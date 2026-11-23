"""Style-slot-aware inference and the labelled listening set.

Loads a trained ``StyleOmniVoice`` checkpoint, reads a style from a reference recording (the style
encoder turns a crop into continuous vectors), and synthesises text lines with the *3-branch* CFG
of ``StyleOmniVoice.generate_tokens``: full (text + prefix), text-only, and nothing, combined as

    log p = log_softmax(c + g_text * (t - u) + g_style * (c - t))

Every generated clip is named ``<style>__<line>__s<seed>__step<N>`` so a human listener can tell
which style, seed and checkpoint produced it.  Human listening decides voice quality; nothing here
computes an automatic score.

Typical real run on the worker (after training):

    python infer_listening.py --checkpoint /path/run/checkpoint-5000 \
        --style-manifest /path/train_manifest.jsonl --max-styles 6 \
        --base-model /path/k2-fsa-OmniVoice-snapshot \
        --out /path/listening-step5000

The style manifest is the same label-blind JSONL that ``data_manifest.py`` wrote (``id``,
``audio_path``, ``text``); the text is used as the reference transcript, so no ASR model is needed.
The audio tokenizer is read from ``--audio-tokenizer`` or ``<base-model>/audio_tokenizer``.

CPU smoke / test path (no audio tokenizer, no downloads): pass tokenised style crops directly and
keep the generated tokens instead of decoding them:

    python infer_listening.py --checkpoint RUN/checkpoint-20 --style-tokens crop.pt \
        --fake-tokenizer --tokens-only --num-step 2 --target-frames 6 --out /tmp/listening
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_OMNISTYLE = os.path.join(os.path.dirname(_HERE), "omnistyle")
for _p in (_HERE, _OMNISTYLE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import style_data as SD  # noqa: E402
from style_omnivoice import StyleOmniVoice  # noqa: E402
from train_ft import verify_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InferConfig:
    checkpoint: str
    out: str
    base_model: str = "k2-fsa/OmniVoice"
    audio_tokenizer: Optional[str] = None
    style_manifest: Optional[str] = None
    style_tokens: List[str] = field(default_factory=list)
    lines_json: str = os.path.join(os.path.dirname(_HERE), "ttstest", "lines.json")
    seeds: Tuple[int, ...] = (1,)
    max_styles: int = 0  # 0 = all
    max_lines: int = 0  # 0 = all
    target_frames: int = 100  # 4 s at 25 fps
    style_crop_frames: int = 75  # 3 s of the reference recording
    style_scale: float = 1.0  # prefix strength knob #1 (scales the vectors)
    guidance_text: float = 2.0
    guidance_style: float = 2.0  # prefix strength knob #2 (3-branch CFG)
    class_temperature: float = 0.0
    layer_penalty_factor: float = 5.0
    num_step: int = 16
    device: str = "cpu"
    tokens_only: bool = False  # skip audio decoding (tests / quick review of token outputs)
    fake_tokenizer: bool = False  # tiny-model checkpoints only: rebuild FakeTextTokenizer
    load_asr: bool = False  # allow transcribing style audio that has no text
    language: str = "en"
    instruct: str = "None"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def checkpoint_step(checkpoint: str) -> Optional[int]:
    name = Path(checkpoint).name
    if name.startswith("checkpoint-") and name.split("-", 1)[1].isdigit():
        return int(name.split("-", 1)[1])
    return None


def load_text_tokenizer(cfg: InferConfig):
    if cfg.fake_tokenizer:
        path = os.path.join(cfg.checkpoint, "fake_tokenizer.json")
        if not os.path.isfile(path):
            raise SystemExit(f"--fake-tokenizer: {path} does not exist")
        with open(path, "r", encoding="utf-8") as f:
            return SD.FakeTextTokenizer(vocab_size=json.load(f)["vocab_size"])
    from transformers import AutoTokenizer

    tok_path = cfg.checkpoint
    try:
        tok = AutoTokenizer.from_pretrained(tok_path)
    except Exception:
        tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def attach_audio_tokenizer(model: StyleOmniVoice, cfg: InferConfig):
    """Load the Higgs v2 audio tokenizer from an explicit local snapshot (never an implicit download)."""
    from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel

    from omnivoice.models.omnivoice import _resolve_model_path

    path = cfg.audio_tokenizer
    if path is None:
        base = _resolve_model_path(cfg.base_model)
        candidate = os.path.join(base, "audio_tokenizer")
        if os.path.isdir(candidate):
            path = candidate
    if path is None or not os.path.isdir(path):
        raise SystemExit(
            "real listening output needs the Higgs v2 audio tokenizer: pass "
            "--audio-tokenizer /path/to/higgs-audio-v2-tokenizer (or put it under "
            f"{cfg.base_model}/audio_tokenizer), or use --tokens-only"
        )
    tokenizer_device = str(model.device)
    model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(path, device_map=tokenizer_device)
    model.feature_extractor = AutoFeatureExtractor.from_pretrained(path)
    model.sampling_rate = model.feature_extractor.sampling_rate
    return model


def load_style_model(cfg: InferConfig) -> StyleOmniVoice:
    model = StyleOmniVoice.from_pretrained(cfg.checkpoint, train=True, dtype=torch.float32)
    model.to(cfg.device)
    model.eval()
    model.configure_training(kl_weight=0.0, metrics_sink=None)
    if not cfg.tokens_only:
        attach_audio_tokenizer(model, cfg)
    return model


# ---------------------------------------------------------------------------
# Style sources
# ---------------------------------------------------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def style_items_from_manifest(path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    items = []
    for row in rows:
        audio = row.get("audio_path") or row.get("audio")
        if not audio:
            raise SystemExit(f"{path}: row {row.get('id')!r} has no audio_path")
        items.append({"id": str(row.get("id") or Path(audio).stem), "audio": audio,
                      "text": str(row.get("text") or "")})
    return items


def style_items_from_tokens(paths: Iterable[str]) -> List[Dict[str, Any]]:
    items = []
    for path in paths:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            tokens = obj.get("tokens", obj.get("audio_tokens"))
            name = str(obj.get("id", Path(path).stem))
        else:
            tokens, name = obj, Path(path).stem
        if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
            raise SystemExit(f"{path}: expected a [C, T] token tensor")
        items.append({"id": name, "tokens": tokens.long()})
    return items


def load_style_items(cfg: InferConfig) -> List[Dict[str, Any]]:
    if cfg.style_manifest and cfg.style_tokens:
        raise SystemExit("give either --style-manifest or --style-tokens, not both")
    if cfg.style_manifest:
        items = style_items_from_manifest(cfg.style_manifest)
    elif cfg.style_tokens:
        items = style_items_from_tokens(cfg.style_tokens)
    else:
        raise SystemExit("give --style-manifest (real audio) or --style-tokens (.pt crops)")
    if cfg.max_styles > 0:
        items = items[: cfg.max_styles]
    if not items:
        raise SystemExit("no style items found")
    return items


def audio_tokens_for(model: StyleOmniVoice, item: Dict[str, Any], cfg: InferConfig) -> torch.Tensor:
    """Encode a style recording into [C, T] audio tokens (upstream voice-clone prompt path)."""
    text = item.get("text") or None
    if text is None:
        if not cfg.load_asr:
            raise SystemExit(
                f"style {item['id']!r} has no transcript; either add text to the manifest or pass "
                "--load-asr (loads Whisper on the fly)"
            )
        if model._asr_pipe is None:
            model.load_asr_model()
    prompt = model.create_voice_clone_prompt(item["audio"], ref_text=text)
    return prompt.ref_audio_tokens.long()


@torch.no_grad()
def style_vectors(model: StyleOmniVoice, tokens: torch.Tensor, crop_frames: int) -> torch.Tensor:
    """Continuous style vectors [n, H] from a deterministic tail crop of the recording."""
    t = tokens.long()
    if t.dim() != 2:
        raise ValueError(f"expected [C, T] tokens, got {tuple(t.shape)}")
    t = t[:, max(0, t.shape[1] - crop_frames):].to(next(model.parameters()).device)
    valid = torch.ones(1, t.shape[1], dtype=torch.bool, device=t.device)
    return model.style_encoder(t.unsqueeze(0), valid).vectors[0].float()


# ---------------------------------------------------------------------------
# Lines and plan
# ---------------------------------------------------------------------------


def load_lines(path: str, max_lines: int = 0) -> List[Dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # one JSON object per line also works
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("lines", list(data.values()))
    lines = [{"id": str(row.get("id", i)), "text": str(row["text"]), **{
        k: str(v) for k, v in row.items() if k not in ("id", "text")}}
        for i, row in enumerate(data)]
    if max_lines > 0:
        lines = lines[:max_lines]
    if not lines:
        raise SystemExit(f"{path}: no lines")
    return lines


def text_ids_for(tokenizer, text: str, language: str = "en", instruct: str = "None") -> torch.Tensor:
    """The same conditioning span the training processor uses (lang + instruct + text)."""
    span = (f"<|lang_start|>{language}<|lang_end|>"
            f"<|instruct_start|>{instruct}<|instruct_end|>"
            f"<|text_start|>{text}<|text_end|>")
    ids = tokenizer(span, return_tensors="pt").input_ids
    return ids[0] if ids.dim() == 2 else ids


def _safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(text)]
    return "".join(keep).strip("-") or "x"


def item_label(style_id: str, line_id: str, seed: int, step: Optional[int]) -> str:
    return f"{_safe_name(style_id)}__{_safe_name(line_id)}__s{seed}__step{step if step is not None else 'na'}"


def build_plan(
    styles: List[Dict[str, Any]],
    lines: List[Dict[str, str]],
    seeds: Iterable[int],
    cfg: InferConfig,
) -> List[Dict[str, Any]]:
    """One row per (style, line, seed); the label is style/seed/step, as the spec asks."""
    step = checkpoint_step(cfg.checkpoint)
    plan = []
    for style in styles:
        for line in lines:
            for seed in seeds:
                label = item_label(style["id"], line["id"], seed, step)
                plan.append({
                    "label": label,
                    "style_id": style["id"],
                    "line_id": line["id"],
                    "register": line.get("register"),
                    "text": line["text"],
                    "seed": int(seed),
                    "step": step,
                    "target_frames": cfg.target_frames,
                    "num_step": cfg.num_step,
                    "guidance_text": cfg.guidance_text,
                    "guidance_style": cfg.guidance_style,
                    "style_scale": cfg.style_scale,
                    "class_temperature": cfg.class_temperature,
                })
    return plan


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _generator(seed: int, label: str, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    # crc32: stable across processes (Python's hash() is salted)
    g.manual_seed((int(seed) * 1_000_003 + zlib.crc32(label.encode("utf-8"))) % (2**31))
    return g


@torch.no_grad()
def generate_for_item(
    model: StyleOmniVoice,
    tokenizer,
    text: str,
    vectors: torch.Tensor,
    row: Dict[str, Any],
    cfg: InferConfig,
) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = text_ids_for(tokenizer, text, cfg.language, cfg.instruct).to(device)
    return model.generate_tokens(
        text_ids=ids,
        target_len=int(row["target_frames"]),
        style_vectors=vectors,
        style_scale=float(row["style_scale"]),
        num_step=int(row["num_step"]),
        guidance_text=float(row["guidance_text"]),
        guidance_style=float(row["guidance_style"]),
        class_temperature=float(row["class_temperature"]),
        layer_penalty_factor=cfg.layer_penalty_factor,
        generator=_generator(row["seed"], row["label"], device),
    )


def _write_wav(path: str, tokens: torch.Tensor, model: StyleOmniVoice):
    import torchaudio

    wav = model.audio_tokenizer.decode(tokens.to(model.audio_tokenizer.device).unsqueeze(0)).audio_values[0]
    wav = wav.detach().cpu().float()
    while wav.dim() > 1 and wav.shape[0] == 1:  # [1, T] / [1, 1, T] -> [T]
        wav = wav[0]
    torchaudio.save(path, wav.unsqueeze(0), int(model.sampling_rate))


def _listening_readme() -> str:
    return """\
This folder is a listening set for a style-prefix fine-tune checkpoint.

Each file is named <style>__<line>__s<seed>__step<N>:
  style  the reference recording whose encoded style was used as the prefix
  line   the text line from jobs/ttstest/lines.json (the register is in listening.jsonl)
  seed   the generation seed
  step   the training step of the checkpoint

Listen to the wav files (or inspect the token tensors if this folder was made with
--tokens-only). Human listening decides voice quality: this folder and its metadata contain
no automatic score, and no file here should be used as a training label.
"""


def run_inference(cfg: InferConfig) -> Dict[str, Any]:
    os.makedirs(cfg.out, exist_ok=True)
    verification = verify_checkpoint(cfg.checkpoint)
    if not verification["ok"]:
        raise SystemExit(f"refusing to run from an unverifiable checkpoint: {verification['problems']}")

    model = load_style_model(cfg)
    tokenizer = load_text_tokenizer(cfg)
    styles = load_style_items(cfg)
    lines = load_lines(cfg.lines_json, cfg.max_lines)

    # Encode every style once; each (line, seed) reuses its vectors.
    encodings: Dict[str, torch.Tensor] = {}
    for style in styles:
        if "tokens" in style:
            tokens = style["tokens"]
        else:
            tokens = audio_tokens_for(model, style, cfg)
        encodings[style["id"]] = style_vectors(model, tokens, cfg.style_crop_frames)

    plan = build_plan(styles, lines, cfg.seeds, cfg)
    out = Path(cfg.out)
    tokens_dir = out / ("tokens" if cfg.tokens_only else "wav")
    tokens_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in plan:
        tokens = generate_for_item(model, tokenizer, row["text"], encodings[row["style_id"]], row, cfg)
        if cfg.tokens_only:
            path = tokens_dir / f"{row['label']}.pt"
            torch.save(tokens.detach().cpu(), path)
        else:
            path = tokens_dir / f"{row['label']}.wav"
            _write_wav(str(path), tokens, model)
        written.append({**row, "file": str(path)})

    with open(out / "listening.jsonl", "w", encoding="utf-8") as f:
        for row in written:
            f.write(json.dumps(row) + "\n")
    (out / "README.txt").write_text(_listening_readme(), encoding="utf-8")
    summary = {
        "checkpoint": cfg.checkpoint,
        "checkpoint_step": checkpoint_step(cfg.checkpoint),
        "checkpoint_verification": verification,
        "out": str(out.resolve()),
        "n_items": len(written),
        "n_styles": len(styles),
        "n_lines": len(lines),
        "seeds": list(cfg.seeds),
        "tokens_only": cfg.tokens_only,
        "config": asdict(cfg),
        "note": "Human listening decides voice quality; this set carries no automatic score.",
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def config_from_args(a: argparse.Namespace) -> InferConfig:
    return InferConfig(
        checkpoint=a.checkpoint,
        out=a.out,
        base_model=a.base_model,
        audio_tokenizer=a.audio_tokenizer,
        style_manifest=a.style_manifest,
        style_tokens=a.style_tokens,
        lines_json=a.lines,
        seeds=tuple(a.seeds),
        max_styles=a.max_styles,
        max_lines=a.max_lines,
        target_frames=a.target_frames,
        style_crop_frames=a.style_crop_frames,
        style_scale=a.style_scale,
        guidance_text=a.guidance_text,
        guidance_style=a.guidance_style,
        class_temperature=a.class_temperature,
        layer_penalty_factor=a.layer_penalty_factor,
        num_step=a.num_step,
        device=a.device,
        tokens_only=a.tokens_only,
        fake_tokenizer=a.fake_tokenizer,
        load_asr=a.load_asr,
        language=a.language,
        instruct=a.instruct,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="a complete checkpoint-N/ directory")
    ap.add_argument("--out", required=True, help="listening-set folder")
    ap.add_argument("--base-model", default="k2-fsa/OmniVoice",
                    help="base snapshot; its audio_tokenizer/ is used when --audio-tokenizer is absent")
    ap.add_argument("--audio-tokenizer", default=None, help="local Higgs v2 tokenizer folder")
    ap.add_argument("--style-manifest", default=None,
                    help="JSONL with id/audio_path/text (the training manifest works)")
    ap.add_argument("--style-tokens", nargs="+", default=[],
                    help=".pt file(s) holding a [C, T] audio-token tensor (or {'id','tokens'})")
    ap.add_argument("--lines", default=InferConfig.lines_json, help="text lines JSON")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--max-styles", type=int, default=0)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--target-frames", type=int, default=100, help="frames at 25 fps (100 = 4 s)")
    ap.add_argument("--style-crop-frames", type=int, default=75, help="tail crop used for the style code")
    ap.add_argument("--style-scale", type=float, default=1.0, help="prefix strength knob #1")
    ap.add_argument("--guidance-text", type=float, default=2.0)
    ap.add_argument("--guidance-style", type=float, default=2.0, help="prefix strength knob #2")
    ap.add_argument("--class-temperature", type=float, default=0.0)
    ap.add_argument("--layer-penalty-factor", type=float, default=5.0)
    ap.add_argument("--num-step", type=int, default=16, help="denoising steps")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens-only", action="store_true", help="write token tensors, no audio decoding")
    ap.add_argument("--fake-tokenizer", action="store_true", help="tiny-model checkpoints (tests)")
    ap.add_argument("--load-asr", action="store_true", help="transcribe style audio that has no text")
    ap.add_argument("--language", default="en")
    ap.add_argument("--instruct", default="None")
    a = ap.parse_args(argv)
    summary = run_inference(config_from_args(a))
    print(json.dumps({k: summary[k] for k in ("out", "n_items", "n_styles", "n_lines")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
