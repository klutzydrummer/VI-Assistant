"""voicelora: one voice LoRA on Qwen3-TTS-12Hz-1.7B-Base that speaks as Tali with no reference clip.
(Stage 5 of the Tali voice plan, Brandon 2026-10-30.) Mood-blind: it learns her voice, accent and
suit from her real clips; moods come later from the projector. One LoRA for every mood, never one
per mood.

Built on the official recipe (QwenLM/Qwen3-TTS finetuning/, read 2026-10-30, research note
research/qwen3tts-finetune-guide.md), with its known problems fixed:
  - alignment: at generation, the hidden state at position p predicts frame p+1's first code, and
    the code predictor gets that same hidden state with frame p+1's first code (talker.forward's
    past_hidden). The official sft_12hz.py instead passes codec_0_labels[:, 1:] to a loss that
    shifts again (each frame learns the frame two ahead: "speech speeds up every epoch"), and pairs
    the hidden state at p with frame p's codes. Here the labels go in unshifted (`train_labels`)
    and the code predictor gets the hidden state one position back (`predictor_hidden_mask`).
  - attention: sdpa, not the hard-coded flash_attention_2 (no flash-attn build on the worker).
  - speaker: her MEAN x-vector over many clips (rescaled to their average norm, as voicearms does)
    sits in the speaker slot for every line and becomes the saved speaker, not whichever clip the
    first shuffled batch held.
  - a LoRA (rank 16) instead of the full fine-tune, so the strength can be turned down at inference
    (others found scale 0.3-0.4 best); lr 2e-6 (2e-5 gives noise and endless output).

    python3 jobs/voicelora.py check   --out <folder>
    python3 jobs/voicelora.py prepare --raw train_raw.jsonl --out <folder> [--device cpu]
    python3 jobs/voicelora.py train   --data <folder>/train_with_codes.jsonl --out <folder>
                                      [--val val_with_codes.jsonl] [--epochs 10] [--max-steps N]
    python3 jobs/voicelora.py speak   --run <folder> --sentences s.txt [--epoch K] [--scales 0.3,0.5,1.0]

train_raw.jsonl (built at home): one line per clip, {"id", "audio" (24 kHz mono wav), "text",
"ref_audio" (any of her clips; only used for the mean x-vector), "game", optional "language"}.
`prepare` adds the codec codes (it fits the home CPU: one ~0.7 GB tokenizer). `train` writes
epoch-<k>/adapter.safetensors, voice.json (her mean x-vector) and train_log.jsonl (loss and peak
GPU memory). `speak` builds a no-reference voice from Base + her x-vector + the adapter and writes
a blind hub list (clips.toml) of every sentence at every scale, plus the Base model's ICL clone
for comparison when --icl-ref is given. Needs qwen-tts==0.1.1 and peft (requirements-lora.txt).
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER = "tali"
SPEAKER_ID = 3000  # the codec-embedding row the official recipe gives a new speaker
SPEAKER_SLOT = 6  # input position that holds the speaker embedding
SR = 24000
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
REQUIRED = ["id", "audio", "text", "ref_audio"]


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def validate(rows, sample_rate=None):
    """Problems with a train_raw/with_codes list, as strings (empty = fine). sample_rate(path) -> Hz."""
    bad = []
    seen = set()
    for n, r in enumerate(rows):
        miss = [k for k in REQUIRED if not r.get(k)]
        if miss:
            bad.append(f"line {n + 1}: missing {', '.join(miss)}")
            continue
        if r["id"] in seen:
            bad.append(f"line {n + 1}: duplicate id {r['id']}")
        seen.add(r["id"])
        if sample_rate is not None:
            for k in ("audio", "ref_audio"):
                sr = sample_rate(r[k])
                if sr != SR:
                    bad.append(f"line {n + 1}: {k} is {sr} Hz, needs {SR}")
    return bad


def train_labels(codec_0_labels):
    """The labels to hand the talker for inputs_embeds[:, :-1]. Its transformers causal-LM loss shifts
    labels by one itself, so they must line up with the inputs, not be pre-shifted (the official
    script's codec_0_labels[:, 1:] trains each frame on the frame two ahead)."""
    return codec_0_labels[:, :-1]


def predictor_hidden_mask(codec_mask):
    """Which hidden states (of inputs_embeds[:, :-1]) feed the code predictor, one per audio frame:
    the position just before each frame, whose output predicted that frame's first code."""
    return codec_mask[:, 1:]


def mean_voice(vectors):
    """Mean of x-vectors, rescaled to their average length (a mean of vectors is shorter than they are)."""
    import numpy as np

    x = np.asarray(vectors, dtype=np.float64)
    m = x.mean(0)
    return m * np.linalg.norm(x, axis=1).mean() / max(np.linalg.norm(m), 1e-12)


def length_bands(rows, key="duration_s"):
    """Clip counts and hours by length band, for the report (short lines matter: issue #343)."""
    bands = {"<2 s": 0, "2-5 s": 0, "5-10 s": 0, "10+ s": 0}
    hours = dict.fromkeys(bands, 0.0)
    for r in rows:
        d = float(r.get(key, 0))
        b = "<2 s" if d < 2 else "2-5 s" if d < 5 else "5-10 s" if d < 10 else "10+ s"
        bands[b] += 1
        hours[b] += d / 3600
    return {b: {"clips": bands[b], "hours": round(hours[b], 3)} for b in bands}


def set_scale(module, scale):
    """Scale every LoRA layer's contribution to `scale` times its trained alpha/r (peft's `scaling`)."""
    n = 0
    for m in module.modules():
        if isinstance(getattr(m, "scaling", None), dict) and hasattr(m, "lora_A"):
            if not hasattr(m, "vi_trained_scaling"):
                m.vi_trained_scaling = dict(m.scaling)
            for k in m.scaling:
                m.scaling[k] = m.vi_trained_scaling[k] * scale
            n += 1
    return n


# --- GPU / model parts (not imported in CI) ---------------------------------------------------------


def load_tts(path=BASE, device="cuda:0"):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return Qwen3TTSModel.from_pretrained(path, device_map=device, dtype=dtype, attn_implementation="sdpa")


def lora_config(rank, alpha):
    from peft import LoraConfig

    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")


def add_lora(talker, rank, alpha):
    from peft import inject_adapter_in_model

    for p in talker.parameters():
        p.requires_grad_(False)
    inject_adapter_in_model(lora_config(rank, alpha), talker)
    return [p for n, p in talker.named_parameters() if "lora_" in n]


def mel(path):
    import librosa
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    wav, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, needs {SR}")
    return mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=SR,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)


def her_voice(model, refs, limit=200):
    import torch

    vecs = []
    with torch.no_grad():
        for p in list(dict.fromkeys(refs))[:limit]:
            m = mel(p).to(model.device).to(model.dtype)
            vecs.append(model.speaker_encoder(m)[0].float().cpu().numpy())
    return mean_voice(vecs), len(vecs)


def cmd_check(a):
    """Everything the run needs, checked in one go before training (plan ahead, don't chase errors)."""
    out = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:  # noqa: BLE001 - report every failure, don't stop at the first
        out["torch_error"] = repr(e)
    for mod in ("qwen_tts", "peft", "transformers", "accelerate", "librosa", "soundfile"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            out[f"{mod}_error"] = repr(e)
    if out.get("transformers") and out["transformers"] != "4.57.3":
        out["transformers_warning"] = "qwen-tts 0.1.1 pins 4.57.3"
    if not a.no_model and "qwen_tts_error" not in out and "peft_error" not in out:
        try:
            import torch

            dev = "cuda:0" if out.get("gpu") else "cpu"
            tts = load_tts(a.base, dev)
            got = add_lora(tts.model.talker, a.rank, a.alpha)
            out["lora_params_m"] = round(sum(p.numel() for p in got) / 1e6, 2)
            out["model"] = "loaded with sdpa, LoRA injected"
            if dev.startswith("cuda"):
                out["vram_after_load_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception as e:  # noqa: BLE001
            out["model_error"] = repr(e)
    errors = [k for k in out if k.endswith("_error")]
    out["ok"] = not errors
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


def cmd_prepare(a):
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    rows = read_jsonl(a.raw)
    bad = validate(rows, lambda p: sf.info(p).samplerate)
    if bad:
        sys.exit("prepare: fix these first:\n" + "\n".join(bad[:50]))
    tok = Qwen3TTSTokenizer.from_pretrained(a.tokenizer, device_map=a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / (Path(a.raw).stem.replace("_raw", "") + "_with_codes.jsonl")
    done = {r["id"] for r in read_jsonl(dst)} if dst.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with open(dst, "a") as f:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            enc = tok.encode([r["audio"] for r in chunk])
            for r, code in zip(chunk, enc.audio_codes):
                f.write(json.dumps({**r, "audio_codes": code.cpu().tolist()}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"prepare: {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"prepare: {dst}")


def batches(dataset, size, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=size, shuffle=shuffle, generator=g, collate_fn=dataset.collate_fn)


def step_loss(model, batch, voice):
    """The official forward pass with the label fix and her fixed voice in the speaker slot."""
    input_ids = batch["input_ids"].to(model.device)
    codec_ids = batch["codec_ids"].to(model.device)
    text_mask = batch["text_embedding_mask"].to(model.device)
    codec_emb_mask = batch["codec_embedding_mask"].to(model.device)
    attention = batch["attention_mask"].to(model.device)
    labels = batch["codec_0_labels"].to(model.device)
    codec_mask = batch["codec_mask"].to(model.device)

    emb = model.talker.model.text_embedding(input_ids[:, :, 0]) * text_mask
    cemb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_emb_mask
    cemb[:, SPEAKER_SLOT, :] = voice.to(cemb.dtype)
    emb = emb + cemb
    for i in range(1, 16):
        e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        emb = emb + e * codec_mask.unsqueeze(-1)
    out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=attention[:, :-1],
                       labels=train_labels(labels), output_hidden_states=True)
    hidden = out.hidden_states[0][-1][predictor_hidden_mask(codec_mask)]
    _, sub_loss = model.talker.forward_sub_talker_finetune(codec_ids[codec_mask], hidden)
    return out.loss + 0.3 * sub_loss


def cmd_train(a):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoConfig

    sys.path.insert(0, str(a.qwen_finetuning))
    from dataset import TTSDataset  # the official dataset/collate (QwenLM/Qwen3-TTS finetuning/)

    torch.manual_seed(a.seed)
    tts = load_tts(a.base, "cuda:0")
    model = tts.model
    config = AutoConfig.from_pretrained(a.base)
    rows = read_jsonl(a.data)
    bad = validate(rows)
    if bad:
        sys.exit("train: " + "; ".join(bad[:20]))
    val = read_jsonl(a.val) if a.val else []

    vec, n_refs = her_voice(model, [r["ref_audio"] for r in rows], a.voice_refs)
    voice = torch.tensor(vec, dtype=model.dtype, device=model.device)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "voice.json").write_text(json.dumps({"speaker": SPEAKER, "refs": n_refs, "xvector": vec.tolist()}))

    params = add_lora(model.talker, a.rank, a.alpha)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    train_dl = batches(TTSDataset(rows, tts.processor, config), a.batch, True, a.seed)
    val_dl = batches(TTSDataset(val, tts.processor, config), a.batch, False, a.seed) if val else None
    log = open(a.out / "train_log.jsonl", "a")
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        opt.zero_grad()
        for k, batch in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = step_loss(model, batch, voice) / a.accum
            loss.backward()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                rec = {"epoch": epoch, "step": step, "loss": round(loss.item() * a.accum, 4),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "s": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 10 == 0:
                    print(rec, flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
        rec = {"epoch": epoch, "end": True}
        if val_dl is not None:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl = [step_loss(model, b, voice).item() for b in val_dl]
            model.train()
            rec["val_loss"] = round(sum(vl) / len(vl), 4)
        ep = a.out / f"epoch-{epoch}"
        ep.mkdir(exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model.talker).items()}
        save_file(state, str(ep / "adapter.safetensors"))
        (ep / "lora.json").write_text(json.dumps({"rank": a.rank, "alpha": a.alpha, "targets": LORA_TARGETS, "base": a.base}))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    print(f"train: done in {time.time() - t0:.0f} s, {step} steps, adapters in {a.out}")


def voice_dir(base, vec, dst):
    """Base turned into a custom_voice model whose speaker `tali` is her mean x-vector (the official
    recipe's checkpoint layout, minus its fine-tuned weights: the LoRA goes on top)."""
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    if (dst / "config.json").exists():
        return dst
    src = Path(base) if Path(base).exists() else Path(snapshot_download(base))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg["tts_model_type"] = "custom_voice"
    t = cfg.setdefault("talker_config", {})
    t["spk_id"] = {SPEAKER: SPEAKER_ID}
    t["spk_is_dialect"] = {SPEAKER: False}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    state = load_file(str(src / "model.safetensors"))
    state = {k: v for k, v in state.items() if not k.startswith("speaker_encoder")}
    w = state["talker.model.codec_embedding.weight"]
    w[SPEAKER_ID] = torch.tensor(vec, dtype=w.dtype)
    save_file(state, str(dst / "model.safetensors"))
    return dst


def cmd_speak(a):
    import hashlib

    import soundfile as sf
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    run = a.run
    voice = json.loads((run / "voice.json").read_text())
    epochs = sorted(int(p.name.split("-")[1]) for p in run.glob("epoch-*") if (p / "adapter.safetensors").exists())
    if not epochs:
        sys.exit("speak: no adapters in the run")
    epoch = a.epoch if a.epoch is not None else epochs[-1]
    ep = run / f"epoch-{epoch}"
    lora = json.loads((ep / "lora.json").read_text())
    sentences = [x.strip() for x in Path(a.sentences).read_text().splitlines() if x.strip()]
    vd = voice_dir(lora["base"], voice["xvector"], run / "voice-model")
    tts = load_tts(str(vd), a.device)
    add_lora(tts.model.talker, lora["rank"], lora["alpha"])
    set_peft_model_state_dict(tts.model.talker, load_file(str(ep / "adapter.safetensors")))
    tts.model.eval()
    out = a.out or (run / f"speak-epoch-{epoch}")
    (out / "wav").mkdir(parents=True, exist_ok=True)
    clips = []
    timing = []
    for scale in [float(s) for s in a.scales.split(",")]:
        set_scale(tts.model.talker, scale)
        for n, text in enumerate(sentences):
            t0 = time.time()
            wavs, sr = tts.generate_custom_voice(text=text, speaker=SPEAKER)
            p = out / "wav" / f"lora{scale:g}--{n:02d}.wav"
            sf.write(p, wavs[0], sr)
            timing.append({"scale": scale, "n": n, "s": round(time.time() - t0, 2), "audio_s": round(len(wavs[0]) / sr, 2)})
            clips.append((f"LoRA epoch {epoch} x{scale:g}, no reference | sentence {n}: {text}", p))
    if a.icl_ref:
        del tts  # one big model at a time
        base = load_tts(lora["base"], a.device)
        for n, text in enumerate(sentences):
            wavs, sr = base.generate_voice_clone(text=text, ref_audio=a.icl_ref, ref_text=a.icl_ref_text)
            p = out / "wav" / f"base-icl--{n:02d}.wav"
            sf.write(p, wavs[0], sr)
            clips.append((f"Base model with her clip as the example | sentence {n}: {text}", p))
    clips.sort(key=lambda c: hashlib.sha256(c[0].encode()).hexdigest())
    hub = ['title = "Voice LoRA: is this Tali, with no reference clip?"',
           'question = "Is this Tali with her suit and accent? Note short-line glitches or speed-ups."',
           'answers = ["her, suit and accent", "her, suit or accent partly", "not quite her", "not her or broken"]',
           "blind = true", 'asked_by = "Modeling Tali\'s suit effect"', ""]
    for label, p in clips:
        hub += ["[[clip]]", f"path = {json.dumps(str(p.relative_to(out)))}", f"label = {json.dumps(label)}", ""]
    (out / "clips.toml").write_text("\n".join(hub))
    write_jsonl(out / "timing.jsonl", timing)
    print(f"speak: {len(clips)} clips in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--base", default=BASE)
    c.add_argument("--no-model", action="store_true", help="imports and GPU only")
    c.add_argument("--rank", type=int, default=16)
    c.add_argument("--alpha", type=int, default=32)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--val", type=Path)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--base", default=BASE)
    t.add_argument("--qwen-finetuning", type=Path, default=Path("Qwen3-TTS/finetuning"),
                   help="the official finetuning/ folder (for dataset.py), pinned commit 022e286b")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-6)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--max-steps", type=int, default=0, help="stop early (smoke)")
    t.add_argument("--voice-refs", type=int, default=200, help="clips averaged into her x-vector")
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("speak")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--sentences", type=Path, required=True)
    s.add_argument("--epoch", type=int)
    s.add_argument("--scales", default="0.3,0.5,1.0")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", type=Path)
    s.add_argument("--icl-ref", help="her clip, for a Base-model ICL comparison")
    s.add_argument("--icl-ref-text", default="")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "prepare": cmd_prepare, "train": cmd_train, "speak": cmd_speak}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
