"""Generate the public-voice replay set for the OmniVoice fine-tune (KL-to-Base guard data).

OmniVoice Base speaks the public-domain line set in the public LibriTTS-R reference voices,
zero-shot, audio-only voice cloning, round-robin (voice = voices[n % len(voices)]). These are
the "replay" rows the fine-tune's KL-to-Base term compares against: Base's own speech in
OTHER voices.

    python3 jobs/omnivoice-ft/replay_public.py --out-dir /path/replay-set [--device cpu]

Writes per line n:

    <out-dir>/replay-wavs/replay-<n:04d>.wav   mono WAV, the model's native rate (24 kHz)
    <out-dir>/replay_public.jsonl              one JSON object per line, in line order:

        {"id": "replay-<n:04d>", "audio_path": "<absolute path>", "text": "<exact text>",
         "language_id": "en", "replay": true}

which is exactly the data_manifest.py replay schema (jobs/omnivoice-ft/data_manifest.py
--replay-jsonl consumes this file). Ids match the old Qwen3 replay set (replay-0000..-0299)
for comparability; only its TEXTS and VOICE SET carry over (its audio codes are Qwen3
16-codebook artifacts and cannot be reused).

Reference clips: the LibriTTS-R clips have no transcripts in this pipeline (the old Qwen3
replay cloned from audio only, x-vector mode), so ref_text is passed as "" -- audio-only
cloning -- which also keeps the Whisper ASR model from ever loading. language is "en".

Resume: a line whose WAV already exists (non-empty) is not re-generated, so a killed run
resumes without redoing work; the manifest is rewritten in full at the end (atomic). To
force a line to regenerate, delete its WAV.

Offline by default: HF_HUB_OFFLINE is set to "1" unless already set; the model is loaded
from the local --base-model snapshot path only. Nothing is ever downloaded.

Data rules (VI Assistant project instructions):
  * only the "text" field of the input JSONL rows is read, in file order; every other field
    (audio, ref_audio, audio_codes, ...) is ignored, never filtered, weighted or selected on;
  * public data only: the public-domain lines and the 8 LibriTTS-R voices;
  * there is no path into the Tali / game-audio pool in this script.

The real adapter is imported only when a WAV actually needs generating (mirrors
jobs/ttstest/generate.py), so the manifest writer can be tested with a stub generator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "ttstest") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "ttstest"))
import wavio  # noqa: E402  (stdlib-only WAV helpers shared by the listening-test scripts)

DEFAULT_TEXTS = "/home/brandon/.cache/vi/voicetune/fresh/data/replay_with_codes.jsonl"
DEFAULT_VOICES = "/home/brandon/.cache/vi/voicetune/abrun/libri/voices"
DEFAULT_BASE_MODEL = ("/home/brandon/.cache/huggingface/hub/models--k2-fsa--OmniVoice/"
                      "snapshots/c5fdb5ccb189668d56333f77ba2629f4cd7535f4")

LANGUAGE_ID = "en"


def read_texts(path, only=None):
    """The input JSONL's "text" field only, in file order; nothing else is ever read.

    `only` limits the result to the first N lines (the script's --only).
    """
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: not valid JSON: {e}") from e
            if "text" not in row:
                raise SystemExit(f"{path}:{line_no}: row has no 'text' field")
            texts.append(row["text"])
            if only is not None and len(texts) >= only:
                break
    return texts


def _np_to_list(w):
    """A model's audio (tensor / ndarray / list, maybe 2-D) as a flat mono list of floats."""
    if hasattr(w, "detach"):
        w = w.detach().float().cpu().numpy()
    if hasattr(w, "ndim") and w.ndim > 1:
        w = w.squeeze()
        if w.ndim > 1:
            w = w.mean(axis=0 if w.shape[0] < w.shape[-1] else 1)
    return [float(v) for v in w]


def load_omnivoice(base_model, device):
    """Real OmniVoice adapter (k2-fsa), imported only when generation needs it.

    Returns synth(text, ref_wav) -> (mono samples, sample_rate). ref_text stays "" (audio-only
    cloning; the reference clips have no transcripts and Whisper must never load).
    """
    import torch
    from omnivoice import OmniVoice

    if device == "cuda":
        m = OmniVoice.from_pretrained(base_model, device_map="cuda:0", dtype=torch.float16)
    else:
        m = OmniVoice.from_pretrained(base_model, device_map="cpu", dtype=torch.float32)
    sr = int(getattr(m, "sampling_rate", None) or 24000)

    def synth(text, ref_wav):
        audio = m.generate(text=text, language=LANGUAGE_ID, ref_text="", ref_audio=str(ref_wav))
        return _np_to_list(audio[0]), sr

    return synth


def generate(out_dir, lines, voices, synth=None, device="cuda",
             base_model=DEFAULT_BASE_MODEL, voice=None):
    """Speak every line in a reference voice and write the wavs + replay manifest.

    Lines with an existing non-empty WAV are skipped (resume); the manifest is rewritten in
    full at the end, rows in line order. Returns the manifest Path.
    """
    out_dir = Path(out_dir)
    wav_dir = out_dir / "replay-wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = []
    n_generated = 0
    for n, text in enumerate(lines):
        wav_path = wav_dir / f"replay-{n:04d}.wav"
        if not (wav_path.is_file() and wav_path.stat().st_size > 0):
            if synth is None:
                t0 = time.time()
                synth = load_omnivoice(base_model, device)
                print(f"loaded {base_model} on {device} in {time.time() - t0:.1f}s")
            ref = voices[voice % len(voices)] if voice is not None else voices[n % len(voices)]
            t1 = time.time()
            try:
                samples, sr = synth(text, ref)
                wavio.write_wav(wav_path, samples, sr)
            except Exception as e:  # noqa: BLE001 - counted below; the guard decides, not this line
                failed.append(n)
                print(f"replay-{n:04d} failed: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            n_generated += 1
            print(f"replay-{n:04d} {len(samples) / sr:.1f}s audio @ {sr} Hz "
                  f"({time.time() - t1:.1f}s) ref={ref.name}")
        rows.append({
            "id": f"replay-{n:04d}",
            "audio_path": str(wav_path.resolve()),
            "text": text,
            "language_id": LANGUAGE_ID,
            "replay": True,
        })
    manifest = out_dir / "replay_public.jsonl"
    tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, manifest)
    print(f"{manifest}: {len(rows)} rows ({n_generated} generated, "
          f"{len(rows) - n_generated} reused)")
    # Count guard: refuse a silent shortfall of more than 1 % of the requested lines. Only
    # counts lines; it never picks or weights them. A rerun resumes the lines that worked.
    if len(failed) * 100 > len(lines):
        raise SystemExit(f"replay_public: {len(failed)} of {len(lines)} requested lines failed "
                         f"(over 1 %); refusing a silently shrunk replay set")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="folder for replay-wavs/ and replay_public.jsonl")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help="local OmniVoice snapshot (absolute path)")
    ap.add_argument("--voices", default=DEFAULT_VOICES,
                    help="reference WAV folder (LibriTTS-R)")
    ap.add_argument("--texts", default=DEFAULT_TEXTS,
                    help="JSONL line set; only its 'text' field is read")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--only", type=int, default=None, metavar="N",
                    help="generate just the first N lines (default: all)")
    ap.add_argument("--voice", type=int, default=None, metavar="N",
                    help="use voices[N] for every line (default: round-robin)")
    a = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download; override if set
    texts = read_texts(a.texts, a.only)
    if not texts:
        raise SystemExit(f"--texts {a.texts}: no lines with a 'text' field")
    voices = sorted(Path(a.voices).glob("*.wav"))
    if not voices:
        raise SystemExit(f"--voices {a.voices}: no .wav files")
    if a.voice is not None and not 0 <= a.voice < len(voices):
        raise SystemExit(f"--voice {a.voice} out of range (0..{len(voices) - 1})")

    manifest = generate(a.out_dir, texts, voices, device=a.device,
                        base_model=a.base_model, voice=a.voice)
    print(f"done: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
