"""Build the OmniVoice fine-tune's data folder at home, label-blind (stage omnivoice_data).

One home CPU job (no GPU, no codec) that produces the one folder both GPU stages
(omnivoice_preflight / omnivoice_train) read through VI_INPUT_VOICETUNE_DATA:

  train_manifest.jsonl   her clips as upstream OmniVoice rows (id, audio_path, text, language_id)
  dev_manifest.jsonl     the held-out conversations, same shape
  style_manifest.jsonl   style-source rows (id, audio_path, text) for the listening sets and probe
  data_config.json       OmniVoice data config; its shard lists are written on the worker
  wav/                   her 24 kHz mono clips, hard-linked (never re-encoded) by jobs/buildset.py
  voices/                the public LibriTTS-R reference voices (replay generation)
  replay_texts.jsonl     the public replay line set: only {"text": ...} rows
  audio_tokenizer/       the Higgs v2 tokenizer when --audio-tokenizer is given, else a
                         PLACEHOLDER file (the Base snapshot carries one; see voicetune-fresh.sh)

Tokenisation (extract_audio_tokens) and replay generation need CUDA, so they run on the worker as
steps 0a/0b of `voicetune-fresh.sh preflight`; they write shards/{train,dev,replay}/ into this
folder, which is where data_config.json already points. Every path inside the folder is relative
to it, so it is relocatable and packs whole; the worker runs with it as the working directory.

Data rules (VI Assistant project instructions): the split comes from jobs/buildset.py, which reads
only game and conversation ids; jobs/omnivoice-ft/data_manifest.py reads only id/text/audio. No
mood label, rating or hand-labelled set is read, filtered, weighted or used to pick clips, and no
clip is dropped for once being in a labelled set. The style rows are picked by a hash of the clip
id, not by any label. Game audio is hard-linked into the folder, never into the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "omnivoice-ft"))
import buildset  # noqa: E402
import data_manifest  # noqa: E402

CACHE = Path.home() / ".cache" / "vi" / "voicetune"
REPLAY_TEXTS = CACHE / "fresh" / "data" / "replay_with_codes.jsonl"
VOICES_SRC = CACHE / "abrun" / "libri" / "voices"
REUSE_WAV = CACHE / "data-v2" / "wav"
STYLE_ROWS = 24

# Shard lists the worker writes (voicetune-fresh.sh, steps 0a/0b); the config only names them.
SHARDS = {"train": "shards/train", "dev": "shards/dev", "replay": "shards/replay"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def relativise(manifest, out):
    """Rewrite audio_path to a path relative to out/ (the worker's working directory)."""
    rows = read_jsonl(manifest)
    base = str(Path(out).resolve())
    for row in rows:
        row["audio_path"] = os.path.relpath(row["audio_path"], base)
    write_jsonl(manifest, rows)
    return rows


def style_rows(rows, n=STYLE_ROWS):
    """n rows ordered by sha256 of the clip id: deterministic, and blind to every label."""
    ranked = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    return [{"id": r["id"], "audio_path": r["audio_path"], "text": r["text"]}
            for r in ranked[:n] if r["text"]]


def data_config():
    shard = lambda key: f"{SHARDS[key]}/data.lst"  # noqa: E731
    return {"train": [{"manifest_path": [shard("train"), shard("replay")], "repeat": 1}],
            "dev": [{"manifest_path": [shard("dev")], "repeat": 1}]}


def copy_voices(src, out):
    vdir = Path(out) / "voices"
    vdir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(src).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"omnivoice_data: no reference voices (*.wav) in {src}")
    for wav in wavs:
        dst = vdir / wav.name
        if not dst.exists():
            try:
                os.link(wav, dst)
            except OSError:
                shutil.copy2(wav, dst)
    return [w.name for w in wavs]


def write_replay_texts(src, out):
    """Only the 'text' field is read, in file order; every other field is ignored."""
    rows = [{"text": r["text"]} for r in read_jsonl(src) if isinstance(r.get("text"), str) and r["text"]]
    if not rows:
        raise SystemExit(f"omnivoice_data: {src} has no rows with a 'text' field")
    write_jsonl(Path(out) / "replay_texts.jsonl", rows)
    return len(rows)


def wire_audio_tokenizer(src, out):
    dst = Path(out) / "audio_tokenizer"
    if src:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return "copied"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "PLACEHOLDER").write_text(
        "No tokenizer was bundled. voicetune-fresh.sh uses $VI_AUDIO_TOKENIZER or the\n"
        "audio_tokenizer/ folder inside the $VI_BASE_MODEL snapshot instead.\n", encoding="utf-8")
    return "placeholder"


def gate_home_disk():
    """The stage's home-disk floor, read from the settings JSON the runner exports."""
    settings = os.environ.get("VI_SETTINGS")
    if not settings:
        return
    cfg = json.load(open(settings))
    if cfg.get("min_home_gb"):
        subprocess.run([sys.executable, str(HERE / "checkpoints.py"), "gate", str(Path.home()),
                        "--min-gb", str(cfg["min_home_gb"]),
                        "--reserve-gb", str(cfg.get("home_gate_reserve_gb", 0))], check=True)


def build(out, repo, reuse_wav, voices, replay_texts, audio_tokenizer, raw_dir=None,
          style_n=STYLE_ROWS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if raw_dir is None:
        buildset.main(["--out", str(out), "--repo", str(repo), "--reuse-wav", str(reuse_wav)])
        raw_dir = out
    raw_dir = Path(raw_dir)
    counts = {}
    for stem, name in (("train", "train_manifest.jsonl"), ("val", "dev_manifest.jsonl")):
        summary = data_manifest.build_manifest(
            raw_jsonls=[str(raw_dir / f"{stem}_raw.jsonl")], audio_root=str(raw_dir),
            out=str(out / name))
        counts[name] = summary["n_rows"]
    relativise(out / "train_manifest.jsonl", out)
    dev = relativise(out / "dev_manifest.jsonl", out)
    style = style_rows(dev, style_n)
    write_jsonl(out / "style_manifest.jsonl", style)
    (out / "data_config.json").write_text(json.dumps(data_config(), indent=1) + "\n", encoding="utf-8")
    summary = {
        "rows": counts, "style_rows": len(style),
        "voices": copy_voices(voices, out),
        "replay_texts": write_replay_texts(replay_texts, out),
        "audio_tokenizer": wire_audio_tokenizer(audio_tokenizer, out),
        "shards_written_on_worker": SHARDS,
        "fields_read": ["id", "text", "audio_path", "audio"],
        "note": "Label-blind: no mood/rating/legacy-label field was read or used.",
    }
    (out / "omnivoice_data_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(os.environ.get("VI_OUT", ".")))
    p.add_argument("--repo", type=Path, default=buildset.REPO)
    p.add_argument("--reuse-wav", type=Path, default=REUSE_WAV)
    p.add_argument("--voices", type=Path, default=VOICES_SRC)
    p.add_argument("--replay-texts", type=Path, default=REPLAY_TEXTS)
    p.add_argument("--audio-tokenizer", type=Path, default=None,
                   help="Higgs v2 tokenizer folder to bundle (default: placeholder)")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="use existing train_raw.jsonl/val_raw.jsonl here instead of running buildset")
    p.add_argument("--style-rows", type=int, default=STYLE_ROWS)
    a = p.parse_args(argv)
    gate_home_disk()
    print(json.dumps(build(a.out, a.repo, a.reuse_wav, a.voices, a.replay_texts, a.audio_tokenizer,
                           a.raw_dir, a.style_rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
