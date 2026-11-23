#!/usr/bin/env python3
"""Build the small data slice a debug pass uploads (standard library only).

    python3 -I jobs/debug_slice.py SRC DST [--train-rows N]

SRC is a prepared OmniVoice data folder (jobs/omnivoice_data.py). DST receives the same layout
with only the first rows of each manifest, in file order: --train-rows train rows (default 128),
4 dev rows, 2 style rows, 8 replay texts. data_config.json, voices/ and audio_tokenizer/ are kept
whole; every audio file a kept row names is hard-linked (copied when a link is impossible). DST is
rebuilt from scratch each time (only when it holds this script's .debug-slice marker), and it lives under home's cache, never in the repo, because the
audio is game audio.

Rows are taken by position only. No labels are read, and nothing is chosen, weighted or scored.
The real run still trains on the whole folder; this only keeps a debug pass's upload small.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

MANIFESTS = {"train_manifest.jsonl": None, "dev_manifest.jsonl": 4, "style_manifest.jsonl": 2,
             "replay_texts.jsonl": 8}
WHOLE_DIRS = ("voices", "audio_tokenizer")
WHOLE_FILES = ("data_config.json",)
MARKER = ".debug-slice"


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def head_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line if line.endswith("\n") else line + "\n")
                if len(rows) >= n:
                    break
    return rows


def build(src, dst, train_rows):
    src, dst = Path(src).resolve(), Path(dst)
    if not (src / "data_config.json").is_file():
        raise SystemExit(f"debug_slice: {src} has no data_config.json; not an OmniVoice data folder")
    if dst.exists():
        # Only a folder this script made (it holds the marker) is ever removed.
        if any(dst.iterdir()) and not (dst / MARKER).is_file():
            raise SystemExit(f"debug_slice: {dst} exists and is not a debug slice "
                             f"(no {MARKER}); refusing to remove it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / MARKER).write_text("made by jobs/debug_slice.py; rebuilt on every debug pass\n")
    counts = {}
    audio = 0
    for name, n in MANIFESTS.items():
        p = src / name
        if not p.is_file():
            continue
        rows = head_rows(p, train_rows if n is None else n)
        (dst / name).write_text("".join(rows), encoding="utf-8")
        counts[name] = len(rows)
        for line in rows:
            try:
                rel = json.loads(line).get("audio_path")
            except (ValueError, AttributeError):
                rel = None
            if not rel or os.path.isabs(rel) or ".." in Path(rel).parts:
                continue
            a = src / rel
            if a.is_file() and not (dst / rel).exists():
                link(a, dst / rel)
                audio += 1
    for name in WHOLE_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    for name in WHOLE_DIRS:
        d = src / name
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    link(f, dst / f.relative_to(src))
    return counts, audio


def main(argv=None):
    p = argparse.ArgumentParser(prog="debug_slice.py", description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--train-rows", type=int, default=128)
    a = p.parse_args(argv)
    if a.train_rows < 1:
        p.error("--train-rows must be at least 1")
    counts, audio = build(a.src, a.dst, a.train_rows)
    rows = ", ".join(f"{k.split('.')[0]} {v}" for k, v in counts.items())
    print(f"debug_slice: {a.dst}: {rows}; {audio} audio files linked", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
