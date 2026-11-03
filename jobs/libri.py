"""LibriTTS-R replay source: public voices and sentences, label-free.

The replay lines for the KL-to-Base term are generic sentences that Base itself
speaks in *other*, public voices (LibriTTS-R, CC BY 4.0), so reading skill and
expressive range survive the fine-tune without pulling her voice back toward
Base's clone of it. This script selects that source from one parquet shard:

    curl -sL -o dev0.parquet \
      https://huggingface.co/datasets/mythicinfinity/libritts_r/resolve/main/data/dev.clean/dev.clean-00000-of-00004.parquet
    python3 jobs/libri.py dev0.parquet <out dir>

It writes `<out>/voices/<speaker_id>.wav` — one 24 kHz clip of 4-10 s per
speaker, at most 10 — and `<out>/sentences.txt` — 300 transcripts of 6-25
words, shuffled with a fixed seed. Nothing but speaker id and transcript is
read: no mood label, rating or hand-labelled set touches this data.

Needs pyarrow, soundfile and numpy (the parquet read). These are not part of
the stage's runtime; this is a standalone regeneration script for public data.
"""

import argparse
import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

MAX_VOICES = 10
MAX_SENTENCES = 300
MIN_WORDS, MAX_WORDS = 6, 25
MIN_SECONDS, MAX_SECONDS = 4.0, 10.0
SEED = 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("parquet", help="one LibriTTS-R parquet shard (see the docstring's curl)")
    p.add_argument("out", type=Path, help="folder to write voices/ and sentences.txt into")
    args = p.parse_args(argv)

    t = pq.read_table(args.parquet).to_pandas()
    out = args.out
    (out / "voices").mkdir(parents=True, exist_ok=True)

    spk_col = "speaker_id" if "speaker_id" in t.columns else [c for c in t.columns if "speaker" in c][0]
    txt_col = "text_normalized" if "text_normalized" in t.columns else [c for c in t.columns if "text" in c][0]

    rng = random.Random(SEED)
    done = 0
    for spk, g in t.groupby(spk_col):
        for _, r in g.iterrows():
            a = r["audio"]
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if sr == 24000 and MIN_SECONDS <= len(x) / sr <= MAX_SECONDS:
                sf.write(out / "voices" / f"{spk}.wav", x, sr)
                done += 1
                break
        if done >= MAX_VOICES:
            break

    texts = [s for s in t[txt_col].tolist() if MIN_WORDS <= len(str(s).split()) <= MAX_WORDS]
    rng.shuffle(texts)
    (out / "sentences.txt").write_text("\n".join(texts[:MAX_SENTENCES]) + "\n")
    print("voices", done, "sentences", min(MAX_SENTENCES, len(texts)))


if __name__ == "__main__":
    main()
