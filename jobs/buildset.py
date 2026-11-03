"""Build the Tali voice fine-tune manifests, label-blind.

A reproducible, label-blind replacement for the 2026-10-30 scratch
`buildset.py`. It ports that script's drop filters unchanged and keeps the
same row schema, but the train/val split no longer depends on rated or
held-out conversations: whole conversations are assigned by a seeded hash of
the conversation id, ~10% per game. No clips.toml, spotcheck scores, mood
pairs, gold set, label_fixes or mood field is read; the only analysis input
is voice_sets.json["excluded"] (effect clips).

Outputs into a new folder: train_raw.jsonl, val_raw.jsonl, val100_raw.jsonl,
summary.json and wav/. Wavs are reused (hard-linked) from an earlier build
when present, otherwise re-encoded from the clean-stage FLACs.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import tomllib
from pathlib import Path

# The repo root: this file lives in <repo>/jobs/, two levels up. Resolved so
# it works from a worktree or a clone, not just one machine's checkout.
REPO = Path(__file__).resolve().parents[1]
OLD_WAV_DIR = Path.home() / ".cache/vi/voicetune/data/wav"

# Fixed seeds, chosen once. Never change them: the conversation split and the
# 100-clip validation sample must stay reproducible across rebuilds.
SPLIT_SEED = 20261030
VAL100_SEED = 20261102
VAL_FRACTION = 0.10
VAL100_SIZE = 100

# LE1 keeps the pre-upgrade suit voice, a different vocal effect from the
# LE2/LE3 suit, so its lines stay out of TTS training.
LE1_REASON = ("LE1 has a different suit vocal effect; LE1 lines are kept out "
              "of TTS training and may later be converted to the LE2/LE3 suit "
              "with an RVC model trained on LE2/LE3, if LE2/LE3 alone prove "
              "too few.")

BAD_CLEAN = {"far_voice", "subtitle_mismatch", "asr_only", "nonverbal", "silent"}
BAD_HARVEST = {"bark", "nonverbal", "no_subtitle", "singing"}

interj = re.compile(
    r"^\W*((a+h+|u+g+h+|a+r+g+h+|g+a+h+|o+w+|o+h+|n+o+|h+a+|e+h+|m+m+|h+m+|h+n+g+)\W*)+$",
    re.I,
)
pain = re.compile(r"\b(burning up|on fire|it hurts|i'm hit|can't breathe|aaa+h|arr+gh)\b", re.I)


def drop_filters(lines, clean, effect, excl):
    """Apply the unchanged drop filters; return (kept_rows, drop_counts).

    `lines` and `clean` are id -> record dicts. Filters run in the original
    order, so a clip counts under the first reason that drops it.
    """
    why = collections.Counter()
    kept = []
    for i, l in lines.items():
        g = l["source"]["game"]
        c = clean.get(i)
        # LE1 keeps the pre-upgrade suit voice (see LE1_REASON); its lines are
        # not part of TTS training.
        if g not in ("le2", "le3"):
            why["le1"] += 1
            continue
        if c is None or not c.get("keep", True) or (set(c.get("flags") or []) & BAD_CLEAN):
            why["clean dropped/flagged"] += 1
            continue
        if set(l.get("flags") or []) & BAD_HARVEST:
            why["bark/nonverbal/no subtitle"] += 1
            continue
        if i in effect:
            why["effect clip"] += 1
            continue
        if i in excl:
            why["refscreen exclude"] += 1
            continue
        text = re.sub(r"\[[^\]]*\]", "", l.get("text", "")).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or interj.match(text) or pain.search(text):
            why["scream/pain/interjection"] += 1
            continue
        d = c["audio"]["duration_s"]
        if not 0.8 <= d <= 15:
            why["length outside 0.8-15 s"] += 1
            continue
        kept.append({
            "id": i,
            "audio": f"wav/{i}.wav",
            "text": text,
            "ref_audio": f"wav/{i}.wav",
            "game": g,
            "language": "English",
            "duration_s": round(d, 3),
            "conversation": l["source"].get("conversation"),
        })
    return kept, why


def conversation_hash(conversation_id):
    """Deterministic seeded hash of a conversation id (stable across runs)."""
    return int.from_bytes(
        hashlib.sha256(f"{SPLIT_SEED}:{conversation_id}".encode("utf-8")).digest()[:8],
        "big",
    )


def val_conversation_count(n):
    """~VAL_FRACTION of n conversations, at least one when any exist."""
    if n <= 0:
        return 0
    return max(1, round(n * VAL_FRACTION))


def split_conversations(rows):
    """Label-blind split: whole conversations by seeded hash, per game.

    Only `game` and `conversation` are read; every other field (text, mood,
    rated status, ...) is ignored.
    """
    by_game = collections.defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)
    train, val = [], []
    for game, game_rows in by_game.items():
        convs = sorted({r["conversation"] for r in game_rows}, key=conversation_hash)
        val_convs = set(convs[: val_conversation_count(len(convs))])
        for r in game_rows:
            (val if r["conversation"] in val_convs else train).append(r)
    return train, val


def sample_val100(val, seed=VAL100_SEED):
    """VAL100_SIZE clips drawn at random (seeded) from val."""
    return random.Random(seed).sample(val, min(VAL100_SIZE, len(val)))


def ensure_wavs(rows, clean, clean_out, out, reuse_wav):
    """Write (or link) each kept clip's 24 kHz mono wav, without duplicating
    audio that an earlier build already wrote."""
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        dest = wav_dir / f"{r['id']}.wav"
        if dest.exists():
            continue
        src = reuse_wav / f"{r['id']}.wav"
        if src.exists():
            try:
                os.link(src, dest)
                continue
            except OSError:
                pass
        _write_wav(clean[r["id"]]["audio"]["path"], clean_out, dest)


def _write_wav(clean_rel, clean_out, dest):
    import soundfile as sf

    x, sr = sf.read(str(clean_out / clean_rel), dtype="float32", always_2d=True)
    assert sr == 24000, (clean_rel, sr)
    sf.write(str(dest), x.mean(1), sr, subtype="PCM_16")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path,
                   help="output folder (created; never overwrite an existing build)")
    p.add_argument("--repo", type=Path, default=REPO, help="repo checkout holding the corpus")
    p.add_argument("--reuse-wav", type=Path, default=OLD_WAV_DIR,
                   help="folder of existing wavs to hard-link from")
    args = p.parse_args(argv)

    repo = args.repo
    stages = repo / "corpus/tali/stages"
    harvest = stages / "harvest" / (stages / "harvest" / "current").read_text().strip() / "out"
    clean_out = stages / "clean" / (stages / "clean" / "current").read_text().strip() / "out"

    lines = {json.loads(l)["id"]: json.loads(l) for l in open(harvest / "lines.jsonl")}
    clean = {json.loads(l)["id"]: json.loads(l) for l in open(clean_out / "clips.jsonl")}
    sets = json.loads((repo / "corpus/tali/analysis/after_labels/voice_sets.json").read_text())
    effect = set(sets["excluded"])
    ds = tomllib.loads((repo / "characters/tali/dataset.toml").read_text())
    excl = set(ds["stage"]["refscreen"]["settings"]["exclude"])

    kept, why = drop_filters(lines, clean, effect, excl)
    train, val = split_conversations(kept)
    val100 = sample_val100(val)

    args.out.mkdir(parents=True, exist_ok=True)
    ensure_wavs(kept, clean, clean_out, args.out, args.reuse_wav)

    def write_rows(path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    write_rows(args.out / "train_raw.jsonl", train)
    write_rows(args.out / "val_raw.jsonl", val)
    write_rows(args.out / "val100_raw.jsonl", val100)

    summary = {
        "train": len(train),
        "val": len(val),
        "val100": len(val100),
        "train_hours": round(sum(r["duration_s"] for r in train) / 3600, 2),
        "val_hours": round(sum(r["duration_s"] for r in val) / 3600, 2),
        "split": "seeded conversation hash, label-blind",
        "seed": SPLIT_SEED,
        "val100_seed": VAL100_SEED,
        "fraction": VAL_FRACTION,
        "le1_reason": LE1_REASON,
        "dropped": dict(why),
        "by_game_train": dict(collections.Counter(r["game"] for r in train)),
        "by_game_val": dict(collections.Counter(r["game"] for r in val)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
