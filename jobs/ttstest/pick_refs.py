"""Choose the two reference clips for the TTS listening test, label-free and reproducible.

Voice T (the main voice): a Tali game clip. Rule, using no mood or rating field at all:
game le2 or le3, 6-9 s, one sentence, transcript present, not in the val / val100 splits,
then the lowest sha256(seed:id). The clip stays outside the repo and is private.

Voice P (the control): ONE female LibriTTS-R clip, 6-9 s, one sentence, from one parquet shard
(OpenSLR 141, listed there as CC BY 4.0; see jobs/libri.py). LibriTTS-R parquet has no gender
column, so pass the LibriTTS `speakers.tsv` (OpenSLR 60) or `--speaker-id`; the pick is the lowest
sha256(seed:utterance id) among that voice set. A person still listens to the clip before use.

    python3 jobs/ttstest/pick_refs.py \\
        --corpus /home/brandon/Projects/VI-Assistant/corpus/tali \\
        --shard <new empty dir>/dev.clean-00000-of-00004.parquet \\
        --speakers-tsv <new empty dir>/speakers.tsv \\
        --out /home/brandon/Projects/VI-Assistant-listen/tts-test-2026-11-10/refs

It writes vT.wav, vP.wav, refs.json and ref-sources.md in --out (24 kHz mono). The shard is read
only; deleting it afterwards is the caller's job (`--delete-shard` does it once the clip validates).
Needs pyarrow for voice P only (lazy import).
"""

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wavio  # noqa: E402

SEED = "tts-test-2026-11-10"
SR = 24000
MIN_S, MAX_S = 6.0, 9.0
GAMES = ("le2", "le3")
OPENSLR_URL = "https://www.openslr.org/141/"
SHARD_URL = "https://huggingface.co/datasets/mythicinfinity/libritts_r/resolve/main/data/dev.clean/dev.clean-00000-of-00004.parquet"
LIBRIVOX_URL = "https://librivox.org/pages/public-domain/"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_key(seed, clip_id):
    """Deterministic, label-free ordering key."""
    return hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()


def single_sentence(text):
    """True for non-empty text with exactly one terminal . ? ! at its end (ellipses allowed)."""
    t = (text or "").strip()
    if not t or t[-1] not in ".?!":
        return False
    body = re.sub(r"\.{2,}|…", " ", t[:-1])
    return not re.search(r"[.?!](\s|$)", body)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tali_candidates(rows, excluded_ids):
    """Rows meeting the label-free rule. Reads only id, game, duration_s, text."""
    out = []
    for r in rows:
        if r.get("id") in excluded_ids or r.get("game") not in GAMES:
            continue
        d = r.get("duration_s")
        if d is None or not (MIN_S <= float(d) <= MAX_S):
            continue
        if single_sentence(r.get("text")):
            out.append(r)
    return out


def pick_lowest_hash(cands, seed, key="id"):
    return min(cands, key=lambda r: hash_key(seed, r[key])) if cands else None


def find_tali_data(corpus):
    """Locate the training-ready data dir (the one holding train_with_codes.jsonl)."""
    hits = sorted(Path(corpus).glob("stages/voicetune_data/*/out/train_with_codes.jsonl"))
    if not hits:
        raise SystemExit(f"no train_with_codes.jsonl under {corpus}/stages/voicetune_data/*/out")
    return hits[-1].parent


def resolve_audio(row, data_dir, corpus):
    """Find the WAV of a train row: its `audio` path as given, else beside the data or harvest."""
    a = row["audio"]
    a = a["path"] if isinstance(a, dict) else a
    for base in (None, data_dir, Path(corpus), Path(corpus) / "harvest"):
        p = Path(a) if base is None else Path(base) / a
        if p.is_file():
            return p
    name = Path(a).name
    for p in (data_dir / "wav" / name, Path(corpus) / "harvest" / "audio" / name):
        if p.is_file():
            return p
    raise SystemExit(f"audio for {row['id']} not found ({a})")


def clip_window_ok(n, sr):
    return MIN_S <= n / sr <= MAX_S


def pick_tali(corpus, out):
    corpus = Path(corpus)
    data = find_tali_data(corpus)
    excluded = set()
    for name in ("val.jsonl", "val100.jsonl", "val_with_codes.jsonl", "val100_with_codes.jsonl"):
        if (data / name).is_file():
            excluded |= {r["id"] for r in read_jsonl(data / name)}
    rows = read_jsonl(data / "train_with_codes.jsonl")
    cands = tali_candidates(rows, excluded)
    row = pick_lowest_hash(cands, SEED)
    if row is None:
        raise SystemExit("no Tali clip meets the rule")
    src = resolve_audio(row, data, corpus)
    x, sr = wavio.read_wav(src)
    x, method = wavio.resample(x, sr, SR)
    if not clip_window_ok(len(x), SR):
        raise SystemExit(f"picked clip {row['id']} is not 6-9 s after load")
    wavio.write_wav(out / "vT.wav", x, SR)
    return {"id": f"tali:{row['id']}", "text": row["text"].strip(), "game": row["game"],
            "n_candidates": len(cands), "n_excluded": len(excluded), "resample": method}


def load_female_speakers(tsv):
    """Speaker ids with gender F from the LibriTTS speakers.tsv (READER GENDER SUBSET NAME)."""
    ids = set()
    for line in Path(tsv).read_text().splitlines():
        parts = line.replace("|", "\t").split("\t")
        if len(parts) >= 2 and parts[1].strip().upper() == "F" and parts[0].strip().isdigit():
            ids.add(parts[0].strip())
    return ids


def pick_libri(shard, speakers_tsv, speaker_id, out):
    import pyarrow.parquet as pq

    t = pq.read_table(shard).to_pandas()
    spk_col = "speaker_id" if "speaker_id" in t.columns else [c for c in t.columns if "speaker" in c][0]
    txt_col = "text_normalized" if "text_normalized" in t.columns else [c for c in t.columns if "text" in c][0]
    t[spk_col] = t[spk_col].astype(str)
    if speaker_id:
        allowed = {str(speaker_id)}
    elif speakers_tsv:
        allowed = load_female_speakers(speakers_tsv)
    else:
        raise SystemExit("voice P needs --speakers-tsv or --speaker-id (the shard has no gender column)")
    rows = []
    for i, r in t[t[spk_col].isin(allowed)].iterrows():
        a = r["audio"]
        x, sr = _read_bytes(a["bytes"])
        if sr == SR and clip_window_ok(len(x), sr) and single_sentence(str(r[txt_col])):
            rows.append({"id": str(r.get("id", i)), "spk": r[spk_col], "text": str(r[txt_col]).strip(),
                         "x": x, "chapter": str(r.get("chapter_id", "")), "n": len(x)})
    row = pick_lowest_hash(rows, SEED)
    if row is None:
        raise SystemExit("no LibriTTS-R clip meets the rule in this shard")
    wavio.write_wav(out / "vP.wav", row["x"], SR)
    return {"id": f"libritts-r:{row['id']}", "text": row["text"], "speaker": row["spk"],
            "chapter": row["chapter"], "n_candidates": len(rows)}


def _read_bytes(b):
    """Read WAV bytes via a temp file in a private dir (wavio reads paths)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.wav"
        p.write_bytes(b)
        return wavio.read_wav(p)


def write_sources(out, t, p, today):
    t_hash, p_hash = sha256_file(out / "vT.wav"), sha256_file(out / "vP.wav")
    (out / "ref-sources.md").write_text(f"""# Reference clip sources (written {today} by pick_refs.py)

Selection seed: `{SEED}`. Rule is label-free: no mood, rating or hand-labelled field was read.

## vT.wav — voice T (main voice)
- Source: Tali game clip `{t['id']}` ({t['game']}), from the home corpus. **Private, do not publish.**
- Rule: game le2/le3, 6-9 s, one sentence, transcript present, not in val/val100, lowest
  sha256(seed:id) of {t['n_candidates']} candidates ({t['n_excluded']} val ids excluded).
- Transcript: in refs.json. SHA-256 of vT.wav (24 kHz mono): `{t_hash}`
- Licence: copyrighted game audio; stays at home and on the rented worker only. Never committed.

## vP.wav — voice P (public control)
- Source: LibriTTS-R `{p['id']}`, speaker {p['speaker']}, chapter {p['chapter']}, from
  {SHARD_URL}
- Licence evidence: OpenSLR {OPENSLR_URL} lists "License: CC BY 4.0" (fetched 2026-11-10, official,
  controlling). The HF card also shows cc-by-4.0; a BibTeX line there and the arXiv paper icon say
  BY-NC-ND 4.0, which is unresolved and probably covers the paper. Private reference use, nothing
  redistributed. Attribution: LibriTTS-R, Koizumi et al. 2023, https://arxiv.org/abs/2305.18802
- LibriVox public domain: NOT verified (no source fetched). The book page for chapter
  {p['chapter']} still has to be checked at {LIBRIVOX_URL}; the clip is not relied on as public domain.
- Gender: taken from the LibriTTS speaker table, not from the audio. A person confirms by listening.
- SHA-256 of vP.wav (24 kHz mono): `{p_hash}`
""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--speakers-tsv", type=Path)
    ap.add_argument("--speaker-id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delete-shard", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t = pick_tali(a.corpus, a.out)
    p = pick_libri(a.shard, a.speakers_tsv, a.speaker_id, a.out)
    (a.out / "refs.json").write_text(json.dumps(
        {"vT": {"id": t["id"], "text": t["text"]}, "vP": {"id": p["id"], "text": p["text"]}}, indent=2) + "\n")
    write_sources(a.out, t, p, datetime.date.today().isoformat())
    if a.delete_shard:
        a.shard.unlink()
    print("vT", t["id"], "vP", p["id"])


if __name__ == "__main__":
    main()
