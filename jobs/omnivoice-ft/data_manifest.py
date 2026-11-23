"""Build the OmniVoice fine-tune manifest from the project's clip pool.

Output schema is the upstream OmniVoice manifest (primer section 1): one JSON object per line

    {"id": "...", "audio_path": "/abs/path/clip.wav", "text": "...", "language_id": "en"}

Public-voice replay rows (the KL-to-Base guard's data) additionally carry ``"replay": true``.
``jsonl_to_webdataset.py`` and ``extract_audio_tokens.py`` preserve extra fields, and the
upstream ``LabelDataset`` hands the whole object to the sample processor, so the flag survives
into training.

Data rules (VI Assistant project instructions):
  * this tool reads only ``id``, ``text`` and one audio-path field from its inputs; every other
    field (mood, ratings, legacy labels, ...) is ignored, never filtered, weighted or selected on;
  * clips are kept in full: nothing is dropped for appearing in an old labelled set, and no
    duration/quality filter is re-applied here (the project's own audio-quality drop filters live
    in ``jobs/buildset.py`` and have already run when its raw manifests are the input);
  * audio is never copied: the manifest references absolute paths in place. Copyrighted clips
    move only home -> rented worker, never into this repo.

Text source. The project already has text in the label-blind raw manifests written by
``jobs/buildset.py`` (``train_raw.jsonl`` / ``val_raw.jsonl``; field ``text``, subtitle-derived,
brackets stripped). Pass those files with ``--raw-jsonl``. For a plain folder of wavs, pass
``--pool-dir`` plus ``--texts-jsonl`` (any JSONL with ``id`` and ``text``, e.g. the same raw
manifests). Missing transcripts are an error unless ``--allow-empty-text`` is given; when they
are empty the clip is still kept (the model trains on its audio with an empty text span) and the
count is reported in the summary.

Usage:
    python data_manifest.py \
        --raw-jsonl /path/to/train_raw.jsonl --raw-jsonl /path/to/val_raw.jsonl \
        --audio-root /path/to/set-folder \
        --replay-jsonl /path/to/replay_public.jsonl \
        --out /path/to/train_manifest.jsonl

    python data_manifest.py --pool-dir /path/to/wavs --texts-jsonl transcripts.jsonl \
        --out manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")

_BRACKETS = re.compile(r"\[[^\]]*\]")
_WS = re.compile(r"\s+")


def sanitise_id(clip_id: str) -> str:
    """A manifest id WebDataset can round-trip. WebDataset cuts a sample key at the first dot
    (base_plus_ext), and extract_audio_tokens writes each member as ``{id}.npy``, so an id with a
    dot (``le2.s178234_m``) comes back as key ``le2``. Dots become underscores; the clip's
    audio path is a path, not a key, and is left alone."""
    return clip_id.replace(".", "_")


def normalise_text(text: str) -> str:
    """Same text cleanup as jobs/buildset.py: drop bracketed annotations, collapse whitespace."""
    return _WS.sub(" ", _BRACKETS.sub("", text)).strip()


def read_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: not valid JSON: {e}") from e
    return rows


def _resolve_audio(audio: str, base: Path) -> str:
    p = Path(audio)
    if not p.is_absolute():
        p = base / p
    return str(p.resolve())


def _audio_field(row: Dict[str, Any], where: str) -> str:
    for key in ("audio_path", "audio"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise SystemExit(f"{where}: row {row.get('id')!r} has no 'audio_path'/'audio' field")


def rows_from_manifests(paths: Iterable[str], audio_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read id/text/audio from existing raw manifests; never look at any other field."""
    out: List[Dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        base = Path(audio_root).resolve() if audio_root else p.resolve().parent
        for row in read_jsonl(p):
            if not isinstance(row.get("id"), str) or not row.get("id"):
                raise SystemExit(f"{p}: a row has no string 'id'")
            out.append(
                {
                    "id": sanitise_id(row["id"]),
                    "audio": _resolve_audio(_audio_field(row, str(p)), base),
                    "text": normalise_text(str(row.get("text") or "")),
                }
            )
    return out


def rows_from_pool(pool_dir: str, texts: Dict[str, str]) -> List[Dict[str, Any]]:
    """Scan a clip folder; ids are file stems, text comes from the id -> text map."""
    root = Path(pool_dir)
    if not root.is_dir():
        raise SystemExit(f"--pool-dir {pool_dir} is not a directory")
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            out.append({"id": sanitise_id(path.stem), "audio": str(path.resolve()),
                        "text": texts.get(path.stem, "")})
    return out


def texts_from_jsonls(paths: Iterable[str]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for path in paths:
        for row in read_jsonl(path):
            if isinstance(row.get("id"), str) and row.get("text") is not None:
                texts[row["id"]] = normalise_text(str(row["text"]))
    return texts


def build_manifest(
    raw_jsonls: Iterable[str] = (),
    pool_dir: Optional[str] = None,
    texts_jsonls: Iterable[str] = (),
    replay_jsonls: Iterable[str] = (),
    audio_root: Optional[str] = None,
    out: str = "manifest.jsonl",
    language_id: str = "en",
    allow_empty_text: bool = False,
    check_audio: bool = True,
) -> Dict[str, Any]:
    """Write the manifest and a summary; return the summary (also written next to it)."""
    texts_paths = list(texts_jsonls)
    if raw_jsonls:
        main = rows_from_manifests(raw_jsonls, audio_root=audio_root)
    elif pool_dir:
        main = rows_from_pool(pool_dir, texts_from_jsonls(texts_paths))
    else:
        raise SystemExit("give at least one --raw-jsonl or --pool-dir")

    replay: List[Dict[str, Any]] = []
    if replay_jsonls:
        replay = rows_from_manifests(replay_jsonls, audio_root=audio_root)

    explicit_ids = {r["id"] for r in main}
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for kind, rows in (("train", main), ("replay", replay)):
        for row in rows:
            rid = row["id"]
            if kind == "replay" and rid in explicit_ids | seen:
                rid = f"{rid}-replay"  # keep both rows addressable in one label dataset
                n = 2
                while rid in seen:
                    rid = f"{row['id']}-replay{n}"
                    n += 1
            if rid in seen:
                raise SystemExit(f"duplicate id {rid!r} (ids are unique after dots become '_', "
                                 "so two source ids may differ only by '.' vs '_'); OmniVoice's "
                                 "LabelDataset requires unique ids")
            seen.add(rid)
            deduped.append(
                {
                    "id": rid,
                    "audio_path": row["audio"],
                    "text": row["text"],
                    "language_id": language_id,
                    **({"replay": True} if kind == "replay" else {}),
                }
            )

    empty_text = [r["id"] for r in deduped if not r["text"]]
    if empty_text and not allow_empty_text:
        shown = ", ".join(empty_text[:10]) + (" ..." if len(empty_text) > 10 else "")
        raise SystemExit(f"{len(empty_text)} clip(s) have no transcript ({shown}); "
                         "pass --allow-empty-text to keep them with an empty text span")
    missing_audio = [r["id"] for r in deduped if not os.path.isfile(r["audio_path"])]
    if missing_audio and check_audio:
        shown = ", ".join(missing_audio[:10]) + (" ..." if len(missing_audio) > 10 else "")
        raise SystemExit(f"{len(missing_audio)} audio file(s) do not exist ({shown})")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    summary = {
        "manifest": str(out_path.resolve()),
        "n_train": sum(1 for r in deduped if not r.get("replay")),
        "n_rows": len(deduped),
        "n_replay": sum(1 for r in deduped if r.get("replay")),
        "n_empty_text": len(empty_text),
        "empty_text_ids": empty_text[:50],
        "n_missing_audio": len(missing_audio),
        "language_id": language_id,
        "sources": {
            "raw_jsonl": list(raw_jsonls),
            "pool_dir": pool_dir,
            "texts_jsonl": texts_paths,
            "replay_jsonl": list(replay_jsonls),
            "audio_root": audio_root,
        },
        # Recorded so the intent is auditable: only these input fields were read.
        "fields_read": ["id", "text", "audio_path", "audio"],
        "audio_copied": False,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "note": ("Label-blind: no mood/rating/legacy-label field was read or used. "
                 "Clipped audio is referenced in place, never copied."),
    }
    with open(out_path.with_suffix(out_path.suffix + ".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-jsonl", action="append", default=[],
                    help="existing raw manifest(s) with id/text/audio (jobs/buildset.py output); repeatable")
    ap.add_argument("--pool-dir", default=None, help="folder of clips to scan (ids = file stems)")
    ap.add_argument("--texts-jsonl", action="append", default=[],
                    help="JSONL with id/text rows, used with --pool-dir; repeatable")
    ap.add_argument("--replay-jsonl", action="append", default=[],
                    help="public-voice rows appended with replay=true for the KL-to-Base guard; repeatable")
    ap.add_argument("--audio-root", default=None,
                    help="base folder for relative audio paths (default: each manifest's folder)")
    ap.add_argument("--out", required=True, help="output manifest JSONL")
    ap.add_argument("--language-id", default="en")
    ap.add_argument("--allow-empty-text", action="store_true",
                    help="keep clips without a transcript (empty text span) instead of failing")
    ap.add_argument("--no-check-audio", action="store_true", help="do not require audio files to exist")
    a = ap.parse_args(argv)
    summary = build_manifest(
        raw_jsonls=a.raw_jsonl, pool_dir=a.pool_dir, texts_jsonls=a.texts_jsonl,
        replay_jsonls=a.replay_jsonl, audio_root=a.audio_root, out=a.out,
        language_id=a.language_id, allow_empty_text=a.allow_empty_text,
        check_audio=not a.no_check_audio,
    )
    print(json.dumps({k: summary[k] for k in ("manifest", "n_rows", "n_replay", "n_empty_text")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
