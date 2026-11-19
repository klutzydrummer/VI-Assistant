"""Home-side check of the clip ids in an omnivoice_data folder (standard library only).

WebDataset reads a tar member `name.ext` back as sample key = the text before the FIRST dot
(webdataset/tariterators.py `base_plus_ext`, regex `^((?:.*/|)[^.]+)[.]([^/]*)$`). omnivoice's
extract_audio_tokens writes each member as `{id}.npy`, so an id that contains a dot reads back
as a shorter key, several clips merge into one sample, and the label lookup raises KeyError
(omnivoice-ft-r5, ids like `le2.s178234_m`). This module repeats that rule in pure Python so the
check needs no webdataset install, and runs on the real manifests before anything is rented.
"""

import json
import re
import tarfile
from pathlib import Path

# webdataset 1.0.2 tariterators.base_plus_ext
_BASE_PLUS_EXT = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")
MANIFESTS = ("train", "dev", "style")


def webdataset_key(member_name):
    """(key, suffix) for a tar member name, as webdataset splits it; (name, '') with no dot."""
    m = _BASE_PLUS_EXT.match(member_name)
    return (m.group(1), m.group(2)) if m else (member_name, "")


def _read_ids(path):
    return [json.loads(ln)["id"] for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _count_lines(path):
    return sum(1 for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip())


def check_ids(folder):
    """(ok, problems, summary) for one omnivoice_data folder. `problems` are plain sentences
    naming the first offending ids and the count; `summary` is a one-line description."""
    folder = Path(folder)
    problems, ids = [], {}
    for name in MANIFESTS:
        path = folder / f"{name}_manifest.jsonl"
        if not path.is_file():
            problems.append(f"{path} is missing")
            continue
        try:
            ids[name] = _read_ids(path)
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"{path} has a row without a readable id ({e})")
    # Replay ids are made on the worker (replay-0000...), one per replay_texts.jsonl line.
    replay_texts = folder / "replay_texts.jsonl"
    if replay_texts.is_file():
        ids["replay"] = [f"replay-{i:04d}" for i in range(_count_lines(replay_texts))]
    for name, rows in ids.items():
        dotted = [i for i in rows if "." in i or webdataset_key(f"{i}.npy")[0] != i]
        if dotted:
            problems.append(
                f"{len(dotted)} of {len(rows)} {name} id(s) contain a '.', which WebDataset cuts "
                f"as the sample key (first: {', '.join(repr(i) for i in dotted[:3])}); rebuild "
                "the omnivoice_data folder with the sanitising jobs/omnivoice_data.py")
    pool = [i for name in ("train", "dev", "replay") for i in ids.get(name, [])]
    seen, dup = set(), []
    for i in pool:
        if i in seen:
            dup.append(i)
        seen.add(i)
    if dup:
        problems.append(f"{len(dup)} id(s) repeat across train/dev/replay "
                        f"(first: {', '.join(repr(i) for i in dup[:3])})")
    if "dev" in ids and "style" in ids:
        stray = [i for i in ids["style"] if i not in set(ids["dev"])]
        if stray:
            problems.append(f"{len(stray)} style id(s) are not dev ids "
                            f"(first: {', '.join(repr(i) for i in stray[:3])})")
    problems += _check_shards(folder)
    total = sum(len(v) for v in ids.values())
    return not problems, problems, f"{total} ids in {', '.join(ids) or 'no manifests'}"


def _check_shards(folder):
    """If tokenised shards are present, every tar member's webdataset key must have a label."""
    problems = []
    for lst in sorted(folder.glob("shards/*/data.lst")):
        for line in lst.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            tar_path, label_path = Path(parts[0]), Path(parts[1])
            if not (tar_path.is_file() and label_path.is_file()):
                continue
            labels = set(_read_ids(label_path))
            with tarfile.open(tar_path) as tar:
                keys = {webdataset_key(n)[0] for n in tar.getnames()}
            missing = sorted(keys - labels)
            if missing:
                problems.append(f"{tar_path.name}: {len(missing)} sample key(s) have no label "
                                f"(first: {', '.join(repr(k) for k in missing[:3])})")
    return problems
