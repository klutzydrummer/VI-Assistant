"""Shared plumbing for the Python dataset jobs (discover, label).

A job reads its inputs from the folders `vi dataset run` (or on-worker.sh)
names in VI_INPUT_*, its settings from VI_SETTINGS, and writes into VI_OUT.
File formats: docs/design/corpus-format.md.

This module uses only the standard library, so the fusion logic and the
fake-model test runs work anywhere, including CI.
"""

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

JOBS = Path(__file__).resolve().parent


def env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


def settings():
    path = env_path("VI_SETTINGS")
    if path is None:
        return {}
    return json.loads(path.read_text())


def out_dir():
    out = env_path("VI_OUT")
    if out is None:
        sys.exit("VI_OUT is not set: run this through `vi dataset run`")
    out.mkdir(parents=True, exist_ok=True)
    return out


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")


# --- inputs -----------------------------------------------------------------


@dataclass
class Clip:
    """One line of hers to score: the harvest record joined with clean's."""

    id: str
    text: str
    audio: Path
    duration_s: float
    flags: list = field(default_factory=list)
    context: list = field(default_factory=list)
    scene_tags: list = field(default_factory=list)
    stage_direction: str | None = None
    game: str = ""

    @property
    def bark(self):
        return "bark" in self.flags

    @property
    def nonverbal(self):
        return "nonverbal" in self.flags

    @property
    def has_words(self):
        """Whether there are spoken words for the text judge to read."""
        return bool(self.text.strip()) and not self.nonverbal


def load_clips():
    """Harvest's lines, with clean's audio and transcript when clean has run.

    Clips clean dropped (`keep: false`) are left out.
    """
    harvest = env_path("VI_INPUT_HARVEST")
    if harvest is None:
        sys.exit("this job needs the harvest stage as an input")
    clean = env_path("VI_INPUT_CLEAN")
    cleaned = {}
    if clean is not None:
        cleaned = {c["id"]: c for c in read_jsonl(clean / "clips.jsonl")}

    clips = []
    for line in read_jsonl(harvest / "lines.jsonl"):
        audio = harvest / line["audio"]["path"]
        duration = line["audio"]["duration_s"]
        text = line.get("text", "")
        flags = list(line.get("flags", []))
        if clean is not None:
            c = cleaned.get(line["id"])
            if c is None or not c.get("keep", True):
                continue
            audio = clean / c["audio"]["path"]
            duration = c["audio"]["duration_s"]
            text = c.get("transcript") or text
            flags += [f for f in c.get("flags", []) if f not in flags]
        clips.append(
            Clip(
                id=line["id"],
                text=text,
                audio=audio,
                duration_s=duration,
                flags=flags,
                context=line.get("context", []),
                scene_tags=line.get("scene_tags", []),
                stage_direction=line.get("stage_direction"),
                game=line.get("source", {}).get("game", ""),
            )
        )
    limit = settings().get("limit")
    if limit:
        clips = spread(clips, int(limit))
    return clips


def spread(clips, n):
    """A deterministic sample of n clips, picked by id hash so every stage
    with the same limit picks the same ones."""
    if n >= len(clips):
        return clips
    ranked = sorted(clips, key=lambda c: hashlib.sha256(c.id.encode()).hexdigest())
    keep = {c.id for c in ranked[:n]}
    return [c for c in clips if c.id in keep]


@dataclass
class Axis:
    id: str
    description: str


def load_axes():
    """The pack's variables, in its order. The pack's states.toml comes in as
    a --put file on the worker (files/states.toml), or by path at home."""
    s = settings()
    candidates = [Path("files/states.toml")]
    if s.get("states"):
        candidates.insert(0, Path(s["states"]))
    for path in candidates:
        if path.exists():
            pack = tomllib.loads(path.read_text())
            return [Axis(v["id"], v.get("description", "")) for v in pack["variable"]]
    sys.exit("no states.toml: pass it with --put states.toml")


def write_axes(out, axes):
    (out / "axes.json").write_text(
        json.dumps([{"id": a.id, "description": a.description} for a in axes], indent=1)
        + "\n"
    )


# --- environment ------------------------------------------------------------


def fake_models():
    """`backend = "fake"` swaps every model for a deterministic stand-in, so
    the jobs' plumbing runs in CI with no GPU and no downloads."""
    return settings().get("backend") == "fake"


def ensure_packages(requirements="requirements-gpu.txt"):
    """On the worker, install the jobs' Python packages once per instance."""
    if fake_models() or os.environ.get("VI_WORKER") != "1":
        return
    req = JOBS / requirements
    stamp = Path.home() / ".cache" / "vi" / (
        "pip-" + hashlib.sha256(req.read_bytes()).hexdigest()[:16]
    )
    if stamp.exists():
        return
    log(f"installing {req.name}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True
    )
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()


def unit_hash(*parts):
    """A float in [0, 1) from strings, for the fake models."""
    h = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64
