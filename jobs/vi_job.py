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
import struct
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



def write_embeddings(folder, name, model, vectors):
    """Save {id: vector} as <name>.npy (float32, one row per clip) and
    <name>.ids (the row order), readable with numpy.load without numpy here."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = list(vectors)
    dim = len(vectors[ids[0]]) if ids else 0
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (len(ids), dim)
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    with open(folder / f"{name}.npy", "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1"))
        for i in ids:
            if len(vectors[i]) != dim:
                raise ValueError(f"{name}: {i} has {len(vectors[i])} dimensions, not {dim}")
            f.write(struct.pack(f"<{dim}f", *vectors[i]))
    (folder / f"{name}.ids").write_text("".join(f"{i}\n" for i in ids))
    return {"model": model, "clips": len(ids), "dim": dim}



class Checkpoint:
    """Per-clip results kept in VI_CHECKPOINT/<name>.jsonl as they're made, so
    a run that dies part way (worker lost, credit gone) skips them when it's
    rerun. on-worker.sh copies the folder home during the run. Values must be
    JSON; tuples come back as lists. Without VI_CHECKPOINT nothing is kept."""

    def __init__(self, name):
        self.name = name
        self.done = {}
        folder = env_path("VI_CHECKPOINT")
        self.path = folder / f"{name}.jsonl" if folder else None
        if self.path is None or not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            # A copy taken mid-write: drop the torn last line.
            data = data[: data.rfind(b"\n") + 1]
            self.path.write_bytes(data)
        for line in data.decode("utf-8").splitlines():
            try:
                r = json.loads(line)
                self.done[r["id"]] = r["v"]
            except (ValueError, KeyError, TypeError):
                continue
        if self.done:
            log(f"{name}: resuming, {len(self.done)} already done")

    def todo(self, clips):
        return [c for c in clips if c.id not in self.done]

    def run(self, clips, compute, chunk=50):
        """compute(list of clips) -> one result per clip, for the clips not
        done yet, saved after every chunk. Returns {id: result} for all."""
        todo = self.todo(clips)
        f = None
        if self.path is not None and todo:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            f = open(self.path, "a", encoding="utf-8", newline="\n")
        try:
            for start in range(0, len(todo), chunk):
                part = todo[start : start + chunk]
                results = compute(part)
                for clip, r in zip(part, results, strict=True):
                    self.done[clip.id] = r
                    if f:
                        f.write(json.dumps({"id": clip.id, "v": r}, ensure_ascii=False) + "\n")
                if f:
                    f.flush()
                    os.fsync(f.fileno())
                log(f"{self.name}: {len(clips) - len(todo) + start + len(part)}/{len(clips)}")
        finally:
            if f:
                f.close()
        return {c.id: self.done[c.id] for c in clips}


class ArrayCheckpoint:
    """Checkpoint for per-clip vectors (embed): each finished chunk is one
    immutable VI_CHECKPOINT/<name>/<n>.npz holding `ids` and one float16
    array per output, so the minute-by-minute copy home only sends new
    chunks and never rewrites a big file on the home disk. Needs numpy.
    Without VI_CHECKPOINT nothing is kept."""

    def __init__(self, name):
        import numpy as np

        self.np = np
        self.name = name
        self.done = {}
        folder = env_path("VI_CHECKPOINT")
        self.folder = folder / name if folder else None
        self.chunks = 0
        if self.folder is None or not self.folder.is_dir():
            return
        for path in sorted(self.folder.glob("*.npz")):
            try:
                with np.load(path) as z:
                    arrays = {k: z[k] for k in z.files if k != "ids"}
                    for row, cid in enumerate(z["ids"]):
                        self.done[str(cid)] = {k: v[row] for k, v in arrays.items()}
            except (OSError, ValueError, KeyError):
                log(f"{name}: skipping unreadable {path.name}")
                continue
            self.chunks = max(self.chunks, int(path.stem) + 1)
        if self.done:
            log(f"{name}: resuming, {len(self.done)} already done")

    def todo(self, clips):
        return [c for c in clips if c.id not in self.done]

    def run(self, clips, compute, chunk=32):
        """compute(list of clips) -> one {output name: vector} per clip, for
        the clips not done yet, saved after every chunk. Returns {id: dict}."""
        np = self.np
        todo = self.todo(clips)
        if self.folder is not None and todo:
            self.folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(todo), chunk):
            part = todo[start : start + chunk]
            results = compute(part)
            stacked = {}
            for clip, r in zip(part, results, strict=True):
                r = {k: np.asarray(v, dtype=np.float16) for k, v in r.items()}
                self.done[clip.id] = r
                for k, v in r.items():
                    stacked.setdefault(k, []).append(v)
            if self.folder is not None:
                tmp = self.folder / f".{self.chunks:05d}.tmp"
                with open(tmp, "wb") as f:
                    np.savez(f, ids=np.array([c.id for c in part]), **{k: np.stack(v) for k, v in stacked.items()})
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.folder / f"{self.chunks:05d}.npz")
                self.chunks += 1
            log(f"{self.name}: {len(clips) - len(todo) + start + len(part)}/{len(clips)}")
        return {c.id: self.done[c.id] for c in clips}


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
    # From prep, when it ran: the trimmed, levelled 16 kHz copy for models,
    # and prep's record (trim bounds, level, noise floor).
    audio16k: Path | None = None
    prep: dict | None = None
    # The harvest record as read, for stages that need more of it
    # (conversation, following, structural tags).
    line: dict = field(default_factory=dict, repr=False)

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


def load_clips(limit=True):
    """Harvest's lines, with prep's trim and 16 kHz copy when prep has run
    and clean's audio and transcript when clean has run. `limit=False`
    ignores the stage's `limit` setting.

    Clips prep found silent or clean dropped (`keep: false`) are left out.
    Without clean, a clip's duration is prep's trimmed one.
    """
    harvest = env_path("VI_INPUT_HARVEST")
    if harvest is None:
        sys.exit("this job needs the harvest stage as an input")
    clean = env_path("VI_INPUT_CLEAN")
    cleaned = {}
    if clean is not None:
        cleaned = {c["id"]: c for c in read_jsonl(clean / "clips.jsonl")}
    prep = env_path("VI_INPUT_PREP")
    prepped = {}
    if prep is not None:
        prepped = {p["id"]: p for p in read_jsonl(prep / "prep.jsonl")}

    clips = []
    for line in read_jsonl(harvest / "lines.jsonl"):
        audio = harvest / line["audio"]["path"]
        duration = line["audio"]["duration_s"]
        text = line.get("text", "")
        flags = list(line.get("flags", []))
        p = None
        if prep is not None:
            p = prepped.get(line["id"])
            if p is None or "audio16k" not in p:
                continue
            duration = p["duration_s"]
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
                audio16k=prep / p["audio16k"]["path"] if p else None,
                prep=p,
                line=line,
            )
        )
    n = settings().get("limit") if limit else None
    if n:
        clips = spread(clips, int(n))
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


def ensure_venv(requirements):
    """On the worker, run this job in its own virtualenv, for jobs whose
    packages clash with requirements-gpu.txt (qwen-tts pins an older
    transformers). The venv sees the image's packages, so torch isn't
    installed twice. Built once per instance, then the job re-runs itself
    inside it."""
    if fake_models() or os.environ.get("VI_WORKER") != "1":
        return
    req = JOBS / requirements
    venv = Path.home() / ".cache" / "vi" / (
        "venv-" + hashlib.sha256(req.read_bytes()).hexdigest()[:16]
    )
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    python = venv / "bin" / "python"
    if not (venv / ".ready").exists():
        log(f"building a virtualenv for {req.name}")
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", "--clear", str(venv)],
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "-r", str(req)], check=True
        )
        (venv / ".ready").touch()
    os.execv(str(python), [str(python), *sys.argv])


def unit_hash(*parts):
    """A float in [0, 1) from strings, for the fake models."""
    h = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64
