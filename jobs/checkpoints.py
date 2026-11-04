"""Checkpoint publishing and verification for voicetune runs (standard library only).

`voicetune train --publish DIR` writes each finished epoch into DIR as `epoch-<k>/`
with a SHA256SUMS manifest (files hard-linked, checksums written last, then the
`.part` dir renamed into place), so home can prove a checkpoint arrived intact. The
CLI here is the other half, run at home:

    python3 jobs/checkpoints.py gate   PATH --min-gb 50 [--reserve-gb 27]
    python3 jobs/checkpoints.py verify DIR --expect N
    python3 jobs/checkpoints.py prune  DIR --keep K [--yes]
    python3 jobs/checkpoints.py audit  ROOT

`gate` refuses to start below the free-space floor, after setting aside a reserve for projected checkpoint bytes; `verify` checks every epoch's
hashes (no symlinks, no leftover `.part` dirs, exactly the N epochs 0..N-1) and
writes VERIFIED;
`prune` keeps at most two verified epochs; `audit` lists runs holding more than two.
"""

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

# The files an epoch folder carries, in the order they are published and listed.
# `style.safetensors` only exists on --speaker style runs; the other two are required.
EPOCH_FILES = ("talker.safetensors", "tune.json", "style.safetensors")
REQUIRED_EPOCH_FILES = ("talker.safetensors", "tune.json")
# Run-level files copied (overwritten) into the publish dir after each epoch.
RUN_FILES = ("voice.json", "xvectors.json", "train_log.jsonl")


def sha256_file(path):
    """A file's SHA-256, hex, streamed so a ~3.4 GB safetensors stays out of memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(src, dst):
    """Hard-link src into dst, falling back to a copy when they cross a device."""
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.copyfile(src, dst)


def publish_epoch(src_epoch_dir, publish_dir):
    """Publish one saved epoch folder (and the run-level files) into `publish_dir`.

    The epoch's files land in `epoch-<k>.part/`, a SHA256SUMS is written last, then
    the folder is renamed to `epoch-<k>` so a reader never sees a half-written epoch."""
    src = Path(src_epoch_dir)
    publish = Path(publish_dir)
    for name in REQUIRED_EPOCH_FILES:
        if not (src / name).is_file():
            raise FileNotFoundError(f"{src / name}")
    publish.mkdir(parents=True, exist_ok=True)
    part = publish / (src.name + ".part")
    dst = publish / src.name
    if part.exists():
        shutil.rmtree(part)  # a previous attempt's debris
    part.mkdir()
    sums = {}
    for name in EPOCH_FILES:
        f = src / name
        if not f.is_file():
            continue  # only style.safetensors is optional
        _link_or_copy(f, part / name)
        sums[name] = sha256_file(part / name)
    (part / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    if dst.exists():
        shutil.rmtree(dst)  # re-publishing the same epoch replaces it
    os.rename(part, dst)
    for name in RUN_FILES:
        f = src.parent / name
        if f.is_file():
            shutil.copyfile(f, publish / name)


def free_gb(path):
    """Free space on `path`, in GiB, available to this user (f_bavail * f_frsize)."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def gate(path, min_gb, reserve_gb=0):
    """Print the free space; return False (with a message) when it is under
    `min_gb` GiB after reserving `reserve_gb` GiB for projected checkpoints."""
    gb = free_gb(path)
    print(f"{path}: {gb:.2f} GiB free")
    if gb - reserve_gb < min_gb:
        need = f"{min_gb} GiB"
        if reserve_gb:
            need += f" plus a {reserve_gb} GiB reserve"
        print(f"gate: {gb:.2f} GiB free, need {need}", file=sys.stderr)
        return False
    return True


def _epoch_key(name):
    """The epoch number of an `epoch-<k>[.part]` name, or None when it isn't one."""
    if not name.startswith("epoch-"):
        return None
    tail = name[len("epoch-"):]
    if tail.endswith(".part"):
        tail = tail[: -len(".part")]
    return int(tail) if tail.isdigit() else None


def _list_epochs(dir):
    """[(key, is_part, path)] for every `epoch-*` entry in `dir`, numeric order."""
    out = []
    for p in dir.iterdir():
        key = _epoch_key(p.name)
        if key is not None:
            out.append((key, p.name.endswith(".part"), p))
    return sorted(out)


def _symlinks_under(dir):
    """Paths of every symlink under `dir` (files or directories, not followed)."""
    out = []
    for root, dirs, files in os.walk(dir):
        for n in dirs + files:
            p = Path(root) / n
            if p.is_symlink():
                out.append(p)
    return out


def _check_epoch(ep):
    """Verify one epoch dir against its SHA256SUMS: ({filename: sha256}, [problems])."""
    problems = []
    sums = ep / "SHA256SUMS"
    if not sums.is_file():
        return {}, [f"{ep.name}/SHA256SUMS: missing"]
    per = {}
    listed = set()
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            problems.append(f"{ep.name}/SHA256SUMS: malformed line {line!r}")
            continue
        hexd, name = parts
        if "/" in name or ".." in name:
            problems.append(f"{ep.name}/SHA256SUMS: unsafe name {name!r}")
            continue
        listed.add(name)
        f = ep / name
        if not f.is_file():
            problems.append(f"{ep.name}/{name}: missing or not a regular file")
            continue
        if f.stat().st_size == 0:
            problems.append(f"{ep.name}/{name}: empty")
            continue
        got = sha256_file(f)
        per[name] = got
        if got != hexd:
            problems.append(f"{ep.name}/{name}: sha256 mismatch")
    for req in REQUIRED_EPOCH_FILES:
        if req not in listed:
            problems.append(f"{ep.name}/SHA256SUMS: does not list {req}")
    return per, problems


def _check(dir):
    """Verify a whole run dir: returns (problems, {epoch_name: {file: sha256}})."""
    problems = []
    epochs = {}
    if not dir.is_dir():
        return [f"{dir}: not a directory"], {}
    for p in _symlinks_under(dir):
        problems.append(f"{p}: is a symlink")
    for key, is_part, p in _list_epochs(dir):
        if p.is_symlink():
            continue  # already rejected above
        if is_part:
            problems.append(f"{p.name}: leftover partial dir")
            continue
        if not p.is_dir():
            problems.append(f"{p.name}: not a directory")
            continue
        per, errs = _check_epoch(p)
        problems += errs
        epochs[p.name] = per
    return problems, epochs


def _write_verified(dir, epochs):
    (dir / "VERIFIED").write_text(json.dumps(
        {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "epochs": epochs},
        indent=1) + "\n")


def state_dir():
    """The same state dir `vi` uses: `$VI_STATE_DIR`, else `$XDG_STATE_HOME/vi`,
    else `~/.local/state/vi`."""
    if os.environ.get("VI_STATE_DIR"):
        return Path(os.environ["VI_STATE_DIR"])
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "vi"
    return Path.home() / ".local" / "state" / "vi"


def _clear_markers(dir):
    """Drop every unverified marker whose recorded checkpoint path is exactly
    `dir`. Nothing else under `<state dir>/unverified/` is touched."""
    dir = dir.resolve()
    marker_dir = state_dir() / "unverified"
    if not marker_dir.is_dir():
        return
    for p in marker_dir.iterdir():
        if p.is_symlink() or not p.is_file():
            continue  # only regular files written by on-worker.sh
        recorded = p.read_text().strip()
        if not recorded:
            continue
        if Path(recorded).resolve() == dir:
            p.unlink()


def verify(dir, expect):
    """Check a run dir has exactly `expect` verified epochs, then write VERIFIED.
    Exits non-zero on any symlink, leftover `.part`, missing/empty/mismatched file,
    or epoch dirs not numbered 0..N-1."""
    dir = Path(dir)
    problems, epochs = _check(dir)
    if len(epochs) != expect:
        problems.append(f"expected {expect} epoch dirs, found {len(epochs)}")
    elif set(epochs) != {f"epoch-{k}" for k in range(expect)}:
        problems.append(
            f"epoch dirs must be epoch-0..epoch-{expect - 1}; found {', '.join(sorted(epochs))}"
        )
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(1)
    _write_verified(dir, epochs)
    _clear_markers(dir)
    print(f"verified: {len(epochs)} epochs ok ({', '.join(sorted(epochs))})")
    return epochs


def _keep_set(ks, keep):
    """The epoch numbers prune keeps: `keep` plus its next (or previous, at the end)."""
    ks = sorted(ks)
    if keep not in ks:
        sys.exit(f"prune: no epoch-{keep} to keep (have {', '.join(f'epoch-{k}' for k in ks)})")
    out = {keep}
    if keep == ks[-1]:
        if keep - 1 in ks:
            out.add(keep - 1)
    else:
        out.add(keep + 1)
    return out


def _rmtree_guarded(dir, p):
    """Remove one epoch dir, refusing anything that resolves outside `dir`."""
    root = dir.resolve()
    target = p.resolve()
    if p.is_symlink() or target.parent != root or not target.name.startswith("epoch-") or not target.is_dir():
        sys.exit(f"prune: refusing to delete {p} (outside {dir})")
    shutil.rmtree(p)


def prune(dir, keep, yes=False):
    """Keep at most two verified epochs, deleting the rest. Dry run unless `yes`."""
    dir = Path(dir)
    vf = dir / "VERIFIED"
    if vf.is_symlink() or not vf.is_file():
        sys.exit(f"prune: no {vf} (run verify first)")
    try:
        recorded = json.loads(vf.read_text())["epochs"]
    except (ValueError, KeyError, TypeError):
        sys.exit(f"prune: {vf} is not a valid verify record")
    problems, epochs = _check(dir)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(1)
    if set(epochs) != set(recorded) or any(epochs[n] != recorded[n] for n in epochs):
        sys.exit(f"prune: {vf} does not match the current epochs (re-run verify)")
    keep_epochs = _keep_set(sorted(_epoch_key(n) for n in epochs), keep)
    doomed = sorted(n for n in epochs if _epoch_key(n) not in keep_epochs)
    if not doomed:
        print(f"prune: nothing to delete (keeping {', '.join(f'epoch-{k}' for k in sorted(keep_epochs))})")
        return
    if not yes:
        for n in doomed:
            print(f"would delete {dir / n}")
        print(f"would keep {', '.join(f'epoch-{k}' for k in sorted(keep_epochs))}")
        return
    for n in doomed:
        _rmtree_guarded(dir, dir / n)
    _write_verified(dir, {n: epochs[n] for n in epochs if _epoch_key(n) in keep_epochs})
    print(f"prune: deleted {', '.join(doomed)}")


def audit(root):
    """[(run_dir, n_epochs)] for run dirs under `root` that hold a VERIFIED file
    and more than two epoch dirs."""
    out = []
    for run in sorted(Path(root).iterdir()):
        if not run.is_dir() or not (run / "VERIFIED").is_file():
            continue
        n = sum(1 for _, is_part, _ in _list_epochs(run) if not is_part)
        if n > 2:
            out.append((run, n))
    return out


def cmd_gate(a):
    if not gate(a.path, a.min_gb, a.reserve_gb):
        sys.exit(1)


def cmd_verify(a):
    verify(a.dir, a.expect)


def cmd_prune(a):
    prune(a.dir, a.keep, a.yes)


def cmd_audit(a):
    for run, n in audit(a.root):
        print(f"{run}: {n} epoch dirs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="refuse to start under the free-space floor")
    g.add_argument("path", type=Path)
    g.add_argument("--min-gb", type=float, default=50.0)
    g.add_argument("--reserve-gb", type=float, default=0.0)
    v = sub.add_parser("verify", help="check every epoch's SHA256SUMS and write VERIFIED")
    v.add_argument("dir", type=Path)
    v.add_argument("--expect", type=int, required=True)
    p = sub.add_parser("prune", help="keep at most two verified epochs")
    p.add_argument("dir", type=Path)
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--yes", action="store_true", help="delete instead of printing")
    u = sub.add_parser("audit", help="list run dirs holding more than two epochs")
    u.add_argument("root", type=Path)
    a = ap.parse_args(argv)
    cmds = {"gate": cmd_gate, "verify": cmd_verify, "prune": cmd_prune, "audit": cmd_audit}
    cmds[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
