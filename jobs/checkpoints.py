"""Checkpoint publishing and verification for worker training runs (standard library only).

Two checkpoint layouts are understood. `omnivoice` (the default) is the
`jobs/omnivoice-ft` pipeline contract: `train_ft.py` writes `checkpoint-<step>/`
dirs whose SHA256SUMS manifest is written last and atomically, so a checkpoint
only exists once its manifest does. `epoch` is the legacy Qwen3 voicetune
layout (`epoch-<k>/` with `talker.safetensors`, `tune.json` and optionally
`style.safetensors`) still published by `jobs/voicetune.py` through
`publish_epoch`; it stays reachable behind `--format epoch` for that path.

The CLI here is the other half of the pipeline, run at home:

    python3 jobs/checkpoints.py [--format omnivoice|epoch] gate PATH --min-gb 50 [--reserve-gb 27] [--retention-gb 15]
    python3 jobs/checkpoints.py [--format ...] verify DIR --expect N
    python3 jobs/checkpoints.py [--format ...] verify-epoch DIR N
    python3 jobs/checkpoints.py [--format ...] prune DIR [--keep K] [--yes] [--exempt-step STEP]...
    python3 jobs/checkpoints.py abandon DIR --reason TEXT
    python3 jobs/checkpoints.py [--format ...] audit ROOT

`gate` refuses to start below the free-space floor, after setting aside a
reserve for projected checkpoint bytes and (in omnivoice mode) the ~15 GB
retention that prune's exempt steps 1000/2500 hold at home; `verify` checks
every checkpoint's hashes (no symlinks, no partial or unrecognized dirs,
exactly the expected count) and writes VERIFIED; `verify-epoch` checks one
numbered checkpoint and writes nothing, so the home supervisor can ack each
checkpoint as it arrives; `prune` keeps at most two verified checkpoints (the
two newest) plus the exempt steps; `abandon` marks a run dir as given up on and
clears its unverified marker while keeping every file; `audit` lists runs
holding more than two.

In omnivoice mode every file inside a checkpoint dir must be listed in its
SHA256SUMS and every listed file must hash cleanly (the pipeline's own
`verify_checkpoint` applies the same contract, so home and pipeline agree).
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

CHECKSUM_NAME = "SHA256SUMS"
# A complete OmniVoice checkpoint holds at least one file from each category
# (jobs/omnivoice-ft/train_ft.py REQUIRED_CHECKPOINT_FILES).
CKPT_REQUIRED = (
    ("config.json",),
    ("model.safetensors", "pytorch_model.bin"),
    ("optimizer.bin",),
    ("scheduler.bin",),
    ("random_states_0.pkl",),
    ("train_config.json",),
    ("ft_config.json",),
    ("tokenizer.json", "tokenizer_config.json", "fake_tokenizer.json"),
)
# Decision Q1b: prune exempts steps 1000 and 2500 so Brandon can pick by ear;
# they stay at home (~15 GB) until then.
EXEMPT_STEPS = (1000, 2500)
# The extra home disk the exemption retains (two checkpoints, ~15 GB), counted
# by `gate` so a run never starts below the floor plus the retention.
RETENTION_GB = 15.0

# Legacy Qwen3 voicetune layout (jobs/voicetune.py still calls publish_epoch).
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


def _refuse_under_checkpoints(path):
    """Exit when `path` resolves inside ~/.cache/vi/checkpoints. Only `prune`
    deletes there, so no other code may treat a path in that store as its own
    debris. On the worker that home store is absent, so this is a no-op."""
    root = (Path.home() / ".cache" / "vi" / "checkpoints").resolve()
    target = Path(path).resolve()
    if target == root or root in target.parents:
        sys.exit(f"refusing to delete inside {root}: {path}")


def publish_epoch(src_epoch_dir, publish_dir):
    """Publish one saved epoch folder (and the run-level files) into `publish_dir`.

    The epoch's files land in `epoch-<k>.part/`, a SHA256SUMS is written last, then
    the folder is renamed to `epoch-<k>` so a reader never sees a half-written epoch."""
    src = Path(src_epoch_dir)
    publish = Path(publish_dir)
    _refuse_under_checkpoints(publish)
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


def gate(path, min_gb, reserve_gb=0, retention_gb=0):
    """Print the free space; return False (with a message) when it is under
    `min_gb` GiB after reserving `reserve_gb` GiB for projected checkpoints and
    `retention_gb` GiB for checkpoints prune will keep at home."""
    gb = free_gb(path)
    print(f"{path}: {gb:.2f} GiB free")
    if gb - reserve_gb - retention_gb < min_gb:
        need = f"{min_gb} GiB"
        if reserve_gb:
            need += f" plus a {reserve_gb} GiB reserve"
        if retention_gb:
            need += f" plus {retention_gb} GiB retained checkpoints"
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


def _checkpoint_key(name):
    """The step number of a `checkpoint-<step>` name, or None when it isn't a
    clean one (e.g. a leftover `.part` dir; the omnivoice contract has no part
    protocol — a checkpoint exists once its SHA256SUMS does)."""
    if not name.startswith("checkpoint-"):
        return None
    tail = name[len("checkpoint-"):]
    return int(tail) if tail.isdigit() else None


def _list_epochs(dir):
    """[(key, is_part, path)] for every `epoch-*` entry in `dir`, numeric order."""
    out = []
    for p in dir.iterdir():
        key = _epoch_key(p.name)
        if key is not None:
            out.append((key, p.name.endswith(".part"), p))
    return sorted(out)


def _list_checkpoints(dir):
    """[(key, name, path)] for every `checkpoint-*` entry in `dir`; key is None
    when the name is not a clean `checkpoint-<digits>` (such entries are
    reported as problems, not skipped)."""
    out = []
    for p in dir.iterdir():
        if p.name.startswith("checkpoint-"):
            out.append((_checkpoint_key(p.name), p.name, p))
    return sorted(out, key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0, t[1]))


def _symlinks_under(dir):
    """Paths of every symlink under `dir` (files or directories, not followed)."""
    out = []
    for root, dirs, files in os.walk(dir):
        for n in dirs + files:
            p = Path(root) / n
            if p.is_symlink():
                out.append(p)
    return out


def _safe_rel_name(name):
    """A SHA256SUMS entry is safe when it is a relative path whose components
    never leave the folder (no absolute paths, no `..`, not the manifest itself)."""
    if name == CHECKSUM_NAME or name.startswith("/"):
        return False
    return all(part not in ("..", "") for part in name.split("/"))


def _read_sums(sums, label):
    """Parse a SHA256SUMS file: ({listed_name: hexdigest}, [problems])."""
    problems = []
    listed = {}
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            problems.append(f"{label}/{CHECKSUM_NAME}: malformed line {line!r}")
            continue
        hexd, name = parts
        if not _safe_rel_name(name):
            problems.append(f"{label}/{CHECKSUM_NAME}: unsafe name {name!r}")
            continue
        listed[name] = hexd
    return listed, problems


def _check_listed(ck, listed, problems):
    """Check every listed file exists, is non-empty and hashes cleanly.
    Returns {relpath: sha256} for the files that passed."""
    per = {}
    for name, hexd in sorted(listed.items()):
        f = ck / name
        if not f.is_file():
            problems.append(f"{ck.name}/{name}: missing or not a regular file")
            continue
        if f.stat().st_size == 0:
            problems.append(f"{ck.name}/{name}: empty")
            continue
        got = sha256_file(f)
        per[name] = got
        if got != hexd:
            problems.append(f"{ck.name}/{name}: sha256 mismatch")
    return per


def _check_epoch(ep):
    """Verify one epoch dir against its SHA256SUMS: ({filename: sha256}, [problems])."""
    problems = []
    sums = ep / CHECKSUM_NAME
    if not sums.is_file():
        return {}, [f"{ep.name}/{CHECKSUM_NAME}: missing"]
    listed, problems = _read_sums(sums, ep.name)
    per = _check_listed(ep, listed, problems)
    for req in REQUIRED_EPOCH_FILES:
        if req not in listed:
            problems.append(f"{ep.name}/{CHECKSUM_NAME}: does not list {req}")
    return per, problems


def _check_ckpt(ck):
    """Verify one checkpoint dir against its SHA256SUMS: ({relpath: sha256}, [problems]).

    The pipeline contract: every file in the folder is listed, the manifest is
    written last (a kill during save leaves no manifest at all), and one file
    from each required category must be present."""
    problems = []
    sums = ck / CHECKSUM_NAME
    if not sums.is_file():
        return {}, [f"{ck.name}/{CHECKSUM_NAME}: missing (a checkpoint is complete only when its manifest exists)"]
    listed, problems = _read_sums(sums, ck.name)
    per = _check_listed(ck, listed, problems)
    for group in CKPT_REQUIRED:
        if not any(name in listed for name in group):
            problems.append(f"{ck.name}/{CHECKSUM_NAME}: does not list {list(group)}")
    for p in sorted(ck.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ck).as_posix()
        if rel == CHECKSUM_NAME:
            continue
        if rel not in listed:
            problems.append(f"{ck.name}/{rel}: not listed in {CHECKSUM_NAME}")
    return per, problems


def _check(dir, fmt):
    """Verify a whole run dir: returns (problems, {checkpoint_name: {file: sha256}})."""
    problems = []
    found = {}
    if not dir.is_dir():
        return [f"{dir}: not a directory"], {}
    for p in _symlinks_under(dir):
        problems.append(f"{p}: is a symlink")
    if fmt == "epoch":
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
            found[p.name] = per
    else:
        for key, name, p in _list_checkpoints(dir):
            if p.is_symlink():
                continue  # already rejected above
            if key is None:
                problems.append(f"{name}: not a checkpoint-<step> dir")
                continue
            if not p.is_dir():
                problems.append(f"{name}: not a directory")
                continue
            per, errs = _check_ckpt(p)
            problems += errs
            found[name] = per
    return problems, found


def _write_verified(dir, found, fmt):
    field = "epochs" if fmt == "epoch" else "checkpoints"
    (dir / "VERIFIED").write_text(json.dumps(
        {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), field: found},
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


def verify(dir, expect, fmt="omnivoice"):
    """Check a run dir has exactly `expect` verified checkpoints, then write VERIFIED.
    Exits non-zero on any symlink, leftover/unrecognized dir, missing/empty/mismatched
    file, or (epoch mode) epoch dirs not numbered 0..N-1."""
    dir = Path(dir)
    problems, found = _check(dir, fmt)
    if fmt == "epoch":
        if len(found) != expect:
            problems.append(f"expected {expect} epoch dirs, found {len(found)}")
        elif set(found) != {f"epoch-{k}" for k in range(expect)}:
            problems.append(
                f"epoch dirs must be epoch-0..epoch-{expect - 1}; found {', '.join(sorted(found))}"
            )
    else:
        if len(found) != expect:
            problems.append(f"expected {expect} checkpoint dirs, found {len(found)}")
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(1)
    _write_verified(dir, found, fmt)
    _clear_markers(dir)
    word = "epochs" if fmt == "epoch" else "checkpoints"
    print(f"verified: {len(found)} {word} ok ({', '.join(sorted(found))})")
    return found


def abandon(dir, reason):
    """Mark a run dir ABANDONED: the explicit human decision to give up on it.
    Clears the exact unverified marker that names `dir` and keeps every file.
    `prune` refuses it (it is not VERIFIED), so only a human outside vi removes
    the files."""
    dir = Path(dir)
    vf = dir / "VERIFIED"
    if not dir.is_dir():
        sys.exit(f"abandon: {dir} is not a directory")
    if vf.is_file() or vf.is_symlink():
        sys.exit(f"abandon: {dir} is VERIFIED; nothing to abandon")
    (dir / "ABANDONED").write_text(json.dumps(
        {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": reason},
        indent=1) + "\n")
    _clear_markers(dir)
    print(f"abandoned: {dir} (files kept)")


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


def _keep_ckpts(ks, exempt):
    """The checkpoint steps prune keeps: the two newest, plus any exempt steps present."""
    ks = sorted(ks)
    keep = set(ks[-2:])
    for s in exempt:
        if s in ks:
            keep.add(s)
    return keep


def _rmtree_guarded(dir, p, prefix):
    """Remove one checkpoint dir, refusing anything that resolves outside `dir`."""
    root = dir.resolve()
    target = p.resolve()
    if p.is_symlink() or target.parent != root or not target.name.startswith(prefix) or not target.is_dir():
        sys.exit(f"prune: refusing to delete {p} (outside {dir})")
    shutil.rmtree(p)


def prune(dir, keep=None, yes=False, fmt="omnivoice", exempt=EXEMPT_STEPS):
    """Keep at most two verified checkpoints, deleting the rest. Dry run unless `yes`.
    Omnivoice mode keeps the two newest steps plus `exempt`; epoch mode keeps
    `keep` plus its neighbour."""
    dir = Path(dir)
    vf = dir / "VERIFIED"
    if vf.is_symlink() or not vf.is_file():
        sys.exit(f"prune: no {vf} (run verify first)")
    field = "epochs" if fmt == "epoch" else "checkpoints"
    try:
        recorded = json.loads(vf.read_text())[field]
    except (ValueError, KeyError, TypeError):
        sys.exit(f"prune: {vf} is not a valid verify record")
    problems, found = _check(dir, fmt)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(1)
    if set(found) != set(recorded) or any(found[n] != recorded[n] for n in found):
        sys.exit(f"prune: {vf} does not match the current checkpoints (re-run verify)")
    if fmt == "epoch":
        keep_epochs = _keep_set(sorted(_epoch_key(n) for n in found), keep)
        kept = {n for n in found if _epoch_key(n) in keep_epochs}
        doomed = sorted(n for n in found if n not in kept)
        keep_desc = ", ".join(f"epoch-{k}" for k in sorted(keep_epochs))
        prefix = "epoch-"
    else:
        keep_steps = _keep_ckpts(sorted(_checkpoint_key(n) for n in found), exempt)
        kept = {n for n in found if _checkpoint_key(n) in keep_steps}
        doomed = sorted(n for n in found if n not in kept)
        keep_desc = ", ".join(f"checkpoint-{k}" for k in sorted(keep_steps))
        prefix = "checkpoint-"
    if not doomed:
        print(f"prune: nothing to delete (keeping {keep_desc})")
        return
    if not yes:
        for n in doomed:
            print(f"would delete {dir / n}")
        print(f"would keep {keep_desc}")
        return
    for n in doomed:
        _rmtree_guarded(dir, dir / n, prefix)
    _write_verified(dir, {n: found[n] for n in kept}, fmt)
    print(f"prune: deleted {', '.join(doomed)}")


def audit(root, fmt="omnivoice", exempt=EXEMPT_STEPS):
    """[(run_dir, n)] for run dirs under `root` that hold a VERIFIED file and
    more than two checkpoint dirs (non-exempt ones, in omnivoice mode)."""
    out = []
    for run in sorted(Path(root).iterdir()):
        if not run.is_dir() or not (run / "VERIFIED").is_file():
            continue
        if fmt == "epoch":
            n = sum(1 for _, is_part, _ in _list_epochs(run) if not is_part)
        else:
            n = sum(1 for key, _, _ in _list_checkpoints(run) if key not in exempt)
        if n > 2:
            out.append((run, n))
    return out


def cmd_gate(a):
    retention = a.retention_gb
    if retention is None:
        retention = RETENTION_GB if a.format == "omnivoice" else 0.0
    if not gate(a.path, a.min_gb, a.reserve_gb, retention):
        sys.exit(1)


def verify_epoch(dir, key, fmt="omnivoice"):
    """Check one numbered checkpoint dir against its SHA256SUMS; return its file
    hashes. Writes no VERIFIED and clears no marker: a run is verified as a whole."""
    dir = Path(dir)
    name = f"checkpoint-{key}" if fmt == "omnivoice" else f"epoch-{key}"
    ep = dir / name
    if not dir.is_dir():
        sys.exit(f"verify-epoch: {dir} is not a directory")
    if ep.is_symlink() or not ep.is_dir():
        sys.exit(f"verify-epoch: {ep} is not a directory")
    checker = _check_ckpt if fmt == "omnivoice" else _check_epoch
    per, problems = checker(ep)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(1)
    print(f"{name}: ok ({len(per)} files)")
    return per


def cmd_verify(a):
    verify(a.dir, a.expect, a.format)


def cmd_verify_epoch(a):
    verify_epoch(a.dir, a.key, a.format)


def cmd_prune(a):
    exempt = tuple(a.exempt_step) if a.exempt_step else EXEMPT_STEPS
    if a.format == "omnivoice":
        if a.keep is not None:
            sys.exit("prune: --keep is epoch-mode only; omnivoice keeps the two newest checkpoints plus exempt steps")
    else:
        if a.keep is None:
            sys.exit("prune: --keep is required in epoch mode")
        if a.exempt_step:
            sys.exit("prune: --exempt-step is omnivoice-mode only")
    prune(a.dir, a.keep, a.yes, a.format, exempt)


def cmd_abandon(a):
    abandon(a.dir, a.reason)


def cmd_audit(a):
    exempt = tuple(a.exempt_step) if a.exempt_step else EXEMPT_STEPS
    word = "epoch dirs" if a.format == "epoch" else "checkpoint dirs"
    for run, n in audit(a.root, a.format, exempt):
        print(f"{run}: {n} {word}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--format", choices=("omnivoice", "epoch"), default="omnivoice",
                    help="checkpoint layout: omnivoice checkpoint-<step>/ (default) or the "
                         "legacy voicetune epoch-<k>/ layout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="refuse to start under the free-space floor")
    g.add_argument("path", type=Path)
    g.add_argument("--min-gb", type=float, default=50.0)
    g.add_argument("--reserve-gb", type=float, default=0.0)
    g.add_argument("--retention-gb", type=float, default=None,
                   help="checkpoints prune will keep at home (default 15 in omnivoice mode, 0 in epoch mode)")
    v = sub.add_parser("verify", help="check every checkpoint's SHA256SUMS and write VERIFIED")
    v.add_argument("dir", type=Path)
    v.add_argument("--expect", type=int, required=True)
    e = sub.add_parser("verify-epoch", help="check one checkpoint's SHA256SUMS; write nothing")
    e.add_argument("dir", type=Path)
    e.add_argument("key", type=int)
    p = sub.add_parser("prune", help="keep at most two verified checkpoints")
    p.add_argument("dir", type=Path)
    p.add_argument("--keep", type=int, default=None, help="epoch mode only: the pivot epoch to keep")
    p.add_argument("--yes", action="store_true", help="delete instead of printing")
    p.add_argument("--exempt-step", type=int, action="append", default=None,
                   help="omnivoice mode only: a step exempt from pruning (default 1000, 2500)")
    b = sub.add_parser("abandon", help="mark a run dir as given up on; keep every file")
    b.add_argument("dir", type=Path)
    b.add_argument("--reason", required=True, help="why this run is being abandoned")
    u = sub.add_parser("audit", help="list run dirs holding more than two checkpoints")
    u.add_argument("root", type=Path)
    u.add_argument("--exempt-step", type=int, action="append", default=None,
                   help="a step that does not count toward the limit (default 1000, 2500)")
    a = ap.parse_args(argv)
    cmds = {"gate": cmd_gate, "verify": cmd_verify, "verify-epoch": cmd_verify_epoch,
            "prune": cmd_prune, "abandon": cmd_abandon, "audit": cmd_audit}
    cmds[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
