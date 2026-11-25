"""Home-side gates before any rental: bundle cache, bundle ids, real-path, rehearsal, markers."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from paidrun.common import REPO, vi_state_dir
from paidrun.limits import REALPATH_TIMEOUT
from paidrun.config import stage_settings
from paidrun.hf import base_model_revision


def bundle_dir(home, env):
    """The bundle cache folder on-worker.sh uses ($VI_BUNDLE_DIR, else the XDG cache)."""
    return Path(env.get("VI_BUNDLE_DIR") or Path(env.get("XDG_CACHE_HOME")
                                                  or Path(home) / ".cache") / "vi" / "bundles")


def bundle_key(folder):
    """The bundle cache key of one input folder, computed as jobs/on-worker.sh does (read-only):
    sha256 of `pwd -P` plus the LC_ALL=C sorted `find . -printf '%P %s %T@\\n'` listing, first 16
    hex digits. %T@ is seconds with ten fractional digits (nanoseconds plus a trailing 0)."""
    top = Path(folder).resolve()

    def row(path, rel):
        st = os.lstat(path)
        ns = st.st_mtime_ns
        return os.fsencode(rel) + b" %d %d.%09d0\n" % (st.st_size, ns // 10**9, ns % 10**9)

    rows = [row(top, "")]  # find lists the folder itself with an empty %P
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            rows.append(row(path, os.path.relpath(path, top)))
    return hashlib.sha256(os.fsencode(top) + b"\n" + b"".join(sorted(rows))).hexdigest()[:16]


def bundle_inputs(stages, repo):
    """The stage inputs the GPU stages read, or None when the stage file cannot be read."""
    try:
        import tomllib
        p = Path(stages)
        p = p if p.is_absolute() else Path(repo) / p
        doc = tomllib.loads(p.read_text())
    except (ImportError, OSError, ValueError):
        return None, None
    names = []
    for stage in ("omnivoice_preflight", "omnivoice_train"):
        for i in doc.get("stage", {}).get(stage, {}).get("inputs", []):
            if i not in names:
                names.append(i)
    return p.parent / doc.get("corpus", "."), names


def bundle_cache_status(stages, repo, home, env):
    """One plan line: is every large input folder's bundle already packed for its current key?"""
    corpus, names = bundle_inputs(stages, repo)
    if not names:
        return "bundle cache    : not checked (stage file unreadable or lists no inputs)"
    try:
        min_kb = int(env.get("VI_PACK_MIN_KB") or 1024)
    except ValueError:
        min_kb = 1024
    cache = bundle_dir(home, env)
    keys, missing = [], []
    for name in names:
        cur = corpus / "stages" / name / "current"
        try:
            folder = cur.parent / cur.read_text().strip() / "out"
        except OSError:
            return (f"bundle cache    : not checked ({name} has not been built at home, so its "
                    "folder, and so the key, is unknown)")
        if not folder.is_dir():
            return f"bundle cache    : not checked ({folder} is missing)"
        r = subprocess.run(["du", "-sk", str(folder)], capture_output=True, text=True)
        try:
            kb = int(r.stdout.split()[0])
        except (IndexError, ValueError):
            return f"bundle cache    : not checked (du failed on {folder})"
        if kb < min_kb:
            continue  # on-worker.sh sends small inputs unpacked
        key = bundle_key(folder)
        keys.append(key)
        if not (cache / f"{key}.tar.zst").is_file() or (cache / f"{key}.tar.zst").stat().st_size == 0:
            missing.append(key)
    if not keys:
        return "bundle cache    : no input is large enough to pack (sent as plain files)"
    if missing:
        return ("bundle cache    : not built, will pack before renting (home CPU/disk only, no "
                "spend)")
    return f"bundle cache    : built and current ({', '.join(keys)})"


def load_bundle_ids():
    """jobs/bundle_ids.py by path (this script runs under `python3 -I`, so jobs/ is not on sys.path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bundle_ids", REPO / "jobs" / "bundle_ids.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def data_folder(stages, repo):
    """(folder, why_not): the built omnivoice_data folder the GPU stages read."""
    corpus, names = bundle_inputs(stages, repo)
    if not names or "omnivoice_data" not in names:
        return None, "stage file unreadable or does not list omnivoice_data"
    cur = corpus / "stages" / "omnivoice_data" / "current"
    try:
        folder = cur.parent / cur.read_text().strip() / "out"
    except OSError:
        return None, "omnivoice_data has not been built at home"
    return (folder, None) if folder.is_dir() else (None, f"{folder} is missing")


def ids_check(stages, repo):
    """(ok, lines): every sample id in the real data folder survives WebDataset's key rule."""
    folder, why = data_folder(stages, repo)
    if folder is None:
        return True, [f"clip ids        : not checked ({why})"]
    ok, problems, summary = load_bundle_ids().check_ids(folder)
    if ok:
        return True, [f"clip ids        : OK, {summary} (no '.', unique, labels cover tokens)"]
    return False, [f"clip ids        : FAIL in {folder}"] + [f"                  - {p}" for p in problems]


REALPATH_REQ = REPO / "jobs" / "requirements-omnivoice-home-cpu.txt"
REALPATH_SCRIPT = REPO / "jobs" / "omnivoice-ft" / "home_realpath_check.py"
REALPATH_MIN_RAM_MB = 4096
REALPATH_FORWARD_RAM_MB = 12288  # model + KL reference + gradients, measured ~9 GB peak by estimate


def mem_available_mb(meminfo="/proc/meminfo"):
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def realpath_venv(home, req=None):
    req = Path(req or REALPATH_REQ)
    return Path(home) / ".cache" / "vi" / f"home-cpu-venv-{hashlib.sha256(req.read_bytes()).hexdigest()[:16]}"


def realpath_code_hash(folder, req=None, repo=None):
    """Hash of everything the check exercises: our data/training code, the pins, the data folder id."""
    repo = Path(repo or REPO)
    h = hashlib.sha256()
    files = [repo / "jobs" / "omnivoice-ft" / n for n in ("ft_data.py", "train_ft.py",
                                                         "home_realpath_check.py", "data_manifest.py")]
    files += sorted((repo / "jobs" / "omnistyle").glob("*.py"))
    files.append(Path(req or REALPATH_REQ))
    # F-HOME-1: the worker's pins and the pinned base-model revision are part of the key too, so
    # a changed pin cannot reuse an OK that was earned with the old ones.
    files.append(repo / "jobs" / "requirements-omnivoice-worker.txt")
    for f in files:
        h.update(f.name.encode() + b"\0")
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"<missing>")
    h.update(Path(folder).parent.name.encode())
    h.update(b"revision\0" + base_model_revision(repo / "jobs" / "voicetune-fresh.sh").encode())
    return h.hexdigest()[:24]


REHEARSAL_CMD = "jobs/rehearsal/vast-limits.sh --expect green"


def rehearsal_state_dir():
    """Where jobs/rehearsal/vast-limits.sh keeps its pass records. VI_REHEARSAL_STATE_DIR moves
    it for the tests only; it does not skip the gate."""
    env = os.environ.get("VI_REHEARSAL_STATE_DIR")
    return Path(env) if env else Path.home() / ".local" / "state" / "vi" / "rehearsal"


def rehearsal_gate(repo):
    """(ok, line). A real run needs the home Vast-limits rehearsal to have passed, with default
    limits on a clean tree, for exactly this HEAD: <state dir>/<HEAD sha>.pass must be JSON whose
    head matches the file name and HEAD and whose tree matches HEAD^{tree}. There is no bypass."""
    label = "rehearsal       : "
    fix = f"run `{REHEARSAL_CMD}` on this commit first (about a minute, home CPU only)"

    def git(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    head, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    if not head or not tree:
        return False, f"{label}INVALID: {repo} has no git HEAD to check a pass record for; {fix}"
    path = rehearsal_state_dir() / f"{head}.pass"
    if not path.is_file():
        others = sorted(rehearsal_state_dir().glob("*.pass"), key=lambda q: q.stat().st_mtime)
        why = (f"STALE: the newest pass record is for {others[-1].stem[:12]}, not HEAD {head[:12]}"
               if others else f"MISSING: no pass record for HEAD {head[:12]}")
        return False, f"{label}{why} ({path}); {fix}"
    try:
        rec = json.loads(path.read_text())
        if not isinstance(rec, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        return False, f"{label}INVALID: {path} is not a readable JSON record ({e}); {fix}"
    if rec.get("head") != head:
        return False, (f"{label}INVALID: {path} names head {str(rec.get('head'))[:12]}, not HEAD "
                       f"{head[:12]}; {fix}")
    if rec.get("tree") != tree:
        return False, (f"{label}STALE: {path} was written for tree {str(rec.get('tree'))[:12]}, "
                       f"HEAD's tree is {tree[:12]}; {fix}")
    if rec.get("expect") != "green" or rec.get("verdict") != "PASS":
        return False, f"{label}INVALID: {path} is not a green PASS record; {fix}"
    return True, (f"{label}ok (green PASS for {head[:12]} at {rec.get('time', '?')}, "
                  f"stage {rec.get('stage_secs', '?')} s, tasks_max {rec.get('tasks_max', '?')})")


def run_realpath(cmd, timeout, env):
    """(rc, output) of the check subprocess; a timeout is rc 124."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        return 124, f"timed out after {timeout} s\n{(e.stdout or b'')[-1500:]!r}"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def realpath_check(stages, repo, forward=False, home=None, mem_mb=None, runner=run_realpath,
                   req=None, script=None):
    """(ok, lines): the worker's real data path on CPU, on a few real clips, before renting.

    Runs jobs/omnivoice-ft/home_realpath_check.py in the home CPU venv. The batch-only check runs
    on every call but its OK result is cached by code hash; the forward+backward step is opt-in and
    only runs with enough free RAM (otherwise the check stops at the collated batch and says so)."""
    label = "real-path check : "
    folder, why = data_folder(stages, repo)
    if folder is None:
        return True, [f"{label}not checked ({why})"]
    home = Path(home or Path.home())
    req = Path(req or REALPATH_REQ)
    script = Path(script or REALPATH_SCRIPT)
    venv = realpath_venv(home, req)
    py = venv / "bin" / "python"
    if not py.is_file():
        return False, [f"{label}NOT RUN (venv missing; build it once, it needs ~0.5 GB of downloads): "
                       f"uv venv -p 3.11 {venv} && uv pip install --index-strategy unsafe-best-match "
                       f"-p {py} -r {req}"]
    rev = base_model_revision()
    base = Path(os.environ.get("VI_BASE_MODEL") or home / ".cache" / "huggingface" / "hub"
                / "models--k2-fsa--OmniVoice" / "snapshots" / rev)
    if not (base / "audio_tokenizer").is_dir():
        return False, [f"{label}NOT RUN (the pinned base-model snapshot is missing at {base}; "
                       "set VI_BASE_MODEL to a local snapshot)"]
    ram = mem_available_mb() if mem_mb is None else mem_mb
    note = []
    if forward and ram < REALPATH_FORWARD_RAM_MB:
        note = [f"                  forward step skipped: {ram} MB RAM available, it needs "
                f"{REALPATH_FORWARD_RAM_MB} MB; the check stops at the collated batch"]
        forward = False
    code = realpath_code_hash(folder, req, Path(script).parents[2])
    cache = home / ".cache" / "vi" / "realpath-check"
    entry = cache / f"{code}.json"
    try:
        cached = json.loads(entry.read_text())
    except (OSError, ValueError):
        cached = {}
    if cached.get("ok") and (cached.get("forward") or not forward):
        return True, [f"{label}OK (cached, {'with' if cached.get('forward') else 'without'} forward "
                      f"step; code {code[:8]})"] + note
    if ram < REALPATH_MIN_RAM_MB:
        return False, [f"{label}NOT RUN (only {ram} MB RAM available, needs {REALPATH_MIN_RAM_MB} MB; "
                       "free memory and run --plan again)"]
    cmd = ["nice", "-n", "19", str(py), "-I", str(script), "--data", str(folder), "--base", str(base),
           "--scratch", str(cache / "scratch"), "--code-hash", code, "--json"]
    if forward:
        cmd.append("--forward")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", HOME=str(home))
    rc, out = runner(cmd, REALPATH_TIMEOUT[forward], env)
    if rc == 0:
        cache.mkdir(parents=True, exist_ok=True)
        entry.write_text(json.dumps({"ok": True, "forward": forward, "code": code,
                                     "when": time.strftime("%Y-%m-%d %H:%M:%S")}))
        last = [ln for ln in out.splitlines() if ln.startswith("real-path check: OK")]
        return True, [f"{label}{last[-1].split(': ', 1)[1] if last else 'OK'}"] + note
    tail = [ln for ln in out.splitlines() if ln.strip()][-12:]
    head = "REFUSED" if rc == 3 else "FAIL"
    return False, [f"{label}{head} (exit {rc}); last lines of the check:"] + [f"                  | {ln}" for ln in tail]


def marker_state(d):
    """What an unverified marker's checkpoint dir holds, read-only: "missing", "empty",
    "incomplete" (files but no SHA256SUMS in the dir or any checkpoint below it, e.g. dbg5's
    partial push) or "complete-unverified" (a manifest exists, `verify` has not passed yet)."""
    if not d.is_dir():
        return "missing"
    try:
        kids = list(d.iterdir())
    except OSError:
        return "missing"
    if not kids:
        return "empty"
    if (d / "SHA256SUMS").is_file() or any(
            k.is_dir() and not k.name.startswith(".") and (k / "SHA256SUMS").is_file() for k in kids):
        return "complete-unverified"
    return "incomplete"


def unverified_markers(state_dir=None):
    """Every unverified marker with its checkpoint dir and marker_state(): [(marker, dir, state)].
    Read-only."""
    root = Path(state_dir) if state_dir else vi_state_dir()
    out = []
    mdir = root / "unverified"
    if not mdir.is_dir():
        return out
    for m in sorted(mdir.iterdir()):
        if not m.is_file():
            continue
        try:
            d = Path(m.read_text().strip())
        except OSError:
            continue
        if str(d):
            out.append((m, d, marker_state(d)))
    return out


def stale_markers(state_dir=None, home=None):
    """Unverified markers whose checkpoint dir is missing or empty: [(marker, dir)]. Read-only."""
    return [(m, d) for m, d, st in unverified_markers(state_dir) if st in ("missing", "empty")]


def stale_marker_lines():
    """r31 F3: the plan lists EVERY unverified marker with its state (dbg6's plan said `none`
    while dbg5's incomplete marker existed). Read-only; abandon commands only for dead dirs."""
    found = unverified_markers()
    if not found:
        return ["stale markers   : none"]
    lines = [f"stale markers   : {len(found)} unverified marker(s); a plain `vi cloud down` refuses "
             "while any marker exists. Nothing is deleted here."]
    for m, d, st in found:
        lines.append(f"                  {st}: {m.name} -> {d}")
    dead = [d for _m, d, st in found if st in ("missing", "empty")]
    if dead:
        lines.append("                  If you agree the empty or missing ones are dead, run for each:")
        for d in dead:
            lines.append(f"                  python3 jobs/checkpoints.py abandon {d} "
                         "--reason 'empty failed-attempt dir'")
    return lines


def home_check(stages, repo, home=None, env=None):
    """(ok, lines): the home-side checks on-worker.sh makes after renting, made before renting.

    The home disk gate is jobs/checkpoints.py's own `gate` with the stages' min_home_gb and
    home_gate_reserve_gb; the bundle cache line computes the same key on-worker.sh does."""
    env = os.environ if env is None else env
    home = home or str(Path.home())
    lines, ok = [], True
    for stage in ("omnivoice_preflight", "omnivoice_train"):
        st = stage_settings(stages, repo, stage)
        if st is None:
            lines.append(f"home disk       : {stage}: stage settings unreadable, not checked")
            continue
        floor = st.get("min_home_gb")
        if floor is None:
            continue
        r = subprocess.run(
            [sys.executable, str(REPO / "jobs" / "checkpoints.py"), "--format",
             str(st.get("checkpoint_format", "omnivoice")), "gate", home, "--min-gb", str(floor),
             "--reserve-gb", str(st.get("home_gate_reserve_gb", 0))],
            capture_output=True, text=True)
        free = r.stdout.strip().splitlines()[-1].split(": ", 1)[-1] if r.stdout.strip() else "?"
        ok &= r.returncode == 0
        lines.append(f"home disk       : {stage}: {free}, floor {floor} + reserve "
                     f"{st.get('home_gate_reserve_gb', 0)} GiB: "
                     f"{'OK' if r.returncode == 0 else 'FAIL'}")
    cache = bundle_dir(home, env)
    lines.append(bundle_cache_status(stages, repo, home, env))
    bundles = sorted(cache.glob("*.tar.zst")) if cache.is_dir() else []
    if bundles:
        mb = sum(b.stat().st_size for b in bundles) // 1048576
        lines.append(f"bundle files    : {len(bundles)} bundle(s), {mb} MB in {cache}")
    return ok, lines
