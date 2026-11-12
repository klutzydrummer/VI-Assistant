"""The TTS listening test as ONE job on the rented worker (README "phase B"). Runs under on-worker.sh:

    ../../jobs/on-worker.sh python3 jobs/ttstest/worker_run.py

on-worker.sh gives it VI_OUT (home receives it), VI_RETURN_PORT and, from a stage input named
`refs`, VI_INPUT_REFS (vP.wav, vT.wav, refs.json: sent home -> worker, never into the repo).

Flow: (1) per model, in its OWN venv and process: build the environment, download the weights,
then a smoke test (line L01, voice P, seed 1, time-limited, checked with verify.py). A model that
fails smoke twice is dropped and recorded. (2) per surviving model: full generation (voice P and T,
seed 1; T also seed 2 with --seed2), self-check with verify.py, a second copy in the keep dir
(default /root/vi-keep, which on-worker.sh's cleanup never touches), then push the model's folders
home through worker-return.sh. (3) last: top-level manifest.sha256, expected-count.txt and
run-info.json, pushed last. vT.wav is deleted from the worker at the very end (finally block).

Exit 0 only when every planned folder was made, verified and pushed. 3: a model was dropped or a
folder failed its check (the culprits are printed under "CULPRITS"). 4: a push failed (every copy
is kept). 2: bad setup. Failures never stop the other models. No Hugging Face token is used.

    worker_run.py --repush [KEEP]    re-send the kept copy through an open return path (README).
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate  # noqa: E402
import verify  # noqa: E402

JOBS = HERE.parent
RETURN = JOBS / "worker-return.sh"
MODELS = ("omnivoice", "moss15", "qwen3base")
CACHE = Path(os.environ.get("TTSTEST_CACHE") or Path.home() / ".cache" / "ttstest")
BACKOFF = (10, 30, 90)  # seconds before download retries 2 and 3 (a 4th wait is never used)
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# uv is installed from PyPI with the image's own pip (no curl|sh of "latest"), pinned to a release range, into
# its own small venv; it then fetches a managed Python 3.12 with `uv python install 3.12`.
# UNVERIFIED against the docs: this was written without network access. Check before the rental:
#   https://docs.astral.sh/uv/getting-started/installation/   ("pip install uv"; version pinning)
#   https://docs.astral.sh/uv/guides/install-python/          (`uv python install 3.12`)
#   https://docs.astral.sh/uv/reference/environment/          (UV_PYTHON_INSTALL_DIR)
UV_SPEC = os.environ.get("TTSTEST_UV_SPEC") or "uv>=0.9,<0.10"


def log(msg):
    print(f"worker_run: {msg}", file=sys.stderr, flush=True)


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def env_num(name, default):
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return float(default)


# --- running commands with a time limit, a log, and the whole process group killed on timeout ---------

class CmdFailed(RuntimeError):
    pass


def run_cmd(cmd, timeout, logfile, env=None, cwd=None):
    """Run cmd, its output appended to logfile. Raises CmdFailed (with the log tail) on failure/timeout."""
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "ab") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n".encode())
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=max(timeout, 1))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise CmdFailed(f"timed out after {int(timeout)} s: {tail(logfile)}")
    if rc != 0:
        raise CmdFailed(f"exit {rc}: {tail(logfile)}")


def tail(path, n=600):
    try:
        return Path(path).read_bytes()[-n:].decode(errors="replace").strip().replace("\n", " | ")
    except OSError:
        return ""


def with_retries(label, steps, logfile, env, deadline):
    """Run each (cmd) in steps, the whole list retried up to 3 times with backoff; never past deadline."""
    last = None
    for attempt in range(3):
        left = deadline - time.time()
        if left <= 0:
            raise CmdFailed(f"{label}: gave up after the download time limit ({last})")
        try:
            for cmd in steps:
                run_cmd(cmd, deadline - time.time(), logfile, env)
            return
        except CmdFailed as e:
            last = e
            log(f"{label}: attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(min(BACKOFF[attempt], max(deadline - time.time(), 0)))
    raise CmdFailed(f"{label}: failed 3 times: {last}")


# --- per-model environment ------------------------------------------------------------------------------

def clean_env(stub):
    """No Hugging Face token ever reaches a model process; the worker has none and must not need one."""
    env = dict(os.environ)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        env.pop(k, None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VI_WORKER"] = "1"
    if stub:
        env["TTSTEST_STUB"] = "1"
    return env


def make_venv(venv, system_site):
    """Create `venv` with the image's python (+ the image's packages, so torch is not fetched twice)."""
    cmd = [sys.executable, "-m", "venv"]
    if system_site:
        cmd.append("--system-site-packages")
    return cmd + ["--clear", str(venv)]


def provision_py312(logdir, env, deadline):
    """Make sure a Python 3.12 can be had. Returns the uv executable to use, or None when a system
    `python3.12` exists. Raises CmdFailed with a clear message if neither can be provisioned. Only public
    package/Python downloads happen here; no credentials of any kind are passed (env is clean_env's)."""
    if shutil.which("uv"):
        uv = shutil.which("uv")
    elif shutil.which("python3.12"):
        return None
    else:
        uvenv = CACHE / "venv-uv"
        uv = str(uvenv / "bin" / "uv")
        if not Path(uv).exists():
            try:
                with_retries("uv-install", [make_venv(uvenv, False),
                                            [uvenv / "bin" / "python", "-m", "pip", "install", "-q", UV_SPEC]],
                             logdir / "uv-setup.log", env, deadline)
            except CmdFailed as e:
                raise CmdFailed(f"moss15 needs Python 3.12 and could not install uv ({UV_SPEC}): {e}")
    env["UV_PYTHON_INSTALL_DIR"] = str(CACHE / "uv-python")
    try:
        with_retries("python3.12", [[uv, "python", "install", "3.12"]], logdir / "uv-setup.log", env, deadline)
    except CmdFailed as e:
        raise CmdFailed(f"moss15 needs Python 3.12 and `uv python install 3.12` failed: {e}")
    return uv


def ensure_ffmpeg_libs(logdir, env, deadline):
    """torchcodec (used by torchaudio 2.9 load/save) needs FFmpeg SHARED libraries (4-8 for torchcodec 0.8.1).
    The Vast image may not have them, so install Ubuntu's `ffmpeg` package with apt when libavcodec/libavutil
    are not in the linker cache. Needs root (the worker is root). Raises CmdFailed so moss15 is dropped early."""
    try:
        cache = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        cache = ""
    if "libavcodec.so" in cache and "libavutil.so" in cache and "libswresample.so" in cache:
        return
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise CmdFailed("moss15 needs FFmpeg shared libraries (torchcodec) and this user cannot apt-get install them")
    aenv = dict(env, DEBIAN_FRONTEND="noninteractive")
    with_retries("ffmpeg-apt", [["apt-get", "update", "-qq"],
                                ["apt-get", "install", "-y", "-qq", "--no-install-recommends", "ffmpeg"]],
                 logdir / "moss15-ffmpeg.log", aenv, deadline)


def check_moss15_env(py, logdir, env, deadline):
    """Import everything the moss15 path needs and round-trip a tiny WAV (moss15_check.py), BEFORE the ~24 GB of
    weights are fetched. Raises CmdFailed (model dropped, exit 3) with the check's own message."""
    run_cmd([py, HERE / "moss15_check.py"], min(deadline - time.time(), 600), logdir / "moss15-check.log", env,
            cwd=HERE.parent)


def prepare(model, stub, logdir, env, deadline):
    """Build the model's environment and download its weights. Returns (python, generate-args-prefix)."""
    gen = str(HERE / "generate.py")
    if stub and model == "moss15" and os.environ.get("TTSTEST_STUB_NO312") == "1":
        # Test hook (stub runs only): behave as if Python 3.12 could not be provisioned.
        raise CmdFailed("moss15 needs Python 3.12: it could not be provisioned (injected by TTSTEST_STUB_NO312)")
    if stub and model == "moss15" and os.environ.get("TTSTEST_STUB_BADENV") == "1":
        # Test hook (stub runs only): the environment check fails, as it does when torchcodec is missing.
        run_cmd([sys.executable, "-c", "import torchcodec_is_missing_for_this_test"], 60, logdir / "moss15-check.log", env)
    if stub:
        return sys.executable, []
    if model == "qwen3base":
        # jobs/vi_job.ensure_venv builds the venv from requirements-tune.txt and re-runs inside it.
        with_retries(model, [[sys.executable, gen, "--model", model, "--venv", "requirements-tune.txt",
                              "--download-only"]], logdir / f"{model}-setup.log", env, deadline)
        return sys.executable, ["--venv", "requirements-tune.txt"]
    venv = CACHE / f"venv-{model}"
    py = venv / "bin" / "python"
    ready = venv / ".ready"
    if not ready.exists():
        if model == "omnivoice":
            # The image's torch is reused (system site packages); only omnivoice and its small deps are fetched.
            steps = [make_venv(venv, True), [py, "-m", "pip", "install", "-q", "omnivoice"]]
        else:
            # MOSS-TTS-v1.5: its own Python 3.12 + torch 2.9.1 + transformers 5.0.0 (the MOSS README). No flash-attn.
            req = HERE / "requirements-moss15.txt"
            ensure_ffmpeg_libs(logdir, env, deadline)  # raises CmdFailed -> model dropped, exit 3
            uv = provision_py312(logdir, env, deadline)  # raises CmdFailed -> model dropped, exit 3
            if uv:
                steps = [[uv, "venv", "--python", "3.12", "--clear", venv],
                         [uv, "pip", "install", "--python", py, "--torch-backend", "cu128", "-r", req]]
            else:
                steps = [["python3.12", "-m", "venv", "--clear", venv],
                         [py, "-m", "pip", "install", "-q", "--extra-index-url", TORCH_INDEX, "-r", req]]
        with_retries(model, steps, logdir / f"{model}-setup.log", env, deadline)
        if model == "moss15":
            check_moss15_env(py, logdir, env, deadline)  # fail fast, before any weights; .ready is not set on failure
        ready.touch()
    elif model == "moss15":
        check_moss15_env(py, logdir, env, deadline)  # a reused venv is checked again (a few seconds)
    with_retries(model, [[py, gen, "--model", model, "--download-only"]],
                 logdir / f"{model}-download.log", env, deadline)
    return str(py), []


def gen_cmd(py, prefix, model, voice, seed, out, refs, only=None):
    cmd = [py, str(HERE / "generate.py"), "--model", model, "--voice", voice, "--seed", str(seed),
           "--out", str(out), "--refs", str(refs), *prefix]
    return cmd + (["--only", only] if only else [])


def versions(py, stub):
    """Package versions in one model's environment, for run-info.json."""
    code = ("import importlib.metadata as m, json\nout = {}\n"
            "for n in ('torch', 'torchaudio', 'transformers', 'qwen-tts', 'omnivoice', 'huggingface_hub'):\n"
            "    try: out[n] = m.version(n)\n    except Exception: pass\nprint(json.dumps(out))")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    except Exception:
        return {}


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# --- smoke test -----------------------------------------------------------------------------------------

def smoke(model, py, prefix, refs, root, env, limit):
    """Up to two attempts at line L01, voice P, seed 1. Returns (ok, [attempt records])."""
    records = []
    for n in (1, 2):
        out = root / "smoke" / f"{model}-try{n}"
        t0 = time.time()
        rec = {"attempt": n}
        try:
            run_cmd(gen_cmd(py, prefix, model, "P", 1, out, refs, "L01"), limit,
                    root / "logs" / f"{model}-smoke{n}.log", env)
            problems = verify.verify_dir(generate.run_dir(out, model, "P", 1))
            if problems:
                raise CmdFailed("smoke output failed verify.py: " + "; ".join(f"{f}: {p}" for f, p in problems))
            rec["ok"] = True
        except CmdFailed as e:
            rec.update(ok=False, error=str(e)[:900])
            ri = generate.run_dir(out, model, "P", 1) / "run-info.json"
            if ri.is_file():
                try:
                    rec["error"] = json.loads(ri.read_text()).get("error", rec["error"])
                except ValueError:
                    pass
            log(f"{model}: smoke attempt {n} FAILED: {rec['error'][:300]}")
        rec["seconds"] = round(time.time() - t0, 1)
        records.append(rec)
        if rec["ok"]:
            return True, records
    return False, records


# --- pushing home and keeping a second copy --------------------------------------------------------------

def rsync_push(out, filters, what, wait):
    env = dict(os.environ)
    if wait is not None:
        env["VI_PUSH_WAIT"] = str(wait)
    # --timeout / --contimeout: a half-dead tunnel must fail the push, not hang the job forever.
    r = subprocess.run(["bash", str(RETURN), "push", f"{out}/", "vi-out", "--timeout=120", "--contimeout=30", *filters],
                       env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise CmdFailed(f"push of {what} failed (rsync exit {r.returncode}): {r.stderr.strip()[-400:]}")


def push_folder(out, name, wait=None):
    """Push one top-level folder of `out` home (worker-return.sh -> the home receiver's vi-out).
    The manifest and count go in a second pass, so a half-pushed folder never looks finished."""
    one = [f"--include=/{name}/", f"--exclude=/{name}/manifest.sha256", f"--exclude=/{name}/expected-count.txt",
           f"--include=/{name}/***", "--exclude=*"]
    two = [f"--include=/{name}/", f"--include=/{name}/***", "--exclude=*"]
    for flt in (one, two):
        rsync_push(out, flt, name, wait)


def push_top(out, wait=None):
    """The run-level files, always the very last thing sent."""
    names = ("run-info.json", "manifest.sha256", "expected-count.txt")
    rsync_push(out, [*[f"--include=/{n}" for n in names], "--exclude=*"], "the run-level files", wait)


def keep_copy(out, keep, name):
    dst = Path(keep) / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(Path(out) / name, dst)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --- the job --------------------------------------------------------------------------------------------

def plan(models, seed2):
    """Every (model, voice, seed) folder this run promises."""
    runs = [("P", 1), ("T", 1)] + ([("T", 2)] if seed2 else [])
    return [(m, v, s) for m in models for v, s in runs]


def delete_vt(refs, info):
    """The private game clip never stays on the worker: delete vT.wav and its transcript."""
    refs = Path(refs)
    try:
        (refs / "vT.wav").unlink(missing_ok=True)
        rj = refs / "refs.json"
        if rj.is_file():
            d = json.loads(rj.read_text())
            d.pop("vT", None)
            rj.write_text(json.dumps(d, indent=2) + "\n")
        info["vT_deleted"] = not (refs / "vT.wav").exists()
    except Exception as e:
        info["vT_deleted"] = False
        log(f"ALERT: could not delete vT.wav: {e}")


def deadman(refs):
    """If this process dies by ANY signal (even SIGKILL), a detached watcher still deletes vT.wav."""
    subprocess.Popen(["bash", "-c", 'while kill -0 "$1" 2>/dev/null; do sleep 1; done; rm -f "$2"', "_",
                      str(os.getpid()), str(Path(refs) / "vT.wav")], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_job(a):
    stub = a.stub or os.environ.get("TTSTEST_STUB") == "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = out.parent / "ttstest-work"  # smoke runs and logs; deliberately outside the pushed folder
    keep = Path(a.keep)
    keep.mkdir(parents=True, exist_ok=True)
    refs = Path(a.refs)
    models = [m for m in a.models.split(",") if m]
    bad = [m for m in models if m not in MODELS]
    if bad or not (refs / "refs.json").is_file() or not (refs / "vP.wav").is_file() or not (refs / "vT.wav").is_file():
        log(f"setup problem: unknown models {bad}, or refs missing in {refs}")
        return 2
    deadman(refs)
    t_start = time.time()
    env = clean_env(stub)
    info = {"started": stamp(), "stub": stub, "seed": 1, "seed2": a.seed2, "python": platform.python_version(),
            "host": platform.node(), "gpu": gpu_info(), "models": {}, "pushed": [], "failed_push": [],
            "culprits": [], "timings_s": {}}
    smoke_limit = env_num("TTSTEST_SMOKE_SECONDS", 900)
    gen_limit = env_num("TTSTEST_GEN_SECONDS", 1800)
    dl_limit = env_num("TTSTEST_DOWNLOAD_MINUTES", 20) * 60
    # After one push has failed, the others wait only briefly: generation goes on, the end retries them all.
    short_wait = int(min(30, env_num("VI_PUSH_WAIT", 300)))
    for m in models:
        info["models"][m] = {"status": "planned"}

    # Phase 1: environment, download, smoke, one model at a time.
    pys = {}
    for m in models:
        t0 = time.time()
        rec = info["models"][m]
        try:
            py, prefix = prepare(m, stub, root / "logs", env, time.time() + dl_limit)
        except (CmdFailed, OSError) as e:
            rec.update(status="dropped", reason=f"setup or download failed: {str(e)[:900]}")
            log(f"{m}: DROPPED ({rec['reason'][:300]})")
            continue
        info["timings_s"][f"{m}_setup"] = round(time.time() - t0, 1)
        rec["versions"] = versions(py, stub)
        ok, rec["smoke"] = smoke(m, py, prefix, refs, root, env, smoke_limit)
        if not ok:
            rec.update(status="dropped", reason="failed the smoke test twice: " + rec["smoke"][-1]["error"][:600])
            log(f"{m}: DROPPED after two failed smoke tests")
            continue
        rec["status"] = "smoke-ok"
        pys[m] = (py, prefix)

    # Phase 2: full generation per surviving model, pushed as each finishes.
    for m in models:
        if m not in pys:
            continue
        py, prefix = pys[m]
        rec = info["models"][m]
        t0 = time.time()
        names = []
        for model, voice, seed in plan([m], a.seed2):
            name = generate.run_dir("", model, voice, seed).name
            names.append(name)
            try:
                run_cmd(gen_cmd(py, prefix, model, voice, seed, out, refs), gen_limit,
                        root / "logs" / f"{name}.log", env)
            except CmdFailed as e:
                rec.setdefault("failed_runs", {})[name] = str(e)[:900]
                log(f"{name}: FAILED: {str(e)[:300]}")
        rec["seconds"] = round(time.time() - t0, 1)
        info["timings_s"][f"{m}_generate"] = rec["seconds"]
        problems = []
        for name in names:
            d = out / name
            if not d.is_dir():
                problems.append((name, [("(folder)", "no output folder was left by the run")]))
                continue
            p = verify.verify_dir(d)
            if p:
                problems.append((name, p))
        for name, p in problems:
            info["culprits"] += [f"{name}/{f}: {why}" for f, why in p]
        rec["status"] = "failed-check" if problems else "generated"
        rec["problems"] = {n: [f"{f}: {w}" for f, w in p] for n, p in problems}
        # Second copy first (cleanup never touches it), then home. Damaged folders are sent too: home's
        # verify.py then names them, and the evidence is not lost.
        for name in names:
            if (out / name).is_dir():
                keep_copy(out, keep, name)
        for name in names:
            if not (out / name).is_dir():
                continue
            try:
                push_folder(out, name, wait=short_wait if info["failed_push"] else None)
                info["pushed"].append(name)
            except CmdFailed as e:
                info["failed_push"].append(name)
                info["culprits"].append(f"{name}: not pushed home ({e})")
                log(f"{name}: PUSH FAILED: {e}")

    delete_vt(refs, info)  # no model needs vT any more; also done again in main()'s finally block

    # Phase 3: one more try for anything not yet home, then the top-level files LAST.
    for name in list(info["failed_push"]):
        try:
            push_folder(out, name)
            info["failed_push"].remove(name)
            info["pushed"].append(name)
        except CmdFailed as e:
            log(f"{name}: final push attempt FAILED: {e}")
    planned = plan(models, a.seed2)
    per_folder = len(json.loads((HERE / "lines.json").read_text()))
    lines = []
    for p in sorted(out.glob("*/*.wav")):
        lines.append(f"{sha256(p)}  {p.parent.name}/{p.name}\n")
    info["finished"] = stamp()
    info["timings_s"]["total"] = round(time.time() - t_start, 1)
    info["expected_wavs"] = per_folder * len(planned)
    info["found_wavs"] = len(lines)
    dropped = [m for m, r in info["models"].items() if r["status"] == "dropped"]
    info["dropped"] = dropped
    info["exit"] = 4 if info["failed_push"] else 3 if (dropped or info["culprits"]) else 0
    (out / "run-info.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "manifest.sha256").write_text("".join(lines))
    (out / "expected-count.txt").write_text(f"{info['expected_wavs']}\n")
    for f in ("run-info.json", "manifest.sha256", "expected-count.txt"):
        shutil.copy2(out / f, keep / f)
    try:
        push_top(out)
    except CmdFailed as e:
        info["exit"] = 4
        log(f"top-level files not pushed: {e}")
    log(f"done: expected {info['expected_wavs']} WAVs, found {len(lines)}; dropped {dropped or 'none'}; exit {info['exit']}")
    if info["culprits"] or dropped or info["failed_push"]:
        print("CULPRITS:", file=sys.stderr)
        for m in dropped:
            print(f"  model {m} dropped: {info['models'][m].get('reason', '')[:200]}", file=sys.stderr)
        for c in info["culprits"]:
            print(f"  {c}", file=sys.stderr)
        print(f"  worker copies kept: {out} and {keep}", file=sys.stderr)
    return info["exit"]


def repush(keep):
    """Send everything in the keep dir again, through the return path that is open right now."""
    keep = Path(keep)
    if not keep.is_dir():
        log(f"no kept copy at {keep}")
        return 2
    failed = []
    for d in sorted(p for p in keep.iterdir() if p.is_dir()):
        try:
            push_folder(keep, d.name)
            log(f"re-pushed {d.name}")
        except CmdFailed as e:
            failed.append(d.name)
            log(f"{d.name}: FAILED: {e}")
    try:
        push_top(keep)
    except CmdFailed as e:
        failed.append("(top-level files)")
        log(f"top-level files: FAILED: {e}")
    return 4 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.environ.get("VI_OUT"))
    ap.add_argument("--refs", default=os.environ.get("VI_INPUT_REFS"))
    ap.add_argument("--keep", default=os.environ.get("TTSTEST_KEEP", "/root/vi-keep"))
    ap.add_argument("--models", default=os.environ.get("TTSTEST_MODELS", ",".join(MODELS)),
                    help="comma list; `--models moss15 --seed2` is the 39-clip moss15-only run (P s1, T s1, T s2)")
    ap.add_argument("--seed2", action="store_true", default=os.environ.get("TTSTEST_SEED2") == "1",
                    help="also run voice T with seed 2")
    ap.add_argument("--stub", action="store_true", help="sine-WAV stand-ins: no venvs, no downloads, no GPU")
    ap.add_argument("--repush", nargs="?", const=os.environ.get("TTSTEST_KEEP", "/root/vi-keep"), metavar="KEEP")
    a = ap.parse_args(argv)
    if a.repush:
        return repush(a.repush)
    if not (a.out and a.refs):
        ap.error("VI_OUT and VI_INPUT_REFS are needed (or --out and --refs)")
    # SIGTERM becomes an exit so the finally block still deletes vT.wav.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    gone = {}
    try:
        return run_job(a)
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        delete_vt(a.refs, gone)
        if not gone.get("vT_deleted"):
            log("ALERT: vT.wav may still be on the worker; delete it before anything else")


if __name__ == "__main__":
    sys.exit(main())
