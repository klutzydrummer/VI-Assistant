"""Free, CPU-only proof that the moss15 venv is complete, run at home (README "moss15 local smoke").

    nice -n 19 ionice -c2 -n7 python3 -I jobs/ttstest/smoke_moss15_local.py [--keep] [--workdir NEW_EMPTY_DIR]

Builds the same venv worker_run.py builds (uv, Python 3.12, requirements-moss15.txt) but with CPU torch
wheels, in a new temp dir outside the repo, then runs moss15_check.py with it. Downloads packages only, never
model weights. It cannot prove CUDA, the real model forward, flash-attn or the Vast image's apt/ffmpeg.
FFmpeg shared libraries must be reachable (system, or LD_LIBRARY_PATH set before calling this).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", help="new, empty directory for the venv (default: a fresh temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the venv afterwards")
    a = ap.parse_args(argv)
    uv = shutil.which("uv")
    if not uv:
        print("smoke: uv is needed on PATH", file=sys.stderr)
        return 2
    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="moss15-smoke-"))
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        print(f"smoke: {work} is not empty", file=sys.stderr)
        return 2
    venv = work / "venv"
    py = venv / "bin" / "python"
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(work / "uv-python"), UV_CACHE_DIR=str(work / "uv-cache"))
    try:
        for cmd in ([uv, "venv", "--python", "3.12", "--clear", venv],
                    [uv, "pip", "install", "--python", py, "--torch-backend", "cpu", "-r", HERE / "requirements-moss15.txt"]):
            if run(cmd, env=env).returncode != 0:
                print("smoke: venv build FAILED", file=sys.stderr)
                return 1
        # Same call and cwd rule as worker_run.py: the script path is passed, the interpreter runs isolated from the repo.
        r = run([py, HERE / "moss15_check.py"], env=env, cwd=work)
        return r.returncode
    finally:
        if a.keep:
            print(f"smoke: kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
