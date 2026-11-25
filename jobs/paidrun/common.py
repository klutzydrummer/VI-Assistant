"""Shared paths, exit codes, stop exceptions, the log tee, clock and state-file helpers for paid_run."""

import os
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "jobs" / "paid-run" / "omnivoice-run-cloud.toml"
DEFAULT_STAGES = "characters/tali/omnivoice-ft.toml"


def nap(seconds):
    """Every retry/backoff wait goes through here. PAID_RUN_SLEEP_SCALE shrinks it for tests and
    PAID_RUN_SLEEP_LOG records each requested wait (seconds, one per line)."""
    log = os.environ.get("PAID_RUN_SLEEP_LOG")
    if log:
        with open(log, "a") as f:
            f.write(f"{seconds}\n")
    time.sleep(seconds * float(os.environ.get("PAID_RUN_SLEEP_SCALE", "1")))


def timeout_scale(seconds):
    """A per-call timeout; PAID_RUN_TIMEOUT_SCALE shrinks it for tests (never below 2 s)."""
    return min(seconds, max(seconds * float(os.environ.get("PAID_RUN_TIMEOUT_SCALE", "1")), 2.0))


EXIT = dict(ok=0, args=2, up=3, preflight=4, push=5, train=6, verify=7, interrupted=8,
            release=9, budget=10, disk=11, ids=12, realpath=13, hf=14, rehearsal=15, debug=16,
            other=1)


# Run logging. Everything this script prints is also appended to <run dir>/run.log (capped), and
# before a FAILED worker is destroyed a bounded diagnostics file is pulled into the run dir.
RUN_LOG_MAX_BYTES = 200 * 1024 * 1024


class Tee:
    """A stdout/stderr wrapper that also appends everything to the run log (best effort)."""

    def __init__(self, stream, path):
        self.stream, self.path, self.written, self.capped = stream, path, 0, False

    def write(self, text):
        n = self.stream.write(text)
        if not self.capped:
            try:
                with open(self.path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(text)
                self.written += len(text)
                if self.written > RUN_LOG_MAX_BYTES:
                    self.capped = True
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(f"\n[run.log reached {RUN_LOG_MAX_BYTES} bytes; no more is written]\n")
            except OSError:
                self.capped = True
        return n

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Stop(Exception):
    """Finish the run with this exit code and message."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class Interrupted(Exception):
    reason = "interrupted by a signal"


class StopRequested(Interrupted):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def proc_start(pid):
    """Start time (clock ticks since boot) of a process, from /proc; None if it is gone."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def proc_is_run(pid, run_id, started):
    """True only if pid is a live paid_run.py process for this run id that started when recorded."""
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return False
    return (any(a.endswith("paid_run.py") for a in argv) and run_id in argv
            and started is not None and proc_start(pid) == started)


def say(msg):
    print(f"paid-run: {msg}", flush=True)


def state_root():
    for var in ("VI_PAID_RUN_STATE_DIR",):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("VI_STATE_DIR"):
        base = Path(os.environ["VI_STATE_DIR"])
    elif os.environ.get("XDG_STATE_HOME"):
        base = Path(os.environ["XDG_STATE_HOME"]) / "vi"
    else:
        base = Path.home() / ".local" / "state" / "vi"
    return base / "paid-run"


def vi_state_dir():
    if os.environ.get("VI_STATE_DIR"):
        return Path(os.environ["VI_STATE_DIR"])
    if os.environ.get("XDG_STATE_HOME"):
        return Path(os.environ["XDG_STATE_HOME"]) / "vi"
    return Path.home() / ".local" / "state" / "vi"


def clock():
    """Wall clock. Tests add a fake offset (seconds) kept in $PAID_RUN_CLOCK_FILE."""
    t = time.time()
    f = os.environ.get("PAID_RUN_CLOCK_FILE")
    if f:
        try:
            t += float(Path(f).read_text().strip() or 0)
        except (OSError, ValueError):
            pass
    return t


def rfc3339(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def state_path(run_id):
    return state_root() / f"{run_id}.json"


def stop_path(run_id):
    return state_root() / f"{run_id}.stop"


def resolve_vi_bin():
    """$VI_BIN wins; else <repo>/target/release/vi if it is an executable file; else `vi`."""
    explicit = os.environ.get("VI_BIN")
    if explicit:
        return explicit
    release = REPO / "target" / "release" / "vi"
    if release.is_file() and os.access(release, os.X_OK):
        return str(release)
    return "vi"
