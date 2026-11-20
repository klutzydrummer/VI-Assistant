#!/usr/bin/env python3
"""One unattended command for the paid OmniVoice fine-tune flow (standard library only).

    python3 -I jobs/paid_run.py --run-id ID --cap USD --max-hours H [options]

It drives the existing `vi` binary ($VI_BIN, else <repo>/target/release/vi if built, else `vi`) so a stub can stand in for tests:
rent the cheapest qualifying offer, run the preflight stage (tunnel, receiver, token push, the
worker gates), then the train stage, then release the worker and print what it cost. If ONLY the
cheap worker gate fails (stderr says `worker-preflight: ...`: disk, RAM, CUDA, VRAM, cores, push
speed; or `worker upload ETA` for a too-slow upload line), the machine is released, that offer is
excluded, and the next cheapest is tried, up to --max-attempts. Any other failure releases the
worker and stops. Before renting anything it also checks home's free disk (the stage's
min_home_gb plus reserve; exit 11) and shows the bundle cache and the pinned base-model revision.

--cap and --max-hours cover ALL attempts together. A state file records every attempt, so a
killed run can be started again with the same --run-id: a worker still up is adopted (if the
train stage had started) or destroyed first, and a second rental is never made on top of it.

Exit codes:
  0 done        2 bad arguments         3 `vi cloud up` failed (boot failure included)
  4 a preflight failure that is not a retryable worker gate
  5 a cheap worker gate failed on every attempt allowed (--max-attempts)
  6 the train stage failed             7 train finished but home checkpoints/listening sets
                                         did not verify
  8 interrupted (SIGINT/SIGTERM); worker released
  9 THE WORKER COULD NOT BE RELEASED: check `vi cloud status` now
 10 no offer or budget left to try (cap, hours, or offers exhausted)
  11 home or the rented disk is too small (checked before renting)
  12 a clip id in the data folder would break WebDataset (checked before renting)
  13 the home real-path check did not pass (checked before renting)
  14 the pinned base-model revision is gone or changed on Hugging Face (checked before renting)
 15 the home Vast-limits rehearsal has no valid pass record for this HEAD (checked before
    renting; run `jobs/rehearsal/vast-limits.sh --expect green` on a clean tree)
  1 anything unexpected (worker release is still attempted)

A failed train stage never rents a second machine. It is recovered IN PLACE on the same worker
(GPU visible, disk free, stage process gone, a verified checkpoint to resume from; at most
INPLACE_MAX_RESTARTS restarts within INPLACE_MAX_SECONDS). Anything else (bounds hit, ENOSPC,
abort.json, a deterministic traceback, no checkpoint, a vanished instance) releases the worker
and stops with exit 6 so that a person decides.

Watching and stopping a run (no vi call unless --credit; neither command can rent, retry or
change the cap, hours or attempts):

    python3 -I jobs/paid_run.py status --run-id ID [--json] [--credit]
    python3 -I jobs/paid_run.py stop   --run-id ID [--now]

`stop` writes <run-id>.stop beside the state file; the run waits for the current checkpoint push
(at most 10 minutes), releases the worker and exits 8. `stop --now` sends SIGTERM to the recorded
paid_run process, which releases the worker at once.

Heavy work happens inside `vi dataset run` (stages `omnivoice_preflight`, `omnivoice_train`).
No Vast credential file is read here; only the `vi` CLI is called.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "jobs" / "paid-run" / "omnivoice-run-cloud.toml"
DEFAULT_STAGES = "characters/tali/omnivoice-ft.toml"
PUSH_GATE_TEXT = "worker push (MB/s)"
# Every failed cheap gate of jobs/worker-preflight.sh prints `worker-preflight: <what failed>`
# (disk, RAM, CUDA, VRAM, cores, push speed, missing tools). They are facts about one offer, so
# the next offer may pass; the success line `worker-preflight: ok` is not a failure.
PREFLIGHT_GATE_RE = re.compile(r"worker-preflight: (?!ok\b)\S")
# on-worker.sh prints this when the timed 8 MB probe says the inputs would take longer than
# max_upload_minutes: also a per-offer fact (this worker's line), also before anything is sent.
UPLOAD_GATE_TEXT = "worker upload ETA"


def retryable_gate(out):
    """The gate text that makes a failed preflight stage retry on another offer, or None."""
    if PUSH_GATE_TEXT in out:
        return "push-speed"
    if UPLOAD_GATE_TEXT in out:
        return "upload-ETA"
    if PREFLIGHT_GATE_RE.search(out):
        return "worker-preflight"
    return None
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


def gate_kind(out):
    """How a failed worker gate should mark the machine: 'cuda' (machine and host), 'memory'
    (VRAM/RAM: raise the next VRAM floor too), or 'other' (machine only)."""
    if CUDA_GATE_RE.search(out):
        return "cuda"
    if MEMORY_GATE_RE.search(out):
        return "memory"
    return "other"


EU_COUNTRIES = ("US AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI "
                "ES SE GB CH NO").split()
# Home keeps the two newest checkpoints plus the exempt steps (jobs/checkpoints.py
# EXEMPT_STEPS = 1000, 2500) when a 5000-step run ends.
EXPECTED_STEPS = (1000, 2500, 4500, 5000)
LISTENING_STEPS = (1000, 2500, 5000)
EXPECT_EPOCHS = 4

OFFER_RE = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+(\d+) GB\s+\$([\d.]+)/hr\s+reliability ([\d.]+)\s+"
    r"net down\s+(\d+) / up\s+(\d+) Mbps\s+([\d.]+) cores\s+(\d+) GB ram\s+"
    r"disk (\d+) MB/s\s*(.*)$")

# Trainer output the train stage streams (train_ft.py runs upstream's OmniTrainer, which logs
# through tqdm/accelerate; save_steps is 500 in train_ft.py). Two shapes are read, the one with
# the larger total wins when a line holds both an epoch bar and a step bar:
#   tqdm bar:   "  3%|▎  | 150/5000 [02:10<1:10:00,  1.15it/s]"
#   plain text: "Step 150/5000" / "step: 150 of 5000"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[\d+:\d+")
STEP_RE = re.compile(r"\b[Ss]teps?\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)")
# on-worker.sh prints this on home once a checkpoint is verified at home and acked to the worker.
ACK_RE = re.compile(r"on-worker\.sh: checkpoint (\d+) verified home and acked")
CKPT_EVERY = 500
PROGRESS_SECONDS = 30.0
STOP_WAIT_SECONDS = 600.0
MIN_SAMPLES = 3

EXIT = dict(ok=0, args=2, up=3, preflight=4, push=5, train=6, verify=7, interrupted=8,
            release=9, budget=10, disk=11, ids=12, realpath=13, hf=14, rehearsal=15, other=1)

# ---- retries, timeouts and recovery bounds (every wait goes through nap(), so tests never wait) ----
OUTER_RETRY_SECONDS = 20          # F-NET-1: one more try of `offers` / `status` after this long
RELEASE_BACKOFF = (10, 20, 40, 80)  # F-REL-1: waits between `down --force` tries before exit 9
QUERY_TIMEOUT_SECONDS = 180       # F-CLK-1: offers / status
DOWN_TIMEOUT_SECONDS = 300        # a hung `down` would bill forever
PREFLIGHT_TIMEOUT_SECONDS = 90 * 60
# r7: the tokeniser's pool died (pthread_create failed -> BrokenProcessPool) and the stage then
# printed nothing for hours. A fatal line followed by this long with no new output ends the
# preflight as a non-retryable failure (fork F2: a crash is a code or limits bug, not a bad
# offer). The same words as VI_FATAL_RE in jobs/worker-limits.sh. PAID_RUN_FATAL_QUIET_SECONDS
# overrides the wait for tests.
FATAL_RE = re.compile(r"BrokenProcessPool|pthread_create failed|can't start new thread|Traceback")
FATAL_QUIET_SECONDS = 180
UP_EXTRA_SECONDS = 5 * 60         # `up` may take boot_minutes plus this
DEFAULT_BOOT_MINUTES = 15
INPLACE_MAX_RESTARTS = 2          # F-RES-1 / F-TRN-2: in-place train restarts on the same worker
INPLACE_MAX_SECONDS = 45 * 60     # ... and the window, from the first failure, they must fit in
REATTACH_MAX = 3                  # F-PUSH-5: stage died but the trainer lives; reattach this often
REATTACH_WAIT_SECONDS = 15
BOOT_FAIL_RE = re.compile(r"no ssh after|disappeared while booting|failed to start|boot (?:timeout|failure)",
                          re.I)
INPLACE_MIN_DISK_GB = 5.0         # a worker with less free disk than this is treated as ENOSPC
HF_REPO = "k2-fsa/OmniVoice"
HF_API = "https://huggingface.co/api/models"  # the tests point it at a local stub server
HF_TRIES = 3
HF_TIMEOUT_SECONDS = 60
HF_BACKOFF = (5, 15)
VRAM_FLOOR_STEP_GB = 4.0          # F-OOM-1: how much a memory failure raises the next offer's VRAM floor
CUDA_GATE_RE = re.compile(r"worker-preflight: .*\b(CUDA|driver)\b", re.I)
MEMORY_GATE_RE = re.compile(r"worker-preflight: (?:vram peak \S+ of \S+ GiB|.*\b(?:VRAM|RAM)\b)", re.I)
# Worker check for an in-place train restart, run through `vi cloud ssh`.
INPLACE_SCRIPT = r"""
exec 2>&1
# INPLACE_CHECK
if timeout 20 nvidia-smi -L >/dev/null 2>&1; then echo "inplace: gpu=ok"; else echo "inplace: gpu=missing"; fi
echo "inplace: disk_free_gb=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -n1 | tr -dc 0-9)"
n=$(pgrep -fc '[w]orker-job.sh|[t]rain_ft.py|[v]oicetune-fresh.sh' 2>/dev/null); echo "inplace: procs=${n:-0}"
ab=$(find "$HOME" /tmp /workspace -maxdepth 8 -name abort.json -path '*voicetune-fresh*' 2>/dev/null | head -n1)
echo "inplace: abort=${ab:-none}"
if find "$HOME" /tmp /workspace -maxdepth 6 -name job.log 2>/dev/null | head -n3 | xargs -r grep -l 'No space left on device' >/dev/null 2>&1; then
  echo "inplace: enospc=1"; else echo "inplace: enospc=0"; fi
ck=$(find "$HOME" /tmp /workspace -maxdepth 8 -path '*voicetune-fresh/run/checkpoint-*' -name SHA256SUMS 2>/dev/null \
  | sed -E 's#.*/checkpoint-([0-9]+)/SHA256SUMS#\1#' | sort -n | tail -n1)
echo "inplace: ckpt=${ck:-none}"
"""

# The worker's preflight (jobs/voicetune-fresh.sh, `min_disk = float(os.environ.get(
# "VI_MIN_DISK_GB", "100"))`) refuses a worker with less free disk than this many GiB. The
# default is read from that script so the two cannot drift; GATE_FALLBACK is only used if the
# script is missing, and a test fails if it differs from the script.
GATE_SCRIPT = REPO / "jobs" / "voicetune-fresh.sh"
GATE_FALLBACK_GB = 100.0
# Measured on omnivoice-ft-r4: a 100 GB rented disk showed 95 GB free at the gate.
DISK_OVERHEAD_GB = 5.0
# Vast's offer filter default when a run config sets no disk_gb (crates/cloud/src/config.rs).
CONFIG_DEFAULT_DISK_GB = 60.0


# Run logging. Everything this script prints is also appended to <run dir>/run.log (capped), and
# before a FAILED worker is destroyed a bounded diagnostics file is pulled into the run dir.
RUN_LOG_MAX_BYTES = 200 * 1024 * 1024
DIAG_MAX_BYTES = 256 * 1024
DIAG_TIMEOUT_SECONDS = 25.0
# Run on the worker through `vi cloud ssh`; every part is bounded by head/tail and `timeout`.
DIAG_SCRIPT = r"""
exec 2>&1
echo "== date: $(date -u +%FT%TZ)"
echo "== nvidia-smi (raw)"
timeout 10 nvidia-smi 2>&1 | head -n 80 || echo "(nvidia-smi failed or is missing)"
echo "== pip freeze, worker venv(s)"
for v in "$HOME"/.cache/vi/venv-*; do
  [ -x "$v/bin/python" ] || continue
  echo "-- $v"
  timeout 15 "$v/bin/python" -m pip freeze 2>&1 | head -n 400
done
echo "== limits (r7: a cgroup pids.max that counts threads)"
ulimit -a 2>&1 | head -n 30
echo "nproc: $(nproc 2>&1)"
for f in pids.max pids.current cpu.max cpuset.cpus.effective; do
  echo "/sys/fs/cgroup/$f: $(head -c 200 /sys/fs/cgroup/$f 2>&1)"
done
echo "threads-max: $(cat /proc/sys/kernel/threads-max 2>&1)"
echo "threads on the machine: $(ps -eLf 2>/dev/null | wc -l)"
echo "-- job tree (pid ppid threads state wait command)"
timeout 10 ps -eo pid,ppid,nlwp,stat,wchan:20,cmd 2>&1 | grep -E 'PID|vi-jobs|omnivoice|python|accelerate|timeout|on-worker' | grep -v grep | head -n 80
echo "== job and stage logs (newest two job dirs, last 20 KB of each log)"
for d in $(ls -dt "${VI_WORKER_ROOT:-/root/vi-jobs}"/*/ 2>/dev/null | head -n 2); do
  for f in "$d"job.log "$d"out/*.log "$d"out/*/*.log; do
    [ -f "$f" ] || continue
    echo "-- $f"
    tail -c 20000 "$f"
  done
done
""" 


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


def parse_step(line):
    """(step, total) from one trainer output line, or None."""
    found = [(int(m.group(1)), int(m.group(2))) for rx in (TQDM_RE, STEP_RE)
             for m in rx.finditer(line)]
    found = [f for f in found if f[1] > 0 and f[0] <= f[1]]
    return max(found, key=lambda f: f[1]) if found else None


def eta_seconds(samples, step, total):
    """Rough seconds left from the recent [time, step] samples; None until MIN_SAMPLES."""
    if not samples or len(samples) < MIN_SAMPLES or not total or step is None:
        return None
    (t0, s0), (t1, s1) = samples[0], samples[-1]
    if t1 <= t0 or s1 <= s0:
        return None
    return max(0.0, (total - step) * (t1 - t0) / (s1 - s0))


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


MACHINE_TAIL_RE = re.compile(r"\s+machine (\d+) host (\d+|\?)\s*$")


def parse_offers(text):
    """[{id, gpu, dph, up, cores, ram, loc, sig, machine, host}] in the order `vi cloud offers`
    printed them. The line ends with `machine N host M` (host `?` when Vast gave none); older
    output without it gives machine/host None."""
    out = []
    for line in text.splitlines():
        m = OFFER_RE.match(line)
        if not m:
            continue
        loc = m.group(11).strip()
        tail = MACHINE_TAIL_RE.search(loc)
        machine = host = None
        if tail:
            machine = int(tail.group(1))
            host = None if tail.group(2) == "?" else int(tail.group(2))
            loc = loc[:tail.start()].strip()
        out.append(dict(
            id=int(m.group(1)), gpu=m.group(2).strip(), dph=float(m.group(4)),
            up=int(m.group(7)), cores=m.group(8), ram=m.group(9), loc=loc,
            machine=machine, host=host,
            # A relisted machine is also recognised by what it is and where it is, for output
            # that carries no machine id.
            sig=f"{m.group(2).strip()}|{loc}|{m.group(8)}|{m.group(9)}|{m.group(7)}"))
    return out


def set_toml_values(text, max_hours=None, countries=None):
    """A copy of a cloud config with top-level max_hours and/or search.countries replaced."""
    lines = text.splitlines()
    if max_hours is not None:
        new = f"max_hours = {max_hours:.4f}"
        for i, ln in enumerate(lines):
            if ln.startswith("[") :
                lines.insert(i, new)
                break
            if re.match(r"\s*max_hours\s*=", ln):
                lines[i] = new
                break
        else:
            lines.append(new)
    if countries is not None:
        new = "countries = [" + ", ".join(f'"{c}"' for c in countries) + "]"
        in_search, at = False, None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("["):
                if in_search:
                    break
                in_search = s == "[search]"
                if in_search:
                    at = i + 1
            elif in_search and re.match(r"countries\s*=", s):
                lines[i] = new
                at = None
                break
        else:
            pass
        if at is not None:
            lines.insert(at, new)
        elif not any(ln.strip() == "[search]" for ln in lines):
            lines += ["[search]", new]
    return "\n".join(lines) + "\n"


def set_search_value(text, key, value):
    """A copy of a cloud config with `key = value` set inside its [search] table."""
    lines = text.splitlines()
    new = f"{key} = {value}"
    in_search, at = False, None
    for i, ln in enumerate(lines):
        t = ln.strip()
        if t.startswith("["):
            if in_search:
                break
            in_search = t == "[search]"
            if in_search:
                at = i + 1
        elif in_search and re.match(rf"{re.escape(key)}\s*=", t):
            lines[i], at = new, None
            break
    if at is not None:
        lines.insert(at, new)
    elif not any(ln.strip() == "[search]" for ln in lines):
        lines += ["[search]", new]
    return "\n".join(lines) + "\n"


def search_value(text, key, default=None):
    in_search = False
    for ln in text.splitlines():
        t = ln.strip()
        if t.startswith("["):
            in_search = t == "[search]"
        elif in_search and re.match(rf"{re.escape(key)}\s*=", t):
            try:
                return float(t.split("=", 1)[1].split("#")[0].strip())
            except ValueError:
                return default
    return default


def config_boot_minutes(text):
    for ln in text.splitlines():
        if ln.startswith("["):
            break
        m = re.match(r"boot_minutes\s*=\s*(\d+)", ln)
        if m:
            return int(m.group(1))
    return DEFAULT_BOOT_MINUTES


def gate_default_gb(script=GATE_SCRIPT):
    """The free-disk gate default (GiB) written in the worker preflight script."""
    try:
        m = re.search(r'"VI_MIN_DISK_GB",\s*"([\d.]+)"', Path(script).read_text())
    except OSError:
        m = None
    return float(m.group(1)) if m else GATE_FALLBACK_GB


def config_disk_gb(text):
    """The top-level disk_gb of a cloud config (the Vast default when it sets none)."""
    for ln in text.splitlines():
        if ln.startswith("["):
            break
        m = re.match(r"\s*disk_gb\s*=\s*([\d.]+)", ln)
        if m:
            return float(m.group(1))
    return CONFIG_DEFAULT_DISK_GB


def disk_check(config_text, env=None):
    """(ok, disk_gb, gate_gb, line): can the rented disk pass the worker's free-disk gate?"""
    env = os.environ if env is None else env
    disk = config_disk_gb(config_text)
    try:
        gate = float(env.get("VI_MIN_DISK_GB") or gate_default_gb())
    except ValueError:
        gate = gate_default_gb()
    ok = disk - DISK_OVERHEAD_GB >= gate
    line = (f"worker disk     : disk_gb {disk:g} - overhead {DISK_OVERHEAD_GB:g} vs gate "
            f"{gate:g} GiB free: {'OK' if ok else 'FAIL'}")
    return ok, disk, gate, line


# Assumed ceiling for Vast's storage price, in $/GB/month. The real storage_cost is only known
# from the live offer, so this is a bound for the plan, not a quote.
DISK_CEILING_PER_GB_MONTH = 1.00
HOURS_PER_MONTH = 730


def disk_cost_line(disk_gb, hours):
    """The plan line about disk billing: Vast charges storage on top of the GPU price, per GB and
    second, until the instance is destroyed (stopping does not end it)."""
    cost = disk_gb * DISK_CEILING_PER_GB_MONTH / HOURS_PER_MONTH * hours
    return (f"disk billing    : Vast bills disk on top of the GPU price, per GB-hour until the "
            f"instance is destroyed; worst case for {disk_gb:g} GB over {hours:g} h is about "
            f"${cost:.2f} at an assumed ${DISK_CEILING_PER_GB_MONTH:.2f}/GB/month (estimate; "
            "the real storage_cost is only known from the live offer)")


def base_model_revision(script=GATE_SCRIPT):
    """The base-model revision pinned in the stage script (shown at --plan, not checked)."""
    try:
        m = re.search(r"VI_BASE_MODEL_REVISION:-([0-9a-f]{12,40})", Path(script).read_text())
    except OSError:
        m = None
    return m.group(1) if m else "unknown"


def stage_settings(stages, repo, stage):
    """The [stage.<stage>.settings] table of the stage file, or None when it cannot be read."""
    try:
        import tomllib
        p = Path(stages)
        p = p if p.is_absolute() else Path(repo) / p
        return tomllib.loads(p.read_text())["stage"][stage].get("settings", {})
    except (ImportError, OSError, KeyError, ValueError):
        return None


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
REALPATH_TIMEOUT = {False: 900, True: 1800}


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


# F-NET-4: the pinned base-model revision must still exist upstream, with the pinned file sizes.
CHECKPOINT_BYTES_CALCULATED = 3 * 2_450_344_112  # weights + two AdamW states (WP3 / w27 report)
WEIGHTS_BYTES_CALCULATED = 2_450_344_112


def load_size_table():
    """(SIZE_TABLE, DEFAULT_REVISION_PREFIX) from jobs/omnivoice-ft/fetch_base_model.py."""
    spec = importlib.util.spec_from_file_location(
        "fetch_base_model_pins", REPO / "jobs" / "omnivoice-ft" / "fetch_base_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SIZE_TABLE), mod.DEFAULT_REVISION_PREFIX


class RevisionNotFoundError(Exception):
    """The Hub answered 404 for the pinned revision (X-Error-Code: RevisionNotFound)."""


class RepositoryNotFoundError(Exception):
    """The Hub answered 401 or 403: the repo is gone, private or gated."""


def hf_fetch(url, body, timeout):
    """GET url (POST when body is a dict) and return the parsed JSON reply. Standard library only,
    because the paid run starts under the system python3 -I, which has no huggingface_hub (r8)."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        why = e.headers.get("X-Error-Message") or e.reason
        if e.code == 404:
            raise RevisionNotFoundError(f"HTTP 404 {e.headers.get('X-Error-Code', '')}: {why}")
        if e.code in (401, 403):
            raise RepositoryNotFoundError(f"HTTP {e.code}: {why}")
        raise  # 5xx, 429 and the like are retried as network failures


def hf_model_info(repo, revision, timeout, fetch=None):
    """The one network step of the gate (tests replace it): the commit the pinned revision
    resolves to, and the sizes of the pinned files at that commit.

    Both endpoints are in the official Hub API spec, https://huggingface.co/.well-known/openapi.md
    (linked from https://huggingface.co/docs/hub/api, read 2026-11-20):
      GET  /api/models/{namespace}/{repo}/commits/{rev}     -> [{"id": <commit sha>, ...}]
      POST /api/models/{namespace}/{repo}/paths-info/{rev}  {"paths": [...]}
                                                            -> [{"path": ..., "size": ...}]
    The spec lists no response fields, so the field names were checked against live replies on
    2026-11-20: commits gives 404 with X-Error-Code RevisionNotFound for an unknown revision,
    an unknown repo gives 401, and paths-info leaves out paths it does not have."""
    fetch = fetch or hf_fetch
    base = f"{HF_API}/{repo}"
    commits = fetch(f"{base}/commits/{revision}?limit=1", None, timeout)
    sha = commits[0]["id"] if commits else None
    table, _ = load_size_table()
    files = fetch(f"{base}/paths-info/{sha or revision}", {"paths": sorted(table)}, timeout)
    return SimpleNamespace(sha=sha, siblings=[{"rfilename": f.get("path"), "size": f.get("size")}
                                              for f in files])


def hf_offline_check(revision=None):
    """(ok, line) for --plan: runs the whole gate on canned Hub replies, with no network, so a
    missing module or a broken parser shows up before anything is rented."""
    rev = revision or base_model_revision()
    urls = []

    def canned(url, body, timeout):
        urls.append(url)
        if "/commits/" in url:
            return [{"id": rev, "title": "canned"}]
        table, _ = load_size_table()
        return [{"type": "file", "path": p, "size": table[p]} for p in body["paths"]]

    try:
        ok, lines = hf_revision_gate(revision=rev, sleep=lambda _s: None,
                                     model_info=lambda r, v, t: hf_model_info(r, v, t, canned))
    except Exception as e:  # noqa: BLE001: any crash here would crash the real gate too
        ok, lines = False, [f"{type(e).__name__}: {e}"]
    label = "hf check code   : "
    if ok and len(urls) == 2:
        return True, f"{label}OK offline on canned Hub replies (no network call in --plan)"
    return False, (f"{label}BROKEN offline ({lines[0].removeprefix('hf revision    : ')}); the "
                   f"real check before renting would stop with exit {EXIT['hf']}")


def hf_revision_gate(home=None, revision=None, model_info=None, sleep=None):
    """(ok, lines). A revision that is gone or has other file sizes fails. A network failure
    after HF_TRIES tries is only a warning when the home HF cache holds the snapshot."""
    model_info = model_info or hf_model_info
    sleep = sleep or nap
    label = "hf revision    : "
    rev = revision or base_model_revision()
    home = Path(home or Path.home())
    snap = (home / ".cache" / "huggingface" / "hub" / "models--k2-fsa--OmniVoice" / "snapshots" / rev)
    last = None
    for i in range(HF_TRIES):
        try:
            info = model_info(HF_REPO, rev, HF_TIMEOUT_SECONDS)
            break
        except Exception as e:  # noqa: BLE001: classified below
            last = e
            if isinstance(e, ImportError):  # ModuleNotFoundError included
                # Retrying cannot fix a missing library, and a cached snapshot does not prove
                # the pinned revision still exists, so this is a hard stop (r7 lost its gate here).
                return False, [f"{label}could not be checked: {type(e).__name__}: {e}. The check "
                               "needs only the Python standard library, so this is a bug in "
                               "jobs/paid_run.py; fix it and run again. Nothing was rented."]
            if type(e).__name__ in ("RevisionNotFoundError", "RepositoryNotFoundError",
                                    "EntryNotFoundError", "GatedRepoError"):
                return False, [f"{label}NOT FOUND: {HF_REPO} @ {rev} ({type(e).__name__}: {e}); the "
                               "worker would fail after being rented. Nothing was rented."]
            if i + 1 < HF_TRIES:
                sleep(HF_BACKOFF[min(i, len(HF_BACKOFF) - 1)])
    else:
        if snap.is_dir():
            return True, [f"{label}WARNING not checked ({type(last).__name__}: {last}); the home "
                          f"cache holds the snapshot {snap}"]
        return False, [f"{label}could not be checked after {HF_TRIES} tries ({type(last).__name__}: "
                       f"{last}) and the home cache has no snapshot at {snap}. Nothing was rented."]
    sha = getattr(info, "sha", None)
    if not sha or not sha.startswith(rev):
        return False, [f"{label}MISMATCH: {HF_REPO} @ {rev} resolved to commit {sha}, not the "
                       "pinned one. Nothing was rented."]
    table, prefix = load_size_table()
    if not rev.startswith(prefix):
        return True, [f"{label}ok ({HF_REPO} @ {rev} exists; the pinned size table is for {prefix}, "
                      "so sizes were not compared)"]
    sizes = {}
    for f in info.siblings or []:
        name, size = (f.get("rfilename"), f.get("size")) if isinstance(f, dict) else (
            getattr(f, "rfilename", None), getattr(f, "size", None))
        sizes[name] = size
    bad = [f"{name}: {sizes.get(name)} bytes upstream, pinned {want}"
           for name, want in table.items() if sizes.get(name) != want]
    if bad:
        return False, [f"{label}CHANGED upstream: " + "; ".join(bad) + ". Nothing was rented."]
    return True, [f"{label}ok ({HF_REPO} @ {rev}; pinned file sizes match)"]


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


def push_hours_line(settings, bytes_full=CHECKPOINT_BYTES_CALCULATED,
                    bytes_weights=WEIGHTS_BYTES_CALCULATED):
    """`push ≈ X h at floor` for one full checkpoint and one weights-only checkpoint, at the
    stage's min_push_mb_s. The sizes are CALCULATED (3 x the 2,450,344,112-byte weights file),
    not measured; jobs/checkpoints.py only enforces lower bounds."""
    floor = float((settings or {}).get("min_push_mb_s", 2.5))
    full, weights = bytes_full / (floor * 1e6) / 3600, bytes_weights / (floor * 1e6) / 3600
    return (f"push time       : push ≈ {full:.2f} h at floor ({floor:g} MB/s) per full checkpoint "
            f"({bytes_full / 1e9:.2f} GB), {weights:.2f} h per weights-only one "
            f"({bytes_weights / 1e9:.2f} GB); sizes calculated, not measured (jobs/checkpoints.py "
            f"floors {'2.0 GB weights + 4.0 GB optimizer'})")


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


def stale_markers(state_dir=None, home=None):
    """Unverified markers whose checkpoint dir is missing or empty: [(marker, dir)]. Read-only."""
    root = Path(state_dir) if state_dir else vi_state_dir()
    out = []
    mdir = root / "unverified"
    if not mdir.is_dir():
        return out
    for m in sorted(mdir.iterdir()):
        try:
            d = Path(m.read_text().strip())
        except OSError:
            continue
        if m.is_file() and str(d) and (not d.is_dir() or not any(d.iterdir())):
            out.append((m, d))
    return out


def stale_marker_lines():
    found = stale_markers()
    if not found:
        return ["stale markers   : none"]
    lines = [f"stale markers   : {len(found)} unverified marker(s) name an empty or missing "
             "checkpoint dir; a plain `vi cloud down` refuses while any marker exists. Nothing is "
             "deleted here. If you agree they are dead, run for each:"]
    for m, d in found:
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


def resolve_vi_bin():
    """$VI_BIN wins; else <repo>/target/release/vi if it is an executable file; else `vi`."""
    explicit = os.environ.get("VI_BIN")
    if explicit:
        return explicit
    release = REPO / "target" / "release" / "vi"
    if release.is_file() and os.access(release, os.X_OK):
        return str(release)
    return "vi"


class PaidRun:
    def __init__(self, args):
        self.a = args
        self.vi = resolve_vi_bin()
        self.dir = state_root() / args.run_id
        self.path = state_root() / f"{args.run_id}.json"
        self.child = None
        self.timed_out = False
        self.fatal_line = self.fatal_stopped = None
        self.last_facts = {}
        self.worker_maybe_up = False
        self.in_teardown = False
        self.config_path = None  # the per-attempt config currently in force
        self.s = None
        self.stop_file = stop_path(args.run_id)
        self.step = self.total = None
        self.samples = []
        self.last_ack = 0
        self.tail = deque(maxlen=30)
        self.last_progress = None  # monotonic time of the last progress write
        self.stop_deadline = self.stop_target = None
        self.stop_wait = float(os.environ.get("PAID_RUN_STOP_WAIT_SECONDS", STOP_WAIT_SECONDS))
        self.progress_every = float(os.environ.get("PAID_RUN_PROGRESS_SECONDS", PROGRESS_SECONDS))

    # ----- state ---------------------------------------------------------------------------
    def load_or_create(self):
        if self.path.exists():
            self.s = json.loads(self.path.read_text())
            for key, want in (("cap", self.a.cap), ("max_hours", self.a.max_hours)):
                if abs(self.s[key] - want) > 1e-9:
                    raise Stop(EXIT["args"], f"run {self.a.run_id} was started with {key} "
                               f"{self.s[key]}, not {want}; use the same limits or a new --run-id")
            for key, default in (("failed_machines", []), ("failed_hosts", []),
                                 ("vram_floor_gb", None)):
                self.s.setdefault(key, default)
            say(f"resuming run {self.a.run_id} from {self.path}")
            return True
        self.s = dict(run_id=self.a.run_id, cap=self.a.cap, max_hours=self.a.max_hours,
                      max_attempts=self.a.max_attempts, created=clock(), status="running",
                      exit_code=None, message="", attempts=[], failed_offers=[], failed_sigs=[],
                      failed_machines=[], failed_hosts=[], vram_floor_gb=None,
                      spent_total=0.0, hours_total=0.0, credit_before_run=None,
                      credit_after_run=None)
        return False

    def recorded_live_attempt(self):
        """The open attempt in this run's state file (its worker may still bill), or None. Reads
        the file only; no vi or Vast call."""
        try:
            s = json.loads(self.path.read_text())
            return next((at for at in s.get("attempts") or []
                         if isinstance(at, dict) and at.get("ended") is None), None)
        except (OSError, ValueError, AttributeError):
            return None

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=1) + "\n")
        os.replace(tmp, self.path)

    # ----- running vi ----------------------------------------------------------------------
    def env(self):
        env = dict(os.environ)
        if self.config_path:
            env["VI_CLOUD_CONFIG"] = str(self.config_path)
        env.update(getattr(self, "extra_env", {}))
        return env

    def run(self, argv, stream=True, watch=False, train=False, timeout=None, fatal_quiet=None):
        """Run a command, returning (returncode, combined stdout+stderr text). With watch=True a
        stop request is noticed even while the command prints nothing; train=True also reads the
        trainer's progress and checkpoint acks from its output. A `timeout` (seconds, scaled by
        PAID_RUN_TIMEOUT_SCALE) kills the command's whole process group: rc 124, and
        self.timed_out is set for the caller. With `fatal_quiet` (seconds) a fatal line
        (FATAL_RE, seen by note_line) followed by that long with no output also kills the group:
        rc 124, and self.fatal_stopped holds the fatal line."""
        self.timed_out = False
        self.fatal_line, self.fatal_stopped = None, None
        limit = timeout_scale(timeout) if timeout else None
        last_line_at = time.monotonic()
        deadline = time.monotonic() + limit if limit else None
        p = subprocess.Popen(argv, cwd=self.a.repo, env=self.env(), text=True, bufsize=1,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, start_new_session=bool(limit))
        self.child = p
        buf = []
        lines = queue.Queue()

        def pump():
            for ln in p.stdout:
                lines.put(ln)
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        try:
            while True:
                try:
                    line = lines.get(timeout=0.2 if (watch or limit) else None)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    last_line_at = time.monotonic()
                    buf.append(line)
                    if stream:
                        print(line, end="", flush=True)
                    if train or watch:
                        self.note_line(line)
                if watch:
                    self.check_stop()
                if fatal_quiet and self.fatal_line \
                        and time.monotonic() - last_line_at >= fatal_quiet:
                    self.fatal_stopped = self.fatal_line
                    msg = (f"paid-run: a fatal line ({self.fatal_line.strip()[:200]}) and no output "
                           f"for {fatal_quiet:.0f} s; killing the stage's process group\n")
                    buf.append(msg)
                    if stream:
                        print(msg, end="", flush=True)
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except OSError:
                        p.kill()
                    p.wait()
                    return 124, "".join(buf)
                if deadline is not None and time.monotonic() > deadline:
                    self.timed_out = True
                    msg = (f"paid-run: {' '.join(map(str, argv[:4]))} timed out after {limit:.0f} s; "
                           "killing its process group\n")
                    buf.append(msg)
                    if stream:
                        print(msg, end="", flush=True)
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except OSError:
                        p.kill()
                    p.wait()
                    return 124, "".join(buf)
            p.wait()
        except BaseException:
            if p.poll() is None:
                try:
                    if limit:
                        os.killpg(p.pid, signal.SIGTERM)
                    else:
                        p.terminate()
                except OSError:
                    p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise
        finally:
            self.child = None
            if train or watch:
                self.write_progress(force=True)  # also leaves stage-tail.log for `status`
        return p.returncode, "".join(buf)

    # ----- progress and stop requests ------------------------------------------------------
    def note_line(self, line):
        self.tail.append(line.rstrip("\n"))
        if FATAL_RE.search(line):
            self.fatal_line = line
        got = parse_step(line)
        if got:
            self.step, self.total = got
        m = ACK_RE.search(line)
        if m:
            self.last_ack = max(self.last_ack, int(m.group(1)))
        self.write_progress()

    def set_phase(self, at, phase):
        at["phase"] = phase
        self.write_progress(force=True)  # also saves the state

    def write_progress(self, force=False):
        """Store the trainer step in the state, at most every PROGRESS_SECONDS (and on phase
        changes). Also keeps the last stage output lines for `status`."""
        now = time.monotonic()
        if not force and self.last_progress is not None \
                and now - self.last_progress < self.progress_every:
            return
        if self.s is None:
            return
        # a phase change does not start the throttle window: the first trainer line still writes
        self.last_progress = None if force else now
        if self.step is not None and (not self.samples or self.step > self.samples[-1][1]):
            self.samples = (self.samples + [[clock(), self.step]])[-10:]
        at = self.open_attempt()
        self.s["progress"] = dict(step=self.step, total=self.total, updated_at=clock(),
                                  phase=at["phase"] if at else None, samples=self.samples)
        self.save()
        if self.tail:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / "stage-tail.tmp"
            tmp.write_text("\n".join(self.tail) + "\n")
            os.replace(tmp, self.dir / "stage-tail.log")

    def check_stop(self):
        """Called while a watched command runs. Raises StopRequested when the worker should now
        be torn down: at once before training, after the checkpoint push (or the wait limit)
        during training."""
        if self.in_teardown or not self.stop_file.exists():
            return
        at = self.open_attempt()
        if self.stop_deadline is None:
            if at is None or at["phase"] != "train":
                raise StopRequested("stop requested before training started; released at once")
            newest = (self.step // CKPT_EVERY) * CKPT_EVERY if self.step is not None else None
            if newest is None:
                say("stop requested, but the trainer's step could not be read, so a pending "
                    "checkpoint push cannot be told; releasing at once")
                raise StopRequested("stop requested; checkpoint push state unreadable, "
                                    "released at once")
            if newest == 0 or newest <= self.last_ack:
                raise StopRequested("stop requested; no checkpoint push was pending")
            self.stop_target = newest
            self.stop_deadline = time.monotonic() + self.stop_wait
            say(f"stop requested: waiting for the checkpoint push (checkpoint {newest}) "
                f"for up to {self.stop_wait:.0f} s, then releasing the worker")
            return
        if self.last_ack >= self.stop_target:
            raise StopRequested(f"stop requested; checkpoint {self.stop_target} reached home")
        if time.monotonic() >= self.stop_deadline:
            say(f"gave up waiting for checkpoint {self.stop_target} after "
                f"{self.stop_wait:.0f} s; releasing the worker now")
            raise StopRequested(f"stop requested; checkpoint {self.stop_target} push did not "
                                "finish in time")

    def vi_cmd(self, *args, stream=False):
        """Run a `vi` subcommand. `cloud offers` and `cloud status` are read-only, so they get a
        timeout and one more try after OUTER_RETRY_SECONDS (F-NET-1, F-CLK-1); `down` gets a
        timeout; nothing else is retried here."""
        kind = args[1] if len(args) > 1 and args[0] == "cloud" else None
        timeout = QUERY_TIMEOUT_SECONDS if kind in ("offers", "status") else (
            DOWN_TIMEOUT_SECONDS if kind == "down" else None)
        rc, out = self.run([self.vi, *args], stream=stream, timeout=timeout)
        if rc != 0 and kind in ("offers", "status"):
            say(f"`vi cloud {kind}` failed (exit {rc}); trying once more in {OUTER_RETRY_SECONDS} s")
            nap(OUTER_RETRY_SECONDS)
            rc, out = self.run([self.vi, *args], stream=stream, timeout=timeout)
        return rc, out

    def credit(self):
        rc, out = self.vi_cmd("cloud", "status")
        m = re.search(r"^credit \$([\d.]+)", out, re.M)
        return float(m.group(1)) if rc == 0 and m else None

    def instance_id(self):
        """The id of the billing worker from `vi cloud status`, or None (best effort)."""
        try:
            rc, out = self.run([self.vi, "cloud", "status"], stream=False,
                               timeout=QUERY_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001
            return None
        m = re.search(r"^worker (\d+) (?!is gone)", out, re.M) or \
            re.search(r"^untracked .* instance (\d+)", out, re.M)
        return m.group(1) if rc == 0 and m else None

    def worker_present(self):
        """(present, text, tracked): is any vi-worker instance on the account?"""
        rc, out = self.vi_cmd("cloud", "status")
        if rc != 0:
            raise Stop(EXIT["release"], "`vi cloud status` failed, so a billing worker cannot be "
                       f"ruled out. Check it by hand.\n{out.strip()}")
        tracked = bool(re.search(r"^worker \d+ (?!is gone)", out, re.M))
        orphan = bool(re.search(r"^untracked .* instance \d+", out, re.M))
        return tracked or orphan, out, tracked

    # ----- the plan ------------------------------------------------------------------------
    def plan_lines(self):
        a = self.a
        self.disk_ok, disk_gb, _, self.disk_line = disk_check(self.base_config_text())
        self.home_ok, home_lines = home_check(a.stages, a.repo)
        self.ids_ok, ids_lines = ids_check(a.stages, a.repo)
        self.realpath_ok, realpath_lines = realpath_check(
            a.stages, a.repo, forward=getattr(a, "realpath_forward", False))
        self.rehearsal_ok, rehearsal_line = rehearsal_gate(a.repo)
        self.hf_code_ok, hf_code_line = hf_offline_check()
        countries = a.countries or "(from the config)"
        return [
            f"run id          : {a.run_id}",
            f"cap (all tries) : ${a.cap:.2f}",
            f"max hours (all) : {a.max_hours} h",
            f"max attempts    : {a.max_attempts} (every attempt counts, including a rental that failed to start)",
            f"vi binary       : {self.vi}",
            f"run config      : {a.config}",
            self.disk_line,
            disk_cost_line(disk_gb, a.max_hours),
            *home_lines,
            *ids_lines,
            *realpath_lines,
            *stale_marker_lines(),
            rehearsal_line + ("" if self.rehearsal_ok else
                              f" (a real run stops here with exit {EXIT['rehearsal']})"),
            f"base model      : k2-fsa/OmniVoice @ {base_model_revision()} (pinned revision); "
            f"checked against Hugging Face before renting (exit {EXIT['hf']}); --plan runs "
            "the check offline",
            hf_code_line,
            push_hours_line(stage_settings(a.stages, a.repo, "omnivoice_train")),
            f"countries       : {countries}",
            f"stage file      : {a.stages}",
            f"state file      : {self.path}",
            "flow            : offers -> cheapest not-failed offer -> up -> preflight stage "
            "(tunnel, receiver, token push, gates) -> train stage -> verify -> down",
            "retry rule      : a failed `worker-preflight:` gate (disk, RAM, CUDA, VRAM, cores, "
            "push speed) or the upload-ETA gate retries on the next offer; every other "
            "failure releases the worker and stops",
            f"keeps at home   : checkpoints {', '.join(map(str, EXPECTED_STEPS))}; listening sets "
            f"{', '.join(map(str, LISTENING_STEPS))} in {a.platter}",
        ]

    # ----- config --------------------------------------------------------------------------
    def base_config_text(self):
        text = Path(self.a.config).read_text()
        if self.a.countries:
            cs = EU_COUNTRIES if self.a.countries.strip().lower() == "eu" else [
                c.strip().upper() for c in self.a.countries.split(",") if c.strip()]
            text = set_toml_values(text, countries=cs)
        return text

    def write_attempt_config(self, n, hours):
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"attempt-{n}.toml"
        text = set_toml_values(self.base_config_text(), max_hours=hours)
        if self.s.get("vram_floor_gb"):
            text = set_search_value(text, "min_gpu_ram_gb", f"{self.s['vram_floor_gb']:g}")
        p.write_text(text)
        self.config_path = p
        return p

    # ----- bookkeeping ---------------------------------------------------------------------
    def remaining(self):
        return (self.s["cap"] - self.s["spent_total"], self.s["max_hours"] - self.s["hours_total"])

    def open_attempt(self):
        for at in self.s["attempts"]:
            if at.get("ended") is None:
                return at
        return None

    def close_attempt(self, at, outcome, spent, spent_estimated=False):
        at["ended"] = clock()
        at["hours"] = max(0.0, (at["ended"] - at["started"]) / 3600)
        at["outcome"] = outcome
        at["spent"] = spent
        at["spent_estimated"] = spent_estimated
        self.s["spent_total"] = round(sum(x.get("spent") or 0 for x in self.s["attempts"]), 4)
        self.s["hours_total"] = round(sum(x.get("hours") or 0 for x in self.s["attempts"]), 4)
        self.save()

    def release(self, at, outcome):
        """`vi cloud down`, then close the attempt with its spend. Raises Stop(9) if the worker
        cannot be released."""
        self.in_teardown = True
        if outcome not in ("done", "up-failed", "train-vanished"):
            self.fetch_diagnostics(at, outcome)
        say("releasing the worker (vi cloud down)")
        rc, out = self.vi_cmd("cloud", "down", stream=True)
        if rc != 0:
            say("plain `down` refused; trying `down --force` so nothing keeps billing")
            rc, out = self.vi_cmd("cloud", "down", "--force", stream=True)
            for wait in RELEASE_BACKOFF:
                if rc == 0:
                    break
                say(f"`down --force` failed (exit {rc}); trying again in {wait} s")
                nap(wait)
                rc, out = self.vi_cmd("cloud", "down", "--force", stream=True)
        if rc != 0:
            self.worker_maybe_up = True
            if at is not None:
                at["outcome"] = outcome + "+release-failed"
                self.save()
            ident = self.instance_id()
            raise Stop(EXIT["release"], "COULD NOT RELEASE THE WORKER. It may still be billing"
                       f"{f' (instance {ident})' if ident else ''}. Run `vi cloud status`, then "
                       "`vi cloud down --force` now.")
        self.worker_maybe_up = False
        spent, est = None, False
        m = re.search(r"\(\$([\d.]+) spent\)", out)
        if m:
            spent = float(m.group(1))
        elif at is not None and at.get("credit_start") is not None:
            now = self.credit()
            if now is not None:
                spent = max(0.0, at["credit_start"] - now)
        if spent is None and at is not None:
            spent, est = round((at.get("dph") or 0) * (clock() - at["started"]) / 3600, 4), True
        if at is not None:
            self.close_attempt(at, outcome, spent or 0.0, est)
        self.in_teardown = False
        return spent

    def fetch_diagnostics(self, at, outcome):
        """Best effort, never raises: pull a bounded diagnostics file from the worker into the run
        dir before it is destroyed. The whole fetch is cut off after DIAG_TIMEOUT_SECONDS (its
        process group is killed), so it can delay the destroy by at most that long."""
        limit = float(os.environ.get("PAID_RUN_DIAG_SECONDS", DIAG_TIMEOUT_SECONDS))
        dest = self.dir / f"diag-attempt-{at['n'] if at else 0}-{outcome}.txt"
        say(f"saving worker diagnostics before release (at most {limit:.0f} s): {dest}")
        data, note = b"", ""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen([self.vi, "cloud", "ssh", DIAG_SCRIPT], cwd=self.a.repo,
                                 env=self.env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            try:
                data, _ = p.communicate(timeout=limit)
                if p.returncode != 0:
                    note = f"(vi cloud ssh exited {p.returncode})"
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                try:
                    data, _ = p.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    data = b""
                note = f"(cut off after {limit:.0f} s; what arrived is kept)"
            if len(data) > DIAG_MAX_BYTES:
                data, note = data[:DIAG_MAX_BYTES], note + f" (truncated to {DIAG_MAX_BYTES} bytes)"
        except BaseException as e:  # noqa: BLE001: nothing here may stop the destroy
            note = f"(diagnostics fetch failed: {type(e).__name__}: {e})"
        try:
            dest.write_bytes(data + (f"\n{note}\n".encode() if note else b""))
            say(f"diagnostics: {len(data)} bytes saved {note}".rstrip())
        except OSError as e:
            say(f"(diagnostics could not be written: {e})")

    # ----- resume --------------------------------------------------------------------------
    def settle_previous(self):
        """Handle an attempt left open by a killed run. Returns the attempt to continue (adopted
        worker in the train phase) or None."""
        at = self.open_attempt()
        present, out, tracked = self.worker_present()
        if at is None:
            if present:
                say("a vi-worker instance exists that this run did not record; destroying it "
                    "before renting anything")
                self.worker_maybe_up = True
                self.release(None, "orphan")
            return None
        say(f"attempt {at['n']} was left open (phase {at['phase']})")
        if present and at["phase"] == "train" and tracked:
            say("the worker is still up and the train stage had started: adopting it, the train "
                "stage reattaches")
            self.worker_maybe_up = True
            return at
        if present:
            self.worker_maybe_up = True
            self.release(at, "interrupted")
        else:
            now = self.credit()
            spent = None
            if now is not None and at.get("credit_start") is not None:
                spent = max(0.0, at["credit_start"] - now)
            est = spent is None
            if est:
                spent = round((at.get("dph") or 0) * (clock() - at["started"]) / 3600, 4)
            self.close_attempt(at, "interrupted", spent, est)
        if at["phase"] == "train":
            raise Stop(EXIT["train"], "the run was interrupted during the train stage and the "
                       "worker is gone; checkpoints that came home are under "
                       f"{self.a.checkpoint_root}. Decide by hand whether to start a new run.")
        return None

    # ----- the flow ------------------------------------------------------------------------
    def choose_offer(self, hours_left, cap_left):
        excl = [x for m in self.s["failed_machines"] for x in ("--exclude-machine", str(m))]
        rc, out = self.vi_cmd("cloud", "offers", *excl)
        if rc != 0:
            raise Stop(EXIT["up"], f"`vi cloud offers` failed:\n{out.strip()}")
        offers = sorted(parse_offers(out), key=lambda o: o["dph"])
        if not offers:
            raise Stop(EXIT["budget"], "no offers match the config right now")

        def failed(o):
            if o["id"] in self.s["failed_offers"]:
                return True
            if o["machine"] is not None and o["machine"] in self.s["failed_machines"]:
                return True
            if o["host"] is not None and o["host"] in self.s["failed_hosts"]:
                return True
            # without a machine id in the listing, fall back to what the machine looks like
            return o["machine"] is None and o["sig"] in self.s["failed_sigs"]

        fresh = [o for o in offers if not failed(o)]
        if not fresh:
            on_failed = len(offers) - len(fresh)
            raise Stop(EXIT["budget"], "every listed offer has already failed this run "
                       f"({len(offers)} offers listed, {on_failed} on failed machines; "
                       f"{len(self.s['failed_machines'])} machines excluded in the search)")
        affordable = [o for o in fresh if o["dph"] * hours_left <= cap_left + 1e-9]
        if not affordable:
            cheapest = fresh[0]
            raise Stop(EXIT["budget"], f"no offer fits what is left: ${cap_left:.2f} for "
                       f"{hours_left:.2f} h (cheapest is ${cheapest['dph']:.3f}/hr, worst case "
                       f"${cheapest['dph'] * hours_left:.2f})")
        return affordable[0]

    # ----- failed machines and memory floors ----------------------------------------------
    def mark_machine(self, at, host_too=False):
        """Never rent this machine again in this run (F-RENT-4); `host_too` also drops every offer
        from the same host, used for CUDA/driver failures where the whole box is suspect."""
        m, h = at.get("machine"), at.get("host")
        if m is not None and m not in self.s["failed_machines"]:
            self.s["failed_machines"].append(m)
        if host_too and h is not None and h not in self.s["failed_hosts"]:
            self.s["failed_hosts"].append(h)
        self.save()

    def raise_vram_floor(self, why):
        """F-OOM-1: after a memory failure the next offer must have more GPU memory."""
        cur = self.s.get("vram_floor_gb") or search_value(self.base_config_text(),
                                                          "min_gpu_ram_gb", 24.0)
        self.s["vram_floor_gb"] = cur + VRAM_FLOOR_STEP_GB
        self.save()
        say(f"{why}: the next offer needs at least {self.s['vram_floor_gb']:g} GB of GPU memory "
            f"(was {cur:g})")

    # ----- train stage with in-place recovery ----------------------------------------------
    def inplace_check(self):
        """The worker facts for an in-place restart as a dict, or None if they cannot be read."""
        rc, out = self.run([self.vi, "cloud", "ssh", INPLACE_SCRIPT], stream=False, timeout=120)
        facts = dict(re.findall(r"^inplace: (\w+)=(\S*)", out, re.M))
        return facts if rc == 0 and {"gpu", "disk_free_gb", "procs"} <= facts.keys() else None

    def train_blocker(self, rc, out, restarts, first_fail):
        """Why the train stage cannot be restarted in place, or None when it can. A string
        starting with 'VANISHED' means the instance is gone."""
        if self.timed_out:
            return "the train stage ran out of its hours and was killed"
        if rc == 76 or "GPU out of memory" in out:
            return "the GPU ran out of memory (exit 76); a restart on this card would do the same"
        if restarts >= INPLACE_MAX_RESTARTS:
            return f"{INPLACE_MAX_RESTARTS} in-place restarts were already used"
        if clock() - first_fail > INPLACE_MAX_SECONDS:
            return (f"the first failure was more than {INPLACE_MAX_SECONDS // 60} min ago "
                    "(the in-place recovery window)")
        present, _, tracked = self.worker_present()
        if not tracked:
            return "VANISHED: the instance is gone"
        if "No space left on device" in out:
            return "the worker disk is full (ENOSPC)"
        facts = self.inplace_check()
        if facts is None:
            return "the worker could not be inspected (`vi cloud ssh` failed)"
        if facts.get("abort", "none") != "none":
            return f"the run was aborted by its own rule ({facts['abort']})"
        if facts.get("enospc") == "1":
            return "the worker disk is full (ENOSPC in the job log)"
        if facts.get("gpu") != "ok":
            return "the GPU is not visible on the worker"
        try:
            if float(facts["disk_free_gb"]) < INPLACE_MIN_DISK_GB:
                return f"the worker has under {INPLACE_MIN_DISK_GB:g} GB of free disk"
        except ValueError:
            return "the worker's free disk could not be read"
        if facts.get("ckpt", "none") == "none":
            return "no verified checkpoint on the worker to restart from"
        self.last_facts = facts
        return None

    def run_train(self, at):
        """The train stage; a failure is recovered in place on this worker (never by renting
        another one) within INPLACE_MAX_RESTARTS and INPLACE_MAX_SECONDS, else the worker is
        released and the run stops so that a person decides. A stage process that exited while
        the trainer still runs on the worker (the supervisor died) is reattached, not counted."""
        a = self.a
        restarts = reattaches = 0
        first_fail = None
        while True:
            _, hours_left = self.remaining()
            left = max(1 / 60, hours_left - (clock() - at["started"]) / 3600)
            # the worker's ETA warning reads this (on-worker.sh forwards it); refreshed on every
            # in-place restart, so it shrinks as the run uses its hours. The stale "projected"
            # marker ($work/eta.warned) is not reset: a restart that is still too slow was
            # already warned about once.
            self.extra_env = {**getattr(self, "extra_env", {}), "VI_HOURS_LEFT": f"{left:.2f}"}
            rc, out = self.run([self.vi, "dataset", "run", a.stages, "omnivoice_train"],
                               watch=True, train=True, timeout=left * 3600)
            if rc == 0:
                return
            if first_fail is None:
                first_fail = clock()
            if rc == 76 or "GPU out of memory" in out:
                self.raise_vram_floor("train ran out of GPU memory")
            why = self.train_blocker(rc, out, restarts, first_fail)
            if why is None and self.last_facts.get("procs") not in ("0", ""):
                if reattaches >= REATTACH_MAX:
                    why = f"the stage process died {REATTACH_MAX} times while the trainer lived"
                else:
                    reattaches += 1
                    say(f"the stage exited (exit {rc}) but the trainer still runs on the worker: "
                        f"reattaching ({reattaches} of {REATTACH_MAX})")
                    nap(REATTACH_WAIT_SECONDS)
                    continue
            if why is None:
                restarts += 1
                say(f"train stage failed (exit {rc}); the worker is healthy (GPU visible, disk "
                    f"free, stage process gone, newest checkpoint {self.last_facts['ckpt']}): "
                    f"in-place restart {restarts} of {INPLACE_MAX_RESTARTS} on the same worker")
                continue
            vanished = why.startswith("VANISHED")
            self.mark_machine(at, host_too=False)
            say(f"train stage failed (exit {rc}) and cannot be recovered in place: {why}")
            self.release(at, "train-vanished" if vanished else "train-failed")
            raise Stop(EXIT["train"], f"the train stage failed (exit {rc}): {why}. No other "
                       "worker was rented; the worker was "
                       f"{'already gone' if vanished else 'released'}. Checkpoints that came home "
                       f"are under {a.checkpoint_root}. STOPPED so that you decide what next "
                       "(this machine is on the run's failed list).")

    def execute(self):
        a = self.a
        adopted = self.settle_previous()
        if self.s["credit_before_run"] is None:
            self.s["credit_before_run"] = self.credit()
            self.save()
        at = adopted
        while True:
            if at is None:
                if self.stop_file.exists():
                    raise StopRequested("stop requested; nothing was rented")
                n = len(self.s["attempts"]) + 1
                cap_left, hours_left = self.remaining()
                # Every attempt counts, including one whose `vi cloud up` failed: attempts bound
                # the number of rentals (a boot timeout bills real minutes), while the cap and
                # the hours bound the spend. Otherwise a long run of dead offers is unbounded.
                if len(self.s["attempts"]) >= a.max_attempts:
                    raise Stop(EXIT["push"], f"all {a.max_attempts} attempts allowed are used "
                               "(each ended in a failed rental or a failed cheap worker gate)")
                if hours_left < a.min_hours:
                    raise Stop(EXIT["budget"], f"only {hours_left:.2f} h of the {a.max_hours} h "
                               f"left (under --min-hours {a.min_hours}); not renting again")
                if cap_left <= 0:
                    raise Stop(EXIT["budget"], f"the ${a.cap:.2f} cap is used up")
                say(f"===== attempt {n} of {a.max_attempts} at {rfc3339(clock())} =====")
                say(f"attempt {n}: ${cap_left:.2f} and {hours_left:.2f} h left")
                self.write_attempt_config(n, hours_left)
                offer = self.choose_offer(hours_left, cap_left)
                say(f"cheapest offer not tried: {offer['id']} {offer['gpu']} {offer['loc']} "
                    f"${offer['dph']:.3f}/hr")
                at = dict(n=n, offer=offer["id"], sig=offer["sig"], dph=offer["dph"],
                          machine=offer["machine"], host=offer["host"],
                          phase="up", started=clock(), ended=None, cap=round(cap_left, 4),
                          hours=None, credit_start=self.credit())
                self.s["attempts"].append(at)
                self.save()
                self.worker_maybe_up = True
                until = rfc3339(clock() + 3600)
                boot_s = config_boot_minutes(self.base_config_text()) * 60 + UP_EXTRA_SECONDS
                rc, out = self.run([self.vi, "cloud", "up", "--offer", str(offer["id"]), "--cap",
                                    f"{cap_left:.2f}", "--until", until, "--yes"], watch=True,
                                   timeout=boot_s)
                if rc != 0:
                    # F-RENT-1/2: the offer is gone, or the box did not boot. Release whatever
                    # exists (a no-op when nothing was created), skip this offer, try the next.
                    boot_fault = self.timed_out or bool(BOOT_FAIL_RE.search(out))
                    say(f"`vi cloud up` failed (exit {rc}) for offer {offer['id']}"
                        f"{' (boot failure: machine marked failed)' if boot_fault else ''}")
                    self.s["failed_offers"].append(offer["id"])
                    if boot_fault:
                        self.s["failed_sigs"].append(offer["sig"])
                        self.mark_machine(at)
                    self.release(at, "up-failed")
                    at = None
                    continue
                self.set_phase(at, "preflight")
                quiet = float(os.environ.get("PAID_RUN_FATAL_QUIET_SECONDS", FATAL_QUIET_SECONDS))
                rc, out = self.run([self.vi, "dataset", "run", a.stages, "omnivoice_preflight"],
                                   watch=True, timeout=PREFLIGHT_TIMEOUT_SECONDS, fatal_quiet=quiet)
                fatal = self.fatal_stopped  # release() runs other commands, which reset it
                if rc != 0 and fatal:
                    self.release(at, "preflight-failed")
                    raise Stop(EXIT["preflight"], "the preflight printed a fatal line ("
                               f"{fatal.strip()[:200]}) and then nothing for "
                               f"{quiet:.0f} s, so it was stopped. A crash is not a per-offer gate: "
                               "the worker was released, no retry; the diag-attempt file in the run "
                               "folder holds the worker's limits and job tree.")
                if rc != 0:
                    gate = retryable_gate(out) or ("stage-timeout" if self.timed_out else None)
                    if gate:
                        say(f"offer {offer['id']} failed a cheap worker gate ({gate})")
                        self.s["failed_offers"].append(offer["id"])
                        self.s["failed_sigs"].append(offer["sig"])
                        kind = gate_kind(out)
                        self.mark_machine(at, host_too=kind == "cuda")
                        if kind == "memory":
                            self.raise_vram_floor("a memory gate failed")
                        spent = self.release(at, "push-gate" if gate == "push-speed" else "worker-gate")
                        say(f"attempt {n} cost ${spent:.2f}; total ${self.s['spent_total']:.2f} "
                            f"of ${a.cap:.2f}, {self.s['hours_total']:.2f} of {a.max_hours} h")
                        at = None
                        continue
                    self.release(at, "preflight-failed")
                    raise Stop(EXIT["preflight"], f"the preflight failed (exit {rc}) on a gate "
                               "that is not a retryable worker gate; the worker was released, no retry")
                self.set_phase(at, "train")
            # -- train (new or adopted) --
            say(f"===== train stage (attempt {at['n']}) at {rfc3339(clock())} =====")
            say("preflight passed; starting the train stage")
            self.run_train(at)
            self.set_phase(at, "verify")
            problems = self.verify_final()
            spent = self.release(at, "verify-failed" if problems else "done")
            if problems:
                raise Stop(EXIT["verify"], "the train stage finished but the results at home do "
                           "not check out:\n  " + "\n  ".join(problems))
            return spent

    # ----- home results --------------------------------------------------------------------
    def run_dir(self):
        root = Path(self.a.checkpoint_root) / "omnivoice_train"
        runs = [d for d in root.glob("*") if d.is_dir()] if root.is_dir() else []
        return max(runs, key=lambda d: d.stat().st_mtime) if runs else None

    def verify_final(self):
        """What home should hold after a good train stage; a list of problems (empty = fine)."""
        problems = []
        d = self.run_dir()
        if d is None:
            return [f"no run folder under {self.a.checkpoint_root}/omnivoice_train"]
        steps = sorted(int(m.group(1)) for p in d.iterdir()
                       if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir())
        if tuple(steps) != EXPECTED_STEPS:
            problems.append(f"{d}: checkpoints present {steps}, expected {list(EXPECTED_STEPS)}")
        if not (d / "VERIFIED").is_file():
            # F-PUSH-6: older checkpoints may have been pushed weights-only; the stage's own verify
            # does not accept that, so try home's weights-only-older verify (the newest must be
            # full). It writes VERIFIED and clears the unverified marker when it passes.
            rc, out = self.run(["nice", "-n", "19", sys.executable, "-I", str(REPO / "jobs" / "checkpoints.py"),
                                "--format", "omnivoice", "verify", str(d), "--expect", str(len(EXPECTED_STEPS)),
                                "--weights-only-older"], stream=False, timeout=1800)
            if rc != 0 or not (d / "VERIFIED").is_file():
                problems.append(f"{d}: no VERIFIED file and the weights-only-older verify failed: "
                                f"{out.strip()[-300:]}")
        markers = vi_state_dir() / "unverified"
        if markers.is_dir():
            for m in markers.iterdir():
                if m.is_file() and str(d) in m.read_text():
                    problems.append(f"unverified marker {m} still names {d}")
        for step in LISTENING_STEPS:
            s = Path(self.a.platter) / f"step-{step}"
            gone = [need for need in ("listening.jsonl", "README.txt") if not (s / need).is_file()]
            problems += [f"listening set step {step}: {s / need} is missing" for need in gone]
            if not gone:
                problems += self.write_clip_list(step, s)
        if not problems:
            say(f"verified: checkpoints {steps} in {d}; listening sets in {self.a.platter}")
        return problems

    def write_clip_list(self, step, s):
        """Make step-N listable in the review hub: write `clips.toml` beside its wav files.

        The hub (`vi review corpus/tali`) already finds any `clips.toml` under a stage's current
        output (crates/review/src/hub.rs `walk`, depth 4) and plays its clips unblinded, with a
        label and the line text, saving his answers beside it. A single-model set has nothing to
        blind against, so `blind` stays off. Only local files are written; returns problems."""
        wavs = []
        try:
            rows = [json.loads(ln) for ln in (s / "listening.jsonl").read_text().splitlines() if ln.strip()]
        except ValueError as e:
            return [f"listening set step {step}: listening.jsonl is not valid JSON lines ({e})"]
        for row in rows:
            label = row.get("label") if isinstance(row, dict) else None
            # The row's `file` is a worker path; the wav came home beside the set under wav/.
            wav = s / "wav" / f"{label}.wav" if isinstance(label, str) and label else None
            if wav is None or Path(label).name != label or not wav.is_file():
                return [f"listening set step {step}: no audio {wav or row} for a row in listening.jsonl"]
            wavs.append((label, str(row.get("text") or "")))
        if not wavs:
            return [f"listening set step {step}: listening.jsonl has no clips"]
        q = json.dumps  # a JSON string is a valid TOML basic string
        toml = (f"title = {q(f'OmniVoice fine-tune, step {step} (run {self.a.run_id})')}\n"
                f"question = {q('Does this sound like her? Listen for the voice; nothing here is scored.')}\n"
                f"answers = [\"her\", \"partly\", \"not her\"]\n"
                f"asked_by = \"paid_run.py\"\n")
        for label, text in wavs:
            toml += f"\n[[clip]]\npath = {q(f'wav/{label}.wav')}\nlabel = {q(label)}\ntext = {q(text)}\n"
        (s / "clips.toml").write_text(toml)
        return []

    # ----- main ----------------------------------------------------------------------------
    def signal_handler(self, signum, _frame):
        if self.in_teardown:
            say(f"signal {signum} ignored while releasing the worker")
            return
        say(f"signal {signum}: stopping and releasing the worker")
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
        raise Interrupted(signum)

    def report(self):
        s = self.s
        say("attempts:")
        for at in s["attempts"]:
            say(f"  #{at['n']} offer {at['offer']} ${at['dph']:.3f}/hr: {at.get('outcome')}, "
                f"${(at.get('spent') or 0):.2f}{' (estimated)' if at.get('spent_estimated') else ''}"
                f", {(at.get('hours') or 0):.2f} h")
        after = self.credit()
        s["credit_after_run"] = after
        before = s["credit_before_run"]
        line = f"total actual spend: ${s['spent_total']:.2f} of ${s['cap']:.2f}; {s['hours_total']:.2f} h"
        if before is not None and after is not None:
            line += f"; account credit ${before:.2f} before, ${after:.2f} now (${before - after:.2f})"
        say(line + " (credit can lag the last billing slice)")
        d = self.run_dir()
        say(f"home results: checkpoints {d}; listening sets {self.a.platter}")

    def main(self):
        for ln in self.plan_lines():
            say(ln)
        if not self.disk_ok:
            print(f"paid-run: STOPPED (exit {EXIT['disk']}): the run config {self.a.config} "
                  "rents a disk too small for the worker's free-disk gate, so the preflight would "
                  "fail after you had paid for the machine. Raise disk_gb (the gate plus "
                  f"{DISK_OVERHEAD_GB:g} GB overhead is the minimum) or lower VI_MIN_DISK_GB. "
                  "Nothing was rented.", file=sys.stderr, flush=True)
            return EXIT["disk"]
        if not self.home_ok:
            print(f"paid-run: STOPPED (exit {EXIT['disk']}): home is short of disk for the "
                  "stage's min_home_gb plus reserve, so the run would fail after the worker was "
                  "rented. Free space at home first. Nothing was rented.", file=sys.stderr,
                  flush=True)
            return EXIT["disk"]
        if not self.ids_ok:
            print(f"paid-run: STOPPED (exit {EXIT['ids']}): the clip ids in the data folder would "
                  "break WebDataset on the worker (see the `clip ids` lines above), so the run "
                  "would fail after the worker was rented. Rebuild the omnivoice_data stage with "
                  "the current jobs/omnivoice_data.py first. Nothing was rented.",
                  file=sys.stderr, flush=True)
            return EXIT["ids"]
        if not self.realpath_ok:
            print(f"paid-run: STOPPED (exit {EXIT['realpath']}): the home real-path check did not "
                  "pass (see the `real-path check` lines above), so the worker's data path is "
                  "unproven and the run could fail after the worker was rented. Fix the code or "
                  "the home setup it names, then run --plan again. Nothing was rented.",
                  file=sys.stderr, flush=True)
            return EXIT["realpath"]
        if not self.hf_code_ok:
            print(f"paid-run: STOPPED (exit {EXIT['hf']}): the Hugging Face revision check "
                  "fails even on canned replies (the `hf check code` line above says why), so "
                  "the real check could not run. Nothing was rented.", file=sys.stderr, flush=True)
            return EXIT["hf"]
        if self.a.plan:
            say("--plan: nothing was called; exiting")
            return 0
        if not self.rehearsal_ok:
            live = self.recorded_live_attempt()
            if live is None:
                tail = "Nothing was rented."
            else:
                tail = (f"This run's state file shows attempt {live.get('n')} (offer "
                        f"{live.get('offer')}) never closed, so its worker may still be up. It "
                        "KEEPS BILLING until brought down; this stop does not release it. Bring it "
                        "down now with `vi cloud down --force`, then check `vi cloud status`.")
            print(f"paid-run: STOPPED (exit {EXIT['rehearsal']}): the home Vast-limits rehearsal "
                  "has not passed for this commit (the `rehearsal` line above says why). Run "
                  f"`{REHEARSAL_CMD}` from the repo with a clean tree, then start this again. "
                  + tail, file=sys.stderr, flush=True)
            return EXIT["rehearsal"]
        if os.environ.get("VI_SKIP_HF_CHECK") != "1":
            hf_ok, hf_lines = hf_revision_gate(Path.home())
            for ln in hf_lines:
                say(ln)
            if not hf_ok:
                print(f"paid-run: STOPPED (exit {EXIT['hf']}): the pinned base-model revision could "
                      "not be confirmed on Hugging Face (the `hf revision` line above says why). "
                      "Nothing was rented.", file=sys.stderr, flush=True)
                return EXIT["hf"]
        try:
            resumed = self.load_or_create()
        except Stop as e:
            print(f"paid-run: STOPPED (exit {e.code}): {e.message}", file=sys.stderr, flush=True)
            return e.code
        if resumed and self.s["status"] not in ("running", "interrupted"):
            say(f"run already finished: {self.s['status']} (exit {self.s['exit_code']}): "
                f"{self.s['message']}")
            return self.s["exit_code"]
        self.s["status"] = "running"
        self.s["pid"], self.s["pid_start"] = os.getpid(), proc_start(os.getpid())
        self.stop_file.unlink(missing_ok=True)  # a stop request left by an older run is void
        self.save()
        self.dir.mkdir(parents=True, exist_ok=True)
        log = self.dir / "run.log"
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        say(f"===== run {self.a.run_id} {'resumed' if resumed else 'started'} at "
            f"{rfc3339(clock())} (pid {os.getpid()}); full output is also in {log} =====")
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.signal_handler)
        code, message = EXIT["ok"], "done"
        try:
            try:
                self.execute()
            except Stop as e:
                code, message = e.code, e.message
            except Interrupted as e:
                code, message = EXIT["interrupted"], e.reason
            except Exception as e:  # noqa: BLE001: always fall through to the release below
                code, message = EXIT["other"], f"unexpected error: {type(e).__name__}: {e}"
        finally:
            if self.worker_maybe_up:
                try:
                    self.in_teardown = True
                    self.release(self.open_attempt(), "aborted")
                except Stop as e:
                    code, message = e.code, e.message
                except BaseException as e:  # noqa: BLE001
                    code, message = EXIT["release"], (
                        f"release failed unexpectedly ({e}); run `vi cloud status` now")
        self.s["status"] = {0: "done", EXIT["interrupted"]: "interrupted"}.get(code, "failed")
        self.s["exit_code"], self.s["message"] = code, message
        self.s["pid"] = self.s["pid_start"] = None
        self.stop_file.unlink(missing_ok=True)
        self.save()
        try:
            self.report()
        except Exception as e:  # noqa: BLE001
            say(f"(report incomplete: {e})")
        (say if code == 0 else lambda m: print(f"paid-run: STOPPED (exit {code}): {m}",
                                               file=sys.stderr, flush=True))(message)
        return code


def load_state(run_id):
    """The state dict for a run id, or None if there is no such run."""
    try:
        return json.loads(state_path(run_id).read_text())
    except (OSError, ValueError):
        return None


def stage_tail(run_id, n=10):
    try:
        return (state_root() / run_id / "stage-tail.log").read_text().splitlines()[-n:]
    except OSError:
        return []


def build_status(s, now):
    """Everything `status` shows, from a state dict. Open-attempt figures are estimates."""
    at = next((x for x in s["attempts"] if x.get("ended") is None), None)
    open_h = max(0.0, (now - at["started"]) / 3600) if at else 0.0
    open_spend = round((at.get("dph") or 0) * open_h, 4) if at else 0.0
    spent = round(s["spent_total"] + open_spend, 4)
    hours = round(s["hours_total"] + open_h, 4)
    pr = s.get("progress") or {}
    step, total = pr.get("step"), pr.get("total")
    eta = eta_seconds(pr.get("samples") or [], step, total) if at else None
    alive = proc_is_run(s["pid"], s["run_id"], s.get("pid_start")) if s.get("pid") else False
    return dict(
        run_id=s["run_id"], status=s["status"], exit_code=s["exit_code"], message=s["message"],
        phase=at["phase"] if at else None,
        attempt=dict(n=at["n"] if at else len(s["attempts"]), max=s["max_attempts"]),
        offer=(at or (s["attempts"][-1] if s["attempts"] else {})).get("offer"),
        price_per_hour=(at or (s["attempts"][-1] if s["attempts"] else {})).get("dph"),
        spend=dict(closed_attempts=s["spent_total"], open_attempt_estimate=open_spend,
                   total=spent, total_is_estimate=at is not None),
        hours_used=hours, max_hours=s["max_hours"], hours_left=round(s["max_hours"] - hours, 4),
        cap=s["cap"], cap_left=round(s["cap"] - spent, 4),
        train=dict(step=step, total=total, updated_at=pr.get("updated_at"),
                   eta_seconds=eta, eta_is_estimate=True),
        process_alive=alive, stop_requested=stop_path(s["run_id"]).exists(),
        stage_log_tail=stage_tail(s["run_id"]))


def fmt_secs(x):
    x = int(x)
    return f"{x // 3600}h{x % 3600 // 60:02d}m"


def cmd_status(args):
    s = load_state(args.run_id)
    if s is None:
        print(f"paid-run: no run with id {args.run_id!r} (looked for {state_path(args.run_id)})",
              file=sys.stderr)
        return EXIT["args"]
    st = build_status(s, clock())
    if args.credit:
        r = subprocess.run([resolve_vi_bin(), "cloud", "status"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        m = re.search(r"^credit \$([\d.]+)", r.stdout, re.M)
        st["credit"] = float(m.group(1)) if r.returncode == 0 and m else None
    if args.json:
        print(json.dumps(st, indent=1))
        return 0
    t, sp = st["train"], st["spend"]
    est = " (estimate)" if sp["total_is_estimate"] else ""
    print(f"run {st['run_id']}: {st['status']}"
          + (f" (exit {st['exit_code']}): {st['message']}" if st["exit_code"] is not None else ""))
    if st["status"] == "running" and not st["process_alive"]:
        print("  WARNING: no live paid_run process for this run; start it again with the same "
              "limits to resume")
    print(f"  phase    : {st['phase'] or '-'}; attempt {st['attempt']['n']}/{st['attempt']['max']}"
          f"; offer {st['offer']} at ${st['price_per_hour'] or 0:.3f}/hr")
    print(f"  spend    : ${sp['total']:.2f}{est} = ${sp['closed_attempts']:.2f} closed attempts "
          f"+ ${sp['open_attempt_estimate']:.2f} open attempt (estimate)")
    print(f"  cap      : ${st['cap']:.2f}, ${st['cap_left']:.2f} left{est}")
    print(f"  hours    : {st['hours_used']:.2f} used of {st['max_hours']}, "
          f"{st['hours_left']:.2f} left{est}")
    if t["step"] is not None:
        eta = f"ETA about {fmt_secs(t['eta_seconds'])} (estimate)" if t["eta_seconds"] is not None \
            else "ETA not known yet (needs 3 samples)"
        print(f"  training : step {t['step']}/{t['total']}; {eta}")
    else:
        print("  training : no step read yet")
    if st["stop_requested"]:
        print("  a stop has been requested")
    if "credit" in st:
        print(f"  credit   : " + (f"${st['credit']:.2f}" if st["credit"] is not None
                                 else "unavailable"))
    if st["stage_log_tail"]:
        print("  last stage output:")
        for ln in st["stage_log_tail"]:
            print(f"    {ln}")
    return 0


def cmd_stop(args):
    s = load_state(args.run_id)
    if s is None:
        print(f"paid-run: no run with id {args.run_id!r}; nothing changed", file=sys.stderr)
        return EXIT["args"]
    if s["status"] != "running":
        print(f"paid-run: run {args.run_id} is not running (status {s['status']}); nothing "
              "changed", file=sys.stderr)
        return EXIT["args"]
    pid = s.get("pid")
    if pid and not proc_is_run(pid, args.run_id, s.get("pid_start")):
        print(f"paid-run: the state says running but no paid_run process for {args.run_id} is "
              "alive (it was killed). Nothing changed; start it again with the same limits to "
              "resume and release the worker.", file=sys.stderr)
        return EXIT["args"]
    if args.now:
        if not pid:
            print("paid-run: the state records no process id; cannot signal. Nothing changed",
                  file=sys.stderr)
            return EXIT["args"]
        os.kill(pid, signal.SIGTERM)
        print(f"paid-run: sent SIGTERM to paid_run (pid {pid}); it releases the worker at once")
        return 0
    stop_path(args.run_id).write_text("stop\n")
    print(f"paid-run: stop requested for {args.run_id}. It waits for the current checkpoint "
          "push (at most 10 minutes), releases the worker and exits 8. Use `status` to follow.")
    return 0


def parse(argv):
    p = argparse.ArgumentParser(prog="paid_run.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True, help="names the state file; reuse it to resume")
    p.add_argument("--cap", type=float, required=True, help="most USD all attempts together")
    p.add_argument("--max-hours", type=float, required=True, help="most hours, all attempts")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--min-hours", type=float, default=0.5,
                   help="do not rent again with fewer hours than this left")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="the run's vi cloud config (default jobs/paid-run/omnivoice-run-cloud.toml)")
    p.add_argument("--countries", help='e.g. "US,DE" or "eu"; written into a copy of the config')
    p.add_argument("--stages", default=DEFAULT_STAGES)
    p.add_argument("--repo", default=str(REPO), help="working directory for vi")
    p.add_argument("--checkpoint-root", default=os.environ.get(
        "VI_CHECKPOINT_ROOT", str(Path.home() / ".cache" / "vi" / "checkpoints")))
    p.add_argument("--platter", default=os.environ.get("VI_PLATTER_DIR", str(
        REPO / "corpus" / "tali" / "stages" / "omnivoice_train" / "current" / "out" / "platter")))
    p.add_argument("--plan", action="store_true", help="print the plan and exit; no vi calls")
    p.add_argument("--realpath-forward", action="store_true",
                   help="also run one forward+backward step of the real model in the home real-path "
                        "check (needs 12 GB free RAM; otherwise it stops at the collated batch)")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.run_id):
        p.error("--run-id may only hold letters, digits, dot, dash and underscore")
    if a.cap <= 0 or a.max_hours <= 0 or a.max_attempts < 1:
        p.error("--cap, --max-hours and --max-attempts must be positive")
    if not Path(a.config).is_file():
        p.error(f"--config {a.config} is not a file")
    return a


def parse_sub(argv):
    p = argparse.ArgumentParser(prog=f"paid_run.py {argv[0]}")
    p.add_argument("--run-id", required=True)
    if argv[0] == "status":
        p.add_argument("--json", action="store_true", help="machine-readable output")
        p.add_argument("--credit", action="store_true",
                       help="also call `vi cloud status` and show the account credit")
    else:
        p.add_argument("--now", action="store_true",
                       help="signal the run to release the worker at once (no push wait)")
    a = p.parse_args(argv[1:])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.run_id):
        p.error("--run-id may only hold letters, digits, dot, dash and underscore")
    return a


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("status", "stop"):
        try:
            a = parse_sub(argv)
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else EXIT["args"]
        return (cmd_status if argv[0] == "status" else cmd_stop)(a)
    try:
        args = parse(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT["args"]
    return PaidRun(args).main()


if __name__ == "__main__":
    sys.exit(main())
