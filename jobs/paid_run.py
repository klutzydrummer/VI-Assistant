#!/usr/bin/env python3
"""One unattended command for the paid OmniVoice fine-tune flow (standard library only).

    python3 -I jobs/paid_run.py --run-id ID --cap USD --max-hours H [options]

It drives the existing `vi` binary ($VI_BIN, else <repo>/target/release/vi if built, else `vi`) so a stub can stand in for tests:
rent the cheapest qualifying offer, run the preflight stage (tunnel, receiver, token push, the
worker gates), then the train stage, then release the worker and print what it cost. If ONLY the
cheap worker gate fails (stderr says `worker-preflight: ...`: disk, RAM, CUDA, VRAM, cores, push
speed; or `worker upload ETA` for a too-slow upload line), the machine is released, that offer is
excluded, and the next cheapest is tried, up to --max-attempts. Right after ssh works, and
before any upload, every mode runs the host-qualification gate (jobs/push-probe.sh host-qual: a
public download of at least 5 MB/s over 30 s, then a return push of at least 1 MB/s); a fail or a
probe that cannot run releases the worker as INVALID(host-unqualified), excludes its host and
machine from the next search, and tries the next offer. Any other failure releases the
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
 16 --debug-pass ran to the end, but at least one of its stages did not pass (worker released)
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

Debug pass (one rental, or two only after a confirmed host drop or a failed host qualification;
every step tried):

    python3 -I jobs/paid_run.py --debug-pass --run-id ID --cap USD --max-hours H [--plan]

rents the cheapest qualifying offer and runs 12 stages, each with its own time limit,
carrying on past failures: 1 host/driver/CUDA and 2 disk and limits (jobs/debug-probe.sh over
`vi cloud ssh`), then the stage `omnivoice_debug` (jobs/debug-pass.sh through on-worker.sh, so
the real tunnel, receiver and token push) for 3 upload, 4 venv and launchers, 5 Base fetch,
6 tokenise, 7 CUDA load and replay, 8 five training steps with the KL reference, 9 checkpoint push
(verified at home here), 10 listening set (checked at home here), 11 resume, and 12 down, which
always runs. A stage whose prerequisite failed prints SKIP-dependent. The results go to
<state dir>/<run-id>/debug-summary.{txt,json}; exit 0 only if every stage passed, else 16. The
same pre-rent gates as a real run apply, and --plan prints the worst-case dollars. After the
worker preflight passes, on-worker.sh runs a transfer-speed probe (jobs/push-probe.sh, at most
5 minutes, random bytes only, fail-soft): MB/s through the tunnel, over 4 parallel ssh streams, and
with curl on the worker against public speed-test URLs; it goes into the summary as push_probe.

Heavy work happens inside `vi dataset run` (stages `omnivoice_preflight`, `omnivoice_train`,
and `omnivoice_debug` for --debug-pass). No Vast credential file is read here; only the `vi`
CLI is called.
"""

import argparse
import hashlib
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

# `python3 -I` (how the paid-run card runs this script) leaves this script's folder off sys.path,
# so put it back before importing the jobs/paidrun package. The names below are re-exported so
# `paid_run.<name>` keeps working for callers and tests.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from paidrun import common
from paidrun.common import (
    REPO, DEFAULT_CONFIG, DEFAULT_STAGES, nap, timeout_scale, EXIT, RUN_LOG_MAX_BYTES, Tee, Stop,
    Interrupted, StopRequested, proc_start, proc_is_run, say, state_root, vi_state_dir, clock,
    rfc3339, state_path, stop_path, resolve_vi_bin,
)
from paidrun import limits
from paidrun.limits import (
    PROGRESS_SECONDS, STOP_WAIT_SECONDS, OUTER_RETRY_SECONDS, RELEASE_BACKOFF,
    QUERY_TIMEOUT_SECONDS, DOWN_TIMEOUT_SECONDS, DOWN_VERIFY_TIMEOUT_SECONDS,
    VERIFY_EPOCH_TIMEOUT_SECONDS, PREFLIGHT_TIMEOUT_SECONDS, FATAL_QUIET_SECONDS, UP_EXTRA_SECONDS,
    DEFAULT_BOOT_MINUTES, INPLACE_MAX_RESTARTS, INPLACE_MAX_SECONDS, REATTACH_MAX,
    REATTACH_WAIT_SECONDS, HF_TRIES, HF_TIMEOUT_SECONDS, HF_BACKOFF, VAST_EVIDENCE_SECONDS,
    HOST_DROP_RECHECK_SECONDS, DIAG_TIMEOUT_SECONDS, DEBUG_PROBE_SECONDS,
    DEBUG_RELEASE_MARGIN_SECONDS, DEBUG_LOST_MINUTES, PUSH_PROBE_MAX_MINUTES, REALPATH_TIMEOUT,
)
from paidrun import signals
from paidrun.signals import (
    PREFLIGHT_GATE_RE, UPLOAD_GATE_TEXT, WORKER_RC_RE, ON_WORKER_EXIT_RE, STAGE_STATUS_RE,
    failure_rc_text, HOST_DROP_RE, DROP_EXIT_RE, drop_signal, DEBUG_WORK_RE, DEBUG_RUNNING_RE,
    debug_run_key, retryable_gate, gate_kind, OFFER_RE, TQDM_RE, STEP_RE, ACK_RE, MIN_SAMPLES,
    FATAL_RE, BOOT_FAIL_RE, CUDA_GATE_RE, MEMORY_GATE_RE, parse_step, eta_seconds, MACHINE_TAIL_RE,
    parse_offers, DEBUG_LINE_RE, DEBUG_EXCERPT_RE, parse_vast_instance, NO_CONTAINER_RE, VAST_GONE,
    VAST_ALIVE, vast_instance_gone, parse_debug_output, PUSH_PROBE_JSON_RE, parse_push_probe,
    push_probe_lines, debug_excerpt_keep,
)
from paidrun import config
from paidrun.config import (
    EU_COUNTRIES, EXPECTED_STEPS, LISTENING_STEPS, EXPECT_EPOCHS, CKPT_EVERY, GATE_SCRIPT,
    GATE_FALLBACK_GB, DISK_OVERHEAD_GB, CONFIG_DEFAULT_DISK_GB, DEBUG_MAX_RENTALS, set_toml_values,
    set_search_value, search_value, config_boot_minutes, gate_default_gb, config_disk_gb,
    disk_check, DISK_CEILING_PER_GB_MONTH, HOURS_PER_MONTH, disk_cost_line, debug_worst_case,
    config_max_dph, stage_settings, CHECKPOINT_BYTES_CALCULATED, WEIGHTS_BYTES_CALCULATED,
    push_hours_line, DEBUG_STAGE,
)
from paidrun import hf
from paidrun.hf import (
    HF_REPO, HF_API, base_model_revision, load_size_table, RevisionNotFoundError,
    RepositoryNotFoundError, hf_fetch, hf_model_info, hf_offline_check, hf_revision_gate,
)
from paidrun import gates
from paidrun.gates import (
    bundle_dir, bundle_key, bundle_inputs, bundle_cache_status, load_bundle_ids, data_folder,
    ids_check, REALPATH_REQ, REALPATH_SCRIPT, REALPATH_MIN_RAM_MB, REALPATH_FORWARD_RAM_MB,
    mem_available_mb, realpath_venv, realpath_code_hash, REHEARSAL_CMD, rehearsal_state_dir,
    rehearsal_gate, run_realpath, realpath_check, marker_state, unverified_markers, stale_markers,
    stale_marker_lines, home_check,
)
from paidrun import status
from paidrun.status import load_state, stage_tail, build_status, fmt_secs, cmd_status, cmd_stop
from paidrun import release
from paidrun.release import DIAG_MAX_BYTES, VAST_LOG_TAIL, DIAG_SCRIPT, ReleaseMixin
from paidrun import debug
from paidrun.debug import (
    verify_epoch_argv, DEBUG_STAGES, DEBUG_NAMES, DEBUG_PROBE_SCRIPT, DEBUG_PROBE_FLOORS, DebugMixin,
)
from paidrun import rental
from paidrun.rental import INPLACE_MIN_DISK_GB, VRAM_FLOOR_STEP_GB, INPLACE_SCRIPT, RentalMixin
from paidrun import orchestrator
from paidrun.orchestrator import PaidRun
# ---- end of paidrun re-exports


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
    p.add_argument("--debug-pass", action="store_true",
                   help="rent once and try every worker stage separately, carrying on past "
                        "failures (stage omnivoice_debug); the worker is always released")
    p.add_argument("--debug-rows", type=int, default=128,
                   help="train rows in the debug pass's data slice (default 128)")
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
    if a.debug_rows < 1:
        p.error("--debug-rows must be at least 1")
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
