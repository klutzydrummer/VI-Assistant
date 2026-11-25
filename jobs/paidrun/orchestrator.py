"""PaidRun: the state, running vi, progress and stop requests, the plan, bookkeeping and main loop."""

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from paidrun.common import (
    EXIT, Interrupted, Stop, StopRequested, Tee, clock, nap, proc_start, resolve_vi_bin, rfc3339,
    say, state_root, stop_path, timeout_scale,
)
from paidrun.limits import (
    DOWN_TIMEOUT_SECONDS, DOWN_VERIFY_TIMEOUT_SECONDS, HOST_DROP_RECHECK_SECONDS,
    OUTER_RETRY_SECONDS, PROGRESS_SECONDS, PUSH_PROBE_MAX_MINUTES, QUERY_TIMEOUT_SECONDS,
    STOP_WAIT_SECONDS,
)
from paidrun.signals import ACK_RE, FATAL_RE, parse_step
from paidrun.config import (
    CKPT_EVERY, DEBUG_MAX_RENTALS, DISK_CEILING_PER_GB_MONTH, DISK_OVERHEAD_GB, EU_COUNTRIES,
    EXPECTED_STEPS, HOURS_PER_MONTH, LISTENING_STEPS, config_max_dph, debug_worst_case, disk_check,
    disk_cost_line, push_hours_line, set_search_value, set_toml_values, stage_settings,
)
from paidrun.hf import base_model_revision, hf_offline_check, hf_revision_gate
from paidrun.gates import (
    REHEARSAL_CMD, home_check, ids_check, realpath_check, rehearsal_gate, stale_marker_lines,
)
from paidrun.release import ReleaseMixin
from paidrun.rental import RentalMixin
from paidrun.debug import DebugMixin


class PaidRun(RentalMixin, DebugMixin, ReleaseMixin):
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
                    # Each argument is cut short and kept on one line: `vi cloud ssh` gets a whole script as
                    # one argument, and its echo lines must not be read back as stage results.
                    shown = " ".join(" ".join(str(x).split())[:80] for x in argv[:4])
                    msg = (f"paid-run: {shown} timed out after {limit:.0f} s; "
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
            (DOWN_VERIFY_TIMEOUT_SECONDS if "--verify" in args else DOWN_TIMEOUT_SECONDS)
            if kind == "down" else None)
        rc, out = self.run([self.vi, *args], stream=stream, timeout=timeout)
        if rc != 0 and kind in ("offers", "status"):
            say(f"`vi cloud {kind}` failed (exit {rc}); trying once more in {OUTER_RETRY_SECONDS} s")
            nap(OUTER_RETRY_SECONDS)
            rc, out = self.run([self.vi, *args], stream=stream, timeout=timeout)
        return rc, out

    def cache_instance(self, at, out):
        """r32 WP7: remember the instance id `vi cloud up` printed ("rented instance N" right
        after the create, "worker N ready" once booted), even when `up` failed, so release()
        needs no 180 s `vi cloud status` lookup. Returns the id or None."""
        m = re.search(r"^(?:rented instance (\d+);|worker (\d+) ready)", out, re.M)
        if m and at is not None:
            at["instance"] = m.group(1) or m.group(2)
            self.save()
        return at.get("instance") if at is not None else None

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
        if getattr(a, "debug_pass", False):
            max_dph = config_max_dph(self.base_config_text())
            total, gpu, disk = debug_worst_case(a.cap, disk_gb, a.max_hours, max_dph)
            h, k = a.max_hours, DEBUG_MAX_RENTALS
            gpu_why = (f"GPU ${gpu:.2f} ({k} rentals x {h:g} h x ${max_dph:.3f}/hr max_dph, "
                       f"inside the ${a.cap:.2f} cap)" if max_dph is not None
                       and gpu < a.cap - 1e-9 else
                       f"GPU ${gpu:.2f} (the cap for both rentals together"
                       + (")" if max_dph is None else
                          f"; {k} x {h:g} h x ${max_dph:.3f}/hr max_dph would be more)"))
            flow = [
                f"mode            : --debug-pass (up to {DEBUG_MAX_RENTALS} rentals: a second one "
                "ONLY after a confirmed host drop (ssh lost the host and two Vast status reads "
                f"{HOST_DROP_RECHECK_SECONDS} s apart agree), never after our own check failing; every stage is tried and a failure "
                "never stops the next one)",
                "host drop       : before release, `vastai show instance <id> --raw` and "
                "`vastai logs <id>` are saved in the run folder; the stage reads HOST-DROP "
                "(status X) only when ssh lost the host and two Vast reads at least "
                f"{HOST_DROP_RECHECK_SECONDS} s apart both say not running or loading (or both "
                "say the instance is gone while ssh is still refused); that worker is "
                f"released and one more offer is rented only if the cap left covers $/hr x "
                f"{h:g} h (each rental has its own {h:g} h; the $ cap is shared)",
                "flow            : offers -> cheapest qualifying offer -> up -> 1 host-cuda, "
                "2 disk-limits (debug-probe.sh over `vi cloud ssh`) -> stage omnivoice_debug "
                "(3 upload + token handshake, 4 venv + launchers, 5 base-model, 6 tokenise, "
                "7 cuda-load, 8 train-steps, 9 ckpt-push, 10 listening, 11 resume) -> 12 down "
                "(always, even after a failure)",
                f"debug data      : the first {a.debug_rows} train rows of omnivoice_data "
                "(jobs/debug_slice.py, by position only), plus 4 dev, 2 style, 8 replay texts",
f"worst case      : ${total:.2f} total = {gpu_why} + disk ${disk:.2f} "
                f"({disk_gb:g} GB x ${DISK_CEILING_PER_GB_MONTH:.2f}/GB/month / "
                f"{HOURS_PER_MONTH} h x {k} x {h:g} h); cap ${a.cap:.2f} for both rentals "
                f"together (an offer is rented only if $/hr x {h:g} h <= the cap left)",
                "                  formula: worst case = min(cap, 2 x max_hours x max_dph) + "
                "disk_gb x 1.00 / 730 x 2 x max_hours; a typical pass should take under 30 min, "
                "so it costs about $/hr x 0.5",
                "push probe      : after the worker preflight passes, at most "
                f"{PUSH_PROBE_MAX_MINUTES} min of transfer-speed probe (jobs/push-probe.sh: "
                "200 MiB random through the tunnel, 4 x 50 MiB parallel ssh streams, curl to "
                "public speed-test URLs; fail-soft) = at most "
                + (f"${PUSH_PROBE_MAX_MINUTES / 60 * max_dph:.2f} at the ${max_dph:.3f}/hr "
                   "max_dph" if max_dph is not None else "$/hr x 0.08")
                + ", about $0.02 at $0.20/hr; it is inside the hours and the cap above, so the "
                "worst case does not grow",
                "time per rental : each stage's limit is cut to the rental's time left minus a "
                "15 min margin; a stage with too little time left reads SKIP-no-time (not "
                "started, shown apart from FAIL); stage 9's limit is 1.5 x checkpoint bytes / "
                "the push rate the preflight measured, at least 600 s",
                f"results         : {self.dir}/debug-summary.txt and .json; exit 0 only if "
                f"every stage passed, else {EXIT['debug']}",
            ]
        else:
            flow = [
                "flow            : offers -> cheapest not-failed offer -> up -> preflight stage "
                "(tunnel, receiver, token push, gates) -> train stage -> verify -> down",
                "retry rule      : a failed `worker-preflight:` gate (disk, RAM, CUDA, VRAM, cores) "
                "or the upload-ETA gate retries on the next offer; a slow push is only a warning "
                "(measured MB/s and checkpoint push time are printed); every other failure "
                "releases the worker and stops",
                f"keeps at home   : checkpoints {', '.join(map(str, EXPECTED_STEPS))}; listening sets "
                f"{', '.join(map(str, LISTENING_STEPS))} in {a.platter}",
            ]
        return [
            f"run id          : {a.run_id}",
            f"cap (all tries) : ${a.cap:.2f}",
(f"max hours       : {a.max_hours:g} h per rental ({DEBUG_MAX_RENTALS} x "
             f"{a.max_hours:g} h at most)" if getattr(a, "debug_pass", False)
             else f"max hours (all) : {a.max_hours} h"),
            (f"max attempts    : up to {DEBUG_MAX_RENTALS} rentals (a second only after a "
             "confirmed host drop, inside the same cap, with its own hours; never a third)"
             if getattr(a, "debug_pass", False)
             else f"max attempts    : {a.max_attempts} (every attempt counts, including a rental "
                  "that failed to start)"),
            f"vi binary       : {self.vi}",
            f"run config      : {a.config}",
            self.disk_line,
            disk_cost_line(disk_gb, a.max_hours * (DEBUG_MAX_RENTALS if getattr(
                a, "debug_pass", False) else 1)),
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
            *flow,
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


    # ----- home results --------------------------------------------------------------------
    def run_dir(self):
        root = Path(self.a.checkpoint_root) / "omnivoice_train"
        runs = [d for d in root.glob("*") if d.is_dir()] if root.is_dir() else []
        return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


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
                (self.execute_debug if getattr(self.a, "debug_pass", False) else self.execute)()
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
