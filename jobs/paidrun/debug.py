"""The debug pass (--debug-pass): rent, run the numbered debug stages, check home, write the summary."""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from paidrun.common import EXIT, Interrupted, REPO, Stop, StopRequested, clock, rfc3339, say
from paidrun.limits import (
    DEBUG_LOST_MINUTES, DEBUG_PROBE_SECONDS, DEBUG_RELEASE_MARGIN_SECONDS, UP_EXTRA_SECONDS,
    VERIFY_EPOCH_TIMEOUT_SECONDS,
)
from paidrun.signals import (
    NO_SUCH_ASK_RE, debug_excerpt_keep, debug_run_key, failure_rc_text, parse_debug_output, parse_host_qual,
    parse_push_probe, push_probe_lines,
)
from paidrun.validity import INVALID, REASON_HOST_DROP, REASON_HOST_UNQUALIFIED, rental2_allowed
from paidrun.config import DEBUG_MAX_RENTALS, DEBUG_STAGE, config_boot_minutes, stage_settings


def verify_epoch_argv(ck):
    """The home check that checkpoint-5 of a debug pass came home whole (patchable in tests)."""
    return [sys.executable, str(REPO / "jobs" / "checkpoints.py"), "--format", "omnivoice",
            "verify-epoch", str(ck), "5"]


# ---- the debug pass (--debug-pass) ---------------------------------------------------------
# (number, name, the stages it needs). 0 is the rental itself; 12 (down) needs nothing.
DEBUG_STAGES = (
    (1, "host-cuda", (0,)), (2, "disk-limits", (0,)), (3, "upload", (0,)),
    (4, "venv", (3,)), (5, "base-model", (4,)), (6, "tokenise", (4, 5)),
    (7, "cuda-load", (4, 5)), (8, "train-steps", (6, 7)), (9, "ckpt-push", (8,)),
    (10, "listening", (8,)), (11, "resume", (8,)), (12, "down", ()))
DEBUG_NAMES = {n: name for n, name, _ in DEBUG_STAGES} | {0: "rental"}
DEBUG_PROBE_SCRIPT = REPO / "jobs" / "debug-probe.sh"
# Settings of the debug stage -> the floors debug-probe.sh reads.
DEBUG_PROBE_FLOORS = (("min_vram_gb", "DEBUG_MIN_VRAM_GB"), ("min_cuda_version", "DEBUG_MIN_CUDA"),
                      ("min_disk_gb", "DEBUG_MIN_DISK_GB"), ("min_ram_gb", "DEBUG_MIN_RAM_GB"),
                      ("min_cpu_cores", "DEBUG_MIN_CORES"))


class DebugMixin:
    # ----- the debug pass ------------------------------------------------------------------
    def debug_record(self, n, result, detail="", excerpt=()):
        self.debug[n] = dict(stage=n, name=DEBUG_NAMES[n], result=result, detail=detail,
                             excerpt=debug_excerpt_keep(excerpt))
        say(f"debug-pass: stage {n} {DEBUG_NAMES[n]}: {result}{' ' + detail if detail else ''}")

    def debug_gate(self, n):
        """True when stage n may run; otherwise records SKIP-dependent naming what failed."""
        needs = next(d for m, _, d in DEBUG_STAGES if m == n)
        for m in needs:
            if self.debug.get(m, {}).get("result") != "PASS":
                self.debug_record(n, "SKIP-dependent",
                                  f"(needs stage {m} {DEBUG_NAMES[m]}, which did not pass)"
                                  if m else "(needs the rental, which failed)")
                return False
        return True

    def debug_probe(self):
        """Stages 1 and 2: jobs/debug-probe.sh over `vi cloud ssh` with the stage's floors."""
        st = stage_settings(self.a.stages, self.a.repo, DEBUG_STAGE) or {}
        env = "".join(f"export {var}={shlex.quote(str(st[key]))}; "
                      for key, var in DEBUG_PROBE_FLOORS if key in st)
        rc, out = self.run([self.vi, "cloud", "ssh", env + DEBUG_PROBE_SCRIPT.read_text()],
                           stream=False, timeout=DEBUG_PROBE_SECONDS)
        timed_out = self.timed_out
        found = parse_debug_output(out)
        drop = None
        for n in (1, 2):
            if n in found:
                res, detail, ex = found[n]
                self.debug_record(n, res, detail, ex)
                continue
            if not timed_out and drop is None:
                drop = self.host_drop(self.open_attempt(), rc, out)
            if drop and drop[0]:
                self.debug_record(n, "HOST-DROP", drop[1].removeprefix("HOST-DROP "),
                                  out.strip().splitlines()[-8:])
                self.dropped = drop[1]
                continue
            why = (f"hit its {DEBUG_PROBE_SECONDS} s limit" if timed_out
                   else f"`vi cloud ssh` exited {rc} without a stage {n} line"
                   + (f"; {drop[1]}" if drop and drop[1] else ""))
            self.debug_record(n, "FAIL", f"({why})", out.strip().splitlines()[-8:])

    def gone_check_cmd(self, at):
        """The command on-worker.sh runs once, read-only, after ~75 s of refused connections: it
        exits 0 when Vast says this instance is gone (jobs/paidrun/vast_gone_check.py). None when
        the instance id is unknown, or in a test with a clock but no stub vastai."""
        if os.environ.get("PAID_RUN_CLOCK_FILE") and not os.environ.get("VI_VASTAI_BIN"):
            return None
        try:
            ident = (at or {}).get("instance") or self.instance_id()
        except Exception:  # noqa: BLE001
            ident = None
        if not ident or not str(ident).isdigit():
            return None
        script = Path(__file__).resolve().parent / "vast_gone_check.py"
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {ident}"

    def debug_worker_stages(self, at):
        """Stages 3 to 11: the stage omnivoice_debug (on-worker.sh + jobs/debug-pass.sh). r21: the
        time is this rental's own --max-hours since it started, not the hours shared with
        rental 1; debug-pass.sh cuts each stage to it (VI_HOURS_LEFT) and prints SKIP-no-time."""
        left_s = self.a.max_hours * 3600 - (clock() - at["started"])
        limit = max(60.0, left_s - DEBUG_RELEASE_MARGIN_SECONDS)
        self.extra_env = {**getattr(self, "extra_env", {}),
                          "VI_HOURS_LEFT": f"{max(left_s, 60) / 3600:.2f}",
                          "VI_DEBUG_PASS": "1", "VI_DEBUG_ROWS": str(self.a.debug_rows),
                          # r21: give up on a lost worker after 5 min, not r9's 30
                          "VI_LOST_MINUTES": str(DEBUG_LOST_MINUTES),
                          # r32 WP5: the probe's leg a is the push half of the host gate
                          "VI_HOST_QUAL_GATE": "1", **self.host_qual_env()}
        gone_cmd = self.gone_check_cmd(at)
        if gone_cmd:  # on-worker.sh's early host-drop stop (VI_EARLY_DROP_SECONDS, default 75 s)
            self.extra_env["VI_GONE_CHECK_CMD"] = gone_cmd
        self.set_phase(at, "debug")
        rc, out = self.run([self.vi, "dataset", "run", self.a.stages, DEBUG_STAGE, "--force"],
                           watch=True, timeout=limit)
        timed_out = self.timed_out
        self.push_probe = at["push_probe"] = parse_push_probe(out)
        for ln in push_probe_lines(self.push_probe):
            say(ln.strip())
        found = parse_debug_output(out)
        tail = [ln for ln in out.strip().splitlines() if "debug-pass:" not in ln][-10:]
        push = parse_host_qual(out).get("push")
        if push and push[0] == "fail" and 3 not in found:
            # r32 WP5: on-worker.sh stopped before the upload (exit 75); stages 4-11 depend on 3
            why = f"return push too slow ({push[1]})"
            self.mark_unqualified(at, why, "push")
            self.unqualified = why
            self.debug_record(3, "HOST-UNQUALIFIED", f"({why}; nothing was uploaded)", tail)
        drop = None
        for n in range(3, 12):
            if n in found:
                res, detail, ex = found[n]
                self.debug_record(n, res, detail, ex)
            elif n in self.debug:
                continue
            elif self.debug_gate(n):
                # Its prerequisites passed, yet no line: the worker stage stopped or was cut off
                # first (for stage 3: on-worker.sh never started debug-pass.sh, e.g. the return
                # path or the token push failed; its own message is in the excerpt). A2: if ssh
                # lost the host and the status check confirms it, that is a HOST-DROP.
                if not timed_out and drop is None:
                    drop = self.host_drop(at, rc, out)
                if drop and drop[0]:
                    self.debug_record(n, "HOST-DROP", drop[1].removeprefix("HOST-DROP ")
                                      + f"; the stage ended ({failure_rc_text(rc, out)})", tail)
                    self.dropped = drop[1]
                    continue
                why = (f"the debug stage hit its {limit:.0f} s limit first" if timed_out else
                       f"never reported; the stage ended ({failure_rc_text(rc, out)}) first"
                       + (f"; {drop[1]}" if drop and drop[1] else ""))
                self.debug_record(n, "FAIL", f"({why})", tail)
        self.debug_home_checks(out)

    def debug_home_dirs(self, out=""):
        """(checkpoint run dir, stage output debug dir) at home for THIS run's stage key, or None
        each. r31 F4: the key (and attempt) come from this run's output; dbg6 took the newest dir
        of any run by mtime and read an older run's summary. No key in the output: None, None."""
        key, attempt = debug_run_key(out)
        if key is None:
            return None, None
        root = Path(self.a.checkpoint_root) / DEBUG_STAGE
        runs = [d for d in root.glob("*") if d.is_dir() and d.name.startswith(key)] \
            if root.is_dir() else []
        exact = [d for d in runs if attempt and d.name.endswith(f"-{attempt}")]
        runs = exact or runs
        ck = max(runs, key=lambda d: d.stat().st_mtime) if runs else None
        corpus_dir = None
        try:
            import tomllib
            sp = Path(self.a.stages)
            sp = sp if sp.is_absolute() else Path(self.a.repo) / sp
            corpus_dir = sp.parent / tomllib.loads(sp.read_text()).get("corpus", ".")
        except (ImportError, OSError, ValueError):
            pass
        outs = []
        if corpus_dir is not None:
            base = corpus_dir / "stages" / DEBUG_STAGE
            outs = [d for d in list(base.glob("*/out/debug")) + list(base.glob(".tmp-*/out/debug"))
                    if d.is_dir() and d.parent.parent.name.removeprefix(".tmp-").startswith(key)]
        out = max(outs, key=lambda d: d.stat().st_mtime) if outs else None
        return ck, out

    def debug_home_checks(self, run_out=""):
        """Stage 9 and 10 pass only if their results are really at home."""
        ck, out = self.debug_home_dirs(run_out)
        if self.debug.get(9, {}).get("result") == "PASS":
            if ck is None:
                key = debug_run_key(run_out)[0]
                self.debug_record(9, "FAIL", "(the worker pushed, but no checkpoint run folder "
                                  f"is under {self.a.checkpoint_root}/{DEBUG_STAGE} for this run's "
                                  + (f"stage key {key})" if key else "stage key: the output "
                                     "named none)"))
            else:
                limit = VERIFY_EPOCH_TIMEOUT_SECONDS
                try:
                    r = subprocess.run(verify_epoch_argv(ck), capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=limit)
                except subprocess.TimeoutExpired as e:  # r32 WP7: never hang home
                    part = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) \
                        else (e.stdout or "")
                    self.debug_record(9, "FAIL", f"(checkpoint-5 in {ck}: the home verify-epoch "
                                      f"did not finish in {limit} s and was stopped)",
                                      part.strip().splitlines()[-6:])
                else:
                    if r.returncode != 0:
                        self.debug_record(9, "FAIL", f"(checkpoint-5 in {ck} did not verify at "
                                          "home)", (r.stdout + r.stderr).strip().splitlines()[-6:])
                    else:
                        self.debug_record(9, "PASS", f"(checkpoint-5 verified at home in {ck})")
        if self.debug.get(10, {}).get("result") == "PASS":
            wavs = sorted((out / "listening").rglob("*.wav")) if out is not None else []
            if not wavs:
                self.debug_record(10, "FAIL", "(the worker made it, but no listening wav came "
                                  "home in the stage output)")
            else:
                self.debug_record(10, "PASS", f"({len(wavs)} wav files at home in "
                                  f"{out / 'listening'}; for human listening only, not scored)")
        if out is not None and (out / "summary.json").is_file():
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "debug-worker-summary.json").write_bytes((out / "summary.json").read_bytes())

    def rental_reason(self):
        """The open rental's INVALID reason class. A confirmed host drop always set
        INVALID(host-drop) through host_drop(), and a failed host qualification
        INVALID(host-unqualified) through mark_unqualified()."""
        v = getattr(self, "validity", None)
        if v is not None and v.state == INVALID:
            return v.reason
        if self.dropped:
            return REASON_HOST_DROP
        return REASON_HOST_UNQUALIFIED if getattr(self, "unqualified", None) else None

    def debug_write_summary(self):
        rows = [self.debug.get(n) or dict(stage=n, name=DEBUG_NAMES[n], result="FAIL",
                                          detail="(never ran: the debug pass stopped early)",
                                          excerpt=[])
                for n, _, _ in DEBUG_STAGES]
        self.dir.mkdir(parents=True, exist_ok=True)
        rentals = []
        for at in self.s["attempts"]:
            status = (at.get("host_drop") or (f"released ({at['outcome']})" if at.get("outcome")
                                              else "open"))
            rentals.append(dict(n=at["n"], offer=at["offer"], instance=at.get("instance"),
                                dph=at.get("dph"), status=status, spent=at.get("spent"),
                                spent_estimated=at.get("spent_estimated", False),
                                vast_status=at.get("vast_status"), vast_file=at.get("vast_file"),
                                invalid=at.get("invalid"), stages=at.get("debug_stages")))
        doc = dict(run_id=self.a.run_id, rental=self.debug.get(0), rentals=rentals,
                   note=self.debug_note, stages=rows,
                   spent_total=self.s["spent_total"], hours_total=self.s["hours_total"],
                   cap=self.s["cap"], max_hours=self.s["max_hours"],
                   listening_set_note="for human listening only; never scored",
                   push_probe=getattr(self, "push_probe", None),
                   push_probe_note=("jobs/push-probe.sh, the last rental's; MB/s is decimal; "
                                    "null when it did not run (it needs a passed worker "
                                    "preflight)"))
        (self.dir / "debug-summary.json").write_text(json.dumps(doc, indent=1) + "\n")
        lines = [f"debug pass {self.a.run_id}: spent ${self.s['spent_total']:.2f} of "
                 f"${self.s['cap']:.2f} cap in {len(rentals)} rental(s) (at most "
                 f"{DEBUG_MAX_RENTALS}), {self.s['hours_total']:.2f} h in all "
                 f"({self.s['max_hours']} h per rental)"]
        for r in rentals:
            spent = "?" if r["spent"] is None else f"${r['spent']:.2f}" + (
                " (estimated)" if r["spent_estimated"] else "")
            lines.append(f"  rental {r['n']}      : offer {r['offer']}, instance "
                         f"{r['instance'] or '?'}, {r['status']}, spent {spent}"
                         + (f", INVALID ({r['invalid']['reason']})" if r["invalid"] else ""))
        if self.debug_note:
            lines.append(f"  note          : {self.debug_note}")
        if len(rentals) > 1:
            lines.append(f"  stages below are rental {rentals[-1]['n']}'s; rental 1's are in "
                         "debug-summary.json (rentals[0].stages)")
        if self.debug.get(0):
            r = self.debug[0]
            lines.append(f"  rental        : {r['result']} {r['detail']}".rstrip())
        lines += push_probe_lines(getattr(self, "push_probe", None))
        for r in rows:
            lines.append(f"  {r['stage']:>2} {r['name']:<12}: {r['result']} {r['detail']}".rstrip())
            lines += [f"       | {x}" for x in r["excerpt"]]
        no_time = [r for r in rows if r["result"] == "SKIP-no-time"]
        if no_time:
            lines.append(f"  {len(no_time)} stage(s) were not started for lack of time "
                         "(SKIP-no-time), which is not a failure of the stage: the rental's "
                         f"{self.s['max_hours']} h ran out first")
        (self.dir / "debug-summary.txt").write_text("\n".join(lines) + "\n")
        say("===== debug pass summary =====")
        for ln in lines:
            if not ln.startswith("       |"):
                say(ln)
        say(f"full summary with log excerpts: {self.dir / 'debug-summary.txt'}")
        return rows

    def debug_rent_and_run(self, n):
        """One debug rental (attempt n): offer -> up -> host qualification -> stages 1-11.
        Returns the rental's INVALID reason class (host-drop for a confirmed host drop (A2),
        host-unqualified for a failed WP5 gate, ...), else None. The worker is left up. r21: each
        rental gets its own --max-hours; only the $ cap is shared, so the offer must fit
        $/hr x max_hours inside the cap left."""
        a = self.a
        self.dropped = None
        self.unqualified = None
        cap_left, hours_left = self.remaining()[0], a.max_hours
        say(f"===== debug pass rental {n} at {rfc3339(clock())}: ${cap_left:.2f}, "
            f"{hours_left:.2f} h left =====")
        self.write_attempt_config(n, hours_left)
        offer = self.choose_offer(hours_left, cap_left)
        say(f"cheapest offer: {offer['id']} {offer['gpu']} {offer['loc']} ${offer['dph']:.3f}/hr")
        if n > 1:  # the first rental's stage table is kept in its attempt record
            self.debug = {}
        at = dict(n=n, offer=offer["id"], sig=offer["sig"], dph=offer["dph"],
                  machine=offer["machine"], host=offer["host"], phase="up", started=clock(),
                  ended=None, cap=round(cap_left, 4), hours=None, credit_start=self.credit())
        self.s["attempts"].append(at)
        self.save()
        self.worker_maybe_up = True
        boot_s = config_boot_minutes(self.base_config_text()) * 60 + UP_EXTRA_SECONDS
        rc, out = self.run([self.vi, "cloud", "up", "--offer", str(offer["id"]), "--cap",
                            f"{cap_left:.2f}", "--until", rfc3339(clock() + hours_left * 3600),
                            "--yes"], watch=True, timeout=boot_s)
        self.cache_instance(at, out)
        m = re.search(r"^worker (\d+) ready", out, re.M)
        if rc != 0 and not m and NO_SUCH_ASK_RE.search(out):
            # The offer vanished before anything was rented (dbg9 rental 2): a market failure,
            # not ours, so a second rental stays allowed.
            why = f"offer {offer['id']} vanished before it was rented (Vast no_such_ask)"
            self.mark_unqualified(at, why, "offer")
            self.unqualified = why
            self.debug_record(0, "HOST-UNQUALIFIED", f"({why}; nothing was rented)",
                              out.strip().splitlines()[-8:])
        elif rc != 0:
            self.debug_record(0, "FAIL", f"(`vi cloud up` exited {rc}"
                              f"{', boot timed out' if self.timed_out else ''})",
                              out.strip().splitlines()[-8:])
        else:
            inst = f", instance {m.group(1)}" if m else ""
            # r32 WP5: qualify the host before anything else runs on it or is uploaded to it
            why = self.host_qualify(at)
            if why:
                self.mark_unqualified(at, why, "download")
                self.unqualified = why
                self.debug_record(0, "HOST-UNQUALIFIED", f"(offer {offer['id']} at "
                                  f"${offer['dph']:.3f}/hr{inst}: {why}; nothing was uploaded)")
            else:
                self.debug_record(0, "PASS", f"(offer {offer['id']} at ${offer['dph']:.3f}/hr"
                                  f"{inst}; host qualified)")
                self.set_phase(at, "debug")
                self.debug_probe()
                if not self.dropped:  # after a drop in 1-2 the worker stages cannot reach it
                    self.debug_worker_stages(at)
        for k in range(1, 12):
            if k not in self.debug:
                self.debug_gate(k)
        return self.rental_reason()

    def execute_debug(self):
        """Stages 1-12, each tried; the worker is always released (stage 12). One rental, plus
        ONE more only when rental2_allowed says so for rental 1's INVALID reason (a confirmed
        host drop (C) or a failed host qualification (WP5/D3)), inside the same $ cap; each rental
        has its own --max-hours (r21)."""
        self.debug = {}
        self.dropped = None
        self.unqualified = None
        self.debug_note = None
        self.push_probe = None
        failure = None
        a_h = self.a.max_hours
        try:
            self.settle_previous()
            if self.s["attempts"]:
                raise Stop(EXIT["budget"], "a debug pass is never resumed and this run id already "
                           "has an attempt; start a new --run-id")
            if self.s["credit_before_run"] is None:
                self.s["credit_before_run"] = self.credit()
                self.save()
            if self.stop_file.exists():
                raise StopRequested("stop requested; nothing was rented")
            reason = self.debug_rent_and_run(1)
            if reason and rental2_allowed(reason):
                # C/C.6: rental 1 INVALID for a reason that allows it -> evidence, release, then
                # at most one more. rental2_allowed: host-drop and host-unqualified (D3) yes;
                # stalled, our-failure and cap never. The failed host and machine ids are
                # excluded from the next search.
                what = "the host drop" if reason == REASON_HOST_DROP else \
                    "the failed host qualification"
                at = self.open_attempt()
                at["debug_stages"] = [self.debug[k] for k in sorted(self.debug)]
                if at["offer"] not in self.s["failed_offers"]:
                    self.s["failed_offers"].append(at["offer"])
                self.s["failed_sigs"].append(at["sig"])
                self.mark_machine(at, host_too=True)
                spent = self.release(at, reason)
                say(f"rental 1 released after {what}: ${spent or 0:.2f} spent; total "
                    f"${self.s['spent_total']:.2f} of ${self.s['cap']:.2f}")
                if self.stop_file.exists():
                    self.debug_note = f"stop requested after {what}; no second rental"
                else:
                    try:
                        again = self.debug_rent_and_run(DEBUG_MAX_RENTALS)
                        if again and rental2_allowed(again):
                            lost = "also lost its host" if again == REASON_HOST_DROP else \
                                "also failed host qualification"
                            self.debug_note = (f"rental {DEBUG_MAX_RENTALS} {lost}; a "
                                               f"debug pass rents at most {DEBUG_MAX_RENTALS} "
                                               "workers, so there is no third")
                    except Stop as e:
                        if e.code != EXIT["budget"] or len(self.s["attempts"]) > 1:
                            raise
                        self.debug_note = (f"no second rental: {e.message}. The cap left "
                                           f"(${self.remaining()[0]:.2f}) does not cover one "
                                           f"for {a_h:g} h")
                if self.debug_note:
                    say(self.debug_note)
        except (Stop, Interrupted, Exception) as e:  # noqa: BLE001: stage 12 still runs
            failure = e
            self.user_stopped = isinstance(e, Interrupted)
            say(f"debug pass stopped early: {getattr(e, 'message', None) or e!r}")
        finally:
            if self.worker_maybe_up:
                try:
                    self.in_teardown = True
                    at = self.open_attempt()
                    if at is not None:
                        at["debug_stages"] = [self.debug[k] for k in sorted(self.debug)]
                    bad = any(r["result"] != "PASS" for r in self.debug.values())
                    outcome = ("host-drop" if self.dropped else
                               "host-unqualified" if self.unqualified else
                               "debug-failed" if bad or failure is not None else "debug-done")
                    spent = self.release(at, outcome)
                    self.debug_record(12, "PASS", f"(worker released; ${spent or 0:.2f} spent)")
                except Stop as e:
                    self.debug_record(12, "FAIL", f"({e.message})")
                    failure = failure if isinstance(failure, Stop) and failure.code == \
                        EXIT["release"] else e
                finally:
                    self.in_teardown = False
            elif self.s["attempts"]:
                self.debug_record(12, "PASS", f"(every worker released; "
                                  f"${self.s['spent_total']:.2f} spent in all)")
            else:
                self.debug_record(12, "PASS", "(no worker was up)")
            rows = self.debug_write_summary()
        if failure is not None:
            raise failure
        bad = [r for r in rows if r["result"] != "PASS"]
        if bad:
            raise Stop(EXIT["debug"], f"the debug pass finished; {len(bad)} of {len(rows)} stages "
                       "did not pass (" + ", ".join(f"{r['stage']} {r['name']}" for r in bad)
                       + f"). The worker was released. Details: {self.dir / 'debug-summary.txt'}")
        say("debug pass: every stage passed")
