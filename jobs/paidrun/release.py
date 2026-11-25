"""Releasing a rental and settling it: diagnostics, Vast evidence, host drops, final verify."""

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from paidrun.common import EXIT, REPO, Stop, clock, nap, say, vi_state_dir
from paidrun.limits import (
    DIAG_TIMEOUT_SECONDS, HOST_DROP_RECHECK_SECONDS, RELEASE_BACKOFF, VAST_EVIDENCE_SECONDS,
)
from paidrun.signals import (
    VAST_ALIVE, VAST_GONE, drop_signal, parse_vast_instance, vast_instance_gone,
)
from paidrun.config import DEBUG_STAGE, EXPECTED_STEPS, LISTENING_STEPS
from paidrun.gates import unverified_markers


DIAG_MAX_BYTES = 256 * 1024
VAST_LOG_TAIL = 200
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


class ReleaseMixin:
    def release(self, at, outcome):
        """r32 WP1/C.7: destroy first, then everything else. `vi cloud down --force --verify`
        DELETEs the worker (the cached instance id from create, else whatever is tracked or
        labelled), ignores unverified markers (it records them), and lists instances until Vast
        shows it gone; exit 9 means the destroy was never verified. Only that blocks: it raises
        Stop(9). Markers and unverified checkpoints never delay or prevent the DELETE. The
        ssh diagnostics (bounded, DIAG_TIMEOUT_SECONDS) need the live worker and so come first;
        Vast's view, the spend and the debug-marker cleanup come after the destroy."""
        self.in_teardown = True
        if outcome not in ("done", "up-failed", "train-vanished"):
            self.fetch_diagnostics(at, outcome)
        ident = (at or {}).get("instance")
        down = ("cloud", "down", "--force", "--verify", *([str(ident)] if ident else []))
        say(f"releasing the worker (vi {' '.join(down)})")
        rc, out = self.vi_cmd(*down, stream=True)
        for wait in RELEASE_BACKOFF:
            if rc == 0:
                break
            say(f"`down --force --verify` failed (exit {rc}); trying again in {wait} s")
            nap(wait)
            rc, out = self.vi_cmd(*down, stream=True)
        if rc != 0:
            self.worker_maybe_up = True
            if at is not None:
                at["outcome"] = outcome + "+release-failed"
                self.save()
            ident = ident or self.instance_id()
            raise Stop(EXIT["release"], "COULD NOT RELEASE THE WORKER: the destroy was not "
                       "verified, so it may still be billing"
                       f"{f' (instance {ident})' if ident else ''}. Run `vi cloud status`, then "
                       "`vi cloud down --force` now.")
        self.worker_maybe_up = False
        gone = re.search(r"^destroyed instance (\d+)", out, re.M)
        if at is not None and not at.get("instance") and gone:
            at["instance"] = gone.group(1)
            self.save()
        if outcome not in ("done", "debug-done", "orphan") and not (at or {}).get("vast_file"):
            self.vast_evidence(at, outcome)
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
        if getattr(self.a, "debug_pass", False):
            self.abandon_empty_debug_markers()
        self.in_teardown = False
        return spent

    def abandon_empty_debug_markers(self):
        """r21 F4: after a debug worker is released, its `omnivoice_debug-*` unverified marker
        names a checkpoint dir that is empty or missing (the debug pass keeps nothing), and that
        marker would make the next plain `vi cloud down` refuse. Abandon exactly those with
        `jobs/checkpoints.py abandon` (marker_state() decides). r31 F3: also a debug dir whose
        checkpoint has no SHA256SUMS (dbg5's partial push): it can never verify, and a debug pass
        is never resumed. Files are kept. A debug dir with a manifest and every other stage's
        marker are left alone. Best effort, never raises."""
        done = 0
        for m, d, st in unverified_markers():
            if not m.name.startswith(f"{DEBUG_STAGE}-") or st == "complete-unverified":
                continue
            why = ("empty debug checkpoint dir" if st in ("missing", "empty") else
                   "debug checkpoint has no SHA256SUMS (partial push)")
            try:
                r = subprocess.run([sys.executable, str(REPO / "jobs" / "checkpoints.py"),
                                    "abandon", str(d), "--reason",
                                    f"debug pass {self.a.run_id}: {why} "
                                    "after release"], env=self.env(), capture_output=True,
                                   text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as e:
                say(f"could not abandon the debug marker {m.name}: {e!r}")
                continue
            if r.returncode == 0:
                done += 1
            else:
                say(f"could not abandon the debug marker {m.name}: "
                    f"{(r.stdout + r.stderr).strip()[-300:]}")
        if done:
            say(f"abandoned {done} empty or incomplete debug marker(s) (jobs/checkpoints.py "
                "abandon; files kept)")

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

    def vast_evidence(self, at, why):
        """A1, best effort, never raises: `vastai show instance <id> --raw` (actual_status,
        status_msg) and `vastai logs <id>` into the run dir, before the worker is destroyed.
        Returns {"status", "msg", "id", "file"}; status is None when it could not be read. Only
        the Vast CLI is run: its key and its config and state files are never read here."""
        info = dict(status=None, msg=None, id=None, file=None)
        try:
            n = at["n"] if at else 0
            dest = self.dir / f"vast-attempt-{n}-{why}.txt"
            info["file"] = str(dest)
            parts = []
            limit = float(os.environ.get("PAID_RUN_VAST_SECONDS", VAST_EVIDENCE_SECONDS))
            vastai = os.environ.get("VI_VASTAI_BIN")
            ident = (at or {}).get("instance") or self.instance_id()
            info["id"] = ident
            if ident is None:
                parts.append("(no instance id from `vi cloud status`; nothing to ask Vast about)")
            elif not vastai and os.environ.get("PAID_RUN_CLOCK_FILE"):
                # a test clock is set but no stub CLI: never let a test reach the real Vast
                parts.append("(skipped: test clock in use and VI_VASTAI_BIN is unset)")
            else:
                vastai = vastai or "vastai"
                answers = {}
                for cmd in (["show", "instance", str(ident), "--raw"],
                            ["logs", str(ident), "--tail", str(VAST_LOG_TAIL)]):
                    say(f"saving Vast's view of instance {ident} (at most {limit:.0f} s): "
                        f"vastai {' '.join(cmd)}")
                    try:
                        r = subprocess.run([vastai, *cmd], stdin=subprocess.DEVNULL,
                                           capture_output=True, timeout=limit,
                                           env={**os.environ, "PAGER": "cat"})
                        text = (r.stdout + r.stderr)[:DIAG_MAX_BYTES].decode(errors="replace")
                        parts.append(f"== vastai {' '.join(cmd)} (exit {r.returncode})\n{text}")
                        answers[cmd[0]] = (r.returncode, r.stdout.decode(errors="replace"), text)
                        if cmd[0] == "show" and r.returncode == 0:
                            info.update(parse_vast_instance(r.stdout.decode(errors="replace")))
                    except subprocess.TimeoutExpired:
                        parts.append(f"== vastai {' '.join(cmd)}: cut off after {limit:.0f} s")
                    except OSError as e:
                        parts.append(f"== vastai {' '.join(cmd)}: could not run ({e})")
                if info["status"] is None and "show" in answers:
                    show_rc, show_out, _ = answers["show"]
                    logs_rc, _, logs_text = answers.get("logs", (None, "", ""))
                    gone = vast_instance_gone(show_rc, show_out, logs_rc, logs_text)
                    if gone:
                        info.update(status=VAST_GONE, msg=gone)
            head = (f"instance {ident}: actual_status {info['status'] or 'unknown'}; "
                    f"status_msg {info['msg'] or '-'}")
            self.dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(head + "\n" + "\n".join(parts) + "\n")
            say(f"Vast evidence: {head} -> {dest}")
            if at is not None:
                at.update(instance=ident, vast_status=info["status"], vast_msg=info["msg"],
                          vast_file=str(dest))
                self.save()
        except BaseException as e:  # noqa: BLE001: nothing here may stop the destroy
            say(f"(Vast evidence could not be saved: {type(e).__name__}: {e})")
        return info

    def host_drop(self, at, rc, out):
        """A2: (confirmed, text) when output that did not report its own result shows ssh losing
        the host; (False, None) when there is no sign of that. r21: a drop is confirmed only by
        TWO Vast status reads (`vastai show instance`, saved as A1 evidence) at least
        HOST_DROP_RECHECK_SECONDS apart that both say not running (r27: nor loading). If home
        cannot read the status from Vast (its own API call failed or hung), that is never a drop:
        our failure to ask is not evidence about the host. r27: Vast saying the instance is gone
        ({"instances": null}, or logs "No such container" next to a status-less answer; see
        vast_instance_gone) is a drop only while `vi cloud ssh true`, tried once more, is still
        refused or unreachable. Otherwise that ssh try is for the record only. Our own check
        failures print their own FAIL and never get here."""
        sig = drop_signal(rc, out)
        if sig is None:
            return False, None
        say(f"ssh may have lost the host ({sig}); checking the instance status with Vast")
        reads = []
        for i in range(2):
            if i:
                say(f"Vast says {reads[0][0]}; asking again in {HOST_DROP_RECHECK_SECONDS} s "
                    "before calling it a host drop")
                nap(HOST_DROP_RECHECK_SECONDS)
            ev = self.vast_evidence(at, "host-drop-check" if i == 0 else "host-drop-recheck")
            reads.append((ev.get("status"), ev.get("msg"), ev.get("file")))
            if reads[-1][0] is None or reads[-1][0].lower() in VAST_ALIVE:
                break
        try:
            r3, o3 = self.run([self.vi, "cloud", "ssh", "true"], stream=False, timeout=60)
        except Exception as e:  # noqa: BLE001
            r3, o3 = None, repr(e)
        again = drop_signal(r3, o3) if r3 != 0 else None
        again_text = "ssh again: ok" if r3 == 0 else f"ssh again: {again or f'exit {r3}'}"
        seen = ", then ".join(st or "unknown" for st, _, _ in reads)
        # r27: "gone" (Vast no longer lists the instance) counts only while ssh is still refused
        # or unreachable; if ssh works again, the host is there whatever the listing said.
        gone_ok = all(st != VAST_GONE for st, _, _ in reads) or (r3 != 0 and again is not None)
        if (len(reads) == 2 and gone_ok
                and all(st is not None and st.lower() not in VAST_ALIVE for st, _, _ in reads)):
            status, msg, f = reads[-1]
            text = (f"HOST-DROP (status {status}{f': {msg}' if msg else ''}); {sig}; Vast said "
                    f"{seen} {HOST_DROP_RECHECK_SECONDS} s apart; {again_text}"
                    f"; Vast evidence in {f or 'the run folder'}")
            say(text)
            if at is not None:
                at["host_drop"] = text
                self.save()
            return True, text
        why = ("home could not read the instance status from Vast, so this is not a host drop"
               if reads[-1][0] is None else
               f"Vast said {seen}, but ssh answers again" if not gone_ok else f"Vast said {seen}")
        text = (f"ssh lost the connection ({sig}), but this is not a confirmed host drop "
                f"({why}; {again_text})")
        say(text)
        return False, text

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
