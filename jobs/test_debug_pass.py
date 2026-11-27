"""Tests for `jobs/paid_run.py --debug-pass` with the stub `vi` from test_paid_run (no Vast, no
GPU, no network), plus the small helpers the debug pass ships (debug_slice.py, the shell scripts).

The stub gains two answers here: `vi cloud ssh <debug-probe text>` prints stage 1 and 2 lines,
and `vi dataset run ... omnivoice_debug --force` prints stage 3 to 11 lines the way
jobs/debug-pass.sh does. Scenario key `probe` picks the probe answer and `debug` the stage's.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import test_paid_run as tp

JOBS = tp.JOBS
SCRIPT = tp.SCRIPT

NAMES = {3: "upload", 4: "venv", 5: "base-model", 6: "tokenise", 7: "cuda-load",
         8: "train-steps", 9: "ckpt-push", 10: "listening", 11: "resume"}
NEEDS = {4: (3,), 5: (4,), 6: (4, 5), 7: (4, 5), 8: (6, 7), 9: (8,), 10: (8,), 11: (8,)}

DEBUG_BRANCH = '''\
    if stage == "omnivoice_debug":
        r = a.get("debug", sc.get("debug", "pass"))
        log("debug_env", os.environ.get("VI_DEBUG_PASS"), os.environ.get("VI_DEBUG_ROWS"))
        log("debug_hours_left", os.environ.get("VI_HOURS_LEFT"))
        tick(a.get("debug_tick", 0))
        if r == "drop":
            # r18: the host closes ssh during the upload, then stops listening; on-worker.sh exits
            # 255 under set -e before debug-pass.sh starts, and the vi CLI exits 1
            st["dropped"] = st["worker"]["id"]; save()
            print("Connection to 38.34.175.110 closed by remote host.", file=sys.stderr)
            print("rsync: connection unexpectedly closed (0 bytes received so far) [sender]", file=sys.stderr)
            print("on-worker.sh: exiting with 255", file=sys.stderr)
            print("error: stage omnivoice_debug failed: exit status: 255", file=sys.stderr); sys.exit(1)
        if r == "drop_dbg6":
            # r31: omnivoice-ft-dbg6 (2026-11-25, run.log 146-216): stages 3 and 4 passed, stage 5
            # never reported, the host vanished; ssh printed no "remote host" and no exit 255 of
            # its own, on-worker.sh exited 1 and so did the vi CLI
            print("debug-pass: stage 3 upload: PASS (5 s)", flush=True)
            print("debug-pass: stage 4 venv: PASS (670 s)", flush=True)
            print("debug-pass: stage 5 base-model: still running (720 s)", flush=True)
            st["dropped"] = st["worker"]["id"]; save()
            for ln in ("worker-preflight: return path: worker could not push through the reverse SSH return tunnel",
                       "Connection closed by 192.0.2.7 port 53880",
                       "2026-11-25T12:02:58Z tunnel down (ssh exit 255); reconnecting in 10s",
                       "on-worker.sh: ALERT: could not re-prove the return path",
                       "on-worker.sh: poll failed; will retry (lost for 308s)",
                       "on-worker.sh: giving up after 5 min without a successful poll; keeping /root/vi-jobs/w on the worker for a reattach",
                       "on-worker.sh: omnivoice_debug did not complete successfully; keeping the worker copy",
                       "on-worker.sh: exiting with 1",
                       "error: stage omnivoice_debug failed: exit status: 1"):
                print(ln, file=sys.stderr)
            sys.exit(1)
        if r == "silent":
            print("on-worker.sh: input upload failed", file=sys.stderr); sys.exit(1)
        # r31 F4: the vi runner's key line and debug-pass.sh's log path name this run's key
        key = a.get("stage_key", sc.get("stage_key", "ab12" * 16))
        print(f"omnivoice_debug: running ({key[:12]})", file=sys.stderr)
        print(f"debug-pass: stage 3 upload: running (limit 60 s; log /root/vi-jobs/omnivoice_debug."
              f"{key}.20261125T113310Z/out/debug/03-upload.log)", flush=True)
        if sc.get("push_probe"):  # r30: on-worker.sh's probe line, before debug-pass.sh starts
            print("push-probe: start (random bytes only)", file=sys.stderr)
            print(sc["push_probe"], file=sys.stderr, flush=True)
        names = %r
        needs = %r
        fail = set(sc.get("debug_fail", []))
        no_time = set(a.get("debug_no_time", sc.get("debug_no_time", [])))
        res = {}
        for n in range(3, 12):
            if r == "hang" and n == 5:
                print("debug-pass: stage 5 base-model: still running (60 s)", flush=True); hang()
            bad = [m for m in needs.get(n, ()) if res.get(m) != "PASS"]
            if bad:
                res[n] = "SKIP"
                print(f"debug-pass: stage {n} {names[n]}: SKIP-dependent (needs stage {bad[0]} "
                      f"{names[bad[0]]}, which did not pass)", flush=True)
            elif n in no_time:
                res[n] = "SKIP-no-time"
                print(f"debug-pass: stage {n} {names[n]}: SKIP-no-time (100 s left before the "
                      "rental's deadline; needs 120 s)", flush=True)
            elif n in fail:
                res[n] = "FAIL"
                print(f"debug-pass: stage {n} {names[n]}: FAIL (exit 1, 3 s)", flush=True)
                print(f"debug-pass:   | stage {n} broke: ModuleNotFoundError: no module x", flush=True)
            else:
                res[n] = "PASS"
                print(f"debug-pass: stage {n} {names[n]}: PASS (2 s)", flush=True)
        if res.get(9) == "PASS" and not sc.get("no_home_ckpt"):
            ck = (Path(os.environ["VI_CHECKPOINT_ROOT"]) / "omnivoice_debug"
                  / f"{key}-20261125T113310Z" / "checkpoint-5")
            ck.mkdir(parents=True, exist_ok=True)
            sums = []
            for f in ("config.json", "model.safetensors", "optimizer.bin", "scheduler.bin",
                      "random_states_0.pkl", "train_config.json", "ft_config.json", "tokenizer.json"):
                (ck / f).write_bytes(b"x" + f.encode())
                import hashlib
                sums.append(hashlib.sha256((ck / f).read_bytes()).hexdigest() + "  " + f)
            (ck / "SHA256SUMS").write_text("\\n".join(sums) + "\\n")
        out = Path(sc["corpus"]) / "stages" / "omnivoice_debug" / f".tmp-{key}" / "out" / "debug"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({str(k): v for k, v in res.items()}))
        if res.get(10) == "PASS" and not sc.get("no_home_wav"):
            (out / "listening").mkdir(exist_ok=True)
            (out / "listening" / "style0__line0.wav").write_bytes(b"RIFF")
        print("debug-pass: summary written")
        sys.exit(0)
''' % (NAMES, NEEDS)

PROBE_BRANCH = '''\
    if "debug-pass: stage 1" in " ".join(args):
        p = att().get("probe", sc.get("probe", "pass"))
        log("probe", "DEBUG_MIN_VRAM_GB" in " ".join(args))
        if p == "drop":
            st["dropped"] = st["worker"]["id"]; save()
            print("Connection to 38.34.175.110 closed by remote host.", file=sys.stderr); sys.exit(255)
        if p == "hang":
            (D / "probe.pid").write_text(str(os.getpid())); time.sleep(600)
        if p == "nothing":
            print("ssh: connection refused", file=sys.stderr); sys.exit(255)
        if p == "fail1":
            print("debug-pass: stage 1 host-cuda: FAIL driver CUDA 12.2 is under 12.8")
            print("debug-pass:   | cuda (driver): 12.2")
        else:
            print("debug-pass: stage 1 host-cuda: PASS CUDA 12.8, 24.0 GiB")
            print("debug-pass:   | driver: 570.1")
        print("debug-pass: stage 2 disk-limits: PASS 120 GB disk, nofile 65536, shm 8.0 GiB")
        print("debug-pass:   | open files (ulimit -n): soft 65536, hard 65536")
        sys.exit(0)
'''


def debug_stub():
    s = tp.STUB
    a1 = '    if stage == "omnivoice_preflight":'
    a2 = '    mode = sc.get("diag", "ok")'
    assert s.count(a1) == 1 and s.count(a2) == 1, "test_paid_run STUB changed; update anchors"
    s = s.replace(a1, DEBUG_BRANCH + a1)
    return s.replace(a2, PROBE_BRANCH + a2)


STAGES_TOML = '''\
corpus = "corpus"

[stage.omnivoice_debug]
command = ["true"]
inputs = []
manual = true

[stage.omnivoice_debug.settings]
min_gpu_ram_gb = 20
min_cuda = "12.8"
min_disk_gb = 1
min_ram_gb = 1
min_cpu_cores = 1
'''


class DebugBase(tp.Base):
    def setUp(self):
        super().setUp()
        self.stub.write_text(debug_stub())
        self.stages_path = self.tmp / "stages.toml"
        self.stages_path.write_text(STAGES_TOML)
        self.env.update(VI_MIN_MODEL_BYTES="1", VI_MIN_OPTIMIZER_BYTES="1")

    def scen(self, **extra):
        self.scenario([dict(spent=0.42)], corpus=str(self.tmp / "corpus"), **extra)

    def run_debug(self, **kw):
        kw.setdefault("stages", self.stages_path)
        return self.run_orch(debug_pass=True, **kw)

    def summary(self, run_id="t1"):
        return json.loads((self.tmp / "paid" / run_id / "debug-summary.json").read_text())

    def results(self, run_id="t1"):
        return {r["stage"]: r["result"] for r in self.summary(run_id)["stages"]}

    def details(self, run_id="t1"):
        return {r["stage"]: r["detail"] for r in self.summary(run_id)["stages"]}

    def assert_destroyed(self):
        self.assertEqual(len(self.downs()), 1, self.out)
        self.assertIsNone(self.worker(), "the stub worker is still up")
        self.assertEqual(self.results()[12], "PASS", self.out)


class Ordering(DebugBase):
    def test_all_pass_in_order_and_exit_0(self):
        self.scen()
        self.assertEqual(self.run_debug(), 0, self.out)
        rows = self.summary()["stages"]
        self.assertEqual([r["stage"] for r in rows], list(range(1, 13)))
        self.assertEqual({r["result"] for r in rows}, {"PASS"}, self.out)
        # stdout status lines appear in stage order 1..12 (home checks may re-record 9 and 10)
        seen = []  # stage 0 is the rental itself
        for ln in self.out.splitlines():
            ln = ln.removeprefix("paid-run: ")
            if ln.startswith("debug-pass: stage "):
                n = int(ln.split()[2])
                if n and n not in seen:
                    seen.append(n)
        self.assertEqual(seen, list(range(1, 13)), self.out)
        self.assertIn("every stage passed", self.out)
        self.assert_destroyed()
        # one rental, one probe, one debug stage, then down; no train or preflight stage
        self.assertEqual(len(self.ups()), 1)
        self.assertEqual(self.stages(), ["omnivoice_debug"])
        kinds = [c[0] for c in self.calls() if c[0] in ("up", "probe", "stage", "down")]
        self.assertEqual(kinds, ["up", "probe", "stage", "down"])
        # the probe got the stage's floors; the worker got VI_DEBUG_PASS and the slice size
        self.assertIn(["probe", True], self.calls())
        self.assertIn(["debug_env", "1", "128"], self.calls())
        # summary files at home, including the worker's own summary
        d = self.tmp / "paid" / "t1"
        self.assertTrue((d / "debug-summary.txt").is_file())
        self.assertTrue((d / "debug-worker-summary.json").is_file())
        self.assertIn("never scored", self.summary()["listening_set_note"])

    def test_r30_push_probe_goes_into_the_summary(self):
        probe = {"ran": True, "cap_seconds": 300, "seconds": 212, "note": "random",
                 "legs": {"a": {"name": "tunnel-rsync", "status": "ok", "bytes": 209715200,
                                "planned": 209715200, "mb_s": 3.1, "seconds": 67.6},
                          "b": {"name": "4-streams", "status": "timeout", "bytes": 150000000,
                                "planned": 209715200, "mb_s": 1.67, "streams": 4,
                                "per_stream_mb_s": [0.4, 0.42, 0.41, 0.44]},
                          "c": {"name": "public-curl", "status": "ok", "down_status": "ok",
                                "down_mb_s": 95.5, "down_url": "https://x.invalid/100MB.bin",
                                "up_status": "ok", "up_mb_s": 40.2,
                                "up_url": "https://y.invalid/up"}}}
        self.scen(push_probe="push-probe: json " + json.dumps(probe))
        self.assertEqual(self.run_debug(), 0, self.out)
        got = self.summary()["push_probe"]
        self.assertEqual(got, probe)
        self.assertIn("null when it did not run", self.summary()["push_probe_note"])
        text = (self.tmp / "paid" / "t1" / "debug-summary.txt").read_text()
        self.assertIn("a tunnel-rsync : 3.1 MB/s, ok", text)
        self.assertIn("b 4-streams    : 1.67 MB/s in all, timeout (per stream 0.4, 0.42, 0.41, 0.44)", text)
        self.assertIn("c public-curl  : download 95.5 MB/s, ok (https://x.invalid/100MB.bin); "
                      "upload 40.2 MB/s, ok (https://y.invalid/up)", text)
        # the run log (stdout) has the legs too, and the stages are unchanged
        self.assertIn("a tunnel-rsync : 3.1 MB/s", self.out)
        self.assertEqual({r["result"] for r in self.summary()["stages"]}, {"PASS"})

    def test_r30_no_probe_line_is_null_and_said(self):
        self.scen()
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertIsNone(self.summary()["push_probe"])
        self.assertIn("push probe    : no result", self.out)

    def test_r30_a_broken_probe_line_never_breaks_the_pass(self):
        self.scen(push_probe="push-probe: json {not json")
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertIsNone(self.summary()["push_probe"])

    def test_r30_plan_names_the_probe_cost_and_time(self):
        self.scen()
        self.assertEqual(self.run_debug(plan=True), 0, self.out)
        self.assertIn("push probe      : after the worker preflight passes, at most 5 min", self.out)
        self.assertIn("worst case does not grow", self.out)

    def test_debug_rows_reaches_worker(self):
        self.scen()
        self.assertEqual(self.run_debug(debug_rows=16), 0, self.out)
        self.assertIn(["debug_env", "1", "16"], self.calls())

    def test_same_run_id_never_rents_twice(self):
        self.scen()
        self.assertEqual(self.run_debug(), 0, self.out)
        self.run_debug()
        self.assertIn("run already finished", self.out)
        self.assertEqual(len(self.ups()), 1, "a second rental happened")

    def test_unfinished_run_id_with_an_attempt_is_refused(self):
        # a debug pass whose state shows an attempt (e.g. killed mid-run) must not rent again
        self.scen(debug_fail=[4])
        self.assertEqual(self.run_debug(), 16, self.out)
        self.run_debug()
        self.assertEqual(len(self.ups()), 1, "a second rental happened: " + self.out)


class KeepGoing(DebugBase):
    def test_stage4_fail_keeps_going_and_skips_dependents(self):
        self.scen(debug_fail=[4])
        rc = self.run_debug()
        self.assertEqual(rc, 16, self.out)
        res = self.results()
        self.assertEqual(res[1], "PASS")
        self.assertEqual(res[2], "PASS")
        self.assertEqual(res[3], "PASS")
        self.assertEqual(res[4], "FAIL")
        for n in range(5, 12):
            self.assertEqual(res[n], "SKIP-dependent", (n, self.out))
        det = self.details()
        self.assertIn("needs stage 4 venv", det[5])
        self.assertIn("needs stage 4 venv", det[6])
        self.assertIn("needs stage 8 train-steps", det[9])
        self.assertIn("STOPPED (exit 16)", self.out)
        self.assert_destroyed()
        # the failing stage's log excerpt is kept
        ex = {r["stage"]: r["excerpt"] for r in self.summary()["stages"]}
        self.assertTrue(any("ModuleNotFoundError" in x for x in ex[4]), ex[4])

    def test_independent_stages_still_run_after_a_failure(self):
        # stage 10 fails, but 9 and 11 (which need only 8) still pass
        self.scen(debug_fail=[10])
        self.assertEqual(self.run_debug(), 16, self.out)
        res = self.results()
        self.assertEqual((res[9], res[10], res[11]), ("PASS", "FAIL", "PASS"))
        self.assert_destroyed()

    def test_probe_fail_does_not_stop_worker_stages(self):
        self.scen(probe="fail1")
        self.assertEqual(self.run_debug(), 16, self.out)
        res = self.results()
        self.assertEqual(res[1], "FAIL")
        self.assertEqual(res[2], "PASS")
        self.assertEqual(res[8], "PASS")
        self.assertIn("under 12.8", self.details()[1])
        self.assert_destroyed()

    def test_probe_unreachable_is_fail_not_skip(self):
        self.scen(probe="nothing")
        self.assertEqual(self.run_debug(), 16, self.out)
        res = self.results()
        self.assertEqual((res[1], res[2]), ("FAIL", "FAIL"))
        self.assertIn("without a stage 1 line", self.details()[1])
        self.assert_destroyed()

    def test_pushed_but_not_at_home_is_fail(self):
        self.scen(no_home_ckpt=True, no_home_wav=True)
        self.assertEqual(self.run_debug(), 16, self.out)
        res = self.results()
        self.assertEqual((res[9], res[10]), ("FAIL", "FAIL"), self.out)
        self.assertIn("no checkpoint run folder", self.details()[9])
        self.assertIn("no listening wav", self.details()[10])


class ThisRunsHomeDirs(DebugBase):
    """r31 F4: stage 9 and 10 read THIS run's stage-key dirs at home, never another run's newer
    ones (dbg6 copied an older run's summary.json because it took the newest dir by mtime)."""

    OTHER = "cd34" * 16

    def other_run(self):
        ck = Path(self.env["VI_CHECKPOINT_ROOT"]) / "omnivoice_debug" / f"{self.OTHER}-20261124T193837Z"
        (ck / "checkpoint-5").mkdir(parents=True)
        (ck / "checkpoint-5" / "model.safetensors").write_bytes(b"old")  # no SHA256SUMS
        out = self.tmp / "corpus" / "stages" / "omnivoice_debug" / self.OTHER / "out" / "debug"
        (out / "listening").mkdir(parents=True)
        (out / "listening" / "old.wav").write_bytes(b"RIFF")
        (out / "summary.json").write_text('{"5": "OLD RUN"}')
        future = 4102444800  # 2100-01-01: the other run's dirs are the newest by mtime
        for d in (ck, out, out.parent.parent):
            os.utime(d, (future, future))
        return ck

    def test_picks_this_runs_key_not_the_newest_dir(self):
        self.other_run()
        self.scen()
        self.assertEqual(self.run_debug(), 0, self.out)
        det = self.details()
        self.assertEqual(self.results()[9], "PASS", self.out)
        self.assertIn("ab12ab12", det[9])
        self.assertNotIn(self.OTHER, det[9])
        self.assertIn(f".tmp-{'ab12' * 16}", det[10])
        got = (self.tmp / "paid" / "t1" / "debug-worker-summary.json").read_text()
        self.assertNotIn("OLD RUN", got)

    def test_another_runs_dirs_never_stand_in_for_this_runs_missing_ones(self):
        self.other_run()
        self.scen(no_home_ckpt=True, no_home_wav=True)
        self.assertEqual(self.run_debug(), 16, self.out)
        res, det = self.results(), self.details()
        self.assertEqual((res[9], res[10]), ("FAIL", "FAIL"), self.out)
        self.assertIn(f"stage key {'ab12' * 16}", det[9])
        self.assertIn("no listening wav", det[10])

    def test_run_key_parsing(self):
        pr = tp.load_paid_run()
        k = "43c8eb65a5f29300ade7f0cb97193c59ba98e58ba02690b88243c679ea5eef0c"
        self.assertEqual(pr.debug_run_key(
            "omnivoice_debug: running (43c8eb65a5f2)\n"
            f"debug-pass: stage 3 upload: running (limit 60 s; log /root/vi-jobs/omnivoice_debug."
            f"{k}.20261125T113310Z/out/debug/03-upload.log)"), (k, "20261125T113310Z"))
        self.assertEqual(pr.debug_run_key("omnivoice_debug: running (43c8eb65a5f2)"),
                         ("43c8eb65a5f2", None))
        self.assertEqual(pr.debug_run_key("nothing here"), (None, None))


class SkipDependent(DebugBase):
    def test_rental_failure_skips_1_to_11_and_names_it(self):
        self.scenario([dict(up="fail")], corpus=str(self.tmp / "corpus"))
        self.assertEqual(self.run_debug(), 16, self.out)
        res, det = self.results(), self.details()
        for n in range(1, 12):
            self.assertEqual(res[n], "SKIP-dependent", (n, self.out))
        self.assertIn("needs the rental", det[1])
        self.assertIn("needs the rental", det[3])
        self.assertEqual(self.summary()["rental"]["result"], "FAIL")
        self.assertEqual(self.stages(), [], "no stage may run without a worker")
        # stage 12 always runs: the failed `up` may have left a machine, so down is called
        self.assertEqual(len(self.downs()), 1, self.out)
        self.assertEqual(res[12], "PASS")

    def test_silent_worker_stage_fails_3_and_skips_the_rest(self):
        self.scen(debug="silent")
        self.assertEqual(self.run_debug(), 16, self.out)
        res, det = self.results(), self.details()
        self.assertEqual(res[3], "FAIL")
        self.assertIn("never reported", det[3])
        self.assertEqual(res[4], "SKIP-dependent")
        self.assertIn("needs stage 3 upload", det[4])
        self.assert_destroyed()


class AlwaysDestroyed(DebugBase):
    def test_worker_stage_timeout_destroys_worker(self):
        # every per-call timeout shrinks to its 2 s floor, so the hung stage is killed fast
        self.env["PAID_RUN_TIMEOUT_SCALE"] = "0.00001"
        self.scen(debug="hang")
        rc = self.run_debug()
        self.assertEqual(rc, 16, self.out)
        res, det = self.results(), self.details()
        self.assertEqual((res[3], res[4]), ("PASS", "PASS"))
        self.assertEqual(res[5], "FAIL")
        self.assertIn("hit its", det[5])
        self.assertEqual(res[6], "SKIP-dependent")
        self.assert_destroyed()

    def test_probe_timeout_destroys_worker(self):
        self.env["PAID_RUN_TIMEOUT_SCALE"] = "0.00001"
        self.scen(probe="hang")
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertIn("hit its", self.details()[1])
        self.assertEqual(self.results()[3], "PASS", "worker stages still ran after the probe hung")
        self.assert_destroyed()

    def test_exception_in_a_stage_still_destroys_worker(self):
        # paid_run under a wrapper whose debug_probe raises: the worker must still come down
        wrap = self.tmp / "raise_wrap.py"
        wrap.write_text(
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('pr', {str(SCRIPT)!r})\n"
            "pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)\n"
            "def boom(self): raise RuntimeError('injected failure in stage 1')\n"
            "pr.PaidRun.debug_probe = boom\n"
            "sys.exit(pr.main(sys.argv[1:]))\n")
        self.scen()
        a = self.args(debug_pass=True, stages=self.stages_path)
        p = subprocess.run([sys.executable, "-I", str(wrap), *a], env=self.env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        self.out = p.stdout
        self.assertNotEqual(p.returncode, 0, self.out)
        self.assertIn("injected failure", self.out)
        self.assert_destroyed()
        res = self.results()
        self.assertEqual(res[0] if 0 in res else self.summary()["rental"]["result"], "PASS")
        # the stages after the exception were never reached; they are reported, not silently lost
        self.assertNotEqual(res[3], "PASS")


class HostDrop(DebugBase):
    """Round 19 A1+A2+C: a confirmed host drop saves Vast's view, then ONE more rental."""

    def scen2(self, first, second=None, cap_spent=(0.30, 0.42), **extra):
        self.scenario([dict(spent=cap_spent[0], **first), dict(spent=cap_spent[1], **(second or {}))],
                      corpus=str(self.tmp / "corpus"), **extra)

    def evidence(self, n=1):
        return sorted((self.tmp / "paid" / "t1").glob(f"vast-attempt-{n}-*.txt"))

    def test_worker_drop_saves_evidence_then_rents_once_more(self):
        self.scen2(dict(debug="drop"))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(len(self.ups()), 2, self.out)
        self.assertEqual(len(self.downs()), 2, self.out)
        sm = self.summary()
        r1, r2 = sm["rentals"]
        self.assertTrue(r1["status"].startswith("HOST-DROP (status exited"), r1)
        self.assertEqual((r1["instance"], r2["instance"]), ("101", "102"))
        self.assertAlmostEqual(r1["spent"], 0.30)
        self.assertAlmostEqual(r2["spent"], 0.42)
        self.assertAlmostEqual(sm["spent_total"], 0.72)
        st1 = {r["stage"]: r for r in r1["stages"]}
        self.assertEqual(st1[3]["result"], "HOST-DROP")
        self.assertNotIn("never reported", st1[3]["detail"])
        self.assertIn("exit 255 from on-worker.sh", st1[3]["detail"])
        self.assertEqual(self.results()[3], "PASS")  # the table shown is rental 2's
        # A1: Vast's view was saved before the first destroy
        ev = self.evidence(1)
        self.assertTrue(ev, self.out)
        self.assertIn("actual_status exited", ev[0].read_text())
        names = [c[0] for c in self.calls()]
        self.assertLess(names.index("vastai"), names.index("down"))
        self.assertIn(["vastai", ["show", "instance", "101", "--raw"]], self.calls())
        self.assertIn(["vastai", ["logs", "101", "--tail", "200"]], self.calls())
        text = (self.tmp / "paid" / "t1" / "debug-summary.txt").read_text()
        self.assertIn("rental 1      : offer 1, instance 101, HOST-DROP (status exited", text)
        self.assertIn("rental 2      : offer 2, instance 102, released", text)
        self.assertIn("spent $0.72 of $5.00 cap in 2 rental(s)", text)
        self.assertIn("HOST-DROP (status exited", self.out)

    def test_r31_dbg6_drop_with_exit_1_is_confirmed_and_rents_once_more(self):
        # r31 F1: dbg6's lines and exit 1 used to give no drop signal, so Vast was never asked
        self.scen2(dict(debug="drop_dbg6", vast_gone=True))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(len(self.ups()), 2, self.out)
        self.assertEqual(len(self.show_calls(101)), 2, self.out)
        r1 = self.summary()["rentals"][0]
        self.assertTrue(r1["status"].startswith("HOST-DROP (status gone"), r1)
        st1 = {r["stage"]: r for r in r1["stages"]}
        self.assertEqual((st1[3]["result"], st1[4]["result"]), ("PASS", "PASS"))
        self.assertEqual(st1[5]["result"], "HOST-DROP", st1[5])
        self.assertIn("ssh may have lost the host", self.out)

    def test_r31_ordinary_exit_1_without_a_drop_line_is_not_a_drop(self):
        # stage 3 never reports and the stage ends with exit 1 and no ssh or tunnel line
        self.scen2(dict(debug="silent"))
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)
        self.assertNotIn("HOST-DROP", self.results().values())
        self.assertNotIn("ssh may have lost the host", self.out)

    def test_two_drops_never_rent_a_third(self):
        self.scenario([dict(spent=0.3, debug="drop"), dict(spent=0.3, debug="drop"), dict(spent=0.3)],
                      corpus=str(self.tmp / "corpus"))
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), 2, "a third rental happened: " + self.out)
        self.assertIsNone(self.worker())
        self.assertIn("no third", self.summary()["note"])
        self.assertEqual(self.results()[3], "HOST-DROP")
        self.assertTrue(self.evidence(2))

    def test_probe_drop_rents_once_more_without_running_worker_stages_on_the_dead_host(self):
        self.scen2(dict(probe="drop"))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(len(self.ups()), 2, self.out)
        r1 = {r["stage"]: r["result"] for r in self.summary()["rentals"][0]["stages"]}
        self.assertEqual((r1[1], r1[2]), ("HOST-DROP", "HOST-DROP"))
        self.assertEqual(self.stages(), ["omnivoice_debug"], "a worker stage ran on the dead host")

    def test_our_own_failures_never_rent_again(self):
        for extra in (dict(debug_fail=[4]), dict(debug="silent"), dict(probe="fail1")):
            with self.subTest(**{k: str(v) for k, v in extra.items()}):
                self.setUp()
                self.scen2({}, **extra)
                self.assertEqual(self.run_debug(), 16, self.out)
                self.assertEqual(len(self.ups()), 1, "a second rental after our own failure")
                self.assertNotIn("HOST-DROP", self.results().values())
                self.assertFalse(self.summary()["rentals"][0]["status"].startswith("HOST-DROP"))

    def test_unconfirmed_ssh_loss_is_not_a_drop(self):
        # ssh said "connection refused" once, but the instance runs and ssh works again
        self.scen2({}, probe="nothing")
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)
        self.assertIn("not a confirmed host drop", self.details()[1])
        self.assertIn("Vast said running", self.details()[1])

    def test_second_rental_only_if_the_cap_left_covers_it(self):
        self.scen2(dict(debug="drop"), cap_spent=(0.95, 0.1))
        self.assertEqual(self.run_debug(cap=1, max_hours=1), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)
        note = self.summary()["note"]
        self.assertIn("no second rental", note)
        self.assertIn("no offer fits what is left", note)
        self.assertIn("every worker released", self.details()[12])

    def test_vastai_failing_or_hanging_never_blocks_the_destroy_and_is_never_a_drop(self):
        # r21 F1: home's own Vast API failing is not a host drop, so no second rental either
        for mode in ("fail", "hang"):
            with self.subTest(mode=mode):
                self.setUp()
                self.env["PAID_RUN_VAST_SECONDS"] = "1"
                self.scen2(dict(debug="drop"), vastai=mode)
                self.assertEqual(self.run_debug(), 16, self.out)
                self.assertEqual(len(self.ups()), 1, self.out)
                self.assertEqual(len(self.downs()), 1, self.out)
                self.assertIsNone(self.worker())
                self.assertFalse(self.summary()["rentals"][0]["status"].startswith("HOST-DROP"),
                                 self.out)
                self.assertNotIn("HOST-DROP", self.results().values())
                self.assertIn("could not read the instance status from Vast", self.details()[3])

    def show_calls(self, ident):
        return [c for c in self.calls() if c == ["vastai", ["show", "instance", str(ident), "--raw"]]]

    def test_drop_needs_two_vast_reads_at_least_60_s_apart(self):
        # r21 F1: two `vastai show instance` reads, a 60 s wait between them, both before the destroy
        self.scen2(dict(debug="drop"))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(len(self.show_calls(101)), 2, self.out)
        sleeps = (self.tmp / "sleeps.log").read_text().split()
        self.assertIn(60.0, [float(x) for x in sleeps], sleeps)
        names = [c[0] if c[0] != "vastai" else " ".join(c[1][:2]) for c in self.calls()]
        first_down = names.index("down")
        self.assertEqual(names[:first_down].count("show instance"), 2, names)

    def test_second_read_running_or_failing_is_not_a_drop(self):
        for seq in (["exited", "running"], ["exited", "FAIL"], ["FAIL", "exited"]):
            with self.subTest(seq=seq):
                self.setUp()
                self.scen2(dict(debug="drop", vast_status_seq=seq))
                self.assertEqual(self.run_debug(), 16, self.out)
                self.assertEqual(len(self.ups()), 1, self.out)
                self.assertNotIn("HOST-DROP", self.results().values())
                self.assertIn("not a confirmed host drop", self.details()[3])

    # r27: the dbg5 transcript (2026-11-24): Vast no longer had instance 62400251, its logs said
    # "No such container", and ssh stayed refused; r26 code called that "not a confirmed host drop".
    def test_r27_instance_gone_with_ssh_refused_is_a_drop_and_rents_once_more(self):
        self.scen2(dict(debug="drop", vast_gone=True))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(len(self.ups()), 2, self.out)
        self.assertEqual(len(self.show_calls(101)), 2, self.out)
        r1 = self.summary()["rentals"][0]
        self.assertTrue(r1["status"].startswith("HOST-DROP (status gone: Vast lists no such "
                                                "instance (instances: null); its logs say No "
                                                "such container)"), r1)
        self.assertIn("ssh again: Connection refused", r1["status"])
        ev = self.evidence(1)[0].read_text()
        self.assertIn('"instances": null', ev)
        self.assertIn("No such container: C.101", ev)

    def test_r27_running_or_loading_with_ssh_refused_is_not_a_drop(self):
        for seq in (["running"], ["loading"], ["loading", "loading"]):
            with self.subTest(seq=seq):
                self.setUp()
                self.scen2(dict(debug="drop", vast_status_seq=seq))
                self.assertEqual(self.run_debug(), 16, self.out)
                self.assertEqual(len(self.ups()), 1, self.out)
                self.assertNotIn("HOST-DROP", self.results().values())
                self.assertIn("not a confirmed host drop", self.details()[3])

    def test_r27_api_error_is_not_a_drop_even_with_no_such_container_in_the_logs(self):
        for seq in (["FAIL"], ["NULL", "FAIL"], ["FAIL", "NULL"]):
            with self.subTest(seq=seq):
                self.setUp()
                self.scen2(dict(debug="drop", vast_gone=True, vast_status_seq=seq))
                self.assertEqual(self.run_debug(), 16, self.out)
                self.assertEqual(len(self.ups()), 1, self.out)
                self.assertNotIn("HOST-DROP", self.results().values())
                self.assertIn("not a confirmed host drop", self.details()[3])

    def test_r27_instance_gone_but_ssh_answers_again_is_not_a_drop(self):
        self.scen2(dict(debug="drop", vast_gone=True, ssh_true_ok=True))
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)
        self.assertNotIn("HOST-DROP", self.results().values())
        self.assertIn("Vast said gone, then gone, but ssh answers again", self.details()[3])

    def test_debug_pass_sets_lost_minutes_5_and_r9_keeps_the_default(self):
        self.scen()
        self.run_debug()
        lost = [c for c in self.calls() if c[0] == "lost_minutes"]
        self.assertEqual(lost, [["lost_minutes", "omnivoice_debug", "5"]], self.out)
        self.setUp()
        self.scenario([dict(spent=0.4)])
        self.run_orch()
        lost = [c[2] for c in self.calls() if c[0] == "lost_minutes"]
        self.assertTrue(lost, self.out)
        self.assertEqual(set(lost), {None}, "r9 must keep on-worker.sh's 30 min default")


# r27: the two Vast answers saved in ~/.local/state/vi/paid-run/omnivoice-ft-dbg5/
# vast-attempt-1-host-drop-check.txt (2026-11-24), verbatim.
DBG5_SHOW = '{\n "instances": null\n}\n'
DBG5_LOGS = "Error response from daemon: No such container: C.62400251\n"


class HungVerifyEpoch(unittest.TestCase):
    """r32 WP7: the home verify-epoch of stage 9 is bounded (VERIFY_EPOCH_TIMEOUT_SECONDS = 300 s),
    so a hung check can never hold home while the rental bills. A fake verify that sleeps stands
    in; the limit is patched to 1 s so the test never waits 300 s."""

    def test_a_hung_verify_epoch_is_stopped_and_fails_stage_9(self):
        import time
        pr = tp.load_paid_run()
        self.assertEqual(pr.VERIFY_EPOCH_TIMEOUT_SECONDS, 300)
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "run"
            ck.mkdir()
            run = pr.PaidRun.__new__(pr.PaidRun)
            run.debug = {}
            run.debug_record(9, "PASS", "(pushed)")
            run.debug_home_dirs = lambda out="": (ck, None)
            run.dir = Path(tmp) / "paid"
            self.addCleanup(setattr, pr.debug, "VERIFY_EPOCH_TIMEOUT_SECONDS",
                            pr.debug.VERIFY_EPOCH_TIMEOUT_SECONDS)
            self.addCleanup(setattr, pr.debug, "verify_epoch_argv", pr.debug.verify_epoch_argv)
            pr.debug.VERIFY_EPOCH_TIMEOUT_SECONDS = 1
            pr.debug.verify_epoch_argv = lambda d: [sys.executable, "-c",
                                              "import time; print('hashing', flush=True); "
                                              "time.sleep(600)"]
            t0 = time.monotonic()
            run.debug_home_checks("")
            self.assertLess(time.monotonic() - t0, 30)
        self.assertEqual(run.debug[9]["result"], "FAIL")
        self.assertIn("did not finish in 1 s and was stopped", run.debug[9]["detail"])


class VastInstanceGone(unittest.TestCase):
    """r27: vast_instance_gone() reads Vast's own "no such instance" answers, and nothing else."""

    def setUp(self):
        self.pr = tp.load_paid_run()

    def test_dbg5_transcript_is_gone(self):
        why = self.pr.vast_instance_gone(0, DBG5_SHOW, 0, DBG5_LOGS)
        self.assertEqual(why, "Vast lists no such instance (instances: null); "
                              "its logs say No such container")
        self.assertTrue(self.pr.vast_instance_gone(0, '{"instances": []}', 1, ""))
        self.assertTrue(self.pr.vast_instance_gone(0, "[]", 0, ""))
        # a status-less answer plus the daemon's "No such container" also says gone
        self.assertTrue(self.pr.vast_instance_gone(0, '{"instances": {"id": 1}}', 0, DBG5_LOGS))

    def test_a_listed_instance_or_an_api_error_is_never_gone(self):
        g = self.pr.vast_instance_gone
        running = '{"instances": {"id": 1, "actual_status": "running"}}'
        self.assertIsNone(g(0, running, 0, DBG5_LOGS))
        self.assertIsNone(g(0, '{"id": 1, "actual_status": "loading"}', 0, DBG5_LOGS))
        self.assertIsNone(g(1, DBG5_SHOW, 0, DBG5_LOGS))            # show failed
        self.assertIsNone(g(1, "failed with error 500: server error", 0, DBG5_LOGS))
        self.assertIsNone(g(0, "not json", 0, DBG5_LOGS))
        self.assertIsNone(g(0, '{"success": false, "error": "invalid_args"}', 0, DBG5_LOGS))
        self.assertIsNone(g(0, '{"instances": {"id": 1}}', 1, DBG5_LOGS))  # logs call failed
        self.assertIsNone(g(0, '{"instances": {"id": 1}}', 0, "ok"))


class AbandonDebugMarkers(DebugBase):
    """r21 F4: after the release, an `omnivoice_debug-*` unverified marker whose checkpoint dir is
    empty or missing is abandoned (jobs/checkpoints.py abandon); r31 F3: so is one whose checkpoint
    has no SHA256SUMS (dbg5's partial push), files kept. A debug dir with a manifest and any other
    stage's marker are never touched. Fake marker dirs only (VI_STATE_DIR is the test's)."""

    def make_markers(self):
        mdir = Path(self.env["VI_STATE_DIR"]) / "unverified"
        mdir.mkdir(parents=True, exist_ok=True)
        ck = self.tmp / "ck"
        dirs = dict(missing=ck / "omnivoice_debug" / "gone", empty=ck / "omnivoice_debug" / "empty",
                    full=ck / "omnivoice_debug" / "full", other=ck / "omnivoice_train" / "empty",
                    partial=ck / "omnivoice_debug" / "partial",
                    other_partial=ck / "omnivoice_train" / "partial")
        dirs["empty"].mkdir(parents=True)
        dirs["full"].mkdir(parents=True)
        (dirs["full"] / "checkpoint-5").mkdir()
        (dirs["full"] / "checkpoint-5" / "SHA256SUMS").write_text("")
        dirs["other"].mkdir(parents=True)
        for k in ("partial", "other_partial"):  # dbg5's shape: weights, an rsync partial, no manifest
            (dirs[k] / "checkpoint-5" / ".rsync-partial").mkdir(parents=True)
            (dirs[k] / "checkpoint-5" / "model.safetensors").write_bytes(b"w")
            (dirs[k] / "checkpoint-5" / ".rsync-partial" / "optimizer.bin").write_bytes(b"o")
        names = dict(missing="omnivoice_debug-gone", empty="omnivoice_debug-empty",
                     full="omnivoice_debug-full", other="omnivoice_train-empty",
                     partial="omnivoice_debug-partial", other_partial="omnivoice_train-partial")
        for k, d in dirs.items():
            (mdir / names[k]).write_text(f"{d}\n")
        return mdir, dirs, names

    def check(self, mdir, dirs, names):
        left = sorted(x.name for x in mdir.iterdir())
        self.assertEqual(left, ["omnivoice_debug-full", "omnivoice_train-empty",
                                "omnivoice_train-partial"], self.out)
        self.assertTrue((dirs["empty"] / "ABANDONED").is_file(), self.out)
        self.assertTrue((dirs["partial"] / "ABANDONED").is_file(), self.out)
        self.assertTrue((dirs["partial"] / "checkpoint-5" / "model.safetensors").is_file())
        self.assertFalse((dirs["other_partial"] / "ABANDONED").exists())
        self.assertFalse((dirs["full"] / "ABANDONED").exists())
        self.assertTrue((dirs["full"] / "checkpoint-5").is_dir())
        self.assertFalse((dirs["other"] / "ABANDONED").exists())
        self.assertFalse(dirs["missing"].exists())
        self.assertIn("abandoned 3 empty or incomplete debug marker(s)", self.out)

    def test_release_abandons_only_empty_or_missing_debug_markers(self):
        mdir, dirs, names = self.make_markers()
        self.scen()
        self.assertEqual(self.run_debug(), 0, self.out)
        self.check(mdir, dirs, names)

    def test_abandon_also_runs_after_a_failed_pass(self):
        mdir, dirs, names = self.make_markers()
        self.scen(debug_fail=[8])
        self.assertEqual(self.run_debug(), 16, self.out)
        self.check(mdir, dirs, names)


class RentalHours(DebugBase):
    """r21 F2: each debug rental has its own hours (--max-hours is per rental); the $ cap stays
    shared, and a stage the worker had no time for reads SKIP-no-time, not FAIL."""

    def test_second_rental_gets_its_own_hours(self):
        # rental 1 uses 1.0 h of 1.25 before its host drops; rental 2 still gets 1.25 h
        self.scenario([dict(spent=0.25, debug="drop", hours=1.0), dict(spent=0.3)],
                      corpus=str(self.tmp / "corpus"))
        self.assertEqual(self.run_debug(cap=1, max_hours=1.25), 0, self.out)
        ups = self.ups()
        self.assertEqual(len(ups), 2, self.out)
        self.assertEqual([u["max_hours"] for u in ups], ["1.2500", "1.2500"], self.out)
        left = [float(c[1]) for c in self.calls() if c[0] == "debug_hours_left"]
        self.assertEqual(len(left), 2, self.out)
        self.assertGreater(left[1], 1.2, "rental 2's worker was told only the shared hours left")
        self.assertIn("1.25 h per rental", self.out)

    def test_cap_is_still_shared_and_must_cover_price_x_rental_hours(self):
        # $1 cap, rental 1 spent $0.80: offer 2 at $0.25/hr x 1.25 h = $0.31 > $0.20 left
        self.scenario([dict(spent=0.80, debug="drop"), dict(spent=0.1)],
                      offer_list=tp.offers((1, 0.20, "US", 800), (2, 0.25, "DE", 900)),
                      corpus=str(self.tmp / "corpus"))
        self.assertEqual(self.run_debug(cap=1, max_hours=1.25), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)
        self.assertIn("no second rental", self.summary()["note"])
        self.assertIn("1.25 h", self.summary()["note"])

    def test_skip_no_time_is_shown_apart_from_fail(self):
        self.scen(debug_no_time=[9, 10, 11])
        self.assertEqual(self.run_debug(), 16, self.out)
        res, det = self.results(), self.details()
        self.assertEqual([res[n] for n in (9, 10, 11)], ["SKIP-no-time"] * 3, self.out)
        self.assertIn("left before the rental's deadline", det[9])
        self.assertNotIn("FAIL", [res[n] for n in range(1, 13)])
        text = (self.tmp / "paid" / "t1" / "debug-summary.txt").read_text()
        self.assertIn(" 9 ckpt-push   : SKIP-no-time", text)
        self.assertIn("3 stage(s) were not started for lack of time (SKIP-no-time), which is "
                      "not a failure of the stage", text)


class Plan(DebugBase):
    def test_debug_plan_offline_exit_0_with_worst_case(self):
        self.scen()
        self.wrap = self.tmp / "vi-wrap"
        self.wrap.write_text(f"#!/bin/sh\necho \"$*\" >> {self.tmp}/any-call.log\n"
                             f"exec {self.stub} \"$@\"\n")
        self.wrap.chmod(0o755)
        self.env["VI_BIN"] = str(self.wrap)
        rc = self.run_debug(plan=True, cap=3, max_hours=1)
        self.assertEqual(rc, 0, self.out)
        self.assertIn("worst case", self.out)
        self.assertIn("cap $3.00", self.out)
        self.assertIn("up to 2 rentals", self.out)
        self.assertIn("a second only after a confirmed host drop", self.out)
        self.assertIn("never a third", self.out)
        self.assertIn("cap $3.00 for both rentals together", self.out)

    def test_debug_plan_describes_two_rentals_of_their_own_hours(self):
        # r21 F2: 2 x 1.25 h; r27: the repo config's max_dph is 0.60 (was 0.31), so the GPU worst
        # case is 2 x 1.25 h x $0.60 = $1.50, inside the dbg6 cap of $1.75
        self.scen()
        self.assertEqual(self.run_debug(plan=True, cap=1.75, max_hours=1.25), 0, self.out)
        self.assertIn("max hours       : 1.25 h per rental (2 x 1.25 h at most)", self.out)
        self.assertIn("GPU $1.50 (2 rentals x 1.25 h x $0.600/hr max_dph, inside the $1.75 cap)",
                      self.out)
        self.assertRegex(self.out, r"worst case      : \$1\.50 total = GPU \$1\.50 \(.*\) "
                                   r"\(max_dph is the full billed rate, so the 125 GB disk is "
                                   r"inside it\)")
        self.assertNotRegex(self.out, r"worst case.*\+ disk")
        self.assertIn("offer search    : datacenter=true", self.out)
        self.assertRegex(self.out, r"countries       : \S.*never CN")
        self.assertNotIn("max hours (all)", self.out)
        self.assertIn("--plan: nothing was called", self.out)
        self.assertFalse((self.tmp / "any-call.log").exists(), "--plan called vi")
        self.assertEqual(self.calls(), [])

    def test_worst_case_formula(self):
        pr = tp.load_paid_run()
        # two rentals of 1.25 h each: GPU min(cap, 2 x 1.25 x max_dph), disk for 2 x 1.25 h
        total, gpu, disk = pr.debug_worst_case(3.0, 125, 1.25, 0.31)
        self.assertAlmostEqual(gpu, 2 * 1.25 * 0.31)
        # max_dph is dph_total, which already prices the rented disk: no second disk charge
        self.assertAlmostEqual(disk, 0.0)
        self.assertAlmostEqual(total, gpu)
        # no max_dph in the config: the cap is the GPU worst case, disk on top
        total, gpu, disk = pr.debug_worst_case(3.0, 125, 1.25, None)
        self.assertAlmostEqual(gpu, 3.0)
        self.assertAlmostEqual(disk, 125 * 1.00 / 730 * 2.5)
        # a cap under 2 x hours x max_dph still bounds the GPU part
        self.assertAlmostEqual(pr.debug_worst_case(0.5, 125, 1.25, 0.31)[1], 0.5)

    def test_plain_plan_still_exit_0(self):
        self.scen()
        self.assertEqual(self.run_orch(plan=True), 0, self.out)
        self.assertNotIn("up to 2 rentals", self.out)

    def test_debug_rows_must_be_positive(self):
        self.scen()
        self.assertNotEqual(self.run_debug(plan=True, debug_rows=0), 0, self.out)


class ParseOutput(unittest.TestCase):
    def test_parse_lines_and_excerpts(self):
        pr = tp.load_paid_run()
        text = ("noise\n"
                "debug-pass: stage 4 venv: FAIL (exit 1, 3 s)\n"
                "debug-pass:   | ImportError: x\n"
                "debug-pass: stage 5 base-model: SKIP-dependent (needs stage 4 venv, which did not pass)\n"
                "debug-pass: stage 6 tokenise: still running (60 s)\n")
        got = pr.parse_debug_output(text)
        self.assertEqual(got[4][0], "FAIL")
        self.assertEqual(got[4][2], ["ImportError: x"])
        self.assertEqual(got[5][0], "SKIP-dependent")
        self.assertNotIn(6, got, "a heartbeat is not a result")

    def test_r23_key_lines_survive_the_12_line_cap(self):
        pr = tp.load_paid_run()
        ex = ["key: train_ft: GPU out of memory", "key: accelerate: child returned non-zero exit status 76",
              "last 12 lines:"] + [f"tail {i}" for i in range(12)]
        kept = pr.debug_excerpt_keep(ex)
        self.assertEqual(kept[:2], ex[:2])
        self.assertEqual(kept[-12:], ex[-12:])
        self.assertEqual(pr.debug_excerpt_keep([f"t{i}" for i in range(20)]),
                         [f"t{i}" for i in range(8, 20)])

    def test_stage_table_matches_worker_script(self):
        pr = tp.load_paid_run()
        self.assertEqual([n for n, _, _ in pr.DEBUG_STAGES], list(range(1, 13)))
        for n, name in NAMES.items():
            self.assertEqual(pr.DEBUG_NAMES[n], name)
        text = (JOBS / "debug-pass.sh").read_text()
        for n, name in NAMES.items():
            self.assertIn(f"[{n}]={name}", text)


class Scripts(unittest.TestCase):
    def test_shell_syntax(self):
        for s in ("debug-pass.sh", "debug-probe.sh", "on-worker.sh", "push-probe.sh",
                  "rehearsal/debug-pass.sh"):
            p = JOBS / s
            if p.exists():
                r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, (s, r.stderr))

    def test_debug_slice_takes_first_rows_by_position(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t) / "src", Path(t) / "dst"
            (src / "audio").mkdir(parents=True)
            (src / "voices").mkdir()
            (src / "voices" / "v.wav").write_bytes(b"v")
            (src / "data_config.json").write_text("{}")
            rows = []
            for i in range(10):
                (src / "audio" / f"{i}.wav").write_bytes(b"a")
                rows.append(json.dumps({"id": f"c{i}", "audio_path": f"audio/{i}.wav"}))
            (src / "train_manifest.jsonl").write_text("\n".join(rows) + "\n")
            (src / "dev_manifest.jsonl").write_text("\n".join(rows) + "\n")
            r = subprocess.run([sys.executable, "-I", str(JOBS / "debug_slice.py"), str(src),
                                str(dst), "--train-rows", "3"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            train = (dst / "train_manifest.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(x)["id"] for x in train], ["c0", "c1", "c2"])
            self.assertEqual(len((dst / "dev_manifest.jsonl").read_text().splitlines()), 4)
            self.assertEqual(sorted(p.name for p in (dst / "audio").iterdir()),
                             ["0.wav", "1.wav", "2.wav", "3.wav"])
            self.assertTrue((dst / "voices" / "v.wav").is_file())
            self.assertTrue((dst / "data_config.json").is_file())
            r = subprocess.run([sys.executable, "-I", str(JOBS / "debug_slice.py"), str(src),
                                str(dst), "--train-rows", "0"], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            # A rebuild over its own slice works; a foreign non-empty folder is never removed.
            r = subprocess.run([sys.executable, "-I", str(JOBS / "debug_slice.py"), str(src),
                                str(dst), "--train-rows", "2"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len((dst / "train_manifest.jsonl").read_text().splitlines()), 2)
            other = Path(t) / "other"
            other.mkdir()
            (other / "keep.txt").write_text("keep")
            r = subprocess.run([sys.executable, "-I", str(JOBS / "debug_slice.py"), str(src),
                                str(other)], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not a debug slice", r.stderr)
            self.assertTrue((other / "keep.txt").is_file())

    def test_probe_runs_locally_and_reports_both_stages(self):
        r = subprocess.run(["bash", str(JOBS / "debug-probe.sh")], capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        for n, name in ((1, "host-cuda"), (2, "disk-limits")):
            self.assertRegex(r.stdout, rf"debug-pass: stage {n} {name}: (PASS|FAIL)")


class DriverScript(unittest.TestCase):
    """r21 F2 + F3: the real jobs/debug-pass.sh driver at home with a fake venv python (no GPU,
    no network). A copy runs from a temp jobs folder so NEEDS can be loosened per test; the
    worker functions it cuts out of voicetune-fresh.sh and worker-limits.sh are the real ones."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="debug-driver."))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)]))
        self.jobs = self.tmp / "jobs"
        self.jobs.mkdir()
        for f in ("voicetune-fresh.sh", "worker-limits.sh"):
            (self.jobs / f).symlink_to(JOBS / f)
        self.work = self.tmp / "work"
        self.out = self.work / "out"
        self.out.mkdir(parents=True)
        self.data = self.tmp / "data"
        self.data.mkdir()
        (self.data / "data_config.json").write_text("{}")
        self.state = self.work / "voicetune-fresh" / "debug-state"
        self.probe_run = self.work / "voicetune-fresh" / "probe-run"

    def script(self, needs=None):
        text = (JOBS / "debug-pass.sh").read_text()
        if needs is not None:
            old = 'NEEDS=([3]="" [4]="3" [5]="4" [6]="4 5" [7]="4 5" [8]="6 7" [9]="8" [10]="8" [11]="8")'
            self.assertIn(old, text, "debug-pass.sh NEEDS changed; update this test")
            text = text.replace(old, needs)
        path = self.jobs / "debug-pass.sh"
        path.write_text(text)
        return path

    def fake_venv(self, trainer_body):
        """A venv python whose `-m accelerate.commands.launch ... train_ft.py` runs trainer_body."""
        self.state.mkdir(parents=True, exist_ok=True)
        py = self.tmp / "fakepy"
        py.write_text("#!/bin/bash\ncase \"$*\" in *train_ft.py*) " + trainer_body
                      + " ;; esac\nexit 0\n")
        py.chmod(0o755)
        (self.state / "py").write_text(f"{py}\n")
        base = self.tmp / "base"
        base.mkdir(exist_ok=True)
        (self.state / "base").write_text(f"{base}\n")

    def run_driver(self, stages, needs=None, timeout=120, **env):
        e = dict(os.environ, VI_OUT=str(self.out), VI_INPUT_OMNIVOICE_DATA=str(self.data),
                 VI_DEBUG_STAGES=stages, VI_DEBUG_POLL_SECONDS="1", VI_DEBUG_EXCERPT_LINES="40")
        for k in ("VI_HOURS_LEFT", "VI_DEBUG_MARGIN_SECONDS"):
            e.pop(k, None)
        e.update(env)
        p = subprocess.run(["bash", str(self.script(needs))], env=e, text=True, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.output = p.stdout
        self.assertEqual(p.returncode, 0, self.output)
        return {r["stage"]: r for r in
                json.loads((self.out / "debug" / "summary.json").read_text())["stages"]}

    def test_f3_stage8_cut_off_mid_run_keeps_its_progress_lines(self):
        # base: train.log was only cat'ed after the trainer ended, so a killed stage 8 left an
        # empty stage log and the FAIL line had no excerpt
        self.fake_venv("echo 'train_ft: step 1/5 loss 3.1'; echo 'train_ft: step 2/5 loss 3.0'; "
                       "sleep 60")
        res = self.run_driver("8", needs='NEEDS=([8]="")', VI_DEBUG_TIMEOUT_8="4")
        self.assertEqual(res[8]["result"], "FAIL", self.output)
        log = (self.out / "debug" / "08-train-steps.log").read_text()
        self.assertIn("train_ft: step 2/5", log)
        self.assertIn("debug-pass:   | train_ft: step 2/5", self.output)
        self.assertIn("train_ft: step 1/5", (self.probe_run / "train.log").read_text())

    def test_f3_stage8_trainer_failure_still_fails_the_stage(self):
        self.fake_venv("echo 'train_ft: step 1/5'; echo 'boom' >&2; exit 3")
        res = self.run_driver("8", needs='NEEDS=([8]="")')
        self.assertEqual(res[8]["result"], "FAIL", self.output)
        self.assertEqual(res[8]["rc"], 3, self.output)
        log = (self.out / "debug" / "08-train-steps.log").read_text()
        self.assertIn("boom", log)
        self.assertIn("train_ft.py exited 3", log)

    def _accelerate_oom_body(self):
        """A stage 8 trainer as the worker ran it in dbg3: train_ft's OOM line (behind a tqdm
        bar), oom.json in the run dir, then accelerate's ~20-line wrapper traceback whose last
        line is longer than 300 characters, and exit 1."""
        long_cmd = "'" + "', '".join(["/root/venv/bin/python", "train_ft.py"] + ["--arg"] * 40) + "'"
        return (f"mkdir -p '{self.probe_run}'; "
                f"echo '{{\"step\": 0, \"batch_tokens\": 8192}}' >'{self.probe_run}/oom.json'; "
                "echo 'Starting Training Loop...'; "
                "printf 'Training:   0%%|  | 0/5 [00:00<?, ?it/s]train_ft: GPU out of memory at "
                "micro-step 0, batch_tokens 8192; wrote oom.json; exit 76\\n' >&2; "
                "echo 'RuntimeError: Training:   0%|  | 0/5train_ft: GPU out of memory at micro-step 0, "
                "batch_tokens 8192; wrote oom.json; exit 76' >&2; "
                "echo 'The above exception was the direct cause of the following exception:' >&2; "
                "echo 'Traceback (most recent call last):' >&2; "
                "for i in $(seq 1 14); do echo \"  File \\\"launch.py\\\", line $i, in f\" >&2; done; "
                f"echo \"subprocess.CalledProcessError: Command '[{long_cmd}]' returned non-zero "
                "exit status 76.\" >&2; exit 1")

    def test_r23_fail_excerpt_shows_key_lines_before_the_tail(self):
        # base: only the last 12 lines, cut at 300 characters: neither the train_ft line nor
        # "exit status 76" made the excerpt
        self.fake_venv(self._accelerate_oom_body())
        res = self.run_driver("8", needs='NEEDS=([8]="")', VI_DEBUG_EXCERPT_LINES="12")
        self.assertEqual(res[8]["result"], "FAIL", self.output)
        self.assertIn("debug-pass:   | key: train_ft: GPU out of memory at micro-step 0", self.output)
        self.assertIn("debug-pass:   | key: accelerate: child returned non-zero exit status 76",
                      self.output)
        # the child stderr accelerate shows twice is one key line, not two
        self.assertEqual(self.output.count("| key: train_ft: GPU out of memory"), 1, self.output)
        key = self.output.index("| key: ")
        self.assertLess(key, self.output.index("| last 12 lines:"), self.output)
        pr = tp.load_paid_run()
        ex = pr.parse_debug_output(self.output)[8][2]
        kept = pr.debug_excerpt_keep(ex)
        self.assertIn("key: accelerate: child returned non-zero exit status 76", kept)
        self.assertTrue(any(x.startswith("key: train_ft: GPU out of memory") for x in kept), kept)

    def test_r23_stage8_oom_brings_oom_json_and_train_log_home_and_reads_76(self):
        # base: oom.json and train.log stayed in $scratch/probe-run (beside VI_OUT), so they
        # died with the instance; the stage read exit 1
        self.fake_venv(self._accelerate_oom_body())
        res = self.run_driver("8", needs='NEEDS=([8]="")')
        self.assertEqual(res[8]["result"], "FAIL", self.output)
        dbg = self.out / "debug"
        self.assertEqual(json.loads((dbg / "08-oom.json").read_text())["batch_tokens"], 8192)
        self.assertIn("train_ft: GPU out of memory", (dbg / "08-train.log").read_text())
        self.assertEqual(res[8]["rc"], 76, self.output)
        self.assertIn("GPU out of memory: oom.json written", (dbg / "08-train-steps.log").read_text())

    def test_r23_stage8_plain_failure_copies_train_log_and_keeps_its_rc(self):
        self.fake_venv("echo 'train_ft: step 1/5'; echo 'boom' >&2; exit 3")
        res = self.run_driver("8", needs='NEEDS=([8]="")')
        self.assertEqual(res[8]["rc"], 3, self.output)
        self.assertTrue((self.out / "debug" / "08-train.log").exists())
        self.assertFalse((self.out / "debug" / "08-oom.json").exists())

    def _r25_trainer_body(self, args_log):
        """A stage 8 trainer that records its arguments, writes checkpoint-5 and prints the r25
        diagnostic lines and the VRAM gate line the way train_ft.py does."""
        ck = self.probe_run / "checkpoint-5"
        return (f"echo \"$*\" >'{args_log}'; mkdir -p '{ck}'; touch '{ck}/SHA256SUMS'; "
                "echo 'train_ft: sdpa-mask-warning: none (no attention-kernel warning in the first forward)'; "
                "echo 'train_ft: vram peak: allocated 15.10 GiB reserved 16.00 GiB (after the first optimizer step)'; "
                "echo 'train_ft: vram peak: allocated 15.20 GiB reserved 16.00 GiB (end of training)'; "
                "echo 'train_ft: vram peak 15.20 of 23.60 GiB'")

    def test_r25_stage8_shows_the_peak_and_sdpa_lines_after_the_trainer(self):
        # base: stage 8 grepped only the kl lines; the r25 lines were lost in the trainer's tee
        args_log = self.tmp / "args"
        self.fake_venv(self._r25_trainer_body(args_log))
        res = self.run_driver("8", needs='NEEDS=([8]="")')
        self.assertEqual(res[8]["result"], "PASS", self.output)
        log = (self.out / "debug" / "08-train-steps.log").read_text()
        gate = log.index("preflight: vram peak 15.20 of 23.60 GiB (limit 90 %)")
        for line in ("train_ft: sdpa-mask-warning: none", "(after the first optimizer step)",
                     "(end of training)"):
            self.assertEqual(log.count(line), 2, log)  # the trainer's own, and the stage's summary
            self.assertLess(log.rindex(line), gate, log)

    def test_r26_stage8_shows_every_kl_rows_line(self):
        # base: the excerpt grep had no kl-rows pattern, so only the generic kl grep's last 3
        # lines survived and the first optimizer steps' replay share was lost
        args_log = self.tmp / "args"
        kl = "".join(f"echo 'train_ft: kl-rows: used {i} of 16 rows (share {i / 16:.3f})'; "
                     for i in range(1, 6))
        self.fake_venv(kl + self._r25_trainer_body(args_log))
        res = self.run_driver("8", needs='NEEDS=([8]="")')
        self.assertEqual(res[8]["result"], "PASS", self.output)
        log = (self.out / "debug" / "08-train-steps.log").read_text()
        gate = log.index("preflight: vram peak 15.20 of 23.60 GiB (limit 90 %)")
        for i in range(1, 6):
            line = f"train_ft: kl-rows: used {i} of 16 rows"
            self.assertGreaterEqual(log.count(line), 2, log)  # the trainer's own, and the summary
            self.assertLess(log.rindex(line), gate, log)
        for line in ("(after the first optimizer step)", "(end of training)",
                     "train_ft: sdpa-mask-warning: none"):
            self.assertEqual(log.count(line), 2, log)

    def test_r25_stage8_uses_the_same_batch_as_the_real_run(self):
        # Stage 8, the preflight probe and the real run pass no batch flags, so all three run
        # train_ft.py's defaults, which come from ft_guards.py alone (base: literals 8192 and 1
        # in train_ft.py's config and its CLI).
        args_log = self.tmp / "args"
        self.fake_venv(self._r25_trainer_body(args_log))
        self.run_driver("8", needs='NEEDS=([8]="")')
        stage8_args = args_log.read_text()
        self.assertIn("train_ft.py", stage8_args)
        vt = (JOBS / "voicetune-fresh.sh").read_text()
        real = vt[vt.index('"$jobs_dir/worker-job.sh" --work'):]
        real = real[:real.index("8>&- &")]
        self.assertIn("train_ft.py", real)
        for text in (stage8_args, real):
            self.assertNotIn("--batch-tokens", text)
            self.assertNotIn("--gradient-accumulation-steps", text)
        import ast
        tree = ast.parse((JOBS / "omnivoice-ft" / "train_ft.py").read_text())
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") in (
                    "batch_tokens", "gradient_accumulation_steps"):
                defaults["cfg." + node.target.id] = ast.unparse(node.value)
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                    and node.args and getattr(node.args[0], "value", "") in (
                        "--batch-tokens", "--gradient-accumulation-steps")):
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                defaults[node.args[0].value] = kw.get("default")
        self.assertEqual(defaults, {
            "cfg.batch_tokens": "FG.DEFAULT_BATCH_TOKENS",
            "cfg.gradient_accumulation_steps": "FG.DEFAULT_GRAD_ACCUM",
            "--batch-tokens": "FG.DEFAULT_BATCH_TOKENS",
            "--gradient-accumulation-steps": "FG.DEFAULT_GRAD_ACCUM"})

    def test_f2_no_time_left_is_skip_no_time_not_run(self):
        # 0.2 h left minus a 700 s margin leaves 20 s, under stage 3's minimum
        res = self.run_driver("3", VI_HOURS_LEFT="0.2", VI_DEBUG_MARGIN_SECONDS="700")
        self.assertEqual(res[3]["result"], "SKIP-no-time", self.output)
        self.assertRegex(self.output, r"debug-pass: stage 3 upload: SKIP-no-time \(\d+ s left")
        self.assertNotIn("stage 3 upload: running", self.output)

    def test_f2_limit_is_cut_to_the_time_left(self):
        # 0.2 h = 720 s left, 600 s margin: a 1000 s limit becomes about 120 s
        self.run_driver("3", VI_HOURS_LEFT="0.2", VI_DEBUG_MARGIN_SECONDS="600",
                        VI_DEBUG_TIMEOUT_3="1000")
        m = __import__("re").search(r"stage 3 upload: running \(limit (\d+) s", self.output)
        self.assertTrue(m, self.output)
        self.assertTrue(100 <= int(m.group(1)) <= 120, self.output)

    def test_f2_without_hours_left_the_limits_are_unchanged(self):
        self.run_driver("3")
        self.assertIn("stage 3 upload: running (limit 60 s", self.output)

    def test_f2_stage9_limit_comes_from_the_measured_push_rate(self):
        # checkpoint-5 of 2 GB (sparse) at the preflight's 1.0 MB/s. The parallel speedup through
        # the one tunnel is unmeasured, so it defaults to 1.0: 1.5 x 2e9 / 1e6 = 3000 s.
        ck = self.probe_run / "checkpoint-5"
        ck.mkdir(parents=True)
        with open(ck / "model.safetensors", "wb") as f:
            f.truncate(2 * 10**9)
        (self.out / "preflight.json").write_text(json.dumps({"push_mb_s": "1.0"}))
        self.run_driver("9", needs='NEEDS=([9]="")')
        self.assertIn("stage 9 ckpt-push: running (limit 3000 s", self.output)
        self.assertIn("1.0 MB/s", self.output)
        self.assertIn("x 1.0 parallel-stream speedup", self.output)
        # a measured speedup of 2.0 (set after the next debug pass) halves it
        self.run_driver("9", needs='NEEDS=([9]="")', VI_PUSH_SPEEDUP="2.0")
        self.assertIn("stage 9 ckpt-push: running (limit 1500 s", self.output)

    def test_stage9_holds_the_push_lock_and_pid_while_it_pushes(self):
        # W16b part 1 (task 15 finding 1): on-worker.sh's poller starts a publish push loop when
        # VI-PUSH reads 0 (push.pid missing or dead) and push.lock is free. While stage 9 pushes,
        # both must say "a push is running"; both are cleared when it is over.
        self.fake_venv("exit 0")
        ck = self.probe_run / "checkpoint-5"
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(b"x" * 1000)
        (ck / "SHA256SUMS").write_text("")
        seen = self.tmp / "seen"
        stub = self.jobs / "worker-return.sh"
        stub.write_text(f"""#!/bin/bash
# stands in for the real push: records what the poller would see while it runs
pid=$(cat {self.work}/push.pid 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "pid-alive" >> {seen}; fi
if ! flock -n {self.work}/push.lock true 2>/dev/null; then echo "lock-held" >> {seen}; fi
cat {self.work}/push.total >> {seen}; echo >> {seen}
exit 0
""")
        stub.chmod(0o755)
        res = self.run_driver("9", needs='NEEDS=([9]="")', VI_WEIGHTS_FIRST="0")
        self.assertEqual(res[9]["result"], "PASS", self.output)
        text = seen.read_text()
        self.assertIn("pid-alive", text)
        self.assertIn("lock-held", text)
        self.assertRegex(text, r"\n?\d{4,}\n")  # push.total: the bytes pushed
        self.assertFalse((self.work / "push.pid").exists(), "push.pid left behind")
        # the poller's two tests now say "nothing running" again
        r = subprocess.run(["flock", "-n", str(self.work / "push.lock"), "true"])
        self.assertEqual(r.returncode, 0)

    def test_f2_stage9_limit_has_a_600_s_floor_and_falls_back_without_a_rate(self):
        ck = self.probe_run / "checkpoint-5"
        ck.mkdir(parents=True)
        (ck / "x").write_bytes(b"x" * 1000)
        (self.out / "preflight.json").write_text(json.dumps({"push_mb_s": "9.0"}))
        self.run_driver("9", needs='NEEDS=([9]="")')
        self.assertIn("stage 9 ckpt-push: running (limit 600 s", self.output)
        (self.out / "preflight.json").unlink()
        self.run_driver("9", needs='NEEDS=([9]="")')
        self.assertIn("stage 9 ckpt-push: running (limit 600 s", self.output)


class Nofile(unittest.TestCase):
    """Round 19 A4: stage 2 judges the open-file limit after raising soft to hard (at most 65536)."""

    def probe(self, soft, hard):
        # Lower the limits in a subshell (an unprivileged process may lower its hard limit), then
        # run the real probe in it, the way a rented box with those limits would.
        cmd = f"ulimit -Sn {soft} && ulimit -Hn {hard} && exec bash {JOBS / 'debug-probe.sh'}"
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(l for l in r.stdout.splitlines() if l.startswith("debug-pass: stage 2 "))
        return line, r.stdout

    def test_low_soft_with_a_high_hard_limit_passes_after_the_raise(self):
        line, out = self.probe(1024, 8192)
        self.assertNotIn("open-file limit", line)
        self.assertIn("soft 1024, hard 8192; raised soft 8192", out)

    def test_a_hard_limit_under_4096_is_a_clear_failure(self):
        line, out = self.probe(1024, 2048)
        self.assertIn("FAIL", line)
        self.assertIn("open-file limit: hard limit 2048 (soft raised to 2048) is under 4096", line)

    def test_the_raise_is_capped_at_65536(self):
        hard = int(subprocess.run(["bash", "-c", "ulimit -Hn"], capture_output=True,
                                  text=True).stdout.strip() or 0)
        if hard <= 65536:
            self.skipTest(f"this host's hard nofile limit is {hard}, not above 65536")
        line, out = self.probe(1024, hard)
        self.assertIn(f"hard {hard}; raised soft 65536", out)

    def test_worker_limits_raises_and_logs_soft_and_hard(self):
        cmd = (f"ulimit -Sn 1024 && ulimit -Hn 8192 && . {JOBS / 'worker-limits.sh'} && "
               "vi_raise_nofile && echo \"now $(ulimit -Sn) $vi_lim_nofile_soft $vi_lim_nofile_hard\" && "
               "worker_limits_measure && worker_limits_line")
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("now 8192 8192 8192", r.stdout)
        self.assertRegex(r.stdout, r"nofile-soft=8192 nofile-hard=8192 tasks_free=\S+ -> threads=")

    def test_job_scripts_raise_nofile_before_work(self):
        fresh = (JOBS / "voicetune-fresh.sh").read_text()
        self.assertLess(fresh.index("vi_raise_nofile"), fresh.index("worker_limits || exit 1"))
        driver = (JOBS / "debug-pass.sh").read_text()
        self.assertLess(driver.index("vi_raise_nofile\n"), driver.index('for n in "${order[@]}"'))
        on_worker = (JOBS / "on-worker.sh").read_text()
        self.assertLess(on_worker.index("ulimit -Sn"), on_worker.index("jobs_dir="))


if __name__ == "__main__":
    unittest.main()
