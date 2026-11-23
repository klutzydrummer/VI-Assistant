"""Tests for `jobs/paid_run.py --debug-pass` with the stub `vi` from test_paid_run (no Vast, no
GPU, no network), plus the small helpers the debug pass ships (debug_slice.py, the shell scripts).

The stub gains two answers here: `vi cloud ssh <debug-probe text>` prints stage 1 and 2 lines,
and `vi dataset run ... omnivoice_debug --force` prints stage 3 to 11 lines the way
jobs/debug-pass.sh does. Scenario key `probe` picks the probe answer and `debug` the stage's.
"""

import hashlib
import json
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
        r = sc.get("debug", "pass")
        log("debug_env", os.environ.get("VI_DEBUG_PASS"), os.environ.get("VI_DEBUG_ROWS"))
        if r == "silent":
            print("on-worker.sh: input upload failed", file=sys.stderr); sys.exit(1)
        names = %r
        needs = %r
        fail = set(sc.get("debug_fail", []))
        res = {}
        for n in range(3, 12):
            if r == "hang" and n == 5:
                print("debug-pass: stage 5 base-model: still running (60 s)", flush=True); hang()
            bad = [m for m in needs.get(n, ()) if res.get(m) != "PASS"]
            if bad:
                res[n] = "SKIP"
                print(f"debug-pass: stage {n} {names[n]}: SKIP-dependent (needs stage {bad[0]} "
                      f"{names[bad[0]]}, which did not pass)", flush=True)
            elif n in fail:
                res[n] = "FAIL"
                print(f"debug-pass: stage {n} {names[n]}: FAIL (exit 1, 3 s)", flush=True)
                print(f"debug-pass:   | stage {n} broke: ModuleNotFoundError: no module x", flush=True)
            else:
                res[n] = "PASS"
                print(f"debug-pass: stage {n} {names[n]}: PASS (2 s)", flush=True)
        if res.get(9) == "PASS" and not sc.get("no_home_ckpt"):
            ck = Path(os.environ["VI_CHECKPOINT_ROOT"]) / "omnivoice_debug" / "run1" / "checkpoint-5"
            ck.mkdir(parents=True, exist_ok=True)
            sums = []
            for f in ("config.json", "model.safetensors", "optimizer.bin", "scheduler.bin",
                      "random_states_0.pkl", "train_config.json", "ft_config.json", "tokenizer.json"):
                (ck / f).write_bytes(b"x" + f.encode())
                import hashlib
                sums.append(hashlib.sha256((ck / f).read_bytes()).hexdigest() + "  " + f)
            (ck / "SHA256SUMS").write_text("\\n".join(sums) + "\\n")
        out = Path(sc["corpus"]) / "stages" / "omnivoice_debug" / ".tmp-k1" / "out" / "debug"
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
        p = sc.get("probe", "pass")
        log("probe", "DEBUG_MIN_VRAM_GB" in " ".join(args))
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
        self.assertIn("a debug pass rents once", self.out)
        self.assertIn("--plan: nothing was called", self.out)
        self.assertFalse((self.tmp / "any-call.log").exists(), "--plan called vi")
        self.assertEqual(self.calls(), [])

    def test_worst_case_formula(self):
        pr = tp.load_paid_run()
        total, cap, disk = pr.debug_worst_case(3.0, 125, 1.0)
        self.assertAlmostEqual(cap, 3.0)
        self.assertAlmostEqual(disk, 125 * 1.00 / 730 * 1.0)
        self.assertAlmostEqual(total, cap + disk)

    def test_plain_plan_still_exit_0(self):
        self.scen()
        self.assertEqual(self.run_orch(plan=True), 0, self.out)
        self.assertNotIn("a debug pass rents once", self.out)

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
        for s in ("debug-pass.sh", "debug-probe.sh", "on-worker.sh", "rehearsal/debug-pass.sh"):
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
        self.assertRegex(r.stdout, r"nofile-soft=8192 nofile-hard=8192 -> threads=")

    def test_job_scripts_raise_nofile_before_work(self):
        fresh = (JOBS / "voicetune-fresh.sh").read_text()
        self.assertLess(fresh.index("vi_raise_nofile"), fresh.index("worker_limits || exit 1"))
        driver = (JOBS / "debug-pass.sh").read_text()
        self.assertLess(driver.index("vi_raise_nofile\n"), driver.index('for n in "${order[@]}"'))
        on_worker = (JOBS / "on-worker.sh").read_text()
        self.assertLess(on_worker.index("ulimit -Sn"), on_worker.index("jobs_dir="))


if __name__ == "__main__":
    unittest.main()
