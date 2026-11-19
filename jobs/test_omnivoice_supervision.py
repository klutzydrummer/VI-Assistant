"""Tests for the OmniVoice supervision pieces (standard library + shell only, no venv):
jobs/omnivoice_abort.py (the voicetune-fresh.sh abort parser) and the
--resume-from-checkpoint wiring in jobs/worker-job.sh."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))
import omnivoice_abort


def _metrics(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _guard(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class AbortTest(unittest.TestCase):
    """The abort parser reads the pipeline's own logs; only the printed reason counts."""

    def test_nan_metric_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            _metrics(m, [{"micro_step": 50, "attempt": 0, "loss/ce": 3.1},
                         {"micro_step": 100, "attempt": 0, "loss/ce": float("nan")}])
            g.touch()
            reason = omnivoice_abort.abort_reason(m, g)
            self.assertIn("loss/ce", reason)
            self.assertIn("micro-step 100", reason)

    def test_infinite_metric_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            _metrics(m, [{"micro_step": 50, "attempt": 0, "style/z_spread": float("inf")}])
            g.touch()
            self.assertIn("z_spread", omnivoice_abort.abort_reason(m, g))

    def test_healthy_metrics_do_not_abort(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            _metrics(m, [{"micro_step": 50, "attempt": 0, "loss/ce": 3.1, "style/z_spread": 1.2}])
            g.touch()
            self.assertEqual(omnivoice_abort.abort_reason(m, g), "")

    def test_missing_files_do_not_abort(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(
                omnivoice_abort.abort_reason(root / "no-metrics.jsonl", root / "no-guard.jsonl"), "")

    def test_half_written_trailing_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            m.write_text('{"micro_step": 50, "loss/ce": 3.1}\n{"micro_step": 100, "loss/ce": ')
            g.touch()
            self.assertEqual(omnivoice_abort.abort_reason(m, g), "")

    def test_guard_fire_in_latest_attempt_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            m.touch()
            _guard(g, [
                {"attempt": 0, "seed": 0, "step": 300, "collapsed": False, "strikes": 0},
                {"attempt": 0, "seed": 0, "step": 400, "collapsed": False, "strikes": 1},
                {"attempt": 0, "seed": 0, "step": 500, "collapsed": True, "strikes": 2},
            ])
            reason = omnivoice_abort.abort_reason(m, g)
            self.assertIn("collapse guard fired", reason)
            self.assertIn("step 500", reason)

    def test_old_attempt_fire_does_not_kill_a_restarted_run(self):
        # The log keeps every attempt; a fire row from attempt 0 must not abort
        # attempt 1, which the pipeline restarted with a fresh seed.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            m.touch()
            _guard(g, [
                {"attempt": 0, "seed": 0, "step": 500, "collapsed": True, "strikes": 2},
                {"attempt": 1, "seed": 1, "step": 300, "collapsed": False, "strikes": 0},
                {"attempt": 1, "seed": 1, "step": 400, "collapsed": False, "strikes": 0},
            ])
            self.assertEqual(omnivoice_abort.abort_reason(m, g), "")

    def test_patience_reads_ft_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g, c = root / "style_metrics.jsonl", root / "collapse_guard.jsonl", root / "ft_config.json"
            m.touch()
            c.write_text('{"guard_patience": 3}')
            _guard(g, [{"attempt": 0, "seed": 0, "step": 500, "collapsed": True, "strikes": 2}])
            self.assertEqual(omnivoice_abort.abort_reason(m, g, c), "")
            _guard(g, [{"attempt": 0, "seed": 0, "step": 500, "collapsed": True, "strikes": 2},
                       {"attempt": 0, "seed": 0, "step": 600, "collapsed": True, "strikes": 3}])
            self.assertIn("collapse guard fired", omnivoice_abort.abort_reason(m, g, c))

    def test_guard_disabled_ignores_fire_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g, c = root / "style_metrics.jsonl", root / "collapse_guard.jsonl", root / "ft_config.json"
            m.touch()
            c.write_text('{"guard": false}')
            _guard(g, [{"attempt": 0, "seed": 0, "step": 500, "collapsed": True, "strikes": 2}])
            self.assertEqual(omnivoice_abort.abort_reason(m, g, c), "")

    def test_cli_prints_reason_exit_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            m, g = root / "style_metrics.jsonl", root / "collapse_guard.jsonl"
            _metrics(m, [{"micro_step": 50, "attempt": 0, "loss/ce": float("nan")}])
            g.touch()
            run = subprocess.run([sys.executable, "-I", str(JOBS / "omnivoice_abort.py"), str(m), str(g)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0)
            self.assertIn("loss/ce", run.stdout)


class ResumeWiringTest(unittest.TestCase):
    """worker-job.sh restarts a failed trainer with --resume-from-checkpoint from the
    newest checkpoint whose SHA256SUMS exists (the manifest is written last)."""

    def _checkpoint(self, run, step, with_manifest=True):
        ck = run / f"checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(b"x")
        if with_manifest:
            (ck / "SHA256SUMS").write_text(
                f"{'0' * 64}  model.safetensors\n")
        return ck

    def _run_worker_job(self, root, retries=1, publish=False):
        """Run worker-job.sh with a command that logs its args and always fails.
        Returns one arg list per invocation, in order."""
        work, run = root / "work", root / "run"
        run.mkdir(exist_ok=True)  # some tests drop checkpoints in there first
        # A fake cgroup memory.events: with no real one visible, worker-job.sh caps
        # retries at one regardless of --retries.
        events = root / "memory.events"
        events.write_text("oom_kill 0\n")
        log = root / "args.log"
        # Inherit the real environment (worker-job.sh needs flock/awk/setsid/bash
        # from the system PATH) and add the test's own variables.
        env = dict(os.environ, LOG=str(log), VI_TRAIN_RETRIES="1",
                   VI_MEMORY_EVENTS=str(events))
        script = ("import os, sys; "
                  "open(os.environ['LOG'], 'a').write('\\x1f'.join(sys.argv[1:]) + '\\n'); "
                  "sys.exit(1)")
        cmd = [sys.executable, "-c", script, "--steps", "5000"]
        args = ["bash", str(JOBS / "worker-job.sh"), "--work", str(work), "--run", str(run),
                "--retries", str(retries)]
        if publish:
            args += ["--publish", str(root / "publish")]
        args += ["--"] + cmd
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env)
        self.assertNotEqual(proc.returncode, 0)
        return [line.split("\x1f") for line in log.read_text().splitlines()]

    def test_retry_appends_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkpoint(root / "run", 1000)
            self._checkpoint(root / "run", 1500)
            invocations = self._run_worker_job(root)
            self.assertEqual(len(invocations), 2)  # one retry
            self.assertNotIn("--resume-from-checkpoint", invocations[0])
            self.assertIn("--resume-from-checkpoint", invocations[1])
            self.assertEqual(invocations[1][invocations[1].index("--resume-from-checkpoint") + 1],
                             str(root / "run" / "checkpoint-1500"))

    def test_no_checkpoints_no_resume_flag(self):
        with tempfile.TemporaryDirectory() as d:
            invocations = self._run_worker_job(Path(d))
            self.assertEqual(len(invocations), 2)
            self.assertNotIn("--resume-from-checkpoint", invocations[0])
            self.assertNotIn("--resume-from-checkpoint", invocations[1])

    def test_kill_during_save_manifest_missing_is_not_resumed(self):
        # The pipeline writes SHA256SUMS last; a dir without it is a partial save
        # and must not be handed to --resume-from-checkpoint.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkpoint(root / "run", 1000, with_manifest=False)
            invocations = self._run_worker_job(root)
            self.assertNotIn("--resume-from-checkpoint", invocations[1])

    def test_resume_flag_appears_once_across_retries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkpoint(root / "run", 1000)
            invocations = self._run_worker_job(root, retries=2)
            self.assertEqual(len(invocations), 3)
            self.assertNotIn("--resume-from-checkpoint", invocations[0])
            self.assertIn("--resume-from-checkpoint", invocations[1])
            self.assertIn("--resume-from-checkpoint", invocations[2])

    def _run_script(self, root, script, retries=2, extra=()):
        """Run worker-job.sh around a python -c script; returns (proc, arg lists per call)."""
        work, run = root / "work", root / "run"
        run.mkdir(exist_ok=True)
        events = root / "memory.events"
        events.write_text("oom_kill 0\n")
        log = root / "args.log"
        env = dict(os.environ, LOG=str(log), VI_MEMORY_EVENTS=str(events))
        args = ["bash", str(JOBS / "worker-job.sh"), "--work", str(work), "--run", str(run),
                "--retries", str(retries), "--", sys.executable, "-c", script, *extra]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120, env=env)
        calls = [l.split("\x1f") for l in log.read_text().splitlines()] if log.exists() else []
        return proc, calls

    def test_worker_job_oom_halves_batch(self):
        # Exit 76 once, then succeed: the retry halves the batch, doubles grad-accum and resumes.
        script = ("import os, sys; log = os.environ['LOG']; "
                  "first = not os.path.exists(log); "
                  "open(log, 'a').write('\\x1f'.join(sys.argv[1:]) + '\\n'); "
                  "sys.exit(76 if first else 0)")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._checkpoint(root / "run", 1000)
            proc, calls = self._run_script(
                root, script, extra=["--batch-tokens", "8192", "--gradient-accumulation-steps", "1"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], ["--batch-tokens", "8192", "--gradient-accumulation-steps", "1"])
            self.assertEqual(calls[1], ["--batch-tokens", "4096", "--gradient-accumulation-steps", "2",
                                        "--resume-from-checkpoint", str(root / "run" / "checkpoint-1000")])
            rows = [json.loads(l) for l in (root / "work" / "retries.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["reason"], "oom-halve")

    def test_worker_job_oom_uses_defaults_and_stops_at_the_floor(self):
        # No flags given: the trainer defaults (8192, 1) are made explicit; 8192 -> 4096 -> 2048,
        # and the next halving would go below the floor, so the job stops with 76.
        script = ("import os, sys; "
                  "open(os.environ['LOG'], 'a').write('\\x1f'.join(sys.argv[1:]) + '\\n'); sys.exit(76)")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc, calls = self._run_script(root, script, retries=5)
            self.assertEqual(proc.returncode, 76)
            self.assertNotIn("--batch-tokens", calls[0])
            self.assertEqual(len(calls), 3)
            self.assertEqual([c[c.index("--batch-tokens") + 1] for c in calls[1:]], ["4096", "2048"])
            self.assertEqual([c[c.index("--gradient-accumulation-steps") + 1] for c in calls[1:]], ["2", "4"])
            self.assertIn("below the 2048 floor", (root / "work" / "job.log").read_text())

    def test_worker_job_same_traceback_stops(self):
        # The same traceback line twice is deterministic: one retry, then stop (not a 3rd run).
        script = ("import os, sys; "
                  "open(os.environ['LOG'], 'a').write('x\\n'); "
                  "sys.stderr.write('Traceback (most recent call last):\\n  File \"/a/b.py\", line 7, in f\\n"
                  "    boom()\\nValueError: boom\\n'); sys.exit(1)")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc, calls = self._run_script(root, script, retries=2)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(len(calls), 2)
            self.assertIn("same traceback line repeated", (root / "work" / "job.log").read_text())
            rows = [json.loads(l) for l in (root / "work" / "retries.jsonl").read_text().splitlines()]
            self.assertEqual([r["reason"] for r in rows], ["crash", "same-traceback-stop"])
            self.assertIn('File "/a/b.py", line 7', rows[1]["signature"])


if __name__ == "__main__":
    unittest.main()
