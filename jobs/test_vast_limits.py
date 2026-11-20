"""Tests for jobs/rehearsal/vast-limits.sh that need no clips, no GPU and no systemd.

They cover argument parsing, the refuse paths (missing venv, low RAM, low disk) and the red/green
comparator against canned logs. The real rehearsal run is opt-in and is never started here.
"""
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "rehearsal" / "vast-limits.sh"

R7_RED_LOG = """\
INFO [extract_audio_tokens.py:427] GPU count: 0, Processes per GPU: 3, Total processes: 3
OpenBLAS blas_thread_init: pthread_create failed for thread 19 of 48: Resource temporarily unavailable
Traceback (most recent call last):
KeyboardInterrupt
concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated
"""
GREEN_LOG = """\
INFO [extract_audio_tokens.py:427] GPU count: 0, Processes per GPU: 2, Total processes: 2
INFO [extract_audio_tokens.py:607] Processing Complete - Successful: 8, Failed: 0, Filtered/Skipped: 0, Shards written: 8
"""


def run(*args, env=None, home=None):
    e = dict(os.environ)
    for k in ("FAKE_NPROC", "TASKS_MAX", "CPU_QUOTA", "N_CLIPS", "BUDGET", "VENV", "MANIFEST",
              "THREADS", "NJ", "VL_MEM_AVAIL_MB", "VL_DISK_FREE_GB"):
        e.pop(k, None)
    if home:
        e["HOME"] = home
    e.update(env or {})
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=60)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        # A fake venv and a manifest so the later gates can be reached; nothing here is real data.
        self.venv = self.dir / "venv"
        (self.venv / "bin").mkdir(parents=True)
        py = self.venv / "bin" / "python"
        py.write_text("#!/bin/sh\nexit 0\n")
        py.chmod(py.stat().st_mode | stat.S_IXUSR)
        self.manifest = self.dir / "manifest.jsonl"
        self.manifest.write_text("".join('{"id": "x%d"}\n' % i for i in range(8)))
        self.base = self.dir / "base"
        self.env = {"VI_REHEARSAL_DIR": str(self.base)}

    def judge(self, expect, log_text, rc, events=0, o_stage=0, o_post=0, extra=()):
        log = self.dir / "stage.log"
        log.write_text(log_text)
        return run("--judge", "--expect", expect, "--log", str(log), "--rc", str(rc),
                   "--events", str(events), "--orphans-stage", str(o_stage),
                   "--orphans-post", str(o_post), *extra)


class ArgumentParsing(Fixture):
    def test_help_exits_zero(self):
        r = run("--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--expect red|green", r.stdout)

    def test_unknown_argument(self):
        r = run("--bogus")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown argument", r.stderr)

    def test_bad_expect(self):
        r = run("--expect", "blue")
        self.assertEqual(r.returncode, 2)
        self.assertIn("red or green", r.stderr)

    def test_non_numeric_values(self):
        for flag in ("--fake-nproc", "--tasks-max", "--clips", "--budget", "--threads", "--nj"):
            with self.subTest(flag=flag):
                r = run(flag, "abc")
                self.assertEqual(r.returncode, 2)
                self.assertIn("positive whole number", r.stderr)

    def test_missing_value(self):
        r = run("--tasks-max")
        self.assertEqual(r.returncode, 2)
        self.assertIn("needs a value", r.stderr)

    def test_environment_values_are_checked_too(self):
        r = run(env={"BUDGET": "0"})
        self.assertEqual(r.returncode, 2)


class RefusePaths(Fixture):
    def go(self, **env):
        e = dict(self.env)
        e.update(env)
        return run("--venv", str(self.venv), "--manifest", str(self.manifest), env=e, home=str(self.dir))

    def test_missing_venv(self):
        r = run("--venv", str(self.dir / "nope"), "--manifest", str(self.manifest), env=self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("no CPU venv", r.stderr)

    def test_default_venv_missing_is_reported(self):
        r = run("--manifest", str(self.manifest), env=self.env, home=str(self.dir))
        self.assertEqual(r.returncode, 2)
        self.assertIn("no CPU venv", r.stderr)

    def test_missing_manifest(self):
        r = run("--venv", str(self.venv), "--manifest", str(self.dir / "nope.jsonl"), env=self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("manifest", r.stderr)

    def test_more_clips_than_rows(self):
        r = run("--venv", str(self.venv), "--manifest", str(self.manifest), "--clips", "9", env=self.env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("more than the 8 rows", r.stderr)

    def test_low_ram(self):
        r = self.go(VL_MEM_AVAIL_MB="3000")
        self.assertEqual(r.returncode, 2)
        self.assertIn("3000 MB RAM", r.stderr)

    def test_low_disk(self):
        r = self.go(VL_MEM_AVAIL_MB="9000", VL_DISK_FREE_GB="12")
        self.assertEqual(r.returncode, 2)
        self.assertIn("12 GB free", r.stderr)

    def test_refusals_start_nothing(self):
        self.go(VL_MEM_AVAIL_MB="3000")
        self.assertEqual(list(self.base.glob("vast-limits.*")), [])


class Comparator(Fixture):
    def test_red_passes_on_r7_signature(self):
        r = self.judge("red", R7_RED_LOG, rc=1, events=73)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("r7_signature=yes", r.stdout)

    def test_red_passes_on_budget_hang_alone(self):
        for rc in (124, 137):
            with self.subTest(rc=rc):
                r = self.judge("red", "nothing useful\n", rc=rc)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("budget_hang=yes", r.stdout)

    def test_red_fails_when_base_ran_clean(self):
        r = self.judge("red", GREEN_LOG, rc=0)
        self.assertEqual(r.returncode, 1)
        self.assertIn("did NOT fail the way r7 did", r.stderr)

    def test_red_needs_the_fake_cpu_count_in_the_message(self):
        log = R7_RED_LOG.replace("of 48", "of 4")
        r = self.judge("red", log, rc=1)
        self.assertEqual(r.returncode, 1)
        r = self.judge("red", log, rc=1, extra=("--fake-nproc", "4"))
        self.assertEqual(r.returncode, 0)

    def test_red_fails_on_orphans_after_teardown(self):
        r = self.judge("red", R7_RED_LOG, rc=124, o_post=2)
        self.assertEqual(r.returncode, 1)
        self.assertIn("still alive after teardown", r.stderr)

    def test_red_tolerates_orphans_before_teardown(self):
        r = self.judge("red", R7_RED_LOG, rc=124, o_stage=40)
        self.assertEqual(r.returncode, 0)

    def test_green_passes_when_everything_holds(self):
        r = self.judge("green", GREEN_LOG, rc=0)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("complete=8/8", r.stdout)

    def test_green_fails_on_each_broken_assertion(self):
        cases = {
            "exit code": dict(log=GREEN_LOG, rc=1),
            "pids events": dict(log=GREEN_LOG, rc=0, events=3),
            "fatal line": dict(log=GREEN_LOG + "Traceback (most recent call last):\n", rc=0),
            "orphans in scope": dict(log=GREEN_LOG, rc=0, o_stage=4),
            "orphans after teardown": dict(log=GREEN_LOG, rc=0, o_post=1),
            "wrong clip count": dict(log=GREEN_LOG.replace("Successful: 8", "Successful: 7"), rc=0),
            "no complete line": dict(log="nothing\n", rc=0),
            "two complete lines": dict(log=GREEN_LOG * 2, rc=0),
        }
        for name, c in cases.items():
            with self.subTest(case=name):
                r = self.judge("green", c["log"], c["rc"], c.get("events", 0), c.get("o_stage", 0),
                               c.get("o_post", 0))
                self.assertEqual(r.returncode, 1, r.stdout)

    def test_green_fails_on_the_r7_log(self):
        r = self.judge("green", R7_RED_LOG, rc=1, events=73)
        self.assertEqual(r.returncode, 1)

    def test_clip_count_is_configurable(self):
        log = GREEN_LOG.replace("Successful: 8", "Successful: 5")
        self.assertEqual(self.judge("green", log, 0, extra=("--clips", "5")).returncode, 0)


class NotPickedUpByDefault(unittest.TestCase):
    def test_not_in_run_sh_default_scenarios_or_ci(self):
        root = SCRIPT.parent.parent.parent
        run_sh = (root / "jobs" / "rehearsal" / "run.sh").read_text()
        all_line = next(line for line in run_sh.splitlines() if line.startswith("all=("))
        self.assertNotIn("vast-limits", all_line)
        for wf in (root / ".github" / "workflows").glob("*.yml"):
            self.assertNotIn("vast-limits", wf.read_text(), wf.name)


if __name__ == "__main__":
    unittest.main()
