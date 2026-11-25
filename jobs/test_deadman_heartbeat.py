"""r32 WP2: the paid run's heartbeat for the home dead-man timer (L2).

`vi cloud up` arms a systemd user timer at the rental's hard cap; while the run is alive and still
judging the rental (VALID or SUSPECT) PaidRun.poll_validity moves it to now + the stall grace
through `vi cloud deadman extend`. These tests use a stub `vi` that records its argv: no systemd,
no Vast, no network. The timer itself is tested in crates/cloud/tests/deadman.rs.
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))

from paidrun import limits  # noqa: E402
from paidrun import validity as V  # noqa: E402
from paidrun.orchestrator import PaidRun  # noqa: E402


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Args:
    pass


def stub_vi(folder, rc=0):
    """A `vi` that appends its argv (and VI_DEADMAN_LOG_DIR) to calls.txt and exits rc."""
    vi = folder / "vi"
    vi.write_text("#!/bin/sh\n"
                  f"echo \"$* log=$VI_DEADMAN_LOG_DIR\" >> '{folder}/calls.txt'\n"
                  "echo 'error: no dead-man timer is recorded' >&2\n"
                  f"exit {rc}\n")
    vi.chmod(0o755)
    return vi


class Heartbeat(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="deadman-hb."))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.clock = FakeClock()
        self.v = V.Validity(started=self.clock(), clock=self.clock, attempt=1,
                            wall_cap_s=10 * 3600)
        self.v.enter("preflight")
        self.at = dict(n=1, phase="preflight", instance="4242")
        run = PaidRun.__new__(PaidRun)
        run.a = Args()
        run.a.repo = str(self.tmp)
        run.vi = str(stub_vi(self.tmp))
        run.dir = self.tmp / "run"
        run.config_path = None
        run.in_teardown = False
        run.deadman_beat_at = None
        run.current_validity = lambda: self.v
        run.open_attempt = lambda: self.at
        run.record_validity = lambda v: None
        self.run_ = run
        self.mono = 5000.0
        patcher = mock.patch("paidrun.orchestrator.time.monotonic", lambda: self.mono)
        patcher.start()
        self.addCleanup(patcher.stop)

    def calls(self):
        f = self.tmp / "calls.txt"
        return f.read_text().splitlines() if f.exists() else []

    def poll(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.run_.poll_validity()
        return out.getvalue()

    def test_a_valid_rental_moves_the_timer_with_the_stall_grace(self):
        self.assertEqual(self.poll(), "")
        grace = limits.DEADMAN_STALL_GRACE_SECONDS
        self.assertEqual(self.calls(), [f"cloud deadman extend 4242 --grace {grace} "
                                        f"log={self.tmp / 'run'}"])
        self.assertEqual(grace, 30 * 60)

    def test_beats_are_throttled(self):
        self.poll()
        self.mono += limits.DEADMAN_HEARTBEAT_SECONDS - 1
        self.poll()
        self.assertEqual(len(self.calls()), 1)
        self.mono += 1
        self.poll()
        self.assertEqual(len(self.calls()), 2)

    def test_a_suspect_rental_still_beats(self):
        # Validity owns SUSPECT -> INVALID; the timer only covers a dead or frozen run.
        self.v.state = V.SUSPECT
        self.poll()
        self.assertEqual(len(self.calls()), 1)

    def test_no_beat_once_invalid_in_teardown_or_without_an_instance(self):
        self.run_.in_teardown = True
        self.poll()
        self.run_.in_teardown = False
        self.at["instance"] = None
        self.poll()
        self.at["instance"] = "12;rm -rf ~"
        self.poll()
        self.at["instance"] = "4242"
        self.v.state = V.INVALID
        self.v.reason = "our-failure"  # not a self reason, so poll_validity does not raise
        self.poll()
        self.run_.deadman_heartbeat(None)
        self.assertEqual(self.calls(), [])

    def test_a_failed_extend_is_an_alert_not_a_crash(self):
        self.run_.vi = str(stub_vi(self.tmp, rc=1))
        out = self.poll()
        self.assertIn("paid-run: ALERT: could not move the dead-man timer for instance 4242 "
                      "(exit 1)", out)
        self.assertIn("no dead-man timer is recorded", out)
        self.assertIn("stays armed", out)

    def test_a_missing_vi_is_an_alert_not_a_crash(self):
        self.run_.vi = str(self.tmp / "no-such-vi")
        out = self.poll()
        self.assertIn("ALERT: could not move the dead-man timer", out)

    def test_a_hung_extend_is_cut_off(self):
        hang = self.tmp / "hang"
        hang.write_text("#!/bin/sh\nexec sleep 30\n")
        hang.chmod(0o755)
        self.run_.vi = str(hang)
        with mock.patch.object(limits, "DEADMAN_EXTEND_TIMEOUT_SECONDS", 0.2), \
                mock.patch("paidrun.orchestrator.DEADMAN_EXTEND_TIMEOUT_SECONDS", 0.2):
            out = self.poll()
        self.assertIn("ALERT", out)
        self.assertIn("timed out", out)

    def test_env_points_the_fire_log_at_the_run_folder(self):
        with mock.patch.dict(os.environ, {"VI_DEADMAN_LOG_DIR": "/elsewhere"}):
            self.assertEqual(self.run_.env()["VI_DEADMAN_LOG_DIR"], str(self.tmp / "run"))


if __name__ == "__main__":
    unittest.main()
