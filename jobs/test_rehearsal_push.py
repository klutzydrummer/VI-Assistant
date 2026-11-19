"""Runs the two real-receiver push cases of jobs/rehearsal/run.sh as unit tests.

They need rsync and a loopback port, write about 100 MB each under a temp dir (honouring
TMPDIR) and take a few seconds. No GPU, no network beyond 127.0.0.1."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

RUN = Path(__file__).resolve().parent / "rehearsal" / "run.sh"


def _run(case, **extra):
    base = tempfile.mkdtemp(prefix="vi-rehearsal-test-")
    env = {**os.environ, "VI_REHEARSAL_DIR": base, **extra}
    try:
        r = subprocess.run(["nice", "-n", "19", "bash", str(RUN), case], env=env,
                           capture_output=True, text=True, timeout=300)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return r


@unittest.skipUnless(shutil.which("rsync"), "rsync not installed")
class RehearsalPushTest(unittest.TestCase):
    def test_rehearsal_ckpt_push_real(self):
        r = _run("ckpt-push-real")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS [ckpt-push-real] VERIFIED exists", r.stdout)
        self.assertIn("PASS [ckpt-push-real] talker.safetensors is byte-identical home", r.stdout)

    def test_rehearsal_receiver_restart(self):
        r = _run("receiver-restart")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS [receiver-restart] push_retry finished the push after the restart", r.stdout)
        self.assertIn("PASS [receiver-restart] talker.safetensors is byte-identical home", r.stdout)

    def test_rehearsal_real_size_case_refuses_low_disk(self):
        r = _run("ckpt-push-real", VI_REHEARSAL_FREE_BYTES="1000")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing", r.stderr)
        self.assertNotIn("PASS [ckpt-push-real] the push succeeded", r.stdout)


if __name__ == "__main__":
    unittest.main()
