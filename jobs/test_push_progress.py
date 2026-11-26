"""W16b parts 2 and 3 helpers: the bytes-received-home line, and the one read-only Vast check
the early host-drop stop runs (stub vastai, no network)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))
import push_progress  # noqa: E402

CHECK = JOBS / "paidrun" / "vast_gone_check.py"


class ProgressLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_counts_finished_partial_and_chunk_files_of_a_partial_dir(self):
        ck = self.root / "checkpoint-5"
        (ck / ".rsync-partial").mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(b"m" * 2_000_000)
        (ck / ".rsync-partial" / "optimizer.bin").write_bytes(b"o" * 500_000)
        (ck / "optimizer.bin.vichunk-0-of-4").write_bytes(b"c" * 250_000)
        (ck / "link").symlink_to(ck / "model.safetensors")  # never counted twice
        self.assertEqual(push_progress.received_bytes(self.root), 2_750_000)
        self.assertEqual(push_progress.progress_line(self.root, 11_000_000),
                         "received home 2.8 MB of 11.0 MB (25.0%)")
        self.assertEqual(push_progress.progress_line(self.root), "received home 2.8 MB")
        self.assertEqual(push_progress.progress_line(self.root, 0), "received home 2.8 MB")

    def test_empty_dir_and_cli(self):
        self.assertEqual(push_progress.progress_line(self.root, 100), "received home 0.0 MB of 0.0 MB (0.0%)")
        (self.root / "f").write_bytes(b"x" * 3_000_000)
        r = subprocess.run([sys.executable, str(JOBS / "push_progress.py"), str(self.root), "6000000"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.stdout.strip(), "received home 3.0 MB of 6.0 MB (50.0%)")
        r = subprocess.run([sys.executable, str(JOBS / "push_progress.py")], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 2)


class GoneCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def vastai(self, show_rc, show_out, logs_rc=0, logs_out=""):
        calls = self.root / "calls"
        stub = self.root / "vastai"
        stub.write_text(f"""#!/bin/bash
echo "$*" >> {calls}
case "$1" in
  show) printf '%s' '{show_out}'; exit {show_rc} ;;
  logs) printf '%s' '{logs_out}'; exit {logs_rc} ;;
esac
exit 1
""")
        stub.chmod(0o755)
        return stub, calls

    def run_check(self, stub, ident="62756405"):
        env = {**os.environ, "VI_VASTAI_BIN": str(stub)}
        return subprocess.run([sys.executable, "-I", str(CHECK), ident], env=env,
                              capture_output=True, text=True, timeout=60)

    def test_gone_exits_0_and_only_reads(self):
        stub, calls = self.vastai(0, '{"instances": null}', 0, "No such container: C.62756405")
        r = self.run_check(stub)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("is gone", r.stdout)
        verbs = {line.split()[0] for line in calls.read_text().splitlines()}
        self.assertEqual(verbs, {"show", "logs"})  # read-only: never destroy, stop or create

    def test_still_running_exits_1(self):
        stub, _ = self.vastai(0, json.dumps({"instances": {"actual_status": "running"}}))
        self.assertEqual(self.run_check(stub).returncode, 1)

    def test_api_failure_is_not_gone(self):
        stub, _ = self.vastai(1, "boom")
        self.assertEqual(self.run_check(stub).returncode, 1)

    def test_bad_usage(self):
        self.assertEqual(self.run_check(self.vastai(0, "{}")[0], "x; rm").returncode, 2)


if __name__ == "__main__":
    unittest.main()
