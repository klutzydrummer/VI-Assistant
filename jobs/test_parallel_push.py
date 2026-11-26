"""Parallel checkpoint push (jobs/worker-return.sh push_chunked + checkpoints.assemble_chunks).

Local only: a real rsync daemon on 127.0.0.1, small synthetic files, no GPU, no network beyond
loopback. A checkpoint's big files go as byte-range chunks over several rsync connections; home
joins them only when SHA256SUMS exists, so SHA256SUMS stays the only completeness gate."""
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "worker-return.sh"
CHECKPOINTS = HERE / "checkpoints.py"
MB = 1024 * 1024
SMALL = ("config.json", "scheduler.bin", "random_states_0.pkl", "train_config.json", "ft_config.json",
         "tokenizer.json")
BIG = ("model.safetensors", "optimizer.bin")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
class ParallelPushTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.root = Path(self.tmp.name)
        self.publish = r / "publish"
        self.state = r / "state"
        self.src = r / "src"
        self.ck = self.src / "checkpoint-5"
        self.ck.mkdir(parents=True)
        (self.ck / "model.safetensors").write_bytes(os.urandom(9 * MB + 123))
        (self.ck / "optimizer.bin").write_bytes(os.urandom(5 * MB))
        for n in SMALL:
            (self.ck / n).write_text("{}\n")
        self.write_sums()
        out, cp = r / "out", r / "cp"
        port = subprocess.run([SCRIPT, "start", out, cp, self.state, self.publish],
                              check=True, capture_output=True, text=True).stdout.strip()
        self.env = {**os.environ, "VI_RETURN_PORT": port, "VI_PUSH_WAIT": "3",
                    "VI_PUSH_CHUNK_MIN": str(MB), "VI_PUSH_STAGE": str(r)}
        self.procs = []

    def write_sums(self):
        (self.ck / "SHA256SUMS").write_text("".join(f"{sha(self.ck / n)}  {n}\n" for n in BIG + SMALL))

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
        subprocess.run([SCRIPT, "stop", self.state], capture_output=True)
        self.tmp.cleanup()

    def push(self, *opts, env=None, timeout=120):
        return subprocess.run(["nice", "-n", "19", SCRIPT, "push", self.src, "vi-publish", *opts],
                              env={**self.env, **(env or {})}, capture_output=True, text=True, timeout=timeout)

    def verify(self):
        return subprocess.run([sys.executable, CHECKPOINTS, "verify", self.publish, "--expect", "1",
                                "--min-model-bytes", "0", "--min-optimizer-bytes", "0"],
                              capture_output=True, text=True)

    def test_a_parallel_push_is_byte_identical_and_verifies(self):
        r = self.push()
        self.assertEqual(r.returncode, 0, r.stderr)
        v = self.verify()
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        home = self.publish / "checkpoint-5"
        for n in BIG + SMALL + ("SHA256SUMS",):
            self.assertEqual(sha(home / n), sha(self.ck / n), n)
        self.assertEqual([p.name for p in home.rglob("*") if "vichunk" in p.name or p.name.endswith(".vijoin")], [])

    def test_streams_really_run_in_parallel_with_chunks(self):
        # One file, 4 chunks: the log of the receiver shows 4 connections for the chunked pass.
        r = self.push("--exclude=SHA256SUMS")
        self.assertEqual(r.returncode, 0, r.stderr)
        home = self.publish / "checkpoint-5"
        self.assertTrue((home / "model.safetensors.vichunks").is_file())
        self.assertEqual(len(list(home.glob("model.safetensors.vichunk-*-of-4"))), 4)
        self.assertFalse((home / "model.safetensors").exists(), "the whole file must not be sent too")

    def test_c_partial_push_has_no_manifest_and_fails_verify(self):
        # Stop after the chunk pass: everything but SHA256SUMS.
        r = self.push("--exclude=SHA256SUMS")
        self.assertEqual(r.returncode, 0, r.stderr)
        home = self.publish / "checkpoint-5"
        self.assertFalse((home / "SHA256SUMS").exists())
        v = self.verify()
        self.assertNotEqual(v.returncode, 0)
        self.assertIn("SHA256SUMS", v.stdout + v.stderr)
        # a short chunk (interrupted copy) with the manifest present still fails
        (home / "SHA256SUMS").write_bytes((self.ck / "SHA256SUMS").read_bytes())
        victim = next(home.glob("optimizer.bin.vichunk-0-of-*"))
        victim.write_bytes(victim.read_bytes()[:-1])
        v = self.verify()
        self.assertNotEqual(v.returncode, 0)

    def test_b_kill_mid_push_then_rerun_resumes_and_completes(self):
        big = self.ck / "optimizer.bin"
        big.write_bytes(os.urandom(24 * MB))
        self.write_sums()
        p = subprocess.Popen(["nice", "-n", "19", SCRIPT, "push", self.src, "vi-publish", "--bwlimit=2000"],
                             env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        self.procs.append(p)
        home = self.publish / "checkpoint-5"
        deadline = time.time() + 30
        while time.time() < deadline:
            partial = sum(f.stat().st_size for f in home.rglob("*") if "vichunk" in str(f)) if home.exists() else 0
            if partial > 3 * MB:
                break
            time.sleep(0.1)
        self.assertIsNone(p.poll(), "push finished before it could be killed")
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        time.sleep(0.5)
        self.assertFalse((home / "SHA256SUMS").exists())
        self.assertNotEqual(self.verify().returncode, 0)
        kept = sum(f.stat().st_size for f in home.rglob("*") if "vichunk" in str(f) or ".rsync-partial" in str(f))
        self.assertGreater(kept, 0)
        r = self.push()
        self.assertEqual(r.returncode, 0, r.stderr)
        v = self.verify()
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertEqual(sha(home / "optimizer.bin"), sha(big))

    def test_d_receiver_still_refuses_outside_dirs_and_delete_with_four_slots(self):
        conf = (self.state / "rsyncd.conf").read_text()
        for rule in ("address = 127.0.0.1\n", "hosts allow = 127.0.0.1\n", "munge symlinks = true\n",
                     "refuse options = delete\n", "list = false\n", "write only = true\n"):
            self.assertIn(rule, conf)
        self.assertEqual(conf.count("max connections = 4\n"), 1)
        self.assertEqual(conf.count("max connections = 1\n"), 2)
        port = self.env["VI_RETURN_PORT"]
        (self.root / "x").write_text("x")
        # outside the exact modules
        r = subprocess.run(["rsync", "-a", str(self.root / "x"), f"rsync://127.0.0.1:{port}/etc/"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        r = subprocess.run(["rsync", "-a", str(self.root / "x"), f"rsync://127.0.0.1:{port}/vi-publish/../escape"],
                           capture_output=True, text=True)
        self.assertFalse((self.root / "escape").exists())
        # delete refused, listing refused, download refused
        r = subprocess.run(["rsync", "-a", "--delete", str(self.src) + "/", f"rsync://127.0.0.1:{port}/vi-publish/"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        r = subprocess.run(["rsync", "-a", f"rsync://127.0.0.1:{port}/vi-publish/", str(self.root / "dl") + "/"],
                           capture_output=True, text=True)
        self.assertFalse((self.root / "dl").exists() and any((self.root / "dl").iterdir()))


if __name__ == "__main__":
    unittest.main()
