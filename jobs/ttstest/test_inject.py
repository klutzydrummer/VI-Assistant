"""Injection test for the TTS worker job: jobs/on-worker.sh + worker_run.py with the "worker" in a
local folder (VI_WORKER_HOST=local, VI_WORKER_ROOT=<tmp>) and STUB models (sine WAVs). No network, no
GPU, no model download, no real audio. Home is a temp folder too (HOME, VI_STATE_DIR, VI_OUT).

    nice -n 19 python3 -I jobs/ttstest/test_inject.py -v

Cases: a baseline run, then (a) the job is killed halfway, (b) the return port is occupied so pushes
fail, (c) a WAV byte is flipped after the manifest, (d) one model raises MemoryError, (e) one model's
output is deleted. Each must end with a clear nonzero exit, a kept worker copy, the private clip gone
from the worker, and a home-side verify.py result that names the right culprit. Needs rsync, flock,
setsid and setpriv (the same as jobs/on-worker.sh). Takes about five minutes (case b waits out the home
supervisor's two-minute lost-contact limit). Not part of `unittest discover -p test_ttstest.py`.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify  # noqa: E402
import wavio  # noqa: E402

JOBS = HERE.parent
MODELS = ("omnivoice", "moss15", "qwen3base")
FOLDERS = [f"{m}__v{v}__s1" for m in MODELS for v in "PT"]
EXPECT = ",".join(f"{m}:{v}:1" for m in MODELS for v in "PT")
RUN_LIMIT = 240  # seconds for one whole on-worker.sh run


def wait_for(what, seconds, cond):
    end = time.time() + seconds
    while time.time() < end:
        got = cond()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def refuse_all(sock):
    """A squatter on the port: takes each connection and drops it at once (rsync then fails, as it should)."""
    while True:
        try:
            c, _ = sock.accept()
            c.close()
        except OSError:
            return


class Harness:
    """One isolated home + worker, and a way to run the stage through on-worker.sh."""

    def __init__(self, extra_env):
        self.tmp = Path(tempfile.mkdtemp(prefix="ttstest-inject-"))
        t = self.tmp
        for d in ("home", "state", "wroot", "out", "stages"):
            (t / d).mkdir()
        refs = t / "refs"
        refs.mkdir()
        wavio.write_wav(refs / "vT.wav", wavio.sine(6.5, 24000, 150), 24000)
        wavio.write_wav(refs / "vP.wav", wavio.sine(6.5, 24000, 190), 24000)
        (refs / "refs.json").write_text(json.dumps({"vT": {"id": "t", "text": "a b c."}, "vP": {"id": "p", "text": "d e f."}}))
        (t / "settings.json").write_text(json.dumps({"expect_epochs": 0}))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.env = dict(os.environ, HOME=str(t / "home"), VI_STATE_DIR=str(t / "state"), VI_WORKER_HOST="local",
                        VI_WORKER_ROOT=str(t / "wroot"), VI_STAGE="ttstest", VI_OUT=str(t / "out"),
                        VI_SETTINGS=str(t / "settings.json"), VI_CHECKPOINT=str(t / "stages" / ".checkpoint-ttstest"),
                        VI_INPUT_REFS=str(refs), VI_RETURN_PORT=str(self.port), VI_POLL_SECONDS="1",
                        VI_LOST_MINUTES="1", VI_PUSH_WAIT="3", TTSTEST_STUB="1", TTSTEST_KEEP=str(t / "keep"),
                        TTSTEST_CACHE=str(t / "cache"))
        self.env.update(extra_env)
        self.proc = None

    out = property(lambda self: self.tmp / "out")
    keep = property(lambda self: self.tmp / "keep")

    def start(self):
        self.log = open(self.tmp / "on-worker.log", "wb")
        self.proc = subprocess.Popen(["bash", str(JOBS / "on-worker.sh"), "python3", "jobs/ttstest/worker_run.py"],
                                     env=self.env, stdout=self.log, stderr=subprocess.STDOUT, cwd=self.tmp,
                                     start_new_session=True)
        return self.proc

    def finish(self):
        try:
            rc = self.proc.wait(timeout=RUN_LIMIT)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            raise AssertionError("on-worker.sh did not finish in time:\n" + self.text()[-2000:])
        return rc

    def text(self):
        self.log.flush()
        return (self.tmp / "on-worker.log").read_text(errors="replace")

    def work(self):
        """The worker's kept job folder."""
        found = sorted((self.tmp / "wroot").glob("ttstest.ttstest.*"))
        return found[-1] if found else None

    def verify(self):
        r = subprocess.run([sys.executable, "-I", str(HERE / "verify.py"), str(self.out), "--expect", EXPECT],
                           capture_output=True, text=True)
        return r.returncode, r.stdout

    def stop_all(self):
        """Tear down: anything still running from this temp home dies, then the folder goes."""
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-f", str(self.tmp)], capture_output=True)
        self.log.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class Injection(unittest.TestCase):
    def harness(self, **env):
        h = Harness(env)
        self.addCleanup(h.stop_all)
        return h

    def assert_failed_but_kept(self, h, rc):
        self.assertNotEqual(rc, 0, h.text()[-1500:])
        w = h.work()
        self.assertIsNotNone(w, "the worker copy was not kept:\n" + h.text()[-1500:])
        self.assertTrue((w / "out").is_dir())
        self.assertTrue(h.keep.is_dir(), "no second copy in the keep dir")
        self.assertIn("keeping", h.text())

    def assert_vt_gone(self, h):
        w = h.work()
        left = [p for p in (w.rglob("vT.wav") if w else [])] + list(h.keep.rglob("vT.wav"))
        self.assertEqual(left, [], "the private clip is still on the worker")

    def test_0_baseline_all_good(self):
        h = self.harness()
        h.start()
        rc = h.finish()
        self.assertEqual(rc, 0, h.text()[-2500:])
        code, out = h.verify()
        self.assertEqual((code, out.strip()), (0, "VERIFY OK"), out)
        self.assertEqual(len(list(h.out.glob("*/*.wav"))), 78)
        self.assertEqual((h.out / "expected-count.txt").read_text().strip(), "78")
        top = (h.out / "manifest.sha256").read_text().splitlines()
        self.assertEqual(len(top), 78)
        self.assertEqual(len(list(h.keep.glob("*/*.wav"))), 78)
        self.assertIsNone(h.work(), "a verified run should delete the worker folder")
        info = json.loads((h.out / "run-info.json").read_text())
        self.assertEqual((info["dropped"], info["exit"]), ([], 0))
        self.assertFalse(list(h.keep.rglob("vT.wav")))

    def test_a_job_killed_halfway(self):
        h = self.harness(TTSTEST_STUB_DELAY="0.08")
        h.start()
        first = h.out / "omnivoice__vP__s1" / "manifest.sha256"
        wait_for("the first folder to reach home", 120, first.exists)
        pid = int(wait_for("stage.pid", 5, lambda: (h.work() / "stage.pid").read_text().strip() if h.work() and (h.work() / "stage.pid").exists() else ""))
        os.killpg(pid, signal.SIGKILL)  # the stage is its own session (setsid -f), as when sshd drops a session
        rc = h.finish()
        self.assert_failed_but_kept(h, rc)
        code, out = h.verify()
        self.assertEqual(code, 1)
        for f in ("qwen3base__vP__s1", "qwen3base__vT__s1", "moss15__vT__s1"):
            self.assertRegex(out, f + r"/\(folder\): expected but not found")
        self.assertNotIn("omnivoice__vP__s1/", out, "the folder that arrived whole must not be blamed")
        wait_for("the dead-man switch to delete vT", 10, lambda: not list(h.work().rglob("vT.wav")))
        self.assert_vt_gone(h)

    def test_b_return_port_occupied(self):
        h = self.harness(TTSTEST_STUB_DELAY="0.08", VI_LOST_MINUTES="2")
        h.start()
        wait_for("the first folder to reach home", 120, (h.out / "omnivoice__vP__s1" / "manifest.sha256").exists)
        # Take the receiver down and sit on its port, so no push (and no restart) can get through.
        blocker = socket.socket()
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            end = time.time() + 30
            while True:
                pids = subprocess.run(["pgrep", "-f", f"rsync --daemon --no-detach --config={h.tmp}/state"],
                                      capture_output=True, text=True).stdout.split()
                for p in pids:
                    try:
                        os.kill(int(p), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    blocker.bind(("127.0.0.1", h.port))
                    blocker.listen(8)
                    threading.Thread(target=refuse_all, args=(blocker,), daemon=True).start()
                    break
                except OSError:
                    self.assertLess(time.time(), end, "could not take over the return port")
                    time.sleep(0.005)
            rc = h.finish()
        finally:
            blocker.close()
        self.assert_failed_but_kept(h, rc)
        self.assertIn("return receiver", h.text())
        code, out = h.verify()
        self.assertEqual(code, 1)
        self.assertRegex(out, r"qwen3base__vT__s1/\(folder\): expected but not found")
        info = json.loads((h.work() / "out" / "run-info.json").read_text())
        self.assertEqual(info["exit"], 4)
        self.assertIn("qwen3base__vT__s1", info["failed_push"])
        self.assertEqual(len(list(h.keep.glob("*/*.wav"))), 78, "the keep dir must hold every clip")
        self.assert_vt_gone(h)

    def test_c_byte_flipped_after_manifest(self):
        h = self.harness(TTSTEST_STUB_CORRUPT="moss15")
        h.start()
        rc = h.finish()
        self.assert_failed_but_kept(h, rc)
        code, out = h.verify()
        self.assertEqual(code, 1)
        for v in "PT":
            self.assertIn(f"moss15__v{v}__s1/moss15__v{v}__F01__s1.wav: SHA-256 does not match manifest", out)
        self.assertNotIn("omnivoice", out)
        self.assertNotIn("qwen3base", out)
        self.assertIn("SHA-256", h.text())  # the job itself reported the culprit on the worker
        self.assert_vt_gone(h)

    def test_d_memory_error_in_one_model(self):
        h = self.harness(TTSTEST_STUB_FAIL="moss15")
        h.start()
        rc = h.finish()
        self.assert_failed_but_kept(h, rc)
        info = json.loads((h.out / "run-info.json").read_text())
        self.assertEqual(info["dropped"], ["moss15"])
        self.assertIn("MemoryError", info["models"]["moss15"]["reason"])
        self.assertEqual(len(info["models"]["moss15"]["smoke"]), 2, "two smoke attempts before dropping")
        for m in ("omnivoice", "qwen3base"):
            self.assertEqual(info["models"][m]["status"], "generated")
        code, out = h.verify()
        self.assertEqual(code, 1)
        for f in ("moss15__vP__s1", "moss15__vT__s1"):
            self.assertIn(f"{f}/(folder): expected but not found", out)
        self.assertNotIn("omnivoice", out)
        self.assertNotIn("qwen3base", out)
        self.assertEqual(len(list(h.out.glob("*/*.wav"))), 52, "the other models must still be complete")
        self.assert_vt_gone(h)

    def test_e_one_models_output_deleted(self):
        h = self.harness(TTSTEST_STUB_DELETE="qwen3base")
        h.start()
        rc = h.finish()
        self.assert_failed_but_kept(h, rc)
        code, out = h.verify()
        self.assertEqual(code, 1)
        for f in ("qwen3base__vP__s1", "qwen3base__vT__s1"):
            self.assertIn(f"{f}/(folder): expected but not found", out)
        self.assertNotIn("omnivoice", out)
        self.assertNotIn("moss15", out)
        self.assertIn("no output folder was left", h.text())  # named by the job on the worker too
        self.assert_vt_gone(h)

    def test_f_repush_recovers_a_failed_push(self):
        """The README's re-push recipe: after case (b)'s loss, an open return path + --repush fixes home."""
        h = self.harness()
        h.start()
        self.assertEqual(h.finish(), 0, h.text()[-1500:])
        shutil.rmtree(h.out / "moss15__vT__s1")  # home lost a folder; the keep dir still has it
        code, out = h.verify()
        self.assertEqual(code, 1)
        self.assertIn("moss15__vT__s1/(folder): expected but not found", out)
        state = h.tmp / "state-repush"
        port = subprocess.run(["bash", str(JOBS / "worker-return.sh"), "start", str(h.out), str(h.tmp / "cp"), str(state)],
                              capture_output=True, text=True, check=True).stdout.strip()
        try:
            r = subprocess.run([sys.executable, "-I", str(HERE / "worker_run.py"), "--repush", str(h.keep)],
                               env=dict(os.environ, VI_RETURN_PORT=port, VI_PUSH_WAIT="10"), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        finally:
            subprocess.run(["bash", str(JOBS / "worker-return.sh"), "stop", str(state)])
        code, out = h.verify()
        self.assertEqual((code, out.strip()), (0, "VERIFY OK"), out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
