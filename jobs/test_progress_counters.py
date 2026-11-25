"""r32 WP4: worker progress counters (`VI-PROGRESS <stage> <counter> <value>`).

Home side: parsing (malformed lines ignored), Validity's strictly-increasing rule and min rate,
PaidRun.note_line feeding the counters into the open rental's Validity, and a real fetch through
fetch_base_model.py from a local HTTP server throttled to 10 KB/s that must turn SUSPECT.
Worker side: on-worker.sh's stall rule (log-only activity is not progress, rising counters are)
and the final push refusing a dead receiver. Everything is local: loopback HTTP and rsync only,
no SSH, no Vast, no GPU.
"""

import http.server
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))

from paidrun import validity as V  # noqa: E402
from paidrun.orchestrator import PaidRun  # noqa: E402
from paidrun.signals import parse_progress, parse_progress_end  # noqa: E402

ON_WORKER = JOBS / "on-worker.sh"
FETCH = JOBS / "omnivoice-ft" / "fetch_base_model.py"


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def machine(clock, **kw):
    kw.setdefault("wall_cap_s", 10 * 3600)
    return V.Validity(started=clock(), clock=clock, attempt=1, **kw)


class Parsing(unittest.TestCase):
    def test_good_lines(self):
        self.assertEqual(parse_progress("VI-PROGRESS venv install-bytes 123\n"),
                         ("venv", "install-bytes", 123))
        self.assertEqual(parse_progress("VI-PROGRESS train step 0"), ("train", "step", 0))
        self.assertEqual(parse_progress("VI-PROGRESS base-model-fetch bytes-on-disk 9 \r\n"),
                         ("base-model-fetch", "bytes-on-disk", 9))
        self.assertEqual(parse_progress_end("VI-PROGRESS-END tokenise\n"), "tokenise")

    def test_malformed_lines_are_ignored(self):
        for bad in ("", "VI-PROGRESS", "VI-PROGRESS venv install-bytes",
                    "VI-PROGRESS venv install-bytes -5", "VI-PROGRESS venv install-bytes 1.5",
                    "VI-PROGRESS venv install-bytes 12abc", "VI-PROGRESS venv install-bytes 1 2",
                    "VI-PROGRESS Venv install-bytes 1", "VI-PROGRESS venv install_bytes 1",
                    "  VI-PROGRESS venv install-bytes 1", "xVI-PROGRESS venv install-bytes 1",
                    "debug-pass:   | VI-PROGRESS venv install-bytes 1",
                    "VI-PROGRESS-END venv", "VI-PROGRESS venv install-bytes " + "9" * 20,
                    None, b"VI-PROGRESS venv install-bytes 1"):
            self.assertIsNone(parse_progress(bad), bad)
        for bad in ("VI-PROGRESS-END", "VI-PROGRESS-END venv extra", "VI-PROGRESS venv a 1",
                    None):
            self.assertIsNone(parse_progress_end(bad), bad)


class ValidityCounters(unittest.TestCase):
    def setUp(self):
        self.c = FakeClock()

    def test_strictly_increasing_counter_stays_valid(self):
        v = machine(self.c)
        v.enter("tokenise", stall_s=120)
        for n in range(20):
            self.c.advance(60)
            v.evidence("progress", n, counter="tokenise/shard-bytes")
            self.assertEqual(v.poll(), V.VALID, n)

    def test_repeated_value_turns_suspect(self):
        v = machine(self.c)
        v.enter("tokenise", stall_s=120)
        v.evidence("progress", 5, counter="tokenise/shard-bytes")
        for _ in range(4):
            self.c.advance(40)
            v.evidence("progress", 5, counter="tokenise/shard-bytes")
            v.poll()
        self.assertEqual(v.state, V.SUSPECT)

    def test_liveness_never_recovers_suspect(self):
        v = machine(self.c)
        v.enter("venv", stall_s=120)
        self.c.advance(121)
        self.assertEqual(v.poll(), V.SUSPECT)
        for kind in V.LIVENESS_KINDS:
            v.evidence(kind)
        self.assertEqual(v.poll(), V.SUSPECT)
        v.evidence("progress", 1, counter="venv/install-bytes")  # a baseline only
        self.assertEqual(v.poll(), V.SUSPECT)
        v.evidence("progress", 2, counter="venv/install-bytes")
        self.assertEqual(v.poll(), V.VALID)

    def test_trickle_below_min_rate_turns_suspect(self):
        spec = V.STAGE_TABLE["base-model-fetch"]
        v = machine(self.c)
        v.enter("base-model-fetch", stall_s=spec.stall_s)
        have = 0
        for _ in range(int(spec.stall_s // 5) + 1):
            self.c.advance(5)
            have += 50_000  # 10 KB/s
            v.evidence("progress", have, counter="base-model-fetch/bytes-on-disk",
                       min_rate=spec.min_rate)
            v.poll()
        self.assertEqual(v.state, V.SUSPECT)

    def test_fast_enough_download_stays_valid(self):
        spec = V.STAGE_TABLE["base-model-fetch"]
        v = machine(self.c)
        v.enter("base-model-fetch", stall_s=spec.stall_s)
        have = 0
        for _ in range(100):
            self.c.advance(5)
            have += 10_000_000  # 2 MB/s
            v.evidence("progress", have, counter="base-model-fetch/bytes-on-disk",
                       min_rate=spec.min_rate)
            self.assertEqual(v.poll(), V.VALID)


def fake_paid_run(v):
    """A PaidRun with only what note_line needs; the open rental's Validity is `v`."""
    run = PaidRun.__new__(PaidRun)
    at = dict(n=1, phase="preflight")
    run.tail = deque(maxlen=50)
    run.fatal_line = None
    run.step = run.total = None
    run.last_ack = -1
    run.in_teardown = False
    run.progress_stage = None
    run.recorded = []
    run.current_validity = lambda: v
    run.open_attempt = lambda: at
    run.record_validity = lambda val: run.recorded.append(val.state)
    run.write_progress = lambda force=False: None
    return run


class NoteLine(unittest.TestCase):
    def setUp(self):
        self.c = FakeClock()
        self.v = machine(self.c)
        self.v.enter("preflight")
        self.run = fake_paid_run(self.v)

    def feed(self, *lines):
        for line in lines:
            self.run.note_line(line + "\n")

    def test_rising_counter_lines_keep_the_stage_valid(self):
        for n in range(10):
            self.c.advance(60)
            self.feed(f"VI-PROGRESS tokenise shard-bytes {n * 1000}")
            self.assertEqual(self.v.poll(), V.VALID, n)
        self.assertEqual(self.v.stage, "tokenise")
        self.assertEqual(self.v.stall_s, V.STAGE_TABLE["tokenise"].stall_s)
        self.assertIsNone(self.v.deadline_s)  # the C.2 deadline is not armed (ad-hoc limits stay)

    def test_log_only_activity_does_not_count(self):
        self.feed("VI-PROGRESS tokenise shard-bytes 10")
        for n in range(6):
            self.c.advance(30)
            self.feed("tokenise: still working", "debug-pass: stage 6 tokenise: still running",
                      "VI-PROGRESS tokenise shard-bytes 10",  # repeated: a stall
                      f"VI-PROGRESS tokenise shard-bytes {11 + n}.5",  # malformed
                      f"VI-PROGRESS tokenise shard-bytes {20 + n} extra")  # malformed
            self.v.poll()
        self.assertEqual(self.v.state, V.SUSPECT)
        self.feed("VI-PROGRESS tokenise shard-bytes 11")
        self.assertEqual(self.v.poll(), V.VALID)
        self.assertEqual(self.run.recorded[-1], V.VALID)  # the recovery was recorded

    def test_end_line_hands_the_stage_back_to_the_phase(self):
        self.feed("VI-PROGRESS venv install-bytes 1", "VI-PROGRESS-END venv")
        self.assertEqual(self.v.stage, "preflight")
        self.assertIsNone(self.v.stall_s)
        self.assertIsNone(self.run.progress_stage)

    def test_checkpoint_push_adds_evidence_without_taking_the_stage(self):
        self.feed("VI-PROGRESS train step 1", "VI-PROGRESS checkpoint-push files-pushed 0",
                  "VI-PROGRESS checkpoint-push files-pushed 1")
        self.assertEqual(self.v.stage, "train")
        self.assertEqual(self.v.counters["checkpoint-push/files-pushed"], 1)

    def test_no_open_rental_is_a_no_op(self):
        self.run.current_validity = lambda: None
        self.feed("VI-PROGRESS venv install-bytes 1", "VI-PROGRESS-END venv")


# ----- a real fetch through a throttled local HTTP server ----------------------------------------
TRICKLE_BYTES_PER_S = 10_000

HUB_STUB = '''
import os, urllib.request
def snapshot_download(repo, revision=None, local_dir=None):
    with urllib.request.urlopen(os.environ["VI_TEST_URL"], timeout=30) as r, \\
            open(os.path.join(local_dir, "model.safetensors"), "wb") as f:
        while True:
            b = r.read(1024)
            if not b:
                break
            f.write(b)
            f.flush()
'''


class TrickleHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        total = TRICKLE_BYTES_PER_S * 120
        self.send_response(200)
        self.send_header("Content-Length", str(total))
        self.end_headers()
        sent = 0
        try:
            while sent < total and not self.server.stopping.is_set():
                self.wfile.write(b"x" * 1000)
                self.wfile.flush()
                sent += 1000
                time.sleep(1000 / TRICKLE_BYTES_PER_S)  # 1000 bytes per 0.1 s: 10 KB/s
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


class TrickleDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        server = self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TrickleHandler)
        server.daemon_threads = True
        server.stopping = threading.Event()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.stopping.set)

    def test_trickle_download_turns_suspect_within_the_stall_window(self):
        hub = self.root / "hub"
        hub.mkdir()
        (hub / "huggingface_hub.py").write_text(HUB_STUB)
        env = dict(os.environ, PYTHONPATH=str(hub),
                   VI_TEST_URL=f"http://127.0.0.1:{self.server.server_address[1]}/model",
                   # fetch_base_model's own floor off (0 MB/s), polling every 0.5 s, so only
                   # home's counter judges the trickle here
                   VI_FETCH_MIN_MBPS="0", VI_FETCH_STALL_WINDOW="2", VI_FETCH_BACKOFF="0")
        p = subprocess.Popen([sys.executable, str(FETCH), "k2-fsa/OmniVoice", "c5fdb5cc" + "0" * 32,
                              str(self.root / "base")], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True)

        def stop():
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            p.stdout.close()
            p.stderr.close()
        self.addCleanup(stop)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(ln) for ln in p.stderr], daemon=True).start()

        stall_s = 3.0  # C.2's 120 s, scaled down; the min rate is the table's own 1 MB/s
        spec = V.STAGE_TABLE["base-model-fetch"]
        v = V.Validity(wall_cap_s=3600, clock=time.monotonic, attempt=1)
        run = fake_paid_run(v)
        run.FOREGROUND_PROGRESS_STAGES = ()  # enter the stage by hand with the test window
        v.enter("base-model-fetch", stall_s=stall_s)
        t0 = time.monotonic()
        values, suspect_at = [], None
        while time.monotonic() - t0 < 20:
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                line = None
            if line:
                got = parse_progress(line)
                if got:
                    values.append(got[2])
                    run.note_progress(*got)
            if v.poll() == V.SUSPECT:
                suspect_at = time.monotonic() - t0
                break
        print(f"\ntrickle: 10 KB/s fetch -> {v.state} after "
              f"{suspect_at if suspect_at is not None else -1:.1f} s; counter {values[:1]} -> "
              f"{values[-1:]}")
        self.assertEqual(v.state, V.SUSPECT, values)
        self.assertLess(suspect_at, stall_s + 2.5)
        self.assertGreaterEqual(suspect_at, stall_s - 0.5)
        # it was a trickle, not a stop: the counter kept rising, just too slowly
        self.assertGreater(values[-1], values[0], values)
        self.assertEqual(values, sorted(values))
        self.assertEqual(spec.min_rate, 1e6)


# ----- on-worker.sh ---------------------------------------------------------------------------------
def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
class OnWorkerStall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_stage(self, name, job):
        settings = self.root / "settings.json"
        settings.write_text("{}\n")
        env = {
            **os.environ,
            "VI_WORKER_HOST": "local",
            "VI_WORKER_ROOT": str(self.root / "local-worker"),
            "VI_STAGE": name,
            "VI_OUT": str(self.root / f"{name}-out"),
            "VI_SETTINGS": str(settings),
            "VI_RETURN_PORT": str(free_port()),
            "VI_POLL_SECONDS": "1",
            "VI_PREFLIGHT_STALL_SECONDS": "3",
        }
        t0 = time.monotonic()
        result = subprocess.run([str(ON_WORKER), "bash", "-c", job, "preflight"], env=env,
                                capture_output=True, text=True, timeout=120)
        return result, time.monotonic() - t0

    def test_log_only_activity_does_not_count_as_progress(self):
        # One real counter opens the stage; after that only log lines, touched files and
        # malformed counter lines (which would "rise") arrive. The 3 s stall limit stops it.
        job = ('echo "VI-PROGRESS venv install-bytes 100"; i=0; while :; do i=$((i + 1)); '
               'echo "pip: still downloading $i"; echo "VI-PROGRESS venv install-bytes $i.5"; '
               'echo "VI-PROGRESS venv install-bytes $((100 + i)) x"; touch "$PWD/busy-$i"; '
               'sleep 0.3; done')
        result, took = self.run_stage("log-only", job)
        print(f"\nlog-only: counter stage with only log activity -> exit {result.returncode} "
              f"after {took:.1f}s")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("(preflight) advanced no progress counter for 3s (log and file activity "
                      "do not count as progress), over its stall limit", result.stderr)
        self.assertLess(took, 60)

    def test_rising_counters_keep_the_stage_alive(self):
        job = ('for i in $(seq 1 14); do echo "VI-PROGRESS venv install-bytes $i"; sleep 0.5; '
               'done; echo "VI-PROGRESS-END venv"; printf ok > "$VI_OUT/ok.txt"')
        result, took = self.run_stage("rising", job)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("over its stall limit", result.stderr)
        self.assertGreater(took, 6)  # it really outlived the 3 s stall limit
        # the lines went home through the forwarded log
        self.assertIn("VI-PROGRESS venv install-bytes 14", result.stderr)


class FinalSync(unittest.TestCase):
    """The pushes after hs_loop go through final_sync, which checks the receiver first."""

    def run_final_sync(self, alive):
        with tempfile.TemporaryDirectory() as tmp:
            called = Path(tmp) / "called"
            script = f'''
eval "$(awk '/^final_sync\\(\\) \\{{/ {{ p = 1 }} p {{ print }} p && /^\\}}/ {{ exit }}' {ON_WORKER})"
receiver_alive() {{ return {0 if alive else 1}; }}
worker_sync() {{ printf '%s\\n' "$*" >{called}; }}
final_sync /work/out/ vi-out
'''
            t0 = time.monotonic()
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
            return r, time.monotonic() - t0, called.read_text() if called.exists() else None

    def test_dead_receiver_fails_fast_with_a_clear_message(self):
        r, took, called = self.run_final_sync(alive=False)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("ALERT: the return receiver is gone; cannot push vi-out home", r.stderr)
        self.assertIsNone(called)  # no push into a dead receiver
        self.assertLess(took, 5)

    def test_live_receiver_pushes(self):
        r, _, called = self.run_final_sync(alive=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(called, "/work/out/ vi-out\n")

    def test_both_final_pushes_use_final_sync(self):
        text = ON_WORKER.read_text()
        after = text[text.index("\nhs_loop\n"):]
        self.assertNotIn('worker_sync "$work/', after)
        self.assertEqual(after.count('final_sync "$work/'), 2)


if __name__ == "__main__":
    unittest.main()
