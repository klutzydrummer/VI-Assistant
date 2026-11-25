"""Local tests for the worker transfer-speed probe (jobs/push-probe.sh); no rental, no network.

The "worker" is a local folder and `bash -c` stands in for ssh, the way on-worker.sh runs a local
worker. The return receiver is the real one (jobs/worker-return.sh start), on loopback, and curl is
a fake on PATH, so nothing leaves the machine.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
PROBE = JOBS / "push-probe.sh"
RETURN = JOBS / "worker-return.sh"

# The fake curl answers like curl's -w '%{http_code} %{size_...} %{time_total}'. FAKE_CURL picks
# how: ok, fail (cannot connect, exit 7), slow (honours --max-time, exit 28) or hang (ignores it).
FAKE_CURL = r'''#!/bin/bash
mode=${FAKE_CURL:-ok}
echo "$*" >> "$FAKE_CURL_LOG"
max=
up=
prev=
for a in "$@"; do
  [ "$prev" = --max-time ] && max=$a
  [ "$prev" = --data-binary ] && up=${a#@}
  prev=$a
done
case $mode in
  ok)
    if [ -n "$up" ]; then printf '200 %s 2.000000' "$(wc -c <"$up")"; else printf '200 104857600 4.000000'; fi
    exit 0 ;;
  fail) printf '000 0 0.010000'; exit 7 ;;
  slow) sleep "$max"; printf '000 0 %s.000000' "$max"; exit 28 ;;
  hang) sleep 1000; exit 0 ;;
esac
'''

RUN = r'''
set -euo pipefail
probe_ssh=(bash -c)
work=$1 VI_OUT=$2 worker_port=$3
. "$4"
push_probe
echo "after-probe rc=$?"
'''


@unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
class PushProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.out = root / "home-out"
        self.state = root / "state"
        self.work = root / "worker"
        (self.work / "jobs").mkdir(parents=True)
        (self.work / "out").mkdir()
        shutil.copy(RETURN, self.work / "jobs" / "worker-return.sh")
        self.port = subprocess.run(
            [RETURN, "start", self.out, root / "checkpoint", self.state],
            check=True, capture_output=True, text=True).stdout.strip()
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "curl").write_text(FAKE_CURL)
        (bin_dir / "curl").chmod(0o755)
        self.curl_log = root / "curl.log"
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "FAKE_CURL_LOG": str(self.curl_log),
                    "VI_PUSH_PROBE_BYTES": str(4 * 1024 * 1024),
                    "VI_PUSH_PROBE_UP_BYTES": str(1024 * 1024),
                    "VI_PUSH_PROBE_URLS": "http://fake.invalid/one http://fake.invalid/two",
                    "VI_PUSH_PROBE_UP_URL": "http://fake.invalid/up"}

    def tearDown(self):
        subprocess.run([RETURN, "stop", self.state], capture_output=True)
        self.tmp.cleanup()

    def probe(self, **env):
        started = time.monotonic()
        result = subprocess.run(
            ["bash", "-c", RUN, "probe", self.work, self.out, self.port, PROBE],
            env={**self.env, **env}, capture_output=True, text=True, timeout=120)
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("after-probe rc=0", result.stdout)
        found = re.findall(r"^push-probe: json (.*)$", result.stderr, re.M)
        self.assertEqual(len(found), 1, result.stderr)
        return json.loads(found[0]), result.stderr, elapsed

    def assert_cleaned(self):
        self.assertFalse((self.out / ".push-probe").exists(), "home's probe folder was left")
        self.assertFalse((self.work / ".push-probe-src").exists(), "worker's probe file was left")
        self.assertFalse((self.work / ".push-probe-up.bin").exists())

    def test_success_logs_three_legs_and_the_json_fields(self):
        doc, err, _ = self.probe()
        for name in ("tunnel-rsync", "4-streams", "public-curl"):
            self.assertRegex(err, rf"push-probe: leg [abc] {name}: (download )?[0-9.]+ MB/s")
        self.assertEqual(set(doc["legs"]), {"a", "b", "c"})
        self.assertTrue(doc["ran"])
        self.assertEqual(doc["cap_seconds"], 300)
        self.assertIsInstance(doc["seconds"], int)
        a, b, c = doc["legs"]["a"], doc["legs"]["b"], doc["legs"]["c"]
        self.assertEqual((a["name"], a["status"], a["bytes"]), ("tunnel-rsync", "ok", 4 * 1024 * 1024))
        self.assertGreater(a["mb_s"], 0)
        self.assertEqual((b["name"], b["status"], b["streams"]), ("4-streams", "ok", 4))
        self.assertEqual(b["bytes"], 4 * 1024 * 1024)
        self.assertEqual(len(b["per_stream_mb_s"]), 4)
        self.assertGreater(b["mb_s"], 0)
        self.assertEqual((c["name"], c["status"]), ("public-curl", "ok"))
        self.assertEqual(c["down_url"], "http://fake.invalid/one")
        self.assertEqual(c["down_mb_s"], 26.21)  # 104857600 bytes in 4 s
        self.assertEqual(c["up_bytes"], 1024 * 1024)
        for key in ("seconds", "mb_s", "status"):
            self.assertIn(key, a)
            self.assertIn(key, b)
        self.assert_cleaned()

    def test_download_only_when_no_sink_is_set(self):
        doc, err, _ = self.probe(VI_PUSH_PROBE_UP_URL="")
        self.assertIn("upload not measured (no sink set)", err)
        self.assertEqual(doc["legs"]["c"]["up_status"], "none")
        self.assertEqual(doc["legs"]["c"]["status"], "ok")
        self.assertNotIn("--data-binary", self.curl_log.read_text())

    def test_failed_legs_are_fail_soft(self):
        # The receiver is gone (leg a fails), curl cannot connect (leg c fails): the probe still
        # reports every leg and returns 0, so `set -e` in on-worker.sh carries on.
        subprocess.run([RETURN, "stop", self.state], check=True, capture_output=True)
        doc, err, _ = self.probe(FAKE_CURL="fail", VI_PUSH_PROBE_LEG_SECONDS="15")
        # worker-return.sh retries a refused connection until the leg's limit, so this ends as
        # a timeout with nothing received (or a failure, if its retry wait ran out first).
        self.assertIn(doc["legs"]["a"]["status"], ("fail", "timeout"))
        self.assertEqual(doc["legs"]["a"]["bytes"], 0)
        self.assertEqual(doc["legs"]["b"]["status"], "ok")
        self.assertEqual(doc["legs"]["c"]["status"], "fail")
        self.assertEqual(doc["legs"]["c"]["down_status"], "fail")
        # Every listed URL was tried before giving up.
        self.assertEqual(self.curl_log.read_text().count("fake.invalid/"), 3)
        self.assertRegex(err, r"leg a tunnel-rsync: null MB/s")
        self.assert_cleaned()

    def test_a_broken_helper_is_fail_soft(self):
        # No python3 at all: the JSON cannot be built, but the probe still returns 0.
        # Hermetic PATH: one dir holding a symlink to every tool on PATH except python. Dropping
        # whole PATH dirs instead would also drop bash and coreutils wherever they share a dir
        # with python3 (/usr/bin on the CI runner). bash is resolved here in the parent.
        bash = shutil.which("bash", path=self.env["PATH"])
        nopy = Path(self.tmp.name) / "nopy-bin"
        nopy.mkdir()
        for d in self.env["PATH"].split(":"):
            if not d or not Path(d).is_dir():
                continue
            for tool in Path(d).iterdir():
                if (re.fullmatch(r"python[0-9.]*(-config)?", tool.name) or (nopy / tool.name).exists()
                        or not os.access(tool, os.X_OK) or tool.is_dir()):
                    continue
                (nopy / tool.name).symlink_to(tool.resolve())
        env = {**self.env, "PATH": str(nopy), "VI_PUSH_PROBE_LEG_SECONDS": "15"}
        check = subprocess.run([bash, "-c", "command -v python3 || command -v python || echo none"],
                               env=env, capture_output=True, text=True)
        self.assertEqual(check.stdout.strip(), "none")
        result = subprocess.run(
            [bash, "-c", RUN, "probe", self.work, self.out, self.port, PROBE],
            env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("after-probe rc=0", result.stdout)
        # The probe itself ran (legs reached), rather than bash failing before it.
        self.assertIn("push-probe: leg a", result.stderr)

    def test_the_time_cap_is_respected(self):
        # curl ignores its own --max-time and hangs; the leg is killed at the cap left and the
        # whole probe ends near VI_PUSH_PROBE_SECONDS, never at curl's 1000 s.
        doc, err, elapsed = self.probe(FAKE_CURL="hang", VI_PUSH_PROBE_SECONDS="20")
        self.assertLess(elapsed, 20 + 8, err)
        self.assertEqual(doc["cap_seconds"], 20)
        c = doc["legs"]["c"]
        self.assertEqual(c["down_status"], "timeout")
        self.assertIn(c["up_status"], ("timeout", "fail"))  # too little time left for the upload
        self.assert_cleaned()

    def test_curl_max_time_counts_as_timeout(self):
        doc, _, elapsed = self.probe(FAKE_CURL="slow", VI_PUSH_PROBE_SECONDS="40",
                                     VI_PUSH_PROBE_CURL_SECONDS="12")
        self.assertLess(elapsed, 40 + 8)
        self.assertEqual(doc["legs"]["c"]["down_status"], "timeout")

    def test_no_time_left_skips_the_legs(self):
        doc, err, elapsed = self.probe(VI_PUSH_PROBE_SECONDS="5")
        self.assertLess(elapsed, 15)
        self.assertEqual({leg["status"] for leg in doc["legs"].values()}, {"skip"})
        self.assertEqual(err.count("SKIP (no time left"), 3)

    def test_a_cut_off_push_still_gives_a_rate_and_is_cleaned(self):
        # A push slower than its limit: rsync is throttled so 4 MiB cannot finish in the 10 s leg.
        bin_dir = Path(self.env["PATH"].split(":")[0])
        real = shutil.which("rsync")
        (bin_dir / "rsync").write_text(f'#!/bin/bash\nexec {real} --bwlimit=100 "$@"\n')
        (bin_dir / "rsync").chmod(0o755)
        doc, err, elapsed = self.probe(VI_PUSH_PROBE_LEG_SECONDS="10", VI_PUSH_PROBE_SECONDS="40")
        a = doc["legs"]["a"]
        self.assertEqual(a["status"], "timeout", err)
        self.assertGreater(a["bytes"], 0)
        self.assertLess(a["bytes"], 4 * 1024 * 1024)
        self.assertLess(elapsed, 40 + 8)
        self.assert_cleaned()
        # No rsync child outlived the leg's limit.
        left = subprocess.run(["pgrep", "-f", f"--bwlimit=100.*{self.port}"],
                              capture_output=True, text=True).stdout.split()
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()


ON_WORKER = JOBS / "on-worker.sh"


@unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
class OnWorkerProbeTest(unittest.TestCase):
    """The probe inside a real local on-worker.sh run: when it runs, and that the gate is unchanged."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "curl").write_text(FAKE_CURL)
        (bin_dir / "curl").chmod(0o755)
        self.out = self.root / "out"
        self.settings = self.root / "settings.json"
        self.settings.write_text("{}\n")
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "FAKE_CURL_LOG": str(self.root / "curl.log"),
                    "VI_WORKER_HOST": "local", "VI_WORKER_ROOT": str(self.root / "local-worker"),
                    "VI_STAGE": "probe-test", "VI_OUT": str(self.out),
                    "VI_SETTINGS": str(self.settings),
                    "VI_PUSH_PROBE_BYTES": str(2 * 1024 * 1024),
                    "VI_PUSH_PROBE_UP_BYTES": "65536",
                    "VI_PUSH_PROBE_URLS": "http://fake.invalid/one",
                    "VI_PUSH_PROBE_UP_URL": "http://fake.invalid/up"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_job(self, **env):
        return subprocess.run([ON_WORKER, "bash", "-c", 'printf ok > "$VI_OUT/ok.txt"'],
                              env={**self.env, **env}, capture_output=True, text=True,
                              timeout=300)

    def test_debug_pass_runs_the_probe_after_the_gate(self):
        result = self.run_job(VI_DEBUG_PASS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.out / "ok.txt").read_text(), "ok")
        err = result.stderr
        self.assertIn("worker-preflight: ok", err)
        self.assertLess(err.index("worker-preflight: ok"), err.index("push-probe: start"))
        self.assertEqual(len(re.findall(r"^push-probe: json ", err, re.M)), 1)
        doc = json.loads(re.search(r"^push-probe: json (.*)$", err, re.M).group(1))
        self.assertEqual([doc["legs"][k]["status"] for k in "abc"], ["ok", "ok", "ok"])
        self.assertFalse((self.out / ".push-probe").exists())
        self.assertFalse((self.out / ".speedtest").exists())

    def test_a_normal_run_does_not_probe(self):
        result = self.run_job()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("push-probe:", result.stderr)

    def test_probe_can_be_turned_off_or_on(self):
        result = self.run_job(VI_DEBUG_PASS="1", VI_PUSH_PROBE="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("push-probe:", result.stderr)
        result = self.run_job(VI_PUSH_PROBE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("push-probe: json ", result.stderr)

    def test_failing_probe_legs_do_not_stop_the_run(self):
        result = self.run_job(VI_DEBUG_PASS="1", FAKE_CURL="fail")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.out / "ok.txt").read_text(), "ok")
        self.assertIn('"status": "fail"', result.stderr)

    def test_gate_unchanged_a_failed_preflight_is_not_probed(self):
        # A refused worker still stops a normal run before any upload, and a debug pass still
        # carries on, but neither probes a worker that failed its gate.
        self.settings.write_text('{"min_ram_gb": 100000}\n')
        result = self.run_job(VI_PUSH_PROBE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the worker preflight failed; nothing was uploaded", result.stderr)
        self.assertNotIn("push-probe:", result.stderr)
        result = self.run_job(VI_DEBUG_PASS="1")
        self.assertIn("debug pass: carrying on", result.stderr)
        self.assertNotIn("push-probe:", result.stderr)
