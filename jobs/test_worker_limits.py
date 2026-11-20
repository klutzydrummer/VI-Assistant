"""Tests for jobs/worker-limits.sh and the watched tokenise step in jobs/voicetune-fresh.sh.

The r7 preflight died on a box with 48 visible CPUs and a few thousand allowed tasks: OpenBLAS
started 48 threads per process, a thread could not be created, the process pool broke, and an
orphaned Manager kept the `| tee` pipe open so the stage never wrote stage.rc. These tests feed the
limits script fake cgroup files, and run the real `tokenise` function (cut out of
voicetune-fresh.sh by name, as the home rehearsal does) against stub tokenisers that crash, leave
orphans or hang, checking that the stage always ends with a nonzero exit code and no survivors.

Each fast-fail test prints how many seconds the stage took to fail (the round14 report quotes them).
"""
import os, re, signal, subprocess, sys, tempfile, textwrap, time, unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
LIMITS = JOBS / "worker-limits.sh"
FRESH = JOBS / "voicetune-fresh.sh"


def cut_functions(names):
    """The named functions' source from voicetune-fresh.sh, like jobs/rehearsal/vast-limits.sh."""
    text = FRESH.read_text().splitlines()
    out = []
    for name in names:
        start = next(i for i, l in enumerate(text) if l.startswith(f"{name}() {{"))
        end = next(i for i in range(start, len(text)) if text[i] == "}")
        out.extend(text[start:end + 1])
    return "\n".join(out) + "\n"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill 0; it holds no pipe and runs no code.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


class Limits(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.cg = self.tmp / "cg"
        self.proc = self.tmp / "proc"
        self.cg.mkdir()
        (self.proc / "sys" / "kernel").mkdir(parents=True)
        (self.proc / "loadavg").write_text("0.10 0.20 0.30 1/50 999\n")
        (self.proc / "sys" / "kernel" / "threads-max").write_text("1000000\n")

    def cgroup(self, pids_max="max", current="10", cpu="max 100000", cpuset="0-47"):
        (self.cg / "pids.max").write_text(f"{pids_max}\n")
        (self.cg / "pids.current").write_text(f"{current}\n")
        (self.cg / "cpu.max").write_text(f"{cpu}\n")
        (self.cg / "cpuset.cpus.effective").write_text(f"{cpuset}\n")

    def limits(self, nproc="48"):
        env = dict(os.environ, CGROUP_ROOT=str(self.cg), VI_LIMITS_PROC=str(self.proc),
                   VI_LIMITS_NPROC=nproc)
        script = f'. {LIMITS}; worker_limits; rc=$?; echo "rc=$rc omp=${{OMP_NUM_THREADS:-}} ' \
                 'openblas=${OPENBLAS_NUM_THREADS:-} mkl=${MKL_NUM_THREADS:-} ' \
                 'numexpr=${NUMEXPR_NUM_THREADS:-} nj=${VI_LIMIT_NJ:-}"'
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                              timeout=30)

    def test_limits_script_caps_threads_from_cgroup(self):
        # The r7 box: 48 CPUs visible, an 11.52-core quota, 2800 tasks allowed.
        self.cgroup(pids_max="2800", current="100", cpu="1152000 100000", cpuset="0-47")
        r = self.limits()
        self.assertIn("limits: nproc=48 cpus=11 quota=11 cpuset=48 pids.max=2800 current=100",
                      r.stdout)
        self.assertIn("-> threads=2 nj=2", r.stdout)
        self.assertIn("rc=0 omp=2 openblas=2 mkl=2 numexpr=2 nj=2", r.stdout)
        self.assertEqual(len(re.findall(r"^limits: ", r.stdout, re.M)), 1)

    def test_one_cpu_quota_means_one_worker_and_one_thread(self):
        self.cgroup(cpu="50000 100000")
        r = self.limits()
        self.assertIn("-> threads=1 nj=1", r.stdout)
        self.assertIn("rc=0 omp=1", r.stdout)

    def test_cpuset_and_task_room_both_cap(self):
        # 3 CPUs in the cpuset; 120 free tasks -> 120 / 2 / 5 / 3 = 4, cpus / nj = 1.
        self.cgroup(pids_max="130", current="10", cpuset="0-1,5")
        r = self.limits()
        self.assertIn("cpus=3", r.stdout)
        self.assertIn("-> threads=1 nj=2", r.stdout)

    def test_too_few_tasks_is_a_retryable_gate(self):
        # nj=2 -> 5 processes -> at least 40 free task slots are needed; 30 are free.
        self.cgroup(pids_max="40", current="10")
        r = self.limits()
        self.assertIn("rc=1 omp=", r.stdout)
        self.assertRegex(r.stderr, r"^worker-preflight: tasks: only 30 free task slots")
        self.assertNotRegex(r.stdout + r.stderr, r"Traceback|BrokenProcessPool|pthread_create")

    def test_missing_cgroup_files_mean_no_limit(self):
        r = self.limits(nproc="8")
        self.assertIn("pids.max=max", r.stdout)
        self.assertIn("-> threads=2 nj=2", r.stdout)
        self.assertIn("rc=0", r.stdout)

    def test_ulimit_and_threads_max_also_cap(self):
        self.cgroup()
        (self.proc / "sys" / "kernel" / "threads-max").write_text("80\n")
        r = self.limits()
        # 80 - 50 running = 30 free, under the 40 needed.
        self.assertIn("rc=1", r.stdout)
        self.assertIn("threads-max=80", r.stderr)


# A stand-in for `python -m omnivoice.scripts.extract_audio_tokens`. STUB_MODE picks the fault:
#   orphan  start a child that holds stdout open and sleeps, then exit 1 (r7's orphaned Manager)
#   hang    print the r7 crash lines, start a child, then block forever (r7's _finalize_manager)
#   noisy   print a Traceback line, keep printing progress for a while, then succeed
# Child pids go to $STUB_PIDS so the test can check they are gone.
STUB = r'''#!/usr/bin/env python3
import os, subprocess, sys, time
mode = os.environ["STUB_MODE"]
pids = open(os.environ["STUB_PIDS"], "a")
def child():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    pids.write(f"{p.pid}\n"); pids.flush()
pids.write(f"{os.getpid()}\n"); pids.flush()
if mode == "orphan":
    child()
    print("Traceback (most recent call last):", flush=True)
    print("concurrent.futures.process.BrokenProcessPool: A process in the pool was terminated", flush=True)
    os._exit(1)
if mode == "hang":
    child()
    print("OpenBLAS blas_thread_init: pthread_create failed for thread 7 of 48: Resource temporarily unavailable", flush=True)
    print("KeyboardInterrupt", flush=True)
    print("concurrent.futures.process.BrokenProcessPool: A process in the pool was terminated", flush=True)
    time.sleep(600)
if mode == "noisy":
    print("Traceback (most recent call last):  (one bad clip, skipped)", flush=True)
    for i in range(6):
        time.sleep(1)
        print(f"progress {i}", flush=True)
    print("Processing Complete - Successful: 2, Failed: 0, Filtered/Skipped: 0", flush=True)
    args = sys.argv
    tar = args[args.index("--tar_output_pattern") + 1]
    d = os.path.dirname(os.path.dirname(tar))
    open(os.path.join(d, "data.lst"), "w").write("x\n")
    sys.exit(0)
'''


class WatchedTokenise(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.stub = self.tmp / "py"
        self.stub.write_text(STUB)
        self.stub.chmod(0o755)
        self.pids = self.tmp / "pids"
        self.pids.write_text("")
        (self.tmp / "data").mkdir()
        self.manifest = self.tmp / "data" / "m.jsonl"
        self.manifest.write_text('{"id": "a"}\n{"id": "b"}\n')
        self.addCleanup(self.kill_leftovers)

    def kill_leftovers(self):
        # Only the pids the stubs this test started wrote down.
        for line in self.pids.read_text().split():
            try:
                os.kill(int(line), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass

    def stage(self, mode, quiet="3", budget=60, name="train"):
        """Run tokenise the way the stage wrapper does and return (stage.rc, seconds, output)."""
        defs = cut_functions(["audio_tokenizer_dir", "tokenise", "tokenise_count_guard"])
        rc_file = self.tmp / "stage.rc"
        script = textwrap.dedent(f'''\
            . {LIMITS}
            {defs}
            PY={self.stub}
            data_dir={self.tmp / "data"}
            export VI_AUDIO_TOKENIZER={self.tmp}
            set -euo pipefail
            rc=0
            tokenise {name} {self.manifest} || rc=$?
            printf '%s\\n' "$rc" > {rc_file}
            ''')
        env = dict(os.environ, STUB_MODE=mode, STUB_PIDS=str(self.pids),
                   VI_FATAL_QUIET_SECONDS=quiet, VI_WATCH_POLL_SECONDS="0.5")
        t0 = time.monotonic()
        # Read through a pipe, like the stage's job.log: an orphan holding it would block here.
        r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                           timeout=budget)
        secs = time.monotonic() - t0
        return int(rc_file.read_text()), secs, r.stdout + r.stderr

    def children_gone(self):
        pids = [int(p) for p in self.pids.read_text().split()]
        self.assertTrue(pids)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(alive(p) for p in pids):
            time.sleep(0.1)
        self.assertEqual([p for p in pids if alive(p)], [])

    def test_tokenise_kills_orphans_and_writes_rc(self):
        rc, secs, out = self.stage("orphan")
        print(f"\nfast-fail orphan: stage.rc={rc} after {secs:.1f}s", file=sys.stderr)
        self.assertNotEqual(rc, 0)
        self.assertIn("extract_audio_tokens exited 1", out)
        self.assertLess(secs, 20)
        self.children_gone()

    def test_tokenise_manager_hang_times_out(self):
        rc, secs, out = self.stage("hang", quiet="3")
        print(f"\nfast-fail manager hang (quiet limit 3 s): stage.rc={rc} after {secs:.1f}s",
              file=sys.stderr)
        self.assertNotEqual(rc, 0)
        self.assertIn("no new output for 3s", out)
        self.assertLess(secs, 20)
        self.children_gone()

    def test_fatal_pattern_watcher_fails_stage(self):
        # A Traceback line alone is not the end: output that keeps coming resets the quiet clock,
        # so a skipped bad clip under --skip_errors does not kill a healthy run.
        rc, secs, out = self.stage("noisy", quiet="3")
        self.assertEqual(rc, 0, out)
        self.assertIn("progress 5", out)
        self.assertNotIn("no new output", out)
        # The same fatal line followed by silence fails the stage with the watcher's code.
        rc, secs, out = self.stage("hang", quiet="2", name="dev")
        print(f"\nfast-fail watcher (quiet limit 2 s): stage.rc={rc} after {secs:.1f}s",
              file=sys.stderr)
        self.assertNotEqual(rc, 0)
        self.assertIn("tokenise: extract_audio_tokens exited 75", out)

    def test_nj_and_thread_caps_reach_the_tokeniser(self):
        text = FRESH.read_text()
        self.assertIn('--nj_per_gpu "${VI_LIMIT_NJ:-1}"', text)
        # The limits are sourced and applied before the venv's Python is first looked up.
        self.assertLess(text.index("worker_limits || exit 1"), text.index("PY=$(ovft_python)"))
        self.assertNotIn('--skip_errors 2>&1 | tee', text)


if __name__ == "__main__":
    unittest.main()
