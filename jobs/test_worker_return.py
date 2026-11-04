"""Local tests for the worker-initiated rsync return path (no SSH or GPU needed).

The receiver and reverse tunnel run on 127.0.0.1 only, so every test here is
local and disposable. The one OpenSSH-server test is skipped when the tools are
missing; the stub-ssh test works with just a shell and rsync.
"""

import getpass
import hashlib
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "worker-return.sh"
ON_WORKER = SCRIPT.parent / "on-worker.sh"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.3)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


@unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
class WorkerReturnTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.checkpoint = self.root / "checkpoint"
        self.publish = self.root / "publish"
        self.state = self.root / "state"
        self.src = self.root / "src"
        self.processes = []
        (self.src / "out").mkdir(parents=True)
        (self.src / "checkpoint").mkdir()
        started = subprocess.run(
            [SCRIPT, "start", self.out, self.checkpoint, self.state, self.publish],
            check=True, capture_output=True, text=True,
        )
        self.port = started.stdout.strip()
        # A short wait absorbs the moment the daemon's previous connection
        # takes to release its module slot; the retry class itself is tested
        # separately. Tests that must fail fast override it.
        self.env = {**os.environ, "VI_RETURN_PORT": self.port, "VI_PUSH_WAIT": "3"}

    def tearDown(self):
        for process in reversed(self.processes):
            process.terminate()
        for process in reversed(self.processes):
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        subprocess.run([SCRIPT, "stop", self.state], check=True, capture_output=True)
        self.tmp.cleanup()

    def push(self, source, module, *options, check=True, env=None):
        return subprocess.run(
            [SCRIPT, "push", source, module, *options],
            env=env or self.env, check=check, capture_output=True, text=True,
        )

    def assert_process_gone(self, pid, what):
        """Poll briefly for the process to be reaped; fail (and kill) if not."""
        deadline = time.time() + 2
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            if time.time() >= deadline:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                self.fail(f"{what} (pid {pid}) still running")
            time.sleep(0.05)

    def receiver_pid(self):
        return int((self.state / "rsyncd.pid").read_text().strip())

    def kill_receiver(self):
        pid = self.receiver_pid()
        os.kill(pid, 9)
        self.assert_process_gone(pid, "rsync daemon")

    def restart(self):
        return subprocess.run([SCRIPT, "restart", str(self.state)], env=self.env,
                              check=True, capture_output=True, text=True)

    def background_push(self, source, module, *options):
        """A push that stays alive until killed: the caller owns the process group."""
        return subprocess.Popen(
            [SCRIPT, "push", str(source), module, *options], env=self.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )

    def stop_background(self, process):
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)

    def assert_refuses_download_and_delete(self, port):
        (self.out / "keep.txt").write_text("keep\n")
        # `refuse options = delete` covers --delete and its --delete-* variants:
        # the daemon rejects the request before any file is removed.
        for option in ("--delete", "--delete-after"):
            with self.subTest(option=option):
                result = subprocess.run(
                    [SCRIPT, "push", str(self.src / "out"), "vi-out", option],
                    env={**self.env, "VI_RETURN_PORT": str(port)},
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((self.out / "keep.txt").read_text(), "keep\n")
        download = subprocess.run(
            ["rsync", "-a", f"rsync://127.0.0.1:{port}/vi-out/keep.txt", str(self.root / "download.txt")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(download.returncode, 0)
        self.assertFalse((self.root / "download.txt").exists())

    def assert_module_list_empty(self, port):
        result = subprocess.run(
            ["rsync", f"rsync://127.0.0.1:{port}/"], capture_output=True, text=True,
        )
        for module in ("vi-out", "vi-checkpoint", "vi-publish"):
            self.assertNotIn(module, result.stdout)

    def assert_symlink_munged(self, port):
        secret = self.root / "secret.txt"
        secret.write_text("outside\n")
        link = self.src / "out" / "link"
        if not link.is_symlink():
            link.symlink_to(secret)
        result = subprocess.run(
            [SCRIPT, "push", str(self.src / "out"), "vi-out"],
            env={**self.env, "VI_RETURN_PORT": str(port)},
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        link = self.out / "link"
        self.assertTrue(link.is_symlink())
        self.assertTrue(os.readlink(link).startswith("/rsyncd-munged/"))
        with self.assertRaises(OSError):
            link.read_text()

    def test_worker_pushes_output_and_appends_checkpoint(self):
        (self.src / "out" / "report.txt").write_text("returned\n")
        self.push(self.src / "out", "vi-out")
        self.assertEqual((self.out / "report.txt").read_text(), "returned\n")

        progress = self.src / "checkpoint" / "progress.jsonl"
        progress.write_text('{"step":1}\n')
        self.push(self.src / "checkpoint", "vi-checkpoint")
        with progress.open("a") as f:
            f.write('{"step":2}\n')
        self.push(self.src / "checkpoint", "vi-checkpoint", "--append")
        self.assertEqual((self.checkpoint / "progress.jsonl").read_text(), '{"step":1}\n{"step":2}\n')

    def test_start_accepts_relative_paths(self):
        # A detached daemon does not keep this cwd, so `start` must resolve
        # OUT/CHECKPOINT/PUBLISH/STATE to absolute paths before writing the config.
        out = self.root / "rel-out"
        checkpoint = self.root / "rel-checkpoint"
        publish = self.root / "rel-publish"
        state = self.root / "rel-state"
        started = subprocess.run(
            [SCRIPT, "start", "rel-out", "rel-checkpoint", "rel-state", "rel-publish"],
            cwd=self.root, check=True, capture_output=True, text=True,
        )
        try:
            port = started.stdout.strip()
            conf = (state / "rsyncd.conf").read_text()
            self.assertIn(f"path = {out}\n", conf)
            self.assertIn(f"path = {checkpoint}\n", conf)
            self.assertIn(f"path = {publish}\n", conf)
            self.assertIn(f"pid file = {state}/rsyncd.pid\n", conf)
            (self.src / "out" / "rel.txt").write_text("relative\n")
            result = subprocess.run(
                [SCRIPT, "push", str(self.src / "out"), "vi-out"],
                env={**os.environ, "VI_RETURN_PORT": port}, check=True, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((out / "rel.txt").read_text(), "relative\n")
        finally:
            subprocess.run([SCRIPT, "stop", str(state)], check=True, capture_output=True)

    def test_publish_module_is_restricted_and_receives_epochs(self):
        publish = self.root / "publish"
        state = self.root / "pub-state"
        started = subprocess.run(
            [SCRIPT, "start", self.out, self.checkpoint, state, publish],
            check=True, capture_output=True, text=True,
        )
        port = started.stdout.strip()
        try:
            conf = (state / "rsyncd.conf").read_text()
            self.assertIn("[vi-publish]\n", conf)
            self.assertIn(f"path = {publish.resolve()}\n", conf)
            for line in ("read only = false\n", "write only = true\n",
                         "refuse options = delete\n", "list = false\n",
                         "hosts allow = 127.0.0.1\n"):
                self.assertIn(line, conf)
            # vi-publish stays hidden from the module list, like the others.
            listing = subprocess.run(
                ["rsync", f"rsync://127.0.0.1:{port}/"],
                capture_output=True, text=True,
            )
            self.assertNotIn("vi-publish", listing.stdout)
            (self.src / "publish" / "epoch-0").mkdir(parents=True)
            (self.src / "publish" / "epoch-0" / "talker.safetensors").write_text("w0\n")
            result = subprocess.run(
                [SCRIPT, "push", str(self.src / "publish"), "vi-publish"],
                env={**os.environ, "VI_RETURN_PORT": port}, check=True, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((publish / "epoch-0" / "talker.safetensors").read_text(), "w0\n")
            # delete is refused on the publish module too.
            refused = subprocess.run(
                [SCRIPT, "push", str(self.src / "publish"), "vi-publish", "--delete"],
                env={**os.environ, "VI_RETURN_PORT": port}, check=False, capture_output=True, text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
        finally:
            subprocess.run([SCRIPT, "stop", state], check=True, capture_output=True)

    def test_receiver_refuses_download_and_delete(self):
        self.assert_refuses_download_and_delete(self.port)

    def test_module_list_is_empty(self):
        self.assert_module_list_empty(self.port)

    def test_second_connection_same_module_refused(self):
        # Hold vi-out's single slot with a slow transfer; a second connection to
        # the same module must be refused (per-module max connections = 1), not
        # queued. --bwlimit throttles even on loopback so the first push stays
        # alive long enough to overlap the second.
        (self.src / "out" / "big.bin").write_bytes(os.urandom(4 * 1024 * 1024))
        first = self.background_push(self.src / "out", "vi-out", "--bwlimit=500")
        try:
            time.sleep(0.5)
            self.assertIsNone(first.poll(), "--bwlimit was ignored; the first transfer already finished")
            result = self.push(self.src / "out", "vi-out", check=False,
                               env={**self.env, "VI_PUSH_WAIT": "0"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("max connections", result.stderr)
        finally:
            self.stop_background(first)

    def test_concurrent_publish_and_checkpoint_both_succeed(self):
        # A publish owns only its own module's slot, so the checkpoint sync is
        # never starved and lands while the publish is still running.
        epoch = self.src / "publish" / "epoch-0"
        epoch.mkdir(parents=True)
        (epoch / "w.bin").write_bytes(os.urandom(4 * 1024 * 1024))
        publish = self.background_push(self.src / "publish", "vi-publish", "--bwlimit=500")
        try:
            time.sleep(0.5)
            self.assertIsNone(publish.poll(), "--bwlimit was ignored; the publish already finished")
            (self.src / "checkpoint" / "progress.jsonl").write_text('{"step":1}\n')
            result = self.push(self.src / "checkpoint", "vi-checkpoint")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((self.checkpoint / "progress.jsonl").read_text(), '{"step":1}\n')
        finally:
            self.stop_background(publish)

    def test_every_module_declares_max_connections_1(self):
        # The module default for max connections is 0 (unlimited), so each module
        # needs its own explicit slot and lock file; a global limit would make a
        # long publish starve the other modules again.
        sections, current = {}, "global"
        for line in (self.state / "rsyncd.conf").read_text().splitlines():
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections[current] = {}
            elif "=" in line:
                key, _, value = line.partition("=")
                sections.setdefault(current, {})[key.strip()] = value.strip()
        self.assertNotIn("max connections", sections["global"])
        self.assertNotIn("lock file", sections["global"])
        for module in ("vi-out", "vi-checkpoint", "vi-publish"):
            with self.subTest(module=module):
                self.assertEqual(sections[module]["max connections"], "1")
                self.assertEqual(sections[module]["lock file"], str(self.state / f"{module}.lock"))
                self.assertEqual(sections[module]["exclude"], "/rsyncd-munged/")

    def test_stop_leaves_no_rsync_process(self):
        pid = int((self.state / "rsyncd.pid").read_text().strip())
        self.assertGreater(pid, 0)
        subprocess.run([SCRIPT, "stop", self.state], check=True, capture_output=True)
        self.assert_process_gone(pid, "rsync daemon")

    def test_receiver_config_is_loopback_and_restricted(self):
        conf = (self.state / "rsyncd.conf").read_text()
        self.assertIn("address = 127.0.0.1\n", conf)
        self.assertIn("hosts allow = 127.0.0.1\n", conf)
        self.assertIn("max connections = 1\n", conf)
        self.assertIn("munge symlinks = true\n", conf)
        self.assertIn("refuse options = delete\n", conf)
        self.assertIn("list = false\n", conf)
        self.assertIn("read only = false\n", conf)
        self.assertIn("write only = true\n", conf)
        self.assertIn(f"path = {self.out}\n", conf)
        self.assertIn(f"path = {self.checkpoint}\n", conf)

    def test_uploaded_symlink_cannot_escape_receiver(self):
        self.assert_symlink_munged(self.port)

    def test_receiver_restart_keeps_identical_restricted_config(self):
        before = hashlib.sha256((self.state / "rsyncd.conf").read_bytes()).hexdigest()
        self.kill_receiver()
        self.assertEqual(self.restart().stdout.strip(), self.port)
        self.assertEqual(hashlib.sha256((self.state / "rsyncd.conf").read_bytes()).hexdigest(), before)
        self.assert_refuses_download_and_delete(self.port)
        self.assert_module_list_empty(self.port)
        self.assert_symlink_munged(self.port)

    def test_push_succeeds_after_receiver_killed_and_restarted(self):
        self.kill_receiver()
        self.assertEqual(self.restart().stdout.strip(), self.port)
        (self.src / "out" / "after.txt").write_text("after\n")
        self.push(self.src / "out", "vi-out")
        self.assertEqual((self.out / "after.txt").read_text(), "after\n")

    def test_worker_pushed_rsyncd_munged_dir_does_not_wedge_restart(self):
        # A worker running before this exclude could have pushed this dir at the
        # module root. It must not stop the receiver from coming back.
        (self.out / "rsyncd-munged").mkdir()
        (self.out / "rsyncd-munged" / "junk").write_text("x\n")
        self.kill_receiver()
        self.assertEqual(self.restart().stdout.strip(), self.port)
        (self.src / "out" / "after-munged.txt").write_text("ok\n")
        self.push(self.src / "out", "vi-out")
        self.assertEqual((self.out / "after-munged.txt").read_text(), "ok\n")

    def test_push_retries_rc10_until_tunnel_returns(self):
        # VI_RETURN_PORT points at a dead listener at first; .return-port is
        # rewritten later, and each attempt re-reads it.
        work = self.root / "work"
        (work / "out").mkdir(parents=True)
        (work / "out" / "retried.txt").write_text("retried\n")
        down = free_port()
        (work / ".return-port").write_text(f"{down}\n")
        timer = threading.Timer(1.5, lambda: (work / ".return-port").write_text(f"{self.port}\n"))
        timer.start()
        try:
            result = subprocess.run(
                [SCRIPT, "push", str(work / "out"), "vi-out"],
                env={**self.env, "VI_OUT": str(work / "out"),
                     "VI_RETURN_PORT": str(down), "VI_PUSH_WAIT": "10"},
                capture_output=True, text=True,
            )
        finally:
            timer.cancel()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.out / "retried.txt").read_text(), "retried\n")

    def test_push_fails_fast_on_rc23(self):
        started = time.time()
        result = subprocess.run(
            [SCRIPT, "push", str(self.root / "missing"), "vi-out"],
            env={**self.env, "VI_PUSH_WAIT": "30"}, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertLess(time.time() - started, 5, "rc 23 was retried instead of failing fast")

    def test_sweep_spares_concurrent_live_receiver(self):
        base = self.root / "return"
        live_state = base / "live"
        live_out = self.root / "live-out"
        started = subprocess.run(
            [SCRIPT, "start", live_out, self.root / "live-checkpoint", live_state],
            check=True, capture_output=True, text=True,
        )
        live_port = started.stdout.strip()
        try:
            live_pid = int((live_state / "rsyncd.pid").read_text())
            # A dir that merely names the live pid must not get it killed: the
            # cmdline check says that pid runs another dir's config.
            stale = base / "stale"
            stale.mkdir()
            (stale / "rsyncd.conf").write_text("")
            (stale / "rsyncd.pid").write_text(f"{live_pid}\n")
            # An abandoned receiver (owner gone, argv names its config) is swept.
            abandoned = base / "abandoned"
            abandoned.mkdir()
            (abandoned / "rsyncd.conf").write_text("")
            victim = subprocess.Popen(
                ["python3", "-c", "import time; time.sleep(60)", str(abandoned / "rsyncd.conf")])
            try:
                (abandoned / "rsyncd.pid").write_text(f"{victim.pid}\n")
                subprocess.run([SCRIPT, "sweep", str(base)], check=True, capture_output=True)
                os.kill(live_pid, 0)
                self.assertEqual(victim.wait(timeout=3), -15, "abandoned receiver was not swept")
                (self.src / "out" / "live.txt").write_text("live\n")
                subprocess.run(
                    [SCRIPT, "push", str(self.src / "out"), "vi-out"],
                    env={**os.environ, "VI_RETURN_PORT": live_port},
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual((live_out / "live.txt").read_text(), "live\n")
            finally:
                victim.kill()
                victim.wait(timeout=3)
        finally:
            subprocess.run([SCRIPT, "stop", str(live_state)], check=True, capture_output=True)

    def interrupt_publish(self, megabytes):
        """Push a publish dir and SIGKILL it once about half the data is home."""
        epoch = self.src / "publish" / "epoch-0"
        epoch.mkdir(parents=True)
        data = epoch / "talker.safetensors"
        with data.open("wb") as f:
            for _ in range(megabytes):
                f.write(os.urandom(1 << 20))
        digest = hashlib.sha256()
        with data.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        (epoch / "SHA256SUMS").write_text(f"{digest.hexdigest()}  talker.safetensors\n")
        log = (self.root / "interrupted-push.log").open("w")
        push = subprocess.Popen(
            [SCRIPT, "push", str(self.src / "publish"), "vi-publish", "--bwlimit=20000"],
            env=self.env, stdout=log, stderr=log, start_new_session=True,
        )
        minimum = megabytes * 1024 * 1024 // 2
        try:
            deadline = time.time() + 30
            while time.time() < deadline:
                # The daemon writes the growing temp file, then moves it into
                # .rsync-partial when the client dies, so count both.
                received = sum(p.stat().st_size for p in self.publish.rglob("*") if p.is_file())
                if received >= minimum:
                    break
                if push.poll() is not None:
                    self.fail("push finished before it could be interrupted: "
                              + (self.root / "interrupted-push.log").read_text())
                time.sleep(0.05)
            else:
                self.fail("timed out waiting for the partial publish")
        finally:
            try:
                os.killpg(push.pid, 9)
            except ProcessLookupError:
                pass
            push.wait(timeout=5)
            log.close()
        # Wait until the receiver has moved the temp file into .rsync-partial
        # and given the module's connection slot back before resuming.
        deadline = time.time() + 10
        while time.time() < deadline:
            if any(".rsync-partial" in p.parts for p in self.publish.rglob("*")):
                break
            time.sleep(0.05)
        return self.publish / "epoch-0"

    def test_publish_resumes_after_interrupt(self):
        epoch = self.interrupt_publish(200)
        result = subprocess.run(
            [SCRIPT, "push", str(self.src / "publish"), "vi-publish", "--stats"],
            env={**self.env, "VI_PUSH_WAIT": "10"}, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        sent = [int(n.replace(",", "")) for n in re.findall(r"Total bytes sent: ([\d,]+)", result.stdout)]
        self.assertTrue(sent, f"no rsync --stats output: {result.stdout[-2000:]}")
        self.assertLess(sum(sent), 0.6 * 200 * 1024 * 1024, f"resume sent {sum(sent)} bytes")
        check = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=epoch,
                               capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

    def test_partial_epoch_never_verifies(self):
        epoch = self.interrupt_publish(32)
        self.assertFalse((epoch / "talker.safetensors").exists())
        check = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=epoch,
                               capture_output=True, text=True)
        self.assertNotEqual(check.returncode, 0)

    def test_receiver_stopped_after_success_and_failure(self):
        settings = self.root / "settings.json"
        settings.write_text("{}\n")
        for fail in (False, True):
            with self.subTest(fail=fail):
                out = self.root / f"stopped-{fail}"
                port = free_port()
                env = {
                    **os.environ,
                    "VI_WORKER_HOST": "local",
                    "VI_WORKER_ROOT": str(self.root / "local-worker"),
                    "VI_STAGE": f"stopped-{fail}",
                    "VI_OUT": str(out),
                    "VI_SETTINGS": str(settings),
                    "VI_RETURN_PORT": str(port),
                }
                job = 'printf ok > "$VI_OUT/ok.txt"; exit 3' if fail else 'printf ok > "$VI_OUT/ok.txt"'
                result = subprocess.run(
                    [str(ON_WORKER), "bash", "-c", job], env=env, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 3 if fail else 0, result.stderr)
                if not fail:
                    self.assertEqual((out / "ok.txt").read_text(), "ok")
                self.assertFalse(port_open(port), f"receiver still listening after {'failure' if fail else 'success'}")

    def test_tunnel_stopped_after_success_and_failure(self):
        # Stub `ssh` so on-worker.sh's non-local path runs its "worker" commands
        # locally, and its reverse-tunnel invocation records its PID and holds
        # the line. This exercises the real launcher's cleanup on both exits.
        bindir = write_stub_ssh(self.root / "bin")
        settings = self.root / "settings.json"
        settings.write_text("{}\n")
        for fail in (False, True):
            with self.subTest(fail=fail):
                out = self.root / f"tunnel-out-{fail}"
                pidfile = self.root / f"tunnel-{fail}.pid"
                env = {
                    **os.environ,
                    "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
                    "VI_WORKER_HOST": "127.0.0.1",
                    "VI_WORKER_PORT": "22",
                    "VI_WORKER_KNOWN_HOSTS": str(self.root / "known_hosts"),
                    "VI_WORKER_ROOT": str(self.root / "worker"),
                    "VI_STAGE": f"tunnel-{fail}",
                    "VI_OUT": str(out),
                    "VI_SETTINGS": str(settings),
                    "VI_BIN": "true",
                    "SSH_STUB_LOG": str(self.root / "stub-tunnels.log"),
                    "SSH_STUB_FORWARDER": str(bindir / "forward.py"),
                    "SSH_STUB_TUNNEL_PID": str(pidfile),
                }
                job = 'printf ok > "$VI_OUT/ok.txt"; exit 3' if fail else 'printf ok > "$VI_OUT/ok.txt"'
                result = subprocess.run(
                    [str(ON_WORKER), "bash", "-c", job], env=env, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 3 if fail else 0, result.stderr)
                if not fail:
                    self.assertEqual((out / "ok.txt").read_text(), "ok")
                tunnel_pid = int(pidfile.read_text().strip())
                self.assert_process_gone(tunnel_pid, "reverse tunnel")

    def wait_port(self, port, process):
        for _ in range(50):
            self.assertIsNone(process.poll(), "SSH process exited before opening its listener")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.1)
        self.fail(f"port {port} did not open")

    def start_disposable_sshd(self):
        """Start an sshd on loopback as this user; return (process, port).

        It runs unprivileged, so it can only accept this user's key. That is
        enough for the session and tunnel tests; on-worker.sh logs in as root
        and cannot use this harness.
        """
        sshd_port = free_port()
        for name in ("host", "client"):
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", self.root / name], check=True)
        config = self.root / "sshd_config"
        config.write_text(f"""Port {sshd_port}
ListenAddress 127.0.0.1
HostKey {self.root / 'host'}
PidFile {self.root / 'sshd.pid'}
AuthorizedKeysFile {self.root / 'client.pub'}
AllowUsers {getpass.getuser()}
AllowTcpForwarding remote
GatewayPorts no
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
StrictModes no
PrintMotd no
""")
        with (self.root / "sshd.log").open("w") as log:
            sshd = subprocess.Popen([shutil.which("sshd"), "-D", "-e", "-f", config],
                                    stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(sshd)
        self.wait_port(sshd_port, sshd)
        return sshd, sshd_port

    def sshd_sessions(self, sshd):
        """Pids of this sshd's session children that are not on a tty.

        OpenSSH 9 titles them `sshd: user@notty`; OpenSSH 10 uses
        `sshd-session: user@notty`. Only descendants of this disposable
        listener are considered, so no other login on the machine matches.
        """
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="],
                             capture_output=True, text=True, check=True).stdout
        children = {}
        for line in out.splitlines():
            pid, ppid, args = line.split(None, 2)
            children.setdefault(int(ppid), []).append((int(pid), args))
        sessions, stack = [], [sshd.pid]
        while stack:
            for pid, args in children.get(stack.pop(), []):
                stack.append(pid)
                if "@notty" in args and "sshd" in args:
                    sessions.append(pid)
        return sessions

    def kill_if_ours(self, pid, marker):
        """SIGKILL pid's group only if its command line still names `marker`."""
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return
        if str(marker).encode() in cmdline:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @unittest.skipUnless(shutil.which("ssh") and shutil.which("sshd") and shutil.which("ssh-keygen"),
                         "OpenSSH client/server tools are not installed")
    def test_stage_session_killed_mid_job_leaves_detached_job_running(self):
        """Row 1: killing the stage's ssh session mid-job must not stop it.

        The sshd harness runs on-worker.sh's launch shape (`setsid -f` plus
        `</dev/null`), then holds the session open like the old long stage
        session. Killing the session and its old process group must leave the
        detached job alive and finishing. A full on-worker.sh run against this
        server is impossible because on-worker.sh logs in as root and the
        disposable server runs as the test user; the supervisor's reattach
        half is covered by the stub-ssh reconnect tests.
        """
        sshd, port = self.start_disposable_sshd()
        work = self.root / "stage-session"
        work.mkdir()
        (work / "job.sh").write_text(
            "for i in 1 2 3 4 5 6 7 8; do\n"
            f'  printf "%s\\n" "$i" >> {work / "steps"}\n'
            "  sleep 0.5\n"
            "done\n"
            f'printf "0\\n" > {work / "stage.rc"}\n'
        )
        launcher = self.root / "launch.sh"
        launcher.write_text(
            f"echo $$ > {work / 'launcher.pid'}\n"
            f"setsid -f bash -c 'echo $$ > {work / 'stage.pid'}; bash {work / 'job.sh'} "
            f">> {work / 'job.log'} 2>&1; printf \"%s\\n\" \"$?\" > {work / 'stage.rc'}' </dev/null\n"
            "sleep 60\n"
        )
        with (self.root / "session.log").open("w") as log:
            ssh = subprocess.Popen([
                shutil.which("ssh"), "-p", str(port),
                "-i", self.root / "client", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                f"{getpass.getuser()}@127.0.0.1", "bash", str(launcher),
            ], stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(ssh)

        deadline = time.time() + 15
        while time.time() < deadline:
            if (work / "stage.pid").exists() and (work / "steps").exists():
                break
            time.sleep(0.05)
        else:
            self.fail("the stage did not start through sshd\n"
                      + (self.root / "session.log").read_text()
                      + (self.root / "sshd.log").read_text())
        stage_pid = int((work / "stage.pid").read_text())
        launcher_pid = int((work / "launcher.pid").read_text())
        self.addCleanup(self.kill_if_ours, launcher_pid, launcher)
        steps_before = len((work / "steps").read_text().splitlines())
        self.assertLess(steps_before, 8, "the job finished before the session was killed")

        sessions = self.sshd_sessions(sshd)
        self.assertTrue(sessions, "no sshd session process to kill")
        for pid in sessions:
            os.kill(pid, signal.SIGTERM)
        # A dying session also hangs up its process group; the setsid'd job is
        # in its own group and must not care.
        try:
            os.killpg(launcher_pid, signal.SIGHUP)
        except ProcessLookupError:
            pass
        ssh.wait(timeout=5)
        self.assertIsNotNone(ssh.poll())

        os.kill(stage_pid, 0)  # still alive right after the session died
        deadline = time.time() + 15
        while time.time() < deadline and not (work / "stage.rc").exists():
            time.sleep(0.05)
        self.assertEqual((work / "stage.rc").read_text().strip(), "0",
                         (work / "job.log").read_text())
        self.assertEqual(len((work / "steps").read_text().splitlines()), 8)
        self.assert_process_gone(stage_pid, "detached stage")

    @unittest.skipUnless(shutil.which("ssh") and shutil.which("sshd") and shutil.which("ssh-keygen"),
                         "OpenSSH client/server tools are not installed")
    def test_reverse_ssh_tunnel_carries_worker_push(self):
        sshd, sshd_port = self.start_disposable_sshd()
        remote_port = free_port()
        with (self.root / "tunnel.log").open("w") as log:
            tunnel = subprocess.Popen([
                shutil.which("ssh"), "-N", "-T", "-p", str(sshd_port),
                "-i", self.root / "client", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=1",
                "-R", f"127.0.0.1:{remote_port}:127.0.0.1:{self.port}", f"{getpass.getuser()}@127.0.0.1",
            ], stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(tunnel)
        self.wait_port(remote_port, tunnel)
        (self.src / "out" / "tunnel.txt").write_text("through ssh\n")
        result = subprocess.run([SCRIPT, "push", self.src / "out", "vi-out"],
                                env={**self.env, "VI_RETURN_PORT": str(remote_port)},
                                capture_output=True, text=True)
        logs = "\n" + (self.root / "sshd.log").read_text() + "\n" + (self.root / "tunnel.log").read_text()
        self.assertEqual(result.returncode, 0, result.stderr + logs)
        self.assertEqual((self.out / "tunnel.txt").read_text(), "through ssh\n")

    def test_on_worker_runs_without_a_checkpoint(self):
        out = self.root / "no-checkpoint-out"
        settings = self.root / "settings.json"
        settings.write_text("{}\n")
        env = {**os.environ, "VI_WORKER_HOST": "local", "VI_WORKER_ROOT": self.root / "local-worker",
               "VI_STAGE": "return-test", "VI_OUT": out, "VI_SETTINGS": settings}
        result = subprocess.run([ON_WORKER, "bash", "-c", "printf ok > \"$VI_OUT/ok.txt\""],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((out / "ok.txt").read_text(), "ok")

    def test_unknown_module_is_rejected(self):
        result = self.push(self.src / "out", "vi-other", check=False)
        self.assertEqual(result.returncode, 2)


STUB_SSH = r'''#!/bin/bash
set -u
# A tunnel invocation (-N): listen on the requested remote loopback port and
# forward it to the home receiver port, the way ssh -R would on the worker.
for a in "$@"; do
  if [ "$a" = -N ]; then
    prev=
    for b in "$@"; do [ "$prev" = -R ] && spec=$b; prev=$b; done
    rp=${spec#127.0.0.1:}; rp=${rp%%:*}
    rest=${spec#*:}; rest=${rest#*:}; p=${rest##*:}
    echo "$$ $rp $p" >> "$SSH_STUB_LOG"
    [ -n "${SSH_STUB_TUNNEL_PID:-}" ] && echo "$$" > "$SSH_STUB_TUNNEL_PID"
    exec python3 "$SSH_STUB_FORWARDER" "$rp" "$p"
  fi
done
# rsync's transport: run the server command locally. A push's client is
# throttled by a PATH wrapper (SSH_STUB_BWLIMIT) so a test can interrupt a
# transfer mid-flight.
i=0
for a in "$@"; do
  i=$((i + 1))
  if [ "$a" = rsync ]; then
    shift $((i - 1))
    exec "$@"
  fi
done
# A short link flap: every direct remote command fails while the flag exists.
if [ -e "${SSH_STUB_FAIL:-/nonexistent-flag}" ]; then exit 255; fi
# A regenerated worker host key: ssh refuses every direct command with the
# message a real client prints, until home accepts the new key through the API.
if [ -e "${SSH_STUB_HOSTKEY:-/nonexistent-hostkey-flag}" ]; then
  echo "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!" >&2
  echo "Host key verification failed." >&2
  exit 255
fi
# A direct remote command: everything after the root@host target.
i=0
for a in "$@"; do
  i=$((i + 1))
  if [[ "$a" == *@* ]]; then shift "$i"; break; fi
done
exec bash -c "$*"
'''

STUB_FORWARDER = r'''import socket, sys, threading
rp, p = int(sys.argv[1]), int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", rp))
s.listen(16)

def pipe(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        try:
            b.shutdown(socket.SHUT_WR)
        except OSError:
            pass

while True:
    client, _ = s.accept()
    peer = socket.create_connection(("127.0.0.1", p))
    threading.Thread(target=pipe, args=(client, peer), daemon=True).start()
    threading.Thread(target=pipe, args=(peer, client), daemon=True).start()
'''


def write_stub_ssh(bindir):
    """A stand-in ssh that runs the 'worker' locally and forwards tunnels.

    Direct commands run through bash; the rsync transport execs the server
    command it is given; -N starts a loopback TCP forwarder from the remote
    port to the local receiver port and records `pid remote_port local_port`.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "ssh").write_text(STUB_SSH)
    (bindir / "ssh").chmod(0o755)
    (bindir / "forward.py").write_text(STUB_FORWARDER)
    real_rsync = shutil.which("rsync")
    (bindir / "rsync").write_text(
        "#!/bin/sh\n"
        'echo "$@" >> "${SSH_STUB_RSYNC_LOG:-/dev/null}"\n'
        'if [ -n "${SSH_STUB_BWLIMIT:-}" ]; then\n'
        '  for a in "$@"; do\n'
        f'    case $a in *@*) exec {real_rsync} --bwlimit="$SSH_STUB_BWLIMIT" "$@";; esac\n'
        "  done\n"
        "fi\n"
        f'exec {real_rsync} "$@"\n'
    )
    (bindir / "rsync").chmod(0o755)
    return bindir


def wait_for(what, seconds, cond):
    deadline = time.time() + seconds
    while True:
        value = cond()
        if value:
            return value
        if time.time() >= deadline:
            raise AssertionError(what)
        time.sleep(0.05)


@unittest.skipUnless(shutil.which("rsync") and shutil.which("python3"),
                     "rsync and python3 are needed")
class HomeSupervisorTest(unittest.TestCase):
    """on-worker.sh's poll loop, tunnel reconnect and reattach (rows 3, 6-8, 33, 36, 46)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.state = self.root / "state"
        self.worker = self.root / "worker"
        self.out = self.root / "home-out"
        self.bindir = write_stub_ssh(self.root / "bin")
        self.settings = self.root / "settings.json"
        self.settings.write_text("{}\n")
        self.job_path = self.root / "job.sh"
        self.checkpoint = self.root / ".checkpoint-test"
        self.checkpoint.mkdir()
        self.tunnel_log = self.root / "tunnels.log"
        self.fail_flag = self.root / "link-down"
        self.hostkey_flag = self.root / "hostkey-changed"
        self.rotations_log = None
        self.logs = []
        self.procs = []

    def tearDown(self):
        for process in self.procs:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for log in self.logs:
            log.close()
        for conf in (self.state / "return").glob("*/rsyncd.conf"):
            subprocess.run([SCRIPT, "stop", str(conf.parent)], capture_output=True)
        for line in self.tunnel_lines():
            try:
                os.kill(line[0], 9)
            except ProcessLookupError:
                pass
        # The detached stage is its own session; kill only the pid that still
        # runs this test's work dir.
        for pidfile in self.worker.glob("*.*.*/stage.pid"):
            try:
                pid = int(pidfile.read_text().strip())
            except (OSError, ValueError):
                continue
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            if str(self.worker).encode() in cmdline:
                try:
                    os.killpg(pid, 9)
                except ProcessLookupError:
                    pass
        self.tmp.cleanup()

    # -- harness -----------------------------------------------------------
    def env(self, stage, **extra):
        env = {
            **os.environ,
            "PATH": f"{self.bindir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.home),
            "VI_STATE_DIR": str(self.state),
            "VI_WORKER_HOST": "127.0.0.1",
            "VI_WORKER_PORT": "22",
            "VI_WORKER_KNOWN_HOSTS": str(self.root / "known_hosts"),
            "VI_WORKER_ROOT": str(self.worker),
            "VI_STAGE": stage,
            "VI_OUT": str(self.out),
            "VI_SETTINGS": str(self.settings),
            "VI_CHECKPOINT": str(self.checkpoint),
            "VI_BIN": "true",
            "VI_POLL_SECONDS": "1",
            "VI_LOST_MINUTES": "1",
            "VI_SYNC_SECONDS": "1",
            "VI_PUSH_SECONDS": "1",
            "SSH_STUB_LOG": str(self.tunnel_log),
            "SSH_STUB_FORWARDER": str(self.bindir / "forward.py"),
            "SSH_STUB_FAIL": str(self.fail_flag),
            "SSH_STUB_HOSTKEY": str(self.hostkey_flag),
        }
        env.update(extra)
        return env

    def start(self, stage, job, **extra):
        self.job_path.write_text("set -e\n" + job)
        log = open(self.root / f"{stage}.err", "w")
        self.logs.append(log)
        process = subprocess.Popen(
            [str(ON_WORKER), "bash", str(self.job_path)],
            env=self.env(stage, **extra), stdout=subprocess.DEVNULL, stderr=log,
        )
        self.procs.append(process)
        return process

    def stderr(self, stage):
        return (self.root / f"{stage}.err").read_text()

    def tunnel_lines(self):
        try:
            text = self.tunnel_log.read_text()
        except OSError:
            return []
        out = []
        for line in text.splitlines():
            try:
                pid, rp, p = line.split()
                out.append((int(pid), int(rp), int(p)))
            except ValueError:
                continue
        return out

    def wait_tunnels(self, count, seconds=25):
        return wait_for(f"fewer than {count} tunnel(s) started",
                        seconds, lambda: len(self.tunnel_lines()) >= count
                        and self.tunnel_lines()[count - 1])

    def return_dir(self, stage):
        def find():
            dirs = [p.parent for p in (self.state / "return").glob(f"{stage}.*/tunnel.log")]
            return dirs[0] if dirs else None
        return find()

    def return_log(self, stage):
        path = self.return_dir(stage)
        return (path / "tunnel.log").read_text() if path else ""

    def worker_work(self, stage):
        def find():
            dirs = list(self.worker.glob(f"{stage}.*"))
            return dirs[0] if dirs else None
        return find()

    def wait_worker_work(self, stage, seconds=20):
        return wait_for(f"the worker dir for {stage} never appeared", seconds,
                        lambda: self.worker_work(stage))

    def write_vi_stub(self, accept=True):
        """A stand-in `vi` for the host-key rotation path.

        `cloud hostkey` records the call in rotations.log and, when accepting,
        clears the host-key-changed flag. The real fingerprint matching and the
        Vast log check are covered by the Rust tests; this stub stands in for a
        successful `vi cloud hostkey`, so the test exercises on-worker.sh's
        response. `cloud env` refuses, which leaves the worker env unchanged.
        """
        self.rotations_log = self.root / "rotations.log"
        decision = (
            f"rm -f {shlex.quote(str(self.hostkey_flag))}\n    exit 0"
            if accept else "exit 1"
        )
        script = self.root / "vi"
        script.write_text(f'''#!/bin/bash
case "$1 $2" in
  "cloud hostkey")
    echo hostkey >> {shlex.quote(str(self.rotations_log))}
    {decision}
    ;;
  "cloud env") exit 1 ;;
  *) exit 0 ;;
esac
''')
        script.chmod(0o755)
        return script

    def rotations(self):
        if self.rotations_log is None:
            return 0
        try:
            return len(self.rotations_log.read_text().splitlines())
        except OSError:
            return 0

    def receiver(self, stage):
        path = self.return_dir(stage)
        if not path:
            return None
        pid_file = path / "rsyncd.pid"
        try:
            return int(pid_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def finish(self, process, seconds=90):
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            self.fail("on-worker.sh did not finish")

    # -- row 3: tunnel reconnect and token re-proof ------------------------
    def test_tunnel_reconnects_after_kill(self):
        stage = "tunnel-reconnect"
        process = self.start(stage, 'sleep 16\necho done > "$VI_OUT/ok.txt"\n')
        first = self.wait_tunnels(1)
        # The first start proves the path on a 5 s budget; let it finish so the
        # kill exercises the reconnect path and not a failed first start.
        wait_for("the first tunnel never proved the path", 20,
                 lambda: "return path re-proved" in self.stderr(stage))
        os.kill(first[0], 9)

        second = self.wait_tunnels(2)
        self.assertNotEqual(second[1], first[1], "the reconnect reused the old remote port")
        work = self.wait_worker_work(stage)
        wait_for("the worker was not told the new remote port", 10,
                 lambda: (work / ".return-port").is_file()
                 and (work / ".return-port").read_text().strip() == str(second[1]))
        # The successful cleanup removes the return dir, so keep the log now.
        # The first "tunnel up" line is not required: the token proof can win
        # the race against the loop's one-second liveness check.
        settled = f"tunnel up: remote loopback port {second[1]}"
        log = ""
        deadline = time.time() + 15
        while time.time() < deadline:
            text = self.return_log(stage)
            if text:
                log = text
            if settled in log:
                break
            time.sleep(0.05)
        self.assertIn(settled, log, log or self.stderr(stage))

        self.assertEqual(self.finish(process), 0, self.stderr(stage))
        self.assertIn("tunnel down", log)
        self.assertEqual((self.out / "ok.txt").read_text(), "done\n")

    def test_reconnect_uses_new_remote_port_when_old_listener_lingers(self):
        stage = "tunnel-lingering-port"
        process = self.start(stage, 'sleep 16\necho done > "$VI_OUT/ok.txt"\n')
        first = self.wait_tunnels(1)
        wait_for("the first tunnel never proved the path", 20,
                 lambda: "return path re-proved" in self.stderr(stage))
        os.kill(first[0], 9)
        # Stand in for a listener the worker-side ssh left on the old port.
        linger = subprocess.Popen([
            "python3", "-c",
            "import socket, time\n"
            "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            f"s.bind(('127.0.0.1', {first[1]})); s.listen()\n"
            "time.sleep(60)\n",
        ])
        try:
            time.sleep(0.3)
            self.assertIsNone(linger.poll(), "could not hold the old remote port")
            second = self.wait_tunnels(2)
            self.assertNotEqual(second[1], first[1])
            self.assertEqual(second[2], first[2], "the local receiver port must stay fixed")
            self.assertEqual(self.finish(process), 0, self.stderr(stage))
            self.assertEqual((self.out / "ok.txt").read_text(), "done\n")
            self.assertIsNone(linger.poll(), "the old port's listener was disturbed")
        finally:
            linger.kill()
            linger.wait(timeout=5)

    def test_reconnect_token_proof_survives_short_flap(self):
        stage = "tunnel-short-flap"
        process = self.start(stage, 'sleep 20\necho done > "$VI_OUT/ok.txt"\n')
        wait_for("the first tunnel never proved the path", 20,
                 lambda: "return path re-proved" in self.stderr(stage))
        self.wait_tunnels(1)
        os.kill(self.tunnel_lines()[0][0], 9)
        self.fail_flag.write_text("")
        time.sleep(3)
        self.fail_flag.unlink()
        self.wait_tunnels(2)
        self.assertEqual(self.finish(process, 90), 0, self.stderr(stage))
        err = self.stderr(stage)
        self.assertIn("poll failed; will retry", err)
        self.assertGreaterEqual(err.count("return path re-proved"), 2, err)
        self.assertEqual((self.out / "ok.txt").read_text(), "done\n")

    def test_security_holds_after_reconnect_and_receiver_restart(self):
        stage = "security-after-reconnect"
        process = self.start(stage, 'sleep 20\necho done > "$VI_OUT/ok.txt"\n')
        first = self.wait_tunnels(1)
        wait_for("the first tunnel never proved the path", 20,
                 lambda: "return path re-proved" in self.stderr(stage))
        os.kill(first[0], 9)
        second = self.wait_tunnels(2)
        old = self.receiver(stage)
        self.assertIsNotNone(old)
        os.kill(old, 9)
        def restarted():
            pid = self.receiver(stage)
            if pid in (None, old) or not Path(f"/proc/{pid}").exists():
                return None
            return pid

        new = wait_for("the receiver was never restarted", 20, restarted)
        self.assertNotEqual(new, old)
        wait_for("the restarted receiver is not listening", 20,
                 lambda: port_open(int((self.return_dir(stage) / "rsyncd.conf").read_text()
                                      .split("port = ")[1].splitlines()[0])))
        port = second[1]
        src = self.root / "security-src"
        (src / "out").mkdir(parents=True)
        (self.out / "keep.txt").write_text("keep\n")
        for option in ("--delete", "--delete-after"):
            with self.subTest(option=option):
                refused = subprocess.run(
                    [str(SCRIPT), "push", str(src / "out"), "vi-out", option],
                    env={**self.env(stage), "VI_RETURN_PORT": str(port)},
                    capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual((self.out / "keep.txt").read_text(), "keep\n")
        download = subprocess.run(
            ["rsync", "-a", f"rsync://127.0.0.1:{port}/vi-out/keep.txt",
             str(self.root / "download.txt")], capture_output=True, text=True)
        self.assertNotEqual(download.returncode, 0)
        self.assertFalse((self.root / "download.txt").exists())
        listing = subprocess.run(["rsync", f"rsync://127.0.0.1:{port}/"],
                                 capture_output=True, text=True)
        for module in ("vi-out", "vi-checkpoint", "vi-publish"):
            self.assertNotIn(module, listing.stdout)

        secret = self.root / "secret.txt"
        secret.write_text("outside\n")
        (src / "out" / "link").symlink_to(secret)
        result = subprocess.run(
            [str(SCRIPT), "push", str(src / "out"), "vi-out"],
            env={**self.env(stage), "VI_RETURN_PORT": str(port)},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        link = self.out / "link"
        self.assertTrue(link.is_symlink())
        self.assertTrue(os.readlink(link).startswith("/rsyncd-munged/"))
        with self.assertRaises(OSError):
            link.read_text()

        self.assertEqual(self.finish(process), 0, self.stderr(stage))
        self.assertEqual((self.out / "ok.txt").read_text(), "done\n")

    # -- row 8: a SIGKILLed runner leaves nothing behind -------------------
    def test_kill9_runner_leaves_no_receiver_or_tunnel(self):
        stage = "kill9-cleanup"
        process = self.start(stage, 'sleep 20\necho done > "$VI_OUT/ok.txt"\n')
        tunnel = self.wait_tunnels(1)
        work = self.wait_worker_work(stage)
        wait_for("the stage never started", 10, lambda: (work / "stage.pid").is_file())
        receiver = self.receiver(stage)
        self.assertIsNotNone(receiver)
        port = int((self.return_dir(stage) / "rsyncd.conf").read_text()
                   .split("port = ")[1].splitlines()[0])

        process.kill()
        process.wait(timeout=5)
        wait_for("the tunnel outlived the killed runner", 8,
                 lambda: not Path(f"/proc/{tunnel[0]}").exists())
        wait_for("the receiver outlived the killed runner", 8,
                 lambda: not Path(f"/proc/{receiver}").exists())
        self.assertFalse(port_open(port), "the receiver still accepts connections")
        # The detached job is untouched: home's death only detaches it (row 1).
        self.assertEqual(subprocess.run(
            ["kill", "-0", (work / "stage.pid").read_text().strip()]).returncode, 0)

    # -- row 7: reattach needs a fresh token -------------------------------
    def test_reattach_requires_fresh_token(self):
        stage = "reattach-token"
        process = self.start(
            stage, 'sleep 12\necho done > "$VI_OUT/ok.txt"\n',
            VI_WORKER_HOST="local",
        )
        work = self.wait_worker_work(stage)
        wait_for("the stage never started", 10, lambda: (work / "stage.pid").is_file())
        stage_pid = (work / "stage.pid").read_text().strip()
        wait_for("the first run never proved its return path", 10,
                 lambda: "return path re-proved" in self.stderr(stage))
        process.kill()
        process.wait(timeout=5)
        # A reattach must prove a FRESH token. Break only the worker-side
        # pusher; the home receiver still starts and answers.
        pusher = work / "jobs" / "worker-return.sh"
        pusher.write_text("#!/bin/sh\nexit 1\n")
        pusher.chmod(0o755)

        again = self.start(
            stage, 'sleep 12\necho done > "$VI_OUT/ok.txt"\n',
            VI_WORKER_HOST="local", VI_LOST_MINUTES="0", VI_PROVE_WAIT="0",
        )
        self.assertNotEqual(self.finish(again, 30), 0)
        err = self.stderr(stage)
        self.assertIn("reattaching to", err)
        self.assertIn("could not push through the reverse SSH return tunnel", err)
        self.assertIn("the return path did not come up; nothing will be sent", err)
        self.assertEqual((work / "stage.pid").read_text().strip(), stage_pid,
                         "the reattach relaunched the stage before proving the path")

    # -- row 46: the watchdog is restarted when missing --------------------
    def test_hs_restarts_missing_watchdog(self):
        stage = "watchdog-restart"
        alive = self.root / "watchdog.alive"
        script = self.root / "vi-watchdog.sh"
        script.write_text(f"#!/bin/sh\ntouch {alive}\nexec sleep 60\n")
        script.chmod(0o755)
        fake_pgrep = self.bindir / "pgrep"
        fake_pgrep.write_text(
            "#!/bin/sh\n"
            f'[ -e {alive} ] && exit 0\n'
            "exit 1\n"
        )
        fake_pgrep.chmod(0o755)
        process = self.start(
            stage, 'sleep 12\necho done > "$VI_OUT/ok.txt"\n',
            VI_WATCHDOG_SCRIPT=str(script),
            VI_WATCHDOG_LOG=str(self.root / "watchdog.log"),
        )
        try:
            wait_for("the missing watchdog was never restarted", 10, alive.exists)
            wait_for("the restart was not reported", 10,
                     lambda: "the worker watchdog was missing; restarted" in self.stderr(stage))
            self.assertEqual(self.finish(process), 0, self.stderr(stage))
        finally:
            subprocess.run(["pkill", "-f", str(script)], capture_output=True)

    # -- row 33: the upload gate reads the stage setting -------------------
    @unittest.skipUnless(shutil.which("zstd"), "zstd is not installed")
    def test_upload_gate_reads_stage_setting(self):
        stage = "upload-gate"
        data = self.root / "data"
        data.mkdir()
        (data / "clip.wav").write_bytes(b"audio\n")
        self.settings.write_text('{"max_upload_minutes": 0}\n')
        process = self.start(
            stage, 'echo ran > "$VI_OUT/ran.txt"\n',
            VI_INPUT_DATA=str(data), VI_PACK_MIN_KB="0",
            VI_BUNDLE_DIR=str(self.root / "bundles"),
        )
        self.assertNotEqual(self.finish(process, 60), 0)
        err = self.stderr(stage)
        self.assertIn("max_upload_minutes is 0", err)
        work = self.worker_work(stage)
        self.assertIsNotNone(work)
        self.assertFalse((work / "inputs" / "DATA").exists(), "inputs were sent")
        self.assertFalse((work / "bundles").exists(), "a bundle was sent")
        self.assertFalse((work / "stage.pid").exists(), "the stage ran")
        # Home finished packing before the refusal, as designed.
        self.assertEqual(len(list((self.root / "bundles").glob("*.tar.zst"))), 1)

    # -- row 36: a killed bundle upload keeps its bytes --------------------
    @unittest.skipUnless(shutil.which("zstd"), "zstd is not installed")
    def test_bundle_upload_resumes_after_kill(self):
        stage = "bundle-resume"
        target = "root@127.0.0.1"
        data = self.root / "data"
        data.mkdir()
        payload = os.urandom(40 * 1024 * 1024)
        (data / "big.bin").write_bytes(payload)
        process = self.start(
            stage, 'echo ran > "$VI_OUT/ran.txt"\n',
            VI_INPUT_DATA=str(data), VI_PACK_MIN_KB="0",
            VI_BUNDLE_DIR=str(self.root / "bundles"),
            SSH_STUB_BWLIMIT="2000",
        )
        work = self.wait_worker_work(stage)
        bundles = work / "bundles"
        src_bundle = wait_for(
            "the bundle was never packed", 60,
            lambda: next(iter((self.root / "bundles").glob("*.tar.zst")), None))
        bundle = bundles / src_bundle.name
        src_size = src_bundle.stat().st_size

        def uploading():
            if not bundles.is_dir():
                return 0
            return sum(p.stat().st_size for p in bundles.iterdir() if p.is_file())

        wait_for("the upload never made progress", 120,
                 lambda: uploading() > 0.6 * src_size)
        start = uploading()

        # Stop home first so its EXIT trap cannot delete the worker dir, then
        # kill only the rsync client: its server sees EOF and keeps the
        # partial under the destination name (--partial).
        os.kill(process.pid, signal.SIGSTOP)
        client = None
        for pid in subprocess.run(["pgrep", "-f", f"rsync.*{bundles}"],
                                  capture_output=True, text=True).stdout.split():
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            if b"@" in cmdline and str(bundles).encode() in cmdline:
                client = int(pid)
        self.assertIsNotNone(client, "no rsync client found")
        os.kill(client, signal.SIGTERM)
        wait_for("the partial bundle was not kept", 20,
                 lambda: bundle.is_file() and bundle.stat().st_size >= start)
        kept = bundle.stat().st_size
        self.assertGreater(kept, 0)
        self.assertLess(kept, src_size)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

        # The same push to the same destination resumes instead of restarting.
        env = {k: v for k, v in self.env(stage).items()
               if not k.startswith("SSH_STUB_") or k in ("SSH_STUB_LOG", "SSH_STUB_FORWARDER")}
        result = subprocess.run(
            ["rsync", "-a", "--partial", "--timeout=300", "--stats",
             "-e", str(self.bindir / "ssh"),
             f"{self.root / 'bundles' / bundle.name}", f"{target}:{bundles}/"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        sent = [int(n.replace(",", "")) for n in re.findall(r"Total bytes sent: ([\d,]+)", result.stdout)]
        self.assertTrue(sent, result.stdout[-2000:])
        self.assertLess(sum(sent), 0.7 * src_size, f"resume sent {sum(sent)} bytes")
        self.assertEqual(bundle.stat().st_size, src_size)

    # -- R50: a regenerated worker host key --------------------------------
    def test_rotation_after_regenerated_key_with_matching_log_succeeds_and_sends_no_inputs(self):
        """The worker's key changed; home accepts it through the API and only
        reattaches. The running stage is never relaunched, so the job writes its
        start marker exactly once."""
        stage = "hostkey-rotation"
        vi = self.write_vi_stub()
        process = self.start(
            stage,
            'echo started >> "$VI_OUT/runs.txt"\n'
            'sleep 20\necho done > "$VI_OUT/ok.txt"\n',
            VI_BIN=str(vi),
        )
        wait_for("the first tunnel never proved the path", 20,
                 lambda: "return path re-proved" in self.stderr(stage))
        work = self.wait_worker_work(stage)
        wait_for("the stage never started", 10, lambda: (work / "stage.pid").is_file())
        stage_pid = (work / "stage.pid").read_text().strip()

        # The worker regenerated its key: direct ssh commands now print the
        # changed-key message. The stub `vi cloud hostkey` stands in for the
        # authenticated API check accepting a matching key.
        self.hostkey_flag.write_text("")
        wait_for("the rotation was never checked through the API", 20,
                 lambda: self.rotations() >= 1)
        wait_for("the rotation was not accepted", 20,
                 lambda: "accepted host-key rotation 1" in self.stderr(stage))
        wait_for("the return path was not re-proved after the rotation", 20,
                 lambda: self.stderr(stage).count("return path re-proved") >= 2)

        self.assertEqual((work / "stage.pid").read_text().strip(), stage_pid,
                         "the rotation relaunched the stage")
        self.assertEqual(self.finish(process, 90), 0, self.stderr(stage))
        err = self.stderr(stage)
        self.assertIn("ALERT: the worker's ssh host key changed", err)
        self.assertIn("only a reattach continues, no inputs are sent", err)
        self.assertGreaterEqual(len(self.tunnel_lines()), 2,
                                "the tunnel was not restarted after the rotation")
        self.assertEqual((self.out / "runs.txt").read_text(), "started\n",
                         "a rotation sent new inputs")
        self.assertEqual((self.out / "ok.txt").read_text(), "done\n")

    def test_unlimited_rotations_each_logged(self):
        """A worker may regenerate its key repeatedly; each rotation is checked
        and logged, with a new number and no stage relaunch."""
        stage = "hostkey-many"
        vi = self.write_vi_stub()
        process = self.start(
            stage, 'sleep 30\necho done > "$VI_OUT/ok.txt"\n', VI_BIN=str(vi))
        wait_for("the first tunnel never proved the path", 20,
                 lambda: "return path re-proved" in self.stderr(stage))
        work = self.wait_worker_work(stage)
        wait_for("the stage never started", 10, lambda: (work / "stage.pid").is_file())
        stage_pid = (work / "stage.pid").read_text().strip()

        for n in (1, 2, 3):
            self.hostkey_flag.write_text("")
            wait_for(f"rotation {n} was never checked", 30,
                     lambda n=n: self.rotations() >= n)
            wait_for(f"rotation {n} was not accepted", 30,
                     lambda n=n: f"accepted host-key rotation {n}" in self.stderr(stage))
            wait_for(f"the path after rotation {n} was not re-proved", 30,
                     lambda n=n: self.stderr(stage).count("return path re-proved") >= n + 1)

        self.assertEqual((work / "stage.pid").read_text().strip(), stage_pid,
                         "a rotation relaunched the stage")
        self.assertEqual(self.finish(process, 120), 0, self.stderr(stage))
        err = self.stderr(stage)
        for n in (1, 2, 3):
            self.assertIn(f"accepted host-key rotation {n} for this worker", err)
            self.assertIn("only a reattach continues, no inputs are sent", err)
        self.assertEqual((self.out / "ok.txt").read_text(), "done\n")


if __name__ == "__main__":
    unittest.main()
