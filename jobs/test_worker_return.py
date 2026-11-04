"""Local tests for the worker-initiated rsync return path (no SSH or GPU needed).

The receiver and reverse tunnel run on 127.0.0.1 only, so every test here is
local and disposable. The one OpenSSH-server test is skipped when the tools are
missing; the stub-ssh test works with just a shell and rsync.
"""

import getpass
import hashlib
import os
import re
import shutil
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
        bindir = self.root / "bin"
        bindir.mkdir()
        (bindir / "ssh").write_text(
            "#!/bin/bash\n"
            "set -u\n"
            'for a in "$@"; do\n'
            '  if [ "$a" = -N ]; then echo "$$" > "$SSH_STUB_TUNNEL_PID"; exec sleep 3600; fi\n'
            "done\n"
            "# rsync transport: rsync splits 'root@host' into '-l root <host>' and\n"
            "# appends 'rsync --server ...', so there is no @ to find. Run the server\n"
            "# locally (it writes to the path rsync already names on this machine).\n"
            "i=0\n"
            'for a in "$@"; do\n'
            "  i=$((i + 1))\n"
            '  if [ "$a" = rsync ]; then shift $((i - 1)); exec "$@"; fi\n'
            "done\n"
            "# direct ssh (remote): the command is everything after the root@host target.\n"
            "i=0\n"
            'for a in "$@"; do\n'
            "  i=$((i + 1))\n"
            '  [[ "$a" == *@* ]] && { shift "$i"; break; }\n'
            "done\n"
            'exec bash -c "$*"\n'
        )
        (bindir / "ssh").chmod(0o755)
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

    @unittest.skipUnless(shutil.which("ssh") and shutil.which("sshd") and shutil.which("ssh-keygen"),
                         "OpenSSH client/server tools are not installed")
    def test_reverse_ssh_tunnel_carries_worker_push(self):
        sshd_port, remote_port = free_port(), free_port()
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

        def wait_port(port, process):
            for _ in range(50):
                self.assertIsNone(process.poll(), "SSH process exited before opening its listener")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        return
                except OSError:
                    time.sleep(0.1)
            self.fail(f"port {port} did not open")

        wait_port(sshd_port, sshd)
        with (self.root / "tunnel.log").open("w") as log:
            tunnel = subprocess.Popen([
                shutil.which("ssh"), "-N", "-T", "-p", str(sshd_port),
                "-i", self.root / "client", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=1",
                "-R", f"127.0.0.1:{remote_port}:127.0.0.1:{self.port}", f"{getpass.getuser()}@127.0.0.1",
            ], stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(tunnel)
        wait_port(remote_port, tunnel)
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


if __name__ == "__main__":
    unittest.main()
