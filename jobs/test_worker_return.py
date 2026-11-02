"""Local tests for the worker-initiated rsync return path (no SSH or GPU needed).

The receiver and reverse tunnel run on 127.0.0.1 only, so every test here is
local and disposable. The one OpenSSH-server test is skipped when the tools are
missing; the stub-ssh test works with just a shell and rsync.
"""

import getpass
import os
import shutil
import socket
import subprocess
import tempfile
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
        self.state = self.root / "state"
        self.src = self.root / "src"
        self.processes = []
        (self.src / "out").mkdir(parents=True)
        (self.src / "checkpoint").mkdir()
        started = subprocess.run(
            [SCRIPT, "start", self.out, self.checkpoint, self.state],
            check=True, capture_output=True, text=True,
        )
        self.port = started.stdout.strip()
        self.env = {**os.environ, "VI_RETURN_PORT": self.port}

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

    def push(self, source, module, *options, check=True):
        return subprocess.run(
            [SCRIPT, "push", source, module, *options],
            env=self.env, check=check, capture_output=True, text=True,
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
        # OUT/CHECKPOINT/STATE to absolute paths before writing the config.
        out = self.root / "rel-out"
        checkpoint = self.root / "rel-checkpoint"
        state = self.root / "rel-state"
        started = subprocess.run(
            [SCRIPT, "start", "rel-out", "rel-checkpoint", "rel-state"],
            cwd=self.root, check=True, capture_output=True, text=True,
        )
        try:
            port = started.stdout.strip()
            conf = (state / "rsyncd.conf").read_text()
            self.assertIn(f"path = {out}\n", conf)
            self.assertIn(f"path = {checkpoint}\n", conf)
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

    def test_receiver_refuses_download_and_delete(self):
        (self.out / "keep.txt").write_text("keep\n")
        # `refuse options = delete` covers --delete and its --delete-* variants:
        # the daemon rejects the request (rc 13) before any file is removed.
        for option in ("--delete", "--delete-after"):
            with self.subTest(option=option):
                result = self.push(self.src / "out", "vi-out", option, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((self.out / "keep.txt").read_text(), "keep\n")

        download = subprocess.run(
            ["rsync", "-a", f"rsync://127.0.0.1:{self.port}/vi-out/keep.txt", str(self.root / "download.txt")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(download.returncode, 0)
        self.assertFalse((self.root / "download.txt").exists())

    def test_module_list_is_empty(self):
        result = subprocess.run(
            ["rsync", f"rsync://127.0.0.1:{self.port}/"],
            capture_output=True, text=True,
        )
        self.assertNotIn("vi-out", result.stdout)
        self.assertNotIn("vi-checkpoint", result.stdout)

    def test_second_concurrent_connection_is_refused(self):
        # Hold the single connection slot with a slow transfer, then a second
        # connection must be refused (max connections = 1), not queued or
        # serialized. --bwlimit throttles even on loopback so the first push
        # stays alive long enough to overlap the second.
        big = self.src / "out" / "big.bin"
        big.write_bytes(os.urandom(4 * 1024 * 1024))
        first = subprocess.Popen(
            ["rsync", "-a", "--bwlimit=500", str(self.src / "out") + "/", f"rsync://127.0.0.1:{self.port}/vi-out/"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.5)
            self.assertIsNone(first.poll(), "--bwlimit was ignored; the first transfer already finished")
            result = subprocess.run(
                ["rsync", "-a", str(self.src / "out") + "/", f"rsync://127.0.0.1:{self.port}/vi-out/"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("max connections", result.stderr)
        finally:
            first.terminate()
            first.wait(timeout=5)

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
        secret = self.root / "secret.txt"
        secret.write_text("outside\n")
        (self.src / "out" / "link").symlink_to(secret)
        self.push(self.src / "out", "vi-out")
        link = self.out / "link"
        self.assertTrue(link.is_symlink())
        self.assertTrue(os.readlink(link).startswith("/rsyncd-munged/"))
        with self.assertRaises(OSError):
            link.read_text()

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
