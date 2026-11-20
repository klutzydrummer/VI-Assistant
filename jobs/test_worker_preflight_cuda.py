"""The CUDA gate of jobs/worker-preflight.sh names what nvidia-smi really printed when it fails.

omnivoice-ft-r5 attempts 1 and 2 failed with "nvidia-smi reported no CUDA version" and nothing
showed why (most likely a `CUDA Version: N/A` header). A fake nvidia-smi feeds sample headers.
"""

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "worker-preflight.sh"

HEADER = ("+-----------------------------------------------------------------------------+\n"
          "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   {cuda}  |\n"
          "+-----------------------------------------------------------------------------+\n")


class CudaGate(unittest.TestCase):
    def run_gate(self, smi_body, min_cuda="12.8", local_push=False, **extra):
        tmp = Path(tempfile.mkdtemp(prefix="cuda-gate."))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(tmp)]))
        env = dict(os.environ)
        script = SCRIPT
        if local_push:
            # A copy of the script whose push home is a stub, so a host that passes every gate
            # finishes and writes preflight.json without any network.
            jobs = tmp / "jobs"
            jobs.mkdir()
            shutil.copy(SCRIPT, jobs / SCRIPT.name)
            shutil.copy(SCRIPT.parent / "worker-limits.sh", jobs / "worker-limits.sh")
            (jobs / "voicetune.py").write_text("def parse_nvidia_smi_csv(text):\n    return []\n")
            ret = jobs / "worker-return.sh"
            ret.write_text("#!/usr/bin/env bash\nexit 0\n")
            ret.chmod(0o755)
            script = jobs / SCRIPT.name
        if smi_body is not None:
            fake = tmp / "bin"
            fake.mkdir()
            smi = fake / "nvidia-smi"
            smi.write_text("#!/usr/bin/env bash\ncase \"$*\" in\n"
                           "  *--query-gpu=memory.total*) echo 24467 ;;\n"
                           "  *--query-gpu*) echo x ;;\n"
                           f"  *) cat <<'EOF'\n{smi_body}EOF\n ;;\nesac\n")
            smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
            env["PATH"] = f"{fake}:{env['PATH']}"
        else:  # a PATH with no nvidia-smi at all
            bindir = tmp / "bin"
            bindir.mkdir()
            for tool in ("python3", "bash", "env", "dirname", "awk", "nproc", "df", "setsid", "flock",
                         "head", "sed", "grep", "tr", "cat", "date", "dd"):
                path = subprocess.run(["which", tool], capture_output=True, text=True).stdout.strip()
                if path:
                    (bindir / tool).symlink_to(path)
            env["PATH"] = str(bindir)
        (tmp / "job" / "out").mkdir(parents=True)
        settings = tmp / "settings.json"
        settings.write_text(json.dumps({"min_cuda_version": min_cuda}))
        env.update(VI_OUT=str(tmp / "job" / "out"), VI_SETTINGS=str(settings), VI_RETURN_PORT="1")
        env.update(extra)
        self.out = tmp / "job" / "out"
        return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True,
                              timeout=60)

    def test_na_header_is_named_and_still_rejected(self):
        r = self.run_gate(HEADER.format(cuda="CUDA Version: N/A"))
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("reported no CUDA version", r.stderr)
        self.assertIn("CUDA Version: N/A", r.stderr)
        self.assertIn("reads N/A", r.stderr)

    def test_missing_header_shows_the_first_lines(self):
        r = self.run_gate(HEADER.format(cuda="(no cuda field)"))
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("no 'CUDA Version' or 'CUDA UMD Version' line", r.stderr)
        self.assertIn("NVIDIA-SMI 550.54.15", r.stderr)
        self.assertNotIn("reads N/A", r.stderr)

    def test_no_nvidia_smi_is_said_plainly(self):
        r = self.run_gate(None)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("nvidia-smi is not installed", r.stderr)

    def test_old_cuda_is_still_rejected_by_number(self):
        r = self.run_gate(HEADER.format(cuda="CUDA Version: 12.4"))
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("CUDA version is 12.4, under the required 12.8", r.stderr)

    def test_13_1_is_parsed_as_a_number(self):
        # A passing host would run on into the 100 MiB push, so prove the parse by a floor above it.
        r = self.run_gate(HEADER.format(cuda="CUDA Version: 13.1"), min_cuda="14.0")
        self.assertIn("CUDA version is 13.1, under the required 14.0", r.stderr)

    # Driver 615 prints the CUDA version as `CUDA UMD Version:` beside a `KMD Version:` field;
    # omnivoice-ft-r8 attempt 1 falsely rejected a CUDA 13.4 host over it. This is its exact header.
    R8_HEADER = ("+-----------------------------------------------------------------------------+\n"
                 "| NVIDIA-SMI 615.71.09  KMD Version: 615.71.09  CUDA UMD Version: 13.4  |\n"
                 "+-----------------------------------------------------------------------------+\n")

    def test_umd_header_passes_a_lower_floor(self):
        r = self.run_gate(self.R8_HEADER, min_cuda="12.8", local_push=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("reported no CUDA version", r.stderr)
        self.assertIn("worker-preflight: ok", r.stderr)
        self.assertEqual(json.loads((self.out / "preflight.json").read_text())["cuda_version"], 13.4)

    def test_umd_header_fails_a_higher_floor_by_number(self):
        r = self.run_gate(self.R8_HEADER, min_cuda="13.5")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("CUDA version is 13.4, under the required 13.5", r.stderr)

    def test_umd_na_header_is_named_and_rejected(self):
        r = self.run_gate(self.R8_HEADER.replace("13.4", "N/A"))
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("raw header: | NVIDIA-SMI 615.71.09  KMD Version: 615.71.09  CUDA UMD Version: N/A",
                      r.stderr)
        self.assertIn("reads N/A", r.stderr)

    def test_classic_header_passes_a_lower_floor(self):
        r = self.run_gate(HEADER.format(cuda="CUDA Version: 12.8"), local_push=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads((self.out / "preflight.json").read_text())["cuda_version"], 12.8)


if __name__ == "__main__":
    unittest.main()


class TasksRow(unittest.TestCase):
    """The tasks row refuses a box with too few free task slots (r7: pids.max counts threads)."""

    def run_row(self, pids_max, pids_current, threads_max="4000000", ulimit_u=None):
        tmp = Path(tempfile.mkdtemp(prefix="tasks-row."))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(tmp)]))
        cg = tmp / "cgroup"
        cg.mkdir()
        (cg / "pids.max").write_text(f"{pids_max}\n")
        (cg / "pids.current").write_text(f"{pids_current}\n")
        (cg / "cpu.max").write_text("1100000 100000\n")
        (cg / "cpuset.cpus.effective").write_text("0-47\n")
        proc = tmp / "proc"
        (proc / "sys" / "kernel").mkdir(parents=True)
        (proc / "loadavg").write_text("0.10 0.10 0.10 1/50 999\n")
        (proc / "sys" / "kernel" / "threads-max").write_text(f"{threads_max}\n")
        gate = CudaGate("test_na_header_is_named_and_still_rejected")
        gate.addCleanup = self.addCleanup
        # A CUDA 13.1 header against a 14.0 floor: a box that passes the tasks row stops at the
        # CUDA gate, so the test never reaches the network push.
        return gate.run_gate(HEADER.format(cuda="CUDA Version: 13.1"), min_cuda="14.0",
                             CGROUP_ROOT=str(cg), VI_LIMITS_PROC=str(proc), VI_LIMITS_NPROC="48")

    def test_too_few_task_slots_fail_the_tasks_row(self):
        r = self.run_row(pids_max=60, pids_current=30)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertRegex(r.stderr, r"(?m)^worker-preflight: tasks: only 30 free task slots "
                                   r"\(pids\.max=60 current=30 .*\), need at least 40 for 5 processes$")
        self.assertRegex(r.stderr, r"(?m)^limits: nproc=48 cpus=11 quota=11 cpuset=48 pids\.max=60 ")
        self.assertNotIn("CUDA version", r.stderr)

    def test_low_threads_max_also_fails_the_tasks_row(self):
        r = self.run_row(pids_max="max", pids_current=5, threads_max="70")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("worker-preflight: tasks: only 20 free task slots", r.stderr)

    def test_enough_task_slots_pass_on_to_the_next_gate(self):
        r = self.run_row(pids_max=2800, pids_current=100)
        self.assertNotIn("worker-preflight: tasks:", r.stderr)
        self.assertRegex(r.stderr, r"(?m)^limits: .* tasks_free=2700 -> threads=2 nj=2$")
        self.assertIn("CUDA version is 13.1, under the required 14.0", r.stderr)
