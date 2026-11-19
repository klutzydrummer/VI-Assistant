"""The CUDA gate of jobs/worker-preflight.sh names what nvidia-smi really printed when it fails.

omnivoice-ft-r5 attempts 1 and 2 failed with "nvidia-smi reported no CUDA version" and nothing
showed why (most likely a `CUDA Version: N/A` header). A fake nvidia-smi feeds sample headers.
"""

import json
import os
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
    def run_gate(self, smi_body, min_cuda="12.8"):
        tmp = Path(tempfile.mkdtemp(prefix="cuda-gate."))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(tmp)]))
        env = dict(os.environ)
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
        return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
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
        self.assertIn("no 'CUDA Version' line", r.stderr)
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


if __name__ == "__main__":
    unittest.main()
