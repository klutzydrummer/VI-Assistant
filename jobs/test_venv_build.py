"""The OmniVoice venv is built the real way and its console scripts work afterwards.

omnivoice-ft-r8 attempt 2 died with exit 126: voicetune-fresh.sh built the venv under
`venv.tmp-PID` and renamed it, so pip's console scripts kept a shebang to the old, gone path.
These tests run the script's own ovft_python() with a real `python3 -m venv` and a real pip
installing a tiny local wheel that has a console script (no network: --no-index).
"""

import base64
import hashlib
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

JOBS = Path(__file__).resolve().parent
SCRIPT = JOBS / "voicetune-fresh.sh"


def build_wheel(dest):
    """A minimal pure-Python wheel `tinycli` whose console script prints a fixed line."""
    files = {
        "tinycli/__init__.py": "def main():\n    print('tinycli ok')\n",
        "tinycli-0.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: tinycli\nVersion: 0.1\n",
        "tinycli-0.1.dist-info/WHEEL": ("Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                                       "Tag: py3-none-any\n"),
        "tinycli-0.1.dist-info/entry_points.txt": "[console_scripts]\ntinycli = tinycli:main\n",
    }
    record = []
    for name, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(text.encode())}")
    record.append("tinycli-0.1.dist-info/RECORD,,")
    files["tinycli-0.1.dist-info/RECORD"] = "\n".join(record) + "\n"
    wheel = dest / "tinycli-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return wheel


class RealVenvBuild(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.jobs = self.root / "jobs"
        self.jobs.mkdir()
        wheel = build_wheel(self.root)
        (self.jobs / "requirements-omnivoice-worker.txt").write_text(f"--no-index\n{wheel}\n")
        self.home = self.root / "home"
        self.home.mkdir()
        # Only the function under test, taken verbatim from the script.
        text = SCRIPT.read_text()
        start = text.index("ovft_python() {")
        self.func = text[start:text.index("\n}\n", start) + 3]

    def ovft_python(self):
        """Run ovft_python() as voicetune-fresh.sh does (OVFT_PYTHON unset); return its python path."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("OVFT_PYTHON", "PIP_INDEX_URL", "PIP_FIND_LINKS", "VIRTUAL_ENV")}
        env.update(HOME=str(self.home), VI_PIP_BACKOFF="0 0", PIP_DISABLE_PIP_VERSION_CHECK="1")
        r = subprocess.run(["bash", "-c", f'set -u\njobs_dir="$1"\n{self.func}\nPY=$(ovft_python) || exit 1\n'
                            'printf "%s\\n" "$PY"', "_", str(self.jobs)],
                           env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        return Path(r.stdout.strip().splitlines()[-1])

    def test_console_script_runs_and_has_no_tmp_path(self):
        py = self.ovft_python()
        venv = py.parent.parent
        self.assertTrue(venv.name.startswith("venv-"), venv)
        self.assertNotIn(".tmp-", str(venv))
        script = venv / "bin" / "tinycli"
        r = subprocess.run([str(script)], capture_output=True, text=True, timeout=60)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "tinycli ok"), r.stderr)
        self.assertEqual(script.read_text().splitlines()[0], f"#!{venv}/bin/python")
        for f in (venv / "bin").iterdir():
            if f.is_file() and not f.is_symlink():
                self.assertNotIn(b".tmp-", f.read_bytes(), f)
        self.assertTrue((venv / ".vi-complete").is_file())
        self.assertEqual([p.name for p in venv.parent.glob("venv-*.tmp-*")], [])

    def test_complete_venv_is_reused(self):
        venv = self.ovft_python().parent.parent
        (venv / "sentinel").write_text("kept")
        self.assertEqual(self.ovft_python().parent.parent, venv)
        self.assertTrue((venv / "sentinel").exists(), "a complete venv must not be rebuilt")

    def test_folder_without_marker_is_removed_and_rebuilt(self):
        venv = self.ovft_python().parent.parent
        (venv / ".vi-complete").unlink()
        (venv / "half-built").write_text("x")
        self.assertEqual(self.ovft_python().parent.parent, venv)
        self.assertFalse((venv / "half-built").exists(), "a venv without the marker must be rebuilt")
        self.assertTrue((venv / ".vi-complete").is_file())
        r = subprocess.run([str(venv / "bin" / "tinycli")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
