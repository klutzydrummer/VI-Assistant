"""Tests for checks.py. Run with `python3 -m unittest discover -s jobs`."""

import json
import tempfile
import unittest
from pathlib import Path

import checks


class ChecksTest(unittest.TestCase):
    def test_eta_squared(self):
        self.assertEqual(checks.eta_squared([1, 1, 3, 3], ["a", "a", "b", "b"]), 1.0)
        self.assertEqual(checks.eta_squared([1, 3, 1, 3], ["a", "a", "b", "b"]), 0.0)
        self.assertIsNone(checks.eta_squared([1, 1, 1], ["a", "b", "c"]))

    def test_finish_fails_only_in_fail_mode(self):
        bad = [checks.Check("x", 0.9, 0.5), checks.Check("y", 0.1, 0.5), checks.Check("z", None, 0.5)]
        with tempfile.TemporaryDirectory() as d:
            checks.finish(Path(d), bad, "warn")
            report = json.loads((Path(d) / "checks.json").read_text())
            self.assertEqual(report["failed"], 1)
            with self.assertRaises(SystemExit):
                checks.finish(Path(d), bad, "fail")


if __name__ == "__main__":
    unittest.main()
