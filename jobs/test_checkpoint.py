"""Tests for vi_job.Checkpoint. Run with `python3 -m unittest discover -s jobs`."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vi_job import Checkpoint


def items(*ids):
    return [SimpleNamespace(id=i) for i in ids]


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        os.environ["VI_CHECKPOINT"] = self.dir.name

    def tearDown(self):
        os.environ.pop("VI_CHECKPOINT", None)
        self.dir.cleanup()

    def test_a_rerun_computes_only_what_is_missing(self):
        calls = []

        def compute(part):
            calls.extend(c.id for c in part)
            return [c.id.upper() for c in part]

        first = Checkpoint("step").run(items("a", "b"), compute)
        self.assertEqual(first, {"a": "A", "b": "B"})
        again = Checkpoint("step").run(items("a", "b", "c"), compute, chunk=2)
        self.assertEqual(again, {"a": "A", "b": "B", "c": "C"})
        self.assertEqual(calls, ["a", "b", "c"])

    def test_a_crash_keeps_finished_chunks(self):
        def boom(part):
            if "c" in [c.id for c in part]:
                raise RuntimeError("worker lost")
            return [1 for _ in part]

        with self.assertRaises(RuntimeError):
            Checkpoint("step").run(items("a", "b", "c", "d"), boom, chunk=2)
        self.assertEqual(Checkpoint("step").done, {"a": 1, "b": 1})

    def test_a_cut_off_last_line_is_ignored_and_redone(self):
        path = Path(self.dir.name) / "step.jsonl"
        path.write_text('{"id": "a", "value": null}\n{"id": "b", "val')
        saved = Checkpoint("step")
        self.assertEqual(saved.done, {"a": None})
        saved.run(items("a", "b"), lambda part: [2 for _ in part])
        self.assertEqual(Checkpoint("step").done, {"a": None, "b": 2})

    def test_without_vi_checkpoint_nothing_is_written(self):
        os.environ.pop("VI_CHECKPOINT")
        out = Checkpoint("step").run(items("a"), lambda part: [3])
        self.assertEqual(out, {"a": 3})
        self.assertEqual(list(Path(self.dir.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
