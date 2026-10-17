"""Tests for progress.py. Run with `python3 -m unittest discover -s jobs`."""

import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import progress


def stage(corpus, name, lines, age_s=0):
    log = corpus / "stages" / name / ".tmp-abc" / "log.txt"
    log.parent.mkdir(parents=True)
    log.write_text("".join(f"{x}\n" for x in lines))
    t = time.time() - age_s
    os.utime(log, (t, t))


class ProgressTest(unittest.TestCase):
    def test_passes_running_and_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            corpus = Path(d)
            stage(corpus, "embed", ["embed: 3516 clips", "wavlm: 50/100", "wavlm: 100/100",
                                    "emotion2vec_plus: 25/100"])  # fmt: skip
            stage(corpus, "clean", ["clean: 10 clips", "asr: 3/10"], age_s=3600)
            buf = io.StringIO()
            with redirect_stdout(buf):
                seen = progress.report(corpus)
            out = buf.getvalue()
            self.assertIn("embed: running", out)
            self.assertIn("wavlm: 100/100 (100%)", out)
            self.assertIn("emotion2vec_plus: 25/100 (25%)  <- now", out)
            self.assertIn("clean: stopped 60 min ago", out)
            self.assertEqual(seen[("embed", "emotion2vec_plus")], 25)

            # A second read 60 s later at 55/100: 30 per minute, so 1.5 min left.
            stage_log = corpus / "stages" / "embed" / ".tmp-abc" / "log.txt"
            stage_log.write_text(stage_log.read_text() + "emotion2vec_plus: 55/100\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                progress.report(corpus, seen, 60)
            self.assertIn("55/100 (55%), about 2 min left in this pass", buf.getvalue())

    def test_nothing_running(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "stages").mkdir()
            buf = io.StringIO()
            with redirect_stdout(buf):
                progress.report(Path(d))
            self.assertIn("no stage is running", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
