"""Tests for refscreen.py, with fake models. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import refscreen
from test_voicearms import corpus


class FakeRunTest(unittest.TestCase):
    def test_ranks_holding_candidates_and_writes_blind_shortlists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files = corpus(tmp)
            out, ck = tmp / "out", tmp / "ck"
            (tmp / "settings.json").write_text(json.dumps({
                "backend": "fake", "moods": ["warmth", "teasing"], "shortlist": 2, "shortlist_weak": 3, "controls": 1}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out), "VI_CHECKPOINT": str(ck),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env):
                refscreen.main()
                first = (out / "report.json").read_text()
                refscreen.main()  # a rerun reuses the checkpoint
                self.assertEqual((out / "report.json").read_text(), first)
            report = json.loads(first)
            self.assertEqual(report["sentences"], [0, 1, 2, 3])
            for mood in ("warmth", "teasing"):
                x = report["moods"][mood]
                self.assertGreater(x["candidates"], 0, mood)
                self.assertEqual(x["candidates"], x["holding"] + len(x["dropped"]))
                lifts = [c["lift"] for c in x["ranked"]]
                self.assertEqual(lifts, sorted(lifts, reverse=True))
                self.assertEqual([c["rank"] for c in x["ranked"]], list(range(1, len(lifts) + 1)))
                for c in x["ranked"]:
                    self.assertTrue((out / "audio" / mood / c["clip"] / "3.wav").exists())
                    # her pairs' positives only, outside the held-out conversations
                    self.assertIn("pos", c["clip"])
                d = out / "shortlist" / mood
                f = tomllib.loads((d / "clips.toml").read_text())
                self.assertTrue(f["blind"])
                self.assertEqual(f["answers"], ["0", "1", "2", "3"])
                refs = [c for c in f["clip"] if "id" in c]
                want = min(len(x["ranked"]), (3 if x["probe_weak"] else 2) + 1)
                self.assertEqual(len(refs), want, mood)
                gen = [c for c in f["clip"] if "path" in c]
                self.assertEqual(len(gen), 2 * len(refs))  # two sentences per candidate
                self.assertTrue(all((d / c["path"]).exists() for c in gen))
            self.assertIn("Reference screening", (out / "report.md").read_text())
            self.assertNotIn("Secret subtitle", (out / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
