"""Tests for codectrip.py and rvctest.py, with fake models. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import codectrip
import rvctest
import steer
from test_voicearms import corpus


class CodecTripTest(unittest.TestCase):
    def test_picks_strong_suit_clips_and_pairs_them_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files = corpus(tmp)
            out = tmp / "out"
            (tmp / "settings.json").write_text(json.dumps({"backend": "fake", "clips": 4, "per_game": 3}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env):
                codectrip.main()
            rep = json.loads((out / "report.json").read_text())
            self.assertEqual(len(rep["clips"]), 3)  # one game in the fake corpus, at most 3 from it
            strengths = [r["suit_strength_db"] for r in rep["clips"]]
            self.assertEqual(strengths, sorted(strengths, reverse=True))
            self.assertTrue(all(r["suit_gap_db"] == 0.0 for r in rep["clips"]))  # the fake codec copies
            hub = tomllib.loads((out / "clips.toml").read_text())
            self.assertTrue(hub["blind"])
            self.assertEqual(len(hub["clip"]), 6)
            self.assertEqual({c.get("id") for c in hub["clip"] if "id" in c}, {r["clip"] for r in rep["clips"]})
            self.assertTrue(all((out / c["path"]).exists() for c in hub["clip"] if "path" in c))


class RvcTestTest(unittest.TestCase):
    def test_collects_inputs_and_writes_a_blind_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            va = tmp / "va"
            def wav(*parts):
                p = va.joinpath("audio", *parts, "8.wav")
                p.parent.mkdir(parents=True, exist_ok=True)
                steer.write_wav(p, steer.fake_wave(str(p), seconds=0.05))
            wav("_neutral", "N-icl")
            wav("example_clips", "E1")
            wav("alarm", "S")
            wav("alarm", "AL21@1")
            wav("alarm", "Dicl")
            (va / "report.json").write_text(json.dumps({
                "example_trials": {"E1": {"clip": "le3.z1_f", "audio": "E1"}},
                "moods": {"alarm": {"arms": {"A": {"audio": "AL21@1", "strength": 1.0}, "D-icl": {"audio": "Dicl", "strength": 0.5}},
                                    "verdicts": {"A": "steers", "D-icl": "doesn't steer"}}}}))
            harvest = tmp / "harvest"
            (harvest / "audio").mkdir(parents=True)
            steer.write_wav(harvest / "audio" / "s.wav", steer.fake_wave("s", seconds=0.05))
            (harvest / "lines.jsonl").write_text(json.dumps(
                {"id": "le3.s_f", "speaker": "tali", "audio": {"path": "audio/s.wav", "duration_s": 1.5}}) + "\n")
            out = tmp / "out"
            rvctest.main(["--voicearms", str(va), "--harvest", str(harvest), "--out", str(out),
                          "--model", str(tmp / "Tali.pth"), "--fake"])
            hub = tomllib.loads((out / "clips.toml").read_text())
            labels = [c["label"] for c in hub["clip"]]
            # neutral, E1, alarm S, alarm A (steers; D-icl doesn't), one short real line: plain and RVC each.
            self.assertEqual(len(labels), 10)
            self.assertEqual(sum(l.endswith("| plain") for l in labels), 5)
            self.assertFalse(any("D-icl" in l for l in labels))
            self.assertTrue(any("her real short line le3.s_f" in l for l in labels))
            self.assertIn("Tali:", (out / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
