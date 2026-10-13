"""Tests for steer.py, with fake models. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import steer
from vi_job import Clip


def clip(i, seconds=4.0, flags=()):
    return Clip(id=f"c{i}", text=f"secret line {i}", audio=Path(f"/x/c{i}.wav"), duration_s=seconds, flags=list(flags))


class PickRefsTest(unittest.TestCase):
    def test_top_clips_per_axis_and_calmest_for_neutral(self):
        clips = {c.id: c for c in [clip(i) for i in range(10)]}
        labels = {f"c{i}": {"anger": i / 10, "intoxication": (9 - i) / 10} for i in range(10)}
        refs = steer.pick_refs(clips, labels, ["anger"], 2, {})
        self.assertEqual([c.id for c in refs["anger"]], ["c9", "c8"])
        self.assertEqual([c.id for c in refs["neutral"]], ["c0", "c1"])

    def test_listed_ids_win_and_bad_clips_are_skipped(self):
        clips = {c.id: c for c in [clip(0, 1.0), clip(1, flags=["far_voice"]), clip(2), clip(3), clip(4)]}
        labels = {i: {"anger": 1.0} for i in clips}
        refs = steer.pick_refs(clips, labels, ["anger"], 5, {"neutral": ["c2"]})
        self.assertEqual({c.id for c in refs["anger"]}, {"c2", "c3", "c4"})
        self.assertEqual([c.id for c in refs["neutral"]], ["c2"])


class WerTest(unittest.TestCase):
    def test_case_and_punctuation_are_ignored(self):
        self.assertEqual(steer.wer("The scanner, picked it up.", "the scanner picked it up"), 0.0)
        self.assertAlmostEqual(steer.wer("one two three four", "one two four"), 0.25)


class FakeRunTest(unittest.TestCase):
    def test_runs_end_to_end_and_writes_no_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, label, out, ck = tmp / "harvest", tmp / "label", tmp / "out", tmp / "ck"
            for d in (harvest / "audio", label):
                d.mkdir(parents=True)
            lines, labels = [], []
            for i in range(30):
                cid = f"le3.s{i}_f"
                path = harvest / "audio" / f"{cid}.wav"
                steer.write_wav(path, steer.fake_wave(cid, seconds=0.1))
                lines.append({
                    "id": cid, "speaker": "tali", "text": f"Secret subtitle {i}",
                    "audio": {"path": f"audio/{cid}.wav", "sample_rate": 16000, "channels": 1, "duration_s": 3.0 + i % 4},
                    "source": {"game": "le3"},
                })  # fmt: skip
                labels.append({"id": cid, "scores": {"anger": {"value": i / 30}, "intoxication": {"value": (i * 7 % 30) / 30}}})
            (harvest / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
            (label / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
            (tmp / "settings.json").write_text(json.dumps({"backend": "fake", "refs_per_side": 4}))
            env = {
                "VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out), "VI_CHECKPOINT": str(ck),
                "VI_INPUT_HARVEST": str(harvest), "VI_INPUT_LABEL": str(label),
            }  # fmt: skip
            with mock.patch.dict(os.environ, env):
                steer.main()
                first = (out / "report.json").read_text()
                steer.main()  # a rerun reuses the checkpoint and gives the same report
                self.assertEqual((out / "report.json").read_text(), first)

            report = json.loads(first)
            self.assertEqual(set(report["verdicts"]), {"anger", "intoxication"})
            for a in ("anger", "intoxication"):
                sources = report["verdicts"][a]["sources"]
                self.assertTrue(sources["designed"]["steers"], a)
                self.assertFalse(sources["tali"]["steers"], a)
                self.assertIn("designed vectors", report["verdicts"][a]["verdict"])
            self.assertEqual(report["references"]["anger"][0], "le3.s29_f")
            n = len(steer.TEST)
            self.assertEqual(len(list((out / "audio").rglob("*.wav"))), n * (1 + 2 * (1 + 2 * 4)))
            self.assertTrue((out / "vectors" / "tali-anger.npy").exists())
            for name in ("report.json", "report.md", "listen.md"):
                self.assertNotIn("Secret subtitle", (out / name).read_text(), name)


if __name__ == "__main__":
    unittest.main()
