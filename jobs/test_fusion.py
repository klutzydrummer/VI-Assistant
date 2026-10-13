"""python3 -m unittest discover -s jobs"""

import os
import struct
import tempfile
import unittest
from pathlib import Path

import fusion
import models
import vi_job


class Fuse(unittest.TestCase):
    def test_agreeing_signals_are_confident(self):
        r = fusion.fuse({"audio": (0.8, 0.8), "text": (0.75, 0.9), "prior": None}, fusion.DEFAULT_WEIGHTS)
        self.assertAlmostEqual(r["value"], 0.774, places=2)
        self.assertGreater(r["confidence"], 0.7)
        self.assertEqual(r["signals"], {"audio": 0.8, "text": 0.75, "prior": None})

    def test_disagreeing_signals_are_not(self):
        r = fusion.fuse({"audio": (0.9, 0.8), "text": (0.05, 0.9)}, fusion.DEFAULT_WEIGHTS)
        self.assertLess(r["confidence"], 0.2)

    def test_no_signals(self):
        r = fusion.fuse({"audio": None, "text": None}, fusion.DEFAULT_WEIGHTS)
        self.assertEqual((r["value"], r["confidence"]), (0.0, 0.0))

    def test_audio_map(self):
        probs = {"angry": 0.4, "disgusted": 0.4, "neutral": 0.2}
        value, confidence = fusion.audio_signal(probs, "anger", fusion.DEFAULT_AUDIO_MAP, 2.0)
        self.assertAlmostEqual(value, 0.6)
        self.assertEqual(confidence, 0.8)
        self.assertIsNone(fusion.audio_signal(probs, "intoxication", fusion.DEFAULT_AUDIO_MAP, 2.0))
        self.assertEqual(fusion.audio_signal(probs, "anger", fusion.DEFAULT_AUDIO_MAP, 0.5)[1], 0.4)

    def test_priors(self):
        p = fusion.priors(["citadel_party"], "Quiet sad delivery. Not flippant here.",
                          fusion.DEFAULT_SCENE_PRIORS, fusion.DEFAULT_DIRECTION_PRIORS)  # fmt: skip
        self.assertEqual(p["sadness"], (0.75, fusion.DIRECTION_PRIOR_CONFIDENCE))
        self.assertEqual(p["intoxication"], (0.5, fusion.SCENE_PRIOR_CONFIDENCE))
        self.assertNotIn("anger", p)

    def test_tags(self):
        self.assertEqual(fusion.tags_for(["bark"], ["effort"], 0.6), ["combat_bark", "effort", "short"])
        self.assertEqual(fusion.tags_for(["nonverbal"], [], 2.0), ["nonverbal"])

    def test_spearman(self):
        xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, None]
        self.assertEqual(fusion.spearman(xs, xs), 1.0)
        self.assertEqual(fusion.spearman(xs, [-x if x else None for x in xs]), -1.0)
        self.assertIsNone(fusion.spearman(xs[:3], xs[:3]))


class Replies(unittest.TestCase):
    def test_parse_json(self):
        self.assertEqual(models.parse_json('```json\n{"a": {"b": 1}}\n```'), {"a": {"b": 1}})
        self.assertEqual(models.parse_json('Sure! {"a": 1} and {"b": 2}'), {"a": 1})
        self.assertEqual(models.parse_json('{broken {"a": 1}'), {"a": 1})
        self.assertIsNone(models.parse_json("no json here"))

    def test_final_answer(self):
        self.assertEqual(models.final_answer('<|channel>thought\nhmm<channel|>{"a": 1}<turn|>'), '{"a": 1}')
        self.assertEqual(models.final_answer('{"a": 1}<eos>'), '{"a": 1}')


class Embeddings(unittest.TestCase):
    def test_npy(self):
        with tempfile.TemporaryDirectory() as d:
            info = vi_job.write_embeddings(Path(d), "e", "m", {"a": [1.0, 2.0, 3.0], "b": [0.5, -1.0, 0.0]})
            self.assertEqual(info, {"model": "m", "clips": 2, "dim": 3})
            self.assertEqual((Path(d) / "e.ids").read_text(), "a\nb\n")
            raw = (Path(d) / "e.npy").read_bytes()
            n = struct.unpack("<H", raw[8:10])[0]
            self.assertEqual((10 + n) % 64, 0)
            self.assertIn("'shape': (2, 3)", raw[10 : 10 + n].decode())
            self.assertEqual(struct.unpack("<6f", raw[10 + n :]), (1.0, 2.0, 3.0, 0.5, -1.0, 0.0))
            try:
                import numpy as np
            except ImportError:
                return
            self.assertEqual(np.load(Path(d) / "e.npy").tolist(), [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])


class Checkpoints(unittest.TestCase):
    def test_resumes_and_drops_a_torn_line(self):
        clips = [vi_job.Clip(id=f"c{i}", text="", audio="", duration_s=1.0) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            os.environ["VI_CHECKPOINT"] = d
            try:
                seen = []

                def compute(part):
                    seen.extend(c.id for c in part)
                    if "c3" in seen:
                        raise RuntimeError("worker lost")
                    return [[c.id, 1] for c in part]

                with self.assertRaises(RuntimeError):
                    vi_job.Checkpoint("x").run(clips, compute, chunk=2)
                path = Path(d) / "x.jsonl"
                path.write_text(path.read_text() + '{"id": "c2", "v"')  # copied mid-write
                done = vi_job.Checkpoint("x").run(clips, lambda part: [[c.id, 2] for c in part])
                self.assertEqual(done, {"c0": ["c0", 1], "c1": ["c1", 1], "c2": ["c2", 2],
                                        "c3": ["c3", 2], "c4": ["c4", 2]})  # fmt: skip
                self.assertEqual(len(path.read_text().splitlines()), 5)
            finally:
                del os.environ["VI_CHECKPOINT"]


if __name__ == "__main__":
    unittest.main()
