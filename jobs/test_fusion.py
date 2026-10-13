"""python3 -m unittest discover -s jobs"""

import unittest

import fusion
import models


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


if __name__ == "__main__":
    unittest.main()
