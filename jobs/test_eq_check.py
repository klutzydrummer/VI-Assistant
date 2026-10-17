"""Tests for eq_check.py on synthetic vectors. Needs numpy and scikit-learn.
Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np

import eq_check


def scores(avd, classes):
    return [{"avd": {"arousal": float(a)}, "emotion2vec_plus_large_classes": {"angry": float(c)}}
            for a, c in zip(avd, classes)]  # fmt: skip


class CompareTest(unittest.TestCase):
    def test_removing_a_game_offset_shows_as_a_lower_probe_and_kept_readings(self):
        rng = np.random.default_rng(0)
        n = 300
        game = np.array(["le1", "le2", "le3"] * (n // 3))
        group = np.array([f"c{i // 2}" for i in range(n)])  # two clips per conversation
        emotion = rng.standard_normal((n, 16))
        offset = {"le1": 2.0, "le2": 0.0, "le3": -2.0}
        shift = np.array([offset[g] for g in game])[:, None] * np.eye(16)[0]
        before, after = emotion + shift + 5, emotion + 5  # the EQ takes the game's offset away
        arousal = rng.random(n)
        r = eq_check.compare(list(range(n)), game, group, before, after,
                             scores(arousal + shift[:, 0] / 10, arousal), scores(arousal, arousal))  # fmt: skip
        p = r["game_probe"]
        self.assertGreater(p["before"], 0.65, p)
        self.assertLess(p["after"], 0.5, p)
        self.assertGreater(r["kept"]["vector_cosine_median"], 0.9)
        a = r["avd"]["arousal"]
        self.assertLess(a["after"]["spread"], a["before"]["spread"])
        self.assertGreater(a["correlation"], 0.5)
        self.assertIn("Chance", eq_check.markdown(r))

    def test_every_embedding_gets_its_own_probe(self):
        rng = np.random.default_rng(1)
        n = 150
        game = np.array(["le1", "le2", "le3"] * (n // 3))
        group = np.array([f"c{i}" for i in range(n)])
        x = rng.standard_normal((n, 8))
        shifted = x + np.array([{"le1": 3.0, "le2": 0.0, "le3": -3.0}[g] for g in game])[:, None]
        before = {"emotion2vec_plus_large": x, "wavlm_L06": shifted}
        after = {"emotion2vec_plus_large": x, "wavlm_L06": x}
        same = scores(rng.random(n), rng.random(n))
        r = eq_check.compare(list(range(n)), game, group, before, after, same, same)
        self.assertEqual(sorted(r["embeddings"]), ["emotion2vec_plus_large", "wavlm_L06"])
        w = r["embeddings"]["wavlm_L06"]["game_probe"]
        self.assertGreater(w["before"] - w["after"], 0.3, w)
        self.assertEqual(r["game_probe"], r["embeddings"]["emotion2vec_plus_large"]["game_probe"])
        self.assertIn("wavlm_L06", eq_check.markdown(r))


if __name__ == "__main__":
    unittest.main()
