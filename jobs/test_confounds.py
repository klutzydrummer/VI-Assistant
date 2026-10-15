"""Tests for confounds.py on a synthetic corpus with known structure. Needs
numpy and scikit-learn. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

import confounds
import synthcorpus

JOBS = Path(__file__).resolve().parent


class TransformTest(unittest.TestCase):
    def test_leace_leaves_no_linear_information(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((500, 2))
        x = rng.standard_normal((500, 10)) + z @ rng.standard_normal((2, 10)) * 3
        e = confounds.Leace().fit(x, z).apply(x)
        cov = (e - e.mean(0)).T @ (z - z.mean(0)) / len(z)
        self.assertLess(np.abs(cov).max(), 1e-8)
        # Only the z directions change.
        self.assertLess(np.linalg.matrix_rank(x - e, tol=1e-6), 3)

    def test_coral_matches_game_covariances_without_whitening(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((400, 4)) * [1, 2, 3, 4]
        b = rng.standard_normal((400, 4)) * [4, 3, 2, 1] + 5
        x = np.vstack([a, b])
        game = np.array(["a"] * 400 + ["b"] * 400)
        out = confounds.Coral().fit(x, game, np.arange(800)).apply(x, game)
        ca, cb = np.cov(out[:400].T), np.cov(out[400:].T)
        gap = lambda p, q: np.linalg.norm(p - q) / np.linalg.norm(p + q)  # noqa: E731
        # Ledoit-Wolf shrinkage keeps it from being exact.
        self.assertLess(gap(ca, cb), 0.1)
        self.assertGreater(gap(np.cov(a.T), np.cov(b.T)), 0.4)
        np.testing.assert_allclose(out[:400].mean(0), out[400:].mean(0), atol=1e-9)
        self.assertGreater(np.trace(ca), 4 * 1.5)  # not whitened to the identity

    def test_syllables(self):
        self.assertEqual(confounds.syllables("{Sighs} Keelah se'lai, Shepard."), 6)
        self.assertEqual(confounds.syllables(""), 0)

    def test_wilson_upper_bound(self):
        self.assertAlmostEqual(confounds.wilson_upper(0.5, 100), 0.5964, places=3)


class GameProbeTest(unittest.TestCase):
    def test_pure_noise_scores_chance_not_below(self):
        # Fitting Coral on the clips the probe tests on put pure noise well
        # below chance (0.11 for the SVM), which let a game still audible
        # pass. Refitted per fold, noise sits at chance.
        rng = np.random.default_rng(0)
        n = 3 * 120
        game = np.repeat(["le1", "le2", "le3"], n // 3)
        meta = SimpleNamespace(game=game, group=np.array([f"{g}:{i // 4}" for i, g in enumerate(game)]),
                               unique=np.ones(n, dtype=bool), bark=np.zeros(n, dtype=bool),
                               excluded=np.zeros(n, dtype=bool), log_syll=rng.standard_normal(n),
                               floor_level=rng.standard_normal(n), log_dur=rng.standard_normal(n))  # fmt: skip
        x = rng.standard_normal((n, 48))
        out = confounds.game_probe(x, meta, np.arange(n), {}, np.arange(n), 16, False)
        for name in ("knn", "svm"):
            self.assertGreater(out[name]["balanced_accuracy"], 0.25, out)
            self.assertLess(out[name]["balanced_accuracy"], 0.42, out)


class JobTest(unittest.TestCase):
    def test_end_to_end_picks_the_candidate_with_the_classes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # A mild EQ: Coral can't fully undo a strong per-dimension one on
            # tight synthetic classes, and the honest game probe hears what's
            # left (the cluster and classes tests keep the default).
            synthcorpus.write(d, eq=0.1)
            (d / "settings.json").write_text(json.dumps({"pca_dims": 16, "permutations": 20, "bootstrap": 50, "game_margin": 0.12}))
            env = {**os.environ, "VI_INPUT_HARVEST": str(d / "harvest"), "VI_INPUT_PREP": str(d / "prep"),
                   "VI_INPUT_CLEAN": str(d / "clean"), "VI_INPUT_EMBED": str(d / "embed"),
                   "VI_OUT": str(d / "out"), "VI_SETTINGS": str(d / "settings.json")}  # fmt: skip
            r = subprocess.run([sys.executable, str(JOBS / "confounds.py")], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            report = json.loads((d / "out" / "confounds.json").read_text())
            good = report["candidates"]["good"]
            # The game was easy to hear before and is gone after.
            self.assertGreater(good["selection"]["before"]["game_knn_balanced_accuracy"], 0.8)
            self.assertTrue(good["selection"]["tests"]["game_gone"], good["selection"]["game"])
            self.assertTrue(good["selection"]["tests"]["text_length_gone"])
            self.assertEqual(report["chosen"], "good", json.dumps(report["candidates"]["lengthy"]["selection"]["tests"]))
            self.assertIn("permutation_p", good["report"]["game"]["svm"])
            self.assertEqual(report["excluded"], 1)  # the singing line
            ids = (d / "out" / "cleaned.ids").read_text().split()
            self.assertEqual(np.load(d / "out" / "cleaned.npy").shape[0], len(ids))
            self.assertEqual(np.load(d / "out" / "cleaned_crops.npy").shape[0], 480)


if __name__ == "__main__":
    unittest.main()
