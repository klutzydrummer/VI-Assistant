"""Tests for voicesteer.py, with fake models. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import steer
import voicesteer


def probe_file(path, moods, dim=4):
    arrays = {"moods": np.array(moods), "prosody_keys": np.array(["f0_median_hz", "loudness_db"]),
              "prosody_mean": np.array([200.0, -25.0]), "prosody_scale": np.array([30.0, 5.0])}
    rng = np.random.default_rng(0)
    for m in moods:
        arrays.update({f"{m}.mean": np.zeros(dim + 2), f"{m}.scale": np.ones(dim + 2), f"{m}.pca_mean": np.zeros(dim + 2),
                       f"{m}.pca_components": rng.normal(size=(2, dim + 2)), f"{m}.coef": np.array([1.0, -1.0]),
                       f"{m}.intercept": np.array([0.1])})
    np.savez(path, **arrays)


class ProbeTest(unittest.TestCase):
    def test_matches_the_sklearn_pipeline_it_stands_for(self):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 7))
        y = (X[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(int)
        m = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0), LogisticRegression()).fit(X, y)
        sc, pca, lr = m.steps[0][1], m.steps[1][1], m.steps[2][1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.npz"
            np.savez(path, moods=np.array(["warm"]), prosody_keys=np.array(["a"]), prosody_mean=np.zeros(1),
                     prosody_scale=np.ones(1), **{"warm.mean": sc.mean_, "warm.scale": sc.scale_, "warm.pca_mean": pca.mean_,
                     "warm.pca_components": pca.components_, "warm.coef": lr.coef_[0], "warm.intercept": lr.intercept_})  # fmt: skip
            p = voicesteer.Probe(path)
            for row in X[:5]:
                self.assertAlmostEqual(p(row)["warm"], m.predict_proba(row[None])[0, 1], places=4)


class FakeRunTest(unittest.TestCase):
    def test_runs_end_to_end_lowers_only_when_needed_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files, out, ck = tmp / "harvest", tmp / "files", tmp / "out", tmp / "ck"
            for d in (harvest / "audio", files):
                d.mkdir(parents=True)
            lines = []
            for i in range(20):
                cid = f"le3.s{i}_f"
                steer.write_wav(harvest / "audio" / f"{cid}.wav", steer.fake_wave(cid, seconds=0.1))
                lines.append({"id": cid, "speaker": "tali", "text": f"Secret subtitle {i}",
                              "audio": {"path": f"audio/{cid}.wav", "sample_rate": 16000, "channels": 1, "duration_s": 4.0},
                              "source": {"game": "le3"}})  # fmt: skip
            (harvest / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
            sets = {"moods": {"warmth": {"pairs": [[f"le3.s{i}_f", f"le3.s{i + 5}_f"] for i in range(4)], "route": "voice_and_chat"},
                              "teasing": {"pairs": [[f"le3.s{i + 10}_f", f"le3.s{i + 15}_f"] for i in range(3)], "borderline": True}},
                    "neutral": ["le3.s9_f", "le3.s19_f"]}  # fmt: skip
            (files / "voice_sets.json").write_text(json.dumps(sets))
            probe_file(files / "voice_probe.npz", ["warmth", "teasing"])
            (tmp / "settings.json").write_text(json.dumps({"backend": "fake", "refs_per_side": 3}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out), "VI_CHECKPOINT": str(ck),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env):
                voicesteer.main()
                first = (out / "report.json").read_text()
                voicesteer.main()  # a rerun reuses the checkpoint and gives the same report
                self.assertEqual((out / "report.json").read_text(), first)
            report = json.loads(first)
            for m in ("warmth", "teasing"):
                r = report["moods"][m]
                # Fake voice similarity drops at strength 1, so it's lowered once, to 0.5, and kept there.
                self.assertEqual([t["strength"] for t in r["tried"]], [1.0, 0.5], m)
                self.assertEqual(r["strength"], 0.5)
                self.assertEqual(r["verdict"], "steers")
            self.assertTrue(report["moods"]["teasing"]["borderline"])
            self.assertEqual(report["moods"]["warmth"]["pairs"], 3)
            self.assertTrue((out / "vectors" / "warmth.npy").exists())
            self.assertEqual(json.loads((out / "vectors" / "vectors.json").read_text())["strength"]["warmth"], 0.5)
            for name in ("report.json", "report.md", "listen.md"):
                self.assertNotIn("Secret subtitle", (out / name).read_text(), name)


if __name__ == "__main__":
    unittest.main()
