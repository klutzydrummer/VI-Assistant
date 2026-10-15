"""Tests for cluster.py on the synthetic corpus (confounds first). Needs
numpy, scipy and scikit-learn. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cluster
import synthcorpus

JOBS = Path(__file__).resolve().parent


def run(script, d, settings, **inputs):
    (d / f"{script}.json").write_text(json.dumps(settings))
    env = {**os.environ, "VI_OUT": str(d / script), "VI_SETTINGS": str(d / f"{script}.json"),
           **{f"VI_INPUT_{k.upper()}": str(d / v) for k, v in inputs.items()}}  # fmt: skip
    r = subprocess.run([sys.executable, str(JOBS / f"{script}.py")], env=env, capture_output=True, text=True)
    return r


class MergeTest(unittest.TestCase):
    def test_merging_joins_the_components_that_overlap(self):
        # Components 0 and 1 share their clips; 2 stands apart.
        resp = np.array([[0.5, 0.5, 0.0]] * 10 + [[0.0, 0.0, 1.0]] * 10)
        path = cluster.merge_path(resp)
        self.assertEqual(sorted(map(sorted, path[1][0])), [[0, 1], [2]])
        self.assertEqual(path[1][2], 10.0)
        self.assertLess(path[1][1], path[0][1])

    def test_merging_recovers_classes_from_an_over_split_mixture(self):
        rng = np.random.default_rng(3)
        centres = rng.standard_normal((4, 6)) * 4
        z = np.vstack([c + rng.standard_normal((300, 6)) for c in centres])
        for k in (4, 6, 8, 10):
            _, _, n = cluster.fit_classes(z, k, "full")
            self.assertEqual(n, 4, k)

    def test_parallel_analysis_finds_the_real_dimensions(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((500, 3)) @ rng.standard_normal((3, 20)) * 3 + rng.standard_normal((500, 20))
        dims, _ = cluster.parallel_analysis(x, rng)
        self.assertEqual(dims, 3)

    def test_spearman_brown_split_half(self):
        rng = np.random.default_rng(0)
        z = np.vstack([rng.standard_normal((200, 4)) + [3, 0, 0, 0], rng.standard_normal((200, 4))])
        out = cluster.split_half(z, np.arange(200), z.mean(0), rng)
        self.assertGreater(out[160], out[10])
        self.assertGreater(out[160], 0.9)


class JobTest(unittest.TestCase):
    def test_end_to_end_finds_the_synthetic_classes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            truth = synthcorpus.write(d)["classes"]
            inputs = {"harvest": "harvest", "prep": "prep", "clean": "clean", "embed": "embed"}
            r = run("confounds", d, {"pca_dims": 16, "permutations": 10, "bootstrap": 20, "game_margin": 0.12}, **inputs)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = run("cluster", d, {"k_max": 8, "resamples": 10, "cov_types": ["full"]}, **inputs, confounds="confounds")
            self.assertEqual(r.returncode, 0, r.stderr)
            report = json.loads((d / "cluster" / "cluster.json").read_text())
            self.assertEqual(report["status"], "ok", report.get("stop_reason"))
            self.assertEqual(report["merge"]["classes"], 4, report["merge"])
            self.assertTrue(report["shortest_usable"]["measured"])
            # 0.5 s crops are too noisy to find their source; 2 s ones aren't.
            rates = report["shortest_usable"]["retrieval"]
            self.assertLess(rates["0.5"], rates["2"])
            recs = [json.loads(x) for x in (d / "cluster" / "clusters.jsonl").read_text().splitlines()]
            from sklearn.metrics import adjusted_rand_score

            ari = adjusted_rand_score([truth[r["id"]] for r in recs], [r["cluster"] for r in recs])
            self.assertGreater(ari, 0.9)
            for p in report["classes"]:
                self.assertEqual(p["stability"], "class")
                self.assertFalse(p["rejected_for_nuisance"])
            self.assertEqual(set(recs[0]["scores"]), {"c0", "c1", "c2", "c3"})

    def test_stops_when_confounds_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            synthcorpus.write(d)
            (d / "confounds").mkdir()
            (d / "confounds" / "confounds.json").write_text(json.dumps({"status": "stopped", "stop_reason": "x"}))
            r = run("cluster", d, {}, harvest="harvest", prep="prep", clean="clean", embed="embed",
                    confounds="confounds")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("confounds stopped", r.stderr)


if __name__ == "__main__":
    unittest.main()
