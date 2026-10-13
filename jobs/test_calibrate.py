"""Tests for calibrate.py with the fake backend (no numpy). Run with
`python3 -m unittest discover -s jobs`."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import calibrate

JOBS = Path(__file__).resolve().parent


def label(i, v):
    return {"id": i, "scores": {"joy": {"value": v, "confidence": 0.5,
            "signals": {"audio": v, "text": 1 - v, "prior": None},
            "signal_confidence": {"audio": 0.8, "text": 0.6}}}, "tags": []}  # fmt: skip


class CalibrateTest(unittest.TestCase):
    def test_calibration_sample_matches_the_review_page(self):
        # Same rule as crates/review/src/queue.rs (tested there with a, b, c).
        self.assertEqual(calibrate.calibration(["a", "b", "c"], 2), ["c", "b"])

    def test_weights_follow_agreement_with_reviews(self):
        ids = [f"c{i}" for i in range(20)]
        truth = {i: k / 20 for k, i in enumerate(ids)}
        cols = {"good": {i: (truth[i], 0.8) for i in ids}, "backwards": {i: (1 - truth[i], 0.8) for i in ids}}
        w = calibrate.fit_weights(cols, truth, ids)
        self.assertEqual(w["good"], 1.0)
        self.assertEqual(w["backwards"], 0.0)

    def test_end_to_end_with_fake_probe(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "label").mkdir()
            ids = [f"c{i:02}" for i in range(40)]
            with open(d / "label" / "labels.jsonl", "w") as f:
                for k, i in enumerate(ids):
                    f.write(json.dumps(label(i, k / 40)) + "\n")
            (d / "label" / "axes.json").write_text(json.dumps([{"id": "joy", "description": ""}]))
            with open(d / "reviews.jsonl", "w") as f:
                for k, i in enumerate(ids[:-1]):
                    f.write(json.dumps({"id": i, "verdict": "keep", "scores": {"joy": k / 40}}) + "\n")
                f.write(json.dumps({"id": ids[-1], "verdict": "flag", "flag": "music"}) + "\n")
            (d / "s.json").write_text(json.dumps({"backend": "fake", "reviews": str(d / "reviews.jsonl"),
                                                   "calibration": 10, "min_reviews": 10}))  # fmt: skip
            env = {**os.environ, "VI_INPUT_LABEL": str(d / "label"), "VI_SETTINGS": str(d / "s.json"),
                   "VI_OUT": str(d / "out")}  # fmt: skip
            subprocess.run([sys.executable, str(JOBS / "calibrate.py")], env=env, check=True, capture_output=True)
            report = json.loads((d / "out" / "calibration.json").read_text())
            self.assertEqual(report["reviewed"], 39)
            joy = report["axes"]["joy"]
            self.assertEqual(joy["weights"]["audio"], 1.0)
            self.assertEqual(joy["weights"]["text"], 0.0)
            self.assertGreater(joy["on_calibration_sample"]["fused_calibrated"]["spearman"], 0.9)
            out = [json.loads(l) for l in open(d / "out" / "labels.jsonl")]
            self.assertTrue(out[0]["reviewed"])
            self.assertEqual(out[-1]["flagged"], "music")
            self.assertEqual(out[0]["scores"]["joy"]["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
