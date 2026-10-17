"""Tests for remap_purity.py and the cluster-run guard in classes.py.
Run with `python3 -m unittest discover -s jobs`."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import classes
import remap_purity as rp

JOBS = Path(__file__).resolve().parent


def rows(groups):
    """{cluster: [ids]} -> clusters.jsonl rows, all formed, plus one unformed."""
    out = [{"id": i, "cluster": c, "formed": True} for c, ids in groups.items() for i in ids]
    return out + [{"id": "zz", "cluster": "c0", "formed": False}]


OLD = rows({"c0": [f"a{i}" for i in range(10)], "c1": [f"b{i}" for i in range(8)], "c2": [f"d{i}" for i in range(6)]})
# The rebuild: old c1 is now c0 (one clip moved), old c0 is c1, old c2 broke apart.
NEW = rows({"c0": [f"b{i}" for i in range(7)] + ["a0"], "c1": [f"a{i}" for i in range(1, 10)] + ["b7"],
            "c2": ["d0", "d1", "d2"], "c3": ["d3", "d4", "d5"]})  # fmt: skip
RECORDS = [
    {"kind": "answer", "id": "a1", "cluster": "c0", "fits": True, "foil": False, "repeat": False, "at": 1},
    {"kind": "decision", "cluster": "c0", "action": "name", "name": "urgency", "at": 2},
    {"kind": "decision", "cluster": "c1", "action": "merge", "into": "c0", "at": 3},
    {"kind": "decision", "cluster": "c2", "action": "name", "name": "sad", "at": 4},
]


class GuardTest(unittest.TestCase):
    def test_fingerprint_follows_the_clusters_not_the_order(self):
        self.assertEqual(classes.fingerprint(OLD), classes.fingerprint(list(reversed(OLD))))
        self.assertNotEqual(classes.fingerprint(OLD), classes.fingerprint(NEW))
        renamed = [{**r, "cluster": {"c0": "c1", "c1": "c0"}.get(r["cluster"], r["cluster"])} for r in OLD]
        self.assertNotEqual(classes.fingerprint(OLD), classes.fingerprint(renamed))

    def test_for_run(self):
        recs = [{"kind": "decision", "cluster": "c0"}, {"kind": "decision", "cluster": "c1", "run": "x"},
                {"kind": "decision", "cluster": "c2", "run": "y"}]  # fmt: skip
        keep, other = classes.for_run(recs, "x")
        self.assertEqual([r["cluster"] for r in keep], ["c0", "c1"])
        self.assertEqual(other, 1)
        keep, other = classes.for_run(recs, "x", legacy_run="y")
        self.assertEqual([r["cluster"] for r in keep], ["c1"])


class RemapTest(unittest.TestCase):
    def test_proposal(self):
        proposal, new_run, old_run = rp.propose(OLD, NEW, RECORDS)
        got = {c: n for c, _, n, _ in proposal}
        self.assertEqual(got, {"c0": "c1", "c1": "c0", "c2": None})  # c2 broke apart: hear it again
        out = rp.remapped(proposal, new_run, now=9)
        self.assertEqual([(r["cluster"], r["action"], r.get("into")) for r in out],
                         [("c1", "name", None), ("c0", "merge", "c1")])  # fmt: skip
        self.assertTrue(all(r["run"] == new_run for r in out))
        self.assertIn("remapped from c0", out[0]["note"])

    def test_write_stamps_old_records_and_classes_sees_only_the_new(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, rs in (("old", OLD), ("new", NEW)):
                (d / name).mkdir()
                (d / name / "clusters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
            purity = d / "purity.jsonl"
            purity.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
            dry = subprocess.run([sys.executable, str(JOBS / "remap_purity.py"), str(d / "old"), str(d / "new"),
                                  str(purity)], capture_output=True, text=True, check=True)  # fmt: skip
            self.assertIn("c2 (sad) -> not carried", dry.stdout)
            self.assertEqual(len(purity.read_text().splitlines()), 4, "a dry run writes nothing")
            subprocess.run([sys.executable, str(JOBS / "remap_purity.py"), str(d / "old"), str(d / "new"),
                            str(purity), "--write"], capture_output=True, check=True)  # fmt: skip
            self.assertEqual(len(list(d.glob("purity.jsonl.bak-*"))), 1)
            recs = classes.read_purity(purity)
            self.assertEqual(len(recs), 6)
            keep, other = classes.for_run(recs, classes.fingerprint(NEW))
            self.assertEqual(other, 4)
            self.assertEqual(classes.decisions(keep)["c1"]["name"], "urgency")

    def test_stamp_only(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "c").mkdir()
            (d / "c" / "clusters.jsonl").write_text("".join(json.dumps(r) + "\n" for r in OLD))
            purity = d / "purity.jsonl"
            purity.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
            subprocess.run([sys.executable, str(JOBS / "remap_purity.py"), "--stamp", str(d / "c"), str(purity)],
                           check=True, capture_output=True)  # fmt: skip
            run = classes.fingerprint(OLD)
            self.assertTrue(all(r["run"] == run for r in classes.read_purity(purity)))
            self.assertEqual(rp.stamp(purity, run), 0, "stamping twice changes nothing")


if __name__ == "__main__":
    unittest.main()
