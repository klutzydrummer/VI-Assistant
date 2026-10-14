"""Tests for classes.py on the synthetic corpus run through confounds and
cluster. Needs numpy, scipy and scikit-learn.
Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import classes
import synthcorpus

JOBS = Path(__file__).resolve().parent


def run(stage, d, settings, **inputs):
    (d / f"{stage}.settings.json").write_text(json.dumps(settings))
    env = {**os.environ, "VI_OUT": str(d / stage), "VI_SETTINGS": str(d / f"{stage}.settings.json"),
           "VI_CORPUS": str(d)}  # fmt: skip
    for k, v in inputs.items():
        env[f"VI_INPUT_{k.upper()}"] = str(d / v)
    return subprocess.run([sys.executable, str(JOBS / f"{stage}.py")], env=env, capture_output=True, text=True)


def answer(cid, cluster, fits, foil=False, repeat=False):
    return {"kind": "answer", "id": cid, "cluster": cluster, "fits": fits, "foil": foil, "repeat": repeat,
            "seconds": 2.0, "at": 0}  # fmt: skip


def decision(cluster, action, name=None, into=None):
    return {"kind": "decision", "cluster": cluster, "action": action, "name": name, "into": into, "note": None, "at": 0}


class PurityTest(unittest.TestCase):
    def test_sprt_and_foil_correction_match_the_review_page(self):
        recs = [answer(f"m{i}", "c0", True) for i in range(7)]
        per, overall = classes.purity_summary(recs, {"c0": 30})
        self.assertEqual(per["c0"]["sprt"], "pure")  # ln 19 / ln 1.6 = 6.3
        recs = [answer(f"m{i}", "c0", False) for i in range(4)]
        self.assertEqual(classes.purity_summary(recs, {"c0": 30})[0]["c0"]["sprt"], "impure")
        recs = [answer("m0", "c0", True), answer("m1", "c0", False), answer("f0", "c0", True, foil=True),
                answer("f1", "c0", False, foil=True), answer("m0", "c0", True, repeat=True),
                answer("m1", "c0", True, repeat=True)]  # fmt: skip
        per, overall = classes.purity_summary(recs, {"c0": 2})
        self.assertEqual(per["c0"]["sprt"], "undecided")
        self.assertEqual(overall["false_yes"], 0.5)
        self.assertEqual(per["c0"]["purity"], 0.0)  # (0.5 - 0.5) / 0.5
        self.assertEqual(overall["repeat_pairs"], 2)
        self.assertEqual(overall["kappa"], 0.0)


class ResolveTest(unittest.TestCase):
    def test_merges_splits_rejects(self):
        clusters = {"c0": {"components": [0, 3]}, "c1": {"components": [1]}, "c2": {"components": [2]},
                    "c3": {"components": [4]}, "c4": {"components": [5]}}  # fmt: skip
        decided = {"c0": decision("c0", "split"), "c1": decision("c1", "merge", into="c3"),
                   "c3": decision("c3", "name", name="Mildly Annoyed"), "c4": decision("c4", "reject")}  # fmt: skip
        out, rejected, problems = classes.resolve(clusters, decided)
        by = {c["key"]: c for c in out}
        self.assertEqual(set(by), {"c0.0", "c0.1", "c2", "mildly_annoyed"})
        self.assertEqual(by["mildly_annoyed"]["components"], [1, 4])
        self.assertEqual(by["c0.1"]["components"], [3])
        self.assertEqual(by["c2"]["status"], "unreviewed")
        self.assertEqual(rejected, ["c4"])
        self.assertEqual(problems, [])

    def test_cycles_and_merges_into_rejected_are_reported(self):
        clusters = {c: {"components": [i]} for i, c in enumerate(["c0", "c1", "c2"])}
        decided = {"c0": decision("c0", "merge", into="c1"), "c1": decision("c1", "merge", into="c0"),
                   "c2": decision("c2", "split")}  # fmt: skip
        out, _, problems = classes.resolve(clusters, decided)
        self.assertTrue(any("cycle" in p for p in problems))
        self.assertEqual({c["status"] for c in out if c["clusters"] == ["c2"]}, {"split_single_component"})
        decided = {"c0": decision("c0", "merge", into="c1"), "c1": decision("c1", "reject")}
        out, rejected, problems = classes.resolve(clusters, decided)
        self.assertEqual(sorted(rejected), ["c0", "c1"])
        self.assertTrue(any("rejected" in p for p in problems))


class JobTest(unittest.TestCase):
    def test_end_to_end_on_the_synthetic_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            synthcorpus.write(d)
            inputs = {"harvest": "harvest", "prep": "prep", "clean": "clean", "embed": "embed"}
            r = run("confounds", d, {"pca_dims": 16, "permutations": 10, "bootstrap": 20}, **inputs)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = run("cluster", d, {"k_max": 8, "resamples": 10, "cov_types": ["full"]}, **inputs, confounds="confounds")
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [json.loads(x) for x in (d / "cluster" / "clusters.jsonl").read_text().splitlines()]
            members = {c: [r["id"] for r in rows if r["cluster"] == c and r["formed"]] for c in ["c0", "c1", "c2", "c3"]}

            # No review yet: every cluster comes through unreviewed.
            r = run("classes", d, {}, cluster="cluster", embed="embed", confounds="confounds")
            self.assertEqual(r.returncode, 0, r.stderr)
            result = json.loads((d / "classes" / "classes.json").read_text())
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual([c["status"] for c in result["classes"]], ["unreviewed"] * 4)

            recs = [answer(i, "c0", True) for i in members["c0"][:7]] + [decision("c0", "name", name="Calm")]
            recs += [decision("c1", "merge", into="c0"), decision("c2", "reject")]
            recs += [answer(i, "c3", True) for i in members["c3"][:4]] + [answer(members["c0"][0], "c3", False, foil=True)]
            recs += [answer(i, "c3", True) for i in members["c3"][4:7]] + [decision("c3", "name", name="teasing")]
            (d / "purity.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
            r = run("classes", d, {}, cluster="cluster", embed="embed", confounds="confounds")
            self.assertEqual(r.returncode, 0, r.stderr)
            result = json.loads((d / "classes" / "classes.json").read_text())
            by = {c["key"]: c for c in result["classes"]}
            self.assertEqual(set(by), {"calm", "teasing"})
            self.assertEqual(by["calm"]["clusters"], ["c0", "c1"])
            self.assertEqual(by["calm"]["stability"]["band"], "not re-measured")
            self.assertEqual(by["teasing"]["purity"]["sprt"], ["pure"])
            self.assertEqual(result["rejected"], ["c2"])
            self.assertEqual(result["status"], "ok")
            self.assertTrue(any("repeats, not 50" in f for f in result["failed"]))

            # Rescored memberships match cluster's for an unmerged class, and
            # a merge sums its clusters' responsibilities.
            got = {m["id"]: m for m in map(json.loads, (d / "classes" / "memberships.jsonl").read_text().splitlines())}
            self.assertEqual(len(got), len(rows))
            for row in rows[:50]:
                m = got[row["id"]]
                self.assertAlmostEqual(m["scores"]["teasing"], row["scores"]["c3"], places=2)
                self.assertAlmostEqual(m["responsibilities"]["teasing"], row["responsibilities"]["c3"], places=3)
                self.assertAlmostEqual(m["responsibilities"]["calm"],
                                       row["responsibilities"]["c0"] + row["responsibilities"]["c1"], places=3)
            gone = [m for i, m in got.items() if any(r["id"] == i and r["cluster"] == "c2" for r in rows)]
            self.assertTrue(all(m["responsibilities"]["calm"] + m["responsibilities"]["teasing"] < 0.5 for m in gone))

            eiv = [json.loads(x) for x in (d / "classes" / "eiv_labels.jsonl").read_text().splitlines()]
            self.assertEqual(len(eiv), len(rows))
            self.assertEqual(len(eiv[0]["top"]), 3)
            report = (d / "classes" / "report.md").read_text()
            for heading in ("## Checks that failed", "## Confounds", "## Clusters", "## Purity review", "## Classes"):
                self.assertIn(heading, report)

    def test_refuses_a_stopped_cluster(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cluster").mkdir()
            (d / "cluster" / "cluster.json").write_text(json.dumps({"status": "stopped", "stop_reason": "1 class"}))
            r = run("classes", d, {}, cluster="cluster")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("cluster stopped", r.stderr)


if __name__ == "__main__":
    unittest.main()
