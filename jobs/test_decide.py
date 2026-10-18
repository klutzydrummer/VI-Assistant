"""Tests for decide.py with the fake backend. Standard library only.
Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path

import decide
from test_textscore import MOODS, harvest, run


class DecideTest(unittest.TestCase):
    def test_every_line_gets_every_question_from_both_models(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            harvest(d / "harvest")
            (d / "c.json").write_text(json.dumps({}))
            env = {**os.environ, "VI_INPUT_HARVEST": str(d / "harvest"), "VI_OUT": str(d / "context"),
                   "VI_SETTINGS": str(d / "c.json")}  # fmt: skip
            run("context.py", env)
            (d / "s.json").write_text(json.dumps({"backend": "fake", "moods": str(MOODS)}))
            env = {**os.environ, "VI_INPUT_CONTEXT": str(d / "context"), "VI_OUT": str(d / "out"),
                   "VI_SETTINGS": str(d / "s.json"), "VI_CHECKPOINT": str(d / "ck")}  # fmt: skip
            run("decide.py", env)

            moods = tomllib.loads(MOODS.read_text())
            want = {m["id"] for m in moods["mood"]}
            want |= {f"{k['id']}:{lv['id']}" for k in moods["knob"] for lv in k["levels"]}
            context = [json.loads(x)["id"] for x in (d / "context" / "context.jsonl").read_text().splitlines()]
            for name in ("kev", "clef"):
                rows = [json.loads(x) for x in (d / "out" / "scores" / f"{name}.jsonl").read_text().splitlines()]
                self.assertEqual([r["id"] for r in rows], context)
                for r in rows:
                    self.assertEqual(set(r["scores"]), want)
                    for k in moods["knob"]:
                        total = sum(r["scores"][f"{k['id']}:{lv['id']}"] for lv in k["levels"])
                        self.assertAlmostEqual(total, 1.0, places=3)
            report = json.loads((d / "out" / "decide.json").read_text())
            self.assertEqual(set(report["models"]), {"kev", "clef"})
            qs = json.loads((d / "out" / "questions.json").read_text())
            self.assertEqual(qs["warm"]["type"], "noul")
            self.assertEqual(qs["energy"]["type"], "score")
            self.assertEqual(len(qs["energy"]["criteria"]), 3)

    def test_scores_reads_both_answer_shapes(self):
        class M:
            moods = [{"id": "warm"}]
            knobs = [{"id": "drunk", "levels": [{"id": "sober"}, {"id": "drunk"}]}]

        answers = {"warm": {"type": "noul", "noul": 0.81234},
                   "drunk": {"type": "score", "score": 0.2, "probabilities": {"0": 0.8, "1": 0.2}}}  # fmt: skip
        self.assertEqual(decide.scores(M, answers), {"warm": 0.8123, "drunk:sober": 0.8, "drunk:drunk": 0.2})

    def test_a_kernel_fallback_stops_the_run(self):
        h = decide.Fallbacks()
        h.seen.append("`chunk_gated_delta_rule` is " + decide.FALLBACK + " because ...")
        with self.assertRaises(SystemExit):
            decide.check_kernels(h, strict=True)
        decide.check_kernels(h, strict=False)  # reported, not fatal


if __name__ == "__main__":
    unittest.main()
