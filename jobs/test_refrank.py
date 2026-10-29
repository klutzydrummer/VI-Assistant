"""Tests for refrank.py, with fake models. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import refrank
import voicearms
from test_voicearms import corpus


class Graded(voicearms.FakeScorer):
    """Clip ids ending in an even number lift the probe by their number; pos0 loses her voice."""

    def __call__(self, path, sentence, mood, probe, her):
        r = super().__call__(path, sentence, mood, probe, her)
        cond = Path(path).parent.name
        if cond.startswith("le3.pos"):
            n = int(cond.removeprefix("le3.pos").removesuffix("_f"))
            r["probe"] = {m: v + (0.05 * n if m == mood else 0.0) for m, v in r["probe"].items()}
            if n == 0:
                r["similarity"] = 0.1
        return r


class RefRankTest(unittest.TestCase):
    def test_ranks_every_example_and_writes_both_clip_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files = corpus(tmp)
            out, ck = tmp / "out", tmp / "ck"
            (tmp / "settings.json").write_text(json.dumps({
                "backend": "fake", "rank_sentences": 2, "shortlist": 2, "shortlist_weak": 3, "controls": 1}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out), "VI_CHECKPOINT": str(ck),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env), mock.patch.object(voicearms, "FakeScorer", Graded):
                refrank.main()
                first = (out / "ranking.json").read_text()
                refrank.main()  # resumes from the checkpoint with the same result
                self.assertEqual((out / "ranking.json").read_text(), first)
            r = json.loads(first)
            warm = r["moods"]["warmth"]
            ids = [x["clip"] for x in warm["ranking"]]
            # Higher-numbered clips lift more; pos0 loses her voice, so it ranks last and is excluded.
            self.assertEqual(ids[-1], "le3.pos0_f")
            self.assertTrue(warm["ranking"][-1]["excluded"])
            lifts = [x["lift"] for x in warm["ranking"] if x["holds"]]
            self.assertEqual(lifts, sorted(lifts, reverse=True))
            self.assertEqual([x["rank"] for x in warm["ranking"]], list(range(1, len(ids) + 1)))
            self.assertFalse(warm["probe_weak"])
            self.assertIn("## warmth", (out / "ranking.md").read_text())
            short = tomllib.loads((out / "shortlist" / "clips.toml").read_text())
            self.assertTrue(short["blind"])
            self.assertTrue(all((out / "shortlist" / c["path"]).exists() for c in short["clip"]))
            warm_short = [c for c in short["clip"] if c["text"].startswith("warmth:")]
            self.assertEqual(len(warm_short), 3)  # top 2 and 1 low-ranked check
            self.assertEqual(sum("low-ranked" in c["label"] for c in warm_short), 1)
            sub = tomllib.loads((out / "submoods" / "clips.toml").read_text())
            self.assertEqual([c["id"] for c in sub["clip"] if c["label"].startswith("warmth")], ids[:2])
            for name in ("ranking.json", "ranking.md", "shortlist/clips.toml"):
                self.assertNotIn("Secret subtitle", (out / name).read_text(), name)

    def test_example_xvec_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files = corpus(tmp)
            (tmp / "settings.json").write_text(json.dumps({"backend": "fake", "example_xvec": "mine"}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(tmp / "out"),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env), self.assertRaises(SystemExit):
                refrank.main()


if __name__ == "__main__":
    unittest.main()
