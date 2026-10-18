"""Tests for context.py and textscore.py with the fake backend. Needs numpy.
Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

import textscore

JOBS = Path(__file__).resolve().parent
MOODS = JOBS.parent / "characters" / "tali" / "moods.toml"


def harvest(folder):
    """Five lines: two in one conversation with context, a bark, a sigh with
    no words, and one with a director's note."""
    (folder / "audio").mkdir(parents=True)
    lines = [
        {"id": "le2.s1_f", "text": "Thank you, Shepard.", "source": {"game": "le2", "conversation": "c1",
         "listener": "shepard", "line_kind": "dialogue_choices"},
         "context": [{"speaker": "shepard", "text": "I got you a present."}],
         "following": [{"speaker": "shepard", "text": "Anytime."}, {"speaker": "shepard", "text": "Don't mention it."}],
         "scene_tags": ["normandy", "dialogue_choices", "to_shepard"]},
        {"id": "le2.s2_f", "text": "You're teasing me.", "source": {"game": "le2", "conversation": "c1"},
         "context": [{"speaker": "shepard", "text": "I got you a present."}, {"speaker": "admiral_raan", "text": "Hm."}]},
        {"id": "le1.b3", "text": "Taking fire!", "flags": ["bark"], "source": {"game": "le1"}},
        {"id": "le1.b4", "text": "", "flags": ["bark", "no_subtitle", "nonverbal"], "source": {"game": "le1"}},
        {"id": "le3.s5_m", "text": "Omni-tooooool!", "source": {"game": "le3"}, "scene_tags": ["citadel_party"],
         "stage_direction": "very drunk"},
    ]  # fmt: skip
    for r in lines:
        r["audio"] = {"path": f"audio/{r['id']}.wav", "duration_s": 1.0}
    (folder / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))


def run(job, env):
    p = subprocess.run([sys.executable, str(JOBS / job)], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p


class ContextTest(unittest.TestCase):
    def test_records_and_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            harvest(d / "harvest")
            (d / "s.json").write_text(json.dumps({"before": 1}))
            env = {**os.environ, "VI_INPUT_HARVEST": str(d / "harvest"), "VI_OUT": str(d / "out"),
                   "VI_SETTINGS": str(d / "s.json")}  # fmt: skip
            run("context.py", env)
            recs = {r["id"]: r for r in map(json.loads, (d / "out" / "context.jsonl").read_text().splitlines())}
            self.assertNotIn("le1.b4", recs)  # no words
            self.assertEqual(len(recs), 4)
            first = recs["le2.s1_f"]
            self.assertEqual(first["scene_tags"], ["normandy"])  # structural tags dropped
            self.assertEqual(
                first["prompt"],
                "Scene: normandy\nBefore:\nShepard: I got you a present.\n>>> Tali (to Shepard): Thank you, Shepard.\n"
                "After (one of):\nShepard: Anytime.\nShepard: Don't mention it.",
            )
            self.assertEqual(recs["le2.s2_f"]["before"], [{"speaker": "Admiral Raan", "text": "Hm."}])  # last 1
            self.assertIn("in combat", recs["le1.b3"]["prompt"])
            self.assertIn("Voice director's note: very drunk", recs["le3.s5_m"]["prompt"])
            report = json.loads((d / "out" / "context.json").read_text())
            self.assertEqual(report["left_out_no_words"], 1)


class ParseTest(unittest.TestCase):
    def setUp(self):
        os.environ["VI_SETTINGS"] = ""
        self.moods = textscore.Moods(
            [{"id": "warm", "description": "warm"}, {"id": "teasing", "description": "teasing"}],
            [{"id": "drunk", "description": "how drunk", "levels": [{"id": "sober", "description": "sober"},
                                                                    {"id": "drunk", "description": "drunk"}]}],
        )  # fmt: skip

    def test_good_answer_and_duplicates(self):
        text = json.dumps({"moods": [{"mood": "warm", "strength": 2, "reason": "a"},
                                     {"mood": "warm", "strength": 3, "reason": "b"}], "drunk": "sober"})  # fmt: skip
        self.assertEqual(textscore.parse(text, self.moods),
                         {"moods": [{"mood": "warm", "strength": 2, "reason": "a"}], "drunk": "sober"})  # fmt: skip

    def test_bad_answers(self):
        for text in ["not json", "[]", json.dumps({"moods": [], "drunk": "wasted"}),
                     json.dumps({"moods": [{"mood": "angry", "strength": 1, "reason": ""}], "drunk": "sober"}),
                     json.dumps({"moods": [{"mood": "warm", "strength": 5, "reason": ""}], "drunk": "sober"})]:  # fmt: skip
            self.assertIsNone(textscore.parse(text, self.moods), text)

    def test_schema_names_every_id(self):
        shape = textscore.schema(self.moods, 3)
        self.assertEqual(shape["properties"]["moods"]["items"]["properties"]["mood"]["enum"], ["warm", "teasing"])
        self.assertEqual(shape["properties"]["drunk"]["enum"], ["sober", "drunk"])
        self.assertEqual(shape["required"], ["moods", "drunk"])
        self.assertEqual([k for k, _ in self.moods.labels()], ["warm", "teasing", "drunk:sober", "drunk:drunk"])


class JobTest(unittest.TestCase):
    def test_end_to_end_with_fake_models(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            harvest(d / "harvest")
            (d / "c.json").write_text("{}")
            base = {**os.environ, "VI_INPUT_HARVEST": str(d / "harvest"), "VI_SETTINGS": str(d / "c.json")}
            run("context.py", {**base, "VI_OUT": str(d / "context")})
            (d / "s.json").write_text(json.dumps({"backend": "fake", "checks": "fail", "moods": str(MOODS)}))
            env = {**os.environ, "VI_INPUT_CONTEXT": str(d / "context"), "VI_OUT": str(d / "out"),
                   "VI_SETTINGS": str(d / "s.json"), "VI_CHECKPOINT": str(d / "ckpt")}  # fmt: skip
            run("textscore.py", env)
            out = d / "out"
            toml = tomllib.loads(MOODS.read_text())
            moods = textscore.Moods(toml["mood"], toml["knob"])
            keys = [k for k, _ in moods.labels()]
            for name in ("goemotions", "nli", "embedder"):
                rows = [json.loads(x) for x in (out / "scores" / f"{name}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 4, name)
                if name != "goemotions":
                    self.assertEqual(sorted(rows[0]["scores"]), sorted(keys))
            for part, n in (("line", 4), ("context", 4), ("moods", len(keys))):
                m = np.load(out / "embeddings" / f"qwen3_{part}.npy")
                self.assertEqual(m.shape[0], n)
                self.assertEqual(len((out / "embeddings" / f"qwen3_{part}.ids").read_text().split()), n)
            for name in textscore.LLMS:
                rows = [json.loads(x) for x in (out / "llm" / f"{name}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 4)
                self.assertEqual(len(rows[0]["samples"]), 2)
                self.assertTrue(all(s is not None for r in rows for s in r["samples"]))
                self.assertIn("teasing: Tali is teasing", (out / "llm" / f"{name}.prompt.txt").read_text())
            report = json.loads((out / "textscore.json").read_text())
            self.assertEqual(report["llms"]["gemma"]["failed"], 0)
            self.assertEqual(json.loads((out / "checks.json").read_text())["failed"], 0)

            # A rerun resumes everything from the checkpoint.
            again = run("textscore.py", {**env, "VI_OUT": str(d / "out2")})
            self.assertIn("resuming, 4 already done", again.stderr)



class CudaLibEnv(unittest.TestCase):
    def test_venv_nvidia_libraries_go_first_and_missing_builtins_stop_it(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d, "nvidia", "cu13", "lib")
            lib.mkdir(parents=True)
            (lib / "libnvrtc.so.13").touch()
            old_path, old_ld = sys.path[:], os.environ.get("LD_LIBRARY_PATH")
            sys.path.insert(0, d)
            os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
            try:
                with self.assertRaises(SystemExit):
                    textscore.cuda_lib_env()
                (lib / "libnvrtc-builtins.so.13.0").touch()
                env = textscore.cuda_lib_env()
                self.assertEqual(env["LD_LIBRARY_PATH"], f"{lib}:/usr/local/cuda/lib64")
            finally:
                sys.path[:] = old_path
                if old_ld is None:
                    os.environ.pop("LD_LIBRARY_PATH", None)
                else:
                    os.environ["LD_LIBRARY_PATH"] = old_ld


if __name__ == "__main__":
    unittest.main()
