"""Tests for embed.py with the fake backend. Needs numpy and soundfile.
Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import models
import vi_job

JOBS = Path(__file__).resolve().parent


def tone(seconds, freq=220.0):
    t = np.arange(int(seconds * 16_000)) / 16_000
    return (0.3 * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


class RealFramesTest(unittest.TestCase):
    def test_frame_count(self):
        self.assertEqual(models.real_frames(16_000), 50)
        self.assertEqual(models.real_frames(16_001), 51)
        self.assertEqual(models.real_frames(0), 1)
        self.assertEqual(models.real_frames(16_000 * 40), 1500)

    def test_padding_does_not_change_the_pooled_vector(self):
        # The same content padded to 30 s (Whisper's way) and to 10 s gives
        # the same vector; a plain mean over all frames does not.
        fake = models.FakeEmbedder([6])
        wav = tone(1.3)
        long, short = fake.frames(wav, total=1500), fake.frames(wav, total=500)
        np.testing.assert_allclose(models.pool_real(long, len(wav)), models.pool_real(short, len(wav)))
        self.assertFalse(np.allclose(long.mean(axis=0), short.mean(axis=0)))

    def test_pooled_vector_tracks_content_not_length_of_padding(self):
        fake = models.FakeEmbedder([6])
        a, b = tone(1.0, 200), tone(1.0, 2000)
        self.assertFalse(np.allclose(models.pool_real(fake.frames(a), len(a)), models.pool_real(fake.frames(b), len(b))))


class ArrayCheckpointTest(unittest.TestCase):
    def test_resume_skips_saved_chunks_and_ignores_torn_ones(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["VI_CHECKPOINT"] = d
            try:
                clips = [vi_job.Clip(id=f"c{i}", text="", audio=Path("x"), duration_s=1.0) for i in range(5)]
                calls = []

                def compute(part):
                    calls.extend(c.id for c in part)
                    return [{"v": np.full(3, float(c.id[1:]))} for c in part]

                first = vi_job.ArrayCheckpoint("emb").run(clips[:3], compute, chunk=2)
                (Path(d) / "emb" / ".00009.tmp").write_bytes(b"half a chunk")
                (Path(d) / "emb" / "00009.npz").write_bytes(b"not a zip")
                again = vi_job.ArrayCheckpoint("emb")
                result = again.run(clips, compute, chunk=2)
            finally:
                del os.environ["VI_CHECKPOINT"]
            self.assertEqual(calls, ["c0", "c1", "c2", "c3", "c4"])
            self.assertEqual(sorted(result), [c.id for c in clips])
            np.testing.assert_array_equal(result["c2"]["v"], first["c2"]["v"])
            self.assertEqual(result["c4"]["v"].dtype, np.float16)


class JobTest(unittest.TestCase):
    def test_end_to_end_with_fake_models(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            harvest, prep = d / "harvest", d / "prep"
            (harvest / "audio").mkdir(parents=True)
            (prep / "audio16k").mkdir(parents=True)
            lines, prepped = [], []
            for i in range(6):
                lid = f"le{i % 3 + 1}.s{i}_f"
                flags = ["bark", "no_subtitle"] if i == 5 else []
                lines.append({"id": lid, "text": "" if flags else f"Line {i}.", "flags": flags,
                              "audio": {"path": f"audio/{lid}.wav", "duration_s": 1.0 + i / 4},
                              "source": {"game": f"le{i % 3 + 1}"}})  # fmt: skip
                sf.write(prep / "audio16k" / f"{lid}.flac", tone(1.0 + i / 4, 200 + 50 * i), 16_000)
                prepped.append({"id": lid, "duration_s": 1.0 + i / 4, "flags": [],
                                "audio16k": {"path": f"audio16k/{lid}.flac", "sample_rate": 16_000}})  # fmt: skip
            (harvest / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
            (prep / "prep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prepped))
            (d / "settings.json").write_text(json.dumps({"backend": "fake", "wavlm_layers": [6, 12, 18]}))
            env = {**os.environ, "VI_INPUT_HARVEST": str(harvest), "VI_INPUT_PREP": str(prep),
                   "VI_OUT": str(d / "out"), "VI_SETTINGS": str(d / "settings.json"),
                   "VI_CHECKPOINT": str(d / "ckpt")}  # fmt: skip
            subprocess.run([sys.executable, str(JOBS / "embed.py")], env=env, check=True, capture_output=True)
            out = d / "out"
            report = json.loads((out / "embed.json").read_text())
            expected = {"wavlm_L06", "wavlm_L12", "wavlm_L18", "emotion2vec_plus_large", "emotion2vec_base",
                        "bude_whisper", "minilm"}  # fmt: skip
            self.assertEqual(set(report["embeddings"]), expected)
            for name in expected:
                m = np.load(out / "embeddings" / f"{name}.npy")
                ids = (out / "embeddings" / f"{name}.ids").read_text().split()
                self.assertEqual(m.dtype, np.float16)
                self.assertEqual(len(ids), m.shape[0])
                self.assertEqual(m.shape[0], 5 if name == "minilm" else 6)  # the bark has no words
            scores = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines()]
            self.assertEqual(set(scores[0]), {"id", "eiv", "eiv_quality", "avd", "emotion2vec_plus_large_classes"})
            self.assertIn("Teasing", scores[0]["eiv"])
            self.assertEqual(len((out / "prosody.jsonl").read_text().splitlines()), 6)
            self.assertEqual(json.loads((out / "checks.json").read_text())["failed"], 0)


if __name__ == "__main__":
    unittest.main()
