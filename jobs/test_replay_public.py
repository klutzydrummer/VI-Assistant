"""Tests for jobs/omnivoice-ft/replay_public.py (CPU, no model, no downloads).

The generator is always a stub: every test injects a fake synth that records (text, ref)
calls and writes a short sine WAV, so the manifest writer, the round-robin / --voice
selection, the --only filter, the resume skip and the data_manifest.py compatibility are
all proven without torch, omnivoice or the 2.4 GB snapshot. The real adapter is only
exercised by the documented smoke run, never here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
for _p in (HERE / "omnivoice-ft", HERE / "ttstest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_manifest as DM  # noqa: E402
import replay_public as RP  # noqa: E402
import wavio  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_texts(path: Path, lines) -> str:
    """Write a texts JSONL whose rows carry extra fields the script must ignore
    (mood/rating/audio_codes/ref_audio/replay): proves only 'text' is read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(lines):
            row = {
                "id": f"replay-{i:04d}",
                "text": t,
                "audio": f"replay/wav/replay-{i:04d}.wav",
                "ref_audio": "libri/voices/84.wav",
                "replay": True,
                "mood": "calm",
                "rating": 5,
                "audio_codes": [[0, 1, 2]],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _write_voices(dir: Path, n: int):
    dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        wavio.write_wav(dir / f"{i}.wav", wavio.sine(0.02, 24000, 200.0 + i), 24000)


def _stub(calls):
    """Fake generator: records (text, ref_wav) and writes a short sine WAV."""

    def synth(text, ref_wav):
        calls.append((text, str(ref_wav)))
        return wavio.sine(0.02, 24000, 300.0), 24000

    return synth


def _read_jsonl(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _precreate_wavs(out: Path, *ns):
    wav_dir = out / "replay-wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for n in ns:
        wavio.write_wav(wav_dir / f"replay-{n:04d}.wav", wavio.sine(0.02, 24000, 300.0), 24000)
    return wav_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class ReplayPublicTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp_path = Path(td.name)

    def test_manifest_schema_ids_and_label_blind(self):
        texts = ['"Or like Sancho Panza\'s island," said North.', "A second line."]
        _write_texts(self.tmp_path / "texts.jsonl", texts)
        voices_dir = self.tmp_path / "voices"
        _write_voices(voices_dir, 2)
        calls = []
        manifest = RP.generate(self.tmp_path / "out", texts,
                               sorted(voices_dir.glob("*.wav")), synth=_stub(calls))
        rows = _read_jsonl(manifest)
        assert [r["id"] for r in rows] == ["replay-0000", "replay-0001"]
        for i, row in enumerate(rows):
            # exact schema: these five fields and nothing else (no mood/rating/... leak)
            assert set(row) == {"id", "audio_path", "text", "language_id", "replay"}
            assert row["text"] == texts[i]
            assert row["language_id"] == "en"
            assert row["replay"] is True
            p = Path(row["audio_path"])
            assert p.is_absolute() and p.is_file() and p.stat().st_size > 44
            assert p == (self.tmp_path / "out" / "replay-wavs" / f"replay-{i:04d}.wav").resolve()
        # the stub saw the exact texts and two different refs (round-robin over 2 voices)
        assert [c[0] for c in calls] == texts
        assert len({c[1] for c in calls}) == 2

    def test_round_robin_and_voice_flag(self):
        texts = ["a", "b", "c", "d"]
        voices_dir = self.tmp_path / "voices"
        _write_voices(voices_dir, 3)
        voices = sorted(voices_dir.glob("*.wav"))
        calls = []
        RP.generate(self.tmp_path / "out", texts, voices, synth=_stub(calls))
        assert [Path(c[1]).name for c in calls] == ["0.wav", "1.wav", "2.wav", "0.wav"]
        calls2 = []
        RP.generate(self.tmp_path / "out2", texts, voices, synth=_stub(calls2), voice=1)
        assert [Path(c[1]).name for c in calls2] == ["1.wav"] * 4

    def test_read_texts_only_field_and_limit(self):
        p = self.tmp_path / "texts.jsonl"
        _write_texts(p, ["one", "two", "three"])
        assert RP.read_texts(p) == ["one", "two", "three"]
        assert RP.read_texts(p, 2) == ["one", "two"]
        bad = self.tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"no_text": "x"}) + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            RP.read_texts(bad)

    def test_main_only_wiring_never_loads_model(self):
        # Both wavs already exist, so main() must not touch the real adapter even for --only 1.
        out = self.tmp_path / "out"
        _precreate_wavs(out, 0, 1)
        _write_texts(self.tmp_path / "texts.jsonl", ["one", "two"])
        voices_dir = self.tmp_path / "voices"
        _write_voices(voices_dir, 1)
        with mock.patch.object(
                RP, "load_omnivoice",
                side_effect=lambda *a, **k: self.fail("real adapter must not load when wavs exist")):
            code = RP.main(["--out-dir", str(out), "--texts", str(self.tmp_path / "texts.jsonl"),
                            "--voices", str(voices_dir), "--device", "cpu", "--only", "1"])
        assert code == 0
        rows = _read_jsonl(out / "replay_public.jsonl")
        assert [r["id"] for r in rows] == ["replay-0000"]

    def test_resume_skips_existing_wavs(self):
        texts = ["one", "two", "three"]
        out = self.tmp_path / "out"
        pre = _precreate_wavs(out, 0)  # only line 0 exists
        pre_bytes = (pre / "replay-0000.wav").read_bytes()
        voices_dir = self.tmp_path / "voices"
        _write_voices(voices_dir, 2)
        calls = []
        manifest = RP.generate(out, texts, sorted(voices_dir.glob("*.wav")), synth=_stub(calls))
        assert [c[0] for c in calls] == ["two", "three"]  # line 0 skipped
        assert (pre / "replay-0000.wav").read_bytes() == pre_bytes  # untouched
        assert [r["id"] for r in _read_jsonl(manifest)] == ["replay-0000", "replay-0001", "replay-0002"]

    def test_manifest_consumable_by_data_manifest(self):
        # End-to-end schema check: data_manifest.py --replay-jsonl accepts the produced manifest.
        texts = ["A replay line."]
        voices_dir = self.tmp_path / "voices"
        _write_voices(voices_dir, 1)
        calls = []
        manifest = RP.generate(self.tmp_path / "replays", texts,
                               sorted(voices_dir.glob("*.wav")), synth=_stub(calls))
        pool = self.tmp_path / "pool"
        pool.mkdir()
        wavio.write_wav(pool / "main.wav", wavio.sine(0.02, 24000, 300.0), 24000)
        texts_p = pool / "texts.jsonl"
        texts_p.write_text(json.dumps({"id": "main", "text": "A main line."}) + "\n", encoding="utf-8")
        out = self.tmp_path / "built.jsonl"
        summary = DM.build_manifest(pool_dir=str(pool), texts_jsonls=[str(texts_p)],
                                    replay_jsonls=[str(manifest)], out=str(out))
        rows = _read_jsonl(out)
        assert summary["n_replay"] == 1 and summary["n_rows"] == 2
        assert {r["id"] for r in rows} == {"main", "replay-0000"}
        replay_row = next(r for r in rows if r["id"] == "replay-0000")
        assert replay_row["replay"] is True and replay_row["text"] == texts[0]


    def test_replay_public_count_guard(self):
        texts = [f"Line number {i}." for i in range(300)]
        voices_dir = self.tmp_path / "voices"
        _write_voices(voices_dir, 2)
        voices = sorted(voices_dir.glob("*.wav"))
        bad = {3, 50, 120, 200, 299}

        def flaky(text, ref):
            if int(text.split()[-1].rstrip(".")) in bad:
                raise RuntimeError("synth blew up")
            return wavio.sine(0.02, 24000, 300.0), 24000

        with self.assertRaises(SystemExit) as cm:  # 5 of 300 is 1.67 %, over the 1 % limit
            RP.generate(self.tmp_path / "out", texts, voices, synth=flaky)
        self.assertIn("5 of 300 requested lines failed", str(cm.exception))
        # 3 of 300 is exactly 1 %: allowed, and the manifest holds the 297 that worked.
        bad.clear()
        bad.update({1, 2, 3})
        manifest = RP.generate(self.tmp_path / "out2", texts, voices, synth=flaky)
        self.assertEqual(len(_read_jsonl(manifest)), 297)


if __name__ == "__main__":
    unittest.main()
