"""Tests for jobs/ttstest (verify, blind, generate with the stub, pick_refs rules).
Run with `python3 -m unittest discover -s jobs`. Standard library only; no network, no GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ttstest"))

import blind  # noqa: E402
import generate  # noqa: E402
import pick_refs  # noqa: E402
import verify  # noqa: E402
import wavio  # noqa: E402

LINES = json.loads((Path(__file__).resolve().parent / "ttstest" / "lines.json").read_text())
STUB = {m: generate.load_stub for m in ("omnivoice", "moss15", "qwen3base", "mossnano", "stub")}


def make_refs(root):
    refs = Path(root) / "refs"
    refs.mkdir()
    for v in "TP":
        wavio.write_wav(refs / f"v{v}.wav", wavio.sine(6.5, 24000, 150 + 40 * (v == "P")), 24000)
    (refs / "refs.json").write_text(json.dumps({"vT": {"id": "t", "text": "a b c."}, "vP": {"id": "p", "text": "d e f."}}))
    return refs


def gen(root, refs, model="omnivoice", voice="T", seed=1, loaders=STUB, only=None):
    return generate.generate(model, voice, seed, Path(root) / "raw", refs, LINES, only, loaders)


class LinesTest(unittest.TestCase):
    def test_thirteen_lines(self):
        self.assertEqual([x["id"] for x in LINES],
                         ["L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08", "L09", "P01", "F01", "F02", "F03"])
        self.assertTrue(all(x["text"].strip() for x in LINES))


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        self.refs = make_refs(self.root)
        self.raw = self.root / "raw"
        code, _ = gen(self.root, self.refs)
        self.assertEqual(code, 0)
        self.d = self.raw / "omnivoice__vT__s1"

    def tearDown(self):
        self._t.cleanup()

    def culprits(self, **kw):
        return verify.verify_all(self.raw, **kw)

    def test_good(self):
        self.assertEqual(self.culprits(expect=[("omnivoice", "T", 1)]), {})
        self.assertEqual(len(list(self.d.glob("*.wav"))), 13)
        self.assertEqual((self.d / "expected-count.txt").read_text().strip(), "13")
        self.assertEqual(verify.main([str(self.raw)]), 0)

    def test_flipped_byte(self):
        w = self.d / "omnivoice__vT__L03__s1.wav"
        b = bytearray(w.read_bytes())
        b[200] ^= 0xFF
        w.write_bytes(bytes(b))
        r = self.culprits()["omnivoice__vT__s1"]
        self.assertEqual([f for f, _ in r], [w.name])
        self.assertIn("SHA-256", r[0][1])
        self.assertEqual(verify.main([str(self.raw)]), 1)

    def test_missing_file(self):
        w = self.d / "omnivoice__vT__L05__s1.wav"
        w.unlink()
        probs = dict(self.culprits()["omnivoice__vT__s1"])
        self.assertIn("missing on disk", probs[w.name])
        self.assertIn("expected 13", probs["expected-count.txt"])

    def test_extra_file(self):
        wavio.write_wav(self.d / "omnivoice__vT__L99__s1.wav", wavio.sine(2, 24000), 24000)
        probs = dict(self.culprits()["omnivoice__vT__s1"])
        self.assertIn("not in manifest", probs["omnivoice__vT__L99__s1.wav"])

    def test_no_manifest(self):
        (self.d / "manifest.sha256").unlink()
        self.assertIn("manifest.sha256", dict(self.culprits()["omnivoice__vT__s1"]))

    def test_expected_folder_missing_or_extra(self):
        r = self.culprits(expect=[("omnivoice", "T", 1), ("moss15", "T", 1)])
        self.assertEqual(r["moss15__vT__s1"], [("(folder)", "expected but not found")])
        r = self.culprits(expect=[("moss15", "T", 1)])
        self.assertIn("omnivoice__vT__s1", r)

    def test_empty_raw(self):
        r = verify.verify_all(self.root / "nothing")
        self.assertTrue(r)

    def test_failed_run_info(self):
        info = json.loads((self.d / "run-info.json").read_text())
        info.update(ok=False, error="MemoryError: boom")
        (self.d / "run-info.json").write_text(json.dumps(info))
        self.assertIn("MemoryError", dict(self.culprits()["omnivoice__vT__s1"])["run-info.json"])


class WavChecksTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def check(self, samples, sr=24000):
        p = self.root / "x.wav"
        wavio.write_wav(p, samples, sr)
        return verify.check_wav(p)

    def test_ok(self):
        self.assertEqual(self.check(wavio.sine(3, 24000)), [])

    def test_silent(self):
        self.assertTrue(any("silent" in p for p in self.check([0.0] * 72000)))

    def test_too_short_and_too_long(self):
        self.assertTrue(any("duration" in p for p in self.check(wavio.sine(0.5, 24000))))
        self.assertTrue(any("duration" in p for p in self.check(wavio.sine(61, 16000), 16000)))

    def test_bad_sample_rate(self):
        self.assertTrue(any("sample rate" in p for p in self.check(wavio.sine(3, 8000), 8000)))

    def test_clipped(self):
        loud = [1.0 if (i // 50) % 2 else -1.0 for i in range(72000)]
        self.assertTrue(any("clipped" in p for p in self.check(loud)))

    def test_constant(self):
        self.assertTrue(any("constant" in p for p in self.check([0.5] * 72000)))

    def test_garbage_file(self):
        p = self.root / "bad.wav"
        p.write_bytes(b"not a wav at all")
        self.assertTrue(any("unreadable" in x for x in verify.check_wav(p)))

    def test_nan(self):
        import struct

        p = self.root / "nan.wav"
        n = 72000
        data = struct.pack("<%df" % n, *([float("nan")] * n))
        hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, 24000, 96000, 4, 32)
               + b"data" + struct.pack("<I", len(data)))
        p.write_bytes(hdr + data)
        probs = verify.check_wav(p)
        self.assertTrue(probs)  # NaN if soundfile reads float WAVs, else unreadable: either way flagged


class GenerateTest(unittest.TestCase):
    def test_failure_has_no_manifest_and_is_named(self):
        with tempfile.TemporaryDirectory() as t:
            refs = make_refs(t)

            def boom():
                raise MemoryError("out of memory")

            code, info = gen(t, refs, loaders={"omnivoice": boom})
            self.assertEqual(code, 1)
            d = Path(t) / "raw" / "omnivoice__vT__s1"
            self.assertFalse((d / "manifest.sha256").exists())
            self.assertIn("MemoryError", json.loads((d / "run-info.json").read_text())["error"])
            probs = dict(verify.verify_all(Path(t) / "raw")["omnivoice__vT__s1"])
            self.assertIn("manifest.sha256", probs)
            self.assertIn("MemoryError", probs["run-info.json"])

    def test_refuses_dirty_dir_and_cli_exit_code(self):
        with tempfile.TemporaryDirectory() as t:
            refs = make_refs(t)
            args = ["--model", "stub", "--voice", "T", "--seed", "1", "--out", str(Path(t) / "o"), "--refs", str(refs)]
            self.assertEqual(generate.main(args + ["--only", "L01,L02"]), 0)
            with self.assertRaises(SystemExit):
                generate.main(args)
            self.assertEqual(generate.main(["--model", "stub", "--voice", "P", "--seed", "1", "--out", t + "/o2",
                                            "--refs", str(Path(t) / "nope")]), 1)


class BlindTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        self.refs = make_refs(self.root)
        for m in ("omnivoice", "moss15", "qwen3base", "mossnano"):
            for v in "TP":
                self.assertEqual(gen(self.root, self.refs, m, v)[0], 0)
        self.raw = self.root / "raw"

    def tearDown(self):
        self._t.cleanup()

    def test_blind_set(self):
        self.assertEqual(verify.verify_all(self.raw), {})
        out = self.root / "listen"
        key = blind.build(self.raw, out, refs=self.refs)
        self.assertEqual(len(key), 2 * 13)
        g = out / "listen-blind" / "vT" / "L01"
        self.assertEqual(sorted(p.name for p in g.glob("*.wav")), ["A.wav", "B.wav", "C.wav", "D.wav"])
        self.assertEqual(sorted(key["vT/L01"].values()), ["mossnano", "moss15", "omnivoice", "qwen3base"][:0] or
                         sorted(["omnivoice", "moss15", "qwen3base", "mossnano"]))
        rmss = []
        for p in g.glob("*.wav"):
            x, sr = wavio.read_wav(p)
            self.assertEqual(sr, 24000)
            rmss.append(wavio.rms(x))
        self.assertLess(max(rmss) - min(rmss), 0.002)
        self.assertEqual(json.loads((out / "listen-key" / "key.json").read_text())["groups"], key)
        self.assertTrue((out / "listen-blind" / "refs" / "vT.wav").is_file())
        # letters are shuffled: not the same order in every group
        self.assertGreater(len({tuple(key[k][c] for c in "ABCD") for k in key}), 1)
        # raw untouched, still verifies
        self.assertEqual(verify.verify_all(self.raw), {})

    def test_sheet_matches_listen_format_and_hides_models(self):
        import csv
        out = self.root / "sheetset"
        key = blind.build(self.raw, out)
        with open(out / "listen-blind" / "sheet.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][:7], ["group", "clip", "sentence", "still_her_voice_y_n",
                                      "mood_heard_0_3", "words_clear_y_n", "notes"])
        self.assertEqual(len(rows) - 1, 2 * 13 * 4)
        lines = {ln["id"]: ln["text"] for ln in json.loads(blind.HERE.joinpath("lines.json").read_text())}
        for r in rows[1:]:
            self.assertEqual(len(r), len(rows[0]))
            self.assertEqual(r[2], lines[r[0].split("/")[1][:3]])
            self.assertEqual(r[3:], [""] * (len(r) - 3))
            self.assertIn(r[1], key[r[0]])
        text = (out / "listen-blind" / "sheet.csv").read_text()
        for m in ("omnivoice", "moss15", "qwen3base", "mossnano"):
            self.assertNotIn(m, text)
            self.assertNotIn(m, " ".join(str(p) for p in (out / "listen-blind").rglob("*")))
        self.assertFalse((out / "listen-blind" / "key.json").exists())

    def test_new_model_dirs_are_picked_up(self):
        loaders = STUB
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            refs = make_refs(root)
            for m in ("stub", "mossnano"):
                self.assertEqual(gen(root, refs, m, "T", loaders=loaders)[0], 0)
            key = blind.build(root / "raw", root / "two")
            self.assertEqual(sorted(key["vT/L01"].values()), ["mossnano", "stub"])
            self.assertEqual(sorted(p.name for p in (root / "two/listen-blind/vT/L01").glob("*.wav")),
                             ["A.wav", "B.wav"])

    def test_nested_layout_raw_model_run_clip(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"
            for m in ("modela", "modelb"):
                d = raw / m / f"{m}__vT__s1"
                d.mkdir(parents=True)
                wavio.write_wav(d / f"{m}__vT__L01__s1.wav", wavio.sine(1.0, 24000, 200.0), 24000)
                (d / "manifest.sha256").write_text("x")
                (d / "run-info.json").write_text("{}")
            key = blind.build(raw, Path(t) / "out")
            self.assertEqual(sorted(key["vT/L01"].values()), ["modela", "modelb"])

    def test_deterministic_and_no_overwrite(self):
        a, b = self.root / "a", self.root / "b"
        ka, kb = blind.build(self.raw, a), blind.build(self.raw, b)
        self.assertEqual(ka, kb)
        self.assertEqual(blind.build(self.raw, self.root / "c", seed=2) == ka, False)
        with self.assertRaises(SystemExit):
            blind.build(self.raw, a)

    def test_second_seed_group_is_separate(self):
        self.assertEqual(gen(self.root, self.refs, "omnivoice", "T", seed=2)[0], 0)
        key = blind.build(self.raw, self.root / "d")
        self.assertIn("vT/L01_s2", key)
        self.assertEqual(list(key["vT/L01_s2"].values()), ["omnivoice"])

    def test_level_match_respects_peak(self):
        scaled, target = blind.level_match([[0.9, -0.9] * 100, [0.01, -0.01] * 100])
        self.assertLessEqual(max(abs(v) for c in scaled for v in c), blind.PEAK_MAX + 1e-9)
        self.assertAlmostEqual(wavio.rms(scaled[0]), wavio.rms(scaled[1]))


class PickRefsTest(unittest.TestCase):
    def test_single_sentence(self):
        ok = pick_refs.single_sentence
        self.assertTrue(ok("I did not expect that."))
        self.assertTrue(ok("Wait... what is that?"))
        self.assertFalse(ok("One. Two."))
        self.assertFalse(ok("no end punctuation"))
        self.assertFalse(ok(""))

    def test_rule_and_determinism(self):
        rows = [
            {"id": "a", "game": "le2", "duration_s": 7.0, "text": "Fine, then."},
            {"id": "b", "game": "le1", "duration_s": 7.0, "text": "Fine, then."},
            {"id": "c", "game": "le3", "duration_s": 5.0, "text": "Fine, then."},
            {"id": "d", "game": "le3", "duration_s": 8.0, "text": "Fine. Then."},
            {"id": "e", "game": "le3", "duration_s": 8.0, "text": "Fine, then?", "mood": "angry"},
            {"id": "f", "game": "le3", "duration_s": 8.0, "text": "Fine, then?"},
        ]
        c = pick_refs.tali_candidates(rows, {"f"})
        self.assertEqual([r["id"] for r in c], ["a", "e"])
        p1 = pick_refs.pick_lowest_hash(c, pick_refs.SEED)
        p2 = pick_refs.pick_lowest_hash(list(reversed(c)), pick_refs.SEED)
        self.assertEqual(p1["id"], p2["id"])

    def test_female_speakers(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "speakers.tsv"
            p.write_text("READER\tGENDER\tSUBSET\tNAME\n19\tF\ttrain-clean-100\tA\n26\tM\ttrain-clean-100\tB\n")
            self.assertEqual(pick_refs.load_female_speakers(p), {"19"})


if __name__ == "__main__":
    unittest.main()
