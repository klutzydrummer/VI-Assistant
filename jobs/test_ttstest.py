"""Tests for jobs/ttstest (verify, blind, generate with the stub, pick_refs rules).
Run with `python3 -m unittest discover -s jobs`. Standard library only; no network, no GPU."""

import json
import os
import subprocess
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
import worker_run  # noqa: E402

HERE_TT = Path(__file__).resolve().parent / "ttstest"
LINES = json.loads((Path(__file__).resolve().parent / "ttstest" / "lines.json").read_text())
STUB = {m: generate.load_stub for m in ("omnivoice", "moss15", "qwen3base", "stub")}


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
        for m in ("omnivoice", "moss15", "qwen3base"):
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
        self.assertEqual(sorted(p.name for p in g.glob("*.wav")), ["A.wav", "B.wav", "C.wav"])
        self.assertEqual(sorted(key["vT/L01"].values()), ["moss15", "omnivoice", "qwen3base"])
        rmss = []
        for p in g.glob("*.wav"):
            x, sr = wavio.read_wav(p)
            self.assertEqual(sr, 24000)
            rmss.append(wavio.rms(x))
        self.assertLess(max(rmss) - min(rmss), 0.002)
        self.assertEqual(json.loads((out / "listen-key" / "key.json").read_text())["groups"], key)
        self.assertTrue((out / "listen-blind" / "refs" / "vT.wav").is_file())
        # letters are shuffled: not the same order in every group
        self.assertGreater(len({tuple(key[k][c] for c in "ABC") for k in key}), 1)
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
        self.assertEqual(len(rows) - 1, 2 * 13 * 3)
        lines = {ln["id"]: ln["text"] for ln in json.loads(blind.HERE.joinpath("lines.json").read_text())}
        for r in rows[1:]:
            self.assertEqual(len(r), len(rows[0]))
            self.assertEqual(r[2], lines[r[0].split("/")[1][:3]])
            self.assertEqual(r[3:-1], [""] * (len(r) - 4))
            self.assertIn(r[1], key[r[0]])
            line = r[0].split("/")[1][:3]
            want = {"L05": "angry", "L01": "neutral"}.get(line, "no tone target" if line in
                                                          ("L09", "P01", "F01", "F02", "F03") else None)
            if want:
                self.assertEqual(r[-1], want)
            self.assertTrue(r[-1])
        self.assertEqual(rows[0][-1], "intended_tone")
        text = (out / "listen-blind" / "sheet.csv").read_text()
        for m in ("omnivoice", "moss15", "qwen3base"):
            self.assertNotIn(m, text)
            self.assertNotIn(m, " ".join(str(p) for p in (out / "listen-blind").rglob("*")))
        self.assertFalse((out / "listen-blind" / "key.json").exists())

    def test_new_model_dirs_are_picked_up(self):
        loaders = STUB
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            refs = make_refs(root)
            for m in ("stub", "qwen3base"):
                self.assertEqual(gen(root, refs, m, "T", loaders=loaders)[0], 0)
            key = blind.build(root / "raw", root / "two")
            self.assertEqual(sorted(key["vT/L01"].values()), ["qwen3base", "stub"])
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


class WorkerRunTest(unittest.TestCase):
    def test_models_and_clip_counts(self):
        self.assertEqual(sorted(generate.MODELS), ["moss15", "omnivoice", "qwen3base", "stub"])
        self.assertEqual(worker_run.MODELS, ("omnivoice", "moss15", "qwen3base"))
        self.assertEqual(len(worker_run.plan(worker_run.MODELS, False)) * len(LINES), 78)
        self.assertEqual(len(worker_run.plan(worker_run.MODELS, True)) * len(LINES), 78 + 39)
        self.assertEqual(sorted({(v, s) for _, v, s in worker_run.plan(worker_run.MODELS, True)}),
                         [("P", 1), ("T", 1), ("T", 2)])

    def test_vt_is_deleted_with_its_transcript(self):
        with tempfile.TemporaryDirectory() as t:
            refs = make_refs(t)
            info = {}
            worker_run.delete_vt(refs, info)
            self.assertTrue(info["vT_deleted"])
            self.assertFalse((refs / "vT.wav").exists())
            self.assertTrue((refs / "vP.wav").exists())
            self.assertNotIn("vT", json.loads((refs / "refs.json").read_text()))

    def test_no_hf_token_reaches_a_model_process(self):
        os.environ["HF_TOKEN"] = "secret"
        try:
            self.assertNotIn("HF_TOKEN", worker_run.clean_env(False))
        finally:
            del os.environ["HF_TOKEN"]

    def test_download_retries_three_times_then_gives_up(self):
        with tempfile.TemporaryDirectory() as t:
            worker_run.BACKOFF, old = (0, 0, 0), worker_run.BACKOFF
            try:
                cmd = [sys.executable, "-c", "import sys; open(sys.argv[1], 'a').write('x'); sys.exit(1)", str(Path(t) / "n")]
                with self.assertRaises(worker_run.CmdFailed):
                    worker_run.with_retries("m", [cmd], Path(t) / "log", dict(os.environ), __import__("time").time() + 60)
                self.assertEqual((Path(t) / "n").read_text(), "xxx")
            finally:
                worker_run.BACKOFF = old

    def test_download_time_limit_stops_a_hung_command(self):
        import time
        with tempfile.TemporaryDirectory() as t:
            t0 = time.time()
            with self.assertRaises(worker_run.CmdFailed):
                worker_run.with_retries("m", [[sys.executable, "-c", "import time; time.sleep(60)"]], Path(t) / "log",
                                        dict(os.environ), time.time() + 1)
            self.assertLess(time.time() - t0, 10)


class SpeedGateAndKeepaliveTest(unittest.TestCase):
    """Phase 7 fixes: the idle keepalive, the download budget and the timed speed gate (no network, no GPU)."""

    def setUp(self):
        self.saved = dict(worker_run.KEEPALIVE)
        self.env = dict(os.environ)
        self.addCleanup(self.restore)

    def restore(self):
        worker_run.KEEPALIVE.update(self.saved)
        os.environ.clear()
        os.environ.update(self.env)

    def test_budget_arithmetic(self):
        # 32 GB at 10 MB/s: 32*1024/10*1.25 + 300 s = 4396 s = 73.3 min, inside the 75 min cap
        self.assertAlmostEqual(worker_run.download_seconds(32, 10), 4396.0, places=0)
        self.assertEqual(worker_run.download_budget(32, 10, 75 * 60), 4396.0)
        # 6.5 MB/s (the failed rental): 6600 s, capped at 75 min
        self.assertEqual(worker_run.download_budget(32, 6.5, 75 * 60), 75 * 60)
        # very fast: never below the 20 min floor
        self.assertEqual(worker_run.download_budget(32, 500, 75 * 60), 1200)

    def test_measure_speed_uses_the_slowest_url_and_counts_failures_as_zero(self):
        class Body:
            def __init__(self, n, delay):
                self.left, self.delay = n, delay
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self, n):
                import time
                time.sleep(self.delay)
                k = min(n, self.left, 1 << 20)
                self.left -= k
                return b"x" * k

        def opener(req, timeout):
            u = req.full_url
            if "boom" in u:
                raise OSError("down")
            return Body(8 << 20, 0.0 if "fast" in u else 0.2)

        mbps, per = worker_run.measure_speed(["http://x/fast", "http://x/slow"], 3, 8 << 20, opener=opener)
        self.assertEqual(mbps, min(per.values()))
        self.assertGreater(per["http://x/fast"], per["http://x/slow"] * 5)
        mbps, per = worker_run.measure_speed(["http://x/fast", "http://x/boom"], 3, 8 << 20, opener=opener)
        self.assertEqual(mbps, 0.0)

    def test_keepalive_is_emitted_during_a_long_command_and_stops_after(self):
        import time
        with tempfile.TemporaryDirectory() as t:
            ka = Path(t) / ".keepalive"
            worker_run.KEEPALIVE.update(path=ka, every=0.05)
            seen = []
            import threading
            stop = threading.Event()

            def watch():
                last = None
                while not stop.is_set():
                    if ka.exists():
                        m = ka.stat().st_mtime_ns
                        if m != last:
                            seen.append(m)
                            last = m
                    time.sleep(0.01)

            w = threading.Thread(target=watch)
            w.start()
            # A quiet command that writes nothing for 1.5 s: with a 0.05 s beat the file must refresh many times,
            # i.e. an idle limit of even a few beats could never be reached.
            worker_run.run_cmd([sys.executable, "-c", "import time; time.sleep(1.5)"], 30, Path(t) / "log")
            n_during = len(seen)
            time.sleep(0.3)
            n_after = len(seen)
            stop.set()
            w.join()
            self.assertGreaterEqual(n_during, 10, seen)
            self.assertLessEqual(n_after - n_during, 1, "the keepalive must stop when the command ends")
            age = time.time() - ka.stat().st_mtime
            self.assertGreater(age, 0.2)

    def test_keepalive_stops_when_a_hung_command_is_killed_at_the_deadline(self):
        import time
        with tempfile.TemporaryDirectory() as t:
            ka = Path(t) / ".keepalive"
            worker_run.KEEPALIVE.update(path=ka, every=0.05)
            with self.assertRaises(worker_run.CmdFailed):
                worker_run.run_cmd([sys.executable, "-c", "import time; time.sleep(60)"], 1, Path(t) / "log")
            m = ka.stat().st_mtime_ns
            time.sleep(0.3)
            self.assertEqual(ka.stat().st_mtime_ns, m)

    def run_job_with(self, mbps, **env):
        """run_job in stub mode with a faked speed measurement; returns (rc, prepare calls, run-info)."""
        import io
        os.environ.update(TTSTEST_SPEED_URLS="http://example.invalid/x", TTSTEST_MIN_MBPS="10", **env)
        calls = []
        orig_m, orig_p = worker_run.measure_speed, worker_run.prepare
        worker_run.measure_speed = lambda *a, **k: (mbps, {"http://example.invalid/x": mbps})
        worker_run.prepare = lambda *a, **k: calls.append(a[0]) or (sys.executable, [])
        try:
            with tempfile.TemporaryDirectory() as t:
                refs = make_refs(t)
                out = Path(t) / "job" / "out"
                a = worker_run.argparse.Namespace(out=str(out), refs=str(refs), keep=str(Path(t) / "keep"),
                                                  models="moss15", seed2=False, stub=True)
                orig_push = worker_run.push_folder, worker_run.push_top
                worker_run.push_folder = lambda *a, **k: None
                worker_run.push_top = lambda *a, **k: None
                try:
                    rc = worker_run.run_job(a)
                finally:
                    worker_run.push_folder, worker_run.push_top = orig_push
                info = json.loads((out / "run-info.json").read_text())
        finally:
            worker_run.measure_speed, worker_run.prepare = orig_m, orig_p
        return rc, calls, info

    def test_slow_download_fails_fast_before_any_venv_or_weights(self):
        rc, calls, info = self.run_job_with(6.5)
        self.assertEqual(rc, worker_run.EXIT_SLOW)
        self.assertEqual(rc, 5)
        self.assertEqual(calls, [], "prepare() (venv + weights) must not start")
        self.assertIn("too slow", info["models"]["moss15"]["reason"])
        self.assertEqual(info["speed_gate"]["mbps"], 6.5)
        self.assertEqual(info["exit"], 5)

    def test_unreadable_speed_url_also_fails_the_gate(self):
        rc, calls, info = self.run_job_with(0.0)
        self.assertEqual((rc, calls), (5, []))

    def test_normal_speed_passes_and_sets_a_derived_budget(self):
        rc, calls, info = self.run_job_with(20.0)
        self.assertEqual(calls, ["moss15"])
        self.assertNotEqual(rc, 5)
        # 32 GB at 20 MB/s: 32*1024/20*1.25 + 300 = 2348 s = 39.1 min
        self.assertEqual(info["models"]["moss15"]["download_budget_min"], 39.1)

    def test_every_model_failing_setup_exits_6(self):
        import io
        os.environ.update(TTSTEST_SPEED_URLS="http://example.invalid/x", TTSTEST_MIN_MBPS="10")
        orig_m, orig_p = worker_run.measure_speed, worker_run.prepare

        def boom(*a, **k):
            raise worker_run.CmdFailed("torchcodec import failed")

        worker_run.measure_speed = lambda *a, **k: (50.0, {})
        worker_run.prepare = boom
        try:
            with tempfile.TemporaryDirectory() as t:
                refs = make_refs(t)
                out = Path(t) / "job" / "out"
                a = worker_run.argparse.Namespace(out=str(out), refs=str(refs), keep=str(Path(t) / "keep"),
                                                  models="moss15", seed2=False, stub=True)
                orig_push = worker_run.push_folder, worker_run.push_top
                worker_run.push_folder = lambda *a, **k: None
                worker_run.push_top = lambda *a, **k: None
                try:
                    rc = worker_run.run_job(a)
                finally:
                    worker_run.push_folder, worker_run.push_top = orig_push
        finally:
            worker_run.measure_speed, worker_run.prepare = orig_m, orig_p
        self.assertEqual(rc, 6)


class RunOrReleaseTest(unittest.TestCase):
    """jobs/ttstest/run-or-release.sh with a fake `vi`: only exits 5 and 6 destroy the worker."""
    SCRIPT = Path(__file__).resolve().parent / "ttstest" / "run-or-release.sh"

    def run_with(self, job_rc, down_rc=0):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "state" / "unverified").mkdir(parents=True)
            ck = t / "state" / "ck"
            ck.mkdir()
            (t / "state" / "unverified" / "tts-abc").write_text(str(ck) + "\n")
            other = t / "state" / "other"
            other.mkdir()
            (t / "state" / "unverified" / "voicetune-xyz").write_text(str(other) + "\n")
            fake = t / "vi"
            fake.write_text(f"#!/bin/bash\necho \"$@\" >> {t}/vi.calls\nexit {down_rc}\n")
            fake.chmod(0o755)
            env = dict(os.environ, VI_BIN=str(fake), VI_STATE_DIR=str(t / "state"))
            r = subprocess.run(["bash", str(self.SCRIPT), "bash", "-c", f"exit {job_rc}"], env=env,
                               capture_output=True, text=True)
            calls = (t / "vi.calls").read_text().strip().splitlines() if (t / "vi.calls").exists() else []
            return (r.returncode, calls, (ck / "VERIFIED").exists(), (other / "VERIFIED").exists(),
                    (t / "state" / "unverified" / "tts-abc").exists(), r.stderr)

    def test_exit_5_and_6_destroy_the_worker(self):
        for rc in (5, 6):
            code, calls, verified, other, marker, err = self.run_with(rc)
            self.assertEqual((code, calls), (rc, ["cloud down"]), err)
            self.assertTrue(verified)
            self.assertFalse(marker)
            self.assertFalse(other, "another stage's checkpoint must be left alone")

    def test_other_exit_codes_never_destroy(self):
        for rc in (0, 2, 3, 4, 1):
            code, calls, *_ = self.run_with(rc)
            self.assertEqual((code, calls), (rc, []))

    def test_failed_release_is_loud(self):
        code, calls, *_ , err = self.run_with(5, down_rc=1)
        self.assertEqual(code, 70)
        self.assertIn("FAILED", err)


class SmokeGateTest(unittest.TestCase):
    """jobs/ttstest/smoke-gate.sh must fail without FFmpeg libs and never accept a smoke without its OK line."""
    SCRIPT = Path(__file__).resolve().parent / "ttstest" / "smoke-gate.sh"

    def run_gate(self, smoke_cmd, with_libs, extra=None):
        with tempfile.TemporaryDirectory() as t:
            libs = Path(t) / "lib"
            libs.mkdir()
            if with_libs:
                for n in ("libavcodec.so.61", "libavutil.so.59", "libswresample.so.5"):
                    (libs / n).write_text("")
            env = dict(os.environ, LD_LIBRARY_PATH=str(libs), SMOKE_CMD=smoke_cmd, TTSTEST_GATE_NO_NIX="1")
            env.update(extra or {})
            r = subprocess.run(["bash", str(self.SCRIPT)], env=env, capture_output=True, text=True)
            return r.returncode, r.stdout + r.stderr

    def ldconfig_has_ffmpeg(self):
        try:
            c = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
        except OSError:
            return False
        return all(x in c for x in ("libavcodec.so", "libavutil.so", "libswresample.so"))

    def test_fails_without_ffmpeg_libs_and_does_not_start_the_smoke(self):
        if self.ldconfig_has_ffmpeg():
            self.skipTest("this host has FFmpeg libs in the linker cache")
        with tempfile.TemporaryDirectory() as t:
            mark = Path(t) / "started"
            code, out = self.run_gate(f"touch {mark}; echo MOSS15 CHECK OK", with_libs=False)
            self.assertEqual(code, 3, out)
            self.assertIn("DO NOT RENT", out)
            self.assertFalse(mark.exists(), "the smoke must not run without the libs")

    def test_passes_with_libs_and_the_ok_line(self):
        code, out = self.run_gate("echo MOSS15 CHECK OK", with_libs=True)
        self.assertEqual(code, 0, out)
        self.assertIn("PASSED", out)

    def test_fails_when_the_smoke_fails_or_lacks_the_ok_line(self):
        code, out = self.run_gate("echo boom; exit 1", with_libs=True)
        self.assertEqual(code, 1, out)
        code, out = self.run_gate("echo all fine", with_libs=True)
        self.assertEqual(code, 1, out)
        self.assertIn("DO NOT RENT", out)


class ReapLoopTest(unittest.TestCase):
    """reap-loop.sh must not need `vi` on PATH, and must fail loudly at the start when it finds none."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        (self.root / "jobs" / "ttstest").mkdir(parents=True)
        self.script = self.root / "jobs" / "ttstest" / "reap-loop.sh"
        self.script.write_text((HERE_TT / "reap-loop.sh").read_text())

    def tearDown(self):
        self._t.cleanup()

    def fake_vi(self, path, tag):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\necho {tag} \"$@\"\nkill -TERM $PPID\n")
        path.chmod(0o755)

    def run_loop(self, **env):
        e = {"PATH": "/usr/bin:/bin:" + os.path.dirname(sys.executable), "REAP_INTERVAL": "1", **env}
        return subprocess.run(["bash", str(self.script)], env=e, capture_output=True, text=True, timeout=30)

    def test_none_found_fails_loudly(self):
        r = self.run_loop()
        self.assertEqual(r.returncode, 2)
        self.assertIn("no usable `vi`", r.stderr)
        self.assertNotIn("reaping", r.stdout)

    def test_repo_release_binary_is_used(self):
        self.fake_vi(self.root / "target" / "release" / "vi", "RELEASE")
        r = self.run_loop()
        self.assertIn("RELEASE cloud reap", r.stdout)

    def test_vi_bin_wins(self):
        self.fake_vi(self.root / "target" / "release" / "vi", "RELEASE")
        self.fake_vi(self.root / "elsewhere" / "myvi", "ENVBIN")
        r = self.run_loop(VI_BIN=str(self.root / "elsewhere" / "myvi"))
        self.assertIn("ENVBIN cloud reap", r.stdout)
        self.assertNotIn("RELEASE", r.stdout)

    def test_bad_vi_bin_fails_loudly(self):
        r = self.run_loop(VI_BIN=str(self.root / "nope"))
        self.assertEqual(r.returncode, 2)


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
