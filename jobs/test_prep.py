"""Tests for prep.py. Needs numpy, scipy and soundfile (CI installs them).
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

import prep

JOBS = Path(__file__).resolve().parent
RATE = 24_000


def tone(seconds, amp=0.3, freq=220.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def hiss(seconds, amp, seed, rate=RATE):
    return amp * np.random.default_rng(seed).standard_normal(int(seconds * rate))


class LoudnessTest(unittest.TestCase):
    def test_k_weighting_matches_the_published_48k_coefficients(self):
        (b, a), (hb, ha) = prep.k_weighting(48_000)
        np.testing.assert_allclose(b, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-9)
        np.testing.assert_allclose(a, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-9)
        np.testing.assert_allclose(ha, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-9)
        self.assertEqual(hb, [1.0, -2.0, 1.0])

    def test_full_scale_1khz_sine_is_minus_3_lufs(self):
        # BS.1770's own reference point.
        lufs = prep.integrated_loudness(tone(5, amp=1.0, freq=1000.0, rate=48_000), 48_000)
        self.assertAlmostEqual(lufs, -3.01, delta=0.05)

    def test_gating_ignores_silence(self):
        loud = tone(3, amp=0.1, freq=1000.0)
        padded = np.concatenate([np.zeros(RATE * 3), loud])
        self.assertAlmostEqual(prep.integrated_loudness(padded, RATE), prep.integrated_loudness(loud, RATE), delta=0.5)


class PrepareTest(unittest.TestCase):
    def test_trim_keeps_the_voice_and_measures_the_silence_it_cut(self):
        noise = 1e-3
        clip = np.concatenate([hiss(0.5, noise, 1), tone(1.5) + hiss(1.5, noise, 2), hiss(0.5, noise, 3)])
        r = prep.prepare(clip, RATE)
        self.assertAlmostEqual(r["trim"]["start_s"], 0.5, delta=0.1)
        self.assertAlmostEqual(r["trim"]["end_s"], 2.0, delta=0.1)
        self.assertAlmostEqual(r["noise_floor"]["level_db"], 20 * np.log10(noise), delta=1.0)
        self.assertEqual(len(r["noise_floor"]["spectrum"]), 16)

    def test_noise_floor_tells_two_recording_chains_apart(self):
        voice = tone(1.5)
        quiet = prep.prepare(np.concatenate([hiss(0.4, 1e-5, 1), voice, hiss(0.4, 1e-5, 2)]), RATE)
        loud = prep.prepare(np.concatenate([hiss(0.4, 1e-3, 1), voice, hiss(0.4, 1e-3, 2)]), RATE)
        self.assertGreater(loud["noise_floor"]["level_db"] - quiet["noise_floor"]["level_db"], 30)

    def test_no_floor_without_trimmed_silence(self):
        self.assertIsNone(prep.prepare(tone(1.5), RATE)["noise_floor"])

    def test_floor_from_tight_edges_without_speech_in_it(self):
        # 40 ms of hiss at each end: the trim's centred 85 ms frames see
        # speech there, the floor's 20 ms frames don't.
        noise = 1e-3
        clip = np.concatenate([hiss(0.04, noise, 1), tone(1.5), hiss(0.04, noise, 2)])
        start, end = prep.trim_bounds(clip, 40)
        self.assertEqual((start, end), (0, len(clip)))
        r = prep.prepare(clip, RATE)
        self.assertIsNotNone(r["noise_floor"])
        self.assertAlmostEqual(r["noise_floor"]["level_db"], 20 * np.log10(noise), delta=1.0)
        self.assertAlmostEqual(r["noise_floor"]["seconds"], 0.08, delta=0.021)

    def test_pauses_mid_line_never_give_a_floor(self):
        clip = np.concatenate([tone(0.8), hiss(0.3, 1e-3, 1), tone(0.8)])
        self.assertIsNone(prep.prepare(clip, RATE)["noise_floor"])

    def test_one_linear_gain_to_the_target(self):
        for amp in (0.01, 0.1):
            r = prep.prepare(tone(3, amp=amp), RATE, target_lufs=-23)
            out = r["audio16k"].astype(np.float64)
            self.assertAlmostEqual(prep.integrated_loudness(out, 16_000), -23, delta=0.3)
            # Linear: the envelope's shape is untouched, only scaled.
            self.assertAlmostEqual(r["gain_db"], -23 - r["level"]["db"], places=2)
        self.assertEqual(r["level"]["method"], "bs1770")

    def test_short_clips_use_rms(self):
        r = prep.prepare(tone(0.6, amp=0.1), RATE)
        self.assertEqual(r["level"]["method"], "rms")

    def test_gain_is_capped_below_clipping(self):
        # A spiky clip: quiet on average, with one near-full-scale click.
        clip = tone(3, amp=0.01)
        clip[RATE] = 0.9
        r = prep.prepare(clip, RATE, target_lufs=-10, max_peak_dbfs=-1)
        self.assertTrue(r["gain_capped"])
        self.assertLessEqual(r["level"]["peak_dbfs"] + r["gain_db"], -1 + 1e-6)

    def test_output_is_16k_and_the_right_length(self):
        r = prep.prepare(tone(2.0), RATE)
        self.assertAlmostEqual(len(r["audio16k"]) / 16_000, r["duration_s"], delta=0.01)

    def test_silence_trims_to_nothing(self):
        r = prep.prepare(np.zeros(RATE), RATE)
        self.assertEqual(r["duration_s"], 0)


class JobTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            harvest = d / "harvest"
            (harvest / "audio").mkdir(parents=True)
            lines = []
            voiced = np.concatenate([hiss(0.3, 1e-4, 1), tone(1.2), hiss(0.3, 1e-4, 2)])
            for i, x in enumerate([voiced, np.zeros(RATE)]):
                lid = f"le2.s{i}_f"
                sf.write(harvest / "audio" / f"{lid}.wav", x, RATE, subtype="PCM_16")
                lines.append({"id": lid, "text": "Keelah.", "audio": {"path": f"audio/{lid}.wav", "duration_s": len(x) / RATE},
                              "source": {"game": "le2"}})  # fmt: skip
            (harvest / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
            (d / "settings.json").write_text("{}")
            env = {**os.environ, "VI_INPUT_HARVEST": str(harvest), "VI_OUT": str(d / "out"),
                   "VI_SETTINGS": str(d / "settings.json")}  # fmt: skip
            subprocess.run([sys.executable, str(JOBS / "prep.py")], env=env, check=True, capture_output=True)
            recs = [json.loads(x) for x in (d / "out" / "prep.jsonl").read_text().splitlines()]
            self.assertEqual([r["flags"] for r in recs], [[], ["silent", "no_noise_floor"]])
            audio, rate = sf.read(d / "out" / recs[0]["audio16k"]["path"])
            self.assertEqual(rate, 16_000)
            self.assertNotIn("audio16k", recs[1])
            self.assertTrue((d / "out" / "checks.json").exists())
            # Later stages see prep's trimmed duration and its 16 kHz copy.
            sys.path.insert(0, str(JOBS))
            import vi_job

            old = dict(os.environ)
            try:
                os.environ.update({"VI_INPUT_HARVEST": str(harvest), "VI_INPUT_PREP": str(d / "out"),
                                   "VI_SETTINGS": str(d / "settings.json")})  # fmt: skip
                clips = vi_job.load_clips()
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertEqual([c.id for c in clips], ["le2.s0_f"])
            self.assertEqual(clips[0].duration_s, recs[0]["duration_s"])
            self.assertTrue(clips[0].audio16k.exists())


class EqMatchTest(unittest.TestCase):
    def test_the_correction_changes_colour_but_not_the_cut(self):
        import eq_match
        import filter_probe as fp

        clip = np.concatenate([hiss(0.4, 1e-4, 1), hiss(1.5, 0.1, 2), hiss(0.4, 1e-4, 3)])
        g = fp.grid()
        dark = np.where(g > 2000, -10.0, 0.0)  # take 10 dB off the top
        plain, eq = prep.prepare(clip, RATE), prep.prepare(clip, RATE, fir=eq_match.fir(dark, RATE))
        self.assertEqual(plain["trim"], eq["trim"])
        self.assertEqual(plain["noise_floor"], eq["noise_floor"])
        for r in (plain, eq):  # both levelled to the target
            self.assertFalse(r["gain_capped"])
            self.assertAlmostEqual(r["level"]["db"] + r["gain_db"], -23, delta=0.01)

        def top_share(x):
            p = np.abs(np.fft.rfft(x)) ** 2
            return p[np.fft.rfftfreq(len(x), 1 / 16_000) > 3000].sum() / p.sum()

        self.assertLess(top_share(eq["audio16k"]), 0.3 * top_share(plain["audio16k"]))

    def test_end_to_end_fits_one_correction_per_game(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            harvest = d / "harvest"
            (harvest / "audio").mkdir(parents=True)
            lines = []
            rng = np.random.default_rng(0)
            for game in ("le1", "le2"):
                for i in range(12):
                    x = hiss(1.0 + 0.1 * i, 0.1, int(rng.integers(1 << 30)))
                    if game == "le1":
                        from scipy import signal

                        x = signal.sosfilt(signal.butter(2, 1500, fs=RATE, output="sos"), x)
                    lid = f"{game}.s{i}_f"
                    sf.write(harvest / "audio" / f"{lid}.wav", x, RATE, subtype="PCM_16")
                    lines.append({"id": lid, "text": "Keelah se'lai.", "source": {"game": game},
                                  "audio": {"path": f"audio/{lid}.wav", "duration_s": len(x) / RATE}})  # fmt: skip
            (harvest / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
            (d / "settings.json").write_text(json.dumps({"eq_match": True}))
            env = {**os.environ, "VI_INPUT_HARVEST": str(harvest), "VI_OUT": str(d / "out"),
                   "VI_SETTINGS": str(d / "settings.json")}  # fmt: skip
            subprocess.run([sys.executable, str(JOBS / "prep.py")], env=env, check=True, capture_output=True)
            recs = [json.loads(x) for x in (d / "out" / "prep.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["eq_matched"] for r in recs))
            report = json.loads((d / "out" / "prep.json").read_text())["eq_match"]
            self.assertEqual(sorted(report["fitted_on"]), ["le1", "le2"])
            self.assertNotIn("rows", report)
            # The darker game is brightened, the other darkened.
            tilt = {g: v["4000"] - v["500"] for g, v in report["correction_db_at_third_octaves"].items()}
            self.assertGreater(tilt["le1"], 6, tilt)
            self.assertLess(tilt["le2"], -6, tilt)


if __name__ == "__main__":
    unittest.main()
