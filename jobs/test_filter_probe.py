"""Tests for filter_probe.py on synthetic voices with a known filter.
Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np
from scipy import signal

import filter_probe as fp

SR = 24000


def voice(rng, seconds=1.5, doubled=None):
    """A pulse train at a pitch that differs per clip and drifts, through
    two random formants, with a syllable-rate loudness contour. `doubled`
    adds a second copy at that pitch ratio, half as loud."""
    n = int(SR * seconds)
    f0 = rng.uniform(140, 300) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(n) / SR))
    if doubled:
        state = rng.bit_generator.state
        base = voice_from(f0, rng, n)
        rng.bit_generator.state = state
        return base + 0.5 * voice_from(f0 * doubled, rng, n)
    return voice_from(f0, rng, n)


def voice_from(f0, rng, n):
    phase = np.cumsum(f0 / SR)
    x = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    for fc in (rng.uniform(500, 900), rng.uniform(1200, 2500)):
        b, a = signal.iirpeak(fc, 4, fs=SR)
        x = signal.lfilter(b, a, x)
    x *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(3, 5) * np.arange(n) / SR))
    return x + 1e-4 * rng.standard_normal(n)


def flanger(x, lo_ms=0.5, hi_ms=1.2, rate_hz=0.7):
    """A comb whose delay sweeps between lo_ms and hi_ms."""
    t = np.arange(len(x))
    d = SR / 1000 * (lo_ms + (hi_ms - lo_ms) * 0.5 * (1 + np.sin(2 * np.pi * rate_hz * t / SR)))
    return x + 0.8 * np.interp(t - d, t, x, left=0.0)


def suit(x, delay_ms=6.0, am_hz=75.0):
    """A known chain: band-pass 400-3500 Hz, an echo, a tremolo."""
    sos = signal.butter(4, [400, 3500], btype="band", fs=SR, output="sos")
    y = signal.sosfilt(sos, x)
    d = int(SR * delay_ms / 1000)
    y = y + 0.6 * np.concatenate([np.zeros(d), y[:-d]])
    return y * (1 + 0.6 * np.sin(2 * np.pi * am_hz * np.arange(len(y)) / SR))


def report(filtered, clips=60, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(clips):
        x = voice(rng)
        x = suit(x) if filtered else x
        results.append(fp.analyse(0.3 * x / np.abs(x).max(), SR))
    return fp.summarise(results)


class FilterProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suit = report(True)
        cls.dry = report(False)

    def test_finds_the_echo_delay(self):
        both = self.suit["echo_ms"]["in_both_halves"]
        self.assertTrue(any(abs(p["at"] - 6.0) < 0.2 for p in both), self.suit["echo_ms"])

    def test_finds_the_tremolo_rate(self):
        both = self.suit["modulation_hz"]["in_both_halves"]
        self.assertTrue(any(abs(p["at"] - 75) <= 2 for p in both), self.suit["modulation_hz"])

    def test_finds_the_band(self):
        lo, hi = self.suit["ltas"]["band_minus20db_hz"]
        self.assertGreater(lo, 200)
        self.assertLess(hi, 6000)

    def test_finds_a_doubled_layer_two_semitones_up(self):
        rng = np.random.default_rng(1)
        two_up = 2 ** (2 / 12)
        results = [fp.analyse(voice(rng, doubled=two_up), SR) for _ in range(60)]
        d = fp.summarise(results)["doubled_layer"]
        for half in d["halves"]:
            self.assertAlmostEqual(half["ratio"], two_up, delta=0.01, msg=d)
        dry = self.dry["doubled_layer"]
        self.assertGreater(d["prominence_db"], dry["prominence_db"] + 1.0, (d, dry))

    def test_a_fixed_comb_stays_put_and_a_flanger_moves(self):
        rng = np.random.default_rng(2)
        fixed, swept = [], []
        for _ in range(30):
            x = voice(rng, seconds=3.0)
            d = int(SR * 0.0009)
            fixed.append(fp.analyse(x + 0.8 * np.concatenate([np.zeros(d), x[:-d]]), SR))
            swept.append(fp.analyse(flanger(x), SR))
        f, s = fp.summarise(fixed)["comb_over_time"], fp.summarise(swept)["comb_over_time"]
        self.assertAlmostEqual(f["most_common_ms"], 0.9, delta=0.04)
        self.assertGreater(f["share_within_0_05ms"], 0.8, f)
        self.assertGreater(f["share_with_a_clear_comb"], 0.8, f)
        dry = self.dry["comb_over_time"]
        self.assertTrue(dry is None or dry["share_with_a_clear_comb"] < 0.3, dry)
        self.assertLess(s["share_within_0_05ms"], 0.4, s)
        self.assertGreater(s["median_within_clip_iqr_ms"], 3 * f["median_within_clip_iqr_ms"] + 0.05, (f, s))

    def test_ripple_depth_tells_a_real_comb_from_a_faint_one(self):
        echo = next(p for p in self.suit["echo_ms"]["all"] if abs(p["at"] - 6.0) < 0.2)
        self.assertGreater(echo["ripple_db"], 5.0, echo)

    def test_a_dry_voice_shows_no_echo_or_tremolo(self):
        self.assertFalse(any(abs(p["at"] - 6.0) < 0.2 for p in self.dry["echo_ms"]["in_both_halves"]))
        self.assertFalse(any(abs(p["at"] - 75) <= 2 for p in self.dry["modulation_hz"]["in_both_halves"]))


if __name__ == "__main__":
    unittest.main()
