"""Tests for filter_probe.py on synthetic voices with a known filter.
Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np
from scipy import signal

import filter_probe as fp

SR = 24000


def voice(rng, seconds=1.5):
    """A pulse train at a pitch that differs per clip and drifts, through
    two random formants, with a syllable-rate loudness contour."""
    n = int(SR * seconds)
    f0 = rng.uniform(140, 300) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(n) / SR))
    phase = np.cumsum(f0 / SR)
    x = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    for fc in (rng.uniform(500, 900), rng.uniform(1200, 2500)):
        b, a = signal.iirpeak(fc, 4, fs=SR)
        x = signal.lfilter(b, a, x)
    x *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(3, 5) * np.arange(n) / SR))
    return x + 1e-4 * rng.standard_normal(n)


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

    def test_a_dry_voice_shows_no_echo_or_tremolo(self):
        self.assertFalse(any(abs(p["at"] - 6.0) < 0.2 for p in self.dry["echo_ms"]["in_both_halves"]))
        self.assertFalse(any(abs(p["at"] - 75) <= 2 for p in self.dry["modulation_hz"]["in_both_halves"]))


if __name__ == "__main__":
    unittest.main()
