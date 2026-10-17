"""Tests for eq_match.py: two "games" that differ only by EQ end up with the
same spectrum. Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np
from scipy import signal

import eq_match as eq
import filter_probe as fp
from test_filter_probe import SR, voice


def ltas(clips):
    return fp.smooth_octave(fp.grid(), 10 * np.log10(np.nanmean([fp.clip_spectrum(x, SR) for x in clips], axis=0)))


class EqMatchTest(unittest.TestCase):
    def test_two_games_that_differ_by_eq_end_up_alike(self):
        rng = np.random.default_rng(0)
        dark = signal.butter(2, 2000, fs=SR, output="sos")  # "LE1": darker
        notch_b, notch_a = signal.iirnotch(800, 2, fs=SR)
        games = {"a": [], "b": []}
        for _ in range(40):
            games["a"].append(signal.lfilter(notch_b, notch_a, signal.sosfilt(dark, voice(rng))))
            games["b"].append(voice(rng))
        spectra = {k: [fp.clip_spectrum(x, SR) for x in v] for k, v in games.items()}
        _, corr = eq.corrections(eq.game_curves(spectra))
        g = fp.grid()
        band = (g >= 300) & (g <= 6000)

        def gap(a, b):
            d = ltas(a) - ltas(b)
            d = d[band] - np.nanmean(d[band])
            return float(np.nanmax(np.abs(d)))

        before = gap(games["a"], games["b"])
        after = gap(*[[eq.apply(x, eq.fir(corr[k], SR)) for x in games[k]] for k in ("a", "b")])
        self.assertGreater(before, 8.0)
        self.assertLess(after, 0.4 * before, (before, after))

    def test_the_filter_keeps_timing_and_the_match_keeps_level(self):
        rng = np.random.default_rng(1)
        x = voice(rng)
        flat = eq.apply(x, eq.fir(np.zeros(len(fp.grid())), SR))
        self.assertLess(np.max(np.abs(flat - x)), 0.02 * np.max(np.abs(x)))
        y = eq.same_loudness(x, 3 * x)
        self.assertAlmostEqual(np.sqrt(np.mean(y**2)), np.sqrt(np.mean(x**2)), places=6)


if __name__ == "__main__":
    unittest.main()
