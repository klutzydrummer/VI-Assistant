"""Tests for suit_fit.py on synthetic voices through a known chain.
Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np
from scipy import signal

import eq_match as eq
import filter_probe as fp
import suit_fit as sfit
from test_filter_probe import SR, voice

RATIOS = np.round(np.arange(1.06, 1.2001, 0.02), 3)  # a coarse grid keeps CI quick


def voices(rng, n):
    return [0.3 * x / np.abs(x).max() for x in (voice(rng) for _ in range(n))]


def dark(x):
    sos = signal.butter(2, 2000, fs=SR, output="sos")
    b, a = signal.iirnotch(800, 2, fs=SR)
    return signal.lfilter(b, a, signal.sosfilt(sos, x))


class SuitFitTest(unittest.TestCase):
    def test_runs_at_the_test_rate(self):
        self.assertEqual(sfit.SR, SR)

    def test_pitch_shift_moves_a_tone_and_keeps_length(self):
        t = np.arange(SR) / SR
        y = sfit.pitch_shift(np.sin(2 * np.pi * 200 * t), 1.122)
        self.assertEqual(len(y), SR)
        f, p = signal.welch(y[2048:-2048], SR, nperseg=8192)
        self.assertAlmostEqual(f[np.argmax(p)], 224.4, delta=4)

    def test_reference_eq_recovers_the_suit_and_flags_a_bad_source(self):
        rng = np.random.default_rng(0)
        game = [dark(x) for x in voices(rng, 40)]
        tilt = signal.butter(1, 6000, fs=SR, output="sos")  # one mic a bit duller
        bump_b, bump_a = signal.iirpeak(3000, 1.5, fs=SR)  # one source with a big resonance
        sources = {"a": voices(rng, 30), "b": [signal.sosfilt(tilt, x) for x in voices(rng, 30)],
                   "c": [x + 2 * signal.lfilter(bump_b, bump_a, x) for x in voices(rng, 30)]}  # fmt: skip
        r = sfit.reference_eq(sources, sfit.curve(game))
        g = fp.grid()
        same = voices(rng, 40)  # the filter's effect on one set of clips, measured the same way
        truth = sfit.curve([dark(x) for x in same]) - sfit.curve(same)
        band = (g >= 500) & (g <= 2500)
        err = (r["suit_db"] - truth)[band]
        self.assertLess(np.max(np.abs(err - np.mean(err))), 3.0)
        self.assertFalse(r["trusted"][np.argmin(np.abs(g - 3000))])  # sources disagree there
        self.assertTrue(r["trusted"][np.argmin(np.abs(g - 800))])

    def test_finds_a_planted_layer_and_not_a_missing_one(self):
        rng = np.random.default_rng(1)
        ref = voices(rng, 32)
        layered = {"layer_ratio": 1.12, "layer_db": -6.0, "drive": 0.0, "snr_db": None}
        game = [dark(sfit.pre_eq(x, layered, rng)) for x in voices(rng, 32)]
        found = sfit.layer_test(ref, game, sfit.scale_of(game), RATIOS)
        self.assertTrue(found["supported"], found)
        self.assertAlmostEqual(found["ratio_odd_half"], 1.12, delta=0.025)
        plain = [dark(x) for x in voices(rng, 32)]
        self.assertFalse(sfit.layer_test(ref, plain, sfit.scale_of(plain), RATIOS)["supported"])

    def test_the_fitted_chain_runs_forwards(self):
        rng = np.random.default_rng(2)
        x = voices(rng, 1)[0]
        chain = {"params": {"layer_ratio": 1.12, "layer_db": -12.0, "drive": 1.5, "snr_db": 30.0},
                 "eq_db": np.zeros(len(fp.grid())).tolist()}  # fmt: skip
        y = sfit.apply_chain(x, SR, chain)
        self.assertEqual(len(y), len(x))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreater(np.max(np.abs(y - x)), 0.01)
        _ = eq  # same FIR path as eq_match


if __name__ == "__main__":
    unittest.main()
