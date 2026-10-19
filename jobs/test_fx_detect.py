"""Tests for fx_detect.py: marks and table parsing, the hand-made measures on
synthetic effects, and the scoring on planted data. None of them need torch.
Run with `python3 -m unittest discover -s jobs`."""

import io
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from scipy import signal

import fx_detect as fd

SR = 24000


def voice(rng, seconds=2.0):
    """Harmonics on a drifting pitch with a syllable-rate loudness contour."""
    t = np.arange(int(SR * seconds)) / SR
    f0 = rng.uniform(170, 260) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(1, 3) * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 30) if k * f0.max() < SR / 2)
    return 0.3 * x * (0.3 + np.sin(2 * np.pi * 2.5 * t) ** 2) + 0.002 * rng.standard_normal(len(t))


def radio(x):
    sos = signal.butter(6, [300, 3400], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


class Inputs(unittest.TestCase):
    def test_marks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "channel.jsonl"
            recs = [{"id": "a", "channel": "radio"}, {"id": "b", "channel": "Other effect"},
                    {"id": "c", "channel": "plain"}, {"id": "a", "channel": "plain"},
                    {"id": "d", "channel": "cant_tell"}, {"id": "c", "channel": "can't tell"}]  # fmt: skip
            p.write_text("\n".join(json.dumps(r) for r in recs))
            self.assertEqual(fd.read_marks(p), {"a": "plain", "b": "other_effect"})
            p.write_text(json.dumps({"id": "x", "mark": "loud"}))
            with self.assertRaises(SystemExit):
                fd.read_marks(p)

    def test_table(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "channel.csv"
            p.write_text("id,game,low_cut_db,noise_db,loudness_db,f0_hz\na,LE1,-3,-40,-20,210\nb,LE3,-9,-30,-25,190\n")
            table, cols = fd.read_table(p)
            self.assertEqual(cols, ["low_cut_db", "noise_db"])
            self.assertEqual(table["b"]["low_cut_db"], -9.0)
            table, cols = fd.read_table(p, ["noise_db"])
            self.assertEqual(table["a"], {"noise_db": -40.0})
            q = Path(d) / "channel.jsonl"
            q.write_text(json.dumps({"id": "a", "high_cut_db": -12.5, "note": "x"}))
            self.assertEqual(fd.read_table(q)[0], {"a": {"high_cut_db": -12.5}})

    def test_cache_keeps_only_complete_clips(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rng = np.random.default_rng(0)
            vecs = {k: rng.normal(size=(3, n)).astype(np.float32) for k, n in fd.EMBED_DIMS.items()}
            lines = [json.dumps({"id": f"c{i}", "extras": {"crest_db": float(i)}}) for i in range(3)]
            (out / "measures.jsonl").write_text("\n".join(lines[:2]) + "\n" + lines[2][:10])  # third line cut off
            vecs["afx"].tofile(out / "afx.f32")
            vecs["fxpp"].tofile(out / "fxpp.f32")
            done = fd.load_cache(out)
            self.assertEqual(sorted(done), ["c0", "c1"])
            np.testing.assert_array_equal(done["c1"]["fxpp"], vecs["fxpp"][1])
            self.assertEqual(len(np.fromfile(out / "afx.f32", np.float32)), 2 * 512)  # files back in step
            self.assertEqual(fd.load_cache(out).keys(), done.keys())

    def test_unknown_classes_unpickle(self):
        gone = types.ModuleType("not_installed_anywhere")

        class Callback:
            def __init__(self):
                self.every = 5

        Callback.__module__ = gone.__name__
        Callback.__qualname__ = "Callback"
        gone.Callback = Callback
        sys.modules[gone.__name__] = gone
        try:
            blob = pickle.dumps({"cb": Callback(), "cls": Callback, "w": [1, 2]}, protocol=2)
        finally:
            del sys.modules[gone.__name__]
        r = fd._TolerantUnpickler(io.BytesIO(blob)).load()
        self.assertEqual(r["w"], [1, 2])
        self.assertEqual(r["cb"].state, {"every": 5})


class Measures(unittest.TestCase):
    def test_radio_band_shows_in_channel_measures(self):
        rng = np.random.default_rng(1)
        x = voice(rng)
        plain, rad = fd.channel_builtin(x, SR), fd.channel_builtin(radio(x), SR)
        self.assertLess(rad["low_cut_db"], plain["low_cut_db"] - 6)
        self.assertLess(rad["high_cut_db"], plain["high_cut_db"] - 6)

    def test_clipping_shows_in_extras(self):
        rng = np.random.default_rng(2)
        x = voice(rng)
        hard = np.clip(x * 4, -0.3, 0.3)
        e0, e1 = fd.extras(x, SR), fd.extras(hard, SR)
        self.assertGreater(e1["clip_share"], e0["clip_share"] + 0.05)
        self.assertLess(e1["crest_db"], e0["crest_db"] - 3)
        self.assertGreater(e1["interharm_db"], e0["interharm_db"])

    def test_beep_at_start_is_a_chirp(self):
        """A 150 ms beep holds its frequency; speech with ordinary
        intonation doesn't. Median over voices, since one voice can hold a
        pitch for a while."""
        gains = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            t = np.arange(int(SR * 2.0)) / SR
            f0 = rng.uniform(170, 260) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(4, 6) * t) - 0.05 * t)
            x = sum(np.sin(k * 2 * np.pi * np.cumsum(f0) / SR) / k for k in range(1, 30))
            x = 0.3 * x * (0.3 + np.sin(2 * np.pi * 2.5 * t) ** 2)
            beep = x.copy()
            beep[: int(0.15 * SR)] += 0.5 * np.sin(2 * np.pi * 1500 * t[: int(0.15 * SR)])
            gains.append(fd.chirp(beep, SR) - fd.chirp(x, SR))
        self.assertGreater(np.median(gains), 60)

    def test_net_input_length_and_level(self):
        y = fd.net_input(np.ones(SR // 2) * 0.1, SR, 48000)
        self.assertEqual(len(y), int(fd.MIN_S * 48000))
        self.assertAlmostEqual(float(np.abs(y).max()), 1.0, places=5)
        self.assertEqual(len(fd.net_input(np.zeros(SR * 20), SR, 44100)), int(fd.MAX_S * 44100))


class Scoring(unittest.TestCase):
    def setUp(self):
        """40 marked clips in 20 conversations, where the effect clips are
        also louder (as shouted radio calls would be), and a 400-clip corpus
        with 5% effects to fit the confounds on. `real` hears the effect;
        `fake` only hears loudness."""
        rng = np.random.default_rng(4)
        n, m = 40, 400
        self.groups = np.repeat(np.arange(20), 2)
        self.labels = ["plain"] * 20 + ["radio"] * 10 + ["other_effect"] * 10
        eff = np.array([lab != "plain" for lab in self.labels])
        eff_all = rng.random(m) < 0.05
        loud, loud_all = rng.normal(0, 1, n) + 1.5 * eff, rng.normal(0, 1, m) + 1.5 * eff_all
        self.conf = np.column_stack([loud, rng.normal(size=n)])
        self.conf_all = np.column_stack([loud_all, rng.normal(size=m)])

        def feats(e, lo, k):
            r = np.random.default_rng(k)
            real = np.column_stack([2.0 * e + 0.5 * lo + r.normal(0, 0.7, len(e)), r.normal(size=len(e))])
            fake = np.column_stack([lo + r.normal(0, 0.1, len(e)), r.normal(size=len(e))])
            return real, fake

        (self.real, self.fake), (self.real_all, self.fake_all) = feats(eff, loud, 5), feats(eff_all, loud_all, 6)

    def test_auc(self):
        self.assertEqual(fd.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]), 1.0)
        self.assertEqual(fd.auc([0, 1, 0, 1], [1, 1, 1, 1]), 0.5)
        self.assertTrue(np.isnan(fd.auc([1, 1], [0.2, 0.3])))

    def test_real_effect_survives_and_loudness_does_not(self):
        a = (self.labels, self.groups, self.conf, 99)
        r_real, _ = fd.score_set("real", self.real, *a, self.real_all, self.conf_all)
        r_fake, _ = fd.score_set("fake", self.fake, *a, self.fake_all, self.conf_all)
        self.assertEqual(r_real["resid_fit"], "corpus")
        self.assertGreater(r_real["effect_auc_resid"], 0.85)
        self.assertLess(r_real["p_resid"], 0.05)
        self.assertGreater(r_fake["effect_auc"], 0.8)  # looks good until loudness is taken out
        self.assertLess(r_fake["effect_auc_resid"], 0.7)
        self.assertGreater(r_fake["p_resid"], 0.05)

    def test_without_corpus_the_fit_uses_plain_training_clips(self):
        r_real, _ = fd.score_set("real", self.real, self.labels, self.groups, self.conf, 0)
        r_fake, _ = fd.score_set("fake", self.fake, self.labels, self.groups, self.conf, 0)
        self.assertEqual(r_real["resid_fit"], "plain clips in each fold")
        self.assertGreater(r_real["effect_auc_resid"], 0.85)
        self.assertLess(r_fake["effect_auc_resid"], 0.7)

    def test_pick(self):
        rows = [{"set": "afx", "effect_auc_resid": 0.90, "p_resid": 0.01, "corpus_conf_r2": 0.1},
                {"set": "channel", "effect_auc_resid": 0.89, "p_resid": 0.01, "corpus_conf_r2": 0.2},
                {"set": "extras", "effect_auc_resid": 0.95, "p_resid": 0.01, "corpus_conf_r2": 0.8},
                {"set": "fxpp", "effect_auc_resid": 0.99, "p_resid": 0.2, "corpus_conf_r2": 0.1}]  # fmt: skip
        self.assertEqual(fd.pick(rows), "channel")  # extras too confounded, fxpp not significant, channel ties afx
        self.assertIsNone(fd.pick(rows[2:]))


if __name__ == "__main__":
    unittest.main()
