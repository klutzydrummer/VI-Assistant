"""Tests for voicelora.py's pure parts (no model). Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np

import voicelora


class VoiceLoraTest(unittest.TestCase):
    def test_validate_flags_missing_fields_duplicates_and_sample_rate(self):
        rows = [
            {"id": "a", "audio": "a.wav", "text": "Keelah se'lai.", "ref_audio": "r.wav"},
            {"id": "a", "audio": "b.wav", "text": "Again.", "ref_audio": "r.wav"},
            {"id": "c", "audio": "c.wav", "ref_audio": "r.wav"},
        ]
        bad = voicelora.validate(rows, lambda p: 44100 if p == "b.wav" else 24000)
        self.assertEqual(len(bad), 3)
        self.assertIn("duplicate id a", bad[0])
        self.assertIn("44100 Hz", bad[1])
        self.assertIn("missing text", bad[2])
        self.assertEqual(voicelora.validate(rows[:1], lambda p: 24000), [])

    def test_alignment_matches_generation(self):
        # A sequence laid out like the official collate: prompt, codec bos at 3, frames 0..2 at 4..6, eos at 7.
        frames = [10, 11, 12]
        eos = 99
        labels = np.full((1, 9), -100)
        labels[0, 4:7] = frames
        labels[0, 7] = eos
        codec_mask = np.zeros((1, 9), dtype=bool)
        codec_mask[0, 4:7] = True
        # The model sees inputs[:, :-1] (positions 0..7) and its loss compares logits[t] with labels[t + 1].
        lab = voicelora.train_labels(labels)
        self.assertEqual(lab.shape[1], 8)
        targets = {t: int(lab[0, t + 1]) for t in range(7) if lab[0, t + 1] != -100}
        self.assertEqual(targets, {3: 10, 4: 11, 5: 12, 6: eos})  # bos predicts frame 0, each frame the next
        # The code predictor gets, for each frame, the hidden state that predicted it.
        hid = voicelora.predictor_hidden_mask(codec_mask)
        self.assertEqual(list(np.flatnonzero(hid[0])), [3, 4, 5])
        self.assertEqual(int(hid.sum()), int(codec_mask.sum()))

    def test_mean_voice_keeps_average_length(self):
        x = np.array([[3.0, 0.0], [0.0, 3.0]])
        m = voicelora.mean_voice(x)
        self.assertAlmostEqual(float(np.linalg.norm(m)), 3.0)
        self.assertAlmostEqual(float(m[0]), float(m[1]))

    def test_length_bands(self):
        b = voicelora.length_bands([{"duration_s": 1.5}, {"duration_s": 3}, {"duration_s": 12}])
        self.assertEqual([b[k]["clips"] for k in b], [1, 1, 0, 1])

    def test_blend_scales_the_weight_change(self):
        base = {"w": np.array([1.0, 2.0])}
        tuned = {"w": np.array([3.0, 2.0])}
        self.assertEqual(list(voicelora.blend(base, tuned, 0.5)["w"]), [2.0, 2.0])
        self.assertEqual(list(voicelora.blend(base, tuned, 1.0)["w"]), [3.0, 2.0])

    def test_set_scale_is_relative_to_trained_scaling(self):
        class Lora:
            def __init__(self):
                self.lora_A = {}
                self.scaling = {"default": 2.0}

        class Mod:
            def __init__(self):
                self.layer = Lora()

            def modules(self):
                return [self, self.layer]

        m = Mod()
        self.assertEqual(voicelora.set_scale(m, 0.5), 1)
        self.assertEqual(m.layer.scaling["default"], 1.0)
        voicelora.set_scale(m, 0.3)
        self.assertAlmostEqual(m.layer.scaling["default"], 0.6)


if __name__ == "__main__":
    unittest.main()
