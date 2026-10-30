"""Tests for voicetune.py's pure parts (no model). Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np

import voicetune


class VoiceLoraTest(unittest.TestCase):
    def test_validate_flags_missing_fields_duplicates_and_sample_rate(self):
        rows = [
            {"id": "a", "audio": "a.wav", "text": "Keelah se'lai.", "ref_audio": "r.wav"},
            {"id": "a", "audio": "b.wav", "text": "Again.", "ref_audio": "r.wav"},
            {"id": "c", "audio": "c.wav", "ref_audio": "r.wav"},
        ]
        bad = voicetune.validate(rows, lambda p: 44100 if p == "b.wav" else 24000)
        self.assertEqual(len(bad), 3)
        self.assertIn("duplicate id a", bad[0])
        self.assertIn("44100 Hz", bad[1])
        self.assertIn("missing text", bad[2])
        self.assertEqual(voicetune.validate(rows[:1], lambda p: 24000), [])

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
        lab = voicetune.train_labels(labels)
        self.assertEqual(lab.shape[1], 8)
        targets = {t: int(lab[0, t + 1]) for t in range(7) if lab[0, t + 1] != -100}
        self.assertEqual(targets, {3: 10, 4: 11, 5: 12, 6: eos})  # bos predicts frame 0, each frame the next
        # The code predictor gets, for each frame, the hidden state that predicted it.
        hid = voicetune.predictor_hidden_mask(codec_mask)
        self.assertEqual(list(np.flatnonzero(hid[0])), [3, 4, 5])
        self.assertEqual(int(hid.sum()), int(codec_mask.sum()))

    def test_mean_voice_keeps_average_length(self):
        x = np.array([[3.0, 0.0], [0.0, 3.0]])
        m = voicetune.mean_voice(x)
        self.assertAlmostEqual(float(np.linalg.norm(m)), 3.0)
        self.assertAlmostEqual(float(m[0]), float(m[1]))

    def test_length_bands(self):
        b = voicetune.length_bands([{"duration_s": 1.5}, {"duration_s": 3}, {"duration_s": 12}])
        self.assertEqual([b[k]["clips"] for k in b], [1, 1, 0, 1])

    def test_kl_to_base_is_zero_for_the_same_model_and_positive_otherwise(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        base = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(voicetune.kl_to_base(base, base.clone(), mask)), 0.0, places=6)
        self.assertGreater(float(voicetune.kl_to_base(base, base.flip(-1), mask)), 0.1)

if __name__ == "__main__":
    unittest.main()
