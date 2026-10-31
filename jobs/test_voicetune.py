"""Tests for voicetune.py's pure parts (no model). Run with `python3 -m unittest discover -s jobs`."""

import unittest

import numpy as np

import voicetune


class VoiceTuneTest(unittest.TestCase):
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

    def test_warmup_factor_ramps_linearly_then_holds(self):
        self.assertEqual([voicetune.warmup_factor(s, 4) for s in range(6)], [0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
        self.assertEqual(voicetune.warmup_factor(0, 0), 1.0)

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

    def test_shuffle_chunks_keeps_frames_and_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        mel = torch.arange(10, dtype=torch.float32).repeat(2, 1).T.contiguous()  # [10, 2], frame i = (i, i)
        out = voicetune.shuffle_chunks(mel, 8, 3, torch.Generator().manual_seed(1))
        self.assertEqual(sorted(out[:8, 0].tolist()), list(range(8)))  # same frames, reordered
        self.assertEqual(out[8:, 0].tolist(), [8.0, 9.0])  # padding untouched
        rows = out[:8, 0].tolist()
        starts = [rows.index(c) for c in (0.0, 3.0, 6.0)]
        self.assertEqual(rows[starts[0]:starts[0] + 3], [0.0, 1.0, 2.0])  # chunks stay whole
        self.assertTrue(torch.equal(voicetune.shuffle_chunks(mel, 8, 0), mel))
        self.assertTrue(torch.equal(voicetune.shuffle_chunks(mel, 3, 5), mel))

    def test_grad_reverse_flips_the_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        x = torch.tensor([1.0, -2.0], requires_grad=True)
        y = voicetune.grad_reverse(x, 0.5)
        self.assertEqual(y.tolist(), [1.0, -2.0])
        (y * torch.tensor([3.0, 4.0])).sum().backward()
        self.assertEqual(x.grad.tolist(), [-1.5, -2.0])

    def test_prosody_targets_on_a_steady_tone(self):
        try:
            import soundfile as sf
            import librosa  # noqa: F401
        except ImportError:
            self.skipTest("no librosa")
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tone.wav"
            t = np.arange(24000 * 2) / 24000
            sf.write(p, 0.3 * np.sin(2 * np.pi * 220 * t), 24000)
            f0_range, loud, spread, pace = voicetune.prosody_targets(str(p), "one two three four")
        self.assertLess(f0_range, 1.0)  # a steady pitch has almost no range
        self.assertAlmostEqual(pace, 2.0, places=1)  # 4 words in 2 s
        self.assertLess(spread, 3.0)

    def test_kl_to_base_is_zero_for_the_same_model_and_positive_otherwise(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        base = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(voicetune.kl_to_base(base, base.clone(), mask)), 0.0, places=6)
        self.assertGreater(float(voicetune.kl_to_base(base, base.flip(-1), mask)), 0.1)

    def test_style_encoder_starts_at_her_mean_voice_and_pads(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        mels, n = voicetune.pad_mels([torch.ones(40, 128), torch.ones(25, 128)])
        self.assertEqual(tuple(mels.shape), (2, 40, 128))
        self.assertEqual(n.tolist(), [40, 25])
        self.assertEqual(float(mels[1, 30:].abs().sum()), 0.0)
        enc = voicetune.style_encoder(16)
        self.assertEqual(tuple(enc.code(mels, n).shape), (2, 16))
        self.assertEqual(float(enc(mels, n).abs().sum()), 0.0)  # zero offset until trained


if __name__ == "__main__":
    unittest.main()
