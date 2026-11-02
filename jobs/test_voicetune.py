"""Tests for voicetune.py's pure parts (no model). Run with `python3 -m unittest discover -s jobs`."""

import json
import re
import unittest
from unittest import mock

import numpy as np

import voicetune


# Tiny stand-ins for the parts of Qwen3-TTS that forward() touches, so the two alignment fixes
# can be tested without loading the real model. Every tensor uses an 8-wide hidden state.
class _Const:
    """Callable returning a constant [..., 8] tensor (an embedding that ignores its indices)."""

    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        import torch

        return torch.full((*x.shape, 8), self.value)


class _Project:
    """text_projection stand-in that scales its input, so projected text is distinct from raw."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


class _CodePredictor:
    def get_input_embeddings(self):
        return [_Const(0.0) for _ in range(15)]


class _Talker:
    def __init__(self, project_scale=1.0, sub_logits=None, sub_loss=0.0):
        import torch

        self.text_projection = _Project(project_scale)
        self.model = type("_TextCodec", (), {"text_embedding": _Const(1.0), "codec_embedding": _Const(0.0)})()
        self.code_predictor = _CodePredictor()
        self._sub_logits = sub_logits
        self._sub_loss = sub_loss
        self.seen = {}

    def __call__(self, inputs_embeds, attention_mask=None, labels=None, output_hidden_states=False):
        import torch

        self.seen["inputs_embeds"] = inputs_embeds
        out = type("_Out", (), {})()
        out.loss = torch.zeros(())
        out.hidden_states = ([torch.zeros(*inputs_embeds.shape)],)
        out.logits = torch.zeros(*inputs_embeds.shape[:2], 8)
        return out

    def forward_sub_talker_finetune(self, codes, hidden):
        import torch

        if self._sub_logits is not None:
            return self._sub_logits, torch.full((), self._sub_loss)
        return torch.zeros(*codes.shape[:1], 15, 8), torch.full((), self._sub_loss)


class _Model:
    device = "cpu"

    def __init__(self, **kw):
        self.talker = _Talker(**kw)


def _batch(frames, rows=None):
    """A collate-like batch: `frames` are the masked codec positions; rows[i] sets codec row i."""
    import torch

    T = 8  # longer than SPEAKER_SLOT (6)
    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[0, list(frames)] = True
    codec_ids = torch.zeros(1, T, 16, dtype=torch.long)
    if rows:
        for p, row in zip(frames, rows):
            codec_ids[0, p] = torch.tensor(row)
    return {
        "input_ids": torch.zeros(1, T, 2, dtype=torch.long),
        "codec_ids": codec_ids,
        "codec_0_labels": torch.zeros(1, T, dtype=torch.long),
        "codec_mask": mask,
        "text_embedding_mask": torch.ones(1, T, 1),
        "codec_embedding_mask": torch.zeros(1, T, 1),
        "attention_mask": torch.ones(1, T, dtype=torch.long),
    }


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

    def test_causal_label_mask_matches_logits_shape_and_marks_targets(self):
        # labels [1, 5] -> train_labels [1, 4] -> logits [1, 4]. The logit at t has target
        # train_labels[:, t + 1], and the last logit has none.
        labels = np.array([[3, 5, -100, 7, 9]])
        mask = voicetune.causal_label_mask(labels)
        self.assertEqual(mask.shape, (1, 4))  # out.logits[:, :, 0] is [b, T - 1]
        self.assertEqual(mask.tolist(), [[True, False, True, False]])

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

    def test_replay_kl_all_codebooks_zero_when_identical_and_positive_when_residual_moves(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        labels = torch.tensor([[3, 5, 7, 9]])  # [1, 4] -> logits [1, 3], targets at positions 0 and 1
        main = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # [1, 3, 3]
        sub = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])  # [1, 2, 3]: two residual codebooks
        kl = voicetune.replay_kl_term(main, main.clone(), sub, sub.clone(), labels)
        self.assertAlmostEqual(float(kl), 0.0, places=6)
        moved = voicetune.replay_kl_term(main, main.clone(), sub, sub.flip(-1), labels)
        self.assertGreater(float(moved), 0.0)

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

    def test_forward_projects_text_through_talker_text_projection(self):
        try:
            import torch
        except ImportError:
            self.skipTest("no torch")
        # text_embedding yields 1s; text_projection doubles them. The talker must see 2s (projected),
        # not 1s (raw), at every text position.
        model = _Model(project_scale=2.0)
        voicetune.forward(model, _batch((2,)), torch.zeros(1, 8))
        emb = model.talker.seen["inputs_embeds"]
        self.assertTrue(torch.allclose(emb, torch.full_like(emb, 2.0)))

    def test_forward_scores_the_sub_talker_loss_aligned(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("no torch")
        V = 4
        frames = (2, 4)  # two frames -> sub_logits [2, 15, V]
        rows = ([1] * 16, [2] * 16)
        sub_logits = torch.randn(2, 15, V)
        # sub_loss sentinel 999: forward() must ignore the model's own (double-shifted) loss.
        model = _Model(sub_logits=sub_logits, sub_loss=999.0)
        batch = _batch(frames, rows)
        _, sub, _ = voicetune.forward(model, batch, torch.zeros(1, 8))
        sub_codes = batch["codec_ids"][batch["codec_mask"]]
        expected = F.cross_entropy(sub_logits.reshape(-1, V).float(), sub_codes[:, 1:].reshape(-1))
        self.assertAlmostEqual(float(sub), float(expected), places=5)
        self.assertNotAlmostEqual(float(sub), 999.0, places=5)

    def test_train_argparse_parses_publish_and_no_resume_file(self):
        with mock.patch.object(voicetune, "cmd_train") as fake:
            fake.return_value = None
            self.assertEqual(voicetune.main(["train", "--data", "d", "--out", "o",
                                             "--publish", "pub", "--no-resume-file"]), 0)
            a = fake.call_args.args[0]
            self.assertEqual(str(a.publish), "pub")
            self.assertTrue(a.no_resume_file)
            voicetune.main(["train", "--data", "d", "--out", "o"])
            a2 = fake.call_args.args[0]
            self.assertIsNone(a2.publish)
            self.assertFalse(a2.no_resume_file)

    def test_speak_argparse_accepts_seed_unblind_title_question_and_delivery_from(self):
        with mock.patch.object(voicetune, "cmd_speak") as fake:
            fake.return_value = None
            self.assertEqual(voicetune.main(["speak", "--run", "r", "--sentences", "s",
                                             "--seed", "3", "--unblind", "--title", "T", "--question", "Q",
                                             "--delivery-from", "clip1=r.wav"]), 0)
            a = fake.call_args.args[0]
            self.assertEqual(a.seed, 3)
            self.assertTrue(a.unblind)
            self.assertEqual(a.title, "T")
            self.assertEqual(a.question, "Q")
            self.assertEqual(a.delivery_from, ["clip1=r.wav"])

    def test_unblind_hub_shape_order_and_labels(self):
        lines = ["the first line", "the second line", "the third line"]
        entries = []
        for epoch in range(6):
            for li, text in enumerate(lines):
                entries.append((epoch, li, text, "plain", f"e{epoch}-{li}.wav"))
        for epoch, ids in ((4, ("x", "y", "z")), (5, ("p", "q", "r"))):
            for li, clip_id in enumerate(ids):
                entries.append((epoch, li, lines[li], f"delivery-of-{clip_id}", f"d{epoch}-{li}.wav"))
        import random

        random.Random(1).shuffle(entries)  # prove the hub sorts, it doesn't trust call order
        text = voicetune.unblind_hub(entries, "T", "Q")
        self.assertIn("blind = false", text)
        self.assertIn('answers = ["acceptable", "close", "not yet"]', text)
        self.assertIn('title = "T"', text)
        self.assertIn('question = "Q"', text)
        self.assertEqual(text.count("[[clip]]"), 24)
        labels = [ln.split(" = ", 1)[1] for ln in text.splitlines() if ln.startswith("label = ")]
        self.assertEqual(len(labels), 24)
        epochs = []
        for lab in labels:
            m = re.match(r'"epoch (\d+), (plain|delivery-of-[^:]+): ', lab)
            self.assertIsNotNone(m, lab)
            epochs.append(int(m.group(1)))
        self.assertEqual(epochs, sorted(epochs))  # epoch then line
        self.assertEqual(epochs.count(4), 6)  # 3 plain lines + 3 delivery clips at the last two epochs
        self.assertEqual(epochs.count(5), 6)
        for epoch in range(4):
            self.assertEqual(epochs.count(epoch), 3)

    def test_check_abort_cases(self):
        step = lambda s, loss, peak=1.0: {"step": s, "loss": loss, "peak_gb": peak}  # noqa: E731
        self.assertEqual(voicetune.check_abort([step(1, 4.5)], 0), "step-1 loss > 4.0")
        self.assertEqual(voicetune.check_abort([step(1, 0.5), step(2, float("nan"))], 0), "NaN loss")
        # Genuinely rising: 50 flat steps then 10 at +50%, so the windowed mean is up 50%.
        rising = [step(i + 1, 1.0) for i in range(50)] + [step(i + 51, 1.5) for i in range(10)]
        self.assertIn("loss rising", voicetune.check_abort(rising, 0))
        # Noisy-but-flat (±25% alternating, mean 1.0) must not trip the windowed rule.
        noisy = [step(i + 1, 1.25 if i % 2 else 0.75) for i in range(70)]
        self.assertIsNone(voicetune.check_abort(noisy, 0))
        self.assertIn("peak memory", voicetune.check_abort([step(1, 0.5, 45.0)], 0))
        self.assertIn("elapsed", voicetune.check_abort([step(1, 0.5)], 6301))
        healthy = [{"step": i + 1, "loss": 4.0 - i * 0.01, "peak_gb": 40.0} for i in range(60)]
        self.assertIsNone(voicetune.check_abort(healthy, 100))

    def test_clip_filenames_are_distinct_across_epochs_and_kinds(self):
        names = set()
        for epoch in (0, 4):
            for n in range(3):
                names.add(voicetune.clip_filename(epoch, n, "plain"))
            for clip_id in ("a", "b", "c"):
                names.add(voicetune.clip_filename(epoch, 0, f"delivery-of-{clip_id}"))
        self.assertEqual(len(names), 2 * (3 + 3))  # none collide within or across the two epochs
        self.assertNotEqual(voicetune.clip_filename(4, 0, "plain"),
                            voicetune.clip_filename(4, 0, "delivery-of-a"))

    def test_delivery_clips_is_seeded_and_ignores_extra_fields(self):
        base = [{"id": f"c{i}", "audio": f"a{i}.wav", "ref_audio": f"r{i}.wav", "text": "x"} for i in range(10)]
        a = [dict(r, mood="happy", label=1) for r in base]
        b = [dict(r, mood="sad", label=0, junk=[1, 2, 3]) for r in base]
        pick_a = voicetune.delivery_clips(a, 0, 6)
        pick_b = voicetune.delivery_clips(b, 0, 6)
        self.assertEqual(pick_a, pick_b)  # same seed -> same ids/paths, extra fields ignored
        self.assertEqual(len(pick_a), 6)
        ids = {r["id"] for r in base}
        refs = {r["ref_audio"] for r in base}
        self.assertTrue({p[0] for p in pick_a} <= ids)
        self.assertTrue({p[1] for p in pick_a} <= refs)

    def test_check_reports_new_keys_and_generate_with_model_mocked(self):
        import sys
        import tempfile
        import types
        from pathlib import Path

        saved = {}
        for mod in ("qwen_tts", "transformers", "accelerate", "librosa", "soundfile", "huggingface_hub"):
            saved[mod] = sys.modules.get(mod)
            sys.modules[mod] = types.ModuleType(mod)
        try:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                tts = mock.Mock()
                tts.model.talker.parameters.return_value = iter([])
                a = mock.Mock(out=d / "check", base=voicetune.BASE, no_model=False,
                              work_dir=d / "work", generate=True)
                with mock.patch.object(voicetune, "load_tts", return_value=tts), \
                     mock.patch.object(voicetune, "_generate_one",
                                       return_value={"line": "hi", "wav": "x.wav", "sr": 24000, "audio_s": 1.0}):
                    rc = voicetune.cmd_check(a)
                out = json.loads((d / "check" / "check.json").read_text())
        finally:
            for mod, m in saved.items():
                if m is None:
                    sys.modules.pop(mod, None)
                else:
                    sys.modules[mod] = m
        self.assertEqual(rc, 0)
        for key in ("torch", "cuda", "bf16", "nvidia", "sox", "gcc", "g++", "free_disk_gb", "generate", "ok"):
            self.assertIn(key, out)
        self.assertEqual(out["generate"]["wav"], "x.wav")


if __name__ == "__main__":
    unittest.main()
