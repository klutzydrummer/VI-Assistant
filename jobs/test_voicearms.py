"""Tests for voicearms.py, with fake models. Run with `python3 -m unittest discover -s jobs`."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import steer
import voicearms
from test_voicesteer import probe_file


def corpus(tmp):
    """20 clips in 20 conversations; warmth and teasing pairs, positives named "pos"."""
    harvest, files = tmp / "harvest", tmp / "files"
    for d in (harvest / "audio", files / "vectors"):
        d.mkdir(parents=True)
    lines = []
    for i in range(24):
        cid = f"le3.{'pos' if i % 2 == 0 and i < 20 else 'z'}{i}_f"
        steer.write_wav(harvest / "audio" / f"{cid}.wav", steer.fake_wave(cid, seconds=0.05))
        lines.append({"id": cid, "speaker": "tali", "text": f"Secret subtitle {i}", "conversation": f"c{i}",
                      "audio": {"path": f"audio/{cid}.wav", "sample_rate": 16000, "channels": 1, "duration_s": 4.0},
                      "source": {"game": "le3"}})  # fmt: skip
    (harvest / "lines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    ids = [r["id"] for r in lines]
    sets = {"moods": {"warmth": {"pairs": [[ids[i], ids[i + 1]] for i in range(0, 12, 2)], "route": "voice_and_chat"},
                      "teasing": {"pairs": [[ids[i], ids[i + 1]] for i in range(12, 20, 2)], "borderline": True}},
            "neutral": [ids[21], ids[23]]}  # fmt: skip
    (files / "voice_sets.json").write_text(json.dumps(sets))
    probe_file(files / "voice_probe.npz", ["warmth", "teasing"])
    np.savez(files / "voice_vectors.npz", warmth=np.ones((6, 4), dtype=np.float32))  # voicesteer's; teasing gets captured
    return harvest, files


class FakeRunTest(unittest.TestCase):
    def test_runs_every_arm_end_to_end_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files = corpus(tmp)
            out, ck = tmp / "out", tmp / "ck"
            (tmp / "settings.json").write_text(json.dumps({
                "backend": "fake", "layer_sets": ["2", "4", "3-5"], "blends": [["warmth", "teasing"]],
                "ladder": [1.0, 0.5, 0.25]}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out), "VI_CHECKPOINT": str(ck),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env):
                voicearms.main()
                first = (out / "report.json").read_text()
                voicearms.main()  # a rerun reuses the checkpoint and gives the same report
                self.assertEqual((out / "report.json").read_text(), first)
            report = json.loads(first)
            self.assertEqual(report["settings"]["layer_sets"], ["2", "4", "3-5"])
            for m in ("warmth", "teasing"):
                r = report["moods"][m]
                # Fake similarity drops at strength 1, so every arm steps down once, to 0.5.
                for name in ("A", "B-icl", "B-xvec", "C-xvec", "C-icl", "D-xvec", "D-icl", "Dpca1-xvec", "Dpca4-icl"):
                    self.assertEqual(r["arms"][name]["strength"], 0.5, (m, name))
                self.assertEqual([t["strength"] for t in r["A_layers"]["AL2"]["tried"]], [1.0, 0.5])
                self.assertLess(r["arms"]["R"]["lift"], r["arms"]["A"]["lift"])
                self.assertLess(r["arms"]["C-xvec"]["random_lift"], r["arms"]["C-xvec"]["lift"])
                self.assertEqual(r["verdicts"]["A"], "steers", m)
                self.assertEqual(r["verdicts"]["C-xvec"], "steers", m)
                self.assertGreater(r["likelihood_gain"]["A"]["mean"], 0)
                self.assertGreater(r["pairs"]["held_out"], 0)
            self.assertTrue(report["moods"]["teasing"]["borderline"])
            self.assertTrue((ck / "vectors" / "teasing.npy").exists())  # captured, not given
            self.assertFalse((ck / "vectors" / "warmth.npy").exists())  # given by voicesteer
            self.assertEqual(set(report["blends"]["warmth+teasing"]), {"A", "B", "C", "D", "Dpca1", "Dpca4"})
            self.assertEqual(report["settings"]["match_targets"], ["mean", "pca:1", "pca:4"])
            match = report["moods"]["warmth"]["match"]
            self.assertEqual(set(match), {"mean", "pca:1", "pca:4"})
            self.assertEqual(match["mean"]["fit"]["layers"], [1, 2, 3, 4])
            self.assertIn("Dpca4-icl", (out / "report.md").read_text())
            self.assertTrue((out / "tokens" / "warmth.json").exists())
            for name in ("report.json", "report.md", "listen.md"):
                self.assertNotIn("Secret subtitle", (out / name).read_text(), name)
            # The blind listening set: numbered clips per mood and blend, her example, the sheet and key.
            listen = out / "listen"
            key = json.loads((listen / "key.json").read_text())
            warm = sorted(k for k in key if k.startswith("warmth/"))
            conds = {key[k]["condition"] for k in warm}
            self.assertTrue({"neutral", "swap", "fitted", "A", "R", "B-icl", "C-xvec", "D-xvec", "Dpca4-icl"} <= conds, conds)
            self.assertTrue(all((listen / f"{k}.wav").exists() for k in key))
            self.assertTrue(any(listen.glob("warmth/her_example.*")))
            self.assertTrue(any(k.startswith("warmth+teasing/") for k in key))
            self.assertNotIn("condition", (listen / "sheet.csv").read_text())
            rows = (listen / "sheet.csv").read_text().splitlines()
            rows[1] = rows[1].replace(",,,,", ",y,2,y,nice")
            (listen / "sheet.csv").write_text("\n".join(rows) + "\n")
            voicearms.unblind(out)
            first_group, first_clip = rows[1].split(",")[:2]
            self.assertIn(f"| {first_group} | {key[first_group + '/' + first_clip]['condition']} | y | 2 | y | nice |",
                          (listen / "results.md").read_text())

    def test_above_is_tried_only_when_full_strength_holds_without_lifting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            harvest, files = corpus(tmp)
            out = tmp / "out"
            (tmp / "settings.json").write_text(json.dumps({
                "backend": "fake", "moods": ["warmth"], "layer_sets": ["2"], "above": [2.0], "max_similarity_drop": 0.5}))
            env = {"VI_SETTINGS": str(tmp / "settings.json"), "VI_OUT": str(out),
                   "VI_INPUT_HARVEST": str(harvest), "VI_FILES": str(files)}  # fmt: skip
            with mock.patch.dict(os.environ, env):
                voicearms.main()
            r = json.loads((out / "report.json").read_text())["moods"]["warmth"]
            # Strength 1 holds and lifts (fake lift 0.3 >= 0.1), so 2.0 is never tried.
            self.assertEqual([t["strength"] for t in r["A_layers"]["AL2"]["tried"]], [1.0])


class HelperTest(unittest.TestCase):
    def test_layer_sets(self):
        self.assertEqual(voicearms.parse_layers("21-25", 28), [21, 22, 23, 24, 25])
        self.assertEqual(voicearms.parse_layers(14, 28), [14])
        with self.assertRaises(SystemExit):
            voicearms.parse_layers("26-28", 28)

    def test_random_control_keeps_each_layers_norm(self):
        vec = [[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]]
        r = voicearms.random_like(vec, [0, 2], "warmth")
        self.assertAlmostEqual(np.linalg.norm(r[0]), 5.0)
        self.assertAlmostEqual(np.linalg.norm(r[2]), 2.0)
        self.assertEqual(r[1], vec[1])
        self.assertEqual(r, voicearms.random_like(vec, [0, 2], "warmth"))

    def test_gain_is_relative_to_neutral(self):
        base = {"pos": [2.0, 2.0, 2.0], "zero": [2.0, 2.0, 2.0]}
        better = {"pos": [1.8, 1.9, 1.7], "zero": [2.0, 2.0, 2.0]}
        g = voicearms.gain(better, base, "m")
        self.assertAlmostEqual(g["mean"], 0.2, places=4)
        self.assertGreater(g["low"], 0)
        self.assertEqual(voicearms.gain(base, base, "m")["mean"], 0.0)


class LayersTest(unittest.TestCase):
    def test_finds_the_layer_where_moods_separate_and_reads_swap(self):
        import voicelayers

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "vectors").mkdir()
            shared = np.array([1.0, 0.0, 0.0, 0.0])
            for m, own in (("warmth", [0, 1.0, 0, 0]), ("sadness", [0, 0, 1.0, 0]), ("alarm", [0, 0, 0, 1.0])):
                v = np.stack([shared + 0.1 * np.array(own), shared + 0.1 * np.array(own), np.array(own)])
                np.save(tmp / "vectors" / f"{m}.npy", v)
            (tmp / "report.json").write_text(json.dumps({"moods": {
                "warmth": {"neutral_probe": 0.4, "swap": {"probe": 0.7}, "verdict": "steers", "strength": 0.5}}}))
            voicelayers.main(tmp, tmp / "bundle.npz")
            with np.load(tmp / "bundle.npz") as z:
                self.assertEqual(sorted(z.files), ["alarm", "sadness", "warmth"])
            rows = json.loads((tmp / "layers.json").read_text())
            self.assertLess(rows["layers"][2]["cross_mood_cosine"], rows["layers"][0]["cross_mood_cosine"])
            self.assertAlmostEqual(rows["swap"]["warmth"]["swap_lift"], 0.3)
            self.assertIn("Most distinct layers: 2", (tmp / "layers.md").read_text())


class ReferenceTest(unittest.TestCase):
    def test_first_in_window_else_longest(self):
        from types import SimpleNamespace as C

        short, ok, long_ = C(id="a", duration_s=1.2), C(id="b", duration_s=5.0), C(id="c", duration_s=20.0)
        self.assertEqual(voicearms.reference([short, ok, long_], (3, 15)).id, "b")
        self.assertEqual(voicearms.reference([short, long_], (3, 15)).id, "c")


class SharedTargetTest(unittest.TestCase):
    """D's target types on made-up shifts: one shared direction plus one loud odd pair."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch isn't installed")
        self.torch = torch

    def test_pca_keeps_the_shared_direction_and_drops_what_one_pair_carries(self):
        torch = self.torch
        g = torch.Generator().manual_seed(0)
        shared = torch.zeros(2, 16)
        shared[:, 0] = 3.0
        pairs = shared.expand(10, 2, 16).clone() + 0.1 * torch.randn(10, 2, 16, generator=g)
        pairs[0, :, 5] += 20.0  # one odd, loud pair
        mean, energy = voicearms.shared_target(pairs, "mean")
        self.assertIsNone(energy)
        self.assertTrue(torch.allclose(mean, pairs.mean(0)))
        one, energy = voicearms.shared_target(pairs, "pca:1")
        self.assertEqual(one.shape, (2, 16))
        self.assertGreater(float(one[0, 0]), 2.5)  # sign and size of the mean kept along the shared axis
        self.assertLess(abs(float(one[0, 5])), abs(float(mean[0, 5])) / 4)  # the odd pair mostly gone
        _, e2 = voicearms.shared_target(pairs, "pca:2")
        self.assertGreater(float(e2[0]), float(energy[0]))
        self.assertLessEqual(float(one[0].norm()), float(mean[0].norm()) + 1e-5)

    def test_targets_are_checked(self):
        self.assertEqual(voicearms.match_targets({}), ["mean", "pca:1", "pca:4"])
        with self.assertRaises(SystemExit):
            voicearms.match_targets({"match_targets": ["median"]})


class HookTest(unittest.TestCase):
    """The norm-preserving add and the decay, on a real tensor, without the model."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch isn't installed")
        self.torch = torch

    def hooked(self, renorm, decay, seq_from=None):
        torch = self.torch
        t = voicearms.ArmTTS.__new__(voicearms.ArmTTS)
        t.torch, t.mode, t.add, t.sums, t.counts = torch, "steer", {0: torch.tensor([0.0, 3.0])}, {}, {}
        t.renorm, t.decay, t.step, t.t, t.measure, t.seq_from = renorm, decay, 0, 0, False, seq_from
        return t, t._hook(0)

    def test_single_step_add_rescales_to_the_old_norm_and_decays(self):
        torch = self.torch
        t, hook = self.hooked(True, 0.5)
        h = torch.tensor([[[4.0, 0.0]]])
        a = hook(None, None, (h,))[0]
        self.assertAlmostEqual(float(a.norm()), 4.0, places=5)
        self.assertGreater(float(a[0, 0, 1]), 0)
        b = hook(None, None, (h,))[0]  # second frame: half the push
        self.assertLess(float(b[0, 0, 1]), float(a[0, 0, 1]))

    def test_teacher_forced_pass_steers_only_frame_positions(self):
        torch = self.torch
        t, hook = self.hooked(False, 1.0, seq_from=2)
        h = torch.zeros(1, 4, 2)
        a = hook(None, None, h)
        self.assertEqual(a[0, :2].abs().sum().item(), 0.0)
        self.assertEqual(a[0, 2:, 1].tolist(), [3.0, 3.0])


class TinyModelTest(unittest.TestCase):
    """The real ArmTTS code on a tiny random-weight Qwen3-TTS (shrunk from the 1.7B Base's config),
    on CPU. Needs qwen-tts (requirements-steer.txt) and VI_QWEN_TTS_DIR, a folder with the Base
    model's config.json and tokenizer files (no weights); skipped otherwise, as in CI.

    The key check: a greedy generation's own codes, teacher-forced through voicearms' layout,
    give back the same codes as each position's best guess and the same hidden states, so the
    fit and the likelihood score exactly what generation would do."""

    def setUp(self):
        folder = os.environ.get("VI_QWEN_TTS_DIR")
        try:
            import qwen_tts  # noqa: F401
            import torch
        except ImportError:
            self.skipTest("qwen-tts isn't installed")
        if not folder:
            self.skipTest("VI_QWEN_TTS_DIR isn't set")
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
        from transformers import AutoTokenizer

        torch.manual_seed(0)
        cfg = json.loads((Path(folder) / "config.json").read_text())
        t = cfg["talker_config"]
        t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=16, text_hidden_size=64)
        t["rope_scaling"]["mrope_section"] = [4, 2, 2]
        t["code_predictor_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                                          num_key_value_heads=2, head_dim=8, layer_types=["full_attention"])
        cfg["speaker_encoder_config"]["enc_dim"] = 64
        model = Qwen3TTSForConditionalGeneration(Qwen3TTSConfig(**cfg)).eval()
        processor = Qwen3TTSProcessor(tokenizer=AutoTokenizer.from_pretrained(folder, fix_mistral_regex=True))
        wrapper = Qwen3TTSModel(model=model, processor=processor, generate_defaults={})
        with mock.patch.object(Qwen3TTSModel, "from_pretrained", return_value=wrapper):
            self.tts = voicearms.ArmTTS("tiny", "cpu")
        self.torch = torch

    def greedy(self, text, spk, add=None):
        """Codes and per-frame hidden states from a greedy x-vector-only generation."""
        torch, tts = self.torch, self.tts
        ids = tts.model._tokenize_texts([tts.model._build_assistant_text(text)])
        prompt = {"ref_code": [None], "ref_spk_embedding": [torch.tensor(spk)], "x_vector_only_mode": [True], "icl_mode": [False]}
        tts._steering(add, renorm=True, decay=0.9)
        try:
            codes, hidden = tts.m.generate(input_ids=ids, voice_clone_prompt=prompt, languages=["english"], max_new_tokens=12,
                                           do_sample=False, subtalker_dosample=False, repetition_penalty=1.0)
        finally:
            tts._off()
        return codes[0], hidden[0]

    def test_teacher_forced_layout_matches_generation(self):
        torch, tts = self.torch, self.tts
        spk = torch.randn(64).tolist()
        for add in (None, {1: [0.5] * 64, 2: [-0.3] * 64}):
            codes, hidden = self.greedy("The supply shuttle docks at the third bay.", spk, add)
            n = codes.shape[0]
            self.assertGreater(n, 2)
            seq, p, targets, cd = tts._layout("The supply shuttle docks at the third bay.", codes.tolist(),
                                              torch.tensor(spk))
            tts._steering(add, renorm=True, decay=0.9)
            tts.seq_from = p if add else None
            try:
                with torch.no_grad():
                    out = tts.talker(inputs_embeds=seq, attention_mask=torch.ones(seq.shape[:2], dtype=torch.long),
                                     output_hidden_states=True, use_cache=False)
            finally:
                tts._off()
            logits = out.logits[0, p - 1: p - 1 + n].clone()
            vocab = tts.m.config.talker_config.vocab_size
            logits[:, vocab - 1024:] = float("-inf")  # generation suppresses these (all but the stop token)
            self.assertEqual(logits.argmax(-1).tolist(), codes[:, 0].tolist(), f"group 0, steered={bool(add)}")
            hid = out.hidden_states[0][-1][0, p - 1: p - 1 + n]
            self.assertTrue(torch.allclose(hid, hidden.to(hid.dtype), atol=1e-4), f"hidden states, steered={bool(add)}")
            sub_logits, _ = tts.talker.forward_sub_talker_finetune(cd, hid)
            self.assertEqual(sub_logits.argmax(-1).tolist(), codes[:, 1:].tolist(), f"groups 1-15, steered={bool(add)}")

    def test_fit_lowers_the_loss_of_its_clips_and_moves_only_the_token(self):
        torch, tts = self.torch, self.tts
        before = [p.clone() for p in tts.m.parameters()]
        target = torch.randn(64).tolist()
        clips = []
        for i, text in enumerate(["We should reach the relay soon.", "I checked the coolant lines twice."]):
            codes, _ = self.greedy(text, target)
            clips.append({"id": f"c{i}", "text": text, "codes": codes.tolist()})
        start = torch.randn(64).tolist()
        fit = tts.fit(clips, clips[:1], start, {"fit_steps": 30, "fit_lr": 0.05, "fit_eval_every": 10, "fit_batch": 2})
        self.assertLess(fit["held_loss"], fit["start_held_loss"])
        self.assertLess(tts.nll(clips[1], fit["token"]), tts.nll(clips[1], start))
        for a, b in zip(before, tts.m.parameters(), strict=True):
            self.assertTrue(torch.equal(a, b))
        self.assertLess(tts.nll(clips[0], start, {1: [0.0] * 64}) - tts.nll(clips[0], start), 1e-6)

    def test_residual_match_fit_moves_the_residual_toward_the_shared_shift(self):
        torch, tts = self.torch, self.tts
        before = [p.clone() for p in tts.m.parameters()]
        texts = ["We should reach the relay soon.", "I checked the coolant lines twice.", "The market closes at noon.",
                 "Someone moved my notes."]
        warm, plain = torch.randn(64), torch.randn(64)
        pos, zero, px, zx = [], [], [], []
        for i, text in enumerate(texts):
            for group, ex_list, xs, centre in (("p", pos, px, warm), ("z", zero, zx, plain)):
                x = (centre + 0.1 * torch.randn(64)).tolist()
                codes, _ = self.greedy(text, x)
                ex_list.append({"id": f"{group}{i}", "text": text, "codes": codes.tolist()})
                xs.append(x)
        init = torch.stack([torch.tensor(x) for x in zx]).mean(0).tolist()
        fit = tts.fit_match(pos[:3], px[:3], zero[:3], zx[:3], zero[3:], zx[3:], init, [1, 2],
                            {"fit_steps": 40, "fit_lr": 0.05, "fit_eval_every": 10, "fit_batch": 3})
        self.assertLess(fit["held_loss"], fit["start_held_loss"])
        self.assertEqual(set(fit["agreement"]), {"1", "2"})
        pca = tts.fit_match(pos[:3], px[:3], zero[:3], zx[:3], zero[3:], zx[3:], init, [1, 2],
                            {"fit_steps": 20, "fit_lr": 0.05, "fit_eval_every": 10, "fit_batch": 3}, "pca:1")
        self.assertLess(pca["held_loss"], pca["start_held_loss"])
        self.assertTrue(all(0 < v <= 1.0001 for v in pca["mean_kept"].values()))
        for a, b in zip(before, tts.m.parameters(), strict=True):
            self.assertTrue(torch.equal(a, b))

    def test_xvector_is_the_speaker_token_width(self):
        clip = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            clip.audio = Path(tmp) / "a.wav"
            steer.write_wav(clip.audio, steer.fake_wave("x", rate=24_000, seconds=1.0))
            self.assertEqual(len(self.tts.xvector(clip)), 64)


if __name__ == "__main__":
    unittest.main()
