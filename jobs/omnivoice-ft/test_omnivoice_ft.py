"""Tests for the OmniVoice full fine-tune pipeline (CPU only, tiny model, no downloads).

Run with:
    nice -n 19 <venv>/bin/python -m pytest jobs/omnivoice-ft/ -q
On NixOS the venv needs its torch lib dir on the loader path first:
    LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/torch/lib

Covers the data manifest (label-blind, replay flag), the shard processor layout, the checkpoint and
checksum contract, KL on replay rows, a simulated mid-run kill followed by resume (model + optimizer
+ scheduler restored; resumed training matches a control run), the collapse-guard restart through the
script's own wiring, and style-slot inference with the 3-branch CFG (tokens-only path).
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "omnistyle"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_manifest as DM  # noqa: E402
import ft_data as FD  # noqa: E402
import infer_listening as IL  # noqa: E402
import smoke_style as S  # noqa: E402
import style_data as SD  # noqa: E402
import train_ft as TF  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _read_jsonl(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _state_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    return a == b


def _tiny_cfg(out: str, **kw) -> TF.FTTrainConfig:
    base = dict(
        output_dir=out, synthetic=True, batch_size=8, logging_steps=10**9, seed=0,
        kl_weight=0.0, guard=False, mixed_precision="no", gradient_accumulation_steps=1,
    )
    base.update(kw)
    return TF.FTTrainConfig(**base)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """3-step tiny run with the KL reference attached (no replay rows -> KL stays 0)."""
    out = str(tmp_path_factory.mktemp("tiny") / "run")
    cfg = _tiny_cfg(out, steps=3, save_steps=3, kl_weight=1.0, synthetic_replay_fraction=0.0)
    trainer, model = TF.run_training(cfg)
    return {"out": out, "ckpt": os.path.join(out, "checkpoint-3"), "cfg": cfg, "trainer": trainer}


# ---------------------------------------------------------------------------
# data_manifest.py: label-blind rows, replay flag, no audio copying
# ---------------------------------------------------------------------------


def test_manifest_is_label_blind_and_marks_replay(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "a.wav").write_bytes(b"")
    (pool / "b.wav").write_bytes(b"")

    raw = _write_jsonl(tmp_path / "train_raw.jsonl", [
        {"id": "a", "audio": "a.wav", "text": "[laughs] hello   there",
         "mood": "SECRET-MOOD-A", "rating": 5},
        {"id": "b", "audio": "b.wav", "text": "bye", "hand_label": "SECRET-LABEL-B"},
    ])
    replay = _write_jsonl(tmp_path / "replay.jsonl", [
        {"id": "a", "audio_path": str(pool / "a.wav"), "text": "hello there"},
    ])
    out = tmp_path / "manifest.jsonl"
    summary = DM.build_manifest(raw_jsonls=[raw], replay_jsonls=[replay], audio_root=str(pool),
                                out=str(out))

    rows = DM.read_jsonl(out)
    assert [r["id"] for r in rows] == ["a", "b", "a-replay"]
    assert rows[0]["text"] == "hello there"  # bracketed annotation stripped, whitespace collapsed
    assert rows[0]["audio_path"] == str((pool / "a.wav").resolve())
    assert rows[2]["replay"] is True and "replay" not in rows[0]
    assert all(set(r) <= {"id", "audio_path", "text", "language_id", "replay"} for r in rows)
    # no legacy label is read or written anywhere in the manifest
    text = out.read_text(encoding="utf-8")
    assert "SECRET" not in text and "mood" not in text and "hand_label" not in text
    assert summary["fields_read"] == ["id", "text", "audio_path", "audio"]
    assert summary["n_replay"] == 1 and summary["audio_copied"] is False
    # audio is referenced, never copied next to the manifest
    assert not (tmp_path / "a.wav").exists()

    empty = _write_jsonl(tmp_path / "empty.jsonl", [{"id": "c", "audio": "c.wav", "text": ""}])
    with pytest.raises(SystemExit):
        DM.build_manifest(raw_jsonls=[empty], audio_root=str(pool), out=str(tmp_path / "m2.jsonl"))
    s2 = DM.build_manifest(raw_jsonls=[empty], audio_root=str(pool), out=str(tmp_path / "m3.jsonl"),
                           allow_empty_text=True, check_audio=False)
    assert s2["n_empty_text"] == 1 and DM.read_jsonl(tmp_path / "m3.jsonl")[0]["text"] == ""


def test_manifest_from_pool_dir_with_texts(tmp_path):
    pool = tmp_path / "wavs"
    pool.mkdir()
    for name in ("x.wav", "y.flac", "notes.txt"):
        (pool / name).write_bytes(b"")
    texts = _write_jsonl(tmp_path / "texts.jsonl", [{"id": "x", "text": "ex"},
                                                    {"id": "y", "text": "why"}])
    out = tmp_path / "m.jsonl"
    summary = DM.build_manifest(pool_dir=str(pool), texts_jsonls=[texts], out=str(out),
                                check_audio=False)
    rows = {r["id"]: r["text"] for r in DM.read_jsonl(out)}
    assert rows == {"x": "ex", "y": "why"}  # notes.txt is not audio
    assert summary["sources"]["pool_dir"] == str(pool)


# ---------------------------------------------------------------------------
# ft_data.py: window choice, layout, replay flag, bundle fields
# ---------------------------------------------------------------------------


def test_choose_windows_uses_different_part_of_the_recording():
    tokens = torch.arange(8 * 40).reshape(8, 40)
    target, crop = FD.choose_windows(tokens, target_frames=10, crop_frames=8,
                                     rng=random.Random(0))
    assert target.shape == (8, 10) and crop.shape == (8, 8)
    t0 = int(target[0, 0]) % 40
    c0 = int(crop[0, 0]) % 40
    assert c0 + 8 <= t0 or t0 + 10 <= c0, (t0, c0)


def test_shard_processor_layout_and_replay(monkeypatch):
    monkeypatch.setattr(SD.random, "random", lambda: 0.999)  # keep conditioning, keep the prefix
    proc = FD.StyleShardProcessor(SD.FakeTextTokenizer(), n_style=3, target_frames=12, crop_frames=6)
    clip = torch.randint(0, 64, (8, 30))
    out = proc({"audio_tokens": clip, "label": {"text": "hi"}})
    assert out["kl"] is False
    assert out["style_tokens"].shape == (8, 6)
    assert int(out["style_slot"].sum()) == 3
    assert out["input_ids"].shape[0] == 8
    assert (out["labels"][:, out["style_slot"]] == -100).all()

    replay = proc({"audio_tokens": clip, "label": {"text": "hi", "replay": True}})
    assert replay["kl"] is True and int(replay["style_slot"].sum()) == 0
    assert int((replay["labels"] != -100).sum()) > 0  # loss positions still exist


def test_shard_processor_collates_through_the_real_collator():
    """Regression (r6): upstream's PaddingDataCollator reads ``processor.text_tokenizer``.

    The collator gets the real StyleShardProcessor here, not a plain StyleSampleProcessor, which
    is what hid the missing attribute from the synthetic-bundle tests."""
    proc = FD.StyleShardProcessor(SD.FakeTextTokenizer(), n_style=3, target_frames=12, crop_frames=6)
    assert proc.text_tokenizer is proc.base.text_tokenizer
    collate = SD.StylePaddingCollator(proc, batch_tokens=256)
    raw = [
        {"audio_tokens": torch.randint(0, 64, (8, 30)), "label": {"text": "hi"}},
        {"audio_tokens": torch.randint(0, 64, (8, 40)), "label": {"text": "there", "replay": True}},
    ]
    batch = collate([proc(r) for r in raw])
    for key in ("input_ids", "labels", "audio_mask", "style_slot", "style_tokens", "style_valid",
                "kl_mask"):
        assert key in batch, key
    assert batch["input_ids"].shape[0] == 2 and batch["input_ids"].shape[1] == 8
    assert batch["kl_mask"].tolist() == [False, True]


def test_synthetic_bundle_exposes_style_and_probe_fields():
    bundle = FD.build_synthetic_bundle(batch_size=4, seed=0)
    batch = next(iter(bundle.loader))
    for key in ("style_slot", "style_tokens", "style_valid", "kl_mask"):
        assert key in batch, key
    assert bundle.probe_kind == "labeled"
    stats = FD.labeled_probe_stats(S.build_model(seed=0), bundle.probe_dataset, n=8)
    assert "sep_ratio" in stats


# ---------------------------------------------------------------------------
# train_ft.py: shipped defaults, checkpoint contract, checksum tamper
# ---------------------------------------------------------------------------


def test_shipped_defaults_are_pinned():
    cfg = TF.FTTrainConfig(output_dir="/tmp/unused")
    assert cfg.learning_rate == 5e-4
    assert cfg.encoder_learning_rate > cfg.learning_rate
    assert cfg.style_quantizer == "none"
    assert tuple(cfg.mask_ratio_range) == (0.5, 1.0)
    assert cfg.prompt_ratio == 0.0
    assert cfg.kl_weight == 1.0
    assert cfg.guard is True
    # r25 B1: 2048 tokens per micro-batch x 4 accumulation, the same 8192 per optimizer step
    assert cfg.batch_tokens == 2048
    assert cfg.gradient_accumulation_steps == 4
    assert TF._resolved_mixed_precision(cfg) in ("bf16", "no")


def test_checkpoint_is_complete_and_verifiable(tiny_run):
    ckpt = tiny_run["ckpt"]
    assert os.path.isdir(ckpt)
    report = TF.verify_checkpoint(ckpt)
    assert report["ok"], report["problems"]
    files = set(report["files"])
    for required in ("config.json", "model.safetensors", "optimizer.bin", "scheduler.bin",
                     "random_states_0.pkl", "train_config.json", "ft_config.json",
                     "fake_tokenizer.json"):
        assert required in files, required

    saved = json.load(open(os.path.join(ckpt, "ft_config.json"), encoding="utf-8"))
    assert saved["learning_rate"] == 5e-4
    assert saved["style_quantizer"] == "none"
    assert saved["mask_ratio_range"] == [0.5, 1.0]
    assert saved["prompt_ratio"] == 0.0
    assert saved["kl_weight"] == 1.0

    sums = Path(ckpt, "SHA256SUMS").read_text().splitlines()
    assert len(sums) == len(files)  # report["files"] already excludes SHA256SUMS itself
    assert all(len(line.split("  ")[0]) == 64 for line in sums)

    # tampering with any file is detected
    tampered = str(Path(tiny_run["out"]).parent / "tampered")
    shutil.copytree(ckpt, tampered)
    with open(os.path.join(tampered, "optimizer.bin"), "ab") as f:
        f.write(b"x")
    bad = TF.verify_checkpoint(tampered)
    assert not bad["ok"]
    assert any("checksum mismatch: optimizer.bin" in p for p in bad["problems"]), bad["problems"]

    # an unlisted extra file is a problem too (strict completeness)
    shutil.copy(os.path.join(ckpt, "ft_config.json"), os.path.join(tampered, "extra.json"))
    bad2 = TF.verify_checkpoint(tampered)
    assert any("not listed" in p for p in bad2["problems"]), bad2["problems"]

    # the run's metrics carry the attempt and the KL term is present but zero (no replay rows)
    rows = _read_jsonl(os.path.join(tiny_run["out"], "style_metrics.jsonl"))
    assert rows and all(r["attempt"] == 0 for r in rows)
    assert all("loss/ce" in r for r in rows)
    assert rows[-1]["loss/kl"] <= 1e-6, rows[-1]


def test_kl_guard_uses_public_replay_rows(tmp_path):
    out = str(tmp_path / "run")
    # 4 steps: step 1 is a warmup no-op (upstream's schedule starts the lr at 0), so the first
    # update lands after micro-step 1; by step 4 the trained model and the frozen Base differ.
    TF.run_training(_tiny_cfg(out, steps=4, save_steps=10**9, logging_steps=1, kl_weight=1.0,
                              synthetic_replay_fraction=1.0))
    rows = _read_jsonl(os.path.join(out, "style_metrics.jsonl"))
    assert rows, "no metrics written"
    assert all("loss/kl" in r for r in rows)
    # The reference snapshot equals the model before any update; r25 stores it in bf16, so the
    # first KL is bf16 rounding only (measured 3.4e-7 on this tiny model; 4e-4 by step 3).
    assert rows[0]["loss/kl"] < 1e-5, rows[0]
    assert rows[-1]["loss/kl"] > 10 * rows[0]["loss/kl"], rows  # after the update the Base differs


# ---------------------------------------------------------------------------
# train_ft.py: simulated kill -> resume restores model/optimizer/scheduler
# ---------------------------------------------------------------------------


def test_kill_then_resume_restores_full_state(tmp_path):
    run_a = str(tmp_path / "a")
    run_c = str(tmp_path / "c")
    # accum 1: kill_at_step counts micro-steps, and this test is about resume, not accumulation
    base = dict(synthetic=True, steps=6, save_steps=3, batch_size=8, logging_steps=10**9,
                seed=0, kl_weight=0.0, guard=False, mixed_precision="no",
                gradient_accumulation_steps=1)

    with pytest.raises(TF.SimulatedKill):
        TF.run_training(TF.FTTrainConfig(output_dir=run_a, kill_at_step=5, **base))
    ck3 = os.path.join(run_a, "checkpoint-3")
    assert os.path.isdir(ck3)
    assert not os.path.exists(os.path.join(run_a, "checkpoint-6"))  # the kill prevented the final save
    assert TF.verify_checkpoint(ck3)["ok"]
    shutil.copytree(run_a, run_c)  # control started from the exact same checkpoint

    trainer_b, _ = TF.run_training(TF.FTTrainConfig(
        output_dir=run_a, resume_from_checkpoint=ck3, **base))
    trainer_c, _ = TF.run_training(TF.FTTrainConfig(
        output_dir=run_c, resume_from_checkpoint=os.path.join(run_c, "checkpoint-3"), **base))
    assert trainer_b.global_step == 6 and trainer_c.global_step == 6

    ck6_b = os.path.join(run_a, "checkpoint-6")
    ck6_c = os.path.join(run_c, "checkpoint-6")
    assert TF.verify_checkpoint(ck6_b)["ok"]
    wb, wc = load_file(os.path.join(ck6_b, "model.safetensors")), load_file(
        os.path.join(ck6_c, "model.safetensors"))
    assert set(wb) == set(wc)
    assert all(torch.equal(wb[k], wc[k]) for k in wb), "resumed run diverged from the control run"

    # resume with nothing left to train: the restored model/optimizer/scheduler are re-saved as-is
    run_d = str(tmp_path / "d")
    trainer_d, _ = TF.run_training(TF.FTTrainConfig(
        output_dir=run_d, resume_from_checkpoint=ck3, **{**base, "steps": 3}))
    assert trainer_d.global_step == 3
    wa = load_file(os.path.join(ck3, "model.safetensors"))
    wd = load_file(os.path.join(run_d, "checkpoint-3", "model.safetensors"))
    assert all(torch.equal(wa[k], wd[k]) for k in wa)
    for name in ("optimizer.bin", "scheduler.bin"):
        sa = torch.load(os.path.join(ck3, name), map_location="cpu", weights_only=False)
        sd = torch.load(os.path.join(run_d, "checkpoint-3", name), map_location="cpu", weights_only=False)
        assert _state_equal(sa, sd), f"{name} was not restored consistently"


# ---------------------------------------------------------------------------
# train_ft.py: collapse guard restart path, through the pipeline's own wiring
# ---------------------------------------------------------------------------


def test_collapse_guard_restarts_with_new_seed(tmp_path):
    out = str(tmp_path / "run")
    cfg = _tiny_cfg(out, steps=150, save_steps=75, batch_size=16, seed=42,
                    guard=True, guard_warmup=40, guard_every=25, guard_patience=1,
                    guard_attempts=2, inject_collapse_at=30)
    trainer, model = TF.run_training(cfg)

    log = trainer.guard_log
    first = [r for r in log if r["attempt"] == 0]
    second = [r for r in log if r["attempt"] == 1]
    assert trainer.guard_attempts == 2, log
    assert first and first[0]["collapsed"] and first[0]["sep_ratio"] == 0.0, first
    assert second and second[0]["seed"] == 43 and all(not r["collapsed"] for r in second), second
    assert trainer.global_step == 150
    assert float(model.style_encoder.to_code.weight.detach().abs().sum()) > 0  # not the injected layer

    # attempt 0's metrics were cleared before the restart; only attempt 1 remains
    rows = _read_jsonl(os.path.join(out, "style_metrics.jsonl"))
    assert [r["attempt"] for r in rows] == [1], rows
    guard_rows = _read_jsonl(os.path.join(out, "collapse_guard.jsonl"))
    assert {r["attempt"] for r in guard_rows} == {0, 1}
    assert os.path.isdir(os.path.join(out, "checkpoint-75"))
    assert TF.verify_checkpoint(os.path.join(out, "checkpoint-150"))["ok"]


# ---------------------------------------------------------------------------
# infer_listening.py: style-slot inference, 3-branch CFG, labelled listening set
# ---------------------------------------------------------------------------


def _lines_json(path: Path) -> str:
    return _write_jsonl(path, [
        {"id": "L01", "register": "neutral", "text": "hello there"},
        {"id": "L02", "register": "warm", "text": "good morning"},
    ])


def test_inference_plan_labels_and_deterministic_tokens(tiny_run, tmp_path):
    tokens = SD.make_clip(style=0, T=24, n_styles=4, bias=0.8, C=8, vocab=64,
                          gen=torch.Generator().manual_seed(0))
    style_pt = tmp_path / "styleA.pt"
    torch.save(tokens, style_pt)
    lines = _lines_json(tmp_path / "lines.jsonl")

    cfg = IL.InferConfig(
        checkpoint=tiny_run["ckpt"], out=str(tmp_path / "listen"),
        style_tokens=[str(style_pt)], lines_json=lines, seeds=(1, 2),
        target_frames=6, num_step=2, tokens_only=True, fake_tokenizer=True,
        device="cpu", max_styles=1, max_lines=1,
    )
    summary = IL.run_inference(cfg)
    rows = _read_jsonl(tmp_path / "listen" / "listening.jsonl")
    assert len(rows) == 2
    assert summary["checkpoint_verification"]["ok"]
    assert summary["n_items"] == 2 and summary["n_styles"] == 1 and summary["n_lines"] == 1
    assert all(r["step"] == 3 and r["style_id"] == "styleA" for r in rows)
    assert {r["seed"] for r in rows} == {1, 2}
    assert all(r["label"].startswith("styleA__L01__s") and r["label"].endswith("__step3")
               for r in rows)
    for row in rows:
        t = torch.load(row["file"], map_location="cpu", weights_only=True)
        assert t.shape == (8, 6)
        assert 0 <= int(t.min()) and int(t.max()) < 1025  # real audio-token ids (mask excluded)
    assert (tmp_path / "listen" / "README.txt").exists()

    # same labels + same seeds -> exactly the same tokens
    summary2 = IL.run_inference(replace(cfg, out=str(tmp_path / "listen2")))
    for row in rows:
        t1 = torch.load(row["file"], map_location="cpu", weights_only=True)
        t2 = torch.load(tmp_path / "listen2" / "tokens" / f"{row['label']}.pt",
                        map_location="cpu", weights_only=True)
        assert torch.equal(t1, t2)
    assert summary2["n_items"] == 2


def test_inference_refuses_a_tampered_checkpoint(tiny_run, tmp_path):
    bad = str(tmp_path / "bad_ckpt")
    shutil.copytree(tiny_run["ckpt"], bad)
    with open(os.path.join(bad, "model.safetensors"), "ab") as f:
        f.write(b"x")
    tokens = SD.make_clip(style=0, T=16, n_styles=4, bias=0.8, C=8, vocab=64,
                          gen=torch.Generator().manual_seed(1))
    style_pt = tmp_path / "s.pt"
    torch.save(tokens, style_pt)
    cfg = IL.InferConfig(checkpoint=bad, out=str(tmp_path / "out"),
                         style_tokens=[str(style_pt)], lines_json=_lines_json(tmp_path / "l.jsonl"),
                         tokens_only=True, fake_tokenizer=True)
    with pytest.raises(SystemExit):
        IL.run_inference(cfg)


# ---------------------------------------------------------------------------
# home_realpath_check.py --save: save / resume / reload proof (F-CKPT-1), F-HOME-3 disk guard
# ---------------------------------------------------------------------------


def test_realpath_save_tiny_synthetic(tmp_path):
    """The --save code path (prove_save) on the tiny synthetic model: save checkpoint-1,
    resume it for one more step, save checkpoint-2, reload it."""
    import home_realpath_check as H
    from style_omnivoice import StyleOmniVoice

    out = tmp_path / "run"
    logs = []

    def make_cfg(steps, resume_from):
        return _tiny_cfg(str(out), steps=steps, save_steps=1, kl_weight=1.0,
                         resume_from_checkpoint=str(resume_from) if resume_from else None)

    def resume(ck1):
        trainer, _ = TF.run_training(make_cfg(2, ck1))
        assert trainer.global_step == 2

    reloaded = {}

    def reload(ck2):
        m = StyleOmniVoice.from_pretrained(str(ck2), train=True, dtype=torch.float32)
        reloaded["n"] = sum(p.numel() for p in m.parameters())

    res = H.prove_save(out, make_cfg, resume, reload, log=logs.append)
    assert res["checkpoint_bytes"] > 0 and res["checkpoint_bytes_label"] == "measured"
    assert reloaded["n"] > 0
    assert TF.verify_checkpoint(str(out / "checkpoint-1"))["ok"]
    assert TF.verify_checkpoint(str(out / "checkpoint-2"))["ok"]
    assert any("fresh process resumed" in m for m in logs)


def test_realpath_save_refuses_low_disk(tmp_path, monkeypatch, capsys):
    """F-HOME-3: --save refuses (exit 3, clear message) below calculated checkpoint size + 20 GB."""
    import home_realpath_check as H

    need = 2 * H.calc_checkpoint_bytes() + int(20 * 1e9)
    H.check_save_disk(tmp_path, free=need)  # exactly enough: allowed
    with pytest.raises(H.SaveRefused, match="needs"):
        H.check_save_disk(tmp_path, free=need - 1)
    # end to end through main(): inject a fake free-space value; nothing is written
    monkeypatch.setattr(H, "free_bytes", lambda p: 10 * 10**9)
    monkeypatch.setattr(H, "mem_available_mb", lambda *a: 64000)
    scratch = tmp_path / "scratch"
    rc = H.main(["--data", str(tmp_path), "--base", str(tmp_path), "--scratch", str(scratch), "--save"])
    assert rc == H.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().out
    assert not (scratch / "result.json").exists()


def test_realpath_save_ram_skip_is_not_proof():
    import home_realpath_check as H

    assert H.calc_checkpoint_bytes() == 3 * 2_450_344_112


# ---------------------------------------------------------------------------
# WP4 guards: skipped-sample counting, GPU OOM exit 76, equal-product resume
# ---------------------------------------------------------------------------


def test_ft_data_counts_skipped_samples(tmp_path, monkeypatch):
    """The processor raises on 30 % of rows; upstream would skip them silently, the bundle raises."""
    import types

    class FakeStream(torch.utils.data.IterableDataset):
        def __init__(self, raw, batch_duration, min_length, max_length, max_sample, processor, length_fn):
            self.raw, self.processor = raw, processor

        def __iter__(self):
            batch = []
            for row in self.raw:
                try:
                    batch.append(self.processor(row))
                except Exception:
                    continue  # what upstream does
                if len(batch) == 2:
                    yield batch
                    batch = []

    class FakeRaw:
        def __init__(self, manifests, evaluation):
            pass

        def __iter__(self):
            return iter(range(100))

    class FakeProc:
        text_tokenizer = None

        def __init__(self, *a, **k):
            pass

        def __call__(self, i):
            if i % 10 < 3:
                raise RuntimeError(f"bad row {i}")
            return {"style_tokens": torch.zeros(1, 1, 1), "style_valid": torch.ones(1, 1)}

    mods = {
        "omnivoice.data.batching": types.SimpleNamespace(StreamLengthGroupDataset=FakeStream),
        "omnivoice.data.dataset": types.SimpleNamespace(
            WebDatasetReader=FakeRaw, prepare_data_manifests_from_json=lambda p: (["m"], None)),
        "transformers.trainer_utils": types.SimpleNamespace(seed_worker=lambda *a, **k: None),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(FD, "StyleShardProcessor", FakeProc)
    monkeypatch.setattr(FD.SD, "StylePaddingCollator", lambda proc, bt: (lambda b: {
        "style_tokens": b[0]["style_tokens"], "style_valid": b[0]["style_valid"]}))
    with pytest.raises(ValueError, match=r"ft_data: \d+ of \d+ samples failed in the processor"):
        FD.build_shard_bundle("cfg.json", None, num_workers=0, probe_batches=50)


def test_train_ft_too_few_batches(tmp_path):
    lst = tmp_path / "data.lst"
    lst.write_text("a.tar a.jsonl 3 2.0\n")
    cfgp = tmp_path / "data.json"
    cfgp.write_text(json.dumps({"train": [{"manifest_path": [str(lst)], "repeat": 1}]}))
    cfg = _tiny_cfg(str(tmp_path / "run"), synthetic=False, data_config=str(cfgp))
    with pytest.raises(ValueError, match=r"expected 1 batch per epoch at batch_tokens \d+, need at least 50"):
        TF._run_training_once(cfg, 0, 0, [])


def test_train_ft_oom_exit_76(tmp_path, monkeypatch, capsys):
    """The model forward raises a CUDA OOM at micro-step 3: exit 76 and oom.json, no crash."""
    from style_omnivoice import StyleOmniVoice

    real_forward = StyleOmniVoice.forward
    calls = {"n": 0}

    def forward(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        return real_forward(self, *a, **k)

    monkeypatch.setattr(StyleOmniVoice, "forward", forward)
    out = str(tmp_path / "run")
    rc = TF.main(["--synthetic", "--output-dir", out, "--steps", "10", "--save-steps", "100",
                  "--batch-size", "4", "--mixed-precision", "no", "--kl-weight", "0",
                  "--no-collapse-guard", "--batch-tokens", "8192"])
    assert rc == 76
    info = json.load(open(os.path.join(out, "oom.json")))
    assert info["batch_tokens"] == 8192 and info["step"] >= 1
    # r23: torch's own first line and the memory figures are printed, and kept in oom.json
    err = capsys.readouterr().err
    assert "train_ft: torch: CUDA out of memory (test)" in err
    assert "train_ft: cuda memory at the OOM:" in err
    assert info["error"] == "CUDA out of memory (test)" and "max_memory_reserved" in info


def _with_replay(monkeypatch, fraction):
    """main() with synthetic replay rows (the CLI has no flag for it), so the KL term runs."""
    real = TF.config_from_args
    monkeypatch.setattr(TF, "config_from_args",
                        lambda a: replace(real(a), synthetic_replay_fraction=fraction))


def _llm_class():
    return type(S.build_model(seed=0).llm)


def _oom_in_llm(monkeypatch, which):
    """Make the backbone (the student's llm or the KL reference's) raise a CUDA OOM in its first
    training-time forward. The reference is the bf16 copy; the student is the one with grads."""
    cls = _llm_class()
    real = cls.forward

    def forward(self, *a, **k):
        is_ref = next(self.parameters()).dtype == torch.bfloat16
        if (which == "reference" and is_ref) or (which == "student" and torch.is_grad_enabled()
                                                 and self.training):
            raise torch.cuda.OutOfMemoryError(f"CUDA out of memory ({which} test)")
        return real(self, *a, **k)

    monkeypatch.setattr(cls, "forward", forward)


@pytest.mark.parametrize("which,where", [("student", "student forward"),
                                         ("reference", "kl reference forward")])
def test_r25_oom_json_keeps_the_traceback(tmp_path, monkeypatch, capsys, which, where):
    # base: oom.json had the message only, so the two cases looked the same
    _oom_in_llm(monkeypatch, which)
    _with_replay(monkeypatch, 1.0)
    out = str(tmp_path / "run")
    rc = TF.main(["--synthetic", "--output-dir", out, "--steps", "3", "--save-steps", "100",
                  "--batch-size", "4", "--mixed-precision", "no", "--kl-weight", "1",
                  "--no-collapse-guard"])
    assert rc == 76
    info = json.load(open(os.path.join(out, "oom.json")))
    assert info["where"] == where, info
    assert any(f.endswith(" in forward") and "style_omnivoice.py" in f for f in info["traceback"])
    assert any(f.endswith(" in _kl_to_reference") for f in info["traceback"]) == (which == "reference")
    err = capsys.readouterr().err
    assert f"train_ft: oom in {where}: " in err


def test_r25_kl_reference_is_bf16_grad_free_and_the_kl_is_fp32(tmp_path, monkeypatch):
    # base: the reference was attached without a dtype, so it stayed fp32
    import torch.nn.functional as F
    seen = []
    real = F.log_softmax

    def log_softmax(x, *a, **k):
        seen.append(x.dtype)
        return real(x, *a, **k)

    monkeypatch.setattr(F, "log_softmax", log_softmax)
    cfg = _tiny_cfg(str(tmp_path / "run"), steps=2, save_steps=10**9, logging_steps=1,
                    kl_weight=1.0, synthetic_replay_fraction=1.0)
    _, model = TF.run_training(cfg)
    ref = model._ref_holder[0]
    assert {p.dtype for p in ref.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in ref.parameters()) and not ref.training
    # floating buffers (the rotary inv_freq) stay fp32: a bf16 copy would shift positions
    assert {b.dtype for b in ref.buffers() if b.is_floating_point()} == {torch.float32}
    # the student is untouched: fp32, trainable
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert any(p.requires_grad for p in model.parameters())
    # every log_softmax (the KL's two and the CE's) saw fp32 logits
    assert seen and set(seen) == {torch.float32}, set(seen)
    rows = _read_jsonl(os.path.join(cfg.output_dir, "style_metrics.jsonl"))
    assert all(math.isfinite(r["loss/kl"]) for r in rows), rows


class _FakeCuda:
    def is_available(self):
        return True

    def max_memory_allocated(self):
        return int(15.25 * 2**30)

    def max_memory_reserved(self):
        return 16 * 2**30


def test_r25_vram_peak_lines_and_sdpa_line_in_the_real_run(tmp_path, monkeypatch, capsys):
    # base: no peak line after the first optimizer step or at the end, no sdpa line
    import ft_guards as FG
    real = FG.vram_peak_line
    monkeypatch.setattr(FG, "vram_peak_line", lambda cuda, when: real(_FakeCuda(), when))
    _with_replay(monkeypatch, 0.5)
    out = str(tmp_path / "run")
    rc = TF.main(["--synthetic", "--output-dir", out, "--steps", "3", "--save-steps", "100",
                  "--batch-size", "4", "--mixed-precision", "no", "--kl-weight", "1",
                  "--no-collapse-guard"])
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    peaks = [l for l in lines if l.startswith("train_ft: vram peak: ")]
    assert peaks == [
        "train_ft: vram peak: allocated 15.25 GiB reserved 16.00 GiB (after the first optimizer step)",
        "train_ft: vram peak: allocated 15.25 GiB reserved 16.00 GiB (end of training)"], peaks
    sdpa = [l for l in lines if l.startswith("train_ft: sdpa-mask-warning: ")]
    assert len(sdpa) == 1, sdpa


def test_r25_no_vram_peak_line_without_cuda(tmp_path, capsys):
    out = str(tmp_path / "run")
    assert TF.main(["--synthetic", "--output-dir", out, "--steps", "1", "--save-steps", "100",
                    "--batch-size", "4", "--mixed-precision", "no", "--kl-weight", "0",
                    "--no-collapse-guard"]) == 0
    assert "vram peak" not in capsys.readouterr().out


def test_r25_accumulation_counts_optimizer_steps(tmp_path, monkeypatch):
    # --steps counts optimizer steps: 3 steps at accumulation 4 are 12 micro-batches, the
    # optimizer and the scheduler move 3 times and checkpoint-3 is written, so the tokens per
    # optimizer step (and the LR schedule over the run) are what they were at 8192 x 1.
    made = {}
    real_opt, real_sched = TF.build_optimizer, TF.build_scheduler

    def build_optimizer(model, cfg):
        opt = made["opt"] = real_opt(model, cfg)
        made["opt_steps"] = 0

        def count(*_):
            made["opt_steps"] += 1

        opt.register_step_post_hook(count)
        return opt

    def build_scheduler(opt, cfg):
        made["sched"] = real_sched(opt, cfg)
        return made["sched"]

    monkeypatch.setattr(TF, "build_optimizer", build_optimizer)
    monkeypatch.setattr(TF, "build_scheduler", build_scheduler)
    out = str(tmp_path / "run")
    cfg = _tiny_cfg(out, steps=3, save_steps=3, logging_steps=1, gradient_accumulation_steps=4)
    TF.run_training(cfg)
    rows = _read_jsonl(os.path.join(out, "style_metrics.jsonl"))
    assert [r["micro_step"] for r in rows] == list(range(1, 13))
    assert made["opt_steps"] == 3
    assert made["sched"].last_epoch == 3
    assert os.path.isdir(os.path.join(out, "checkpoint-3"))
    assert not os.path.isdir(os.path.join(out, "checkpoint-12"))


def test_train_ft_resume_allows_equal_product(tmp_path):
    ck = tmp_path / "checkpoint-5"
    ck.mkdir()
    saved = TF._config_dict(_tiny_cfg(str(tmp_path / "run"), batch_tokens=8192, gradient_accumulation_steps=1))
    (ck / "ft_config.json").write_text(json.dumps(saved))
    ok = _tiny_cfg(str(tmp_path / "run"), batch_tokens=4096, gradient_accumulation_steps=2,
                   resume_from_checkpoint=str(ck))
    TF._check_resume_config(ok)  # same tokens per optimizer step: allowed
    bad = replace(ok, gradient_accumulation_steps=1)
    with pytest.raises(SystemExit, match="product"):
        TF._check_resume_config(bad)


# ---------------------------------------------------------------------------
# r26: the KL reference forward runs only on the KL rows (the public replay rows)
# ---------------------------------------------------------------------------


def _full_batch_kl(model, logits, labels, input_ids, audio_mask, attention_mask, style_slot, kl_mask):
    """The e05a3d4 _kl_to_reference: the reference forward over the WHOLE batch (test oracle)."""
    import torch.nn.functional as F
    ref = model._ref_holder[0]
    B, C, S_, V = logits.shape
    dev = logits.device
    keep = ~style_slot if style_slot is not None else torch.ones(B, S_, dtype=torch.bool, device=dev)
    base_mask = attention_mask if attention_mask is not None else torch.ones(B, 1, S_, S_, dtype=torch.bool, device=dev)
    ref_mask = base_mask & keep[:, None, None, :]
    ref_pos = (keep.long().cumsum(-1) - 1).clamp(min=0)
    sel = labels != -100
    if kl_mask is not None:
        sel = sel & kl_mask.view(B, 1, 1)
    pos_any = sel.any(1)
    if not bool(pos_any.any()):
        return logits.new_zeros(())
    with torch.no_grad():
        embeds = ref._prepare_embed_inputs(input_ids, audio_mask).to(next(ref.parameters()).dtype)
        hidden = ref.llm(inputs_embeds=embeds, attention_mask=ref_mask, position_ids=ref_pos, return_dict=True)[0]
        ref_lp = F.log_softmax(ref.audio_heads(hidden[pos_any]).view(-1, C, V).float(), dim=-1)
    sel_m = sel.permute(0, 2, 1)[pos_any]
    mdl_lp = F.log_softmax(logits.permute(0, 2, 1, 3)[pos_any].float(), dim=-1)
    kl = (ref_lp.exp() * (ref_lp - mdl_lp)).sum(-1)
    layer_means = (kl * sel_m).sum(0) / sel_m.sum(0).clamp(min=1).float()
    return (layer_means * torch.tensor(model.normalized_audio_codebook_weights, device=dev)).sum()


def _r26_model_and_batch(kl_rows):
    """Tiny student + bf16 reference (as train_ft attaches it) and one 6-row batch whose kl_mask
    is set by hand; rows outside kl_rows keep whatever style slot the processor gave them."""
    random.seed(0)
    torch.manual_seed(0)
    model = S.build_model(seed=0)
    TF.attach_kl_reference(model)
    model.configure_training(kl_weight=1.0, style_aux_weight=1.0)
    model.train()
    _, _, loader = S.build_loader(batch_size=6, seed=0, replay_fraction=0.5)
    batch = next(iter(loader))
    batch["kl_mask"] = torch.tensor([i in kl_rows for i in range(6)])
    ref_rows = []
    model._ref_holder[0].llm.register_forward_pre_hook(
        lambda m, a, k: ref_rows.append(k["inputs_embeds"].shape[0]), with_kwargs=True)
    return model, batch, ref_rows


@pytest.mark.parametrize("kl_rows", [(1, 4), (0, 1, 2, 3, 4, 5)], ids=["mixed", "all"])
def test_r26_sliced_kl_equals_the_whole_batch_kl(kl_rows):
    # base: the reference forward ran on all 6 rows (ref_rows == [6])
    model, batch, ref_rows = _r26_model_and_batch(kl_rows)
    out = model(**batch)
    assert ref_rows == [len(kl_rows)], ref_rows
    assert model._kl_rows == (len(kl_rows), 6)
    want = _full_batch_kl(model, out.logits, batch["labels"], batch["input_ids"], batch["audio_mask"],
                          batch["attention_mask"], batch.get("style_slot"), batch["kl_mask"])
    assert float(want.detach()) > 0
    torch.testing.assert_close(out.kl_loss.detach(), want.detach(), rtol=1e-4, atol=1e-6)


def test_r26_no_kl_rows_returns_zero_without_touching_the_reference(monkeypatch):
    # base: same zero and no forward, but nothing recorded the row counts (no _kl_rows)
    model, batch, ref_rows = _r26_model_and_batch(())
    ref = model._ref_holder[0]
    touched = []
    monkeypatch.setattr(ref, "to", lambda *a, **k: touched.append("to"))
    monkeypatch.setattr(ref, "_prepare_embed_inputs", lambda *a, **k: touched.append("embed"))
    out = model(**batch)
    assert ref_rows == [] and touched == []
    assert model._kl_rows == (0, 6)
    kl = out.kl_loss
    assert float(kl) == 0.0 and kl.shape == () and kl.dtype == out.logits.dtype
    assert not kl.requires_grad


def test_r26_kl_gradient_reaches_the_student_only():
    # base: gradients were the same, but the reference forward still ran on all 6 rows
    model, batch, ref_rows = _r26_model_and_batch((2, 5))
    out = model(**batch)
    assert ref_rows == [2]
    out.kl_loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert all(p.grad is None for p in model._ref_holder[0].parameters())


def test_r26_kl_rows_line_per_optimizer_step(tmp_path, capsys):
    # base: no "train_ft: kl-rows:" line at all
    cfg = _tiny_cfg(str(tmp_path / "run"), steps=3, save_steps=10**9, logging_steps=1,
                    kl_weight=1.0, synthetic_replay_fraction=0.5, batch_size=4,
                    gradient_accumulation_steps=2)
    TF.run_training(cfg)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("train_ft: kl-rows: ")]
    rows = _read_jsonl(os.path.join(cfg.output_dir, "style_metrics.jsonl"))
    assert len(rows) == 6
    want = []
    for i in range(0, 6, 2):
        used = int(rows[i]["kl/rows_used"] + rows[i + 1]["kl/rows_used"])
        total = int(rows[i]["kl/rows_total"] + rows[i + 1]["kl/rows_total"])
        want.append(f"train_ft: kl-rows: used {used} of {total} rows (share {used / total:.3f})")
    assert lines == want, (lines, want)
    assert all(int(r["kl/rows_total"]) == 4 for r in rows)
