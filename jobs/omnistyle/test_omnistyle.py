"""Fast CPU tests for the style-prefix route (tiny Qwen3, synthetic tokens).  No GPU, no real data, no downloads.

    nice -n 19 python -m pytest -q test_omnistyle.py            # tiny checks, a few minutes
    OMNISTYLE_REAL=1 nice -n 19 python -m pytest -q test_omnistyle.py -k real   # also load k2-fsa/OmniVoice
"""

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smoke_style as S  # noqa: E402
from style_omnivoice import StyleOmniVoice, StyleOmniVoiceConfig  # noqa: E402
from train_style import run_training  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


@pytest.fixture(scope="module")
def trained():
    # shipped default: continuous bottleneck, phase-2 lr 5e-4
    model, ds, rows, secs = S.train_reference_model("none", pre_steps=300, steps=1000)
    return model, ds, rows


def test_a_prefix_helps(trained):
    model, ds, _ = trained
    r = S.check_a_prefix_helps(model, ds)
    assert r["pass"], r


def test_b_decode_bias(trained):
    model, ds, _ = trained
    r = S.check_b_continuous(model, ds)
    assert r["pass"], r


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_b_continuous_decode_bias_all_seeds(seed):
    """Continuous check (b) on all five seeds with the shipped default (continuous bottleneck, lr 5e-4).

    There are no discrete codes to decode from, so check (b) decodes from each style's centroid vectors:
    per style, own-share >= 0.5 and > 2x the best other style.  The old FSQ default failed this on seeds
    0 and 3 (see workers/smoke/fsq-fix-report.md); the continuous route at phase-2 lr 1e-3 failed on seed 4,
    which the 5e-4 default fixes (workers/smoke/contin-fix-report.md).
    """
    model, ds, _, _ = S.train_reference_model("none", pre_steps=300, steps=1000, seed=seed)
    r = S.check_b_continuous(model, ds)
    assert r["pass"], r


def test_c_save_load_roundtrip(trained, tmp_path):
    r = S.check_c_roundtrip(trained[0], workdir=str(tmp_path))
    assert r["pass"], r


@pytest.mark.parametrize("quantizer", ["fsq", "none", "vq"])
def test_d_style_dropped_still_gives_all_encoder_grads(quantizer):
    r = S.check_d_style_dropped_grads(quantizer)
    assert r["pass"], r


def test_e_resume(tmp_path):
    r = S.check_e_resume(str(tmp_path))
    assert r["pass"], r


def test_g_cfg_and_scaling(trained):
    model, ds, _ = trained
    r = S.check_g_cfg_and_scale(model, ds)
    assert r["pass"], r


def test_kl_guard():
    r = S.check_kl_guard()
    assert r["pass"], r


def test_collapse_metrics_are_logged(trained):
    rows = trained[2]
    for key in ("style/z_spread", "style/out_rms"):
        assert key in rows[-1], key
    assert rows[-1]["style/z_spread"] > 0  # not collapsed onto one code across the batch


def test_collapse_probe_detects_forced_collapse(trained):
    """The guard's probe separates a healthy run from the seed-4 collapse mode (a style-blind `to_code`)."""
    model, ds, _ = trained
    healthy = S.bottleneck_stats(model, ds, n=32)
    assert healthy["sep_ratio"] >= 0.5, healthy
    collapsed = copy.deepcopy(model)
    with torch.no_grad():
        collapsed.style_encoder.to_code.weight.zero_()
        collapsed.style_encoder.to_code.bias.zero_()
    probe = S.bottleneck_stats(collapsed, ds, n=32)
    assert probe["sep_ratio"] < 0.5 and probe["within_rms"] == 0.0, probe


def test_collapse_guard_restarts_with_new_seed(tmp_path):
    """Injection test: force the collapse, the guard abandons the attempt and retrains with seed + 1.

    A from-scratch tiny model needs a few hundred steps to become gate-healthy, so this run uses the
    guard's real sep_ratio threshold (0.5) with a shorter warm start; the guard check after the injection
    sees the forced style-blind code (sep_ratio 0), and the restarted attempt stays above the threshold.
    """
    trainer, model = run_training(
        str(tmp_path / "run"), steps=150, save_steps=10**9, batch_size=16, logging_steps=10**9,
        seed=0, guard_warmup=40, guard_every=25, guard_patience=1, inject_collapse_at=30,
    )
    log = trainer.guard_log
    first = [r for r in log if r["attempt"] == 0]
    second = [r for r in log if r["attempt"] == 1]
    assert trainer.guard_attempts == 2, log
    assert first and first[0]["collapsed"] and first[0]["sep_ratio"] == 0.0, first
    assert second and second[0]["seed"] == 1 and all(not r["collapsed"] for r in second), second
    assert trainer.global_step == 150
    assert float(model.style_encoder.to_code.weight.abs().sum()) > 0  # not the injected degenerate layer
    probe = S.bottleneck_stats(model, S.build_loader(seed=1)[0], n=32)
    assert probe["sep_ratio"] >= 0.5, probe


def test_fsq_quantiser_unit():
    m = S.build_model(style_quantizer="fsq", style_fsq_levels=[5, 3, 7])
    enc = m.style_encoder
    z = torch.randn(4, 2, 3, requires_grad=True)
    zq, idx = enc._fsq(z)
    assert zq.min() >= -1 and zq.max() <= 1
    assert int(idx.max()) < 5 * 3 * 7 and int(idx.min()) >= 0
    # index -> vector is the exact inverse of the quantiser, and the gradient passes straight through
    assert torch.allclose(enc.codes_to_z(idx), zq.detach())
    zq.sum().backward()
    assert z.grad is not None and bool((z.grad != 0).any())
    with pytest.raises(AssertionError):
        StyleOmniVoiceConfig(style_fsq_levels=[4, 5])  # even levels are not supported


def test_vq_path_plumbing_only():
    """VQ is kept to document its collapse; here only that it runs, logs usage and saves."""
    m = S.build_model(style_quantizer="vq")
    _, _, loader = S.build_loader()
    rows, _ = S.train_steps(m, loader, 5, log_every=1)
    assert "style/commit_loss" in rows[-1] and "style/codes_used" in rows[-1]


def test_style_scale_matches_audio_embedding_norm():
    m = S.build_model()
    batch = S._fixed_batch()
    m.eval()
    enc = m.style_encoder(batch["style_tokens"], batch["style_valid"])
    C, V = m.config.num_audio_codebook, m.config.audio_vocab_size
    ids = torch.randint(0, V - 1, (2048, C)) + m.codebook_layer_offsets.view(1, -1)
    real = m.audio_embeddings(ids).sum(1).norm(dim=-1).mean()
    assert abs(float(enc.vectors.norm(dim=-1).mean()) / float(real) - 1) < 0.05


@pytest.mark.skipif(not os.environ.get("OMNISTYLE_REAL"), reason="set OMNISTYLE_REAL=1 (downloads the public k2-fsa/OmniVoice)")
def test_f_real_model_load():
    r = S.check_f_real_model()
    if r.get("pass") is None:
        pytest.skip(r["skipped"])
    assert r["pass"], r
