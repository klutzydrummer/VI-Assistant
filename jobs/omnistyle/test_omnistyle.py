"""Fast CPU tests for the style-prefix route (tiny Qwen3, synthetic tokens).  No GPU, no real data, no downloads.

    nice -n 19 python -m pytest -q test_omnistyle.py            # tiny checks, a few minutes
    OMNISTYLE_REAL=1 nice -n 19 python -m pytest -q test_omnistyle.py -k real   # also load k2-fsa/OmniVoice
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smoke_style as S  # noqa: E402
from style_omnivoice import StyleOmniVoice, StyleOmniVoiceConfig  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


@pytest.fixture(scope="module")
def trained():
    model, ds, rows, secs = S.train_reference_model("fsq", pre_steps=300, steps=1000)
    return model, ds, rows


def test_a_prefix_helps(trained):
    model, ds, _ = trained
    r = S.check_a_prefix_helps(model, ds)
    assert r["pass"], r


def test_b_decode_bias(trained):
    model, ds, _ = trained
    r = S.check_b_decode_bias(model, ds)
    assert r["pass"], r


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_b_decode_bias_all_seeds(seed):
    """Check (b) on all five seeds with the shipped FSQ defaults ([7, 7, 5, 5], tanh bound, fan-in init).
    The old [7, 5, 5] tanh configuration passed on 2 of 5 seeds but merged every style onto one code on the
    others (see README "Known defect"), so a single-seed test missed it.  Measured on 2026-11-13: the
    shipped defaults pass seeds 1, 2 and 4 and fail seeds 0 and 3, so this test is expected red on those two
    seeds until the owner picks a follow-up (see workers/smoke/fsq-fix-report.md)."""
    model, ds, _, _ = S.train_reference_model("fsq", pre_steps=300, steps=1000, seed=seed)
    r = S.check_b_decode_bias(model, ds)
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
    for key in ("style/z_spread", "style/codes_used", "style/code_perplexity", "style/codes_used_per_slot"):
        assert key in rows[-1], key
    assert rows[-1]["style/codes_used"] >= 2  # not collapsed onto a single code


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
