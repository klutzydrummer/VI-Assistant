"""CPU smoke checks (a)-(g) for the style-prefix route.  Tiny Qwen3, synthetic [8,T] token clips.

Each `check_*` function returns a dict (with a boolean "pass") so that both pytest and the CLI report
use the same code.  CLI:  nice -n 19 python smoke_style.py [--skip-real] [--out results.json]

Quantizer: the default is FSQ.  The VQ path is kept only so that its measured collapse stays reproducible
(`--quantizer vq`); `--quantizer none` is the continuous diagnosis baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import Qwen3Config  # noqa: E402

from style_data import (  # noqa: E402
    FakeTextTokenizer,
    StylePaddingCollator,
    StyleSampleProcessor,
    SyntheticStyleDataset,
    style_block,
)
from style_omnivoice import StyleOmniVoice, StyleOmniVoiceConfig  # noqa: E402

N_STYLES = 4
N_VEC = 2


# vocab must exceed the audio vocab (1025): upstream embeds input_ids[:,0,:] for audio positions too.
def tiny_llm_config(hidden: int = 64, layers: int = 2, vocab: int = 1100) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=512,
        attn_implementation="sdpa",
    )


def tiny_config(**kw) -> StyleOmniVoiceConfig:
    llm = kw.pop("llm_config", None) or tiny_llm_config()
    defaults = dict(
        llm_config=llm,
        audio_codebook_weights=[8, 8, 6, 6, 4, 4, 2, 2],
        style_n_vectors=N_VEC,
        style_enc_dim=64,
        style_enc_layers=2,
        style_enc_heads=4,
        style_code_dim=8,
        style_codebook_size=16,
        style_quantizer="fsq",
        style_noise_std=0.1,
    )
    defaults.update(kw)
    return StyleOmniVoiceConfig(**defaults)


def build_model(seed: int = 0, calibrate: bool = True, **kw) -> StyleOmniVoice:
    torch.manual_seed(seed)
    model = StyleOmniVoice(tiny_config(**kw))
    if calibrate:
        model.ensure_style_scale_calibrated()
    return model


def build_loader(batch_size: int = 16, seed: int = 0, n_clips: int = 100_000, replay_fraction: float = 0.0,
                 style_drop_ratio: float = 0.2, drop_cond_ratio: float = 0.1, shuffle: bool = True,
                 mask_ratio_range=(0.8, 1.0), bias: float = 0.8):
    ds = SyntheticStyleDataset(n_clips=n_clips, n_styles=N_STYLES, seed=seed, replay_fraction=replay_fraction, bias=bias)
    proc = StyleSampleProcessor(
        FakeTextTokenizer(), 8, 1024, n_style=N_VEC, mask_ratio_range=mask_ratio_range,
        drop_cond_ratio=drop_cond_ratio, style_drop_ratio=style_drop_ratio,
    )

    class Wrapped(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return proc(ds[i])

    return ds, proc, DataLoader(Wrapped(), batch_size=batch_size, shuffle=shuffle, collate_fn=StylePaddingCollator(proc))


def train_steps(model, loader, steps: int, lr: float = 3e-3, log_every: int = 25, log=None, seed: int = 0):
    """Plain CPU loop (the OmniTrainer path is exercised separately in the resume check)."""
    random.seed(seed)
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rows = []
    last = {}
    model.metrics_sink = lambda r: last.update(r)
    it = iter(loader)
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % log_every == 0 or step == 1:
            row = {"step": step, "loss": float(out.loss.detach()), **{k: round(v, 4) for k, v in last.items()}}
            rows.append(row)
            if log:
                log(row)
    return rows, time.time() - t0


@torch.no_grad()
def eval_ce(model, ds, shuffle_style: bool, n_batches: int = 8, seed: int = 7) -> float:
    """CE on fresh batches with the prefix kept for every sample; optionally the crops are shuffled across clips of other styles."""
    model.eval()
    # fully masked target: nothing in the clip itself reveals the style, only the prefix does
    _, proc, _ = build_loader(style_drop_ratio=0.0, drop_cond_ratio=0.0, seed=seed + 1000, mask_ratio_range=(1.0, 1.0))
    coll = StylePaddingCollator(proc)
    losses = []
    for b in range(n_batches):
        gen_ds = SyntheticStyleDataset(n_clips=32, n_styles=N_STYLES, seed=seed + b)
        random.seed(seed + b)
        torch.manual_seed(seed + b)
        batch = coll([proc(gen_ds[i]) for i in range(32)])
        if shuffle_style:
            # roll by 1 within the styles cycle: sample i now gets the crop of a clip of style (i+1) % K
            batch["style_tokens"] = batch["style_tokens"].roll(shifts=1, dims=0)
            batch["style_valid"] = batch["style_valid"].roll(shifts=1, dims=0)
        losses.append(float(model(**batch).ce_loss))
    return sum(losses) / len(losses)


@torch.no_grad()
def modal_codes(model, ds, style: int, n: int = 64):
    model.eval()
    crops = ds.clips_of_style(style, n)
    enc = model.style_encoder(crops, torch.ones(n, crops.shape[2], dtype=torch.bool))
    codes = enc.codes  # [n, n_vec]
    tuples = [tuple(c.tolist()) for c in codes]
    modal = max(set(tuples), key=tuples.count)
    return modal, tuples.count(modal) / n


@torch.no_grad()
def decode_bias(model, ds, codes, n_clips: int = 24, target_len: int = 16, seed: int = 0):
    """Decode with one code tuple (one code per slot); return, per style j, the share of codebook-0/1 tokens that
    fall in style j's preferred blocks (cb0 in block(j,0), cb1 in block(j,1)).  Uniform noise gives about 8/64."""
    vecs = model.style_encoder.vectors_from_codes(torch.tensor(codes)).view(len(codes), -1)
    g = torch.Generator().manual_seed(seed)
    text = FakeTextTokenizer()("<|text_start|>abcdef<|text_end|>").input_ids[0]
    counts = torch.zeros(N_STYLES)
    total = 0
    for _ in range(n_clips):
        toks = model.generate_tokens(
            text, target_len, style_vectors=vecs, num_step=8, guidance_text=0.0, guidance_style=0.0,
            class_temperature=0.5, generator=g,
        )
        for s in range(N_STYLES):
            for cb in (0, 1):
                blk = style_block(s, cb, N_STYLES)
                counts[s] += ((toks[cb] >= blk.start) & (toks[cb] < blk.stop)).sum()
        total += 2 * toks.shape[1]
    return (counts / total).tolist()


def pretrain_base(model, steps: int = 300, lr: float = 1e-3, log=None, seed: int = 0):
    """Phase 1: train the tiny LLM WITHOUT the prefix (it plays the role of the pretrained Base model).

    The encoder is frozen and out of the loss, then re-initialised and re-calibrated afterwards, so phase 2
    starts from a fresh encoder on top of a base that already knows the marginal token statistics.
    """
    _, _, loader = build_loader(style_drop_ratio=1.0, seed=seed + 11)
    for p in model.style_encoder.parameters():
        p.requires_grad_(False)
    aux_w = model.style_aux_weight
    model.style_aux_weight = 0.0
    rows, secs = train_steps(model, loader, steps, lr=lr, log=log, seed=seed)
    model.style_aux_weight = aux_w
    for p in model.style_encoder.parameters():
        p.requires_grad_(True)
    model.style_encoder.reset_parameters()
    model.ensure_style_scale_calibrated()
    return rows, secs


def train_reference_model(quantizer: str = "fsq", pre_steps: int = 300, steps: int = 1000, lr: float = 1e-3,
                          log=None, seed: int = 0, **kw):
    """Phase 1 (base without prefix) + phase 2 (encoder + prefix).  Returns (model, dataset, rows, seconds)."""
    ds, _, loader = build_loader(seed=seed)
    model = build_model(seed=seed, style_quantizer=quantizer, **kw)
    pretrain_base(model, pre_steps, log=log, seed=seed)
    rows, secs = train_steps(model, loader, steps, lr=lr, log=log, seed=seed, log_every=max(1, steps // 10))
    return model, ds, rows, secs


# ---------------------------------------------------------------------------
# Checks (a)-(g)
# ---------------------------------------------------------------------------


def check_a_prefix_helps(model, ds, margin: float = 0.3) -> dict:
    """(a) CE with the true prefix is lower than with a prefix taken from a clip of another style."""
    true, shuf = eval_ce(model, ds, False), eval_ce(model, ds, True)
    return {"check": "a", "ce_true": true, "ce_shuffled": shuf, "margin_required": margin,
            "pass": bool(true + margin < shuf)}


def check_b_decode_bias(model, ds, min_own: float = 0.5) -> dict:
    """(b) Decoding with style k's code yields style k's token bias (own share the highest and above `min_own`)."""
    per_style = []
    ok = True
    for k in range(N_STYLES):
        code, purity = modal_codes(model, ds, k)
        shares = decode_bias(model, ds, code)
        top = max(range(N_STYLES), key=lambda j: shares[j])
        others = max(shares[j] for j in range(N_STYLES) if j != k)
        good = top == k and shares[k] >= min_own and shares[k] > 2 * others
        ok &= good
        per_style.append({"style": k, "code": list(code), "purity": round(purity, 3),
                          "shares": [round(x, 3) for x in shares], "pass": good})
    distinct = len({tuple(r["code"]) for r in per_style})
    return {"check": "b", "per_style": per_style, "distinct_codes": distinct, "pass": bool(ok and distinct == N_STYLES)}


def _fixed_batch(n: int = 8, seed: int = 5, style_drop: float = 0.0, replay_fraction: float = 0.0, mask=(1.0, 1.0)):
    _, proc, _ = build_loader(style_drop_ratio=style_drop, drop_cond_ratio=0.0, mask_ratio_range=mask)
    coll = StylePaddingCollator(proc)
    ds = SyntheticStyleDataset(n_clips=n, n_styles=N_STYLES, seed=seed, replay_fraction=replay_fraction)
    random.seed(seed)
    torch.manual_seed(seed)
    return coll([proc(ds[i]) for i in range(n)])


@torch.no_grad()
def check_c_roundtrip(model, workdir: str | None = None) -> dict:
    """(c) save_pretrained -> from_pretrained keeps config, encoder weights, buffers and logits."""
    model.eval()
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        model.save_pretrained(d)
        files = sorted(os.listdir(d))
        loaded = StyleOmniVoice.from_pretrained(d, train=True)
    loaded.eval()
    sd_a, sd_b = model.state_dict(), loaded.state_dict()
    enc_keys = [k for k in sd_a if k.startswith("style_encoder.")]
    missing = sorted(set(sd_a) - set(sd_b))
    extra = sorted(set(sd_b) - set(sd_a))
    max_diff_enc = max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in enc_keys if k in sd_b)
    max_diff_all = max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a if k in sd_b)
    batch = _fixed_batch()
    la, lb = model(**batch).logits, loaded(**batch).logits
    logit_diff = float((la - lb).abs().max())
    cfg_ok = (loaded.config.style_quantizer == model.config.style_quantizer
              and loaded.config.style_fsq_levels == model.config.style_fsq_levels
              and loaded.config.style_fsq_bound == model.config.style_fsq_bound
              and loaded.config.style_fsq_scale == model.config.style_fsq_scale
              and loaded.config.style_code_init_std == model.config.style_code_init_std
              and loaded.config.style_n_vectors == model.config.style_n_vectors)
    return {"check": "c", "files": files, "n_encoder_tensors": len(enc_keys), "missing": missing, "unexpected": extra,
            "max_abs_diff_encoder": max_diff_enc, "max_abs_diff_all": max_diff_all, "max_abs_logit_diff": logit_diff,
            "config_fields_ok": cfg_ok,
            "pass": bool(not missing and not extra and max_diff_all == 0.0 and logit_diff == 0.0 and cfg_ok)}


def check_d_style_dropped_grads(quantizer: str = "fsq") -> dict:
    """(d) A step where every sample dropped its prefix still gives a finite gradient to ALL encoder parameters."""
    out = {"check": "d", "quantizer": quantizer}
    ok = True
    for tag, drop in (("all_dropped", 1.0), ("all_kept", 0.0)):
        model = build_model(style_quantizer=quantizer)
        model.train()
        batch = _fixed_batch(n=8, style_drop=drop, mask=(0.8, 1.0))
        assert bool(batch["style_slot"].any()) == (drop == 0.0)
        model(**batch).loss.backward()
        params = dict(model.style_encoder.named_parameters())
        none = [k for k, p in params.items() if p.grad is None]
        nonfinite = [k for k, p in params.items() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        nonzero = sum(1 for p in params.values() if p.grad is not None and float(p.grad.abs().max()) > 0)
        res = {"n_params": len(params), "grad_none": none, "grad_nonfinite": nonfinite, "n_nonzero_grad": nonzero}
        out[tag] = res
        ok &= not none and not nonfinite
    out["pass"] = bool(ok)
    return out


def check_e_resume(workdir: str, quantizer: str = "fsq") -> dict:
    """(e) Train 6 steps with save_steps=3, then resume from checkpoint-3 with the unmodified OmniTrainer."""
    from safetensors.torch import load_file

    from train_style import run_training

    run1 = os.path.join(workdir, "run1")
    run_training(run1, steps=6, save_steps=3, batch_size=8, logging_steps=3, style_quantizer=quantizer)
    ck3 = os.path.join(run1, "checkpoint-3")
    # a) resume with nothing left to train: the loaded weights are re-saved untouched
    run2 = os.path.join(workdir, "run2")
    _, m2 = run_training(run2, steps=3, save_steps=3, resume_from=ck3, batch_size=8, logging_steps=3,
                         seed=1, style_quantizer=quantizer)
    w1 = load_file(os.path.join(ck3, "model.safetensors"))
    w2 = load_file(os.path.join(run2, "checkpoint-3", "model.safetensors"))
    same = {k: bool(torch.equal(w1[k], w2[k])) for k in w1}
    weights_equal = all(same.values()) and set(w1) == set(w2)
    # b) resume and continue to step 6
    run3 = os.path.join(workdir, "run3")
    t3, _ = run_training(run3, steps=6, save_steps=3, resume_from=ck3, batch_size=8, logging_steps=3,
                         seed=1, style_quantizer=quantizer)
    w6 = load_file(os.path.join(run3, "checkpoint-6", "model.safetensors"))
    moved = any(not torch.equal(w1[k], w6[k]) for k in w1)
    opt_state = os.path.exists(os.path.join(ck3, "optimizer.bin"))
    return {"check": "e", "checkpoint_files": sorted(os.listdir(ck3)), "weights_equal_after_load": weights_equal,
            "n_tensors": len(w1), "continued_to_step": t3.global_step, "weights_moved_after_resume": moved,
            "optimizer_state_saved": opt_state,
            "pass": bool(weights_equal and t3.global_step == 6 and moved and opt_state)}


@torch.no_grad()
def check_g_cfg_and_scale(model, ds) -> dict:
    """(g) 3-branch CFG decoding and prefix scaling run; CFG reduces to upstream's form when g_text == g_style."""
    model.eval()
    text = FakeTextTokenizer()("<|text_start|>abcdef<|text_end|>").input_ids[0]
    code, _ = modal_codes(model, ds, 1)
    vecs = model.style_encoder.vectors_from_codes(torch.tensor(code)).view(len(code), -1)
    g = torch.Generator().manual_seed(0)
    runs, ok = {}, True
    for name, kw in {
        "text_only_cfg": dict(style_vectors=None, guidance_text=2.0, guidance_style=0.0),
        "style_cfg_2_0": dict(style_vectors=vecs, guidance_text=0.0, guidance_style=2.0),
        "full_3branch_2_2": dict(style_vectors=vecs, guidance_text=2.0, guidance_style=2.0),
        "text_1_style_3": dict(style_vectors=vecs, guidance_text=1.0, guidance_style=3.0),
        "scale_0.5": dict(style_vectors=vecs, style_scale=0.5, guidance_text=0.0, guidance_style=0.0),
        "negative_style_guidance": dict(style_vectors=vecs, guidance_text=0.0, guidance_style=-1.0),
    }.items():
        try:
            toks = model.generate_tokens(text, 16, num_step=6, class_temperature=0.5, generator=g, **kw)
            good = tuple(toks.shape) == (8, 16) and bool((toks != model.config.audio_mask_id).all())
        except Exception as e:  # report, never hide
            good, toks = False, None
            runs[name] = {"error": repr(e)}
        ok &= good
        if toks is not None:
            runs[name] = {"ok": good}
    # identity: c + g_t*(t-u) + g_s*(c-t) == c + g*(c-u) at g_t == g_s == g
    tgt = torch.full((8, 16), model.config.audio_mask_id)
    lc = torch.log_softmax(model._branch_logits(text, vecs, tgt, None), -1)
    lt = torch.log_softmax(model._branch_logits(text, None, tgt, None), -1)
    lu = torch.log_softmax(model._branch_logits(None, None, tgt, None), -1)
    gg = 2.0
    three = torch.log_softmax(lc + gg * (lt - lu) + gg * (lc - lt), -1)
    upstream = torch.log_softmax(lc + gg * (lc - lu), -1)
    ident = float((three - upstream).abs().max())
    # prefix scaling: CE at scale 0 / 1 / 2 on the evaluation batches
    ce = {}
    # Qwen3 layers start with RMSNorm, so a positive rescale of the slot vectors is almost invisible to the other
    # positions; the sweep documents where the knob actually bites (near 0).
    for sc in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        model.config.style_scale = sc
        ce[sc] = eval_ce(model, ds, False, n_batches=4)
    model.config.style_scale = 1.0
    branch_diff = float((lc - lt).abs().max())
    return {"check": "g", "runs": runs, "cfg_identity_max_abs_diff": ident, "branch_full_vs_textonly_max_diff": branch_diff,
            "ce_by_prefix_scale": ce, "pass": bool(ok and ident < 1e-4 and branch_diff > 1e-3 and ce[0.0] > ce[1.0])}


def check_kl_guard() -> dict:
    """KL-to-Base: exactly 0 for an untouched model on a no-slot replay batch, > 0 once the model has been trained."""
    model = build_model(style_quantizer="fsq")
    model.attach_reference(StyleOmniVoice.reference_from(model))
    model.configure_training(kl_weight=1.0)
    model.train()
    batch = _fixed_batch(n=8, replay_fraction=1.0, mask=(0.8, 1.0))
    assert bool(batch["kl_mask"].all()) and not bool(batch["style_slot"].any())
    kl0 = float(model(**batch).kl_loss.detach())
    _, _, loader = build_loader(seed=3)
    model.configure_training(kl_weight=0.0)
    train_steps(model, loader, 30, lr=3e-3, log_every=30)
    model.configure_training(kl_weight=1.0)
    kl1 = float(model(**batch).kl_loss.detach())
    n_ref_in_state = sum(1 for k in model.state_dict() if k.startswith("_ref"))
    return {"check": "kl", "kl_untouched": kl0, "kl_after_30_steps": kl1, "reference_keys_in_state_dict": n_ref_in_state,
            "pass": bool(kl0 < 1e-6 and kl1 > 1e-4 and n_ref_in_state == 0)}


def mem_available_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return 0


def check_f_real_model(model_id: str = "k2-fsa/OmniVoice", min_free_factor: float = 4.0) -> dict:
    """(f) Load the real public Base into the subclass, forward only: base weights intact, only encoder keys new."""
    from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path
    from safetensors import safe_open

    free0 = mem_available_mb()
    if free0 < 6000:
        return {"check": "f", "skipped": f"only {free0} MB RAM available", "pass": None}
    path = _resolve_model_path(model_id)  # downloads the public model if it is not cached
    st = os.path.join(path, "model.safetensors")
    size_mb = os.path.getsize(st) // (1024 * 1024)
    if free0 < min_free_factor * size_mb:
        return {"check": "f", "skipped": f"{free0} MB free < {min_free_factor} x {size_mb} MB", "pass": None}
    t0 = time.time()
    model = StyleOmniVoice.from_pretrained(path, train=True, dtype=torch.float32)
    load_s = time.time() - t0
    sd = model.state_dict()
    with safe_open(st, "pt") as f:
        file_keys = list(f.keys())
        diffs = {}
        for k in file_keys:
            if k in sd:
                diffs[k] = float((f.get_tensor(k).float() - sd[k].float()).abs().max())
    not_in_model = [k for k in file_keys if k not in sd]
    missing = sorted(k for k in sd if k not in set(file_keys))
    nonenc_missing = [k for k in missing if not k.startswith("style_encoder.")]
    max_base_diff = max(diffs.values()) if diffs else None
    # forward-only: style-dropped logits of the subclass == the plain OmniVoice's logits
    torch.manual_seed(0)
    C, V = model.config.num_audio_codebook, model.config.audio_vocab_size
    text = torch.randint(0, 1000, (1, 12))
    tgt = torch.randint(0, V - 1, (C, 10))
    ids = torch.cat([text.expand(C, -1), tgt], dim=1).unsqueeze(0)
    am = torch.cat([torch.zeros(12, dtype=torch.bool), torch.ones(10, dtype=torch.bool)]).unsqueeze(0)
    attn = torch.ones(1, 1, 22, 22, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        ours = model(input_ids=ids, audio_mask=am, attention_mask=attn).logits.float()
        # with a prefix slot: runs, finite, norm-matched
        n = model.config.style_n_vectors
        ids2 = torch.cat([text.expand(C, -1), torch.full((C, n), model.config.audio_mask_id), tgt], dim=1).unsqueeze(0)
        am2 = torch.cat([torch.zeros(12, dtype=torch.bool), torch.ones(n + 10, dtype=torch.bool)]).unsqueeze(0)
        slot = torch.zeros(1, 12 + n + 10, dtype=torch.bool)
        slot[0, 12:12 + n] = True
        crop = torch.randint(0, V - 1, (1, C, 20))
        model.ensure_style_scale_calibrated()
        with_prefix = model(input_ids=ids2, audio_mask=am2, attention_mask=torch.ones(1, 1, 22 + n, 22 + n, dtype=torch.bool),
                            style_slot=slot, style_tokens=crop).logits.float()
        calibrated_norm = float(model.style_encoder(crop, torch.ones(1, 20, dtype=torch.bool)).vectors.float().norm(dim=-1).mean())
    peak_mb = None
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        pass
    del model
    plain = OmniVoice.from_pretrained(path, train=True, dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = plain(input_ids=ids, audio_mask=am, attention_mask=attn).logits.float()
    logit_diff = float((ours - theirs).abs().max())
    return {"check": "f", "model_dir": path, "safetensors_mb": size_mb, "ram_free_before_mb": free0, "peak_rss_mb": peak_mb,
            "load_seconds": round(load_s, 1), "n_file_tensors": len(file_keys), "file_keys_not_in_model": not_in_model[:10],
            "keys_missing_from_file": missing[:6] + (["..."] if len(missing) > 6 else []), "n_missing": len(missing),
            "missing_not_encoder": nonenc_missing, "max_abs_base_weight_diff": max_base_diff,
            "logit_diff_vs_plain_omnivoice": logit_diff, "prefix_forward_finite": bool(torch.isfinite(with_prefix).all()),
            "prefix_vector_norm": calibrated_norm,
            "pass": bool(not nonenc_missing and not not_in_model and max_base_diff == 0.0 and logit_diff == 0.0
                         and torch.isfinite(with_prefix).all())}


def collapse_log_sample(rows, k: int = 6):
    keys = ["step", "loss", "style/z_spread", "style/codes_used", "style/code_perplexity", "style/codes_used_per_slot",
            "style/commit_loss", "style/out_rms"]
    pick = rows[:: max(1, len(rows) // k)][: k + 1]
    return [{kk: r.get(kk) for kk in keys} for r in pick]


def run_all(steps: int = 1000, quantizer: str = "fsq", log=print, skip_real: bool = False) -> dict:
    res: dict = {}
    t0 = time.time()
    model, ds, rows, secs = train_reference_model(quantizer, steps=steps, log=None)
    res["train"] = {"quantizer": quantizer, "steps": steps, "seconds": round(secs, 1), "collapse_log": collapse_log_sample(rows)}
    log(f"trained in {secs:.0f}s")
    res["a"] = check_a_prefix_helps(model, ds)
    log(res["a"])
    res["b"] = check_b_decode_bias(model, ds)
    log(res["b"])
    res["c"] = check_c_roundtrip(model)
    log(res["c"])
    res["d"] = check_d_style_dropped_grads(quantizer)
    log(res["d"])
    with tempfile.TemporaryDirectory() as wd:
        res["e"] = check_e_resume(wd, quantizer)
    log(res["e"])
    res["g"] = check_g_cfg_and_scale(model, ds)
    log(res["g"])
    res["kl"] = check_kl_guard()
    log(res["kl"])
    if not skip_real:
        res["f"] = check_f_real_model()
        log(res["f"])
    res["total_seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--quantizer", default="fsq", choices=["fsq", "vq", "none"])
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run_all(args.steps, args.quantizer, skip_real=args.skip_real)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
