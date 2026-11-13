"""FSQ bound x levels x init sweep for the style encoder (CPU, synthetic tokens).

The old FSQ [7,5,5] + tanh bound passed check (b) on only 2 of 5 seeds (0 and 3): the bound saturated,
most code dimensions sat on the outermost level, and on the other seeds every style merged onto one code.
This sweep searches the approved fix space: more levels/dims, a softer bound ("clip" or "softsign") and a
small pre-quantiser init.  Each run reports check (a), check (b) (modal code, own-share, distinct codes) and
the state of the bottleneck (corner fraction, |z|, bound slope).

    nice -n 19 python sweep_fsq.py --arms DEFAULT --seeds 0 1 2 3 4 --out sweep_fsq_default.json
    nice -n 19 python sweep_fsq.py --mode none --seeds 0 1 2 3 4 --out fallback_none.json

`--mode none` is the fallback measurement: a continuous bottleneck + noise (no quantiser), checked for
tightness (within-style spread vs centroid separation, centroid purity) and leakage (shuffled-prefix CE
margin and a text-line probe on z).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_style as S  # noqa: E402

BIG = [7, 7, 5, 5]
ARMS = {
    # historical default, kept as the control
    "BASE": dict(style_fsq_levels=[7, 5, 5], style_fsq_bound="tanh", style_code_init_std=None),
    # shipped default: no overrides, i.e. the values in StyleOmniVoiceConfig
    "DEFAULT": dict(),
    # more levels/dims only
    "TANH_BIG": dict(style_fsq_levels=BIG, style_fsq_bound="tanh", style_code_init_std=None),
    # approved fix space: more dims + softer bound (+ optional small init / z scale)
    "CLIP_BIG": dict(style_fsq_levels=BIG, style_fsq_bound="clip", style_fsq_scale=1.0,
                     style_code_init_std=None),
    "CLIP_BIG_SMALL_INIT": dict(style_fsq_levels=BIG, style_fsq_bound="clip", style_fsq_scale=1.0,
                                style_code_init_std=0.02),
    "CLIP_BIG_SCALED": dict(style_fsq_levels=BIG, style_fsq_bound="clip", style_fsq_scale=0.25,
                            style_code_init_std=None),
    "CLIP_BIG_SCALED_SMALL_INIT": dict(style_fsq_levels=BIG, style_fsq_bound="clip", style_fsq_scale=0.25,
                                       style_code_init_std=0.02),
    "SOFTSIGN_BIG": dict(style_fsq_levels=BIG, style_fsq_bound="softsign", style_fsq_scale=1.0,
                         style_code_init_std=None),
    "SOFTSIGN_BIG_SMALL_INIT": dict(style_fsq_levels=BIG, style_fsq_bound="softsign", style_fsq_scale=1.0,
                                    style_code_init_std=0.02),
}


@torch.no_grad()
def bound_stats(model, ds, n: int = 64) -> dict:
    """Corner fraction and bound slope of the pre-quantisation code, measured on fresh style crops."""
    model.eval()
    enc = model.style_encoder
    half = torch.tensor([l // 2 for l in enc.fsq_levels], dtype=torch.float32)
    zs = []
    for k in range(S.N_STYLES):
        crops = ds.clips_of_style(k, n)
        zs.append(enc.encode_z(crops, torch.ones(n, crops.shape[2], dtype=torch.bool)).float())
    z = torch.cat(zs)  # [4n, n_vec, d]
    b = enc._bounded(z)
    u = z * enc.fsq_scale
    if enc.fsq_bound == "tanh":
        slope = (1 - torch.tanh(u) ** 2) * enc.fsq_scale
    elif enc.fsq_bound == "softsign":
        slope = enc.fsq_scale / (1 + u.abs()) ** 2
    else:  # value slope of the clip; the backward pass uses the straight-through slope (= fsq_scale) everywhere
        slope = (u.abs() <= 1).float() * enc.fsq_scale
    return {
        "corner_fraction": round(float((torch.round(b).abs() == half).float().mean()), 3),
        "bound_slope_mean": round(float(slope.mean()), 3),
        "frac_outside_linear": round(float((u.abs() > 1).float().mean()), 3),
        "z_abs_mean": round(float(z.abs().mean()), 3),
    }


def run(arm: str, seed: int, steps: int) -> dict:
    model, ds, rows, secs = S.train_reference_model("fsq", steps=steps, seed=seed, **ARMS[arm])
    a = S.check_a_prefix_helps(model, ds)
    b = S.check_b_decode_bias(model, ds)
    return {
        "arm": arm, "seed": seed, "steps": steps, "secs": round(secs),
        "a_true": round(a["ce_true"], 3), "a_shuffled": round(a["ce_shuffled"], 3), "a_pass": a["pass"],
        "b_codes": [p["code"][0] for p in b["per_style"]],
        "b_own_share": [p["shares"][i] for i, p in enumerate(b["per_style"])],
        "b_distinct": b["distinct_codes"], "b_pass": b["pass"],
        **bound_stats(model, ds),
    }


# ---------------------------------------------------------------------------
# Fallback: continuous bottleneck + noise (style_quantizer="none")
# ---------------------------------------------------------------------------

N_BOTTLENECK_CLIPS = 64
TEXT_PROBE_SAMPLES = 800
TEXT_PROBE_STEPS = 300


@torch.no_grad()
def style_centroids_none(model, ds, n: int = N_BOTTLENECK_CLIPS) -> torch.Tensor:
    """Per-style centroids of the continuous code z, shape [K, n_vec, d]."""
    model.eval()
    enc = model.style_encoder
    mus = []
    for k in range(S.N_STYLES):
        crops = ds.clips_of_style(k, n)
        z = enc.encode_z(crops, torch.ones(n, crops.shape[2], dtype=torch.bool)).float()
        mus.append(z.mean(0))
    return torch.stack(mus)


@torch.no_grad()
def bottleneck_stats(model, ds, n: int = N_BOTTLENECK_CLIPS) -> dict:
    """Tightness of the continuous code: within-style spread vs centroid separation, centroid purity.

    Spread is the RMS distance of a sample to its own style centroid; separation is the distance between
    style centroids (min and mean over pairs).  Purity assigns every sample to its nearest centroid; the
    leave-one-out variant rebuilds the own centroid without the sample so it is not optimistically biased.
    """
    model.eval()
    enc = model.style_encoder
    V = []
    for k in range(S.N_STYLES):
        crops = ds.clips_of_style(k, n)
        z = enc.encode_z(crops, torch.ones(n, crops.shape[2], dtype=torch.bool)).float()
        V.append(z.reshape(n, -1))
    V = torch.stack(V)  # [K, n, D]
    K, _, D = V.shape
    mu = V.mean(1)  # [K, D]
    within = (V - mu[:, None]).pow(2).sum(-1).mean().sqrt()
    dist = torch.cdist(mu, mu)
    off = dist[~torch.eye(K, dtype=torch.bool)]
    pts, ids = V.reshape(K * n, D), torch.arange(K * n)
    own = ids // n
    d_plain = torch.cdist(pts, mu)
    d_loo = d_plain.clone()
    d_loo[ids, own] = (V - (V.sum(1, keepdim=True) - V) / (n - 1)).reshape(K * n, D).norm(dim=-1)
    return {
        "within_rms": round(float(within), 3),
        "between_min": round(float(off.min()), 3),
        "between_mean": round(float(off.mean()), 3),
        "sep_ratio": round(float(off.min() / within), 2),
        "centroid_purity": round(float((d_loo.argmin(-1) == own).float().mean()), 3),
        "centroid_purity_plain": round(float((d_plain.argmin(-1) == own).float().mean()), 3),
    }


def text_probe(model, ds, n: int = TEXT_PROBE_SAMPLES, seed: int = 0) -> dict:
    """Linear probe from the continuous code z to the sample's text line.

    The text is drawn independently of the style, so a probe at chance means z carries no text-line
    information (the style crop must not leak it).  Trains only the probe's weights on half the samples
    and reports held-out accuracy against the chance and majority baselines.
    """
    model.eval()
    enc = model.style_encoder
    crops, texts = [], []
    for i in range(n):
        s = ds[i]
        crops.append(s["style_tokens"])
        texts.append(s["label"]["text"])
    crops = torch.stack(crops)
    with torch.no_grad():
        z = enc.encode_z(crops, torch.ones(n, crops.shape[2], dtype=torch.bool)).float().reshape(n, -1)
    classes = sorted(set(texts))
    y = torch.tensor([classes.index(t) for t in texts])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[: n // 2], perm[n // 2:]
    torch.manual_seed(seed)
    W = torch.zeros(z.shape[1], len(classes), requires_grad=True)
    b = torch.zeros(len(classes), requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(TEXT_PROBE_STEPS):
        opt.zero_grad()
        F.cross_entropy(z[tr] @ W + b, y[tr]).backward()
        opt.step()
    with torch.no_grad():
        acc = float(((z[te] @ W + b).argmax(-1) == y[te]).float().mean())
        majority = float((y[te] == torch.bincount(y[tr], minlength=len(classes)).argmax()).float().mean())
    return {"text_probe_acc": round(acc, 3), "text_probe_majority": round(majority, 3),
            "text_probe_chance": round(1 / len(classes), 3), "text_probe_n_test": len(te)}


@torch.no_grad()
def decode_bias_from_vectors(model, vecs, n_clips: int = 24, target_len: int = 16, seed: int = 0) -> list:
    """Share of decoded tokens in each style's preferred blocks for one style's prefix vectors [n, hidden].

    The continuous analogue of `check_b_decode_bias`, which decodes from FSQ code indices.
    """
    g = torch.Generator().manual_seed(seed)
    text = S.FakeTextTokenizer()("<|text_start|>abcdef<|text_end|>").input_ids[0]
    counts = torch.zeros(S.N_STYLES)
    total = 0
    for _ in range(n_clips):
        toks = model.generate_tokens(
            text, target_len, style_vectors=vecs, num_step=8, guidance_text=0.0, guidance_style=0.0,
            class_temperature=0.5, generator=g,
        )
        for s in range(S.N_STYLES):
            for cb in (0, 1):
                blk = S.style_block(s, cb, S.N_STYLES)
                counts[s] += ((toks[cb] >= blk.start) & (toks[cb] < blk.stop)).sum()
        total += 2 * toks.shape[1]
    return (counts / total).tolist()


def vec_decode_pass(shares: list) -> bool:
    """Same per-style criterion as check (b), applied to the continuous decode-bias shares."""
    return all(
        max(range(S.N_STYLES), key=lambda j: sh[j]) == k
        and sh[k] >= 0.5
        and sh[k] > 2 * max(sh[j] for j in range(S.N_STYLES) if j != k)
        for k, sh in enumerate(shares)
    )


def run_none(seed: int, steps: int) -> dict:
    """Continuous bottleneck + noise: tightness and leakage checks (fallback when the FSQ fix is fragile)."""
    model, ds, rows, secs = S.train_reference_model("none", steps=steps, seed=seed)
    a = S.check_a_prefix_helps(model, ds)
    enc = model.style_encoder
    vecs = enc.vectors_from_z(style_centroids_none(model, ds))  # [K, n_vec, hidden]
    shares = [decode_bias_from_vectors(model, vecs[k]) for k in range(S.N_STYLES)]
    return {
        "mode": "none", "seed": seed, "steps": steps, "secs": round(secs),
        "a_true": round(a["ce_true"], 3), "a_shuffled": round(a["ce_shuffled"], 3), "a_pass": a["pass"],
        "vec_b_shares": [[round(x, 3) for x in sh] for sh in shares],
        "vec_b_pass": bool(vec_decode_pass(shares)),
        **bottleneck_stats(model, ds),
        **text_probe(model, ds),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fsq", "none"], default="fsq",
                    help="fsq: bound/levels/init sweep; none: continuous-bottleneck fallback checks")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    if a.mode == "none":
        jobs = [(None, seed) for seed in a.seeds]
    else:
        jobs = [(arm, seed) for seed in a.seeds for arm in a.arms]
    for arm, seed in jobs:
        r = run_none(seed, a.steps) if arm is None else run(arm, seed, a.steps)
        results.append(r)
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(results, f, indent=1)
    print("SWEEPDONE")
