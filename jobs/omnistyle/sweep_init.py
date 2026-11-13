"""Seed x output-layer-init sweep for the FSQ style encoder (CPU, synthetic tokens).

Arms (all FSQ [7,5,5], noise 0.1, lr 1e-3, 300 base steps + 1000 prefix steps unless --steps says otherwise):
  OLD      to_out init std 0.02, LayerNorm eps 1e-5   (output norm ~0.48 of target at init; the former default)
  OLD_EPS  to_out init std 0.02, LayerNorm eps 1e-10  (norm matches, small init kept; the current default)
  NEW      to_out fan-in init,   LayerNorm eps 1e-5   (norm matches, unit-variance output)

Per run it logs check (a)/(b) numbers plus saturation metrics of the bottleneck: the fraction of code dimensions
sitting on the outermost FSQ level (a "corner") and the mean tanh derivative 1 - tanh(z)^2.

    nice -n 19 python sweep_init.py --arms OLD OLD_EPS NEW --seeds 0 1 2 --out sweep_init.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_style as S  # noqa: E402

ARMS = {
    "OLD": dict(style_out_init_std=0.02, style_out_ln_eps=1e-5),
    "OLD_EPS": dict(style_out_init_std=0.02, style_out_ln_eps=1e-10),
    "NEW": dict(style_out_init_std=None, style_out_ln_eps=1e-5),
}


@torch.no_grad()
def saturation(model, ds, n: int = 64) -> dict:
    model.eval()
    enc = model.style_encoder
    half = torch.tensor([l // 2 for l in enc.fsq_levels], dtype=torch.float32)
    zs = []
    for k in range(S.N_STYLES):
        crops = ds.clips_of_style(k, n)
        zs.append(enc.encode_z(crops, torch.ones(n, crops.shape[2], dtype=torch.bool)).float())
    z = torch.cat(zs)  # [4n, n_vec, d]
    b = half * torch.tanh(z)
    corner = (torch.round(b).abs() == half).float().mean()
    return {"corner_fraction": round(float(corner), 3), "mean_tanh_grad": round(float((1 - torch.tanh(z) ** 2).mean()), 3),
            "z_abs_mean": round(float(z.abs().mean()), 3)}


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
        **saturation(model, ds),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["OLD", "OLD_EPS", "NEW"], choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    for seed in a.seeds:
        for arm in a.arms:
            r = run(arm, seed, a.steps)
            results.append(r)
            print(json.dumps(r), flush=True)
            if a.out:
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=1)
    print("SWEEPDONE")
