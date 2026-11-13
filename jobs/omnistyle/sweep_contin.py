"""Continuous-bottleneck (`style_quantizer="none"`) collapse fix sweep and gate table.

At the old phase-2 learning rate (1e-3) seed 4 does not latch onto style: its encoder drifts to a code
that is clip-sensitive but style-blind, check (a) fails and every style decodes to the same distribution
(`workers/smoke/fallback_none.json`).  This sweep measures the suggested fix arms (phase-2 lr, noise,
warm-start length, output init) on that seed, then confirms the shipped default on seeds 0-4.

    nice -n 19 python sweep_contin.py --arms BASE NOISE_02 NOISE_03 PRE_600 PRE_900 OUT_FANIN \
        --seeds 4 --out .../workers/smoke/contin_arms.json
    nice -n 19 python sweep_contin.py --arms DEFAULT --seeds 0 1 2 3 4 \
        --out .../workers/smoke/contin_default.json

Every run reports the gate set from the handoff: check (a) margin > 0.3, leave-one-out centroid purity
>= 0.9, sep_ratio = between_min / within_rms (reported; > 1 required for a non-merged code), text-probe
accuracy near chance (|acc - chance| <= 0.05) and the continuous check (b) (centroid-decode own-share
>= 0.5 and > 2x the best other style).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_style as S  # noqa: E402
import sweep_fsq as W  # noqa: E402

# Phase 2 (prefix) training defaults: lr 1e-3 is the old control that collapses on seed 4.
BASE_LR = 1e-3
DEFAULT_LR = 5e-4

ARMS = {
    # shipped defaults: continuous code, noise 0.1, 300 base + 1000 prefix steps, phase-2 lr 5e-4
    "DEFAULT": dict(),
    # the old phase-2 lr, kept as the collapse control; the arms below are the old rate + one change
    "BASE": dict(lr=BASE_LR),
    "LR_7E4": dict(lr=7e-4),
    # more training noise
    "NOISE_02": dict(lr=BASE_LR, style_noise_std=0.2),
    "NOISE_03": dict(lr=BASE_LR, style_noise_std=0.3),
    # longer base (warm-start) phase; also changes the encoder's post-reset init draw
    "PRE_600": dict(lr=BASE_LR, pre_steps=600),
    "PRE_900": dict(lr=BASE_LR, pre_steps=900),
    # output-MLP init (fan-in instead of a fixed std 0.02)
    "OUT_FANIN": dict(lr=BASE_LR, style_out_init_std=None),
}


def gates(r: dict) -> dict:
    """Gate booleans for one continuous-bottleneck run."""
    g = {
        "purity": r["centroid_purity"] >= 0.9,
        "no_collapse": r["sep_ratio"] > 1.0,
        "a_margin": (r["a_shuffled"] - r["a_true"]) > 0.3,
        "text_probe": abs(r["text_probe_acc"] - r["text_probe_chance"]) <= 0.05,
        "vec_decode": r["vec_b_pass"],
    }
    g["all_pass"] = all(g.values())
    return g


def run_contin(seed: int, steps: int = 1000, pre_steps: int = 300, lr: float = DEFAULT_LR, **kw) -> dict:
    """One continuous-bottleneck run with the full gate set (same measurements as the fallback table)."""
    t0 = time.time()
    model, ds, rows, secs = S.train_reference_model(
        "none", pre_steps=pre_steps, steps=steps, lr=lr, seed=seed, **kw)
    a = S.check_a_prefix_helps(model, ds)
    b = S.check_b_continuous(model, ds)
    r = {
        "mode": "none", "seed": seed, "steps": steps, "pre_steps": pre_steps,
        "lr": lr, "overrides": dict(kw),
        "secs": round(time.time() - t0),
        "a_true": round(a["ce_true"], 3), "a_shuffled": round(a["ce_shuffled"], 3), "a_pass": a["pass"],
        "vec_b_shares": b["vec_b_shares"], "vec_b_pass": b["pass"],
        **S.bottleneck_stats(model, ds),
        **W.text_probe(model, ds),
    }
    r.update(gates(r))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["DEFAULT"], choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    for seed in a.seeds:
        for arm in a.arms:
            kw = dict(ARMS[arm])
            if "pre_steps" in kw:
                kw["pre_steps"] = int(kw["pre_steps"])
            else:
                kw["pre_steps"] = 300
            r = run_contin(seed, steps=a.steps, **kw)
            r["arm"] = arm
            results.append(r)
            print(json.dumps(r), flush=True)
            if a.out:
                with open(a.out, "w") as f:
                    json.dump(results, f, indent=1)
    print("SWEEPDONE")
