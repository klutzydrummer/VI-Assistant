"""Measure the cost of the KL-to-Base guard so the rental plan can extrapolate to the 0.6B model.

Driver mode (default) runs each (config, kl on/off) in its own subprocess so that peak RSS is per run:
    nice -n 19 python bench_kl.py --out bench_kl.json
Worker mode (used by the driver):
    nice -n 19 python bench_kl.py --worker mid --kl 1

CPU only, synthetic tokens.  Configs: tiny (hidden 64, 2 layers) and mid (hidden 256, 6 layers).
The 0.6B figures are an extrapolation (parameter counts from the real Qwen3-0.6B shape on the meta device;
memory and time scale from the measured ratios) and are labelled as such in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIGS = {
    "tiny": dict(hidden=64, layers=2),
    "mid": dict(hidden=256, layers=6),
}


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def worker(name: str, kl: bool, steps: int = 20, warmup: int = 3, batch_size: int = 8) -> dict:
    """One process per (config, kl) for peak RSS, plus INTERLEAVED step timing: every iteration times one step
    with the KL term off and one with it on, on the same batch, so machine load drifts cancel in the ratio."""
    import smoke_style as S
    from style_omnivoice import StyleOmniVoice

    torch.set_num_threads(2)
    cfg = CONFIGS[name]
    torch.manual_seed(0)
    model = S.StyleOmniVoice(S.tiny_config(llm_config=S.tiny_llm_config(**cfg)))
    model.ensure_style_scale_calibrated()
    ref = None
    if kl:
        ref = StyleOmniVoice.reference_from(model)
        model.attach_reference(ref)
    model.configure_training(kl_weight=1.0 if kl else 0.0)
    _, _, loader = S.build_loader(batch_size=batch_size, replay_fraction=0.5)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(loader)

    def one_step(batch, with_kl):
        model.kl_weight = 1.0 if (with_kl and kl) else 0.0
        t0 = time.perf_counter()
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        return time.perf_counter() - t0

    off, on = [], []
    for i in range(steps + warmup):
        batch = next(it)
        a = one_step(batch, False)
        b = one_step(batch, True)
        if i >= warmup:
            off.append(a)
            on.append(b)
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    ratios = [b / a for a, b in zip(off, on)]
    return {
        "config": name,
        "kl": kl,
        "params_model": n_params(model),
        "params_encoder": n_params(model.style_encoder),
        "params_reference": n_params(ref) if ref is not None else 0,
        "step_s_off_median": med(off),
        "step_s_on_median": med(on),
        "step_s_off_min": min(off),
        "step_s_on_min": min(on),
        "pair_ratio_median": med(ratios),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
    }


def params_0p6b() -> dict:
    """Parameter counts for the real Qwen3-0.6B-shaped StyleOmniVoice, built on the meta device (no memory)."""
    from transformers import Qwen3Config

    import smoke_style as S

    llm = Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, max_position_embeddings=40960, tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    with torch.device("meta"):
        m = S.StyleOmniVoice(
            S.StyleOmniVoiceConfig(
                llm_config=llm, audio_codebook_weights=[8, 8, 6, 6, 4, 4, 2, 2], style_n_vectors=4, style_enc_dim=256,
                style_enc_layers=2, style_enc_heads=4, style_quantizer="fsq", style_fsq_levels=[7, 5, 5, 5],
            )
        )
    total = n_params(m)
    enc = n_params(m.style_encoder)
    return {"params_total": total, "params_encoder": enc, "params_reference": total - enc}


def driver(out: str | None, repeats: int = 1):
    rows = []
    for name in CONFIGS:
        for kl in (0, 1):
            cmd = ["nice", "-n", "19", sys.executable, os.path.abspath(__file__), "--worker", name, "--kl", str(kl)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise SystemExit(f"worker failed: {' '.join(cmd)}\n{res.stderr[-2000:]}")
            row = json.loads(res.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(row, flush=True)
    summary = {}
    for name in CONFIGS:
        a = next(r for r in rows if r["config"] == name and not r["kl"])
        b = next(r for r in rows if r["config"] == name and r["kl"])
        summary[name] = {
            "params_model": a["params_model"],
            "params_reference_extra": b["params_reference"],
            "peak_rss_mb_no_kl": a["peak_rss_mb"],
            "peak_rss_mb_kl": b["peak_rss_mb"],
            "peak_rss_extra_mb": b["peak_rss_mb"] - a["peak_rss_mb"],
            "step_s_kl_off_median": round(b["step_s_off_median"], 4),
            "step_s_kl_on_median": round(b["step_s_on_median"], 4),
            "step_s_kl_off_min": round(b["step_s_off_min"], 4),
            "step_s_kl_on_min": round(b["step_s_on_min"], 4),
            "step_time_ratio_median_of_pairs": round(b["pair_ratio_median"], 3),
            "step_time_ratio_of_mins": round(b["step_s_on_min"] / b["step_s_off_min"], 3),
        }
    big = params_0p6b()
    # Extrapolation (labelled): the frozen copy costs 4 bytes/param in fp32 (2 in bf16) next to ~16 bytes/param for
    # weights+grads+Adam; its forward is ~1/3 of a train step (forward is 1/3 of forward+backward FLOPs).
    ref_params = big["params_reference"]
    summary["extrapolation_0p6b"] = {
        **big,
        "extra_gb_ref_fp32": round(ref_params * 4 / 1e9, 2),
        "extra_gb_ref_bf16": round(ref_params * 2 / 1e9, 2),
        "train_gb_weights_grads_adam_fp32": round(big["params_total"] * 16 / 1e9, 2),
        "extra_memory_share_fp32": round(ref_params * 4 / (big["params_total"] * 16), 3),
        "extra_step_time_expected": "about +33% of a training step (one extra no-grad forward over the same tokens); "
        "the measured CPU ratios above are the evidence for tiny/mid",
        "label": "extrapolation, not measured at 0.6B",
    }
    print(json.dumps(summary, indent=1))
    if out:
        with open(out, "w") as f:
            json.dump({"rows": rows, "summary": summary}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, choices=list(CONFIGS))
    ap.add_argument("--kl", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.worker:
        print(json.dumps(worker(a.worker, bool(a.kl))))
    else:
        driver(a.out)
