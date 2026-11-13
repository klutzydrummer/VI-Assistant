"""Entry script: train StyleOmniVoice with the unmodified upstream `OmniTrainer`.

Nothing in upstream `omnivoice.training` is patched.  This script only supplies the subclass model, the
style processor/collator, and (optionally) the frozen reference for the KL guard.

Smoke usage (CPU, synthetic tokens, tiny Qwen3):
    nice -n 19 python train_style.py --synthetic --output-dir /tmp/style-run --steps 200

Real-model usage needs a tokenised dataset in the processor's raw-sample format (audio_tokens, style_tokens,
label.text, label.replay); loading it is not part of this smoke job.  `--init-from k2-fsa/OmniVoice` loads
the public Base weights into the subclass; the encoder keys are the only new (randomly initialised) ones.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnivoice.training.config import TrainingConfig  # noqa: E402
from omnivoice.training.trainer import OmniTrainer  # noqa: E402

import smoke_style as S  # noqa: E402
from style_omnivoice import StyleOmniVoice  # noqa: E402


def run_training(
    output_dir: str,
    steps: int = 200,
    save_steps: int = 100,
    resume_from: str | None = None,
    init_from: str | None = None,
    batch_size: int = 16,
    lr: float = 1e-3,
    kl_weight: float = 0.0,
    replay_fraction: float = 0.0,
    seed: int = 0,
    keep_last_n: int = -1,
    logging_steps: int = 25,
    model: StyleOmniVoice | None = None,
    **model_kwargs,
):
    """Build everything and run `OmniTrainer.train()`.  Returns (trainer, model)."""
    if model is None:
        if init_from:
            model = StyleOmniVoice.from_pretrained(init_from, train=True)
            model.ensure_style_scale_calibrated()
        else:
            model = S.build_model(seed=seed, **model_kwargs)
    ds, proc, loader = S.build_loader(batch_size=batch_size, seed=seed, replay_fraction=replay_fraction)

    # Frozen reference for the KL guard: a plain OmniVoice snapshot of the starting base weights.  It is not a
    # registered submodule, so it never enters state_dict, DDP, optimizer or checkpoints.
    if kl_weight > 0:
        model.attach_reference(StyleOmniVoice.reference_from(model))

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "style_metrics.jsonl")
    counter = {"n": 0}

    def sink(row):
        counter["n"] += 1
        if counter["n"] % logging_steps == 0 or counter["n"] == 1:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"micro_step": counter["n"], **row}) + "\n")

    model.configure_training(kl_weight=kl_weight, metrics_sink=sink)

    cfg = TrainingConfig(
        output_dir=output_dir,
        steps=steps,
        seed=seed,
        learning_rate=lr,
        weight_decay=0.0,
        warmup_type="steps",
        warmup_steps=0,
        lr_scheduler_type="constant",
        mixed_precision="no",
        num_workers=0,
        attn_implementation="sdpa",
        logging_steps=logging_steps,
        eval_steps=10**9,
        save_steps=save_steps,
        keep_last_n_checkpoints=keep_last_n,
        resume_from_checkpoint=resume_from,
        batch_tokens=0,
    )
    trainer = OmniTrainer(model=model, config=cfg, train_dataloader=loader, tokenizer=proc.text_tokenizer)
    trainer.train()
    return trainer, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="synthetic tokens + tiny model (the only supported data)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--kl-weight", type=float, default=0.0)
    ap.add_argument("--replay-fraction", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--quantizer", default="fsq", choices=["fsq", "vq", "none"])
    ap.add_argument("--fsq-levels", type=int, nargs="+", default=None, help="odd levels per FSQ dimension")
    ap.add_argument("--fsq-bound", default=None, choices=["tanh", "clip", "softsign"])
    a = ap.parse_args()
    if not a.synthetic:
        sys.exit("only --synthetic data is wired in this smoke job")
    kwargs = {}
    if a.fsq_levels is not None:
        kwargs["style_fsq_levels"] = a.fsq_levels
    if a.fsq_bound is not None:
        kwargs["style_fsq_bound"] = a.fsq_bound
    run_training(
        a.output_dir, a.steps, a.save_steps, a.resume_from, a.init_from, lr=a.lr, kl_weight=a.kl_weight,
        replay_fraction=a.replay_fraction, style_quantizer=a.quantizer, **kwargs,
    )
