"""Entry script: train StyleOmniVoice with the unmodified upstream `OmniTrainer`.

Nothing in upstream `omnivoice.training` is patched.  This script only supplies the subclass model, the
style processor/collator, and (optionally) the frozen reference for the KL guard.

The shipped quantiser is the continuous bottleneck (`style_quantizer="none"`) at phase-2 lr 5e-4.  A
training-time collapse guard probes the style code every `guard_every` steps: when its between-centroid
separation (or centroid purity) stays collapsed for `guard_patience` probes, the attempt is abandoned and
retrained from scratch with seed + attempt.  Disable it with `--no-collapse-guard`.

Smoke usage (CPU, synthetic tokens, tiny Qwen3):
    nice -n 19 python train_style.py --synthetic --output-dir /tmp/style-run --steps 200

Real-model usage needs a tokenised dataset in the processor's raw-sample format (audio_tokens, style_tokens,
label.text, label.replay); loading it is not part of this smoke job.  `--init-from k2-fsa/OmniVoice` loads
the public Base weights into the subclass; the encoder keys are the only new (randomly initialised) ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnivoice.training.config import TrainingConfig  # noqa: E402
from omnivoice.training.trainer import OmniTrainer  # noqa: E402

import smoke_style as S  # noqa: E402
from style_omnivoice import StyleOmniVoice  # noqa: E402


class CollapseDetected(RuntimeError):
    """The training-time collapse guard fired: the style code stopped separating the styles.

    Carries the full guard log (one row per probe) in `.guard_log`.
    """

    def __init__(self, message: str, guard_log: list):
        super().__init__(message)
        self.guard_log = guard_log


@torch.no_grad()
def collapse_probe(model: StyleOmniVoice, ds, n: int = 32) -> dict:
    """Label-aware probe of the style code: within-style spread vs between-centroid separation.

    Uses fresh deterministic crops per style (`ds.clips_of_style`), so it does not disturb the training RNG.
    The model's train/eval mode is restored afterwards.
    """
    was_training = model.training
    try:
        return S.bottleneck_stats(model, ds, n=n)
    finally:
        model.train(was_training)


def _clear_partial_run(output_dir: str):
    """Remove the checkpoints and metrics of an aborted attempt before restarting (only this run's files)."""
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if os.path.isdir(path):
            shutil.rmtree(path)
    metrics = os.path.join(output_dir, "style_metrics.jsonl")
    if os.path.exists(metrics):
        os.remove(metrics)


def _run_training_once(
    output_dir: str,
    steps: int,
    save_steps: int,
    resume_from: str | None,
    init_from: str | None,
    batch_size: int,
    lr: float,
    kl_weight: float,
    replay_fraction: float,
    seed: int,
    keep_last_n: int,
    logging_steps: int,
    model: StyleOmniVoice | None,
    model_kwargs: dict,
    guard: bool,
    guard_warmup: int,
    guard_every: int,
    guard_patience: int,
    guard_min_sep: float,
    inject_collapse_at: int | None,
    attempt: int,
    guard_log: list,
):
    """One training attempt.  Raises `CollapseDetected` when the guard fires."""
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
    guard_path = os.path.join(output_dir, "collapse_guard.jsonl")
    counter = {"n": 0, "strikes": 0}
    can_probe = guard and hasattr(ds, "clips_of_style")

    def guard_check(step: int):
        stats = collapse_probe(model, ds)
        collapsed = stats["sep_ratio"] < guard_min_sep
        counter["strikes"] = counter["strikes"] + 1 if collapsed else 0
        row = {"attempt": attempt, "seed": seed, "step": step, **stats,
               "collapsed": collapsed, "strikes": counter["strikes"]}
        guard_log.append(row)
        with open(guard_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if counter["strikes"] >= guard_patience:
            raise CollapseDetected(
                f"style code collapsed at step {step} (attempt {attempt}, seed {seed}): "
                f"sep_ratio={stats['sep_ratio']}, purity={stats['centroid_purity']}",
                guard_log,
            )

    def sink(row):
        counter["n"] += 1
        if counter["n"] % logging_steps == 0 or counter["n"] == 1:
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"micro_step": counter["n"], **row}) + "\n")
        if inject_collapse_at is not None and attempt == 0 and counter["n"] >= inject_collapse_at:
            # Injection test hook: force the exact collapse mode seen on seed 4 (a style-blind code).  The
            # layer is frozen after zeroing so the straight-through gradient through the layer norm cannot
            # nudge it back before the next probe.
            with torch.no_grad():
                model.style_encoder.to_code.weight.zero_()
                if model.style_encoder.to_code.bias is not None:
                    model.style_encoder.to_code.bias.zero_()
            model.style_encoder.to_code.weight.requires_grad_(False)
            if model.style_encoder.to_code.bias is not None:
                model.style_encoder.to_code.bias.requires_grad_(False)
        if can_probe and counter["n"] >= guard_warmup and counter["n"] % guard_every == 0:
            guard_check(counter["n"])

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


def run_training(
    output_dir: str,
    steps: int = 200,
    save_steps: int = 100,
    resume_from: str | None = None,
    init_from: str | None = None,
    batch_size: int = 16,
    lr: float = 5e-4,
    kl_weight: float = 0.0,
    replay_fraction: float = 0.0,
    seed: int = 0,
    keep_last_n: int = -1,
    logging_steps: int = 25,
    model: StyleOmniVoice | None = None,
    guard: bool = True,
    guard_warmup: int = 200,
    guard_every: int = 100,
    guard_patience: int = 2,
    guard_attempts: int = 3,
    guard_min_sep: float = 0.5,
    inject_collapse_at: int | None = None,
    **model_kwargs,
):
    """Build everything and run `OmniTrainer.train()` with the training-time collapse guard.

    The guard probes the style code every `guard_every` steps after a `guard_warmup` warm start: if
    sep_ratio = between-centroid distance / within-style spread stays below `guard_min_sep` for
    `guard_patience` consecutive probes, the attempt is abandoned and retrained from scratch with the
    deterministic new seed `seed + attempt`.  (Final-measurement sep_ratio on seed 4 was 0.01-0.19 for the
    collapsed lr-1e-3 arms and 2.34-5.49 for the non-collapsing runs, so 0.5 sits between the two modes;
    see `workers/smoke/contin-fix-report.md`.)  Returns (trainer, model) of the successful attempt; the
    trainer also carries `.guard_log` and `.guard_attempts`.

    `inject_collapse_at` (micro-step, attempt 0 only) zeroes and freezes the encoder's `to_code` layer from
    that step on, forcing the exact collapse mode seen on seed 4 (a style-blind code); it exists for the
    guard's injection test.  Resuming from a checkpoint disables retries (the same checkpoint would
    reproduce the same run), so a collapse there raises.
    """
    attempts = guard_attempts if (guard and resume_from is None and model is None) else 1
    guard_log: list = []
    os.makedirs(output_dir, exist_ok=True)
    guard_path = os.path.join(output_dir, "collapse_guard.jsonl")
    if os.path.exists(guard_path):
        os.remove(guard_path)  # this call's log, not a previous call's
    last: CollapseDetected | None = None
    for attempt in range(attempts):
        try:
            trainer, trained = _run_training_once(
                output_dir, steps, save_steps, resume_from, init_from, batch_size, lr, kl_weight,
                replay_fraction, seed + attempt, keep_last_n, logging_steps, model, model_kwargs,
                guard, guard_warmup, guard_every, guard_patience, guard_min_sep,
                inject_collapse_at, attempt, guard_log,
            )
            trainer.guard_log = guard_log
            trainer.guard_attempts = attempt + 1
            return trainer, trained
        except CollapseDetected as e:
            last = e
            if attempt + 1 < attempts:
                _clear_partial_run(output_dir)
    raise last


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
    ap.add_argument("--lr", type=float, default=5e-4, help="phase-2 lr; 1e-3 collapses the continuous code on seed 4")
    ap.add_argument("--quantizer", default="none", choices=["none", "fsq", "vq"])
    ap.add_argument("--fsq-levels", type=int, nargs="+", default=None, help="odd levels per FSQ dimension")
    ap.add_argument("--fsq-bound", default=None, choices=["tanh", "clip", "softsign"])
    ap.add_argument("--no-collapse-guard", action="store_true", help="disable the training-time collapse guard")
    ap.add_argument("--guard-attempts", type=int, default=3, help="max attempts; each restarts with seed + attempt")
    ap.add_argument("--guard-warmup", type=int, default=200)
    ap.add_argument("--guard-every", type=int, default=100)
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
        replay_fraction=a.replay_fraction, style_quantizer=a.quantizer, guard=not a.no_collapse_guard,
        guard_attempts=a.guard_attempts, guard_warmup=a.guard_warmup, guard_every=a.guard_every, **kwargs,
    )
