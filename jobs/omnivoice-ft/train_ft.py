"""OmniVoice full fine-tune with the learned style prefix: entry script.

Reuses the validated ``jobs/omnistyle`` model/processor/collator and the *unmodified* upstream
``OmniTrainer``.  This script only supplies:

* the subclass model (``StyleOmniVoice.from_pretrained("k2-fsa/OmniVoice", train=True)``),
* our own optimizer, so the style encoder can use a higher learning rate than the backbone
  (conditioning.md patch item 5),
* the style data pipeline (``ft_data``), and
* the training-time collapse guard and a per-checkpoint checksum contract (``SHA256SUMS`` plus
  ``ft_config.json`` in every ``checkpoint-N/`` directory).

Shipped defaults (parent's decisions, do not change):
  * phase-2 lr 5e-4; encoder lr is a separate, higher knob.  The default is 1e-3 (2x): on the
    tiny continuous-bottleneck probe, encoder lr 5e-3 collapsed (sep_ratio 0.24 at step 50) while
    1e-3 stayed healthy (sep_ratio 0.91 -> 1.82 over 150 steps) and 5e-4 reached 1.02.
  * continuous bottleneck (``style_quantizer="none"``),
  * KL-to-Base guard on public replay rows (``--kl-weight 1.0``; rows marked ``replay: true``
    by ``data_manifest.py``),
  * collapse guard ON: probe every ``guard_every`` micro-steps after ``guard_warmup``; when the
    style code stays collapsed for ``guard_patience`` probes, abandon the attempt and retrain
    from scratch with seed + attempt, up to ``guard_attempts`` attempts,
  * ``mask_ratio_range = [0.5, 1.0]`` (biased upward so the prefix is not ignored),
  * ``prompt_ratio = 0``: no real audio prompt, the learned prefix occupies the prompt slot,
  * ``batch_tokens = 8192`` per GPU, bf16, TF32, weight decay 0.01, max grad norm 1.0,
    drop_cond_ratio 0.1, steps 5000, save every 500, keep all checkpoints.

Real run (needs the tokenised shards first; see README):
    accelerate launch --num_processes 1 train_ft.py \
        --data-config /path/to/data_config.json --output-dir /path/to/run

Free CPU smoke (tiny synthetic model, no data, no downloads):
    nice -n 19 python train_ft.py --synthetic --steps 20 --save-steps 10 --output-dir /tmp/ft-smoke

Checkpoint verification:
    python train_ft.py --verify-checkpoint /path/to/run/checkpoint-500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import torch
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_OMNISTYLE = os.path.join(os.path.dirname(_HERE), "omnistyle")
for _p in (_HERE, _OMNISTYLE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omnivoice.training.config import TrainingConfig  # noqa: E402
from omnivoice.training.trainer import OmniTrainer  # noqa: E402

import ft_data  # noqa: E402
import smoke_style as S  # noqa: E402
import train_style as TS  # noqa: E402
from style_omnivoice import StyleOmniVoice, StyleOmniVoiceConfig  # noqa: E402


CHECKSUM_NAME = "SHA256SUMS"

# category -> one of these relative file names must exist in every complete checkpoint
REQUIRED_CHECKPOINT_FILES: Dict[str, Tuple[str, ...]] = {
    "model config": ("config.json",),
    "model weights": ("model.safetensors", "pytorch_model.bin"),
    "optimizer state": ("optimizer.bin",),
    "scheduler state": ("scheduler.bin",),
    "rng state": ("random_states_0.pkl",),
    "trainer config": ("train_config.json",),
    "pipeline config": ("ft_config.json",),
    "text tokenizer": ("tokenizer.json", "tokenizer_config.json", "fake_tokenizer.json"),
}


class SimulatedKill(RuntimeError):
    """Raised by the metrics sink to simulate a mid-run worker kill (tests only)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FTTrainConfig:
    output_dir: str = ""

    # model / data
    synthetic: bool = False
    init_from: str = "k2-fsa/OmniVoice"
    data_config: Optional[str] = None
    style_n_vectors: int = 4
    style_quantizer: str = "none"  # shipped default; fsq/vq remain selectable
    target_frames: int = 250
    crop_frames: int = 75

    # optimisation (primer train_config_finetune keys + shipped defaults)
    steps: int = 5000
    save_steps: int = 500
    keep_last_n_checkpoints: int = -1
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    learning_rate: float = 5e-4
    encoder_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    batch_tokens: int = 8192
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "cosine"
    warmup_type: str = "ratio"
    warmup_ratio: float = 0.01
    warmup_steps: int = 0

    # conditioning (prompt_ratio is fixed at 0: the prefix replaces the audio prompt)
    prompt_ratio: float = 0.0
    mask_ratio_range: Tuple[float, float] = (0.5, 1.0)
    drop_cond_ratio: float = 0.1
    style_drop_ratio: float = 0.2
    language_ratio: float = 0.8
    instruct_ratio: float = 0.0
    kl_weight: float = 1.0

    # system / data loader
    mixed_precision: str = "bf16"
    allow_tf32: bool = True
    attn_implementation: str = "sdpa"
    logging_steps: int = 50
    eval_steps: int = 10**9
    num_workers: int = 2
    min_sample_tokens: int = 50
    max_sample_tokens: int = 2000
    max_batch_size: int = 64
    batch_size: int = 16  # synthetic loader only
    synthetic_replay_fraction: float = 0.0

    # collapse guard (train_style.py machinery; ON by default)
    guard: bool = True
    guard_warmup: int = 200
    guard_every: int = 100
    guard_patience: int = 2
    guard_attempts: int = 3
    guard_min_sep: float = 0.5  # labelled synthetic probe (validated threshold)
    guard_min_z_spread: float = 0.01  # floor for the label-free real-data probe
    guard_z_spread_frac: float = 0.1  # ... and a fraction of the pre-training spread

    # test / failure-injection hooks (micro-steps; attempt 0 only)
    inject_collapse_at: Optional[int] = None
    kill_at_step: Optional[int] = None


def _config_dict(cfg: FTTrainConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d["mask_ratio_range"] = list(cfg.mask_ratio_range)
    return d


def _resolved_mixed_precision(cfg: FTTrainConfig) -> str:
    mp = cfg.mixed_precision
    if mp == "bf16" and not torch.cuda.is_available():
        # bf16 autocast on CPU is not what the primer's GPU recipe means; the CPU smoke runs fp32.
        return "no"
    return mp


# ---------------------------------------------------------------------------
# Model / data / optimizer
# ---------------------------------------------------------------------------


def load_text_tokenizer(cfg: FTTrainConfig) -> Tuple[Any, str]:
    """(tokenizer, resolved init path): the text tokenizer with upstream's pad and special tokens.

    Split out of ``load_style_model`` so the home real-path check can build the data path without
    loading the 2.5 GB model."""
    from transformers import AutoTokenizer

    from omnivoice.models.omnivoice import _resolve_model_path

    init_path = _resolve_model_path(cfg.init_from)
    tokenizer_path = cfg.resume_from_checkpoint or init_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special = [
        "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
        "<|instruct_end|>", "<|text_start|>", "<|text_end|>",
    ]
    to_add = [t for t in special if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return tokenizer, init_path


def load_style_model(cfg: FTTrainConfig) -> Tuple[StyleOmniVoice, Any]:
    """Load the public Base into the subclass; tokenizer from the resume checkpoint when given."""
    tokenizer, init_path = load_text_tokenizer(cfg)

    model_cfg = StyleOmniVoiceConfig.from_pretrained(init_path)
    model_cfg.style_n_vectors = cfg.style_n_vectors
    model_cfg.style_quantizer = cfg.style_quantizer
    model = StyleOmniVoice.from_pretrained(
        init_path,
        config=model_cfg,
        attn_implementation=cfg.attn_implementation,
        dtype=torch.float32,  # fp32 weights + bf16 autocast, as upstream's builder does
        train=True,
    )
    if len(tokenizer) != model.config.llm_config.vocab_size:
        model.llm.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.ensure_style_scale_calibrated()
    return model, tokenizer


def build_optimizer(model: StyleOmniVoice, cfg: FTTrainConfig) -> torch.optim.Optimizer:
    """AdamW with a separate, higher lr for the new style encoder."""
    enc = [p for p in model.style_encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    if not enc or not rest:
        raise ValueError("optimizer needs both encoder and backbone parameters")
    return torch.optim.AdamW(
        [
            # backbone first so the trainer's logged lr is the shipped 5e-4, not the encoder's higher one
            {"params": rest, "lr": cfg.learning_rate},
            {"params": enc, "lr": cfg.encoder_learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: FTTrainConfig):
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    if cfg.warmup_type == "ratio":
        warmup = math.ceil(cfg.steps * cfg.warmup_ratio)
    else:
        warmup = cfg.warmup_steps
    if cfg.lr_scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=cfg.steps
    )


def _build_data(cfg: FTTrainConfig, seed: int, tokenizer) -> ft_data.DataBundle:
    if cfg.synthetic:
        return ft_data.build_synthetic_bundle(
            batch_size=cfg.batch_size,
            seed=seed,
            replay_fraction=cfg.synthetic_replay_fraction,
            style_drop_ratio=cfg.style_drop_ratio,
            drop_cond_ratio=cfg.drop_cond_ratio,
            mask_ratio_range=cfg.mask_ratio_range,
            n_style=cfg.style_n_vectors,
        )
    if not cfg.data_config:
        raise SystemExit("a real run needs --data-config (see README: manifest -> tokenise -> data.lst)")
    return ft_data.build_shard_bundle(
        cfg.data_config,
        tokenizer,
        batch_tokens=cfg.batch_tokens,
        min_sample_tokens=cfg.min_sample_tokens,
        max_sample_tokens=cfg.max_sample_tokens,
        max_batch_size=cfg.max_batch_size,
        num_workers=cfg.num_workers,
        n_style=cfg.style_n_vectors,
        target_frames=cfg.target_frames,
        crop_frames=cfg.crop_frames,
        mask_ratio_range=cfg.mask_ratio_range,
        drop_cond_ratio=cfg.drop_cond_ratio,
        style_drop_ratio=cfg.style_drop_ratio,
        language_ratio=cfg.language_ratio,
        instruct_ratio=cfg.instruct_ratio,
    )


def _build_trainer_config(cfg: FTTrainConfig, seed: int) -> TrainingConfig:
    return TrainingConfig(
        output_dir=cfg.output_dir,
        init_from_checkpoint=cfg.init_from,
        steps=cfg.steps,
        seed=seed,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_type=cfg.warmup_type,
        warmup_ratio=cfg.warmup_ratio,
        warmup_steps=cfg.warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        batch_tokens=cfg.batch_tokens,
        num_workers=0 if cfg.synthetic else cfg.num_workers,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        keep_last_n_checkpoints=cfg.keep_last_n_checkpoints,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        mixed_precision=_resolved_mixed_precision(cfg),
        allow_tf32=cfg.allow_tf32,
        attn_implementation=cfg.attn_implementation,
        max_sample_tokens=cfg.max_sample_tokens,
        min_sample_tokens=cfg.min_sample_tokens,
        max_batch_size=cfg.max_batch_size,
        drop_cond_ratio=cfg.drop_cond_ratio,
        prompt_ratio_range=(cfg.prompt_ratio, cfg.prompt_ratio),  # prompt_length = 0
        mask_ratio_range=tuple(cfg.mask_ratio_range),
        language_ratio=cfg.language_ratio,
        instruct_ratio=cfg.instruct_ratio,
        only_instruct_ratio=0.0,
        use_pinyin_ratio=0.0,
    )


# ---------------------------------------------------------------------------
# Checkpoint contract: required files + SHA256SUMS + ft_config.json
# ---------------------------------------------------------------------------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative_files(root: str) -> List[str]:
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if rel == CHECKSUM_NAME:
                continue
            out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def write_checkpoint_manifest(checkpoint_dir: str) -> str:
    """Write ``SHA256SUMS`` (sha256sum-compatible: ``<hash>  <relative path>``)."""
    lines = [f"{_sha256(os.path.join(checkpoint_dir, rel))}  {rel}" for rel in _relative_files(checkpoint_dir)]
    path = os.path.join(checkpoint_dir, CHECKSUM_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


def verify_checkpoint(checkpoint_dir: str, strict_files: bool = True) -> Dict[str, Any]:
    """Check completeness, hashes and (strictly) that every file is listed in SHA256SUMS."""
    problems: List[str] = []
    if not os.path.isdir(checkpoint_dir):
        return {"path": checkpoint_dir, "ok": False, "problems": ["not a directory"], "files": []}
    files = _relative_files(checkpoint_dir)
    for category, options in REQUIRED_CHECKPOINT_FILES.items():
        if not any(name in files for name in options):
            problems.append(f"missing {category} (one of {list(options)})")
    sums_path = os.path.join(checkpoint_dir, CHECKSUM_NAME)
    if not os.path.isfile(sums_path):
        problems.append(f"missing {CHECKSUM_NAME}")
    else:
        listed: Dict[str, str] = {}
        with open(sums_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                digest, _, rel = line.partition("  ")
                if not rel:
                    problems.append(f"malformed {CHECKSUM_NAME} line: {line!r}")
                    continue
                listed[rel] = digest
        for rel, digest in sorted(listed.items()):
            path = os.path.join(checkpoint_dir, rel)
            if not os.path.isfile(path):
                problems.append(f"listed file is missing: {rel}")
            elif _sha256(path) != digest:
                problems.append(f"checksum mismatch: {rel}")
        if strict_files:
            for rel in files:
                if rel not in listed:
                    problems.append(f"file not listed in {CHECKSUM_NAME}: {rel}")
    return {"path": checkpoint_dir, "ok": not problems, "problems": problems, "files": files}


def _install_checkpoint_hook(trainer: OmniTrainer, ft_config: Dict[str, Any]):
    """After every upstream save: drop ft_config.json in and write SHA256SUMS.

    The upstream ``OmniTrainer`` itself is untouched; only this instance's ``save_checkpoint``
    attribute is wrapped, so a checkpoint is never left without its manifest.
    """
    original = trainer.save_checkpoint

    def save_with_manifest(step: int):
        original(step)
        if not trainer.accelerator.is_main_process:
            return
        ckpt = os.path.join(trainer.config.output_dir, f"checkpoint-{step}")
        with open(os.path.join(ckpt, "ft_config.json"), "w", encoding="utf-8") as f:
            json.dump(ft_config, f, indent=1)
        write_checkpoint_manifest(ckpt)

    trainer.save_checkpoint = save_with_manifest  # instance attribute shadows the method


# ---------------------------------------------------------------------------
# Collapse guard
# ---------------------------------------------------------------------------


def _guard_stats(model, data: ft_data.DataBundle, cfg: FTTrainConfig, baseline_z: Optional[float]) -> Dict[str, Any]:
    if data.probe_kind == "labeled":
        stats = ft_data.labeled_probe_stats(model, data.probe_dataset)
        collapsed = stats["sep_ratio"] < cfg.guard_min_sep
        return {**stats, "probe": "labeled", "metric": "sep_ratio",
                "threshold": cfg.guard_min_sep, "collapsed": bool(collapsed)}
    stats = ft_data.unsupervised_probe_stats(model, data.probe_crops)
    threshold = cfg.guard_min_z_spread
    if baseline_z is not None:
        threshold = max(threshold, cfg.guard_z_spread_frac * baseline_z)
    return {**stats, "probe": "z_spread", "metric": "z_spread_min",
            "threshold": round(threshold, 6),
            "collapsed": bool(stats["z_spread_min"] < threshold)}


def _run_training_once(
    cfg: FTTrainConfig,
    attempt_seed: int,
    attempt: int,
    guard_log: List[Dict[str, Any]],
) -> Tuple[OmniTrainer, StyleOmniVoice]:
    """One training attempt.  Raises ``train_style.CollapseDetected`` when the guard fires."""
    if cfg.synthetic:
        model = S.build_model(
            seed=attempt_seed,
            style_n_vectors=cfg.style_n_vectors,
            style_quantizer=cfg.style_quantizer,
        )
        tokenizer = None
    else:
        model, tokenizer = load_style_model(cfg)
    data = _build_data(cfg, attempt_seed, tokenizer)
    tokenizer = data.tokenizer

    # Frozen Base snapshot for the KL term (not a submodule: never in state_dict/DDP/checkpoints).
    if cfg.kl_weight > 0:
        model.attach_reference(StyleOmniVoice.reference_from(model))

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    trainer_config = _build_trainer_config(cfg, attempt_seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.output_dir, "style_metrics.jsonl")
    guard_path = os.path.join(cfg.output_dir, "collapse_guard.jsonl")
    counter = {"n": 0, "strikes": 0, "injected": False}
    baseline_z = None
    if cfg.guard and data.probe_kind == "unsupervised":
        baseline_z = ft_data.unsupervised_probe_stats(model, data.probe_crops)["z_spread_mean"]

    def sink(row: Dict[str, Any]):
        counter["n"] += 1
        if counter["n"] % cfg.logging_steps == 0 or counter["n"] == 1:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"micro_step": counter["n"], "attempt": attempt,
                                    **{k: float(v) for k, v in row.items()}}) + "\n")
        if cfg.kill_at_step is not None and attempt == 0 and counter["n"] >= cfg.kill_at_step:
            raise SimulatedKill(f"simulated mid-run kill at micro-step {counter['n']}")
        if (cfg.inject_collapse_at is not None and attempt == 0 and not counter["injected"]
                and counter["n"] >= cfg.inject_collapse_at):
            # Same injection as train_style: swap in zeroed, frozen parameters (never zero the live
            # weight mid-forward: that bumps the version of the tensor the Linear's backward saved).
            with torch.no_grad():
                lin = model.style_encoder.to_code
                lin.weight = torch.nn.Parameter(torch.zeros_like(lin.weight), requires_grad=False)
                if lin.bias is not None:
                    lin.bias = torch.nn.Parameter(torch.zeros_like(lin.bias), requires_grad=False)
            counter["injected"] = True
        if cfg.guard and counter["n"] >= cfg.guard_warmup and counter["n"] % cfg.guard_every == 0:
            stats = _guard_stats(model, data, cfg, baseline_z)
            counter["strikes"] = counter["strikes"] + 1 if stats["collapsed"] else 0
            row = {"attempt": attempt, "seed": attempt_seed, "step": counter["n"], **stats,
                   "strikes": counter["strikes"]}
            guard_log.append(row)
            with open(guard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if counter["strikes"] >= cfg.guard_patience:
                raise TS.CollapseDetected(
                    f"style code collapsed at step {counter['n']} (attempt {attempt}, seed {attempt_seed}): "
                    f"{stats['metric']}={stats[stats['metric']]} threshold={stats['threshold']}",
                    guard_log,
                )

    model.configure_training(kl_weight=cfg.kl_weight, style_aux_weight=1.0, metrics_sink=sink)
    trainer = OmniTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=data.loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    _install_checkpoint_hook(trainer, _config_dict(cfg))
    trainer.train()
    return trainer, model


def _check_resume_config(cfg: FTTrainConfig):
    """Refuse a resume whose architecture knobs differ from the checkpoint's."""
    path = os.path.join(cfg.resume_from_checkpoint or "", "ft_config.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    arch = ("style_n_vectors", "style_quantizer", "target_frames", "crop_frames", "init_from")
    bad = [k for k in arch if k in saved and saved[k] != getattr(cfg, k)]
    if bad:
        raise SystemExit(
            f"resume config mismatch in {path}: {bad}; the checkpoint's architecture must be reused "
            "(pass the same flags, or edit the saved ft_config.json deliberately)"
        )


def run_training(cfg: FTTrainConfig) -> Tuple[OmniTrainer, StyleOmniVoice]:
    """Run the pipeline with the collapse guard and checkpoint contract; returns the last attempt."""
    if not cfg.output_dir:
        raise SystemExit("--output-dir is required")
    if not cfg.synthetic and not cfg.data_config:
        raise SystemExit("a real run needs --data-config (or use --synthetic)")
    if cfg.resume_from_checkpoint:
        _check_resume_config(cfg)
        report = verify_checkpoint(cfg.resume_from_checkpoint, strict_files=True)
        if not report["ok"]:
            raise SystemExit(f"refusing to resume from an unverifiable checkpoint: {report['problems']}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "ft_config.json"), "w", encoding="utf-8") as f:
        json.dump(_config_dict(cfg), f, indent=1)
    guard_path = os.path.join(cfg.output_dir, "collapse_guard.jsonl")
    if os.path.exists(guard_path):
        os.remove(guard_path)  # this call's log

    attempts = cfg.guard_attempts if (cfg.guard and cfg.resume_from_checkpoint is None) else 1
    guard_log: List[Dict[str, Any]] = []
    last: Optional[TS.CollapseDetected] = None
    for attempt in range(attempts):
        try:
            trainer, model = _run_training_once(cfg, cfg.seed + attempt, attempt, guard_log)
            trainer.guard_log = guard_log
            trainer.guard_attempts = attempt + 1
            return trainer, model
        except TS.CollapseDetected as e:
            last = e
            if attempt + 1 < attempts:
                TS._clear_partial_run(cfg.output_dir)  # only this run's checkpoints/metrics
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(ap: argparse.ArgumentParser):
    ap.add_argument("--output-dir", default=None, help="run folder; checkpoints land here")
    ap.add_argument("--synthetic", action="store_true", help="tiny model + synthetic tokens (CPU smoke)")
    ap.add_argument("--init-from", default="k2-fsa/OmniVoice")
    ap.add_argument("--data-config", default=None, help="upstream data config JSON (manifest_path -> data.lst)")
    ap.add_argument("--style-n-vectors", type=int, default=4)
    ap.add_argument("--style-quantizer", default="none", choices=["none", "fsq", "vq"])
    ap.add_argument("--target-frames", type=int, default=250)
    ap.add_argument("--crop-frames", type=int, default=75)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--keep-last-n-checkpoints", type=int, default=-1)
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=5e-4, help="backbone lr (shipped default 5e-4)")
    ap.add_argument("--encoder-lr", type=float, default=1e-3, help="style-encoder lr (higher by design; 5e-3 collapsed the tiny probe)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "constant"])
    ap.add_argument("--warmup-ratio", type=float, default=0.01)
    ap.add_argument("--mask-ratio-range", type=float, nargs=2, default=(0.5, 1.0))
    ap.add_argument("--drop-cond-ratio", type=float, default=0.1)
    ap.add_argument("--style-drop-ratio", type=float, default=0.2)
    ap.add_argument("--language-ratio", type=float, default=0.8)
    ap.add_argument("--instruct-ratio", type=float, default=0.0)
    ap.add_argument("--kl-weight", type=float, default=1.0, help="KL-to-Base on public replay rows; 0 disables")
    ap.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "no"])
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16, help="synthetic loader only")
    ap.add_argument("--no-collapse-guard", action="store_true")
    ap.add_argument("--guard-attempts", type=int, default=3)
    ap.add_argument("--guard-warmup", type=int, default=200)
    ap.add_argument("--guard-every", type=int, default=100)
    ap.add_argument("--guard-patience", type=int, default=2)
    ap.add_argument("--guard-min-sep", type=float, default=0.5)
    ap.add_argument("--guard-min-z-spread", type=float, default=0.01)
    ap.add_argument("--guard-z-spread-frac", type=float, default=0.1)
    ap.add_argument("--inject-collapse-at", type=int, default=None, help="test hook (micro-step)")
    ap.add_argument("--kill-at-step", type=int, default=None, help="test hook (micro-step)")
    ap.add_argument("--verify-checkpoint", nargs="+", default=None,
                    help="verify checkpoint dir(s) and exit")


def config_from_args(a: argparse.Namespace) -> FTTrainConfig:
    return FTTrainConfig(
        output_dir=a.output_dir or "",
        synthetic=a.synthetic,
        init_from=a.init_from,
        data_config=a.data_config,
        style_n_vectors=a.style_n_vectors,
        style_quantizer=a.style_quantizer,
        target_frames=a.target_frames,
        crop_frames=a.crop_frames,
        steps=a.steps,
        save_steps=a.save_steps,
        keep_last_n_checkpoints=a.keep_last_n_checkpoints,
        resume_from_checkpoint=a.resume_from_checkpoint,
        seed=a.seed,
        learning_rate=a.lr,
        encoder_learning_rate=a.encoder_lr,
        weight_decay=a.weight_decay,
        max_grad_norm=a.max_grad_norm,
        batch_tokens=a.batch_tokens,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        lr_scheduler_type=a.lr_scheduler,
        warmup_ratio=a.warmup_ratio,
        mask_ratio_range=tuple(a.mask_ratio_range),
        drop_cond_ratio=a.drop_cond_ratio,
        style_drop_ratio=a.style_drop_ratio,
        language_ratio=a.language_ratio,
        instruct_ratio=a.instruct_ratio,
        kl_weight=a.kl_weight,
        mixed_precision=a.mixed_precision,
        allow_tf32=not a.no_tf32,
        num_workers=a.num_workers,
        logging_steps=a.logging_steps,
        batch_size=a.batch_size,
        guard=not a.no_collapse_guard,
        guard_attempts=a.guard_attempts,
        guard_warmup=a.guard_warmup,
        guard_every=a.guard_every,
        guard_patience=a.guard_patience,
        guard_min_sep=a.guard_min_sep,
        guard_min_z_spread=a.guard_min_z_spread,
        guard_z_spread_frac=a.guard_z_spread_frac,
        inject_collapse_at=a.inject_collapse_at,
        kill_at_step=a.kill_at_step,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    _add_common(ap)
    a = ap.parse_args(argv)
    if a.verify_checkpoint:
        reports = [verify_checkpoint(p) for p in a.verify_checkpoint]
        print(json.dumps(reports, indent=1))
        return 0 if all(r["ok"] for r in reports) else 1
    if not a.synthetic and not a.data_config:
        ap.error("give --data-config for a real run (or --synthetic for the CPU smoke)")
    cfg = config_from_args(a)
    trainer, model = run_training(cfg)
    print(json.dumps({"output_dir": cfg.output_dir, "global_step": trainer.global_step,
                      "attempts": getattr(trainer, "guard_attempts", None)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
