"""Data side of the OmniVoice full fine-tune: real shard samples and the synthetic CPU bundle.

The sequence layout, the dropouts and the collator are reused unchanged from ``jobs/omnistyle``
(``style_data.StyleSampleProcessor`` / ``StylePaddingCollator``).  This module adds two things:

* ``StyleShardProcessor``: turns an upstream OmniVoice training sample
  (``{"audio_tokens": [C, T], "label": {...}}``, read from the offline-tokenised WebDataset
  shards) into that same synthetic-sample shape.  The target is one random window of the clip and
  the encoder reads a *different* window of the same recording, so the style prefix cannot simply
  copy the target (conditioning.md risk 3).  ``prompt_length`` stays 0: no real audio is used as a
  prompt, the learned prefix occupies the prompt slot.
* ``build_shard_bundle`` / ``build_synthetic_bundle``: assemble the batched loader plus a guard
  probe reference.  The synthetic bundle (tiny Qwen3 + known style biases) is what the CPU tests
  use; the shard bundle is the real training path and is not exercised in this task.

Data rules: the processor reads only ``audio_tokens`` and ``label.text`` / ``label.replay``.  It
never looks at mood or any legacy label.  ``label.replay`` marks public-voice replay rows for the
KL-to-Base guard and is set by ``data_manifest.py``.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

_OMNISTYLE = Path(__file__).resolve().parents[1] / "omnistyle"
if str(_OMNISTYLE) not in sys.path:
    sys.path.insert(0, str(_OMNISTYLE))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ft_guards as FG  # noqa: E402
import style_data as SD  # noqa: E402
import smoke_style as S  # noqa: E402
import train_style as TS  # noqa: E402


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


def choose_windows(
    tokens: torch.Tensor,
    target_frames: int,
    crop_frames: int,
    rng: random.Random = random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick a target window and a non-overlapping (when the clip allows) encoder window.

    ``tokens`` is the whole clip ``[C, T]``.  Returns ``(target, crop)`` as views, both
    ``[C, min(T, frames)]``.  The encoder window is a different part of the same recording; when a
    non-overlapping placement does not exist it is drawn from anywhere in the clip.
    """
    _, total = tokens.shape
    target_len = min(total, max(1, target_frames))
    crop_len = min(total, max(1, crop_frames))
    t0 = rng.randrange(0, total - target_len + 1)
    starts = list(range(0, t0 - crop_len + 1)) + list(
        range(t0 + target_len, total - crop_len + 1)
    )
    if starts:
        c0 = starts[rng.randrange(len(starts))]
    else:
        c0 = rng.randrange(0, total - crop_len + 1)
    return tokens[:, t0 : t0 + target_len], tokens[:, c0 : c0 + crop_len]


class StyleShardProcessor:
    """Upstream shard sample -> the validated synthetic-sample layout.

    Delegates the layout, masking and dropouts to ``style_data.StyleSampleProcessor``; adds the
    window choice, the replay flag and the encoder crop.  Every sample carries a crop, so the
    style encoder runs on every training step (DDP ``find_unused_parameters=False``).
    """

    def __init__(
        self,
        text_tokenizer: Any,
        num_channels: int = 8,
        audio_mask_id: int = 1024,
        n_style: int = 4,
        target_frames: int = 250,
        crop_frames: int = 75,
        mask_ratio_range: Tuple[float, float] = (0.5, 1.0),
        drop_cond_ratio: float = 0.1,
        style_drop_ratio: float = 0.2,
        language_ratio: float = 0.8,
        instruct_ratio: float = 0.0,
        rng: random.Random = random,
    ):
        # Upstream's PaddingDataCollator reads ``processor.text_tokenizer.pad_token_id``, so the
        # wrapper must expose it too (r6 failed with an AttributeError here).
        self.text_tokenizer = text_tokenizer
        self.base = SD.StyleSampleProcessor(
            text_tokenizer,
            num_channels=num_channels,
            audio_mask_id=audio_mask_id,
            n_style=n_style,
            mask_ratio_range=mask_ratio_range,
            drop_cond_ratio=drop_cond_ratio,
            style_drop_ratio=style_drop_ratio,
            language_ratio=language_ratio,
            instruct_ratio=instruct_ratio,
        )
        self.target_frames = target_frames
        self.crop_frames = crop_frames
        self.rng = rng

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        tokens = sample["audio_tokens"].long()
        label = sample["label"]
        target, crop = choose_windows(tokens, self.target_frames, self.crop_frames, self.rng)
        out = self.base(
            {"audio_tokens": target, "style_tokens": crop, "label": label}
        )
        # A public-voice replay row always drops the prefix and takes part in the KL-to-Base term.
        out["kl"] = bool(label.get("replay", False))
        return out


# ---------------------------------------------------------------------------
# Bundles: loader + guard probe + tokenizer
# ---------------------------------------------------------------------------


@dataclass
class DataBundle:
    loader: DataLoader
    tokenizer: Any
    # "labeled": probe needs a synthetic dataset with known styles (validated path, tests only).
    # "unsupervised": probe uses fixed real crops and the encoder's spread (real training).
    probe_kind: str
    probe_dataset: Any = None
    probe_crops: Any = None  # list of (tokens [B,C,Tc], valid [B,Tc])
    sample_counter: Any = None  # ft_guards.SampleCounter on the real loader's processor


def build_synthetic_bundle(
    batch_size: int = 16,
    seed: int = 0,
    replay_fraction: float = 0.0,
    style_drop_ratio: float = 0.2,
    drop_cond_ratio: float = 0.1,
    mask_ratio_range: Tuple[float, float] = (0.5, 1.0),
    n_style: int = 4,
) -> DataBundle:
    """Tiny known-style synthetic clips (tests / CPU smoke); label-aware probe allowed here.

    ``n_style`` must match the tiny model's ``style_n_vectors``; the layout is otherwise the same as
    ``smoke_style.build_loader`` (which fixes the tiny default slot count, so it is not reused here).
    """
    ds = S.SyntheticStyleDataset(
        n_clips=100_000, n_styles=S.N_STYLES, seed=seed, replay_fraction=replay_fraction
    )
    proc = SD.StyleSampleProcessor(
        SD.FakeTextTokenizer(), 8, 1024, n_style=n_style, mask_ratio_range=mask_ratio_range,
        drop_cond_ratio=drop_cond_ratio, style_drop_ratio=style_drop_ratio,
    )

    class Wrapped(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return proc(ds[i])

    loader = DataLoader(
        Wrapped(), batch_size=batch_size, shuffle=True, collate_fn=SD.StylePaddingCollator(proc)
    )
    return DataBundle(
        loader=loader,
        tokenizer=proc.text_tokenizer,
        probe_kind="labeled",
        probe_dataset=ds,
    )


def build_shard_bundle(
    data_config: str,
    tokenizer: Any,
    *,
    batch_tokens: int = 8192,
    min_sample_tokens: int = 50,
    max_sample_tokens: int = 2000,
    max_batch_size: int = 64,
    num_workers: int = 2,
    n_style: int = 4,
    target_frames: int = 250,
    crop_frames: int = 75,
    mask_ratio_range: Tuple[float, float] = (0.5, 1.0),
    drop_cond_ratio: float = 0.1,
    style_drop_ratio: float = 0.2,
    language_ratio: float = 0.8,
    instruct_ratio: float = 0.0,
    probe_batches: int = 2,
) -> DataBundle:
    """Real training loader over upstream ``data.lst`` shards (sdpa / padding collator)."""
    from transformers.trainer_utils import seed_worker  # local: keep the module import light

    from omnivoice.data.batching import StreamLengthGroupDataset
    from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json

    manifests, _ = prepare_data_manifests_from_json(data_config)
    if not manifests:
        raise ValueError(f"{data_config}: no train manifests")
    proc = StyleShardProcessor(
        tokenizer,
        num_channels=8,
        audio_mask_id=1024,
        n_style=n_style,
        target_frames=target_frames,
        crop_frames=crop_frames,
        mask_ratio_range=mask_ratio_range,
        drop_cond_ratio=drop_cond_ratio,
        style_drop_ratio=style_drop_ratio,
        language_ratio=language_ratio,
        instruct_ratio=instruct_ratio,
    )
    collate = SD.StylePaddingCollator(proc, batch_tokens)
    # Upstream skips a sample whose processing raises; count them so a shrunk set is loud (F-DATA-2).
    counted = FG.SampleCounter(proc)

    def make_dataset(evaluation: bool):
        raw = WebDatasetReader(manifests=manifests, evaluation=evaluation)
        return StreamLengthGroupDataset(
            raw,
            batch_duration=batch_tokens,
            min_length=min_sample_tokens,
            max_length=max_sample_tokens,
            max_sample=max_batch_size,
            processor=counted,
            length_fn=lambda s: s["length"],
        )

    def make_loader(dataset, workers: int):
        kwargs = dict(
            batch_size=None,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )
        if workers > 0:
            kwargs["prefetch_factor"] = 4
            kwargs["worker_init_fn"] = partial(
                seed_worker,
                num_workers=workers,
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            )
        return DataLoader(dataset, **kwargs)

    loader = make_loader(make_dataset(evaluation=False), num_workers)

    # Fixed probe crops come from a separate, unshuffled pass so probing never consumes or
    # reorders the training stream.
    probe_loader = make_loader(make_dataset(evaluation=True), 0)
    crops: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, batch in enumerate(probe_loader):
        crops.append((batch["style_tokens"], batch["style_valid"]))
        if i + 1 >= probe_batches:
            break
    if not crops:
        raise ValueError("probe pass produced no batches; check the data config and tokenised shards")
    FG.check_sample_counts(counted, "probe pass")

    return DataBundle(
        loader=loader,
        tokenizer=tokenizer,
        probe_kind="unsupervised",
        probe_crops=crops,
        sample_counter=counted,
    )


# ---------------------------------------------------------------------------
# Collapse-guard probes
# ---------------------------------------------------------------------------


@torch.no_grad()
def unsupervised_probe_stats(model, probe_crops) -> Dict[str, float]:
    """Label-free spread of the continuous code on fixed real crops.

    ``z`` is unit-RMS per vector (``StyleEncoder._encode``), so a collapsed encoder maps every
    crop to the same direction and the across-batch std goes to 0.  Reports the mean and minimum
    spread over the fixed probe batches.  No style labels are used anywhere.

    NOTE: the threshold is unverified on real data; the only validated probe is the labelled
    synthetic one (`train_style.collapse_probe`).  The runner calibrates the real threshold from
    the encoder's spread before training starts (see ``train_ft._guard_stats``).
    """
    enc = model.style_encoder
    was_training = model.training
    model.eval()
    try:
        spreads = []
        for tokens, valid in probe_crops:
            z = enc.encode_z(tokens, valid).float()
            spreads.append(float(z.std(dim=0).pow(2).mean().sqrt()))
        return {
            "z_spread_mean": sum(spreads) / len(spreads),
            "z_spread_min": min(spreads),
        }
    finally:
        model.train(was_training)


def labeled_probe_stats(model, dataset, n: int = 32) -> Dict[str, float]:
    """The validated synthetic probe (leave-one-out centroid purity / sep_ratio)."""
    return TS.collapse_probe(model, dataset, n=n)
