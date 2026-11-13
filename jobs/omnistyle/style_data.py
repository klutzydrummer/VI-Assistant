"""Data side of the style-prefix route: fake tokenizer, sample processor, collator, synthetic clips.

Sequence layout per sample (style-prefix kept):

    [lang/instruct tokens][text][n style slots][masked target audio]

The slots are `audio_mask=True` placeholder positions (token id = mask id, no label); the model overwrites
their embeddings with the style-encoder vectors.  No real audio is used as a prompt (`prompt_length = 0`).

Independent dropouts (all per sample):
  * `drop_cond_ratio`   : everything dropped -> audio-only sequence (as upstream)
  * `style_drop_ratio`  : only the prefix dropped, text kept -> the "text-only" CFG branch
  * `replay` samples    : prefix always dropped, flagged for the KL-to-Base term
The style encoder crop is returned for EVERY sample, so the encoder runs on every step.
"""

from __future__ import annotations

import random
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from omnivoice.data.collator import PaddingDataCollator


class FakeTextTokenizer:
    """Tiny stand-in for the Qwen tokenizer: `<|tag|>` is one token, other characters hash into the vocab."""

    pad_token_id = 0

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

    def save_pretrained(self, path: str):  # OmniTrainer's checkpoint writer calls this
        import json
        import os

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_tokenizer.json"), "w") as f:
            json.dump({"vocab_size": self.vocab_size}, f)

    def _tok(self, piece: str) -> int:
        return 1 + (sum(ord(c) * (i + 1) for i, c in enumerate(piece)) % (self.vocab_size - 2))

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids: List[int] = []
        for part in re.split(r"(<\|[^|]*\|>)", text):
            if not part:
                continue
            if part.startswith("<|"):
                ids.append(self._tok(part))
            else:
                ids.extend(self._tok(c) for c in part)
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class StyleSampleProcessor:
    """Raw sample -> model tensors, with the style slot in the prompt position.

    Raw sample: {"audio_tokens": [C,T], "style_tokens": [C,Tc], "label": {"text": str, "replay": bool?}}.
    `style_tokens` is a DIFFERENT crop of the same recording (the caller chooses it).
    """

    def __init__(
        self,
        text_tokenizer: Any,
        num_channels: int,
        audio_mask_id: int,
        n_style: int,
        mask_ratio_range: Tuple[float, float] = (0.5, 1.0),
        drop_cond_ratio: float = 0.1,
        style_drop_ratio: float = 0.2,
        language_ratio: float = 0.8,
        instruct_ratio: float = 1.0,
    ):
        self.text_tokenizer = text_tokenizer
        self.num_channels = num_channels
        self.audio_mask_id = audio_mask_id
        self.n_style = n_style
        self.mask_ratio_range = mask_ratio_range
        self.drop_cond_ratio = drop_cond_ratio
        self.style_drop_ratio = style_drop_ratio
        self.language_ratio = language_ratio
        self.instruct_ratio = instruct_ratio

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        C = self.num_channels
        label = sample["label"]
        replay = bool(label.get("replay", False))
        drop_cond = random.random() < self.drop_cond_ratio
        keep_style = (not drop_cond) and (not replay) and (random.random() >= self.style_drop_ratio)
        mask_ratio = random.uniform(*self.mask_ratio_range)

        audio_tokens = sample["audio_tokens"].long()
        T = audio_tokens.shape[1]
        token_mask = torch.rand(C, T) < mask_ratio
        if not bool(token_mask.any()):  # never emit a sample without any loss position
            token_mask[0, random.randrange(T)] = True
        audio_inputs = torch.where(token_mask, torch.full_like(audio_tokens, self.audio_mask_id), audio_tokens)
        audio_labels = torch.where(token_mask, audio_tokens, torch.full_like(audio_tokens, -100))

        if drop_cond:
            input_ids, labels = audio_inputs, audio_labels
            audio_mask = torch.ones(T, dtype=torch.bool)
            style_slot = torch.zeros(T, dtype=torch.bool)
        else:
            language = "en" if random.random() < self.language_ratio else "None"
            instruct = "None" if random.random() < self.instruct_ratio else "None"
            style = f"<|lang_start|>{language}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>"
            style_ids = self.text_tokenizer(style, return_tensors="pt").input_ids.repeat(C, 1)
            text_ids = self.text_tokenizer(
                f"<|text_start|>{label['text']}<|text_end|>", return_tensors="pt"
            ).input_ids.repeat(C, 1)
            pieces = [style_ids, text_ids]
            lab_pieces = [torch.full(style_ids.shape, -100), torch.full(text_ids.shape, -100)]
            pre = style_ids.shape[1] + text_ids.shape[1]
            if keep_style:
                n = self.n_style
                pieces.append(torch.full((C, n), self.audio_mask_id, dtype=torch.long))
                lab_pieces.append(torch.full((C, n), -100))
                pre_slot = pre
                pre += n
            pieces.append(audio_inputs)
            lab_pieces.append(audio_labels)
            input_ids = torch.cat(pieces, dim=1)
            labels = torch.cat(lab_pieces, dim=1)
            L = input_ids.shape[1]
            audio_mask = torch.zeros(L, dtype=torch.bool)
            audio_mask[pre - (self.n_style if keep_style else 0) :] = True  # slot (if any) + target
            style_slot = torch.zeros(L, dtype=torch.bool)
            if keep_style:
                style_slot[pre_slot : pre_slot + self.n_style] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "audio_mask": audio_mask,
            "style_slot": style_slot,
            "style_tokens": sample["style_tokens"].long(),
            "kl": replay,
            "length": input_ids.shape[1],
        }


class StylePaddingCollator:
    """Upstream `PaddingDataCollator` plus the style fields (slot mask, encoder crops, KL flags)."""

    def __init__(self, processor, batch_tokens: int = 0):
        self.base = PaddingDataCollator(processor, batch_tokens)
        self.processor = processor

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base(samples)
        B = len(samples)
        L = batch["input_ids"].shape[-1]
        slot = torch.zeros(B, L, dtype=torch.bool)
        for i, s in enumerate(samples):
            slot[i, : s["length"]] = s["style_slot"]
        Tc = max(s["style_tokens"].shape[1] for s in samples)
        C = samples[0]["style_tokens"].shape[0]
        toks = torch.zeros(B, C, Tc, dtype=torch.long)
        valid = torch.zeros(B, Tc, dtype=torch.bool)
        for i, s in enumerate(samples):
            t = s["style_tokens"].shape[1]
            toks[i, :, :t] = s["style_tokens"]
            valid[i, :t] = True
        batch["style_slot"] = slot
        batch["style_tokens"] = toks
        batch["style_valid"] = valid
        batch["kl_mask"] = torch.tensor([bool(s["kl"]) for s in samples])
        return batch


# ---------------------------------------------------------------------------
# Synthetic clips: K "styles" are known token-frequency biases on codebooks 0 and 1
# ---------------------------------------------------------------------------

TEXTS = ["hello there", "good morning", "thank you", "see you soon", "what is that"]
BLOCK = 8  # preferred tokens per style per codebook (clips use a small effective vocab so a tiny CPU model can learn the bias)


def style_block(style: int, codebook: int, n_styles: int) -> range:
    """Token range that style `style` prefers on codebook 0 or 1 (disjoint between styles)."""
    k = style if codebook == 0 else (style + 1) % n_styles  # different block order on codebook 1
    return range(k * BLOCK, (k + 1) * BLOCK)


def make_clip(style: int, T: int, n_styles: int, bias: float, C: int, vocab: int, gen: torch.Generator) -> torch.Tensor:
    """[C,T] tokens: codebooks 0,1 prefer the style block with probability `bias`; the rest is uniform noise."""
    clip = torch.randint(0, vocab, (C, T), generator=gen)
    for cb in (0, 1):
        blk = style_block(style, cb, n_styles)
        pick = torch.randint(blk.start, blk.stop, (T,), generator=gen)
        use = torch.rand(T, generator=gen) < bias
        clip[cb] = torch.where(use, pick, clip[cb])
    return clip


class SyntheticStyleDataset(Dataset):
    """Clips with a known style; encoder crop and target crop are different windows of one clip."""

    def __init__(
        self,
        n_clips: int = 100_000,
        n_styles: int = 4,
        clip_len: int = 48,
        target_len: int = 16,
        crop_len: int = 16,
        bias: float = 0.8,
        C: int = 8,
        vocab: int = 64,
        seed: int = 0,
        replay_fraction: float = 0.0,
    ):
        self.n_clips, self.n_styles = n_clips, n_styles
        self.clip_len, self.target_len, self.crop_len = clip_len, target_len, crop_len
        self.bias, self.C, self.vocab, self.seed = bias, C, vocab, seed
        self.replay_fraction = replay_fraction

    def __len__(self):
        return self.n_clips

    def style_of(self, idx: int) -> int:
        return idx % self.n_styles

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        style = self.style_of(idx)
        clip = make_clip(style, self.clip_len, self.n_styles, self.bias, self.C, self.vocab, gen)
        # target window and a DIFFERENT (non-overlapping when it fits) encoder window
        t0 = int(torch.randint(0, self.clip_len - self.target_len + 1, (1,), generator=gen))
        target = clip[:, t0 : t0 + self.target_len]
        if t0 >= self.crop_len:
            c0 = int(torch.randint(0, t0 - self.crop_len + 1, (1,), generator=gen))
        else:
            c0 = int(torch.randint(t0 + self.target_len, self.clip_len - self.crop_len + 1, (1,), generator=gen))
        crop = clip[:, c0 : c0 + self.crop_len]
        # Few distinct texts that carry no style information, so the text cannot identify a clip (a unique
        # text per clip lets the tiny LLM memorise clips instead of reading the prefix).
        text = TEXTS[int(torch.randint(0, len(TEXTS), (1,), generator=gen))]
        replay = bool(torch.rand(1, generator=gen) < self.replay_fraction)
        return {
            "audio_tokens": target,
            "style_tokens": crop,
            "label": {"text": text, "replay": replay, "style": style},
        }

    def clips_of_style(self, style: int, n: int, seed: int = 123) -> torch.Tensor:
        """[n,C,crop_len] fresh encoder crops of one style (for the decode and shuffle checks)."""
        gen = torch.Generator().manual_seed(seed + style)
        return torch.stack(
            [make_clip(style, self.crop_len, self.n_styles, self.bias, self.C, self.vocab, gen) for _ in range(n)]
        )
