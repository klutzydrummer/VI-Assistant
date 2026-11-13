"""OmniVoice with a learned style prefix (project-local subclass, upstream is not patched).

A `StyleEncoder` reads a *different crop of the same recording* (audio tokens only, no labels),
squeezes it through a tight bottleneck (small VQ code, or a small continuous code) plus noise,
and emits `n` continuous vectors.  Those vectors are written into the *prompt slot* of the
OmniVoice input: after the text, before the masked target frames.  They replace the
reference-audio prompt (the real audio is not used as a prompt: `prompt_length = 0`).

Design: /home/brandon/.cache/orch/tts-finetune-omnivoice-2026-11-13/workers/research/conditioning.md
Nothing in upstream `omnivoice` is patched; this file only subclasses `OmniVoice`.

Needs the `omnivoice` package (0.2.x), torch and transformers>=5.3.  See README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceConfig,
    OmniVoiceModelOutput,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class StyleOmniVoiceConfig(OmniVoiceConfig):
    """`OmniVoiceConfig` plus the style-encoder fields.

    A base OmniVoice `config.json` (no style fields) loads into this class with the defaults below.
    """

    model_type = "omnivoice_style"

    def __init__(
        self,
        style_n_vectors: int = 4,  # prefix length n (vectors written into the prompt slot)
        style_enc_dim: int = 128,  # encoder width
        style_enc_layers: int = 2,
        style_enc_heads: int = 4,
        style_code_dim: int = 16,  # bottleneck width for "vq" / "none" (FSQ uses len(style_fsq_levels))
        style_codebook_size: int = 64,  # VQ codebook size (only for style_quantizer="vq")
        # "none": continuous tight bottleneck: z is unit-RMS per vector, no quantiser.  Shipped default:
        #         the FSQ route failed check (b) on 2 of 5 seeds even after the fix sweep
        #         (`workers/smoke/fsq-fix-report.md`), while the continuous route passes every gate on
        #         5 of 5 seeds at phase-2 lr 5e-4 (`workers/smoke/contin-fix-report.md`).
        # "fsq": finite scalar quantisation (Mentzer et al. 2023, arXiv 2309.15505): each of the d code
        #        dimensions is bounded with floor(L/2)*tanh(z), rounded (straight-through), no auxiliary
        #        loss, no reseeding.  Kept selectable; seed-fragile check (b).
        # "vq" : learned codebook (kept to document its collapse in the smoke test).
        style_quantizer: str = "none",
        style_fsq_levels: Optional[list] = None,  # odd levels per dimension, paper advice L_i >= 5, d < 10
        # FSQ bound.  "tanh" is the paper's bound and saturates (its slope goes to 0 in the corners, so a style
        # stuck on an outermost level gets no straight-through gradient).  On the [7, 7, 5, 5] grid it passed
        # the three previously failing screened seeds (1, 2, 4) but still failed the five-seed confirmation on
        # seeds 0 and 3 (3/5).  "clip" is a scaled linear bound whose value is clipped but whose gradient passes
        # straight through the clip, and "softsign" is z/(1+|z|), bounded with a gradient that stays positive.
        # `sweep_fsq.py` has the evidence; `workers/smoke/fsq-fix-report.md` has the five-seed table.
        style_fsq_bound: str = "tanh",
        style_fsq_scale: float = 1.0,  # z multiplier before the bound (>1 saturates sooner, <1 keeps z in range)
        style_code_init_std: Optional[float] = None,  # FSQ to_code init (None = fan-in); small keeps z off the corners
        style_noise_std: float = 0.1,  # Gaussian noise on the code during training
        style_commit_beta: float = 0.25,  # VQ commitment weight (codebook loss weight is 1)
        style_scale: float = 1.0,  # default prefix-strength multiplier (training and inference)
        style_commit_delay_steps: int = 0,  # commitment weight is 0 for this many train steps ...
        style_commit_warmup_steps: int = 200,  # ... then ramps 0 -> beta over this many more
        style_dead_code_reset: bool = True,  # VQ upkeep: re-seed codes that stopped being used
        # Output MLP of the encoder: init std of its two Linears (None = fan-in scaling, in_features**-0.5) and the
        # eps of the LayerNorm that fixes the vector norm.  See README "Output-layer init" for why both exist.
        style_out_init_std: Optional[float] = 0.02,
        style_out_ln_eps: float = 1e-10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.style_n_vectors = style_n_vectors
        self.style_enc_dim = style_enc_dim
        self.style_enc_layers = style_enc_layers
        self.style_enc_heads = style_enc_heads
        self.style_code_dim = style_code_dim
        self.style_codebook_size = style_codebook_size
        assert style_quantizer in ("fsq", "vq", "none"), style_quantizer
        self.style_quantizer = style_quantizer
        levels = list(style_fsq_levels) if style_fsq_levels is not None else [7, 7, 5, 5]
        assert all(l >= 3 and l % 2 == 1 for l in levels), "FSQ levels must be odd (symmetric bound)"
        self.style_fsq_levels = levels
        assert style_fsq_bound in ("tanh", "clip", "softsign"), style_fsq_bound
        self.style_fsq_bound = style_fsq_bound
        self.style_fsq_scale = style_fsq_scale
        self.style_code_init_std = style_code_init_std
        self.style_noise_std = style_noise_std
        self.style_commit_beta = style_commit_beta
        self.style_scale = style_scale
        self.style_commit_delay_steps = style_commit_delay_steps
        self.style_commit_warmup_steps = style_commit_warmup_steps
        self.style_dead_code_reset = style_dead_code_reset
        self.style_out_init_std = style_out_init_std
        self.style_out_ln_eps = style_out_ln_eps


# ---------------------------------------------------------------------------
# Style encoder
# ---------------------------------------------------------------------------


def _sinusoidal(length: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
    return pe.to(dtype)


class _Attention(nn.Module):
    """Plain multi-head attention (self or cross) with a key-validity mask.

    Written with plain `nn.Linear` layers so that every parameter is covered by
    `StyleEncoder.reset_parameters` (transformers 5 materialises missing keys from the meta device).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, kv, kv_valid):
        B, Lq, D = x.shape
        Lk = kv.shape[1]
        h = self.heads

        def split(t, L):
            return t.view(B, L, h, D // h).transpose(1, 2)

        q, k, v = split(self.q(x), Lq), split(self.k(kv), Lk), split(self.v(kv), Lk)
        mask = kv_valid[:, None, None, :]  # [B,1,1,Lk] bool, True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Lq, D))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, valid):
        h = self.n1(x)
        x = x + self.attn(h, h, valid)
        return x + self.mlp(self.n2(x))


@dataclass
class StyleEncoderOutput:
    vectors: torch.Tensor  # [B, n, hidden]
    codes: Optional[torch.Tensor]  # [B, n] long (FSQ: flat mixed-radix index), None for "none"
    aux_loss: torch.Tensor  # codebook + commitment (scalar, 0 without VQ)
    metrics: Dict[str, torch.Tensor]  # detached scalars for logging


class StyleEncoder(nn.Module):
    """Audio-token crop -> n style vectors through a tight bottleneck."""

    def __init__(self, cfg: StyleOmniVoiceConfig, hidden_size: int):
        super().__init__()
        self.n = cfg.style_n_vectors
        self.quantizer = cfg.style_quantizer
        self.use_vq = self.quantizer == "vq"
        self.use_fsq = self.quantizer == "fsq"
        self.fsq_levels = list(cfg.style_fsq_levels)
        self.fsq_bound = cfg.style_fsq_bound
        self.fsq_scale = cfg.style_fsq_scale
        self.code_init_std = cfg.style_code_init_std
        code_dim = len(self.fsq_levels) if self.use_fsq else cfg.style_code_dim
        self.code_dim = code_dim
        # Constants are plain Python values turned into tensors on use.  Non-persistent buffers are NOT restored by
        # `from_pretrained` in transformers 5 (they come back uninitialised), so none are used here.
        self._fsq_half = [l // 2 for l in self.fsq_levels]
        self._fsq_radix = [1]
        for l in self.fsq_levels[:-1]:
            self._fsq_radix.append(self._fsq_radix[-1] * l)
        self.n_codes = math.prod(self.fsq_levels) if self.use_fsq else cfg.style_codebook_size
        self.noise_std = cfg.style_noise_std
        self.out_init_std = cfg.style_out_init_std
        self.out_ln_eps = cfg.style_out_ln_eps
        self.dead_reset = cfg.style_dead_code_reset
        self.commit_warmup = cfg.style_commit_warmup_steps
        self.commit_delay = cfg.style_commit_delay_steps
        self.beta = cfg.style_commit_beta
        self.hidden_size = hidden_size
        d = cfg.style_enc_dim
        C, V = cfg.num_audio_codebook, cfg.audio_vocab_size
        self.audio_vocab_size = V
        self.audio_mask_id = cfg.audio_mask_id

        self.embed = nn.Embedding(C * V, d)
        self.blocks = nn.ModuleList([_Block(d, cfg.style_enc_heads) for _ in range(cfg.style_enc_layers)])
        self.queries = nn.Parameter(torch.zeros(self.n, d))
        self.pool_norm = nn.LayerNorm(d)
        self.pool = _Attention(d, cfg.style_enc_heads)
        self.to_code_norm = nn.LayerNorm(d)
        self.to_code = nn.Linear(d, code_dim)
        # Only the VQ path owns a codebook: a parameter that never receives a gradient would break DDP
        # (find_unused_parameters=False).
        self.codebook = nn.Embedding(cfg.style_codebook_size, code_dim) if self.use_vq else None
        self.to_out = nn.Sequential(nn.Linear(code_dim, d), nn.GELU(), nn.Linear(d, hidden_size))
        # Output is layer-normalised (norm = sqrt(hidden)) then multiplied by this learned scalar.
        self.out_scale = nn.Parameter(torch.ones(()))
        # 0 until `StyleOmniVoice.ensure_style_scale_calibrated` matched out_scale to the real audio
        # embedding norm; saved in checkpoints so a resume never recalibrates.
        self.register_buffer("scale_calibrated", torch.zeros(()))
        # VQ upkeep state (saved with the checkpoint): codebook seeded from data on the first training
        # step, and an EMA of per-code usage used to re-seed codes that died.
        self.register_buffer("codebook_inited", torch.zeros(()))
        # Training steps seen: ramps the commitment weight.  Without the ramp Adam makes the always-consistent
        # commitment gradient win against the still-noisy CE gradient and every clip collapses onto one code.
        self.register_buffer("vq_steps", torch.zeros(()))
        self.register_buffer("code_usage", torch.zeros(cfg.style_codebook_size if self.use_vq else 1))

    @torch.no_grad()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.0)  # O(1) so the clip content is visible from step 0
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if self.use_fsq:
            # Fan-in (style_code_init_std=None) is the shipped default on the [7, 7, 5, 5] grid; a small init
            # keeps z in the linear part of the bound instead of on the outer levels.
            std = self.to_code.in_features ** -0.5 if self.code_init_std is None else self.code_init_std
            nn.init.normal_(self.to_code.weight, std=std)
        # Output MLP init.  The Linear loop above used std 0.02; fan-in scaling (style_out_init_std=None) gives a
        # unit-variance output, a fixed std overrides it.  Measured in the smoke report: it changes FSQ training.
        for lin in (self.to_out[0], self.to_out[2]):
            std = lin.in_features ** -0.5 if self.out_init_std is None else self.out_init_std
            nn.init.normal_(lin.weight, std=std)
        nn.init.normal_(self.queries, std=1.0)
        if self.codebook is not None:
            nn.init.normal_(self.codebook.weight, std=1.0)  # same scale as the unit-RMS z
        # transformers 5 calls this AFTER loading checkpoint weights and only guards the `nn.init.*` functions
        # (they skip tensors it already loaded, flagged `_is_hf_initialized`).  Direct fills are not guarded, so
        # skip the loaded ones here or a save/load round trip would reset the calibrated scale and counters.
        for t, v in (
            (self.out_scale, 1.0), (self.scale_calibrated, 0.0), (self.codebook_inited, 0.0),
            (self.vq_steps, 0.0), (self.code_usage, 0.0),
        ):
            if not getattr(t, "_is_hf_initialized", False):
                t.fill_(v)

    @torch.no_grad()
    def _vq_upkeep(self, z: torch.Tensor, codes: torch.Tensor):
        """Seed the codebook from the first batch's codes; afterwards re-seed codes whose usage EMA died."""
        flat = z.reshape(-1, z.shape[-1])
        K = self.codebook.num_embeddings
        pick = lambda m: flat[torch.randint(0, flat.shape[0], (m,), device=flat.device)]  # noqa: E731
        if float(self.codebook_inited) == 0.0:
            self.codebook.weight.copy_(pick(K) + 0.01 * torch.randn(K, flat.shape[-1], device=flat.device))
            self.codebook_inited.fill_(1.0)
            self.code_usage.fill_(1.0)
            return
        hist = torch.bincount(codes.flatten(), minlength=K).float()
        self.code_usage.mul_(0.99).add_(hist)
        dead = self.code_usage < 0.03
        if self.dead_reset and bool(dead.any()):
            m = int(dead.sum())
            self.codebook.weight[dead] = pick(m) + 0.01 * torch.randn(m, flat.shape[-1], device=flat.device)
            self.code_usage[dead] = self.code_usage.mean().clamp(min=1.0)

    def _offsets(self, device) -> torch.Tensor:
        return torch.arange(self.embed.num_embeddings // self.audio_vocab_size, device=device) * self.audio_vocab_size

    def _half(self, device) -> torch.Tensor:
        return torch.tensor(self._fsq_half, dtype=torch.float32, device=device)

    def _radix(self, device) -> torch.Tensor:
        return torch.tensor(self._fsq_radix, dtype=torch.long, device=device)

    def _encode(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """tokens [B,C,T] long (no mask ids), valid [B,T] bool -> z [B,n,code_dim]."""
        ids = tokens.clamp(0, self.audio_vocab_size - 1) + self._offsets(tokens.device).view(1, -1, 1)
        x = self.embed(ids).sum(1)  # [B,T,d]
        x = x + _sinusoidal(x.shape[1], x.shape[2], x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, valid)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        pooled = self.pool(q, self.pool_norm(x), valid)  # [B,n,d]; no query residual: it is clip-independent
        z = self.to_code(self.to_code_norm(pooled))
        if self.use_fsq:
            return z.float()  # bounded and rounded in `_fsq`
        # Unit-RMS per vector: the commitment loss can no longer shrink the whole code space to a point.
        return F.layer_norm(z.float(), (z.shape[-1],))

    def _bounded(self, z: torch.Tensor) -> torch.Tensor:
        """Map the pre-quantisation code z onto the FSQ bound (values in [-half, half] per dimension).

        `tanh` is the paper's saturating bound.  `clip` is a scaled linear bound: the value is clipped to
        the bound but the gradient is passed straight through the clip (the usual straight-through trick),
        so a style pushed onto an outer level can still be moved back.  `softsign` never zeroes the
        gradient at all.  See README "Known defect" and the sweep report for the measurements.
        """
        u = z.float() * self.fsq_scale
        half = self._half(z.device)
        if self.fsq_bound == "tanh":
            return half * torch.tanh(u)
        if self.fsq_bound == "softsign":
            return half * u / (1.0 + u.abs())
        clipped = u.clamp(-1.0, 1.0)
        return half * (u + (clipped - u).detach())  # value clipped, gradient 1 through the clip

    def _fsq(self, z: torch.Tensor):
        """FSQ: round the bounded z per dimension, straight-through; returns (zq in [-1,1], flat code index)."""
        half = self._half(z.device)
        b = self._bounded(z)
        q = b + (torch.round(b) - b).detach()  # STE: gradient of the rounding is 1
        idx = ((torch.round(b.detach()) + half).long() * self._radix(z.device)).sum(-1)
        return q / half, idx

    def codes_to_z(self, codes: torch.Tensor) -> torch.Tensor:
        """Flat FSQ code index [B,n] -> quantised vector in [-1,1]^d."""
        digits = []
        for l, r in zip(self.fsq_levels, self._fsq_radix):
            digits.append((codes // r) % l)
        half = self._half(codes.device)
        q = torch.stack(digits, -1).float() - half
        return q / half

    def _decode(self, zq: torch.Tensor) -> torch.Tensor:
        out = self.to_out(zq)
        out = F.layer_norm(out, (out.shape[-1],), eps=self.out_ln_eps)
        return out * self.out_scale

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> StyleEncoderOutput:
        z = self._encode(tokens, valid).float()  # bottleneck in fp32
        zero = z.new_zeros(())
        codes = None
        if self.use_vq:
            cb = self.codebook.weight.float()  # [K,code_dim]
            dist = (z.unsqueeze(-2) - cb).pow(2).sum(-1)  # [B,n,K]
            codes = dist.argmin(-1)  # [B,n]
            if self.training:
                self._vq_upkeep(z.detach(), codes)
                dist = (z.unsqueeze(-2) - self.codebook.weight.float()).pow(2).sum(-1)
                codes = dist.argmin(-1)
            zq = self.codebook(codes).float()
            codebook_loss = F.mse_loss(zq, z.detach())
            commit_loss = F.mse_loss(z, zq.detach())
            since = float(self.vq_steps) - self.commit_delay
            ramp = min(1.0, max(0.0, since / self.commit_warmup)) if self.commit_warmup > 0 else float(since >= 0)
            if self.training:
                self.vq_steps += 1
            aux = codebook_loss + self.beta * ramp * commit_loss
            zq_st = z + (zq - z).detach()  # straight-through
        elif self.use_fsq:
            codebook_loss = commit_loss = zero
            aux = zero
            zq_st, codes = self._fsq(z)
            ramp = 0.0
        else:
            codebook_loss = commit_loss = zero
            aux = zero
            zq_st = z
            ramp = 0.0
        if self.training and self.noise_std > 0:
            zq_st = zq_st + self.noise_std * torch.randn_like(zq_st)
        vectors = self._decode(zq_st.to(self.to_out[0].weight.dtype))

        with torch.no_grad():
            metrics = {
                "style/commit_loss": commit_loss.detach(),
                "style/commit_ramp": z.new_tensor(ramp),
                "style/codebook_loss": codebook_loss.detach(),
                # spread of the pre-quantisation code across the batch (0 = every clip looks the same)
                "style/z_spread": z.detach().std(dim=0).mean() if z.shape[0] > 1 else zero,
                "style/out_rms": vectors.detach().float().pow(2).mean().sqrt(),
            }
            if codes is not None:
                K = self.n_codes
                hist = torch.bincount(codes.flatten(), minlength=K).float()
                p = hist / hist.sum().clamp(min=1)
                perplexity = torch.exp(-(p * torch.log(p.clamp(min=1e-10))).sum())
                metrics["style/codes_used"] = (hist > 0).sum().float()
                metrics["style/code_perplexity"] = perplexity
                # distinct codes per slot, averaged over the n slots
                metrics["style/codes_used_per_slot"] = torch.stack(
                    [codes[:, s].unique().numel() * torch.ones((), device=codes.device) for s in range(codes.shape[1])]
                ).mean()
        return StyleEncoderOutput(vectors=vectors, codes=codes, aux_loss=aux, metrics=metrics)

    @torch.no_grad()
    def encode_z(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Bottleneck code z [B,n,code_dim] before quantisation (no noise)."""
        return self._encode(tokens, valid)

    def vectors_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck codes [B,n,code_dim] (or [n,code_dim]) into prefix vectors (works with or without VQ)."""
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return self._decode(z.float().to(self.to_out[0].weight.dtype))

    def vectors_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode code indices [B,n] (or [n]) into prefix vectors, with no noise."""
        if not (self.use_vq or self.use_fsq):
            raise ValueError("vectors_from_codes needs style_quantizer='fsq' or 'vq'")
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        zq = self.codes_to_z(codes) if self.use_fsq else self.codebook(codes)
        return self._decode(zq)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class StyleModelOutput(OmniVoiceModelOutput):
    ce_loss: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None


class StyleOmniVoice(OmniVoice):
    config_class = StyleOmniVoiceConfig

    def __init__(self, config: StyleOmniVoiceConfig, llm=None):
        super().__init__(config, llm)
        self.style_encoder = StyleEncoder(config, config.llm_config.hidden_size)
        # Runtime training options (set with `configure_training`; not part of the saved model).
        self.kl_weight = 0.0
        self.style_aux_weight = 1.0
        self.metrics_sink: Optional[Callable[[dict], None]] = None
        # The frozen reference is deliberately NOT a registered submodule: it must not appear in
        # state_dict(), save_pretrained, accelerate state or DDP.  A plain list hides it from nn.Module.
        self._ref_holder: List[nn.Module] = []
        # Re-run post_init so the encoder created after `OmniVoice.__init__` is initialised too.
        self.post_init()

    # -- init ----------------------------------------------------------------

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, StyleEncoder):
            module.reset_parameters()

    @torch.no_grad()
    def ensure_style_scale_calibrated(self, target_norm: Optional[float] = None, n_samples: int = 4096):
        """Match the encoder output norm to the norm of a real audio-position embedding.

        A real audio position is a sum of 8 codebook embeddings.  Its mean norm is measured from the
        current `audio_embeddings` over random codes (or `target_norm` if given).  Runs once: the
        `scale_calibrated` buffer is saved in checkpoints, so resumed/trained encoders are untouched.
        Call it after loading the base weights.  Returns the target norm used (or None if skipped).
        """
        enc = self.style_encoder
        if float(enc.scale_calibrated) != 0.0:
            return None
        if target_norm is None:
            C, V = self.config.num_audio_codebook, self.config.audio_vocab_size
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(0, V - 1, (n_samples, C), generator=g).to(self.codebook_layer_offsets.device)
            emb = self.audio_embeddings(ids + self.codebook_layer_offsets.view(1, -1)).sum(1)
            target_norm = float(emb.float().norm(dim=-1).mean())
        enc.out_scale.fill_(target_norm / math.sqrt(enc.hidden_size))
        enc.scale_calibrated.fill_(1.0)
        return target_norm

    def configure_training(self, kl_weight: float = 0.0, style_aux_weight: float = 1.0, metrics_sink=None):
        self.kl_weight = kl_weight
        self.style_aux_weight = style_aux_weight
        self.metrics_sink = metrics_sink

    # -- frozen reference for the KL term -------------------------------------

    def attach_reference(self, ref: OmniVoice, dtype: Optional[torch.dtype] = None):
        """Attach a frozen original model for the KL term (see `_kl_to_reference`)."""
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        if dtype is not None:
            ref.to(dtype)
        self._ref_holder = [ref]

    @staticmethod
    def reference_from(model: "StyleOmniVoice") -> OmniVoice:
        """Frozen snapshot of the current (not yet trained) base weights, as a plain OmniVoice."""
        llm_cfg = model.config.llm_config
        ref = OmniVoice(
            OmniVoiceConfig(
                audio_vocab_size=model.config.audio_vocab_size,
                audio_mask_id=model.config.audio_mask_id,
                num_audio_codebook=model.config.num_audio_codebook,
                audio_codebook_weights=model.config.audio_codebook_weights,
                llm_config=llm_cfg,
            )
        )
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("style_encoder.")}
        missing, unexpected = ref.load_state_dict(sd, strict=False)
        assert not unexpected and not [k for k in missing if k != "codebook_layer_offsets"], (missing, unexpected)
        return ref.to(next(model.parameters()).device)

    def _kl_to_reference(self, logits, labels, input_ids, audio_mask, attention_mask, style_slot, kl_mask):
        """KL(p_ref || p_model) at masked (label != -100) positions, codebook-weighted like the CE.

        The reference is the original model.  It never saw the style slot, so the slot positions are
        removed from its view (key columns masked out, positions renumbered).  Only the masked
        positions go through the output head and the softmax.
        """
        ref = self._ref_holder[0]
        dev = logits.device
        if next(ref.parameters()).device != dev:
            ref.to(dev)
        B, C, S, V = logits.shape
        keep = ~style_slot if style_slot is not None else torch.ones(B, S, dtype=torch.bool, device=dev)
        base_mask = attention_mask if attention_mask is not None else torch.ones(B, 1, S, S, dtype=torch.bool, device=dev)
        ref_mask = base_mask & keep[:, None, None, :]
        ref_pos = (keep.long().cumsum(-1) - 1).clamp(min=0)

        sel = labels != -100  # [B,C,S]
        if kl_mask is not None:
            sel = sel & kl_mask.view(B, 1, 1)
        pos_any = sel.any(1)  # [B,S]
        if not bool(pos_any.any()):
            return logits.new_zeros(())

        with torch.no_grad():
            ref_dtype = next(ref.parameters()).dtype
            embeds = ref._prepare_embed_inputs(input_ids, audio_mask).to(ref_dtype)
            hidden = ref.llm(inputs_embeds=embeds, attention_mask=ref_mask, position_ids=ref_pos, return_dict=True)[0]
            ref_logits = ref.audio_heads(hidden[pos_any]).view(-1, C, V).float()  # [M,C,V]
            ref_lp = F.log_softmax(ref_logits, dim=-1)
        sel_m = sel.permute(0, 2, 1)[pos_any]  # [M,C]
        mdl_logits = logits.permute(0, 2, 1, 3)[pos_any].float()  # [M,C,V]
        mdl_lp = F.log_softmax(mdl_logits, dim=-1)
        kl = (ref_lp.exp() * (ref_lp - mdl_lp)).sum(-1)  # [M,C]
        layer_means = (kl * sel_m).sum(0) / sel_m.sum(0).clamp(min=1).float()
        w = torch.tensor(self.normalized_audio_codebook_weights, device=dev)
        return (layer_means * w).sum()

    # -- forward -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        style_slot: Optional[torch.Tensor] = None,  # [B,L] bool, n True per sample that uses the prefix
        style_tokens: Optional[torch.LongTensor] = None,  # [B,C,Tc] crop for the encoder
        style_valid: Optional[torch.Tensor] = None,  # [B,Tc] bool
        style_vectors: Optional[torch.Tensor] = None,  # [B,n,H] given directly (inference)
        style_scale=None,  # float or [B]; None -> config.style_scale
        kl_mask: Optional[torch.Tensor] = None,  # [B] bool: samples that take part in the KL term
    ):
        if document_ids is not None and attention_mask is None:
            raise NotImplementedError(
                "StyleOmniVoice supports the padding collator (sdpa) only; flex-attention packing is not implemented"
            )
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

        aux = inputs_embeds.new_zeros((), dtype=torch.float32)
        zero_dep = None
        metrics: Dict[str, torch.Tensor] = {}
        vecs = None
        if style_tokens is not None:
            # The encoder runs on EVERY step, also when every sample dropped its style, so that every
            # encoder parameter is in the autograd graph (DDP find_unused_parameters=False).
            if style_valid is None:
                style_valid = torch.ones(style_tokens.shape[0], style_tokens.shape[2], dtype=torch.bool, device=style_tokens.device)
            enc = self.style_encoder(style_tokens, style_valid)
            vecs, aux, metrics = enc.vectors, enc.aux_loss, enc.metrics
        elif style_vectors is not None:
            vecs = style_vectors
        if vecs is not None:
            scale = self.config.style_scale if style_scale is None else style_scale
            if torch.is_tensor(scale) and scale.dim() == 1:
                scale = scale.view(-1, 1, 1)
            vecs = vecs * scale
            zero_dep = vecs.sum() * 0.0  # keeps out_scale/to_out in the graph when no sample has a slot
            if style_slot is not None and bool(style_slot.any()):
                has = style_slot.any(-1)
                n = self.config.style_n_vectors
                if not bool((style_slot.sum(-1)[has] == n).all()):
                    raise ValueError(f"every sample with a style slot needs exactly {n} slot positions")
                inputs_embeds = inputs_embeds.masked_scatter(
                    style_slot.unsqueeze(-1),
                    vecs[has].reshape(-1, vecs.shape[-1]).to(inputs_embeds.dtype),
                )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        hidden_states = llm_outputs[0]
        B, S, _ = hidden_states.shape
        logits = (
            self.audio_heads(hidden_states)
            .view(B, S, self.config.num_audio_codebook, self.config.audio_vocab_size)
            .permute(0, 2, 1, 3)
        )

        loss = ce = kl = None
        if labels is not None:
            per_token = F.cross_entropy(logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100)
            valid = (labels != -100).float()
            layer_means = (per_token * valid).sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp(min=1.0)
            w = torch.tensor(self.normalized_audio_codebook_weights, device=logits.device)
            ce = (layer_means * w).sum()
            loss = ce
            if self.style_aux_weight:
                loss = loss + self.style_aux_weight * aux.to(loss.dtype)
            if zero_dep is not None:
                loss = loss + zero_dep.to(loss.dtype)
            if self.kl_weight > 0 and self._ref_holder and self.training:
                kl = self._kl_to_reference(logits, labels, input_ids, audio_mask, attention_mask, style_slot, kl_mask)
                loss = loss + self.kl_weight * kl
            if self.metrics_sink is not None and self.training:
                row = {k: float(v) for k, v in metrics.items()}
                row["loss/ce"] = float(ce.detach())
                row["loss/aux"] = float(aux.detach())
                if kl is not None:
                    row["loss/kl"] = float(kl.detach())
                self.metrics_sink(row)
        return StyleModelOutput(loss=loss, logits=logits, ce_loss=ce, kl_loss=kl, aux_loss=aux)

    # -- inference -----------------------------------------------------------------

    @torch.no_grad()
    def _branch_logits(self, text_ids, vecs, target_tokens, style_scale):
        """Logits [C,T,V] for the target frames of one branch: text (+ optional prefix slot) + target."""
        C = self.config.num_audio_codebook
        dev = target_tokens.device
        T = target_tokens.shape[1]
        parts, amask, slot = [], [], []
        if text_ids is not None:
            Lt = text_ids.numel()
            parts.append(text_ids.to(dev).view(1, -1).expand(C, -1))
            amask.append(torch.zeros(Lt, dtype=torch.bool, device=dev))
            slot.append(torch.zeros(Lt, dtype=torch.bool, device=dev))
        if vecs is not None:
            n = vecs.shape[-2]
            parts.append(torch.full((C, n), self.config.audio_mask_id, dtype=torch.long, device=dev))
            amask.append(torch.ones(n, dtype=torch.bool, device=dev))
            slot.append(torch.ones(n, dtype=torch.bool, device=dev))
        parts.append(target_tokens)
        amask.append(torch.ones(T, dtype=torch.bool, device=dev))
        slot.append(torch.zeros(T, dtype=torch.bool, device=dev))
        ids = torch.cat(parts, dim=1).unsqueeze(0)
        am = torch.cat(amask).unsqueeze(0)
        sl = torch.cat(slot).unsqueeze(0)
        L = ids.shape[-1]
        out = self(
            input_ids=ids,
            audio_mask=am,
            attention_mask=torch.ones(1, 1, L, L, dtype=torch.bool, device=dev),
            style_slot=sl if vecs is not None else None,
            style_vectors=vecs.view(1, *vecs.shape[-2:]) if vecs is not None else None,
            style_scale=style_scale,
        )
        return out.logits[0, :, -T:, :].float()

    @torch.no_grad()
    def generate_tokens(
        self,
        text_ids: Optional[torch.Tensor],
        target_len: int,
        style_vectors: Optional[torch.Tensor] = None,  # [n,H]
        style_scale: Optional[float] = None,  # prefix strength knob #1: scale the prefix vectors
        num_step: int = 16,
        guidance_text: float = 2.0,
        guidance_style: float = 2.0,  # prefix strength knob #2: 3-branch CFG weight on (full - text-only)
        class_temperature: float = 0.0,
        layer_penalty_factor: float = 5.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Iterative masked decoding for one clip with 3-branch classifier-free guidance.

        Branches: full (text + prefix), text-only (text), nothing (target slots only).
            log p = log_softmax(c + g_text * (t - u) + g_style * (c - t))
        With g_text == g_style == g this is exactly upstream's `c + g * (c - u)` (identity checked in the
        smoke test).  With no prefix given, c == t, which reduces to upstream's two-branch CFG on text.
        `guidance_style = 0` keeps the prefix in the full branch but drops its CFG push; a negative
        value pushes away from the prefix.  Linear unmasking schedule (upstream uses a shifted one).
        Returns tokens [C, target_len].
        """
        C = self.config.num_audio_codebook
        dev = next(self.parameters()).device
        mask_id = self.config.audio_mask_id
        tokens = torch.full((C, target_len), mask_id, dtype=torch.long, device=dev)
        total = C * target_len
        per_step = [math.ceil(total * (s + 1) / num_step) - math.ceil(total * s / num_step) for s in range(num_step)]
        layer_ids = torch.arange(C, device=dev).view(C, 1)
        was_training = self.training
        self.eval()
        try:
            for step in range(num_step):
                k = per_step[step]
                if k <= 0:
                    continue
                lp_c = F.log_softmax(self._branch_logits(text_ids, style_vectors, tokens, style_scale), -1)
                has_style = style_vectors is not None
                need_t = guidance_text != 0 or (has_style and guidance_style != 0)
                lp_t = (
                    F.log_softmax(self._branch_logits(text_ids, None, tokens, None), -1)
                    if (need_t and has_style)
                    else lp_c
                )
                if guidance_text != 0:
                    lp_u = F.log_softmax(self._branch_logits(None, None, tokens, None), -1)
                    comb = lp_c + guidance_text * (lp_t - lp_u)
                else:
                    comb = lp_c
                if has_style and guidance_style != 0:
                    comb = comb + guidance_style * (lp_c - lp_t)
                lp = torch.log_softmax(comb, dim=-1)
                lp[..., mask_id] = -float("inf")
                if class_temperature > 0:
                    g = -torch.log(-torch.log(torch.rand(lp.shape, generator=generator, device=dev).clamp(1e-10, 1 - 1e-10)))
                    pred = (lp / class_temperature + g).argmax(-1)
                else:
                    pred = lp.argmax(-1)
                scores = lp.max(-1)[0] - layer_ids * layer_penalty_factor
                scores = scores.masked_fill(tokens != mask_id, -float("inf"))
                _, idx = torch.topk(scores.flatten(), min(k, int((tokens == mask_id).sum())))
                flat = tokens.flatten()
                flat[idx] = pred.flatten()[idx]
                tokens = flat.view(C, target_len)
        finally:
            self.train(was_training)
        return tokens
