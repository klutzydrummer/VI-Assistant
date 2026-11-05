"""voicetune: a full fine-tune of Qwen3-TTS-12Hz-1.7B-Base that speaks as Tali with no reference clip.
(Stage 5 of the Tali voice plan; Brandon 2026-10-30: "Let's just do a full fine tune".) One model
for every mood, never one per mood. There's no strength knob: the app's mood state drives the voice
through the projector, which later learns to write the speaker slot.

Built on the official recipe (QwenLM/Qwen3-TTS finetuning/ at 022e286b, read 2026-10-30; notes in
docs/design/qwen3tts-finetune-guide.md), with its problems fixed:
  - alignment: at generation, the hidden state at position p predicts frame p+1's first code, and
    the code predictor gets that same hidden state with frame p+1's first code (talker.forward's
    past_hidden). The official sft_12hz.py passes codec_0_labels[:, 1:] to a loss that shifts
    again (each frame learns the frame two ahead: "speech speeds up every epoch") and pairs the
    hidden state at p with frame p's codes. Here the labels go in unshifted (`train_labels`) and
    the code predictor gets the hidden state one position back (`predictor_hidden_mask`).
  - text input: generation runs text_embedding through talker.text_projection; training does too
    here (`forward`), so the talker sees the text the generator does.
  - sub-talker loss: forward_sub_talker_finetune's own loss shifts labels again (codebook k scored
    against codebook k+1); here its logits are scored against codebook k+1 directly (`forward`),
    the pairing the code predictor is fed at generation.
  - attention: sdpa, not the hard-coded flash_attention_2.
  - speaker slot: each clip's OWN x-vector (`--speaker clip`, the default), not one fixed vector.
    Her x-vector moves with her mood (Brito 2026), so training on a fixed one would teach the model
    to ignore the slot the projector needs. The saved speaker `tali` is her mean x-vector
    (rescaled to the clips' average norm, as voicearms does). `--speaker mean` puts the mean in for
    every line instead.
  - label-less style (`--speaker style`, Brandon 2026-10-30: "what if the labels are not
    accurate?"): a small style encoder hears each clip's own audio and squeezes its delivery into
    `--style-dim` numbers, added to her mean x-vector in the slot, trained with the model (global
    style tokens / reference encoder, Wang et al. 2018). No labels anywhere: the state space comes
    from her clips, to be clustered and named by ear afterwards. The game goes in as its own
    learned vector, so the style numbers needn't carry each game's suit; `speak --game` picks it.
  - style extras (Brandon 2026-10-30: reward the delivery we want, penalise the rest; each off at 0):
    `--style-shuffle S` hands the encoder the clip cut into S-second chunks in a random order, so it can
    hear delivery but not read the words back (content leakage); chunks of ~1 s keep local rhythm.
    `--game-adv W` adds a game classifier on the style code behind gradient reversal, so the code
    can't carry which game (suit) a clip is from; small W, since game and mood overlap (drunk is LE3).
    `--prosody-aux W` adds a head that predicts the clip's measured pitch range, loudness and pace
    from the code, so the code can't be ignored. `--style-dropout P` zeroes the code for a share P
    of lines, so her mean voice stays a good default. None of these exist at inference.
  - forgetting: `--kl` adds a KL-to-Base term on replay lines, i.e. lines Base itself spoke in
    OTHER voices (`replay`). Teacher-forcing on the frozen model's own samples estimates the KL to
    it (Context-Free Synthetic Data, arXiv 2505.13811), so this keeps reading skill and expressive
    range without pulling her own voice back toward Base's version of it. The term covers all 16
    codebooks — the talker logits plus the 15 residual code-predictor logits, the latter at the
    residual-CE weight — so the residual codebooks are held to Base too. lr 2e-6 (2e-5 gives noise
    and endless output); a held-out loss per epoch picks the checkpoint.

    python3 jobs/voicetune.py check   --out <folder>
    python3 jobs/voicetune.py replay  --sentences s.txt --voices <dir of 24 kHz wavs> --out <folder>
    python3 jobs/voicetune.py prepare --raw train_raw.jsonl --out <folder> [--device cpu]
    python3 jobs/voicetune.py train   --data train_with_codes.jsonl --val val_with_codes.jsonl --out <folder>
                                      [--replay replay_with_codes.jsonl --kl 0.1] [--epochs 10] [--max-steps N] [--resume | --init-from epoch-K --warmup 50]
    python3 jobs/voicetune.py speak   --run <folder> --sentences s.txt [--epoch K] [--icl-ref clip.wav]
                                      [--style-from her_clip.wav ...] [--delivery-from id=her_clip.wav ...]
                                      [--seed N] [--unblind --title T --question Q] [--game le3]

train_raw.jsonl (built at home): one line per clip, {"id", "audio" (24 kHz mono wav), "text",
"ref_audio" (the clip itself, for its own x-vector), "game", optional "language"}. `prepare` adds
codec codes (fits the home CPU: one ~0.7 GB tokenizer). `train` writes epoch-<k>/talker.safetensors
(~3.4 GB each; bring home only the chosen one), voice.json (her mean x-vector) and train_log.jsonl
(losses and peak GPU memory). `speak` makes the no-reference voice model from an epoch and writes a
blind hub list (clips.toml), with the Base model's clone from her clip beside it when --icl-ref is
given. Needs qwen-tts==0.1.1 (requirements-tune.txt) and, for training, a 48 GB GPU.
"""

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

import checkpoints

BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER = "tali"
SPEAKER_ID = 3000  # the codec-embedding row the official recipe gives a new speaker
SPEAKER_SLOT = 6  # input position that holds the speaker embedding
SR = 24000
SUB_TALKER_WEIGHT = 0.3  # official residual-CE weight; use the same weight for residual replay KL
REQUIRED = ["id", "audio", "text", "ref_audio"]
CHECK_LINE = "This is a quick check of the voice pipeline."  # a neutral line for check --generate
UNBLIND_ANSWERS = ["acceptable", "close", "not yet"]


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def validate(rows, sample_rate=None):
    """Problems with a train_raw/with_codes list, as strings (empty = fine). sample_rate(path) -> Hz."""
    bad = []
    seen = set()
    for n, r in enumerate(rows):
        miss = [k for k in REQUIRED if not r.get(k)]
        if miss:
            bad.append(f"line {n + 1}: missing {', '.join(miss)}")
            continue
        if r["id"] in seen:
            bad.append(f"line {n + 1}: duplicate id {r['id']}")
        seen.add(r["id"])
        if sample_rate is not None:
            for k in ("audio", "ref_audio"):
                # A row that already carries codec codes never re-opens its audio
                # (the loader reads `audio_codes`), so a missing audio file is fine
                # there. This is how replay lines ship: Base's spoken clips were
                # generated on an old worker and never came home, only their codes.
                if k == "audio" and r.get("audio_codes"):
                    continue
                sr = sample_rate(r[k])
                if sr != SR:
                    bad.append(f"line {n + 1}: {k} is {sr} Hz, needs {SR}")
    return bad


def train_labels(codec_0_labels):
    """The labels to hand the talker for inputs_embeds[:, :-1]. Its transformers causal-LM loss shifts
    labels by one itself, so they must line up with the inputs, not be pre-shifted (the official
    script's codec_0_labels[:, 1:] trains each frame on the frame two ahead)."""
    return codec_0_labels[:, :-1]


def predictor_hidden_mask(codec_mask):
    """Which hidden states (of inputs_embeds[:, :-1]) feed the code predictor, one per audio frame:
    the position just before each frame, whose output predicted that frame's first code."""
    return codec_mask[:, 1:]


def causal_label_mask(codec_0_labels):
    """Which returned talker logits have a causal-LM target after transformers shifts the labels.

    The talker scores logits[t] against train_labels(labels)[:, t + 1] itself, so position t has a
    CE target iff that shifted label is not -100; the last logit (t = T - 1) has none. Returns a
    bool mask shaped like out.logits[:, :, 0], i.e. [b, T - 1]: the shifted labels are [b, T - 2],
    padded with a trailing False for the final, untargeted logit."""
    shifted = train_labels(codec_0_labels)[:, 1:] != -100  # targets for t = 0 .. T - 3
    if hasattr(shifted, "new_zeros"):  # a torch tensor
        import torch

        return torch.cat([shifted, shifted.new_zeros(shifted.shape[0], 1)], dim=1)
    import numpy as np

    return np.concatenate([shifted, np.zeros((shifted.shape[0], 1), dtype=bool)], axis=1)


def mean_voice(vectors):
    """Mean of x-vectors, rescaled to their average length (a mean of vectors is shorter than they are)."""
    import numpy as np

    x = np.asarray(vectors, dtype=np.float64)
    m = x.mean(0)
    return m * np.linalg.norm(x, axis=1).mean() / max(np.linalg.norm(m), 1e-12)


def length_bands(rows, key="duration_s"):
    """Clip counts and hours by length band, for the report (short lines matter: issue #343)."""
    bands = {"<2 s": 0, "2-5 s": 0, "5-10 s": 0, "10+ s": 0}
    hours = dict.fromkeys(bands, 0.0)
    for r in rows:
        d = float(r.get(key, 0))
        b = "<2 s" if d < 2 else "2-5 s" if d < 5 else "5-10 s" if d < 10 else "10+ s"
        bands[b] += 1
        hours[b] += d / 3600
    return {b: {"clips": bands[b], "hours": round(hours[b], 3)} for b in bands}


def pad_mels(mels):
    """[T_i, 128] mels -> ([b, T, 128] zero-padded, [b] lengths) (torch)."""
    import torch

    n = torch.tensor([m.shape[0] for m in mels])
    out = torch.zeros(len(mels), int(n.max()), mels[0].shape[1])
    for i, m in enumerate(mels):
        out[i, : m.shape[0]] = m
    return out, n


def shuffle_chunks(mel, length, chunk, gen=None):
    """mel [T, d] (first `length` frames real) cut into `chunk`-frame pieces, put back in a random order (torch)."""
    import torch

    n = int(length)
    if chunk <= 0 or n <= chunk:
        return mel
    pieces = list(mel[:n].split(chunk))
    order = torch.randperm(len(pieces), generator=gen).tolist()
    out = mel.clone()
    out[:n] = torch.cat([pieces[i] for i in order])
    return out


def grad_reverse(x, scale=1.0):
    """Identity forward, gradient times -scale backward (torch)."""
    import torch

    class _Rev(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.view_as(t)

        @staticmethod
        def backward(ctx, g):
            return -scale * g

    return _Rev.apply(x)


def prosody_targets(path, text):
    """[f0 range in semitones (10-90th pct), mean loudness dB, loudness spread dB, words per second] of a clip."""
    import librosa
    import numpy as np

    wav, sr = librosa.load(path, sr=16000, mono=True)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=320)[0]
    voiced = rms > max(1e-4, 0.1 * float(rms.max(initial=0)))
    f0 = librosa.yin(wav, fmin=75, fmax=600, sr=sr, frame_length=1024, hop_length=320)
    k = min(len(f0), len(voiced))
    f0 = f0[:k][voiced[:k]]
    semis = 12 * np.log2(f0 / 100.0) if len(f0) else np.array([0.0])
    db = 20 * np.log10(rms + 1e-9)
    dur = max(len(wav) / sr, 1e-3)
    return [float(np.percentile(semis, 90) - np.percentile(semis, 10)), float(db.mean()), float(db.std()),
            len(text.split()) / dur]


def kl_to_base(base_logits, logits, mask):
    """Mean KL(base || tuned) over the masked positions, from raw logits [b, t, v] (torch)."""
    import torch.nn.functional as F

    lb = F.log_softmax(base_logits[mask].float(), -1)
    lt = F.log_softmax(logits[mask].float(), -1)
    return (lb.exp() * (lb - lt)).sum(-1).mean()


def replay_kl_term(base_logits, logits, base_sub, sub, codec_0_labels):
    """The replay KL-to-Base term over all 16 codebooks: the talker (codebook 0) logits at the
    positions the causal-LM loss scores, plus every residual code-predictor logit at the residual-CE
    weight. Pure (no model), so the weighting is testable without a GPU."""
    import torch

    mask = causal_label_mask(codec_0_labels).to(logits.device)
    main = kl_to_base(base_logits, logits, mask)
    residual_mask = torch.ones(sub.shape[:-1], dtype=torch.bool, device=sub.device)
    residual = kl_to_base(base_sub, sub, residual_mask)
    return main + SUB_TALKER_WEIGHT * residual


def delivery_clips(rows, seed, n, ref_key="ref_audio"):
    """The `n` clips for the delivery listening conditions, chosen by seed alone:
    `random.Random(seed).sample` over the manifest rows. Ignores every field but `id` and `ref_key`,
    so mood tags, labels and hand-labelled sets never steer the choice. Returns [(id, ref_path)] in
    the sample's order."""
    import random

    if n > len(rows):
        raise ValueError(f"delivery_clips: asked for {n} clips, manifest has {len(rows)} rows")
    return [(r["id"], r[ref_key]) for r in random.Random(seed).sample(rows, n)]


def check_abort(log_records, elapsed_s, mem_limit_gb=44.0, time_limit_s=6300.0):
    """The fresh run's abort rule, as one testable function. `log_records` is the list of per-step
    dicts train writes to train_log.jsonl (step, loss, peak_gb, ...); returns a reason to stop, or
    None to carry on. Aborts on: a step-1 loss above 4.0, a non-finite (NaN or ±Inf) loss, a
    windowed mean that has risen more than 20% over 50 steps (mean of the last 10 losses vs the 10
    losses ending 50 steps earlier, so one noisy batch can't trip it), peak memory above
    `mem_limit_gb` (44 GB), or `elapsed_s` over `time_limit_s` (1 h 45 m)."""
    if elapsed_s > time_limit_s:
        return f"elapsed {elapsed_s:.0f} s > {time_limit_s:.0f} s"
    losses = []
    for r in log_records:
        loss = r.get("loss")
        if loss is not None:
            if r.get("step") == 1 and loss > 4.0:
                return "step-1 loss > 4.0"
            if not math.isfinite(loss):  # NaN or ±Inf
                return "non-finite loss"
            losses.append(loss)
        peak = r.get("peak_gb")
        if peak is not None and peak > mem_limit_gb:
            return f"peak memory {peak} GB > {mem_limit_gb} GB"
    if len(losses) >= 60:
        recent = sum(losses[-10:]) / 10
        older = sum(losses[-60:-50]) / 10
        if recent > older * 1.2:
            return f"loss rising {100 * (recent / older - 1):.0f}% over 50 steps"
    return None


def parse_nvidia_smi_csv(text):
    """Parse
    `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_throttle_reasons.active --format=csv`
    (with or without the header) into one dict per GPU. A value nvidia-smi reports as `[N/A]` (a
    laptop GPU's power, for example) becomes None; the throttle-reasons field stays the raw string
    because it is a hex bitmask (`0x0000000000000000` means no reason is active)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("clocks.sm"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def number(s):
            # "1980 MHz" / "320.45 W" -> the number; "[N/A]" -> None.
            token = s.split()[0] if s.split() else s
            try:
                return float(token)
            except ValueError:
                return None

        rows.append({
            "sm_clock_mhz": number(parts[0]),
            "max_sm_clock_mhz": number(parts[1]),
            "power_draw_w": number(parts[2]),
            "power_limit_w": number(parts[3]),
            "throttle_reasons": parts[4] or None,
        })
    return rows


def gpu_clock_gate(gpus, ratio=0.70):
    """(ok, message) for row 31: under load, the median `clocks.sm` must be at least `ratio` of
    `clocks.max.sm`. `gpus` are dicts from parse_nvidia_smi_csv, sampled while the loadbench
    training loop is busy. Active throttle reasons are named in the message (all zeros counts as
    no reason). With no usable sample it fails closed: a paid run must not start on an unproven
    card."""
    pairs = [(g["sm_clock_mhz"], g["max_sm_clock_mhz"]) for g in gpus
             if g.get("sm_clock_mhz") is not None and g.get("max_sm_clock_mhz")]
    if not pairs:
        return False, "no usable nvidia-smi clock sample under load"
    clocks = sorted(c for c, _ in pairs)
    top = max(m for _, m in pairs)
    sm = clocks[len(clocks) // 2]

    def is_active(reason):
        reason = str(reason).strip()
        if not reason.lower().startswith("0x"):
            return False
        try:
            return int(reason, 16) != 0
        except ValueError:
            return False

    reasons = sorted({str(g["throttle_reasons"]).strip() for g in gpus if is_active(g.get("throttle_reasons"))})
    active = f"; throttle reasons {', '.join(reasons)}" if reasons else ""
    ok = sm >= ratio * top
    return ok, f"GPU sm clock {sm:.0f}/{top:.0f} MHz ({sm / top:.0%}) under load{active}"


def push_gate_from_steady_epoch_time_and_bytes(steady_epoch_seconds, epoch_bytes, measured_mb_s,
                                               margin=1.2):
    """Row 35: a push rate could keep up with an epoch's publishing if it is at least `margin`
    times epoch_bytes / steady_epoch_seconds. MB is decimal (1e6 bytes), matching the numbers the
    2026-11-04 run was judged on. Returns (ok, required_mb_s). The caller supplies the measured
    rate from the row-34 dummy push measured on the worker, and `steady_epoch_seconds` from the
    loadbench steady step time times steps per epoch (not the 5-step probe, which underestimates
    the steady rate ~4x)."""
    required_mb_s = margin * epoch_bytes / steady_epoch_seconds / 1e6
    return measured_mb_s >= required_mb_s, required_mb_s


def parse_delivery(arg):
    """A --delivery-from argument: `id=path` (the id goes in the clip's kind label), or a bare path
    (its stem is the id). Returns (id, path)."""
    clip_id, sep, path = arg.partition("=")
    if not sep:
        return Path(arg).stem, arg
    return clip_id or Path(path).stem, path


def clip_filename(epoch, n, kind):
    """A filename-safe WAV name for one listening clip, unique across epochs and slots:
    `e{epoch}-{kind}-{n:02d}.wav` with `kind` ("plain" or "delivery-of-<id>") lowercased and every
    non-alphanumeric turned into `-`, so two epochs or slots never overwrite each other."""
    safe = "".join(c if c.isalnum() else "-" for c in kind.lower())
    return f"e{epoch}-{safe}-{n:02d}.wav"


def seed_all(seed):
    """Reseed torch (CPU and every CUDA device) so the same line is generated the same across
    checkpoints; a no-op for seed None."""
    if seed is None:
        return
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unblind_entries(entries):
    """The [[clip]] block lines for `entries` = [(epoch, line, text, kind, path)], ordered by epoch
    then line (plain before delivery on the same line). `kind` is "plain" or "delivery-of-<id>"."""
    out = []
    ordered = sorted(entries, key=lambda e: (e[0], e[1], 0 if e[3] == "plain" else 1))
    for epoch, line, text, kind, path in ordered:
        out += ["[[clip]]", f"path = {json.dumps(path)}",
                f"label = {json.dumps(f'epoch {epoch}, {kind}: {text}')}", ""]
    return out


def unblind_hub(entries, title, question):
    """Full clips.toml text for an unblinded listening set: the title/question given on the command
    line, the fixed human-ear answers, `blind = false`, then the entries (see unblind_entries)."""
    head = [f"title = {json.dumps(title)}", f"question = {json.dumps(question)}",
            f"answers = {json.dumps(UNBLIND_ANSWERS)}", "blind = false", ""]
    return "\n".join(head + unblind_entries(entries))


# --- GPU / model parts (not imported in CI) ---------------------------------------------------------


def load_tts(path=BASE, device="cuda:0"):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return Qwen3TTSModel.from_pretrained(path, device_map=device, dtype=dtype, attn_implementation="sdpa")


@functools.lru_cache(maxsize=None)  # each clip's mel is read every epoch for the style encoder: decode it once
def mel(path):
    import librosa
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    wav, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, needs {SR}")
    return mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=SR,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)


def style_encoder(dim, width=256, out=2048, norm=None):
    """Reference encoder: mel [b, T, 128] + lengths -> a dim-number bottleneck -> a slot offset [b, out].
    norm="batch" (every run since 2026-10-31) standardises each of the dim numbers across the batch
    before the tanh. Without it the 2026-10-30 full runs collapsed: the code learned one constant,
    saturated the tanh at +-1 and stopped depending on the clip. With it the code has zero mean and
    unit spread across clips, so it can't be a constant (constants belong in up.bias and the game
    vectors) and the tanh stays in its working range. None = the old encoder, to load old runs."""
    import torch
    from torch import nn

    class StyleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(128, width, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU())
            self.gru = nn.GRU(width, width, batch_first=True)
            self.bottleneck = nn.Linear(width, dim)
            self.up = nn.Linear(dim, out)
            nn.init.zeros_(self.up.weight)  # starts as "no offset": her mean voice
            nn.init.zeros_(self.up.bias)
            self.norm = nn.BatchNorm1d(dim, affine=False) if norm == "batch" else None

        def code(self, mels, lengths):
            h = self.conv(mels.transpose(1, 2)).transpose(1, 2)
            n = torch.clamp((lengths + 7) // 8, min=1, max=h.shape[1]).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, n, batch_first=True, enforce_sorted=False)
            _, last = self.gru(packed)
            z = self.bottleneck(last[-1])
            if self.norm is not None:
                n = self.norm
                z = nn.functional.batch_norm(z, n.running_mean, n.running_var, training=self.training and z.shape[0] > 1,
                                             momentum=n.momentum, eps=n.eps)  # a 1-line batch uses the running stats
            return torch.tanh(z)

        def forward(self, mels, lengths):
            return self.up(self.code(mels, lengths))

    return StyleEncoder()


def xvectors(model, paths, cache):
    """One x-vector per ref clip, cached in a JSON file keyed by path."""
    import torch

    got = json.loads(cache.read_text()) if cache.exists() else {}
    with torch.no_grad():
        for p in dict.fromkeys(paths):
            if p not in got:
                m = mel(p).to(model.device).to(model.dtype)
                got[p] = model.speaker_encoder(m)[0].float().cpu().tolist()
    cache.write_text(json.dumps(got))
    return got


def _tool_version(exe):
    """The first line of `exe --version`, or None when it isn't installed (reported, not gated)."""
    import shutil
    import subprocess

    if not shutil.which(exe):
        return None
    try:
        return subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


def _generate_one(tts, out):
    """Generate and decode one short line from Base and write it as a WAV, so `check --generate`
    proves the whole generate-and-decode path before the paid run. Base clones a voice, so a short
    synthetic tone stands in for the reference; the point is the path, not the voice."""
    import numpy as np
    import soundfile as sf

    ref = out / "check-ref.wav"
    sf.write(ref, 0.05 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR), SR)
    wavs, got_sr = tts.generate_voice_clone(text=CHECK_LINE, ref_audio=str(ref), x_vector_only_mode=True)
    gen = out / "check-generate.wav"
    sf.write(gen, wavs[0], got_sr)
    return {"line": CHECK_LINE, "wav": str(gen), "sr": got_sr, "audio_s": round(len(wavs[0]) / got_sr, 2)}


def cmd_check(a):
    """Everything the run needs, checked in one go before training (plan ahead, don't chase errors)."""
    import shutil
    import subprocess

    out = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:  # noqa: BLE001 - report every failure, don't stop at the first
        out["torch_error"] = repr(e)
    # The driver version torch can't see; nvidia-smi reports it (None without a GPU).
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        out["nvidia"] = [x.strip() for x in drv.stdout.splitlines() if x.strip()] if drv.returncode == 0 else None
    except Exception:  # noqa: BLE001
        out["nvidia"] = None
    out["sox"] = shutil.which("sox")
    out["gcc"] = _tool_version("gcc")  # report only: this sdpa/wheel-only job does not compile anything
    out["g++"] = _tool_version("g++")
    for mod in ("qwen_tts", "transformers", "accelerate", "librosa", "soundfile", "huggingface_hub"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            out[f"{mod}_error"] = repr(e)
    if out.get("transformers") and out["transformers"] != "4.57.3":
        out["transformers_warning"] = "qwen-tts 0.1.1 pins 4.57.3"
    if out.get("vram_gb") is not None and out["vram_gb"] < 40:
        out["vram_warning"] = "a full fine-tune of 1.7B with --kl wants a 48 GB card"
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.work_dir or a.out
    work.mkdir(parents=True, exist_ok=True)
    out["free_disk_gb"] = round(checkpoints.free_gb(work), 1)
    tts = None
    if (not a.no_model or a.generate) and "qwen_tts_error" not in out:
        try:
            dev = "cuda:0" if out.get("gpu") else "cpu"
            tts = load_tts(a.base, dev)
            out["talker_params_m"] = round(sum(p.numel() for p in tts.model.talker.parameters()) / 1e6, 1)
            out["model"] = "loaded with sdpa"
            if dev.startswith("cuda"):
                import torch

                out["vram_after_load_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        except Exception as e:  # noqa: BLE001
            out["model_error"] = repr(e)
    if a.generate and tts is not None:
        try:
            out["generate"] = _generate_one(tts, a.out)
        except Exception as e:  # noqa: BLE001
            out["generate_error"] = repr(e)
    out["ok"] = not [k for k in out if k.endswith("_error")]
    (a.out / "check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


def cmd_replay(a):
    """Base speaks generic sentences in other (public) voices, x-vector only, the layout training uses."""
    import soundfile as sf

    voices = sorted(Path(a.voices).glob("*.wav"))
    if not voices:
        sys.exit("replay: no voices")
    sentences = [x.strip() for x in Path(a.sentences).read_text().splitlines() if x.strip()]
    tts = load_tts(a.base, a.device)
    (a.out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, text in enumerate(sentences):
        v = voices[n % len(voices)]
        wavs, sr = tts.generate_voice_clone(text=text, ref_audio=str(v), x_vector_only_mode=True)
        if sr != SR:
            sys.exit(f"replay: Base spoke at {sr} Hz")
        p = a.out / "wav" / f"replay-{n:04d}.wav"
        sf.write(p, wavs[0], sr)
        rows.append({"id": f"replay-{n:04d}", "audio": str(p), "text": text, "ref_audio": str(v), "replay": True})
    write_jsonl(a.out / "replay_raw.jsonl", rows)
    print(f"replay: {len(rows)} lines in {a.out}")


def cmd_prepare(a):
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    rows = read_jsonl(a.raw)
    bad = validate(rows, lambda p: sf.info(p).samplerate)
    if bad:
        sys.exit("prepare: fix these first:\n" + "\n".join(bad[:50]))
    tok = Qwen3TTSTokenizer.from_pretrained(a.tokenizer, device_map=a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / (Path(a.raw).stem.replace("_raw", "") + "_with_codes.jsonl")
    done = {r["id"] for r in read_jsonl(dst)} if dst.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with open(dst, "a") as f:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            enc = tok.encode([r["audio"] for r in chunk])
            for r, code in zip(chunk, enc.audio_codes):
                f.write(json.dumps({**r, "audio_codes": code.cpu().tolist()}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"prepare: {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"prepare: {dst}")


def loader(rows, tts, config, size, shuffle, seed, xvec, qwen_finetuning, style=False, games=None):
    """The official dataset/collate, plus each line's own x-vector. The official collate stacks every
    line's reference mel, which only works when all lines share one ref_audio, so it is handed one
    fixed clip and the x-vectors travel separately."""
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(qwen_finetuning))
    from dataset import TTSDataset

    fixed = rows[0]["ref_audio"]

    class WithVoice(TTSDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            row = self.data_list[idx]
            item["xvec"] = torch.tensor(xvec[row["_ref"]])
            if style:
                item["style_mel"] = mel(row["audio"])[0]
                item["game"] = games.get(row.get("game", ""), 0)
                item["audio_path"] = row["audio"]
            return item

        def collate_fn(self, batch):
            out = super().collate_fn(batch)
            out["xvec"] = torch.stack([b["xvec"] for b in batch])
            if style:
                out["style_mel"], out["style_len"] = pad_mels([b["style_mel"] for b in batch])
                out["game"] = torch.tensor([b["game"] for b in batch])
                out["audio_path"] = [b["audio_path"] for b in batch]
            return out

    data = [{**r, "_ref": r["ref_audio"], "ref_audio": fixed} for r in rows]
    ds = WithVoice(data, tts.processor, config)
    g = torch.Generator().manual_seed(seed) if shuffle else None
    # Worker processes keep the GPU fed while the main process collates. The
    # sampler stays in the main process, so a given seed yields the same batch
    # order for any worker count. worker-job.sh sets VI_NUM_WORKERS=0 on a retry
    # only after the cgroup oom_kill counter rose (loader processes cost RAM).
    workers = max(0, min(4, (os.cpu_count() or 1) - 2))
    workers = int(os.environ.get("VI_NUM_WORKERS", workers))
    return DataLoader(ds, batch_size=size, shuffle=shuffle, generator=g,
                      collate_fn=ds.collate_fn, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=True)


def forward(model, batch, voice):
    """The official forward pass with the alignment fixes; voice = [b, d] for the speaker slot.
    Returns (talker outputs, aligned sub-talker loss, sub-talker logits)."""
    import torch.nn.functional as F

    dev = model.device
    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    labels = batch["codec_0_labels"].to(dev)
    codec_mask = batch["codec_mask"].to(dev)
    emb = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * batch["text_embedding_mask"].to(dev)
    cemb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"].to(dev)
    cemb[:, SPEAKER_SLOT, :] = voice.to(dev, cemb.dtype)
    emb = emb + cemb
    for i in range(1, 16):
        e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        emb = emb + e * codec_mask.unsqueeze(-1)
    out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"].to(dev)[:, :-1],
                       labels=train_labels(labels), output_hidden_states=True)
    hidden = out.hidden_states[0][-1][predictor_hidden_mask(codec_mask)]
    sub_codes = codec_ids[codec_mask]
    sub_logits, _ = model.talker.forward_sub_talker_finetune(sub_codes, hidden)
    # Aligned like generation: codebook k's logit scores codebook k+1's code (its own loss shifts again).
    sub_loss = F.cross_entropy(sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                               sub_codes[:, 1:].reshape(-1))
    return out, sub_loss, sub_logits


def save_resume(out, epoch, step, heads, opt, gen):
    """What `train --resume` needs beyond epoch-<k>/ (talker, style): optimizer state, helper heads, step
    and the shuffle generator. One file, replaced atomically after each epoch (~8 bytes per talker
    parameter, so about 12 GB)."""
    import torch

    tmp = out / "resume.pt.tmp"
    torch.save({"epoch": epoch, "step": step, "opt": opt.state_dict(), "gen": gen.get_state(),
                "heads": {k: h.state_dict() for k, h in heads.items() if h is not None}}, tmp)
    tmp.replace(out / "resume.pt")


def warmup_factor(step, warmup):
    """Learning-rate multiplier for optimizer step `step` (0-based): a linear ramp over `warmup` steps."""
    return 1.0 if warmup <= 0 else min(1.0, (step + 1) / warmup)


def load_weights(ep, model, enc, game_emb):
    """An epoch folder's talker (and style encoder and game vectors, for style runs) into the models."""
    from safetensors.torch import load_file

    model.talker.load_state_dict(load_file(str(ep / "talker.safetensors")))
    if enc is not None:
        st = load_file(str(ep / "style.safetensors"))
        enc.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith("enc.")})
        game_emb.weight.data.copy_(st["game"].to(game_emb.weight.device))


def resume_state(out, model, enc, game_emb, heads, opt, gen):
    """Load the last finished epoch back in (see save_resume); returns (step, first epoch to run).
    A stop loses at most the epoch in progress. The data order of later epochs differs from an
    unbroken run's, which doesn't matter for training."""
    import torch

    f = out / "resume.pt"
    if not f.exists():
        sys.exit(f"train --resume: no {f} (no epoch finished yet: start again without --resume)")
    r = torch.load(f, map_location="cpu", weights_only=False)
    load_weights(out / f"epoch-{r['epoch']}", model, enc, game_emb)
    for k, h in heads.items():
        if h is not None:
            h.load_state_dict(r["heads"][k])
    opt.load_state_dict(r["opt"])
    gen.set_state(r["gen"])
    return r["step"], r["epoch"] + 1


def cmd_train(a):
    import itertools

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    torch.manual_seed(a.seed)
    tts = load_tts(a.base, "cuda:0")
    model = tts.model
    config = AutoConfig.from_pretrained(a.base)
    rows = read_jsonl(a.data)
    val = read_jsonl(a.val) if a.val else []
    replay = read_jsonl(a.replay) if a.replay else []
    if not rows:
        sys.exit("train: no training rows")
    if a.kl and not replay:
        sys.exit("train: --kl needs --replay")
    bad = validate(rows + val + replay)
    if bad:
        sys.exit("train: " + "; ".join(bad[:20]))
    a.out.mkdir(parents=True, exist_ok=True)

    xvec = xvectors(model, [r["ref_audio"] for r in rows + val + replay], a.out / "xvectors.json")
    her = mean_voice([xvec[r["ref_audio"]] for r in rows])
    (a.out / "voice.json").write_text(json.dumps({"speaker": SPEAKER, "clips": len(rows), "xvector": her.tolist()}))
    if a.speaker == "mean":
        for r in rows + val:
            xvec[r["ref_audio"]] = her.tolist()
    style = a.speaker == "style"
    games = {g: i for i, g in enumerate(sorted({r.get("game", "") for r in rows + val}))}
    extras = any([a.style_shuffle, a.game_adv, a.prosody_aux, a.style_dropout])
    if extras and not style:
        sys.exit("train: --style-shuffle/--game-adv/--prosody-aux/--style-dropout need --speaker style")
    pros = {}
    if a.prosody_aux:
        cache = a.out / "prosody.json"
        pros = json.loads(cache.read_text()) if cache.exists() else {}
        for r in rows + val:
            if r["audio"] not in pros:
                pros[r["audio"]] = prosody_targets(r["audio"], r["text"])
        cache.write_text(json.dumps(pros))
        pv = np.array([pros[r["audio"]] for r in rows])
        pm, ps = pv.mean(0), pv.std(0) + 1e-6
        pros = {k: ((np.array(v) - pm) / ps).tolist() for k, v in pros.items()}

    base = None
    if a.kl:
        base = load_tts(a.base, "cuda:0").model  # frozen copy for the KL term
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.talker.parameters())
    for p in params:
        p.requires_grad_(True)
    her_t = torch.tensor(her, dtype=torch.float32, device=model.device)
    enc = game_emb = game_head = pros_head = None
    fresh = []  # new, randomly started parts: they get their own, much higher learning rate
    if style:
        enc = style_encoder(a.style_dim, norm="batch").to(model.device)
        game_emb = torch.nn.Embedding(len(games), her_t.shape[0]).to(model.device)
        torch.nn.init.zeros_(game_emb.weight)
        fresh = list(enc.parameters()) + list(game_emb.parameters())
        if a.game_adv:
            game_head = torch.nn.Sequential(torch.nn.Linear(a.style_dim, 64), torch.nn.GELU(),
                                            torch.nn.Linear(64, len(games))).to(model.device)
            fresh += list(game_head.parameters())
        if a.prosody_aux:
            pros_head = torch.nn.Linear(a.style_dim, 4).to(model.device)
            fresh += list(pros_head.parameters())
        (a.out / "style.json").write_text(json.dumps({"dim": a.style_dim, "games": games, "norm": "batch"}))
    shuffle_gen = torch.Generator().manual_seed(a.seed + 7)
    chunk_frames = int(round(a.style_shuffle * SR / 256))  # mel hop 256
    # At the talker's 2e-6, a network that starts from scratch barely moves in ~500 steps (Adam moves each
    # weight about lr per step), so the style code would stay near zero and the slot would not steer.
    groups = [{"params": params, "lr": a.lr}] + ([{"params": fresh, "lr": a.style_lr}] if fresh else [])
    params = params + fresh
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    train_dl = loader(rows, tts, config, a.batch, True, a.seed, xvec, a.qwen_finetuning, style, games)
    val_dl = loader(val, tts, config, a.batch, False, a.seed, xvec, a.qwen_finetuning, style, games) if val else None
    replay_it = itertools.cycle(loader(replay, tts, config, a.batch, True, a.seed + 1, xvec, a.qwen_finetuning)) if a.kl else None

    stats = {}

    def slot(batch, training=True):
        """The speaker slot for her lines, and the style extras' loss (0 when off or not training)."""
        if not style:
            return batch["xvec"], torch.zeros((), device=model.device)
        dev = model.device
        mels, lens = batch["style_mel"], batch["style_len"]
        if training and chunk_frames:
            mels = torch.stack([shuffle_chunks(m, n, chunk_frames, shuffle_gen) for m, n in zip(mels, lens)])
        code = enc.code(mels.to(dev), lens.to(dev))
        if training and code.shape[0] > 1:  # collapse check: a code that stops varying across clips is dead
            stats["code_std"] = round(code.detach().float().std(0).mean().item(), 4)
            stats["offset"] = round(enc.up(code.detach()).float().norm(dim=1).mean().item(), 3)
        aux = torch.zeros((), device=dev)
        if training and game_head is not None:
            aux = aux + a.game_adv * torch.nn.functional.cross_entropy(game_head(grad_reverse(code)).float(), batch["game"].to(dev))
        if training and pros_head is not None:
            tgt = torch.tensor([pros[p] for p in batch["audio_path"]], device=dev)
            aux = aux + a.prosody_aux * torch.nn.functional.mse_loss(pros_head(code).float(), tgt)
        if training and a.style_dropout:
            keep = (torch.rand(code.shape[0], 1, device=dev) >= a.style_dropout).to(code.dtype)
            code = code * keep
        return her_t + enc.up(code) + game_emb(batch["game"].to(dev)), aux

    def her_loss(batch, training=True):
        voice, aux = slot(batch, training)
        out, sub, _ = forward(model, batch, voice)
        return out.loss + SUB_TALKER_WEIGHT * sub, aux

    def replay_kl(batch):
        out, _, sub = forward(model, batch, batch["xvec"])
        with torch.no_grad():
            ref, _, ref_sub = forward(base, batch, batch["xvec"])
        return replay_kl_term(ref.logits, out.logits, ref_sub, sub, batch["codec_0_labels"])

    step, first = 0, 0
    if a.resume:
        step, first = resume_state(a.out, model, enc, game_emb, {"game_head": game_head, "pros_head": pros_head}, opt,
                                   shuffle_gen)
        print(f"train: resuming after epoch {first - 1} (step {step})", flush=True)
    elif a.init_from:
        fresh_style = a.fresh_style
        old = a.init_from.parent / "style.json"
        if style and not fresh_style and (not old.exists() or not json.loads(old.read_text()).get("norm")):
            fresh_style = True  # an old, unnormalised (collapsed) style encoder isn't worth keeping
        load_weights(a.init_from, model, None if fresh_style else enc, game_emb if not fresh_style else None)
        print(f"train: starting from the weights in {a.init_from}{' (talker only, fresh style encoder)' if fresh_style else ''},"
              " fresh optimizer", flush=True)
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])
    log = open(a.out / "train_log.jsonl", "a")
    model.train()
    t0 = time.time()

    if a.loadbench_steps:
        # Row 35's loadbench: a short steady-state timing run at the real
        # batch/accum/kl, so the push gate uses a steady step time rather than
        # the 5-step probe's cold start (which read ~4x slow on 2026-11-04). It
        # writes no checkpoint and no train_log record; nvidia-smi is sampled
        # after each timed step, while this loop is the only real GPU load, for
        # row 31's clock gate. `steps_per_epoch` is what the real run would do:
        # floor(batches / accum), because an epoch ends without stepping a
        # partial accumulation group (a ceil here would overstate the epoch
        # time and so loosen the row-35 push gate).
        import statistics
        import subprocess

        times = []
        gpu_samples = []
        opt.zero_grad()
        for k, batch in enumerate(train_dl):
            t_step = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, aux = her_loss(batch)
                kl = replay_kl(next(replay_it)) if a.kl else torch.zeros((), device=model.device)
                total = (loss + aux + a.kl * kl) / a.accum
            total.backward()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"]
                opt.step()
                opt.zero_grad()
                step += 1
                times.append(time.time() - t_step)
                try:
                    smi = subprocess.run(
                        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,"
                         "power.limit,clocks_throttle_reasons.active", "--format=csv"],
                        capture_output=True, text=True, timeout=15)
                    if smi.returncode == 0:
                        gpu_samples.extend(parse_nvidia_smi_csv(smi.stdout))
                except Exception:  # noqa: BLE001 - no nvidia-smi is reported by the clock gate
                    pass
                if len(times) >= a.loadbench_steps:
                    break
        measured = times[a.loadbench_warmup:]
        if not measured:
            sys.exit(f"train: --loadbench-steps {a.loadbench_steps} left no timed step after "
                     f"{a.loadbench_warmup} warmup steps")
        steady = statistics.median(measured)
        steps_per_epoch = len(train_dl) // a.accum
        (a.out / "loadbench.json").write_text(json.dumps({
            "steady_step_s": round(steady, 3), "steps_s": [round(t, 3) for t in times],
            "warmup_steps": a.loadbench_warmup, "steps_per_epoch": steps_per_epoch,
            "batch": a.batch, "accum": a.accum, "kl": a.kl, "gpu_under_load": gpu_samples}, indent=1))
        print(f"loadbench: {steady:.2f} s/step steady, {steps_per_epoch} steps/epoch", flush=True)
        return

    for epoch in range(first, a.epochs):
        opt.zero_grad()
        for k, batch in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, aux = her_loss(batch)
                kl = replay_kl(next(replay_it)) if a.kl else torch.zeros((), device=model.device)
                total = (loss + aux + a.kl * kl) / a.accum
            total.backward()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * (1.0 if a.resume else warmup_factor(step, a.warmup))
                opt.step()
                opt.zero_grad()
                step += 1
                rec = {"epoch": epoch, "step": step, "loss": round(loss.item(), 4), "kl": round(kl.item(), 5), "aux": round(aux.item(), 5),
                       "peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2), "s": round(time.time() - t0), **stats}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 10 == 0:
                    print(rec, flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
        rec = {"epoch": epoch, "end": True}
        if val_dl is not None:
            model.eval()
            if enc is not None:
                enc.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rec["val_loss"] = round(float(np.mean([her_loss(b, training=False)[0].item() for b in val_dl])), 4)
            model.train()
            if enc is not None:
                enc.train()
        ep = a.out / f"epoch-{epoch}"
        ep.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.talker.state_dict().items()},
                  str(ep / "talker.safetensors"))
        if style:
            save_file({**{f"enc.{k}": v.detach().cpu().contiguous() for k, v in enc.state_dict().items()},
                       "game": game_emb.weight.detach().cpu().contiguous()}, str(ep / "style.safetensors"))
        (ep / "tune.json").write_text(json.dumps({"base": a.base, "lr": a.lr, "kl": a.kl, "speaker": a.speaker,
                                                  "style_lr": a.style_lr if style else None, "style_shuffle": a.style_shuffle, "game_adv": a.game_adv,
                                                  "prosody_aux": a.prosody_aux, "style_dropout": a.style_dropout}))
        if not a.no_resume_file:
            save_resume(a.out, epoch, step, {"game_head": game_head, "pros_head": pros_head}, opt, shuffle_gen)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if a.publish:
            checkpoints.publish_epoch(ep, a.publish)
        if a.max_steps and step >= a.max_steps:
            break
    print(f"train: done in {time.time() - t0:.0f} s, {step} steps, checkpoints in {a.out}")


def voice_model(base, vec, talker_file, dst):
    """Base turned into a custom_voice model with the tuned talker and speaker `tali` = her mean
    x-vector (the official recipe's checkpoint layout)."""
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = Path(base) if Path(base).exists() else Path(snapshot_download(base))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg["tts_model_type"] = "custom_voice"
    t = cfg.setdefault("talker_config", {})
    t["spk_id"] = {SPEAKER: SPEAKER_ID}
    t["spk_is_dialect"] = {SPEAKER: False}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    state = {k: v for k, v in load_file(str(src / "model.safetensors")).items() if not k.startswith("speaker_encoder")}
    state.update({f"talker.{k}": v for k, v in load_file(str(talker_file)).items()})
    w = state["talker.model.codec_embedding.weight"]
    w[SPEAKER_ID] = torch.tensor(vec, dtype=w.dtype)
    save_file(state, str(dst / "model.safetensors"))
    return dst


def speak_slots(a, tune, voice, ep, deliveries, no_plain=False):
    """What goes in the speaker slot for each listening condition, as (kind, label, vector). Her
    mean voice first (kind "plain"), then one per delivery clip (kind "delivery-of-<id>"): for a
    clip/mean run that clip's own x-vector (Base speaker encoder), for a style run the style
    encoder's reading of the clip. `no_plain` drops the plain mean-voice slot, leaving only the
    delivery slots (for the delivery-only speak calls)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    her = np.asarray(voice["xvector"], dtype=np.float64)
    style = tune.get("speaker") == "style"
    game_vec = np.zeros_like(her)
    enc = None
    if style:
        info = json.loads((ep.parent / "style.json").read_text())
        st = load_file(str(ep / "style.safetensors"))
        enc = style_encoder(info["dim"], norm=info.get("norm"))
        enc.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith("enc.")})
        enc.eval()
        g = info["games"].get(a.game)
        if g is None:
            sys.exit(f"speak: game {a.game!r} not in {sorted(info['games'])}")
        # her plain voice = code 0 (what style dropout trains): her mean + the game's vector + up's bias
        game_vec = st["game"][g].float().numpy() + st["enc.up.bias"].float().numpy()
    out = [] if no_plain else [("plain", f"her mean voice{f', {a.game} suit' if style else ''}",
                                 (her + game_vec).tolist())]
    if not deliveries:
        return out
    if style:
        with torch.no_grad():
            for clip_id, w in deliveries:
                m = mel(w)
                off = (enc(m, torch.tensor([m.shape[1]])) - enc.up.bias)[0].numpy()  # bias is in game_vec
                out.append((f"delivery-of-{clip_id}", f"delivery of {Path(w).stem}", (her + game_vec + off).tolist()))
    else:
        base = load_tts(tune["base"], a.device).model  # only for its speaker encoder
        with torch.no_grad():
            for clip_id, w in deliveries:
                m = mel(w).to(base.device).to(base.dtype)
                out.append((f"delivery-of-{clip_id}", f"delivery of {Path(w).stem}",
                            base.speaker_encoder(m)[0].float().cpu().tolist()))
        del base
    return out


def cmd_speak(a):
    if a.no_plain and not (a.delivery_from or a.style_from):
        sys.exit("speak: --no-plain needs at least one --delivery-from or --style-from")
    import hashlib

    import soundfile as sf
    import torch

    run = a.run
    voice = json.loads((run / "voice.json").read_text())
    epochs = sorted(int(p.name.split("-")[1]) for p in run.glob("epoch-*") if (p / "tune.json").exists())
    if not epochs:
        sys.exit("speak: no checkpoints in the run")
    epoch = a.epoch if a.epoch is not None else epochs[-1]
    ep = run / f"epoch-{epoch}"
    tune = json.loads((ep / "tune.json").read_text())
    sentences = [x.strip() for x in Path(a.sentences).read_text().splitlines() if x.strip()]
    deliveries = [parse_delivery(x) for x in a.delivery_from] + [(Path(x).stem, x) for x in a.style_from]
    slots = speak_slots(a, tune, voice, ep, deliveries, no_plain=a.no_plain)  # [(kind, label, vector)]
    vm = voice_model(tune["base"], voice["xvector"], ep / "talker.safetensors", run / f"voice-epoch-{epoch}")
    tts = load_tts(str(vm), a.device)
    row = tts.model.talker.model.codec_embedding.weight
    out = a.out or (run / f"speak-epoch-{epoch}")
    (out / "wav").mkdir(parents=True, exist_ok=True)
    clips, entries, timing = [], [], []
    for j, (kind, slot_label, vec) in enumerate(slots):
        with torch.no_grad():
            row[SPEAKER_ID] = torch.tensor(vec, dtype=row.dtype, device=row.device)
        for n, text in enumerate(sentences):
            seed_all(a.seed)
            t0 = time.time()
            wavs, sr = tts.generate_custom_voice(text=text, speaker=SPEAKER)
            p = out / "wav" / clip_filename(epoch, n, kind)
            sf.write(p, wavs[0], sr)
            timing.append({"slot": slot_label, "n": n, "s": round(time.time() - t0, 2), "audio_s": round(len(wavs[0]) / sr, 2)})
            clips.append((f"fine-tuned epoch {epoch}, no reference, {slot_label} | sentence {n}: {text}", p))
            entries.append((epoch, n, text, kind, str(p.relative_to(out))))
    if a.icl_ref:
        del tts  # one big model at a time
        base = load_tts(tune["base"], a.device)
        for n, text in enumerate(sentences):
            seed_all(a.seed)
            wavs, sr = base.generate_voice_clone(text=text, ref_audio=a.icl_ref, ref_text=a.icl_ref_text)
            p = out / "wav" / f"base-icl--{n:02d}.wav"
            sf.write(p, wavs[0], sr)
            clips.append((f"Base model with her clip as the example | sentence {n}: {text}", p))
    if a.unblind:
        title = a.title or "Fine-tuned voice: is this Tali, with no reference clip?"
        question = a.question or "Is this Tali with her suit and accent? Note short-line glitches or speed-ups."
        hub = out / "clips.toml"
        if hub.exists():
            hub.write_text(hub.read_text().rstrip() + "\n\n" + "\n".join(unblind_entries(entries)) + "\n")
        else:
            hub.write_text(unblind_hub(entries, title, question))
    else:
        clips.sort(key=lambda c: hashlib.sha256(c[0].encode()).hexdigest())
        hub = ['title = "Fine-tuned voice: is this Tali, with no reference clip?"',
               'question = "Is this Tali with her suit and accent? Note short-line glitches or speed-ups."',
               'answers = ["her, suit and accent", "her, suit or accent partly", "not quite her", "not her or broken"]',
               "blind = true", 'asked_by = "Modeling Tali\'s suit effect"', ""]
        for label, p in clips:
            hub += ["[[clip]]", f"path = {json.dumps(str(p.relative_to(out)))}", f"label = {json.dumps(label)}", ""]
        (out / "clips.toml").write_text("\n".join(hub))
    write_jsonl(out / "timing.jsonl", timing)
    print(f"speak: {len(clips)} clips in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--base", default=BASE)
    c.add_argument("--no-model", action="store_true", help="imports and GPU only")
    c.add_argument("--generate", action="store_true", help="load Base, generate and decode one line, write a WAV")
    c.add_argument("--work-dir", type=Path, help="directory whose free disk is reported (default --out)")
    r = sub.add_parser("replay")
    r.add_argument("--sentences", type=Path, required=True)
    r.add_argument("--voices", type=Path, required=True, help="24 kHz wavs of other (public) speakers")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--base", default=BASE)
    r.add_argument("--device", default="cuda:0")
    p = sub.add_parser("prepare")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--val", type=Path)
    t.add_argument("--replay", type=Path, help="replay_with_codes.jsonl, for --kl")
    t.add_argument("--kl", type=float, default=0.0, help="weight of the KL-to-Base term on replay lines")
    t.add_argument("--speaker", choices=["clip", "mean", "style"], default="clip")
    t.add_argument("--style-dim", type=int, default=16, help="--speaker style: numbers per clip's delivery")
    t.add_argument("--style-lr", type=float, default=1e-4,
                   help="--speaker style: learning rate of the style encoder, game vectors and helper heads")
    t.add_argument("--style-shuffle", type=float, default=0.0, help="--speaker style: shuffle the encoder's input in chunks of this many seconds")
    t.add_argument("--game-adv", type=float, default=0.0, help="--speaker style: weight of the gradient-reversed game classifier on the code")
    t.add_argument("--prosody-aux", type=float, default=0.0, help="--speaker style: weight of the pitch/loudness/pace head on the code")
    t.add_argument("--style-dropout", type=float, default=0.0, help="--speaker style: share of lines trained with the code zeroed")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--base", default=BASE)
    t.add_argument("--qwen-finetuning", type=Path, default=Path("Qwen3-TTS/finetuning"),
                   help="the official finetuning/ folder (for dataset.py), pinned commit 022e286b")
    t.add_argument("--lr", type=float, default=2e-6)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--max-steps", type=int, default=0, help="stop early (smoke)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--init-from", type=Path,
                   help="an epoch-<k> folder to start from (talker and style weights) with a fresh optimizer; use --warmup")
    t.add_argument("--fresh-style", action="store_true",
                   help="with --init-from: take only the talker, start the style encoder and game vectors fresh "
                        "(automatic when the folder's style encoder is the old, collapsed kind)")
    t.add_argument("--warmup", type=int, default=0,
                   help="optimizer steps over which the learning rates ramp up from zero (e.g. 50 with --init-from)")
    t.add_argument("--resume", action="store_true",
                   help="carry on after the last finished epoch in --out (same flags as the stopped run)")
    t.add_argument("--publish", type=Path,
                   help="copy each finished epoch into this folder, hash-verified (see checkpoints.py)")
    t.add_argument("--no-resume-file", action="store_true",
                   help="skip save_resume (the ~12 GB optimizer file)")
    t.add_argument("--loadbench-steps", type=int, default=0,
                   help="time this many optimizer steps after model load, write loadbench.json "
                        "(steady step time, steps per epoch, GPU samples), then exit without "
                        "writing a checkpoint (the preflight's row-35 loadbench)")
    t.add_argument("--loadbench-warmup", type=int, default=2,
                   help="loadbench: optimizer steps discarded before timing")
    s = sub.add_parser("speak")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--sentences", type=Path, required=True)
    s.add_argument("--epoch", type=int)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", type=Path)
    s.add_argument("--icl-ref", help="her clip, for a Base-model ICL comparison")
    s.add_argument("--style-from", action="append", default=[], help="her clip whose delivery goes in the slot")
    s.add_argument("--delivery-from", action="append", default=[], metavar="ID=PATH",
                   help="her clip whose own x-vector (clip/mean run) goes in the slot; ID labels the entry")
    s.add_argument("--no-plain", action="store_true",
                   help="speak only the delivery/style slots, never the plain mean-voice slot; "
                        "needs at least one --delivery-from or --style-from")
    s.add_argument("--seed", type=int, help="reseed torch before each clip, so the same line is generated the same across checkpoints")
    s.add_argument("--unblind", action="store_true",
                   help="write an unblinded clips.toml (epoch-ordered, labelled), not the hash-shuffled blind one")
    s.add_argument("--title", help="clips.toml title with --unblind")
    s.add_argument("--question", help="clips.toml question with --unblind")
    s.add_argument("--game", default="le3", help="--speaker style runs: which game's suit")
    s.add_argument("--icl-ref-text", default="")
    a = ap.parse_args(argv)
    cmds = {"check": cmd_check, "replay": cmd_replay, "prepare": cmd_prepare, "train": cmd_train, "speak": cmd_speak}
    return cmds[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
