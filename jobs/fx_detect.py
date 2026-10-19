"""Which detector hears the extra effects on her clips? Tests two audio-effect
embeddings and the channel measures against Brandon's by-ear marks
(read-only on the corpus; home, CPU, free).

Some of Tali's lines carry a radio (walkie-talkie) or other effect on top of
the suit. Check A of the voice gate needs a per-clip channel score so a mood
can't pass on the radio instead of her voice. Brandon tapped ~40 clips in the
review page's channel mode as plain / radio / other_effect / cant_tell
(corpus/tali/channel.jsonl). Those marks are the test set for four feature
sets, each scored the same way:

  channel   the six channel measures from after_labels/channel.py (low and
            high cut, band edges, noise, compression), read from its
            per-clip table; rebuilt here (channel_builtin) if no table is
            given
  extras    six cheap measures aimed at what those miss: crest factor,
            clipped share, spectral flatness, periodicity, energy between
            harmonics (distortion), and a tonal burst at either end (chirp)
  afx       AFx-Rep (Steinmetz et al., ST-ITO, ISMIR 2024): a PANNs Cnn14
            trained to tell audio-effect presets apart, 512-d, 48 kHz
  fxpp      Fx-Encoder++ (Sony Research, 2025): a Cnn14-style effects
            encoder trained contrastively on mixes, 128-d, 44.1 kHz

Both models were trained on music, not voice; the marks are what tells
whether they transfer. Only their effects encoders are used, rebuilt here in
plain torch (no laion-clap, lightning or torchlibrosa), and loaded strictly
from the published checkpoints, so a missing or extra weight stops the run.

How each set is scored (fixed before any result):
  - Targets: effect (radio or other_effect) vs plain, the main one; radio
    vs the rest; and all three classes (balanced accuracy). cant_tell is
    left out.
  - Folds: leave one conversation out (a bark is its own group), since
    radio clips come in runs from the same scene.
  - Model: standardise, PCA to at most 8 dims for the embeddings, L2
    logistic regression (C=1, balanced classes). Nothing is tuned on the
    marks.
  - Shuffled-label baseline: the whole fold loop rerun with labels
    shuffled, --perms times; p = share at or above the real AUC.
  - Confounds: the earlier low-cut measure mostly tracked her shouting
    (pitch, loudness and game explained R2 0.79). So each set is also
    scored residualised: what loudness, pitch, duration and game do to
    each feature is fitted on the whole corpus (no labels needed, and
    nearly every clip is plain) and taken out. Fitting it on the marks
    instead would also remove a real effect whenever the marked radio
    clips happen to be shouted. The confounds alone are scored as a
    baseline, and the detector fitted on all marks scores every clip,
    where the R2 of its score on those confounds is reported, before and
    after. With --marked-only the fit falls back to the plain clips of
    each training fold.
  - Pick: the highest residualised effect AUC with p < 0.05 and
    residualised confound R2 < 0.5; within 0.02 AUC the cheaper set wins
    (channel, extras, fxpp, afx). Its residualised score is the one to
    use downstream.

Usage (from the repo root; torch, numpy, scipy, scikit-learn, soundfile):
  python3 jobs/fx_detect.py fetch [--weights DIR]
      download both checkpoints (2.8 GB, one at a time), check their
      sha256, keep only the encoder weights (about 0.65 GB) and delete the
      rest; stops first if DIR lacks the space
  python3 jobs/fx_detect.py smoke corpus/tali [--weights DIR] [--marks F]
      [--channel F] [--channel-cols a,b]
      every check the run needs, in seconds: packages, weights load
      strictly, both nets on a synthetic tone and two real clips, marks and
      channel table parsed; prints the projected run time
  python3 jobs/fx_detect.py run corpus/tali [--weights DIR] [--marks F]
      [--channel F] [--channel-cols a,b] [--marked-only] [--perms 500]
      [--threads 4] [--out DIR]
      smoke, then measure and embed every clip (resumable), then score
Writes report.md, summary.json, scores.csv (every clip's score from each
set) and next_to_mark.csv to DIR (default corpus/tali/analysis/fx_detect).
Audio is read from harvest's raw WAVs (clean may have removed the effect)
and never leaves this machine; only the weights are downloaded.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
import types
import urllib.request
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal

import filter_probe as fp

LABELS = ("plain", "radio", "other_effect")
DROP = {"cant_tell", "cannot_tell", "unsure", "skip", "skipped", "unknown", ""}
ALIASES = {"other": "other_effect", "effect": "other_effect", "other_fx": "other_effect",
           "walkie_talkie": "radio", "comms": "radio", "none": "plain", "clean": "plain"}  # fmt: skip
MARK_KEYS = ("channel", "mark", "verdict", "label", "value", "class")
SETS = ("channel", "extras", "fxpp", "afx")  # cheapest first: breaks near-ties
CONFOUNDS = ("loud_db", "f0_med", "log_dur")
TIE = 0.02
MAX_P = 0.05
MAX_R2 = 0.5
PCA_DIMS = 8
MIN_S, MAX_S = 1.5, 10.0  # nets see 1.5-10 s: short clips tiled, long ones centre-cropped

# Pinned to the revisions checked on 2026-10-19.
MODELS = {
    "afx": {"repo": "csteinmetz1/afx-rep", "rev": "6582cb5414094146759319819a4242967ecc7e39",
            "file": "afx-rep.ckpt", "size": 1156477784, "rate": 48000,
            "sha256": "3587c4f3a1a8ecbc53b8023c480a0e6ff80719bcc26ce6ee6d08b8daf41d75d4"},
    "fxpp": {"repo": "yytung/fxencoder-plusplus", "rev": "d6ed786a5fa20126402121ba8e28e1a6427eea52",
             "file": "fxenc_plusplus_default.pt", "size": 1608449162, "rate": 44100,
             "sha256": "9291f51c4beb6a3bfd1c50aafeab3e3cbcc5d20f9a87234f2205f8156177c14c"},
}  # fmt: skip
ENCODER_GB = 0.35  # each extracted encoder, float32
FREE_MARGIN = 1 << 30


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def default_weights():
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vi-fx"


# --- the two effects encoders, in plain torch ----------------------------------


def nets():
    """(AfxRep, FxppEncoder) module classes. torch is imported here so the
    rest of the file (and its tests) runs without it. Parameter names
    match the checkpoints exactly, torchlibrosa's included."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class Stft(nn.Module):
        def __init__(self, n_fft, hop):
            super().__init__()
            self.n_fft, self.hop = n_fft, hop
            bins = n_fft // 2 + 1
            n = np.arange(n_fft)
            win = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)  # periodic Hann, as librosa's fftbins=True
            w = np.exp(-2j * np.pi * np.outer(n, n[:bins]) / n_fft) * win[:, None]
            self.conv_real = nn.Conv1d(1, bins, n_fft, stride=hop, bias=False)
            self.conv_imag = nn.Conv1d(1, bins, n_fft, stride=hop, bias=False)
            self.conv_real.weight.data = torch.tensor(np.real(w).T[:, None, :], dtype=torch.float32)
            self.conv_imag.weight.data = torch.tensor(np.imag(w).T[:, None, :], dtype=torch.float32)

        def forward(self, x):  # (B, L) -> (B, 1, T, bins), power
            x = F.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
            re, im = self.conv_real(x), self.conv_imag(x)
            return (re**2 + im**2)[:, None].transpose(2, 3)

    class Spectrogram(nn.Module):
        def __init__(self, n_fft, hop):
            super().__init__()
            self.stft = Stft(n_fft, hop)

        def forward(self, x):
            return self.stft(x)

    class Logmel(nn.Module):
        """melW comes from the checkpoint (librosa's Slaney filters); the
        placeholder here is only its shape."""

        def __init__(self, n_fft, mels):
            super().__init__()
            self.melW = nn.Parameter(torch.zeros(n_fft // 2 + 1, mels), requires_grad=False)

        def forward(self, x):
            return 10.0 * torch.log10(torch.clamp(x @ self.melW, min=1e-10))

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.bn2 = nn.BatchNorm2d(cout)

        def forward(self, x, pool):
            x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
            return F.avg_pool2d(x, pool)

    class Trunk(nn.Module):
        """Cnn14 as both models use it: log-mel, min-max to [-1, 1], six
        conv blocks, mean over mels, max + mean over time -> 2048."""

        def __init__(self, n_fft, hop, mels, cin):
            super().__init__()
            self.spectrogram_extractor = Spectrogram(n_fft, hop)
            self.logmel_extractor = Logmel(n_fft, mels)
            self.bn0 = nn.BatchNorm2d(mels)  # unused (min-max input norm) but in the weights
            widths = [cin, 64, 128, 256, 512, 1024, 2048]
            for i in range(6):
                setattr(self, f"conv_block{i + 1}", ConvBlock(widths[i], widths[i + 1]))

        def trunk(self, x):  # x: (B, C, L) -> (B, 2048)
            b, c, n = x.shape
            m = self.logmel_extractor(self.spectrogram_extractor(x.reshape(b * c, n)))
            m = (m.clamp(-80.0, 40.0) + 80.0) / 120.0 * 2.0 - 1.0
            m = m.reshape(b, c, m.shape[-2], m.shape[-1])
            for i in range(6):
                m = getattr(self, f"conv_block{i + 1}")(m, (2, 2) if i < 5 else (1, 1))
            m = m.mean(dim=3)
            return m.max(dim=2).values + m.mean(dim=2)

    class AfxRep(Trunk):
        """st_ito.models.panns.Cnn14 (config: 48 kHz, n_fft 2048, hop 1024,
        128 mels, 20-20000 Hz, batchnorm, min-max). Mono goes in as one
        channel: the mid embedding, L2-normalised as get_param_embeds does."""

        rate = 48000

        def __init__(self):
            super().__init__(2048, 1024, 128, 1)
            self.fc_mid = nn.Linear(2048, 512)
            self.fc_side = nn.Linear(2048, 512)

        def forward(self, x):  # (B, L)
            return F.normalize(self.fc_mid(self.trunk(x[:, None, :])), dim=-1)

    class FxppEncoder(nn.Module):
        """fxencoder_plusplus' MixtureFxEncoder (44.1 kHz, n_fft 2048,
        hop 512, 64 mels, 50-18000 Hz) plus its projection: the
        normalised 128-d get_fx_embedding. Stereo in: mono is duplicated,
        as the authors' README does."""

        rate = 44100

        def __init__(self):
            super().__init__()
            self.fx_encoder = Trunk(2048, 512, 64, 2)
            self.fx_encoder.fc_1 = nn.Linear(2048, 2048)
            self.fx_encoder_projection = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 128))

        def forward(self, x):  # (B, L)
            e = self.fx_encoder
            h = F.relu(e.fc_1(e.trunk(x[:, None, :].repeat(1, 2, 1))))
            return F.normalize(self.fx_encoder_projection(h), dim=-1)

    return AfxRep, FxppEncoder


# --- checkpoints ----------------------------------------------------------------


class _Stub:
    """Stands in for any class a checkpoint pickles that isn't installed
    here (Lightning callbacks, optimiser configs): only tensors are used."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, k, v):
        pass

    def append(self, v):
        pass

    def extend(self, v):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_Stub,), {"__module__": module})


tolerant_pickle = types.SimpleNamespace(Unpickler=_TolerantUnpickler, load=pickle.load, __name__="tolerant_pickle")


def encoder_state(name, path):
    """The encoder's tensors from a published checkpoint, renamed to the
    modules in nets(). The file comes from a pinned revision whose sha256
    fetch checked, so the full unpickler is acceptable; unknown classes
    become stubs instead of failing."""
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True, pickle_module=tolerant_pickle)
    except RuntimeError:  # not the zip format, so no mmap
        ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=tolerant_pickle)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if name == "afx":
        keep = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    else:
        keep = {k: v for k, v in sd.items() if k.startswith(("fx_encoder.", "fx_encoder_projection."))}
    return {k: v.detach().clone().float() if v.is_floating_point() else v.clone() for k, v in keep.items()}


def load_net(name, weights):
    """The encoder in eval mode, loaded strictly: any missing or extra
    weight is an error that names them."""
    import torch

    afx, fxpp = nets()
    net = afx() if name == "afx" else fxpp()
    path = Path(weights) / f"{name}-encoder.pt"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `python3 jobs/fx_detect.py fetch --weights {weights}` first")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    missing, extra = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or extra:
        raise SystemExit(f"{name}: weights don't match the network. missing {missing[:8]} extra {extra[:8]}")
    return net.eval()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size, tries=4):
    """url to dest with a .part file that resumes where it stopped."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored the range: start over
                    f.truncate(0)
                    have = 0
                last = time.time()
                while block := r.read(1 << 20):
                    f.write(block)
                    have += len(block)
                    if time.time() - last > 15:
                        log(f"  {dest.name}: {have / 1e9:.2f} of {size / 1e9:.2f} GB")
                        last = time.time()
        except OSError as e:
            log(f"  {dest.name}: {e}; retrying ({attempt + 1}/{tries})")
            time.sleep(2 ** (attempt + 1))
    if not part.exists() or part.stat().st_size != size:
        raise SystemExit(f"{dest.name}: download incomplete; rerun fetch to resume")
    part.rename(dest)


def fetch(weights):
    import torch

    weights = Path(weights)
    weights.mkdir(parents=True, exist_ok=True)
    for name, m in MODELS.items():
        out = weights / f"{name}-encoder.pt"
        if out.exists():
            log(f"{name}: {out} already there")
            continue
        raw = weights / m["file"]
        need = (0 if raw.exists() else m["size"]) + int(ENCODER_GB * 1e9) + FREE_MARGIN
        free = shutil.disk_usage(weights).free
        if free < need:
            raise SystemExit(f"{weights} has {free / 1e9:.1f} GB free; {name} needs {need / 1e9:.1f} GB while it "
                             f"downloads. Pass --weights on a disk with more room.")  # fmt: skip
        if not raw.exists():
            url = f"https://huggingface.co/{m['repo']}/resolve/{m['rev']}/{m['file']}"
            log(f"{name}: downloading {m['size'] / 1e9:.2f} GB from {m['repo']}")
            download(url, raw, m["size"])
        log(f"{name}: checking sha256")
        if sha256(raw) != m["sha256"]:
            raw.unlink()
            raise SystemExit(f"{name}: sha256 mismatch, file deleted; rerun fetch")
        sd = encoder_state(name, raw)
        torch.save(sd, out)
        load_net(name, weights)  # strict check before the big file goes
        raw.unlink()
        log(f"{name}: kept {out} ({out.stat().st_size / 1e9:.2f} GB, {len(sd)} tensors)")


# --- audio and hand-made measures ---------------------------------------------


def mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def resample(x, sr, to):
    if sr == to:
        return x
    g = gcd(int(sr), int(to))
    return signal.resample_poly(x, to // g, int(sr) // g)


def net_input(x, sr, rate):
    """Mono at the net's rate, MIN_S-MAX_S long, peak-normalised (as
    AFx-Rep's get_param_embeds does; done for both so level doesn't count)."""
    y = resample(x, sr, rate)
    lo, hi = int(MIN_S * rate), int(MAX_S * rate)
    if len(y) > hi:
        s = (len(y) - hi) // 2
        y = y[s : s + hi]
    elif len(y) < lo:
        y = np.tile(y, math.ceil(lo / max(len(y), 1)))[:lo]
    return (y / max(np.abs(y).max(), 1e-8)).astype(np.float32)


def band_db(f, p, lo, hi):
    return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-20)


def edge(f, p_db, peak, below, low_side):
    """Lowest (or highest) frequency within `below` dB of the peak, on a
    smoothed long-term spectrum."""
    ok = np.where(p_db > peak - below)[0]
    if not len(ok):
        return float("nan")
    return float(f[ok[0]] if low_side else f[ok[-1]])


def frame_db(x, frame=1024, hop=256):
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)


def channel_builtin(x, sr):
    """The six channel measures, rebuilt for when channel.py's table isn't
    given: low cut, high cut, low and high band edge, noise floor,
    compression. Spectra are of the active frames, levels relative."""
    frames = fp.active_frames(x, sr)
    win = np.hanning(frames.shape[1])
    p = np.mean(np.abs(np.fft.rfft(frames * win, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(frames.shape[1], 1 / sr)
    mid = band_db(f, p, 250, 4000)
    db = 10 * np.log10(np.convolve(p, np.ones(9) / 9, mode="same") + 1e-20)
    keep = (f >= 50) & (f <= min(sr / 2, 12000))
    peak = db[keep].max()
    fdb = frame_db(x)
    loud = fdb[fdb > fdb.max() - 30]
    return {"low_cut_db": band_db(f, p, 50, 250) - mid,
            "high_cut_db": band_db(f, p, 4000, min(8000, sr / 2)) - mid,
            "low_edge_hz": edge(f[keep], db[keep], peak, 20, True),
            "high_edge_hz": edge(f[keep], db[keep], peak, 30, False),
            "noise_db": float(np.percentile(fdb, 5) - np.percentile(fdb, 95)),
            "dyn_range_db": float(np.percentile(loud, 95) - np.percentile(loud, 10))}  # fmt: skip


def extras(x, sr):
    """Measures for what the channel set can't see. Distortion and limiting
    flatten peaks (crest_db, clip_share) and add energy between her
    harmonics (interharm_db); noise and bit-crushing raise flatness and
    lower periodicity; a radio squelch or beep is a tonal burst at either
    end (chirp_ms: how much longer a fixed tone holds at the edges than in
    the middle)."""
    frames = fp.active_frames(x, sr)
    act = frames.ravel()
    peak = np.abs(x).max() + 1e-12
    crest = 20 * np.log10(peak / (np.sqrt(np.mean(act**2)) + 1e-12))
    clip_share = float(np.mean(np.abs(x) >= peak * 10 ** (-0.1 / 20)))
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2 + 1e-20
    f = np.fft.rfftfreq(frames.shape[1], 1 / sr)
    band = (f >= 300) & (f <= 3400)
    s = spec[:, band]
    flat = np.exp(np.mean(np.log(s), axis=1)) / np.mean(s, axis=1)
    inter = []
    for fr in frames[:: max(1, len(frames) // 40)]:
        f0 = fp.f0_of(fr, sr)
        if f0 is None:
            continue
        db = 10 * np.log10(np.abs(np.fft.rfft((fr - fr.mean()) * win)) ** 2 + 1e-20)
        k = np.arange(1, 9)
        k = k[(k + 0.5) * f0 < min(sr / 2, 4000)]
        if len(k) >= 2:
            inter.append(float(np.mean(np.interp((k + 0.5) * f0, f, db) - np.interp(k * f0, f, db))))
    return {"crest_db": float(crest), "clip_share": clip_share,
            "flatness": float(np.median(flat)), "periodicity": fp.periodicity(frames, sr) or float("nan"),
            "interharm_db": float(np.median(inter)) if inter else float("nan"),
            "chirp_ms": chirp(x, sr)}  # fmt: skip


def longest_run(mask):
    """Per row of a boolean (bins, frames) array, the most consecutive True
    frames; the largest over rows."""
    best = 0
    run = np.zeros(mask.shape[0], int)
    for col in mask.T:
        run = np.where(col, run + 1, 0)
        best = max(best, int(run.max(initial=0)))
    return best


def chirp(x, sr, edge_s=0.4, prominence_db=20.0):
    """A squelch or beep holds one exact frequency; her pitch drifts by more
    than a few Hz within tens of ms. So: the longest time (ms) one frequency
    (300-6000 Hz, within one ~3 Hz bin) stays a prominent spectral peak, in
    the first or last edge_s, minus the same in the rest of the clip."""
    # Short frames, zero-padded to ~3 Hz bins: a fixed tone stays in its bin,
    # a harmonic of a drifting voice leaves it within a few frames.
    n = 1024
    hop = n // 4
    nfft = 1 << int(np.ceil(np.log2(sr / 3.0)))
    f, t, z = signal.stft(x, sr, nperseg=n, noverlap=n - hop, nfft=nfft, padded=False, boundary=None)
    if z.shape[1] < 4:
        return 0.0
    db = 20 * np.log10(np.abs(z[(f >= 300) & (f <= 6000)]) + 1e-10)
    level = 10 * np.log10(np.sum(np.abs(z) ** 2, axis=0) + 1e-20)
    peak = (db - np.median(db, axis=0) > prominence_db) & (level > level.max() - 40)
    peak[1:-1] &= (db[1:-1] >= db[:-2]) & (db[1:-1] >= db[2:])  # local maxima only
    peak[1:-1] |= peak[:-2] | peak[2:]  # a tone may sit between two bins
    ends = (t < edge_s) | (t > t[-1] - edge_s)
    head, tail = ends & (t < t[-1] / 2), ends & (t >= t[-1] / 2)
    edge = max(longest_run(peak[:, m]) for m in (head, tail))
    middle = longest_run(peak[:, ~ends]) if (~ends).any() else 0
    return float((edge - middle) * hop / sr * 1000)


def confounds(x, sr, duration):
    """What a channel detector must not just be tracking: how loud and how
    high she is, and how long the line is (game is added from the corpus)."""
    frames = fp.active_frames(x, sr)
    f0s = [fp.f0_of(fr, sr) for fr in frames[:: max(1, len(frames) // 60)]]
    f0s = [v for v in f0s if v]
    return {"loud_db": float(10 * np.log10(np.mean(frames**2) + 1e-12)),
            "f0_med": float(np.median(f0s)) if f0s else float("nan"),
            "log_dur": float(np.log(max(duration, 0.05)))}  # fmt: skip


# --- inputs ---------------------------------------------------------------------


def norm_label(v):
    v = str(v or "").strip().lower().replace("-", "_").replace(" ", "_").replace("'", "")
    return ALIASES.get(v, v)


def read_marks(path):
    """{id: label} from the review page's channel marks; the last record per
    id wins, cant_tell is dropped, anything unknown is an error."""
    marks, bad = {}, set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = next((k for k in MARK_KEYS if k in r), None)
        if "id" not in r or key is None:
            raise SystemExit(f"{path}: a record without id or one of {MARK_KEYS}: {line[:120]}")
        v = norm_label(r[key])
        if v in DROP:
            marks.pop(r["id"], None)
        elif v in LABELS:
            marks[r["id"]] = v
        else:
            bad.add(r[key])
    if bad:
        raise SystemExit(f"{path}: unknown marks {sorted(bad)}; expected {LABELS} or cant_tell")
    return marks


def read_table(path, cols=None):
    """{id: {col: float}} from channel.py's per-clip table (JSONL, JSON list
    or CSV with an id column). Without cols, every numeric column is used
    except confound-like ones, and the choice is printed."""
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        text = path.read_text()
        rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    table = {}
    for r in rows:
        vals = {}
        for k, v in r.items():
            try:
                vals[k] = float(v)
            except (TypeError, ValueError):
                continue
        table[str(r.get("id") or r.get("clip") or r.get("clip_id"))] = vals
    numeric = sorted(set().union(*[set(v) for v in table.values()])) if table else []
    if cols:
        missing = [c for c in cols if c not in numeric]
        if missing:
            raise SystemExit(f"{path}: no columns {missing}; numeric columns are {numeric}")
    else:
        skip = ("loud", "pitch", "f0", "dur", "energy", "rms", "level", "game", "resid", "pred", "score")
        cols = [c for c in numeric if not any(s in c.lower() for s in skip)]
        dropped = sorted(set(numeric) - set(cols))
        log(f"channel table: using {cols}" + (f"; left out {dropped} (confound-like)" if dropped else ""))
    return {i: {c: v.get(c, float("nan")) for c in cols} for i, v in table.items()}, cols


def find_channel_table(corpus):
    d = Path(corpus) / "analysis" / "after_labels"
    found = sorted(p for ext in ("jsonl", "csv", "json") for p in d.glob(f"channel*.{ext}")) if d.exists() else []
    return found[0] if found else None


def current_out(corpus, stage):
    d = Path(corpus) / "stages" / stage
    return d / (d / "current").read_text().strip() / "out"


def corpus_clips(corpus):
    from vi_job import load_clips

    harvest = current_out(corpus, "harvest")
    os.environ["VI_INPUT_HARVEST"] = str(harvest)
    try:
        os.environ["VI_INPUT_CLEAN"] = str(current_out(corpus, "clean"))  # leave out what clean dropped
    except FileNotFoundError:
        pass
    clips = load_clips(limit=False)
    for c in clips:
        c.raw = harvest / c.line["audio"]["path"]  # as shipped: clean may have removed the effect
        c.group = c.line.get("source", {}).get("conversation") or f"bark:{c.id}"
    return clips


# --- scoring --------------------------------------------------------------------


def auc(y, s):
    """Mann-Whitney AUC; NaN when one class is missing."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def impute(train, test):
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(train), train, med), np.where(np.isfinite(test), test, med)


def confound_fit(x, conf):
    """Linear coefficients of each feature on the confounds."""
    a = np.column_stack([np.ones(len(conf)), conf])
    x, _ = impute(x, x)
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return coef


def residualise(x, conf, coef):
    return x - np.column_stack([np.ones(len(conf)), conf]) @ coef


def fit_predict(xtr, ytr, xte, multi=False):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(xtr)
    xtr, xte = sc.transform(xtr), sc.transform(xte)
    if xtr.shape[1] > 16:
        pca = PCA(n_components=min(PCA_DIMS, len(xtr) - 2), random_state=0).fit(xtr)
        xtr, xte = pca.transform(xtr), pca.transform(xte)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(xtr, ytr)
    return m.predict(xte) if multi else m.predict_proba(xte)[:, 1]


def conf_matrix(clips, game_levels):
    """Confound columns: loudness, pitch, duration, game one-hot (first level dropped)."""
    c = np.array([[cl["conf"][k] for k in CONFOUNDS] for cl in clips], float)
    g = np.array([[cl["game"] == lv for lv in game_levels[1:]] for cl in clips], float).reshape(len(clips), -1)
    out = np.column_stack([c, g])
    med = np.nanmedian(out, axis=0)
    return np.where(np.isfinite(out), out, np.where(np.isfinite(med), med, 0.0))


def cv_scores(x, y, groups, conf=None, multi=False):
    """Out-of-fold predictions, leaving one group out. Folds whose training
    part lacks two of each class are left as NaN. With conf, the confounds
    are taken out inside each fold, fitted on that fold's plain (y == 0)
    clips only, so a real effect that happens to come with shouting isn't
    removed along with the shouting."""
    groups = np.asarray(groups)
    out = np.full(len(y), np.nan) if not multi else np.full(len(y), None, dtype=object)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        classes, counts = np.unique(y[tr], return_counts=True)
        if len(classes) < (3 if multi else 2) or counts.min() < 2:
            continue
        xtr, xte = impute(x[tr], x[te])
        if conf is not None:
            plain = y[tr] == 0
            coef = confound_fit(xtr[plain], conf[tr][plain])
            xtr, xte = residualise(xtr, conf[tr], coef), residualise(xte, conf[te], coef)
        out[te] = fit_predict(xtr, y[tr], xte, multi)
    return out


def balanced_acc(y, pred):
    ok = np.array([p is not None for p in pred])
    if not ok.any():
        return float("nan")
    y, pred = y[ok], pred[ok]
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def boot_ci(y, s, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        a = auc(y[pick], s[pick])
        if np.isfinite(a):
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def perm_p(x, y, groups, observed, perms, conf=None, seed=0):
    """Share of label shuffles whose out-of-fold AUC is at or above the
    real one (with the +1 correction)."""
    if not np.isfinite(observed) or perms <= 0:
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(perms):
        yp = rng.permutation(y)
        a = auc(yp, cv_scores(x, yp, groups, conf))
        hits += bool(np.isfinite(a) and a >= observed)
    return (hits + 1) / (perms + 1)


def score_set(name, x, labels, groups, conf, perms, x_all=None, conf_all=None):
    """Every number the report shows for one feature set on the marks.

    Residualised scores take loudness, pitch, duration and game out of the
    features first. With x_all (every clip in the corpus, nearly all of
    them plain) the coefficients come from the whole corpus, which needs no
    labels and sees how her voice alone moves each feature; without it,
    from the plain clips of each training fold."""
    eff = np.array([lab != "plain" for lab in labels])
    radio = np.array([lab == "radio" for lab in labels])
    y3 = np.array(labels)
    s_raw = cv_scores(x, eff.astype(int), groups)
    if conf is None:
        x_res, fold_conf = None, None
    elif x_all is not None:
        x_res, fold_conf = residualise(impute(x, x)[0], conf, confound_fit(x_all, conf_all)), None
    else:
        x_res, fold_conf = x, conf
    s_res = cv_scores(x_res, eff.astype(int), groups, fold_conf) if x_res is not None else np.full(len(eff), np.nan)
    a_raw, a_res = auc(eff, s_raw), auc(eff, s_res)
    r = {"set": name, "dims": int(x.shape[1]),
         "effect_auc": a_raw, "effect_auc_ci": boot_ci(eff, s_raw, groups),
         "effect_auc_resid": a_res, "effect_auc_resid_ci": boot_ci(eff, s_res, groups),
         "resid_fit": None if conf is None else ("corpus" if x_all is not None else "plain clips in each fold"),
         "radio_auc": auc(radio, cv_scores(x, radio.astype(int), groups)),
         "three_class_bal_acc": balanced_acc(y3, cv_scores(x, y3, groups, multi=True)),
         "p_raw": perm_p(x, eff.astype(int), groups, a_raw, perms, seed=1),
         "p_resid": None if x_res is None else perm_p(x_res, eff.astype(int), groups, a_res, perms, fold_conf, seed=2)}  # fmt: skip
    return r, s_res


def corpus_r2(x_all, x_marked, y_marked, conf_all):
    """Fit on every mark, score every clip, and report how much of that
    score (log-odds) loudness, pitch, duration and game explain."""
    xtr, xall = impute(x_marked, x_all)
    p = np.clip(fit_predict(xtr, y_marked, xall), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    a = np.column_stack([np.ones(len(conf_all)), conf_all])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    resid = z - a @ coef
    r2 = 1 - resid.var() / (z.var() + 1e-20)
    return p, float(r2)


def pick(results):
    """The rule in the module docstring, applied to the rows as given."""
    ok = [r for r in results if r["set"] in SETS and r["p_resid"] is not None and r["p_resid"] < MAX_P
          and (r.get("corpus_conf_r2") is None or r["corpus_conf_r2"] < MAX_R2)
          and np.isfinite(r["effect_auc_resid"])]  # fmt: skip
    if not ok:
        return None
    best = max(r["effect_auc_resid"] for r in ok)
    near = [r for r in ok if r["effect_auc_resid"] >= best - TIE]
    return min(near, key=lambda r: SETS.index(r["set"]))["set"]


def coefficients(x, y, cols):
    """Standardised logistic weights on all marks, for the small sets: which
    measure carries the effect."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x, _ = impute(x, x)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(StandardScaler().fit_transform(x), y)
    return dict(sorted(zip(cols, map(float, m.coef_[0])), key=lambda kv: -abs(kv[1])))


# --- the run ------------------------------------------------------------------


def measure(clip, nets_by_name):
    import soundfile as sf
    import torch

    x, sr = sf.read(str(clip.raw), always_2d=False)
    x = mono(x)
    r = {"channel_builtin": channel_builtin(x, sr), "extras": extras(x, sr),
         "conf": confounds(x, sr, len(x) / sr)}  # fmt: skip
    with torch.no_grad():
        for name, net in nets_by_name.items():
            inp = torch.from_numpy(net_input(x, sr, net.rate))[None]
            r[name] = net(inp)[0].numpy().astype(np.float32)
    return r


def synthetic_tone(sr=24000, seconds=2.0):
    t = np.arange(int(sr * seconds)) / sr
    f0 = 200 + 20 * np.sin(2 * np.pi * 3 * t)
    x = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / sr) / k for k in range(1, 12))
    return x * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t) ** 2) * 0.3, sr


def smoke(a, clips=None, need_marks=False):
    """Everything the run needs, checked before the long part starts.
    Returns (clips, marks, channel table, cols, nets)."""
    import importlib

    problems = []
    for mod in ("numpy", "scipy", "sklearn", "soundfile", "torch"):
        try:
            m = importlib.import_module(mod)
            log(f"ok   {mod} {getattr(m, '__version__', '')}")
        except ImportError as e:
            problems.append(f"{mod} missing ({e})")
    if problems:
        raise SystemExit("smoke failed:\n  " + "\n  ".join(problems))
    import torch

    torch.set_num_threads(a.threads)
    loaded = {}
    for name in MODELS:
        loaded[name] = load_net(name, a.weights)
        log(f"ok   {name} weights load strictly")
    x, sr = synthetic_tone()
    for name, net in loaded.items():
        with torch.no_grad():
            e = net(torch.from_numpy(net_input(x, sr, net.rate))[None])[0].numpy()
        if not np.all(np.isfinite(e)) or abs(np.linalg.norm(e) - 1) > 1e-3:
            problems.append(f"{name}: bad embedding on a synthetic tone")
        log(f"ok   {name} on a synthetic tone: {e.shape[0]}-d")

    if Path(a.marks).exists():
        marks = read_marks(a.marks)
        counts = {lab: sum(v == lab for v in marks.values()) for lab in LABELS}
        log(f"ok   marks: {counts} from {a.marks}")
        if counts["plain"] < 3 or counts["radio"] + counts["other_effect"] < 3:
            if need_marks:
                problems.append("fewer than 3 plain or 3 effect marks: too few to score")
            else:
                log("note fewer than 3 plain or 3 effect marks so far: run needs more")
    elif need_marks:
        raise SystemExit(f"smoke failed: no marks at {a.marks} yet (tap the channel review first)")
    else:
        marks = {}
        log(f"note no marks at {a.marks} yet: run needs them; everything else is checked")

    table, cols = None, None
    path = a.channel or find_channel_table(a.corpus)
    if path:
        table, cols = read_table(path, a.channel_cols.split(",") if a.channel_cols else None)
        log(f"ok   channel table {path}: {len(table)} clips, {len(cols)} measures")
    else:
        log("note no channel.py table found: the channel set is rebuilt here (channel_builtin)")

    clips = clips if clips is not None else corpus_clips(a.corpus)
    by_id = {c.id: c for c in clips}
    unknown = [i for i in marks if i not in by_id]
    if unknown:
        problems.append(f"{len(unknown)} marked ids not in the corpus, e.g. {unknown[:3]}")
    if table is not None:
        nomeas = [i for i in marks if i not in table]
        if nomeas:
            problems.append(f"{len(nomeas)} marked clips missing from the channel table, e.g. {nomeas[:3]}")
    sample = [by_id[i] for i in list(marks)[:2] if i in by_id] or clips[:2]
    t0 = time.time()
    for c in sample:
        r = measure(c, loaded)
        bad = [k for k in ("afx", "fxpp") if not np.all(np.isfinite(r[k]))]
        if bad:
            problems.append(f"{c.id}: non-finite {bad}")
    per = (time.time() - t0) / max(len(sample), 1)
    n = len(marks) if a.marked_only else len(clips)
    log(f"ok   {len(sample)} real clips measured, {per:.1f} s each: about {per * n / 60:.0f} min for {n} clips")
    if problems:
        raise SystemExit("smoke failed:\n  " + "\n  ".join(problems))
    log("smoke passed")
    return clips, marks, table, cols, loaded


EMBED_DIMS = {"afx": 512, "fxpp": 128}


def load_cache(out):
    """Measures already taken: measures.jsonl holds the small ones, one line
    per clip, and <model>.f32 the embeddings as raw float32 rows in the same
    order. A run stopped mid-write keeps only the clips complete in all
    three."""
    path = out / "measures.jsonl"
    if not path.exists():
        return {}
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip().endswith("}")]
    vecs = {k: np.fromfile(out / f"{k}.f32", np.float32) if (out / f"{k}.f32").exists() else np.zeros(0, np.float32)
            for k in EMBED_DIMS}  # fmt: skip
    n = min([len(recs)] + [len(v) // EMBED_DIMS[k] for k, v in vecs.items()])
    done = {}
    for i, r in enumerate(recs[:n]):
        for k, d in EMBED_DIMS.items():
            r[k] = vecs[k][i * d : (i + 1) * d]
        done[r["id"]] = r
    with open(path, "w") as f:  # drop any partial tail so the files stay in step
        f.writelines(json.dumps({k: v for k, v in r.items() if k not in EMBED_DIMS}) + "\n" for r in recs[:n])
    for k, d in EMBED_DIMS.items():
        vecs[k][: n * d].tofile(out / f"{k}.f32")
    return done


def run(a):
    clips, marks, table, cols, loaded = smoke(a, need_marks=True)
    out = Path(a.out) if a.out else Path(a.corpus) / "analysis" / "fx_detect"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if c.id in marks] + [c for c in clips if c.id not in marks and not a.marked_only]
    done = load_cache(out)
    log(f"{len(done)} clips already measured; {sum(c.id not in done for c in todo)} to go")
    files = {k: open(out / f"{k}.f32", "ab") for k in EMBED_DIMS}
    with open(out / "measures.jsonl", "a") as f:
        for i, c in enumerate(todo):
            if c.id in done:
                continue
            try:
                r = measure(c, loaded)
            except Exception as e:  # noqa: BLE001 - report and carry on
                log(f"{c.id}: {e}")
                continue
            done[c.id] = r
            for k in EMBED_DIMS:
                files[k].write(r[k].astype(np.float32).tobytes())
                files[k].flush()
            f.write(json.dumps({"id": c.id, **{k: v for k, v in r.items() if k not in EMBED_DIMS}}) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                log(f"{i + 1}/{len(todo)}")
    for fh in files.values():
        fh.close()

    rows = []
    for c in todo:
        if c.id not in done:
            continue
        r = done[c.id]
        rows.append({"id": c.id, "game": c.game, "group": c.group, "label": marks.get(c.id),
                     "radio_tag": "radio" in (c.scene_tags or []), "conf": r["conf"],
                     "extras": r["extras"], "channel_builtin": r["channel_builtin"], "afx": r["afx"], "fxpp": r["fxpp"]})  # fmt: skip
    marked = [r for r in rows if r["label"]]
    if table is not None:
        rows_t = [r for r in rows if r["id"] in table]
        if len(rows_t) < len(rows):
            log(f"note {len(rows) - len(rows_t)} clips have no channel.py row: their channel score is blank")
    games = sorted({r["game"] for r in rows})

    def features(name, rs):
        if name == "channel" and table is not None:
            return np.array([[table.get(r["id"], {}).get(k, np.nan) for k in cols] for r in rs], float), cols
        if name == "channel":
            name = "channel_builtin"
        if name in ("channel_builtin", "extras"):
            keys = list(rs[0][name])
            return np.array([[r[name][k] for k in keys] for r in rs], float), keys
        return np.stack([r[name] for r in rs]).astype(float), None

    labels = [r["label"] for r in marked]
    groups = [r["group"] for r in marked]
    conf_m, conf_all = conf_matrix(marked, games), conf_matrix(rows, games)
    eff = np.array([lab != "plain" for lab in labels]).astype(int)
    results, scores = [], {}
    names = [("confounds", None)] + [(s, s) for s in SETS]
    corpus_fit = not a.marked_only
    for label, name in names:
        if name is None:
            x, keys = conf_m, None
        else:
            x, keys = features(name, marked)
            x_all, _ = features(name, rows)
        log(f"scoring {label} ({x.shape[1]} dims)")
        if name is None:
            res, _ = score_set(label, x, labels, groups, None, a.perms)
        elif corpus_fit:
            res, _ = score_set(label, x, labels, groups, conf_m, a.perms, x_all, conf_all)
        else:
            res, _ = score_set(label, x, labels, groups, conf_m, a.perms)
        if name is not None:
            p_raw, res["corpus_conf_r2_raw"] = corpus_r2(x_all, x, eff, conf_all)
            # The detector to use: fitted on all marks after the confounds
            # are taken out (coefficients as in the residualised score).
            plain = eff == 0
            coef = confound_fit(x_all, conf_all) if corpus_fit else confound_fit(x[plain], conf_m[plain])
            xr_all = residualise(impute(x_all, x_all)[0], conf_all, coef)
            xr = residualise(impute(x, x)[0], conf_m, coef)
            p_res, r2 = corpus_r2(xr_all, xr, eff, conf_all)
            # Over the marks alone (--marked-only) this R2 mostly measures the
            # labels, so it is left out there and gates only a corpus run.
            res["corpus_conf_r2"] = r2 if corpus_fit else None
            if not corpus_fit:
                res["corpus_conf_r2_raw"] = None
            res["corpus_radio_tag_auc"] = auc([r["radio_tag"] for r in rows], p_res)
            scores[label] = p_res
            scores[label + "_raw"] = p_raw
            if keys:
                res["weights"] = coefficients(xr, eff, keys)
        if name == "channel" and table is None:
            res["source"] = "channel_builtin (no channel.py table given)"
        results.append(res)
    winner = pick(results)

    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "game", "conversation", "mark", "radio_tag"] + [f"{s}_effect_p" for s in scores])
        for i, r in enumerate(rows):
            w.writerow([r["id"], r["game"], r["group"], r["label"] or "", int(r["radio_tag"])]
                       + [f"{scores[s][i]:.4f}" for s in scores])  # fmt: skip
    nxt = []
    if winner:
        order = np.argsort(-scores[winner])
        unmarked = [i for i in order if not rows[i]["label"]]
        # The most and least confident unmarked clips plus the most unsure:
        # a second round of taps checks both ends and the boundary.
        mid = sorted(unmarked, key=lambda i: abs(scores[winner][i] - 0.5))
        for i in list(dict.fromkeys(unmarked[:15] + mid[:10] + unmarked[-5:])):
            nxt.append((rows[i]["id"], rows[i]["game"], f"{scores[winner][i]:.3f}"))
    with open(out / "next_to_mark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "game", f"{winner}_effect_p" if winner else "score"])
        w.writerows(nxt)
    summary = {"marks": {lab: labels.count(lab) for lab in LABELS}, "conversations": len(set(groups)),
               "clips_scored": len(rows), "channel_source": str(a.channel or find_channel_table(a.corpus) or "builtin"),
               "channel_cols": cols, "perms": a.perms, "winner": winner, "results": results,
               "rule": f"max residualised effect AUC with p<{MAX_P}, residualised confound R2<{MAX_R2}; "
                       f"within {TIE} the cheaper of {SETS}"}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "report.md").write_text(report(summary))
    print(report(summary))


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def ci(v):
    return f"[{fmt(v[0])}, {fmt(v[1])}]" if v and v[0] is not None else ""


def report(s):
    lines = ["# Effect detector vs Brandon's marks\n",
             f"Marks: {s['marks']} over {s['conversations']} conversations; {s['clips_scored']} clips scored. "
             f"Channel measures from {s['channel_source']}.\n",
             "| set | dims | effect AUC | resid. AUC | p (resid.) | radio AUC | 3-class bal. acc | confound R2 raw / resid. | radio-tag AUC |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in s["results"]:
        lines.append(f"| {r['set']} | {r['dims']} | {fmt(r['effect_auc'])} {ci(r['effect_auc_ci'])} | "
                     f"{fmt(r['effect_auc_resid'])} {ci(r['effect_auc_resid_ci'])} | {fmt(r['p_resid'], 3)} | "
                     f"{fmt(r['radio_auc'])} | {fmt(r['three_class_bal_acc'])} | "
                     f"{fmt(r.get('corpus_conf_r2_raw'))} / {fmt(r.get('corpus_conf_r2'))} | "
                     f"{fmt(r.get('corpus_radio_tag_auc'))} |")  # fmt: skip
    lines += ["", f"Pick: **{s['winner'] or 'none passed'}** ({s['rule']}).", "",
              "effect AUC: effect vs plain, out of fold, one conversation left out at a time (0.5 is chance). "
              "resid.: the same after taking out what loudness, pitch, duration and game do to each feature "
              "across the whole corpus (nearly all plain), so a set that only tracks her shouting falls to chance. "
              "confounds row: those four alone. confound R2: how much of a detector's score on every clip those "
              "four explain, before and after. radio-tag AUC: how well the residualised score finds the game's "
              "radio scene tag (incomplete, only a hint). scores.csv has every clip's score from each set "
              "(<set>_effect_p residualised, <set>_raw_effect_p not).", ""]  # fmt: skip
    for r in s["results"]:
        if r.get("weights"):
            top = ", ".join(f"{k} {v:+.2f}" for k, v in list(r["weights"].items())[:6])
            lines.append(f"- {r['set']} weights (standardised, + means more effect-like): {top}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--weights", default=str(default_weights()))
    for name in ("smoke", "run"):
        p = sub.add_parser(name)
        p.add_argument("corpus", nargs="?", default="corpus/tali")
        p.add_argument("--weights", default=str(default_weights()))
        p.add_argument("--marks", default=None, help="default <corpus>/channel.jsonl")
        p.add_argument("--channel", default=None, help="channel.py's per-clip table (default: found in analysis/after_labels)")
        p.add_argument("--channel-cols", default=None, help="comma-separated columns of that table to use")
        p.add_argument("--marked-only", action="store_true", help="measure only the marked clips (no corpus R2)")
        p.add_argument("--perms", type=int, default=500)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.weights)
        return
    a.marks = a.marks or str(Path(a.corpus) / "channel.jsonl")
    if a.cmd == "smoke":
        smoke(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
