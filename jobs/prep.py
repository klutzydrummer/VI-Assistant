"""prep stage: one consistent trim and level for every clip, at home on the CPU.

Plan: docs/design/corpus-pipeline.md, step 1. Format:
docs/design/corpus-format.md, prep. Per clip:

  noise floor  the level and spectrum of the silence the trim cuts away,
               measured *before* trimming. Pure recording chain with no voice
               in it, so it is the cleanest nuisance covariate there is.
  trim         leading and trailing silence cut where the frame level drops
               `top_db` below the clip's loudest frame, keeping `pad_s`. Later
               stages cut at these bounds, so every stage sees the same clip.
  level        one linear gain to `target_lufs` (ITU-R BS.1770 integrated
               loudness; RMS for clips under `rms_below_s`, where the 400 ms
               gating is unreliable). Never dynamic compression: dynamics carry
               vocal effort. The gain is capped so the peak stays under
               `max_peak_dbfs` (flag gain_capped). The level before the gain is
               kept as a covariate.
  audio16k     a 16 kHz FLAC copy, trimmed and levelled, for embedding only.
               The 24 kHz originals stay as harvest wrote them.

Outputs: prep.jsonl, audio16k/<id>.flac, prep.json, checks.json.

Settings: top_db, pad_s, target_lufs, rms_below_s, max_peak_dbfs, bands,
limit. Needs numpy, scipy and soundfile (nix-shell at home); no GPU.
"""

import json
import math

import checks
from vi_job import load_clips, log, out_dir, settings, write_jsonl

RATE_16K = 16_000
FRAME = 2048  # librosa.effects.trim's defaults, which clean used before prep
HOP = 512
MIN_NOISE_S = 0.05  # less trimmed-away silence than this gives no floor
SPECTRUM_N = 512


def main():
    import numpy as np
    import soundfile as sf

    s = settings()
    out = out_dir()
    clips = load_clips()
    top_db = float(s.get("top_db", 40))
    pad_s = float(s.get("pad_s", 0.05))
    target = float(s.get("target_lufs", -23))
    rms_below_s = float(s.get("rms_below_s", 1.0))
    max_peak = float(s.get("max_peak_dbfs", -1))
    edges = band_edges(int(s.get("bands", 16)))
    log(f"prep: {len(clips)} clips")

    (out / "audio16k").mkdir(exist_ok=True)
    records = []
    for c in clips:
        data, rate = sf.read(str(c.audio), dtype="float64", always_2d=True)
        data = data.mean(axis=1)
        r = prepare(data, rate, top_db=top_db, pad_s=pad_s, target_lufs=target,
                    rms_below_s=rms_below_s, max_peak_dbfs=max_peak, edges=edges)  # fmt: skip
        audio = r.pop("audio16k")
        rec = {"id": c.id, **r, "flags": []}
        if r["duration_s"] < 0.2:
            rec["flags"].append("silent")
        else:
            rec["audio16k"] = {"path": f"audio16k/{c.id}.flac", "sample_rate": RATE_16K}
            sf.write(str(out / rec["audio16k"]["path"]), audio, RATE_16K, format="FLAC", subtype="PCM_16")
        if r["gain_capped"]:
            rec["flags"].append("gain_capped")
        if r["noise_floor"] is None:
            rec["flags"].append("no_noise_floor")
        records.append(rec)

    write_jsonl(out / "prep.jsonl", records)
    report = summary(records, s, edges)
    (out / "prep.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log(f"prep: {report['flags']}")
    n = max(1, len(records))
    results = [
        checks.Check("silent share", round(report["flags"].get("silent", 0) / n, 3), 0.05),
        checks.Check("gain_capped share", round(report["flags"].get("gain_capped", 0) / n, 3), 0.2,
                     note="levelling barely applied; lower target_lufs"),
        checks.Check("no_noise_floor share", round(report["flags"].get("no_noise_floor", 0) / n, 3), 0.5,
                     note="too little trimmed silence to measure the recording chain"),
    ]  # fmt: skip
    checks.finish(out, results, s.get("checks", "warn"))


def prepare(data, rate, top_db=40.0, pad_s=0.05, target_lufs=-23.0, rms_below_s=1.0,
            max_peak_dbfs=-1.0, edges=None):  # fmt: skip
    """Everything prep does to one mono float clip. Returns the record fields
    plus `audio16k`, the trimmed, levelled 16 kHz samples."""
    import numpy as np

    edges = edges or band_edges(16)
    start, end = trim_bounds(data, top_db)
    pad = int(round(pad_s * rate))
    lo, hi = (max(0, start - pad), min(len(data), end + pad)) if end > start else (0, 0)
    # The floor comes from what the trim found silent, not the kept padding.
    noise = np.concatenate([data[:start], data[end:]]) if end > start else data[:0]
    floor = noise_floor(noise, rate, edges) if len(noise) >= MIN_NOISE_S * rate else None

    kept = data[lo:hi]
    duration = len(kept) / rate
    method = "rms" if duration < rms_below_s else "bs1770"
    level = rms_db(kept) if method == "rms" else integrated_loudness(kept, rate)
    if level is None:  # all blocks gated out
        method, level = "rms", rms_db(kept)
    peak = 20 * math.log10(float(np.max(np.abs(kept))) + 1e-12) if len(kept) else -240.0
    gain = target_lufs - level if level > -200 else 0.0
    capped = peak + gain > max_peak_dbfs
    if capped:
        gain = max_peak_dbfs - peak
    audio = resample(kept * 10 ** (gain / 20), rate, RATE_16K)
    return {
        "trim": {"start_s": round(lo / rate, 4), "end_s": round(hi / rate, 4)},
        "duration_s": round(duration, 4),
        "trimmed_s": round((len(data) - len(kept)) / rate, 4),
        "level": {"method": method, "db": round(level, 2), "peak_dbfs": round(peak, 2)},
        "gain_db": round(gain, 2),
        "gain_capped": bool(capped),
        "noise_floor": floor,
        "audio16k": audio.astype(np.float32),
    }


def trim_bounds(data, top_db):
    """Sample bounds of the non-silent part: frames within top_db of the
    loudest frame, as librosa.effects.trim (centred frames)."""
    import numpy as np

    if len(data) == 0:
        return 0, 0
    padded = np.pad(data, FRAME // 2)
    n = 1 + (len(padded) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(padded[idx] ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    loud = np.flatnonzero(db > db.max() - top_db)
    if rms.max() <= 1e-10 or len(loud) == 0:
        return 0, 0
    start = int(loud[0] * HOP)
    end = min(len(data), int((loud[-1] + 1) * HOP))
    return start, end


def band_edges(n, lo=50.0, hi=7600.0):
    """n log-spaced band edges, inside the 16 kHz copy's band so the floor
    means the same for every game."""
    return [round(lo * (hi / lo) ** (k / n), 1) for k in range(n + 1)]


def noise_floor(noise, rate, edges):
    """Level (dBFS) and per-band power spectrum (dB) of the trimmed silence."""
    import numpy as np

    level = rms_db(noise)
    n = SPECTRUM_N * rate // 16_000  # about 32 ms whatever the rate
    if len(noise) < n:
        noise = np.pad(noise, (0, n - len(noise)))
    frames = np.lib.stride_tricks.sliding_window_view(noise, n)[:: n // 2]
    window = np.hanning(n)
    power = np.mean(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0) / np.sum(window**2)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    spectrum = []
    for a, b in zip(edges, edges[1:]):
        inside = (freqs >= a) & (freqs < b)
        p = float(power[inside].mean()) if inside.any() else float(np.interp((a + b) / 2, freqs, power))
        spectrum.append(round(10 * math.log10(p + 1e-20), 2))
    return {"level_db": round(level, 2), "seconds": round(len(noise) / rate, 4), "spectrum": spectrum}


def rms_db(x):
    import numpy as np

    if len(x) == 0:
        return -240.0
    return 10 * math.log10(float(np.mean(np.square(x))) + 1e-24)


def k_weighting(rate):
    """BS.1770 K-weighting as two biquads (shelf, then high-pass), for any
    sample rate (the published coefficients are for 48 kHz only)."""
    # Pre-filter (high shelf), after Mansbridge et al. / pyloudnorm.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = ([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
             [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])  # fmt: skip
    # RLB high-pass.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    high = ([1.0, -2.0, 1.0], [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    return [shelf, high]


def integrated_loudness(x, rate):
    """ITU-R BS.1770-4 integrated loudness of a mono clip in LUFS, or None if
    every 400 ms block is gated out."""
    import numpy as np
    from scipy.signal import lfilter

    y = x
    for b, a in k_weighting(rate):
        y = lfilter(b, a, y)
    block, step = int(0.4 * rate), int(0.1 * rate)
    if len(y) < block:
        return None
    starts = range(0, len(y) - block + 1, step)
    z = np.array([np.mean(y[i : i + block] ** 2) for i in starts])
    lk = -0.691 + 10 * np.log10(z + 1e-24)
    z = z[lk > -70]
    if len(z) == 0:
        return None
    relative = -0.691 + 10 * math.log10(float(z.mean())) - 10
    z = z[-0.691 + 10 * np.log10(z) > relative]
    if len(z) == 0:
        return None
    return -0.691 + 10 * math.log10(float(z.mean()))


def resample(x, rate, to):
    import numpy as np
    from scipy.signal import resample_poly

    if rate == to:
        return np.asarray(x)
    g = math.gcd(rate, to)
    return resample_poly(x, to // g, rate // g)


def summary(records, s, edges):
    flags = {}
    for r in records:
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    floors = [r["noise_floor"]["level_db"] for r in records if r["noise_floor"]]
    return {
        "clips": len(records),
        "flags": flags,
        "packages": packages("numpy", "scipy", "soundfile"),
        "band_edges_hz": edges,
        "settings": {k: s[k] for k in sorted(s) if k != "backend"},
        "gain_db": quantiles([r["gain_db"] for r in records]),
        "noise_floor_db": quantiles(floors),
        "trimmed_s": quantiles([r["trimmed_s"] for r in records]),
    }


def packages(*names):
    from importlib.metadata import version

    return {n: version(n) for n in names}


def quantiles(xs):
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    at = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {"n": len(xs), "min": xs[0], "p05": at(0.05), "median": at(0.5), "p95": at(0.95), "max": xs[-1]}


if __name__ == "__main__":
    main()
