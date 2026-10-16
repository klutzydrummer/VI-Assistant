"""Estimate the suit filter on Tali's voice from her own clips (read-only).

Nothing has been published about how the quarian helmet voice was made, and
it is baked into the shipped audio. A fixed effect leaves the same mark on
every clip while her voice varies, so averaging over thousands of clips
brings it out. Per game, from the raw harvest audio:

  1. EQ: the long-term average spectrum (LTAS), its -20 dB band edges, its
     tilt from 500 Hz to 4 kHz, and a third-octave table.
  2. Echo or comb filter: a delayed copy adds a ripple across the spectrum
     at 1/delay Hz. Her pitch harmonics ripple too, but at a different
     spacing on every clip, so they blur in the average and a fixed delay
     stands out as a cepstral peak at its delay.
  3. Modulation (tremolo, ring modulator, vocoder carrier): the envelope of
     her voice pulses at her pitch, which also varies; a fixed carrier
     shows as a sharp peak in the average envelope spectrum (20-500 Hz;
     below 20 Hz is her syllable rhythm).
  4. Periodicity: how cleanly periodic her voiced frames are. A noise or
     ring-mod layer lowers it. There is no unfiltered baseline, so compare
     games with each other rather than with a fixed number.
  5. A doubled, pitch-shifted layer (fans guess one about 2 semitones up):
     in voiced frames, the spectrum at k * f0 * r for k = 1..3, relative to
     the frame, as a function of the ratio r. Her own harmonics peak at
     r = 1; a second copy at a fixed ratio peaks there too (1.122 for +2
     semitones). k stops at 3 so no r in 1.03-1.3 lands on a harmonic.
  6. Moving notches (a flanger rather than a fixed delay): the comb's delay
     tracked through each clip in 85 ms blocks. A fixed delay stays put; a
     swept one spreads. Reported as the most common delay and the share of
     blocks within 0.05 ms of it.

Peaks are reported with a split-half check (odd and even clips): a real
effect shows in both halves at the same place.

Usage (from the repo root, at nice 19; reads the game disk once):
  python3 jobs/filter_probe.py corpus/tali [--limit N] [--out DIR]
Prints JSON and writes ltas.csv, cepstrum.csv, envelope.csv, harmonics.csv and
summary.json
to DIR (default corpus/tali/analysis/filter). Changes no stage outputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal

GRID_HZ = 5.0  # LTAS frequency grid
F_MAX = 12000.0
ENV_SR = 2000  # envelope sample rate
THIRDS = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
          2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]  # fmt: skip


def grid():
    return np.arange(0.0, F_MAX + GRID_HZ, GRID_HZ)


def active_frames(x, sr, frame=2048, hop=512, floor_db=30.0):
    """Frames within floor_db of the clip's loudest, above -60 dBFS."""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    keep = (db > db.max() - floor_db) & (db > -60)
    return frames[keep]


def clip_spectrum(x, sr):
    """Mean power spectrum of the active part on the common grid, scaled to
    unit total power so loud clips don't dominate; NaN above Nyquist."""
    # Bins exactly on the grid: interpolating between two spacings would
    # itself ripple, and show as a false echo.
    nfft = round(sr / GRID_HZ)
    f, _, z = signal.stft(x, sr, nperseg=nfft, noverlap=nfft * 3 // 4, padded=False, boundary=None)
    p = np.abs(z) ** 2
    if p.shape[1] == 0:
        return None
    e = p.sum(axis=0)
    p = p[:, e > e.max() * 1e-3]
    s = p.mean(axis=1)
    s /= s.sum() + 1e-20
    g = grid()
    out = np.interp(g, f, s, right=np.nan)
    return out / GRID_HZ  # density per Hz on the grid


def clip_envelope(x, sr):
    """Normalised spectrum of the 300-3400 Hz envelope, 1-500 Hz, at 1 Hz."""
    sos = signal.butter(4, [300, min(3400, sr / 2 * 0.95)], btype="band", fs=sr, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
    env = signal.resample_poly(env, ENV_SR, sr)
    if len(env) < ENV_SR // 4:
        return None
    env = env - env.mean()
    f, p = signal.welch(env, ENV_SR, nperseg=min(len(env), ENV_SR), detrend="linear")
    freqs = np.arange(1.0, 501.0)
    q = np.interp(freqs, f, p)
    return q / (q.sum() + 1e-20)


def periodicity(frames, sr):
    """Median peak of the normalised autocorrelation at 2.5-12.5 ms lag
    (pitch 80-400 Hz) over the loudest frames."""
    lo, hi = int(sr * 0.0025), int(sr * 0.0125)
    vals = []
    for fr in frames[:: max(1, len(frames) // 40)]:
        fr = fr - fr.mean()
        spec = np.fft.rfft(fr, 2 * len(fr))
        ac = np.fft.irfft(np.abs(spec) ** 2)[: len(fr)]
        if ac[0] <= 0:
            continue
        vals.append(float(ac[lo:hi].max() / ac[0]))
    return float(np.median(vals)) if vals else None


RATIOS = np.round(np.arange(0.9, 1.3001, 0.005), 3)


def f0_of(fr, sr):
    """Pitch of a frame by autocorrelation (80-400 Hz), or None if the frame
    isn't clearly voiced."""
    lo, hi = int(sr * 0.0025), int(sr * 0.0125)
    fr = fr - fr.mean()
    spec = np.fft.rfft(fr, 2 * len(fr))
    ac = np.fft.irfft(np.abs(spec) ** 2)[: len(fr)]
    if ac[0] <= 0:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] / ac[0] < 0.5:
        return None
    # Parabolic interpolation for a sub-sample lag.
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return sr / (lag + shift)


def harmonic_profile(frames, sr, nfft=8192):
    """Mean spectrum level (dB re the frame's median, 100-3000 Hz) at
    k * f0 * r for k = 1..3, per ratio r in RATIOS, over voiced frames."""
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= 100) & (freqs <= 3000)
    win = np.hanning(frames.shape[1])
    rows = []
    for fr in frames[:: max(1, len(frames) // 40)]:
        f0 = f0_of(fr, sr)
        if f0 is None:
            continue
        db = 20 * np.log10(np.abs(np.fft.rfft((fr - fr.mean()) * win, nfft)) + 1e-12)
        db -= np.median(db[band])
        rows.append(np.mean([np.interp(k * f0 * RATIOS, freqs, db) for k in (1, 2, 3)], axis=0))
    return np.mean(rows, axis=0) if rows else None


def comb_track(x, sr, nperseg=1024, block=8, lo_ms=0.45, hi_ms=1.3):
    """The comb's delay (ms) in each 85 ms block of a clip: the peak of the
    cepstrum of the block's log spectrum (200-8000 Hz) between lo_ms and
    hi_ms."""
    f, _, z = signal.stft(x, sr, nperseg=nperseg, noverlap=nperseg * 3 // 4, padded=False, boundary=None)
    p = np.abs(z) ** 2
    if p.shape[1] < block:
        return []
    e = p.sum(axis=0)
    loud = e > e.max() * 1e-2
    band = (f >= 200) & (f <= 8000)
    df = f[1] - f[0]
    out = []
    for i in range(0, p.shape[1] - block + 1, block):
        sel = loud[i : i + block]
        if sel.sum() < block // 2:
            continue
        y = np.log(p[band, i : i + block][:, sel].mean(axis=1) + 1e-20)
        y = (y - y.mean()) * np.hanning(len(y))
        n = 16 * len(y)
        c = np.abs(np.fft.rfft(y, n))
        q = np.fft.rfftfreq(n, d=df) * 1000.0
        keep = (q >= lo_ms) & (q <= hi_ms)
        # Above its own slope (formants make the low end tall), and not at
        # an edge of the range, where a slope would put it.
        ck = c[keep] - signal.medfilt(c[keep], 31)
        i = int(np.argmax(ck))
        mad = np.median(np.abs(ck - np.median(ck))) * 1.4826 + 1e-20
        # Only blocks with a clear comb: a voice alone mostly has none.
        if 0 < i < len(ck) - 1 and ck[i] / mad > 5:
            out.append(float(q[keep][i]))
        else:
            out.append(None)
    return out


def analyse(x, sr):
    """Per-clip measurements, or None if the clip is too short or silent."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    frames = active_frames(x, sr)
    if len(frames) < 2:
        return None
    # The whole clip: clip_spectrum drops its own quiet frames, and joining
    # overlapping frames would repeat audio at a fixed lag, a false echo.
    spec = clip_spectrum(x, sr)
    env = clip_envelope(x, sr)
    if spec is None:
        return None
    return {"spec": spec, "env": env, "periodicity": periodicity(frames, sr), "sr": sr,
            "harmonics": harmonic_profile(frames, sr), "comb": comb_track(x, sr)}  # fmt: skip


def smooth_octave(g, db, fraction=3):
    """Fractional-octave smoothing of a dB curve on the linear grid."""
    out = np.full_like(db, np.nan)
    for i, f in enumerate(g):
        if f <= 0 or np.isnan(db[i]):
            continue
        lo, hi = f * 2 ** (-0.5 / fraction), f * 2 ** (0.5 / fraction)
        sel = (g >= lo) & (g <= hi) & ~np.isnan(db)
        out[i] = np.mean(db[sel])
    return out


def ltas_summary(g, spec):
    db = 10 * np.log10(spec + 1e-30)
    sm = smooth_octave(g, db)
    ok = (g >= 50) & ~np.isnan(sm)
    peak = np.nanmax(sm[ok])
    within = g[ok][sm[ok] >= peak - 20]
    band = (g >= 500) & (g <= 4000) & ~np.isnan(sm)
    tilt = np.polyfit(np.log2(g[band]), sm[band], 1)[0] if band.sum() > 2 else None
    thirds = {}
    for c in THIRDS:
        lo, hi = c * 2 ** (-1 / 6), c * 2 ** (1 / 6)
        sel = (g >= lo) & (g < hi) & ~np.isnan(db)
        if sel.any():
            thirds[str(c)] = round(float(10 * np.log10(np.mean(spec[sel]) + 1e-30) - peak), 1)
    return {"peak_hz": float(g[ok][np.nanargmax(sm[ok])]),
            "band_minus20db_hz": [float(within.min()), float(within.max())] if len(within) else None,
            "tilt_db_per_octave_500_4k": round(float(tilt), 2) if tilt is not None else None,
            "third_octaves_db_re_peak": thirds}, sm  # fmt: skip


def cepstrum(g, spec, fmin=100.0, fmax=8000.0):
    """Quefrency (ms) and magnitude of the ripple in the EQ-flattened log
    spectrum, from 0.5 to 50 ms."""
    db = 10 * np.log10(spec + 1e-30)
    flat = db - smooth_octave(g, db, fraction=3)
    sel = (g >= fmin) & (g <= fmax) & ~np.isnan(flat)
    y = flat[sel] - flat[sel].mean()
    y = y * np.hanning(len(y))
    c = np.abs(np.fft.rfft(y, n=8 * len(y)))
    q = np.fft.rfftfreq(8 * len(y), d=GRID_HZ) * 1000.0  # ms
    keep = (q >= 0.5) & (q <= 50)
    return q[keep], c[keep]


def ripple_db(g, spec, q_ms, fmin=150.0, fmax=5000.0, bins=20):
    """Peak-to-peak depth (dB) of the EQ-flattened average spectrum folded
    at a spacing of 1/q_ms. A real comb has deep notches (a delayed copy at
    30% level gives about 5 dB, an equal one 10 dB or more); a large
    cepstral z can still mean a ripple of a dB or less."""
    db = 10 * np.log10(spec + 1e-30)
    flat = db - smooth_octave(g, db, fraction=3)
    sel = (g >= fmin) & (g <= fmax) & ~np.isnan(flat)
    phase = (g[sel] * q_ms / 1000.0) % 1.0
    idx = np.minimum((phase * bins).astype(int), bins - 1)
    means = np.array([flat[sel][idx == b].mean() for b in range(bins)])
    return round(float(means.max() - means.min()), 2)


def peaks(x, y, min_z=6.0, top=5):
    """Peaks standing out from a running median, as robust z-scores."""
    base = signal.medfilt(y, 101 if len(y) > 101 else (len(y) // 2) * 2 - 1)
    r = y - base
    mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-20
    z = r / mad
    idx, _ = signal.find_peaks(z, height=min_z, distance=max(1, len(y) // 200))
    idx = idx[np.argsort(-z[idx])][:top]
    return [{"at": round(float(x[i]), 2), "z": round(float(z[i]), 1)} for i in idx]


def both_halves(a, b, tol):
    """Peaks of half a that half b also has within tol."""
    return [p for p in a if any(abs(p["at"] - q["at"]) <= tol for q in b)]


def summarise(results):
    """results: list of per-clip dicts from analyse(). The game-level report."""
    g = grid()
    halves = [results[0::2], results[1::2]]

    def mean_spec(rs):
        return np.nanmean(np.stack([r["spec"] for r in rs]), axis=0)

    def mean_env(rs):
        envs = [r["env"] for r in rs if r["env"] is not None]
        return np.mean(np.stack(envs), axis=0) if envs else None

    spec = mean_spec(results)
    ltas, smooth = ltas_summary(g, spec)
    q, c = cepstrum(g, spec)
    cep = {"all": [p | {"ripple_db": ripple_db(g, spec, p["at"])} for p in peaks(q, c)]}
    env = mean_env(results)
    freqs = np.arange(1.0, 501.0)
    # From 20 Hz: below that is syllable rhythm, which is hers.
    lo = 19
    mod = {"all": peaks(freqs[lo:], 10 * np.log10(env[lo:] + 1e-30))} if env is not None else {}
    if min(len(h) for h in halves) >= 20:
        hc = [peaks(*cepstrum(g, mean_spec(h))) for h in halves]
        cep["in_both_halves"] = both_halves(hc[0], hc[1], 0.15)
        he = [mean_env(h) for h in halves]
        if all(e is not None for e in he):
            hm = [peaks(freqs[lo:], 10 * np.log10(e[lo:] + 1e-30)) for e in he]
            mod["in_both_halves"] = both_halves(hm[0], hm[1], 2.0)
    per = [r["periodicity"] for r in results if r["periodicity"] is not None]

    def mean_harm(rs):
        hs = [r["harmonics"] for r in rs if r.get("harmonics") is not None]
        return np.mean(np.stack(hs), axis=0) if hs else None

    harm = mean_harm(results)
    doubled = None
    if harm is not None:
        # From one semitone up: closer in, her own harmonic's skirt leaks.
        side = RATIOS >= 1.06

        def best_of(h):
            i = int(np.argmax(np.where(side, h, -np.inf)))
            return float(RATIOS[i]), float(h[i] - np.median(h[side]))

        at1 = float(harm[np.argmin(np.abs(RATIOS - 1.0))])
        ratio, prominence = best_of(harm)
        doubled = {"best_ratio": ratio,
                   "semitones": round(12 * float(np.log2(ratio)), 2),
                   "prominence_db": round(prominence, 2),
                   # At an end of the range it's the slope of her own
                   # harmonic or of the next one, not a second layer.
                   "at_range_end": bool(ratio <= 1.07 or ratio >= 1.29),
                   "own_harmonic_db": round(at1 - float(np.median(harm[side])), 2)}  # fmt: skip
        if min(len(h) for h in halves) >= 20:
            hh = [mean_harm(h) for h in halves]
            if all(h is not None for h in hh):
                doubled["halves"] = [{"ratio": r, "prominence_db": round(p, 2)} for r, p in map(best_of, hh)]
    tracked = [d for r in results for d in r.get("comb", [])]
    delays = np.array([d for d in tracked if d is not None])
    moving = None
    if len(delays) >= 20:
        hist, edges = np.histogram(delays, bins=np.arange(0.45, 1.3001, 0.01))
        mode = float(edges[np.argmax(hist)] + 0.005)
        found = [[d for d in r.get("comb", []) if d is not None] for r in results]
        spreads = [np.subtract(*np.percentile(f, [75, 25])) for f in found if len(f) >= 4]
        moving = {"blocks": len(tracked),
                  "share_with_a_clear_comb": round(len(delays) / len(tracked), 3),
                  "most_common_ms": round(mode, 3),
                  "share_within_0_05ms": round(float(np.mean(np.abs(delays - mode) <= 0.05)), 3),
                  "median_within_clip_iqr_ms": round(float(np.median(spreads)), 3) if spreads else None}  # fmt: skip
    return {"clips": len(results),
            "sample_rates": sorted({r["sr"] for r in results}),
            "ltas": ltas,
            "echo_ms": cep,
            "modulation_hz": mod,
            "doubled_layer": doubled,
            "comb_over_time": moving,
            "periodicity": {"median": round(float(np.median(per)), 3),
                            "p10": round(float(np.quantile(per, 0.1)), 3),
                            "p90": round(float(np.quantile(per, 0.9)), 3)} if per else None,
            "_curves": {"spec": spec, "smooth": smooth, "cep": (q, c), "env": env, "harm": harm}}  # fmt: skip


def current_out(corpus, stage):
    d = Path(corpus) / "stages" / stage
    return d / (d / "current").read_text().strip() / "out"


def main():
    import soundfile as sf

    from vi_job import load_clips, spread

    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="corpus/tali")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    harvest = current_out(a.corpus, "harvest")
    os.environ["VI_INPUT_HARVEST"] = str(harvest)
    try:
        os.environ["VI_INPUT_CLEAN"] = str(current_out(a.corpus, "clean"))  # leave out what clean dropped
    except FileNotFoundError:
        pass
    clips = load_clips()
    if a.limit:
        clips = spread(clips, a.limit)
    by_game = {}
    for i, clip in enumerate(clips):
        raw = harvest / clip.line["audio"]["path"]  # as shipped, before clean touched it
        try:
            x, sr = sf.read(raw, always_2d=False)
        except Exception as e:  # noqa: BLE001 - report and carry on
            print(f"{clip.id}: {e}", file=sys.stderr)
            continue
        r = analyse(x, sr)
        if r is not None:
            by_game.setdefault(clip.game or "?", []).append(r)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(clips)}", file=sys.stderr, flush=True)
    games = {gm: summarise(rs) for gm, rs in sorted(by_game.items())}
    games["all"] = summarise([r for rs in by_game.values() for r in rs])

    out = Path(a.out) if a.out else Path(a.corpus) / "analysis" / "filter"
    out.mkdir(parents=True, exist_ok=True)
    g = grid()
    names = list(games)
    with open(out / "ltas.csv", "w") as f:
        f.write("hz," + ",".join(f"{n}_db,{n}_smooth_db" for n in names) + "\n")
        for i, hz in enumerate(g):
            vals = []
            for n in names:
                cv = games[n]["_curves"]
                vals += [10 * np.log10(cv["spec"][i] + 1e-30), cv["smooth"][i]]
            f.write(f"{hz:g}," + ",".join("" if np.isnan(v) else f"{v:.2f}" for v in vals) + "\n")
    with open(out / "cepstrum.csv", "w") as f:
        q = games["all"]["_curves"]["cep"][0]
        f.write("ms," + ",".join(names) + "\n")
        for i, ms in enumerate(q):
            f.write(f"{ms:.3f}," + ",".join(f"{games[n]['_curves']['cep'][1][i]:.4g}" for n in names) + "\n")
    with open(out / "envelope.csv", "w") as f:
        f.write("hz," + ",".join(names) + "\n")
        for i in range(500):
            vals = [games[n]["_curves"]["env"] for n in names]
            f.write(f"{i + 1}," + ",".join("" if v is None else f"{10 * np.log10(v[i] + 1e-30):.2f}" for v in vals) + "\n")
    with open(out / "harmonics.csv", "w") as f:
        f.write("ratio," + ",".join(names) + "\n")
        for i, r in enumerate(RATIOS):
            vals = [games[n]["_curves"]["harm"] for n in names]
            f.write(f"{r:g}," + ",".join("" if v is None else f"{v[i]:.2f}" for v in vals) + "\n")
    report = {n: {k: v for k, v in s.items() if k != "_curves"} for n, s in games.items()}
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"curves in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
