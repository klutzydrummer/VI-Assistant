"""Measure Tali's suit effect against Ash Sroka's own clean voice, and fit a
model of the effect chain (read-only on the corpus).

filter_probe found the suit is mostly an EQ per game, with no echo, flanger
or modulation, and could not decide whether a quiet doubled layer (fans
guess +2 semitones) is in there: without a clean recording of the same
speaker there is nothing to compare her harmonics with. This uses one:
recordings of Ash Sroka speaking without the suit (ideally doing the Tali
voice at a panel or stream), put in a folder outside the corpus. They are
only measured, never trained or cloned on, and never leave this machine.

Two parts:

  1. Reference EQ. Per game, the third-octave average spectrum of her
     comparable clips minus that of each reference source. Each source adds
     its own mic, room and codec, so the suit EQ is the median over sources
     and a band is trusted only where the sources agree within
     TRUST_SPREAD_DB. Undoing it (restore_db) is for analysis only, in the
     same form as eq_match's correction.

  2. Chain fit. A forward model of the suit, run on the clean voice:
       doubled layer (pitch ratio, level) -> soft-clip drive -> noise -> EQ
     The EQ is solved in closed form (her clips' spectrum minus the
     spectrum of the chain so far); the rest is a grid search matching
     what an EQ can't change: her harmonic profile (filter_probe's measure
     at k * f0 * r) and how periodic her voice is. Whether the layer is
     real is decided on held-out halves (fit on odd clips, score on even),
     and two planted self-tests say whether that decision can be trusted:
     a known layer at --planted-db must be found, and a chain without one
     must not be. The fitted chain run forwards is also the suit effect a
     restored-voice clone would need (the TTS backup).

Usage (from the repo root, at nice 19; reads the game disk once):
  python3 jobs/suit_fit.py corpus/tali --ref ~/vi-reference/ash-sroka
      [--per-game 300] [--chunks 160] [--planted-db -12] [--games LE1,LE3]
      [--out DIR]
--ref holds one subfolder per source (a panel, a stream, an interview) of
WAV or FLAC files with only her voice in them; a flat folder is one source.
Writes suit_fit.json, WAVs and index.html to DIR (default
corpus/tali/analysis/suit_fit). Changes no stage outputs.
"""

import argparse
import html
import json
import os
import sys
import warnings
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal

import eq_match as eq
import filter_probe as fp

SR = 24000  # everything is compared at one rate; the games cut off near 10 kHz
CHUNK_S = 3.0  # reference recordings are cut into clips this long
TRUST_SPREAD_DB = 3.0
LAYER_RATIOS = np.round(np.arange(1.03, 1.3001, 0.01), 3)
LAYER_DB = [-24.0, -18.0, -12.0, -6.0, 0.0]
DRIVES = [0.0, 1.5, 3.0]
SNRS_DB = [None, 40.0, 30.0, 24.0, 18.0]
MARGIN = 1.0  # held-out improvement, in standard errors, for a layer to count
AUDIO = (".wav", ".flac", ".ogg")


def to_sr(x, sr):
    if sr == SR:
        return x
    g = gcd(SR, sr)
    return signal.resample_poly(x, SR // g, sr // g)


def pitch_shift(x, ratio, n=2048, hop=512):
    """x a ratio higher in pitch, same length: a phase-vocoder stretch by
    ratio, then resampled back to len(x)."""
    _, _, d = signal.stft(x, nperseg=n, noverlap=n - hop)
    adv = 2 * np.pi * hop * np.arange(d.shape[0]) / n
    steps = np.arange(0, d.shape[1] - 1, 1 / ratio)
    phase = np.angle(d[:, 0])
    out = np.empty((d.shape[0], len(steps)), complex)
    for j, t in enumerate(steps):
        i, a = int(t), t - int(t)
        out[:, j] = ((1 - a) * np.abs(d[:, i]) + a * np.abs(d[:, i + 1])) * np.exp(1j * phase)
        dp = np.angle(d[:, i + 1]) - np.angle(d[:, i]) - adv
        phase = phase + adv + dp - 2 * np.pi * np.round(dp / (2 * np.pi))
    _, y = signal.istft(out, nperseg=n, noverlap=n - hop)
    return signal.resample(y, len(x))


def drive(x, amount):
    """tanh soft clipping; the peak stays where it was."""
    if amount <= 0:
        return x
    p = np.abs(x).max() + 1e-12
    return np.tanh(amount * x / p) * p / np.tanh(amount)


def active_rms(x):
    fr = fp.active_frames(x, SR)
    return float(np.sqrt(np.mean(fr**2))) if len(fr) else float(np.sqrt(np.mean(x**2)))


def pre_eq(x, p, rng):
    """The chain before its EQ: the doubled layer, the drive, the noise."""
    y = x
    if p.get("layer_ratio"):
        y = y + 10 ** (p["layer_db"] / 20) * pitch_shift(x, p["layer_ratio"])
    y = drive(y, p.get("drive", 0.0))
    if p.get("snr_db") is not None:
        y = y + rng.standard_normal(len(y)) * active_rms(y) * 10 ** (-p["snr_db"] / 20)
    return y


def curve(xs, sr=SR):
    """Third-octave average spectrum in dB, at unit total power."""
    return eq.game_curves({"x": [s for s in (fp.clip_spectrum(x, sr) for x in xs) if s is not None]})["x"]


def chain_eq(target, pre):
    """The EQ taking the chain-so-far's spectrum to her clips': free to cut
    to -60 dB (the games' band edges), boost at most +12."""
    d = np.nan_to_num(target - pre, nan=-60.0)
    band = (fp.grid() >= eq.F_LO) & (fp.grid() <= eq.F_HI)
    return np.clip(d - np.mean(d[band]), -60.0, eq.MAX_DB)


def apply_chain(x, sr, chain, seed=0):
    """Run audio through a fitted chain (suit_fit.json's chains[game]): the
    suit effect for a clean or restored voice."""
    y = pre_eq(np.asarray(x, dtype=np.float64), chain["params"], np.random.default_rng(seed))
    return eq.apply(y, eq.fir(np.asarray(chain["eq_db"]), sr))


def simulate(ref, p, target, seed=0):
    rng = np.random.default_rng(seed)
    pre = [pre_eq(x, p, rng) for x in ref]
    e = chain_eq(target, curve(pre))
    h = eq.fir(e, SR)
    return [eq.apply(y, h) for y in pre], e


def stats(xs):
    """What an EQ can't change much: the harmonic profile and periodicity."""
    harm, per = [], []
    for x in xs:
        fr = fp.active_frames(x, SR)
        if len(fr) < 2:
            continue
        h = fp.harmonic_profile(fr, SR)
        if h is not None:
            harm.append(h)
        q = fp.periodicity(fr, SR)
        if q is not None:
            per.append(q)
    return {"harm": np.mean(harm, axis=0), "per": float(np.median(per))}


def scale_of(xs):
    """Standard errors of the stats, from the gap between odd and even clips."""
    a, b = stats(xs[0::2]), stats(xs[1::2])
    dh = (a["harm"] - np.median(a["harm"])) - (b["harm"] - np.median(b["harm"]))
    return {"harm": max(float(np.sqrt(np.mean(dh**2))) / 2, 0.05), "per": max(abs(a["per"] - b["per"]) / 2, 0.003)}


def distance(a, b, scale):
    dh = (a["harm"] - np.median(a["harm"])) - (b["harm"] - np.median(b["harm"]))
    return float(np.sqrt(np.mean(dh**2)) / scale["harm"] + abs(a["per"] - b["per"]) / scale["per"])


def fit(ref, game, scale, layer=True, ratios=LAYER_RATIOS):
    """Coordinate search: the layer's ratio (at -12 dB), its level, then
    drive and noise together. Returns params and their distance."""
    target, want = curve(game), stats(game)

    def score(p):
        return distance(stats(simulate(ref, p, target)[0]), want, scale)

    best = {"layer_ratio": None, "layer_db": None, "drive": 0.0, "snr_db": None}
    trace = {}
    if layer:
        by_ratio = {float(r): score(best | {"layer_ratio": float(r), "layer_db": -12.0}) for r in ratios}
        r = min(by_ratio, key=by_ratio.get)
        by_db = {d: score(best | {"layer_ratio": r, "layer_db": d}) for d in LAYER_DB}
        best |= {"layer_ratio": r, "layer_db": min(by_db, key=by_db.get)}
        trace = {"by_ratio": {f"{k:g}": round(v, 2) for k, v in by_ratio.items()},
                 "by_layer_db": {f"{k:g}": round(v, 2) for k, v in by_db.items()}}  # fmt: skip
    by_dn = {(dr, sn): score(best | {"drive": dr, "snr_db": sn}) for dr in DRIVES for sn in SNRS_DB}
    dr, sn = min(by_dn, key=by_dn.get)
    trace["by_drive_snr"] = {f"{d:g}/{s}": round(v, 2) for (d, s), v in by_dn.items()}
    return best | {"drive": dr, "snr_db": sn}, by_dn[(dr, sn)], trace


def layer_test(ref, game, scale, ratios=LAYER_RATIOS):
    """Is a doubled layer real? Fit with and without one on odd clips of
    both, score on even ones; the layer counts if it wins there by MARGIN
    and a fit on the even half puts it at the same ratio."""
    ra, rb, ga, gb = ref[0::2], ref[1::2], game[0::2], game[1::2]
    with_l, _, _ = fit(ra, ga, scale, True, ratios)
    without, _, _ = fit(ra, ga, scale, False)
    target, want = curve(gb), stats(gb)
    held = {k: distance(stats(simulate(rb, p, target)[0]), want, scale) for k, p in (("with", with_l), ("without", without))}
    other, _, _ = fit(rb, gb, scale, True, ratios)
    same_place = abs(with_l["layer_ratio"] - other["layer_ratio"]) <= 0.02
    return {"supported": bool(held["with"] < held["without"] - MARGIN and same_place),
            "held_out_distance": {k: round(v, 2) for k, v in held.items()},
            "ratio_odd_half": with_l["layer_ratio"], "ratio_even_half": other["layer_ratio"],
            "layer_db_odd_half": with_l["layer_db"]}  # fmt: skip


def reference_eq(sources, game_curve):
    """The suit EQ for one game against each reference source, their median
    and spread, and the analysis-only correction that undoes it."""
    g = fp.grid()
    band = (g >= eq.F_LO) & (g <= eq.F_HI)
    centres = [t for t in fp.THIRDS if eq.F_LO <= t <= eq.F_HI]
    per = {}
    for s, xs in sources.items():
        d = game_curve - curve(xs)
        # Aligned on the median over third-octave centres: on the linear
        # grid most points sit above 3 kHz, and one source's resonance would
        # shift its whole curve.
        per[s] = d - np.nanmedian(np.interp(centres, g, d))
    stack = np.stack(list(per.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # grid points no source reaches
        med = np.nanmedian(stack, axis=0)
        spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0) if len(per) > 1 else np.full(len(g), np.nan)
    trusted = band & (spread <= TRUST_SPREAD_DB)
    restore = np.where(band, np.clip(-np.nan_to_num(med), -eq.MAX_DB, eq.MAX_DB), 0.0)
    restore[band] -= np.mean(restore[band])
    return {"per_source": per, "suit_db": med, "spread_db": spread, "trusted": trusted, "restore_db": restore}


def thirds(curve_db):
    g = fp.grid()
    return {str(t): (None if np.isnan(v := float(np.interp(t, g, curve_db))) else round(v, 1))
            for t in fp.THIRDS if eq.F_LO <= t <= eq.F_HI}  # fmt: skip


def load_reference(root, chunk_s=CHUNK_S):
    """{source: [clips]}: each file mono at SR, cut into chunk_s clips, the
    clips with too little voice dropped, each at the same peak."""
    import soundfile as sf

    root = Path(root).expanduser()
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()] or [root]
    n = int(chunk_s * SR)
    out = {}
    for d in dirs:
        for f in sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO):
            x, sr = sf.read(f, always_2d=False)
            x = to_sr(np.asarray(x.mean(axis=1) if x.ndim > 1 else x, dtype=np.float64), sr)
            for i in range(0, len(x) - n + 1, n):
                c = x[i : i + n]
                if len(fp.active_frames(c, SR)) >= 10:
                    out.setdefault(d.name if d != root else "ref", []).append(0.3 * c / (np.abs(c).max() + 1e-12))
    return out


def load_game(corpus, per_game, games=None):
    """{game: [(clip, audio at SR)]}: the comparable clips as shipped, spread
    evenly, at most per_game each."""
    import soundfile as sf

    import confounds as cf
    from vi_job import load_clips

    harvest = fp.current_out(corpus, "harvest")
    os.environ["VI_INPUT_HARVEST"] = str(harvest)
    for stage in ("prep", "clean"):
        try:
            os.environ[f"VI_INPUT_{stage.upper()}"] = str(fp.current_out(corpus, stage))
        except FileNotFoundError:
            pass
    clips = load_clips()
    comparable, rule = cf.Meta(clips).comparable(np.random.default_rng(cf.SEED))
    by_game = {}
    for i in comparable:
        if games is None or clips[i].game in games:
            by_game.setdefault(clips[i].game, []).append(clips[i])
    out = {}
    for game, cs in sorted(by_game.items()):
        pick = [cs[j] for j in np.unique(np.linspace(0, len(cs) - 1, min(per_game, len(cs))).astype(int))]
        out[game] = []
        for c in pick:
            x, sr = sf.read(harvest / c.line["audio"]["path"], always_2d=False)
            x = to_sr(np.asarray(x.mean(axis=1) if x.ndim > 1 else x, dtype=np.float64), sr)
            out[game].append((c, 0.3 * x / (np.abs(x).max() + 1e-12)))
        print(f"{game}: {len(out[game])} clips", file=sys.stderr, flush=True)
    return out, rule


def page(rows, out):
    parts = ["<!doctype html><meta charset='utf-8'><title>Suit fit</title>",
             "<style>body{font:15px system-ui;margin:24px auto;max-width:900px;padding:0 16px}"
             "td{padding:6px 10px;border-bottom:1px solid #ddd;vertical-align:top}"
             "audio{width:260px}.t{color:#666;font-size:13px}</style>",
             "<h1>Suit fit</h1>",
             "<p class='t'>Per game: her clips as shipped next to the reference EQ undone (analysis only), "
             "and the clean reference voice next to it run through that game's fitted chain. "
             "All at the same loudness.</p>"]  # fmt: skip
    for game in sorted({r["game"] for r in rows}):
        parts.append(f"<h2>{html.escape(game)}</h2><table>")
        for r in (r for r in rows if r["game"] == game):
            parts.append(f"<tr><td><b>{html.escape(r['id'])}</b><div class='t'>{html.escape(r['text'])}</div></td>"
                         f"<td>{r['a_label']}<br><audio controls preload='none' src='{r['a']}'></audio></td>"
                         f"<td>{r['b_label']}<br><audio controls preload='none' src='{r['b']}'></audio></td></tr>")  # fmt: skip
        parts.append("</table>")
    (out / "index.html").write_text("\n".join(parts))


def main():
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="corpus/tali")
    ap.add_argument("--ref", required=True, help="folder of clean reference recordings, one subfolder per source")
    ap.add_argument("--per-game", type=int, default=300)
    ap.add_argument("--chunks", type=int, default=160, help="reference clips used in the chain fit")
    ap.add_argument("--planted-db", type=float, default=-12.0)
    ap.add_argument("--games", default=None, help="comma-separated, default all")
    ap.add_argument("--listen", type=int, default=4, help="clips per game on the page")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sources = load_reference(a.ref)
    if not sources:
        sys.exit(f"no usable reference audio in {a.ref}")
    print({s: len(v) for s, v in sources.items()}, file=sys.stderr)
    # The chain fit pools the sources evenly, interleaved so odd/even halves
    # both see every source.
    per_src = max(1, a.chunks // len(sources))
    pooled = [x for group in zip(*[v[:per_src] for v in sources.values()]) for x in group] if len(sources) > 1 else sources["ref"][: a.chunks]
    games, rule = load_game(a.corpus, a.per_game, a.games.split(",") if a.games else None)

    out = Path(a.out) if a.out else Path(a.corpus) / "analysis" / "suit_fit"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    report = {"reference": {s: {"clips": len(v), "minutes": round(len(v) * CHUNK_S / 60, 1)} for s, v in sources.items()},
              "fit_reference_clips": len(pooled), "comparable_rule": rule,
              "trust_spread_db": TRUST_SPREAD_DB, "games": {}}  # fmt: skip
    chains, restores, rows = {}, {}, []
    for game, items in games.items():
        xs = [x for _, x in items]
        print(f"{game}: reference EQ", file=sys.stderr, flush=True)
        r = reference_eq(sources, curve(xs))
        restores[game] = r["restore_db"]
        scale = scale_of(xs)
        print(f"{game}: chain fit", file=sys.stderr, flush=True)
        params, dist, trace = fit(pooled, xs, scale)
        _, e = simulate(pooled, params, curve(xs))
        chains[game] = {"params": params, "eq_db": e.tolist()}
        print(f"{game}: is the layer real?", file=sys.stderr, flush=True)
        real = layer_test(pooled, xs, scale)
        # Self-tests on the reference alone: half of it through a known
        # chain stands in for her clips, the other half is the reference.
        rng = np.random.default_rng(0)
        known = {"drive": params["drive"], "snr_db": params["snr_db"]}
        planted = {}
        for name, p in (("layer", known | {"layer_ratio": 1.122, "layer_db": a.planted_db}), ("none", known)):
            print(f"{game}: self-test, {name}", file=sys.stderr, flush=True)
            fake = [eq.apply(pre_eq(x, p, rng), eq.fir(e, SR)) for x in pooled[0::2]]
            planted[name] = layer_test(pooled[1::2], fake, scale_of(fake))
        trustworthy = planted["layer"]["supported"] and not planted["none"]["supported"]
        report["games"][game] = {
            "clips": len(xs),
            "reference_eq": {"suit_db_at_third_octaves": thirds(r["suit_db"]),
                             "spread_db_at_third_octaves": thirds(r["spread_db"]),
                             "trusted_share_of_band": round(float(r["trusted"].sum() / max(1, ((fp.grid() >= eq.F_LO) & (fp.grid() <= eq.F_HI)).sum())), 2),
                             "per_source_at_third_octaves": {s: thirds(d) for s, d in r["per_source"].items()}},  # fmt: skip
            "chain": {"params": params, "distance": round(dist, 2), "eq_db_at_third_octaves": thirds(e), "search": trace},
            "doubled_layer": real | {"self_test": planted, "test_trustworthy": trustworthy,
                                     "verdict": ("unsettled: the self-tests failed" if not trustworthy
                                                 else "present" if real["supported"] else
                                                 f"absent, or quieter than {a.planted_db:g} dB")},  # fmt: skip
        }
        print(json.dumps({game: report["games"][game]["doubled_layer"]["verdict"]}), file=sys.stderr, flush=True)
        # To hear: her clips with the reference EQ undone, and the clean
        # voice through the chain.
        h = eq.fir(r["restore_db"], SR)
        for j in np.linspace(0, len(items) - 1, min(a.listen, len(items))).astype(int):
            c, x = items[j]
            name = c.id.replace("/", "_")
            sf.write(out / "audio" / f"{name}.orig.wav", eq.same_loudness(x, x), SR)
            sf.write(out / "audio" / f"{name}.restored.wav", eq.same_loudness(x, eq.apply(x, h)), SR)
            rows.append({"id": c.id, "game": game, "text": c.text, "a_label": "As shipped", "a": f"audio/{name}.orig.wav",
                         "b_label": "Reference EQ undone", "b": f"audio/{name}.restored.wav"})  # fmt: skip
        for j in np.linspace(0, len(pooled) - 1, min(a.listen, len(pooled))).astype(int):
            x = pooled[j]
            name = f"ref{j:03d}.{game}"
            sf.write(out / "audio" / f"{name}.clean.wav", eq.same_loudness(x, x), SR)
            sf.write(out / "audio" / f"{name}.suit.wav", eq.same_loudness(x, apply_chain(x, SR, chains[game])), SR)
            rows.append({"id": f"reference clip {j}", "game": game, "text": "", "a_label": "Clean reference",
                         "a": f"audio/{name}.clean.wav", "b_label": "Through the fitted chain", "b": f"audio/{name}.suit.wav"})  # fmt: skip

    (out / "suit_fit.json").write_text(json.dumps(report | {"grid_hz": fp.grid().tolist(), "sr": SR, "chains": chains,
        "correction_db": {k: v.tolist() for k, v in restores.items()}}))  # fmt: skip
    print(json.dumps(report, indent=1))
    page(rows, out)
    print(f"open {out / 'index.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
