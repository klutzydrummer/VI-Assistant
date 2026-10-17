"""Per-game EQ match, with a page to hear it (read-only on the corpus).

corpus-pipeline.md lists an optional per-game EQ match to a common long-term
spectrum, fitted on the comparable-clip subset because a game with more
high-arousal lines is brighter for real reasons. filter_probe found the suit
filter is mostly that: a different EQ per game (LE1 darker, with a dip near
800 Hz), with no echo, flanger or modulation.

This fits it and lets Brandon hear it before anything uses it:
  - each game's average spectrum over the comparable clips, smoothed to a
    third of an octave;
  - the target, the mean of the games' curves;
  - per game, a correction of target minus game, from 100 Hz to 9 kHz,
    capped at +-12 dB, as a linear-phase FIR filter;
  - a local page with clips from every game, each original next to its
    matched version at the same loudness, so the louder one doesn't win.

Usage (from the repo root, at nice 19):
  python3 jobs/eq_match.py corpus/tali [--per-game 8] [--out DIR]
Writes eq_match.json (the curves and corrections), WAVs and index.html to DIR
(default corpus/tali/analysis/eq_match). Open index.html in a browser. Changes
no stage outputs; the audio stays on this machine.
"""

import argparse
import html
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

import filter_probe as fp

F_LO, F_HI = 100.0, 9000.0
MAX_DB = 12.0
TAPS = 1025


def game_curves(spectra_by_game):
    """Per game, the third-octave-smoothed average spectrum in dB on the
    probe's grid, each scaled to the same total power."""
    g = fp.grid()
    out = {}
    for game, specs in spectra_by_game.items():
        s = np.nanmean(np.stack(specs), axis=0)
        s /= np.nansum(s)
        out[game] = fp.smooth_octave(g, 10 * np.log10(s + 1e-30))
    return out


def corrections(curves):
    """Target (mean of the games) minus each game, in dB, inside F_LO-F_HI
    and capped; flat (0 dB) outside, so the filter doesn't chase the band
    edges or the codec cliff."""
    g = fp.grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # grid points every game lacks
        target = np.nanmean(np.stack(list(curves.values())), axis=0)
    band = (g >= F_LO) & (g <= F_HI)
    out = {}
    for game, c in curves.items():
        d = np.where(band, target - c, 0.0)
        d = np.clip(np.nan_to_num(d), -MAX_DB, MAX_DB)
        # Remove the overall offset so the match changes colour, not level.
        d[band] -= np.mean(d[band])
        out[game] = d
    return target, out


def fir(correction_db, sr, taps=TAPS):
    """A linear-phase FIR with the given gain curve on the probe's grid."""
    g = fp.grid()
    nyq = sr / 2
    f = np.concatenate([g[g < nyq], [nyq]])
    gain = 10 ** (np.interp(f, g, correction_db) / 20)
    return signal.firwin2(taps, f / nyq, gain)


def apply(x, h):
    """Filter without shifting in time (the FIR's delay is taken back out)."""
    y = signal.fftconvolve(x, h, mode="full")
    d = (len(h) - 1) // 2
    return y[d : d + len(x)]


def same_loudness(ref, y):
    r = np.sqrt(np.mean(ref**2)) + 1e-12
    out = y * r / (np.sqrt(np.mean(y**2)) + 1e-12)
    peak = np.abs(out).max()
    return out * (0.98 / peak) if peak > 0.98 else out


def page(rows, games, out):
    """index.html: per game, each clip with Original and EQ-matched players."""
    parts = ["<!doctype html><meta charset='utf-8'><title>EQ match</title>",
             "<style>body{font:15px system-ui;margin:24px auto;max-width:900px;padding:0 16px}"
             "td{padding:6px 10px;border-bottom:1px solid #ddd;vertical-align:top}"
             "audio{width:260px}.t{color:#666;font-size:13px}</style>",
             "<h1>Per-game EQ match</h1>",
             "<p class='t'>Each row: the clip as shipped, then matched to the average of all three games. "
             "Both play at the same loudness. Space bar plays the focused player.</p>"]  # fmt: skip
    for game in games:
        parts.append(f"<h2>{html.escape(game)}</h2><table>")
        for r in rows:
            if r["game"] != game:
                continue
            parts.append(f"<tr><td><b>{html.escape(r['id'])}</b><div class='t'>{html.escape(r['text'])}</div></td>"
                         f"<td>Original<br><audio controls preload='none' src='{r['orig']}'></audio></td>"
                         f"<td>EQ-matched<br><audio controls preload='none' src='{r['eq']}'></audio></td></tr>")  # fmt: skip
        parts.append("</table>")
    (out / "index.html").write_text("\n".join(parts))


def main():
    import soundfile as sf

    import confounds as cf
    from vi_job import load_clips

    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="corpus/tali")
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    harvest = fp.current_out(a.corpus, "harvest")
    os.environ["VI_INPUT_HARVEST"] = str(harvest)
    for stage in ("prep", "clean"):
        try:
            os.environ[f"VI_INPUT_{stage.upper()}"] = str(fp.current_out(a.corpus, stage))
        except FileNotFoundError:
            pass
    clips = load_clips()
    meta = cf.Meta(clips)
    rng = np.random.default_rng(cf.SEED)
    comparable, rule = meta.comparable(rng)
    print(f"fitting on {len(comparable)} comparable clips ({rule})", file=sys.stderr)

    def raw(c):
        return harvest / c.line["audio"]["path"]  # as shipped

    spectra = {}
    for n, i in enumerate(comparable):
        c = clips[i]
        x, sr = sf.read(raw(c), always_2d=False)
        x = x.mean(axis=1) if x.ndim > 1 else x
        s = fp.clip_spectrum(np.asarray(x, dtype=np.float64), sr)
        if s is not None:
            spectra.setdefault(c.game, []).append(s)
        if (n + 1) % 250 == 0:
            print(f"{n + 1}/{len(comparable)}", file=sys.stderr, flush=True)
    curves = game_curves(spectra)
    target, corr = corrections(curves)

    out = Path(a.out) if a.out else Path(a.corpus) / "analysis" / "eq_match"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    g = fp.grid()
    thirds = {}
    for game, d in corr.items():
        thirds[game] = {str(t): round(float(np.interp(t, g, d)), 1) for t in fp.THIRDS if F_LO <= t <= F_HI}
    report = {"fitted_on": {gm: len(v) for gm, v in spectra.items()}, "rule": rule,
              "range_hz": [F_LO, F_HI], "max_db": MAX_DB,
              "correction_db_at_third_octaves": thirds}  # fmt: skip
    (out / "eq_match.json").write_text(json.dumps(report | {"grid_hz": g.tolist(),
        "target_db": np.nan_to_num(target).tolist(), "correction_db": {k: v.tolist() for k, v in corr.items()}}))  # fmt: skip
    print(json.dumps(report, indent=1))

    # Clips to hear: spread over each game's comparable clips (same rule, so
    # what you hear is what the fit saw), plus barks for the short end.
    rows = []
    for game in sorted(corr):
        pool = [clips[i] for i in comparable if clips[i].game == game]
        pick = [pool[j] for j in np.linspace(0, len(pool) - 1, min(a.per_game, len(pool))).astype(int)]
        for c in pick:
            x, sr = sf.read(raw(c), always_2d=False)
            x = np.asarray(x.mean(axis=1) if x.ndim > 1 else x, dtype=np.float64)
            y = same_loudness(x, apply(x, fir(corr[game], sr)))
            name = c.id.replace("/", "_")
            sf.write(out / "audio" / f"{name}.orig.wav", same_loudness(x, x), sr)
            sf.write(out / "audio" / f"{name}.eq.wav", y, sr)
            rows.append({"id": c.id, "game": game, "text": c.text,
                         "orig": f"audio/{name}.orig.wav", "eq": f"audio/{name}.eq.wav"})  # fmt: skip
    page(rows, sorted(corr), out)
    print(f"open {out / 'index.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
