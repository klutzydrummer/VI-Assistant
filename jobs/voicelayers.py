"""Home check before voicearms (research/steering-qwen3-tts.md, section 4): read voicesteer's output
and say, per talker layer, how the mood vectors separate, and per mood, whether test D's probe
heard her real mood clip in generated speech (swap vs neutral). CPU, numpy only, seconds.

    python3 jobs/voicelayers.py <voicesteer output folder> [--npz PATH]   # writes layers.md there

--npz also bundles the vectors into one file for voicearms (its stage sends
corpus/tali/analysis/after_labels/voice_vectors.npz).

Per layer: the vectors' mean norm, the mean cosine between different moods (lower = they separate
more), and the share of each vector that's common to all moods (the "away from neutral" part).
The published Qwen3-TTS-1.7B recipe found moods separate best at layers 21-25 (cosine 0.24);
voicesteer steered 9-17.
"""

import json
import sys
from pathlib import Path

import numpy as np


def layer_table(vectors):
    """{mood: (layers, hidden)} -> one row per layer."""
    moods = sorted(vectors)
    stack = np.stack([vectors[m] for m in moods])  # moods x layers x hidden
    rows = []
    for i in range(stack.shape[1]):
        v = stack[:, i]
        norms = np.linalg.norm(v, axis=1)
        unit = v / np.maximum(norms[:, None], 1e-12)
        cos = unit @ unit.T
        off = cos[~np.eye(len(moods), dtype=bool)]
        shared = unit.mean(0)
        common = float(np.mean(unit @ shared / max(np.linalg.norm(shared), 1e-12))) if len(moods) > 1 else 1.0
        rows.append({"layer": i, "norm": round(float(norms.mean()), 4),
                     "cross_mood_cosine": round(float(off.mean()), 4) if len(off) else None,
                     "common_share": round(common, 4)})
    return rows


def swap_table(report):
    out = {}
    for m, r in sorted(report.get("moods", {}).items()):
        n, s = r.get("neutral_probe"), r.get("swap", {}).get("probe")
        out[m] = {"neutral": n, "swap": s, "swap_lift": None if n is None or s is None else round(s - n, 4),
                  "verdict": r.get("verdict"), "strength": r.get("strength")}
    return out


def render(rows, swaps, n_layers):
    lines = ["# Voice vectors by layer, and whether the probe hears swap", "",
             f"{n_layers} talker layers. Lower cross-mood cosine = moods more distinct there.", "",
             "| layer | depth | mean norm | cross-mood cosine | common share |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['layer']} | {r['layer'] / n_layers:.2f} | {r['norm']} | {r['cross_mood_cosine']} | {r['common_share']} |")
    best = sorted((r for r in rows if r["cross_mood_cosine"] is not None), key=lambda r: r["cross_mood_cosine"])[:5]
    lines += ["", "Most distinct layers: " + ", ".join(f"{r['layer']} ({r['cross_mood_cosine']})" for r in best), ""]
    if swaps:
        lines += ["Swap (her top mood clip as the reference) against neutral, on test D's probe. If swap doesn't lift",
                  "the probe, the probe can't hear that mood in generated speech and its automatic verdicts don't count.", "",
                  "| mood | neutral | swap | swap lift | voicesteer verdict | kept strength |", "|---|---|---|---|---|---|"]
        for m, s in swaps.items():
            lines.append(f"| {m} | {s['neutral']} | {s['swap']} | {s['swap_lift']} | {s['verdict']} | {s['strength']} |")
    return "\n".join(lines) + "\n"


def main(folder, npz=None):
    folder = Path(folder)
    vectors = {p.stem: np.load(p) for p in sorted((folder / "vectors").glob("*.npy"))}
    if not vectors:
        sys.exit(f"no vectors/*.npy in {folder}")
    shapes = {v.shape for v in vectors.values()}
    if len(shapes) != 1:
        sys.exit(f"vectors have different shapes: {shapes}")
    rows = layer_table(vectors)
    report = folder / "report.json"
    swaps = swap_table(json.loads(report.read_text())) if report.exists() else {}
    (folder / "layers.md").write_text(render(rows, swaps, next(iter(shapes))[0]))
    (folder / "layers.json").write_text(json.dumps({"layers": rows, "swap": swaps}, indent=1) + "\n")
    print(f"wrote {folder / 'layers.md'}")
    if npz:
        np.savez(npz, **{m: v.astype(np.float32) for m, v in vectors.items()})
        print(f"wrote {npz} ({len(vectors)} moods)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[1] == "--npz":
        main(args[0], args[2])
    elif len(args) == 1:
        main(args[0])
    else:
        sys.exit(__doc__)
