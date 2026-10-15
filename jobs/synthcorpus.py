"""A synthetic corpus for the home-stage tests (confounds, cluster, classes):
harvest, prep, clean and embed outputs in the real formats, with known
classes, a per-game recording chain, a text-length direction and a noise
floor baked into the embeddings. No audio: these stages don't read any.

    write(folder, n_conversations=60, eq=0.3) -> {"classes": {id: class}, ...}
"""

import json
from pathlib import Path

GAMES = ["le1", "le2", "le3"]
WORDS = ["keelah", "shepard", "the", "drive", "core", "is", "humming", "nicely", "today", "flotilla"]


def write(folder, n_conversations=60, per_conversation=8, dim=32, classes=4, seed=0, eq=0.3):
    import numpy as np

    rng = np.random.default_rng(seed)
    folder = Path(folder)
    centres = rng.standard_normal((classes, dim)) * 3.0
    length_dir = rng.standard_normal(dim)
    floor_dir = rng.standard_normal(dim)
    # Each game's recording chain: an offset and a per-dimension gain, like
    # an EQ and a mix level.
    chains = {g: (rng.standard_normal(dim) * 2.0, np.diag(np.exp(eq * rng.standard_normal(dim)))) for g in GAMES}
    lines, prep, clips, rows, truth = [], [], [], [], {}
    for conv in range(n_conversations):
        game = GAMES[conv % 3]
        offset, mix = chains[game]
        name = f"{game}_conv{conv:03d}"
        for k in range(per_conversation):
            cls = int(rng.integers(classes))
            n_words = int(rng.integers(2, 14))
            text = " ".join(rng.choice(WORDS, n_words))
            floor = float(-75 + 10 * rng.standard_normal() + {"le1": 8, "le2": 0, "le3": -6}[game])
            base = centres[cls] + 0.8 * rng.standard_normal(dim)
            duration = 0.3 * n_words * float(np.exp(0.2 * rng.standard_normal()))
            takes = ["m", "f"] if k % 4 == 0 else ["f"]
            for take in takes:
                cid = f"{game}.s{conv * 100 + k}_{take}"
                v = base + 0.2 * rng.standard_normal(dim)
                v = v + 0.6 * np.log1p(n_words) * length_dir + 0.05 * (floor + 75) * floor_dir
                v = (v @ mix.T) + offset
                truth[cid] = cls
                tags = ["dialogue_choices" if conv % 2 else "exchange", "to_shepard", ["normandy", "combat"][cls % 2]]
                lines.append({"id": cid, "speaker": "tali", "text": text, "flags": [],
                              "audio": {"path": f"audio/{cid}.wav", "sha256": f"{cid}-sha", "duration_s": duration},
                              "source": {"game": game, "conversation": name, "line_kind": tags[0],
                                         "listener": "shepard"},
                              "scene_tags": tags, "stage_direction": None})  # fmt: skip
                prep.append({"id": cid, "duration_s": duration, "flags": [], "gain_db": 0.0,
                             "level": {"method": "bs1770", "db": -25.0 + rng.standard_normal(), "peak_dbfs": -6.0},
                             "noise_floor": {"level_db": floor, "seconds": 0.2,
                                             "spectrum": list(floor - 10 + rng.standard_normal(4))},
                             "audio16k": {"path": f"audio16k/{cid}.flac", "sample_rate": 16000}})  # fmt: skip
                clips.append({"id": cid, "keep": True, "audio": {"path": f"audio/{cid}.flac", "duration_s": duration},
                              "flags": []})  # fmt: skip
                rows.append((cid, v, cls))
    # One singing line, excluded from fitting but kept.
    lines[3]["flags"] = ["singing"]

    harvest = folder / "harvest"
    harvest.mkdir(parents=True, exist_ok=True)
    _jsonl(harvest / "lines.jsonl", lines)
    (folder / "prep").mkdir(exist_ok=True)
    _jsonl(folder / "prep" / "prep.jsonl", prep)
    (folder / "clean").mkdir(exist_ok=True)
    _jsonl(folder / "clean" / "clips.jsonl", clips)
    emb = folder / "embed" / "embeddings"
    emb.mkdir(parents=True, exist_ok=True)
    ids = [r[0] for r in rows]
    good = np.stack([r[1] for r in rows])
    # A second candidate that only hears length and game: it must lose.
    durations = np.log([p["duration_s"] for p in prep])
    lengthy = np.column_stack([durations, durations**2] + [rng.standard_normal(len(ids)) for _ in range(dim - 2)])
    lengthy += np.stack([chains[i.split(".")[0]][0] for i in ids])
    for name, m in {"good": good, "lengthy": lengthy}.items():
        np.save(emb / f"{name}.npy", m.astype(np.float16))
        (emb / f"{name}.ids").write_text("".join(f"{i}\n" for i in ids))
    # Crops of the first 120 clips: shorter crops are noisier copies.
    crops = folder / "embed" / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    crop_ids, crop_rows = [], []
    for k in range(120):
        for n, noise in [(0.5, 6.0), (1, 2.0), (1.5, 0.8), (2, 0.4)]:
            crop_ids.append(f"{ids[k]}@{n:g}")
            crop_rows.append(good[k] + noise * rng.standard_normal(dim))
    for name, m in {"good": np.stack(crop_rows), "lengthy": np.stack(crop_rows)}.items():
        np.save(crops / f"{name}.npy", m.astype(np.float16))
        (crops / f"{name}.ids").write_text("".join(f"{i}\n" for i in crop_ids))
    avd = {cid: [0.3 + 0.1 * cls, 0.5, 0.4 + 0.05 * cls] for cid, _, cls in rows}
    eiv_names = ["Anger", "Amusement", "Sadness", "Teasing"]
    _jsonl(folder / "embed" / "scores.jsonl", [
        {"id": cid, "avd": dict(zip(["arousal", "dominance", "valence"], avd[cid])),
         "eiv": {n: round(float((cls == j) + 0.1 * rng.standard_normal()), 4) for j, n in enumerate(eiv_names)},
         "eiv_quality": {"score_overall_quality": 2.5}}
        for cid, _, cls in rows])  # fmt: skip
    _jsonl(folder / "embed" / "prosody.jsonl", [{"id": cid, "f0_median_hz": 200.0 + 10 * cls, "words_per_s": 3.0}
                                                for cid, _, cls in rows])  # fmt: skip
    (folder / "embed" / "embed.json").write_text(json.dumps({"embeddings": {"good": {}, "lengthy": {}}}))
    return {"classes": truth}


def _jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
