"""eq_check stage: does the per-game EQ match help the analysis models? At
home on the CPU, before the whole corpus is re-embedded with it.

characters/tali/eq_check.toml preps a subset of clips with the EQ match on
and embeds only them, with emotion2vec+ large and the audeering A/V/D model.
This compares those with the same clips' vectors and scores from the main
corpus's last embed, which heard them without the EQ match. Same clips, same
words, same levelling: the EQ is the only difference.

  game probe   how well a linear classifier tells the game from each
               embedding both sides have (emotion2vec+, and the WavLM layers
               when embed ran them; balanced accuracy, folds grouped by
               conversation, several shuffles). The suit's EQ differs per game,
               so if the match helps, this drops toward chance.
  kept         how much each clip's readings survive the EQ: per-clip cosine of
               the vectors and the correlation of each A/V/D dimension. If the
               match helps, these stay high: it changes colour, not delivery.
  game gaps    each A/V/D dimension's and each emotion2vec+ class's per-game
               means, and their spread across games, before and after.

Writes report.json and report.md. Nothing is a gate: Brandon reads it and
decides whether the whole corpus is re-embedded.

Usage (from the pipeline): python3 ../../jobs/eq_check.py MAIN_CORPUS
Needs numpy and scikit-learn.
"""

import json
import sys
from pathlib import Path

from vi_job import env_path, load_clips, log, out_dir, read_jsonl, settings

VECTOR = "emotion2vec_plus_large"
SHUFFLES = 5


def current_out(corpus, stage):
    d = Path(corpus) / "stages" / stage
    return d / (d / "current").read_text().strip() / "out"


def read_embed(folder, names=None):
    """({name: {id: vector}} for each embedding in `names` (default: all
    but the text one), {id: scores record}) from one embed output."""
    import numpy as np

    names = names or sorted(f.stem for f in (folder / "embeddings").glob("*.npy") if f.stem != "minilm")
    vectors = {}
    for name in names:
        m = np.load(folder / "embeddings" / f"{name}.npy").astype(np.float64)
        ids = (folder / "embeddings" / f"{name}.ids").read_text().split()
        vectors[name] = dict(zip(ids, m, strict=True))
    scores = {r["id"]: r for r in read_jsonl(folder / "scores.jsonl")}
    return vectors, scores


def game_probe(x, game, group, seeds=SHUFFLES):
    """Balanced accuracy of telling the game from x, conversation-grouped
    5-fold CV, over `seeds` shuffles: (mean, sd)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = []
    for seed in range(seeds):
        pred = np.empty_like(game)
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, game, group):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            pred[te] = clf.fit(x[tr], game[tr]).predict(x[te])
        scores.append(balanced_accuracy_score(game, pred))
    return float(np.mean(scores)), float(np.std(scores))


def by_game(values, game):
    """Per-game means of a 1-d array and their spread (max - min)."""
    import numpy as np

    means = {g: round(float(np.mean(values[game == g])), 4) for g in sorted(set(game))}
    return {"means": means, "spread": round(max(means.values()) - min(means.values()), 4)}


def compare(ids, game, group, before, after, scores_before, scores_after):
    """The report, from paired vectors ({name: rows in `ids` order}, or one
    array for VECTOR) and scores. `game_probe` and `kept` are VECTOR's;
    `embeddings` has the same for every name."""
    import numpy as np

    if not isinstance(before, dict):
        before, after = {VECTOR: before}, {VECTOR: after}
    out = {"clips": len(ids), "per_game": {g: int((game == g).sum()) for g in sorted(set(game))},
           "chance": round(1 / len(set(game)), 4), "embeddings": {}}  # fmt: skip
    for name in sorted(before):
        x, y = before[name], after[name]
        b, sb = game_probe(x, game, group)
        a, sa = game_probe(y, game, group)
        cos = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
        out["embeddings"][name] = {
            "game_probe": {"before": round(b, 4), "before_sd": round(sb, 4), "after": round(a, 4), "after_sd": round(sa, 4)},
            "kept": {"vector_cosine_median": round(float(np.median(cos)), 4),
                     "vector_cosine_p05": round(float(np.quantile(cos, 0.05)), 4)},
        }  # fmt: skip
    first = out["embeddings"].get(VECTOR) or next(iter(out["embeddings"].values()))
    out["game_probe"], out["kept"] = first["game_probe"], first["kept"]
    for key in ("avd", "emotion2vec_plus_large_classes"):
        names = sorted(set.intersection(*[set(s.get(key, {})) for s in scores_before + scores_after]))
        if not names:
            continue
        section = {}
        for n in names:
            x = np.array([s[key][n] for s in scores_before])
            y = np.array([s[key][n] for s in scores_after])
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            section[n] = {"before": by_game(x, game), "after": by_game(y, game), "correlation": round(r, 4)}
        out[key] = section
    return out


def markdown(r):
    p = r["game_probe"]
    lines = [
        "# Does the per-game EQ match help?",
        "",
        f"{r['clips']} clips ({', '.join(f'{g} {n}' for g, n in r['per_game'].items())}), each embedded "
        "without the EQ match (main corpus) and with it (this check).",
        "",
        "## Telling the game from emotion2vec+ (lower is better)",
        "",
        f"- Without the EQ match: {p['before']:.3f} (sd {p['before_sd']:.3f})",
        f"- With it: {p['after']:.3f} (sd {p['after_sd']:.3f})",
        f"- Chance: {r['chance']:.3f}",
        "",
        "## What each clip kept (higher is better)",
        "",
        f"- emotion2vec+ vector cosine, median {r['kept']['vector_cosine_median']:.3f}, "
        f"5th percentile {r['kept']['vector_cosine_p05']:.3f}",
    ]
    if len(r.get("embeddings", {})) > 1:
        lines += ["", "## Every embedding: game probe before → after (lower is better), median cosine", "",
                  "| | without | with | cosine |", "|---|---|---|---|"]  # fmt: skip
        for n, e in r["embeddings"].items():
            g = e["game_probe"]
            lines.append(f"| {n} | {g['before']:.3f} | {g['after']:.3f} | {e['kept']['vector_cosine_median']:.3f} |")
    for key, title in (("avd", "A/V/D"), ("emotion2vec_plus_large_classes", "emotion2vec+ classes")):
        if key not in r:
            continue
        lines += ["", f"## {title}: per-game means, before and after", "",
                  "| | correlation | spread before | spread after | " + " | ".join(r["per_game"]) + " |",
                  "|---|---|---|---|" + "---|" * len(r["per_game"])]  # fmt: skip
        for n, v in r[key].items():
            cells = " | ".join(f"{v['before']['means'][g]:.3f} → {v['after']['means'][g]:.3f}" for g in r["per_game"])
            lines.append(f"| {n} | {v['correlation']:.3f} | {v['before']['spread']:.3f} | "
                         f"{v['after']['spread']:.3f} | {cells} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main():
    import numpy as np

    import confounds as cf

    if len(sys.argv) != 2:
        sys.exit("usage: eq_check.py MAIN_CORPUS")
    main_embed = current_out(sys.argv[1], "embed")
    out = out_dir()
    clips = [c for c in load_clips() if c.audio16k is not None]
    after, scores_after = read_embed(env_path("VI_INPUT_EMBED"))
    before, scores_before = read_embed(main_embed, sorted(after))
    clips = [c for c in clips if c.id in scores_before and all(c.id in v for v in [*after.values(), *before.values()])]
    if len(clips) < 30:
        sys.exit(f"eq_check: only {len(clips)} clips embedded both ways")
    meta = cf.Meta(clips)
    ids = [c.id for c in clips]
    r = compare(ids, meta.game, meta.group, {n: np.stack([v[i] for i in ids]) for n, v in before.items()},
                {n: np.stack([v[i] for i in ids]) for n, v in after.items()},
                [scores_before[i] for i in ids], [scores_after[i] for i in ids])  # fmt: skip
    r["main_embed"] = str(main_embed)
    r["settings"] = {k: v for k, v in settings().items() if k != "backend"}
    (out / "report.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(markdown(r))
    for n, e in r["embeddings"].items():
        log(f"eq_check: {n} game probe {e['game_probe']['before']:.3f} -> {e['game_probe']['after']:.3f}")
    log(f"eq_check: chance {r['chance']:.3f}; read {out / 'report.md'}")


if __name__ == "__main__":
    main()
