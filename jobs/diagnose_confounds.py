"""Diagnose why confounds stopped (read-only; writes nothing to the corpus).

Two questions about the stopped run, answered on the report split:

  1. Is the below-chance game probe an artifact? Coral is fitted on rows
     the probe later tests on, and equalises the game means. With folds
     grouped by conversation, a held-out conversation drags its game's mean
     toward itself, so the remaining clips of that game sit on the far side
     and the probe guesses against them. Checked on pure noise (no game
     signal at all) and on the real candidates with the whole transform
     refitted inside each fold, on training rows only.
  2. How much of the A/V/D drop is the confound itself? The A/V/D scores come
     from a model that also hears each game's mix and the clip's length. We
     measure how much of A/V/D the nuisances alone predict, then redo the
     guard against A/V/D with the nuisances taken out (out of fold).

Usage (from the repo root, at nice 19):
  python3 jobs/diagnose_confounds.py corpus/tali [candidate ...]
Prints JSON. Changes no code, thresholds or stage outputs.
"""

import copy
import json
import os
import sys
from pathlib import Path

import confounds as cf
from vi_job import load_clips, read_jsonl

DEFAULT = ["wavlm_L17", "wavlm_L14", "emotion2vec_plus_large"]


def current_out(corpus, stage):
    d = Path(corpus) / "stages" / stage
    return d / (d / "current").read_text().strip() / "out"


def setup(corpus):
    """The same rows, splits and targets confounds.main uses."""
    import numpy as np

    for stage in ("harvest", "prep", "clean", "embed"):
        os.environ[f"VI_INPUT_{stage.upper()}"] = str(current_out(corpus, stage))
    embed = current_out(corpus, "embed")
    clips = load_clips()
    by_id = {c.id: c for c in clips}
    report = json.loads((embed / "embed.json").read_text())
    names = sorted(k for k in report["embeddings"] if k != "minilm")
    _, ids = cf.load_matrix(embed, names[0])
    ids = [i for i in ids if i in by_id]
    meta = cf.Meta([by_id[i] for i in ids])
    comparable, _ = meta.comparable(np.random.default_rng(cf.SEED))
    selection = meta.split(0.3)
    unique = set(np.flatnonzero(meta.unique))
    usable = np.array([i in unique and not meta.excluded[i] for i in range(len(ids))])
    comp = np.zeros(len(ids), dtype=bool)
    comp[comparable] = True
    comp &= usable
    avd = {r["id"]: [r["avd"][k] for k in ("arousal", "dominance", "valence")]
           for r in read_jsonl(embed / "scores.jsonl") if r.get("avd")}  # fmt: skip
    return {"embed": embed, "ids": ids, "meta": meta, "comparable": comparable, "avd": avd,
            "rep_rows": np.flatnonzero(usable & ~selection), "rep_comp": np.flatnonzero(comp & ~selection)}  # fmt: skip


def probes(s):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    return {"knn": KNeighborsClassifier(n_neighbors=int(s.get("knn_k", 15))),
            "svm": SVC(kernel="rbf", gamma="scale", C=1.0)}  # fmt: skip


def noise_check(s, games=3, per_game=360, conv=5, dims=96, seed=0):
    """Check 1a: pure noise through PCA and Coral fitted on every row."""
    import numpy as np
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    n = games * per_game
    x = rng.standard_normal((n, 256))
    game = np.repeat([f"g{i}" for i in range(games)], per_game)
    group = np.array([f"{g}:{i // conv}" for i, g in enumerate(game)])
    rows = np.arange(n)
    raw = PCA(n_components=dims, random_state=cf.SEED).fit_transform(x)
    coloured = cf.Coral().fit(raw, game, rows).apply(raw, game)
    out = {"chance": round(1 / games, 4)}
    for label, v in (("raw", raw), ("coral_on_all_rows", coloured)):
        out[label] = {}
        for name, model in probes(s).items():
            pred, done = cf.oof_classify(v, game, group, model, int(s.get("folds", 5)))
            out[label][name] = round(cf.balanced_accuracy(game[done], pred[done]), 4)
    return out


def inductive_game_probe(x, d, s, dims, with_duration, permutations=20):
    """Check 1b: the whole transform refitted per fold on training rows only,
    so held-out conversations never touch PCA, Coral or LEACE. Folds are
    grouped by conversation and stratified by game: each conversation is one
    game, so plain grouped folds can leave a fold heavy in one game and push
    the probe below chance on their own. The null refits with game shuffled
    across conversations, through exactly the same procedure."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta, rows = d["meta"], d["rep_comp"]
    y, g = meta.game[rows], meta.group[rows]
    split = StratifiedGroupKFold(n_splits=int(s.get("folds", 5)), shuffle=True, random_state=cf.SEED)
    prepared = []
    for train, test in split.split(rows, y, g):
        held = np.isin(meta.group, g[test])
        keep = ~held
        m = copy.copy(meta)
        m.excluded = meta.excluded | held  # out of LEACE's fit
        prepared.append((train, test, m, d["comparable"][keep[d["comparable"]]]))

    def score(game):
        yr = game[rows]
        pred = {name: np.empty(len(rows), dtype=yr.dtype) for name in probes(s)}
        for train, test, m, fit_comp in prepared:
            m = copy.copy(m)
            m.game = game
            cx, _, _, _ = cf.clean(x, m, fit_comp, dims, with_duration)
            for name, model in probes(s).items():
                p = make_pipeline(StandardScaler(), clone(model)).fit(cx[rows[train]], yr[train])
                pred[name][test] = p.predict(cx[rows[test]])
        return {name: cf.balanced_accuracy(yr, p) for name, p in pred.items()}

    observed = score(meta.game)
    game_of = {c: meta.game[meta.group == c][0] for c in sorted(set(meta.group))}
    convs = sorted(game_of)
    rng = np.random.default_rng(cf.SEED)
    null = {name: [] for name in probes(s)}
    for _ in range(permutations):
        shuffled = dict(zip(convs, rng.permutation([game_of[c] for c in convs])))
        for name, v in score(np.array([shuffled[c] for c in meta.group])).items():
            null[name].append(v)
    return {name: {"balanced_accuracy": round(observed[name], 4),
                   "null_median": round(float(np.median(null[name])), 4) if null[name] else None,
                   "null_p95": round(float(np.quantile(null[name], 0.95)), 4) if null[name] else None,
                   "permutation_p": round((1 + sum(v >= observed[name] for v in null[name])) / (1 + len(null[name])), 4)}
            for name in observed}  # fmt: skip


def nuisance_matrix(meta, rows):
    import numpy as np

    games = sorted(set(meta.game))
    cols = [(meta.game[rows] == gm).astype(float) for gm in games[1:]]
    for c in (meta.log_dur, meta.log_dur**2, meta.log_syll, meta.floor_level):
        v = c[rows].astype(float)
        cols.append(np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0.0))
    return np.column_stack(cols)


def oof_residual(z, y, groups, k):
    """y minus its out-of-fold ridge prediction from z."""
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    resid = np.full_like(y, np.nan, dtype=float)
    for train, test in cf.folds(groups, k):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(z[train], y[train])
        resid[test] = y[test] - m.predict(z[test])
    return resid


def avd_check(raw, cx, d, s):
    """Check 2: the guard against A/V/D with the nuisances taken out."""
    import numpy as np

    meta, k = d["meta"], int(s.get("folds", 5))
    rows = d["rep_rows"][[meta.ids[i] in d["avd"] for i in d["rep_rows"]]]
    y = np.array([d["avd"][meta.ids[i]] for i in rows])
    g = meta.group[rows]
    z = nuisance_matrix(meta, rows)
    resid = oof_residual(z, y, g, k)
    ok = np.isfinite(resid).all(axis=1)
    rows, y, g, z, resid = rows[ok], y[ok], g[ok], z[ok], resid[ok]
    before, after = cf.ridge_r2(raw[rows], y, g, k), cf.ridge_r2(cx[rows], y, g, k)
    rb, ra = cf.ridge_r2(raw[rows], resid, g, k), cf.ridge_r2(cx[rows], resid, g, k)
    return {"clips": len(rows),
            "avd_from_nuisance_alone_r2": cf.ridge_r2(z, y, g, k),
            "avd_r2": {"before": before, "after": after, "drop": round(before - after, 4)},
            "avd_minus_nuisance_r2": {"before": rb, "after": ra, "drop": round(rb - ra, 4)},
            "limit": float(s.get("avd_max_drop", 0.10))}  # fmt: skip


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus/tali"
    names = sys.argv[2:] or DEFAULT
    run = json.loads((current_out(corpus, "confounds") / "confounds.json").read_text())
    s = run.get("settings", {})
    dims = int(s.get("pca_dims", 96))
    d = setup(corpus)
    out = {"noise_check": noise_check(s), "candidates": {}}
    for name in names:
        x, xids = cf.load_matrix(d["embed"], name)
        pos = {cid: i for i, cid in enumerate(xids)}
        x = x[[pos[i] for i in d["ids"]]]
        with_duration = bool(run.get("candidates", {}).get(name, {}).get("transform", {}).get("duration_erased", True))
        cx, raw, _, _ = cf.clean(x, d["meta"], d["comparable"], dims, with_duration)
        game = cf.game_probe(cx, d["meta"], d["rep_comp"], s)
        r = {"duration_erased": with_duration,
             "game_probe_as_run": {m: game[m]["balanced_accuracy"] for m in ("knn", "svm") if game.get(m)},
             "game_probe_refit_per_fold": inductive_game_probe(x, d, s, dims, with_duration),
             "avd": avd_check(raw, cx, d, s)}  # fmt: skip
        out["candidates"][name] = r
        print(f"{name}: done", file=sys.stderr, flush=True)
    out["chance"] = round(1 / len(set(d["meta"].game[d["rep_comp"]])), 4)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
