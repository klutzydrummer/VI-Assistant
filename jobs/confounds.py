"""confounds stage: take the recording out of each candidate embedding, check
that it's gone, and pick the candidate to cluster. Home, CPU (numpy,
scikit-learn); no GPU.

Plan: docs/design/corpus-pipeline.md, step 4 (followed exactly; thresholds
are settings fixed before results). Format: docs/design/corpus-format.md,
confounds. For each candidate embedding from embed:

  1. Measure first: probes for every nuisance and situation factor, folds
     grouped by conversation (the "before" table).
  2. PCA on the pooled comparable subset, `pca_dims` dimensions.
  3. Re-colour each game to the pooled covariance (CORAL with Ledoit-Wolf
     shrinkage, fitted on the comparable subset, applied to every clip).
     This also equalises the game means.
  4. LEACE for text length (log syllables) and the noise-floor level,
     fitted on every eligible clip but barks.
  5. Acceptance on the selection split (30% of conversations): game gone to
     k-NN and RBF-SVM probes, text length and duration gone to a linear
     probe, the A/V/D guard holds. Then paired-take agreement.
  6. The best passing candidate (paired-take agreement above chance, then
     A/V/D) is reported on the other 70%, with conversation-level bootstrap
     bounds and permutations of game across conversations.

Nuisance factors are erased; situation factors (scene and structural tags,
bark, stage direction, level, text content) are only measured. Clips flagged
singing, subtitle_mismatch or far_voice never fit anything but are cleaned and
kept. Byte-identical paired takes count once.

If a text-length or duration effect is left, log duration is added to the
eraser and re-checked. If no candidate passes, nothing is chosen: the stage
writes the tables and `status: "stopped"` for Brandon, and cluster refuses
to run. A bound is never loosened to pass.

Outputs: confounds.json, cleaned.npy + cleaned.ids (the chosen candidate,
float32), cleaned_crops.npy + .ids (embed's crops through the same
transform), checks.json.

Settings: pca_dims, selection_share, game_margin, max_length_r2,
avd_max_drop, knn_k, folds, permutations, bootstrap, candidates, limit.
"""

import hashlib
import json
import math
import re

import checks
from vi_job import env_path, load_clips, log, out_dir, read_jsonl, settings

EXCLUDE = {"singing", "subtitle_mismatch", "far_voice"}
COMPARABLE_KINDS = {"exchange", "dialogue_choices"}
SEED = 7


# --- data ---------------------------------------------------------------------


def load_matrix(folder, name, sub="embeddings"):
    import numpy as np

    m = np.load(folder / sub / f"{name}.npy").astype(np.float64)
    ids = (folder / sub / f"{name}.ids").read_text().split()
    return m, ids


def syllables(text):
    """Syllables in the subtitle, by vowel groups per word (at least one per
    word). Stage notes like {Sighs} don't count."""
    text = re.sub(r"[{\[(][^}\])]*[}\])]", " ", text or "").lower()
    n = 0
    for word in re.findall(r"[a-z']+", text):
        groups = re.findall(r"[aeiouy]+", word)
        n += max(1, len(groups) - (word.endswith("e") and len(groups) > 1 and not word.endswith("le")))
    return n


def take_base(cid):
    """The line a take records: le2.s348112_f and _m are both le2.s348112."""
    return re.sub(r"_[mf]$", "", cid)


class Meta:
    """Everything the probes and the eraser need, per clip, in one order."""

    def __init__(self, clips):
        import numpy as np

        self.ids = [c.id for c in clips]
        self.index = {cid: i for i, cid in enumerate(self.ids)}
        lines = [c.line for c in clips]
        src = [ln.get("source", {}) for ln in lines]
        self.game = np.array([c.game for c in clips])
        self.group = np.array([s.get("conversation") or f"{c.game}:{take_base(c.id)}" for c, s in zip(clips, src)])
        self.bark = np.array(["bark" in c.flags for c in clips])
        self.excluded = np.array([bool(EXCLUDE & set(c.flags)) for c in clips])
        self.has_text = np.array([bool((c.text or "").strip()) and "no_subtitle" not in c.flags for c in clips])
        self.log_syll = np.array([math.log1p(syllables(c.text)) if t else np.nan
                                  for c, t in zip(clips, self.has_text)])  # fmt: skip
        self.log_dur = np.log(np.maximum([c.duration_s for c in clips], 0.05))
        floors = [(c.prep or {}).get("noise_floor") for c in clips]
        self.floor_level = np.array([f["level_db"] if f else np.nan for f in floors])
        bands = max((len(f["spectrum"]) for f in floors if f), default=0)
        self.floor_spectrum = np.array([f["spectrum"] if f else [np.nan] * bands for f in floors], dtype=float)
        self.level = np.array([((c.prep or {}).get("level") or {}).get("db", np.nan) for c in clips], dtype=float)
        self.tags = [set(c.scene_tags) for c in clips]
        self.stage_direction = np.array([bool(c.stage_direction) for c in clips])
        self.sha = [ln.get("audio", {}).get("sha256") or c.id for c, ln in zip(clips, lines)]
        self.unique = np.zeros(len(clips), dtype=bool)
        self.unique[self.dedupe()] = True
        kinds = [s.get("line_kind") for s in src]
        self.structural = any(k for k in kinds) or any(t & COMPARABLE_KINDS for t in self.tags)
        self.kind = [k or next(iter(t & (COMPARABLE_KINDS | {"one_liner"})), None) for k, t in zip(kinds, self.tags)]

    def comparable(self, rng):
        """The comparable-clip subset (hard data only): exchange or
        dialogue_choices lines spoken to Shepard, not barks, length-matched
        across games by duration quintile. Without the structural tags
        (an older harvest) it falls back to every non-bark line, and says so."""
        import numpy as np

        ok = ~self.bark & ~self.excluded
        if self.structural:
            ok &= np.array([k in COMPARABLE_KINDS and "to_shepard" in t for k, t in zip(self.kind, self.tags)])
            rule = "exchange or dialogue_choices, to_shepard, not a bark, length-matched"
        else:
            rule = "fallback: no structural tags in this harvest; every non-bark line, length-matched"
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return idx, rule
        edges = np.quantile(self.log_dur[idx], [0.2, 0.4, 0.6, 0.8])
        band = np.searchsorted(edges, self.log_dur[idx])
        games = sorted(set(self.game[idx]))
        keep = []
        for b in range(5):
            per = {g: idx[(band == b) & (self.game[idx] == g)] for g in games}
            n = min(len(v) for v in per.values())
            for v in per.values():
                keep.extend(rng.choice(v, n, replace=False) if n else [])
        return np.array(sorted(keep), dtype=int), rule

    def split(self, share):
        """Selection (True) or report (False) per clip, by conversation hash:
        the same conversations every run."""
        import numpy as np

        def pick(g):
            return int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) / 2**32 < share

        chosen = {g: pick(g) for g in set(self.group)}
        return np.array([chosen[g] for g in self.group])

    def dedupe(self):
        """Row indices with byte-identical takes counted once."""
        import numpy as np

        seen, keep = set(), []
        for i, h in enumerate(self.sha):
            if h not in seen:
                seen.add(h)
                keep.append(i)
        return np.array(keep, dtype=int)

    def pairs(self, rows):
        """Paired takes among `rows` whose audio differs: (i, j) row indices."""
        by = {}
        for i in rows:
            by.setdefault(take_base(self.ids[i]), []).append(i)
        return [(a, b) for v in by.values() if len(v) == 2 for a, b in [v] if self.sha[a] != self.sha[b]]


# --- transforms -----------------------------------------------------------------


def sqrtm(cov, inverse=False, tol=1e-10):
    import numpy as np

    w, v = np.linalg.eigh(cov)
    w = np.where(w > tol * max(w.max(), tol), w, 0.0)
    p = np.where(w > 0, w ** (-0.5 if inverse else 0.5), 0.0)
    return (v * p) @ v.T


def ledoit_wolf(x):
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(x).covariance_


class Coral:
    """Re-colour each game to the pooled within-game covariance:
    x -> S_p^1/2 S_g^-1/2 (x - mu_g) + mu_p, Ledoit-Wolf shrunk, fitted on
    `fit` rows and applied to all. Not whitening: the pooled shape stays."""

    def fit(self, x, game, fit):
        import numpy as np

        self.mu, self.a = {}, {}
        centred = []
        for g in sorted(set(game)):
            rows = fit[game[fit] == g]
            if len(rows) < 3:
                rows = np.flatnonzero(game == g)
            self.mu[g] = x[rows].mean(axis=0)
            centred.append(x[rows] - self.mu[g])
        pooled = np.vstack(centred)
        self.mu_p = x[fit].mean(axis=0)
        root = sqrtm(ledoit_wolf(pooled))
        for g, c in zip(sorted(self.mu), centred):
            self.a[g] = root @ sqrtm(ledoit_wolf(c), inverse=True)
        return self

    def apply(self, x, game):
        out = x.copy()
        for g, a in self.a.items():
            rows = game == g
            out[rows] = (x[rows] - self.mu[g]) @ a.T + self.mu_p
        return out


class Leace:
    """LEACE (Belrose et al. 2023): the least-squares edit that leaves no
    linear information about z in x. x' = x - W+ P W (x - mu), with W the
    whitening of x and P the projection onto W cov(x, z)."""

    def fit(self, x, z):
        import numpy as np

        self.mu = x.mean(axis=0)
        xc, zc = x - self.mu, z - z.mean(axis=0)
        cov = xc.T @ xc / len(x)
        w, w_inv = sqrtm(cov, inverse=True), sqrtm(cov)
        u, s, _ = np.linalg.svd(w @ (xc.T @ zc / len(x)), full_matrices=False)
        u = u[:, s > 1e-12 * max(s.max(initial=0), 1e-300)]
        self.edit = w_inv @ (u @ u.T) @ w
        return self

    def apply(self, x):
        return x - (x - self.mu) @ self.edit.T


def nuisance(meta, with_duration):
    """The erased factors as columns: log syllables and the noise-floor
    level; optionally log duration and its square. Missing values take the
    median (those rows don't fit). The floor's spectrum shape is not erased:
    LEACE takes out a whole direction for even a chance correlation, which
    costs real signal, and the spec's fallback for a game left audible is
    the EQ match."""
    import numpy as np

    cols = [meta.log_syll, meta.floor_level]
    if with_duration:
        cols += [meta.log_dur, meta.log_dur**2]
    cols = [c for c in cols if np.isfinite(c).sum() >= 3]  # a factor this corpus never measured
    z = np.column_stack(cols)
    known = np.isfinite(z).all(axis=1)
    med = np.nanmedian(z, axis=0)
    return np.where(np.isfinite(z), z, med), known


def clean(x, meta, comparable, dims, with_duration=False):
    """PCA -> CORAL -> LEACE. Returns (cleaned, pca-space raw, info, apply),
    where apply(x, game) runs the same fitted transform on other rows (the
    crops)."""
    import numpy as np
    from sklearn.decomposition import PCA

    dims = int(min(dims, len(comparable) - 1, x.shape[1]))
    comparable = comparable[meta.unique[comparable]]
    pca = PCA(n_components=dims, random_state=SEED).fit(x[comparable])
    raw = pca.transform(x)
    coral = Coral().fit(raw, meta.game, comparable)
    coloured = coral.apply(raw, meta.game)
    z, known = nuisance(meta, with_duration)
    fit = np.flatnonzero(known & ~meta.bark & ~meta.excluded & meta.unique)
    leace = Leace().fit(coloured[fit], z[fit])
    info = {"dims": dims, "explained": round(float(pca.explained_variance_ratio_.sum()), 4),
            "leace_fit_clips": int(len(fit)), "erased": int(z.shape[1]), "duration_erased": with_duration}  # fmt: skip
    def apply(other, game):
        return leace.apply(coral.apply(pca.transform(other), game))

    return leace.apply(coloured), raw, info, apply


# --- probes ---------------------------------------------------------------------


def folds(groups, k):
    from sklearn.model_selection import GroupKFold

    k = min(k, len(set(groups)))
    return list(GroupKFold(n_splits=k).split(groups, groups=groups)) if k >= 2 else []


def oof_classify(x, y, groups, model, k):
    """Out-of-fold predictions, folds grouped by conversation."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.empty(len(y), dtype=y.dtype)
    done = np.zeros(len(y), dtype=bool)
    for train, test in folds(groups, k):
        if len(set(y[train])) < 2:
            continue
        est = clone(model)
        if "n_neighbors" in est.get_params():  # tiny corpora (the smoke test)
            est.set_params(n_neighbors=min(est.n_neighbors, len(train)))
        m = make_pipeline(StandardScaler(), est).fit(x[train], y[train])
        pred[test] = m.predict(x[test])
        done[test] = True
    return pred, done


def balanced_accuracy(y, pred):
    import numpy as np

    classes = sorted(set(y))
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))


def wilson_upper(p, n, z=1.96):
    if n == 0:
        return 1.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre + half


def game_probe(x, meta, rows, s, permutations=0, bootstrap=0):
    """Can k-NN or an RBF-SVM still tell the games apart? Balanced accuracy
    out of fold, its analytic upper bound (rows as trials), and optionally a
    conversation-level bootstrap bound and a permutation p-value with game
    shuffled across conversations, not clips."""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    y, g = meta.game[rows], meta.group[rows]
    chance = 1 / max(1, len(set(y)))
    models = {"knn": KNeighborsClassifier(n_neighbors=int(s.get("knn_k", 15))),
              "svm": SVC(kernel="rbf", gamma="scale", C=1.0)}  # fmt: skip
    k = int(s.get("folds", 5))
    out = {"chance": round(chance, 4), "clips": int(len(rows)), "conversations": int(len(set(g)))}
    rng = np.random.default_rng(SEED)
    for name, model in models.items():
        pred, done = oof_classify(x[rows], y, g, model, k)
        if not done.any() or len(set(y)) < 2:
            out[name] = None
            continue
        ba = balanced_accuracy(y[done], pred[done])
        r = {"balanced_accuracy": round(ba, 4), "upper": round(wilson_upper(ba, int(done.sum())), 4), "bound": "binomial"}
        if bootstrap:
            convs = np.array(sorted(set(g)))
            stats = []
            for _ in range(bootstrap):
                pick = rng.choice(convs, len(convs), replace=True)
                idx = np.concatenate([np.flatnonzero((g == c) & done) for c in pick])
                if len(set(y[idx])) == len(set(y)):
                    stats.append(balanced_accuracy(y[idx], pred[idx]))
            r["upper"] = round(float(np.quantile(stats, 0.975)), 4)
            r["bound"] = f"conversation bootstrap, {len(stats)} draws"
        if permutations:
            game_of = {c: y[g == c][0] for c in set(g)}
            convs = sorted(game_of)
            null = []
            for _ in range(permutations):
                shuffled = dict(zip(convs, rng.permutation([game_of[c] for c in convs])))
                yp = np.array([shuffled[c] for c in g])
                p, d = oof_classify(x[rows], yp, g, model, k)
                null.append(balanced_accuracy(yp[d], p[d]))
            r["permutation_p"] = round((1 + sum(n >= ba for n in null)) / (1 + len(null)), 4)
            r["permutation_null_p95"] = round(float(np.quantile(null, 0.95)), 4)
        out[name] = r
    return out


def ridge_r2(x, y, groups, k):
    """Out-of-fold R^2 of a ridge probe (grouped folds); y may be 2-D."""
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(y) < 3:
        return None
    y = y.reshape(len(y), -1)
    pred = np.full_like(y, np.nan, dtype=float)
    for train, test in folds(groups, k):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(x[train], y[train])
        pred[test] = m.predict(x[test]).reshape(len(test), -1)
    ok = np.isfinite(pred).all(axis=1)
    if ok.sum() < 3:
        return None
    resid = ((y[ok] - pred[ok]) ** 2).sum(axis=0)
    total = ((y[ok] - y[ok].mean(axis=0)) ** 2).sum(axis=0)
    return round(float(np.mean(1 - resid / np.maximum(total, 1e-12))), 4)


def auc_probe(x, labels, groups, k):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if labels.sum() < 5 or (~labels).sum() < 5:
        return None
    score = np.full(len(labels), np.nan)
    for train, test in folds(groups, k):
        if len(set(labels[train])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(x[train], labels[train])
        score[test] = m.decision_function(x[test])
    ok = np.isfinite(score)
    if len(set(labels[ok])) < 2:
        return None
    return round(float(roc_auc_score(labels[ok], score[ok])), 4)


def factor_table(x, meta, rows, s, text=None, avd=None):
    """Every nuisance and situation factor a probe can find in x[rows]."""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    k = int(s.get("folds", 5))
    g = meta.group[rows]
    t = {}
    pred, done = oof_classify(x[rows], meta.game[rows], g, KNeighborsClassifier(int(s.get("knn_k", 15))), k)
    t["game_knn_balanced_accuracy"] = round(balanced_accuracy(meta.game[rows][done], pred[done]), 4) if done.any() else None
    worded = rows[meta.has_text[rows]]
    t["text_length_r2"] = ridge_r2(x[worded], meta.log_syll[worded], meta.group[worded], k)
    t["log_duration_r2"] = ridge_r2(x[rows], np.column_stack([meta.log_dur[rows], meta.log_dur[rows] ** 2]), g, k)
    floored = rows[np.isfinite(meta.floor_level[rows])]
    t["noise_floor_r2"] = ridge_r2(x[floored], meta.floor_level[floored], meta.group[floored], k)
    leveled = rows[np.isfinite(meta.level[rows])]
    t["level_r2"] = ridge_r2(x[leveled], meta.level[leveled], meta.group[leveled], k)
    t["bark_auc"] = auc_probe(x[rows], meta.bark[rows], g, k)
    t["stage_direction_auc"] = auc_probe(x[rows], meta.stage_direction[rows], g, k)
    counts = {}
    for i in rows:
        for tag in meta.tags[i]:
            counts[tag] = counts.get(tag, 0) + 1
    t["tag_auc"] = {tag: auc_probe(x[rows], np.array([tag in meta.tags[i] for i in rows]), g, k)
                    for tag, n in sorted(counts.items()) if n >= 20}  # fmt: skip
    if text is not None:
        have = np.array([meta.ids[i] in text for i in rows])
        tr = rows[have]
        if len(tr) > 10:
            te = np.stack([text[meta.ids[i]] for i in tr])
            te = te - te.mean(axis=0)
            _, _, vt = np.linalg.svd(te, full_matrices=False)
            t["text_content_r2"] = ridge_r2(x[tr], te @ vt[:5].T, meta.group[tr], k)
    if avd is not None:
        t["avd_r2"] = avd_r2(x, rows, meta, avd, k)
    return t


def avd_r2(x, rows, meta, avd, k):
    import numpy as np

    have = rows[[meta.ids[i] in avd for i in rows]]
    if len(have) < 10:
        return None
    y = np.array([avd[meta.ids[i]] for i in have])
    return ridge_r2(x[have], y, meta.group[have], k)


def paired_agreement(x, meta, rows):
    """How often a take's nearest neighbour (cosine, centred) among `rows`
    is its own paired take, against the chance rate 1/(n-1). Label-free
    reliability: higher means the space hears the same delivery twice."""
    import numpy as np

    pairs = meta.pairs(rows)
    if not pairs:
        return {"pairs": 0}
    xc = x[rows] - x[rows].mean(axis=0)
    xc /= np.maximum(np.linalg.norm(xc, axis=1, keepdims=True), 1e-12)
    pos = {r: i for i, r in enumerate(rows)}
    hits = ranks = 0.0
    for a, b in pairs:
        for p, q in ((a, b), (b, a)):
            sim = xc @ xc[pos[p]]
            sim[pos[p]] = -np.inf
            hits += float(np.argmax(sim) == pos[q])
            ranks += float((sim > sim[pos[q]]).sum()) / (len(rows) - 2 or 1)
    n = 2 * len(pairs)
    return {"pairs": len(pairs), "top1": round(hits / n, 4), "chance": round(1 / (len(rows) - 1), 5),
            "mean_rank_percentile": round(ranks / n, 4)}  # fmt: skip


# --- acceptance -----------------------------------------------------------------


def accept(after, before, game, s):
    """The fixed acceptance tests for one candidate on one split."""
    margin = float(s.get("game_margin", 0.05))
    max_r2 = float(s.get("max_length_r2", 0.05))
    max_drop = float(s.get("avd_max_drop", 0.10))
    chance = game["chance"]
    probes = [game[m] for m in ("knn", "svm") if game.get(m)]
    tests = {
        "game_gone": bool(probes) and all(p["upper"] <= chance + margin for p in probes),
        "text_length_gone": (after["text_length_r2"] is None) or after["text_length_r2"] <= max_r2,
        "duration_gone": (after["log_duration_r2"] is None) or after["log_duration_r2"] <= max_r2,
        "avd_guard": (after.get("avd_r2") is None or before.get("avd_r2") is None)
        or before["avd_r2"] - after["avd_r2"] <= max_drop,
    }
    return tests


def evaluate(x, raw, meta, rows, comparable_rows, s, avd, text, permutations=0, bootstrap=0):
    before = factor_table(raw, meta, rows, s, text, avd)
    after = factor_table(x, meta, rows, s, text, avd)
    game = game_probe(x, meta, comparable_rows, s, permutations, bootstrap)
    return {"before": before, "after": after, "game": game, "paired": paired_agreement(x, meta, rows),
            "tests": accept(after, before, game, s)}  # fmt: skip


# --- the job --------------------------------------------------------------------


def main():
    import numpy as np

    s = settings()
    out = out_dir()
    embed = env_path("VI_INPUT_EMBED")
    if embed is None:
        raise SystemExit("confounds needs embed as an input")
    clips = load_clips()
    report = json.loads((embed / "embed.json").read_text())
    names = s.get("candidates") or sorted(k for k in report["embeddings"] if k != "minilm")
    by_id = {c.id: c for c in clips}
    first, ids = load_matrix(embed, names[0])
    ids = [i for i in ids if i in by_id]
    clips = [by_id[i] for i in ids]
    meta = Meta(clips)
    rng = np.random.default_rng(SEED)
    comparable, rule = meta.comparable(rng)
    unique = set(np.flatnonzero(meta.unique))
    selection = meta.split(float(s.get("selection_share", 0.3)))
    usable = np.array([i in unique and not meta.excluded[i] for i in range(len(ids))])
    sel_rows, rep_rows = np.flatnonzero(usable & selection), np.flatnonzero(usable & ~selection)
    comp = np.zeros(len(ids), dtype=bool)
    comp[comparable] = True
    comp &= usable
    sel_comp, rep_comp = np.flatnonzero(comp & selection), np.flatnonzero(comp & ~selection)
    avd = {r["id"]: [r["avd"][k] for k in ("arousal", "dominance", "valence")]
           for r in read_jsonl(embed / "scores.jsonl") if r.get("avd")}  # fmt: skip
    text = {}
    if (embed / "embeddings" / "minilm.npy").exists():
        tm, tids = load_matrix(embed, "minilm")
        text = dict(zip(tids, tm))
    log(f"confounds: {len(ids)} clips, {len(names)} candidates, comparable subset {int(comp.sum())} ({rule})")

    summary = {
        "clips": len(ids),
        "used": int(usable.sum()),
        "excluded": int(meta.excluded.sum()),
        "duplicate_takes": len(ids) - len(unique),
        "comparable": {"rule": rule, "per_game": {g: int((comp & (meta.game == g)).sum()) for g in sorted(set(meta.game))}},
        "split": {"selection_conversations": len(set(meta.group[selection])),
                  "report_conversations": len(set(meta.group[~selection]))},
        "settings": {k: s[k] for k in sorted(s) if k != "backend"},
        "packages": packages(),
    }  # fmt: skip
    folds_needed = int(s.get("folds", 5))
    too_small = [f"{name} has {n} clips" for name, n in
                 [("the comparable subset", len(comparable)), ("its selection split", len(sel_comp)),
                  ("its report split", len(rep_comp))] if n < int(s.get("min_comparable", 30))]  # fmt: skip
    too_small += [f"the {name} split has {n} conversations, under {folds_needed} folds" for name, n in
                  [("selection", len(set(meta.group[sel_comp]))), ("report", len(set(meta.group[rep_comp])))]
                  if n < folds_needed]  # fmt: skip
    if too_small:
        stop(out, s, {**summary, "status": "stopped", "chosen": None, "candidates": {},
                      "stop_reason": "too little data to test anything: " + "; ".join(too_small)})  # fmt: skip
        return

    dims = int(s.get("pca_dims", 96))
    results = {}
    cleaned = {}
    for name in names:
        x, xids = load_matrix(embed, name)
        pos = {cid: i for i, cid in enumerate(xids)}
        x = x[[pos[i] for i in ids]]
        variants = {}
        for with_duration in (False, True):
            cx, raw, info, apply = clean(x, meta, comparable, dims, with_duration)
            ev = evaluate(cx, raw, meta, sel_rows, sel_comp, s, avd, text)
            variants[with_duration] = (cx, raw, info, ev, apply)
            t = ev["tests"]
            if t["text_length_gone"] and t["duration_gone"]:
                break  # duration is only erased when its effect is left
        with_duration = max(variants)  # the last one tried
        cx, raw, info, ev, apply = variants[with_duration]
        results[name] = {"transform": info, "selection": ev, "passed": all(ev["tests"].values())}
        if with_duration:
            results[name]["without_duration_erasure"] = variants[False][3]["tests"]
        cleaned[name] = (cx, raw, apply)
        log(f"confounds: {name}: {ev['tests']}")

    passing = [n for n in names if results[n]["passed"]]

    def rank(n):
        ev = results[n]["selection"]
        p = ev["paired"]
        lift = (p.get("top1", 0) - p.get("chance", 0)) if p.get("pairs") else 0
        return (lift, ev["after"].get("avd_r2") or 0)

    shortlist = sorted(passing, key=rank, reverse=True)[:2]
    perms, boots = int(s.get("permutations", 200)), int(s.get("bootstrap", 200))
    for n in shortlist:
        cx, raw, _ = cleaned[n]
        results[n]["report"] = evaluate(cx, raw, meta, rep_rows, rep_comp, s, avd, text, perms, boots)
        results[n]["report_passed"] = all(results[n]["report"]["tests"].values())
    chosen = next((n for n in shortlist if results[n]["report_passed"]), None)

    summary |= {"status": "chosen" if chosen else "stopped", "chosen": chosen, "shortlist": shortlist,
                "candidates": results}  # fmt: skip
    if chosen is None:
        summary["stop_reason"] = ("no candidate passed every acceptance test on the selection split"
                                  if not passing else "no shortlisted candidate passed on the report split")  # fmt: skip
    else:
        cx, _, apply = cleaned[chosen]
        np.save(out / "cleaned.npy", cx.astype(np.float32))
        (out / "cleaned.ids").write_text("".join(f"{i}\n" for i in ids))
        crops = embed / "crops" / f"{chosen}.npy"
        if crops.exists():
            cm, cids = load_matrix(embed, chosen, "crops")
            keep = [k for k, c in enumerate(cids) if c.rsplit("@", 1)[0] in meta.index]
            if keep:
                cids = [cids[k] for k in keep]
                game = np.array([meta.game[meta.index[c.rsplit("@", 1)[0]]] for c in cids])
                np.save(out / "cleaned_crops.npy", apply(cm[keep], game).astype(np.float32))
                (out / "cleaned_crops.ids").write_text("".join(f"{i}\n" for i in cids))
        log(f"confounds: chose {chosen}")
    stop(out, s, summary)


def stop(out, s, summary):
    """Write confounds.json and the check; a stop is reported, not a crash."""
    if summary["status"] == "stopped":
        log(f"confounds: STOPPED, {summary['stop_reason']}; see confounds.json")
    (out / "confounds.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n")
    checks.finish(out, [checks.Check("a candidate passed every acceptance test", int(summary["status"] == "chosen"), 1,
                                     kind="min", note="no cleaned space to cluster; see confounds.json")],
                  s.get("checks", "warn"))  # fmt: skip


def packages():
    from importlib.metadata import version

    return {n: version(n) for n in ("numpy", "scikit-learn")}


if __name__ == "__main__":
    main()
