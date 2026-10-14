"""cluster stage: find Tali's classes in the cleaned space. Home, CPU
(numpy, scikit-learn); no GPU.

Plan: docs/design/corpus-pipeline.md, step 5 and "Sample sizes, derived"
1-4 (thresholds are settings fixed before results). Format:
docs/design/corpus-format.md, cluster. Steps:

  1. Shortest usable clip: how often embed's crops (0.5-2 s) find their own
     source among the full clips; the minimum length is where that reaches
     `retrieval_share` of the 2 s rate. Shorter clips, LE1 barks included,
     don't form clusters. Paired-take agreement per length band cross-checks.
  2. PCA, with the number of components chosen by parallel analysis.
  3. GMM: k and covariance type by held-out likelihood (folds grouped by
     conversation); BIC and ICL reported for every fit.
  4. Components merged into classes by entropy (Baudry et al. 2010),
     while a merge saves at least `merge_min_slope` nats per clip it
     involves; the whole path is reported.
  5. HDBSCAN as a second opinion.
  6. Stability: `resamples` draws of ~80% of conversations without
     replacement, k and covariance type fixed; mean Jaccard per class
     (below 0.6 not trusted, 0.6-0.75 a pattern, 0.75+ a class).
  7. Audits: nuisance-only rejection (game, text length, noise floor),
     enrichment by tag and "topic, not tone?" flags, the recording-session
     check (flagged for Brandon), split-half class size, and one sensitivity
     run without the LE1 barks.
  8. Memberships for every clip: GMM responsibilities per class and a
     continuous score (log-likelihood ratio of the class against the rest).

If fewer than 2 classes reach Jaccard 0.75, or HDBSCAN calls more than 60%
noise, it stops and reports to Brandon: no automatic fallback to continuous
axes or EIV labels.

Outputs: clusters.jsonl, cluster.json, components.npy + .ids, checks.json.

Settings: max_dims, k_min, k_max, cov_types, folds, resamples, subsample,
merge_min_slope, hdbscan_min_size, keep_jaccard, trust_jaccard, max_noise, nuisance_max_auc,
retrieval_share, class_cosine, crop_lengths.
"""

import json
import math

import checks
from confounds import (
    Meta,
    auc_probe,
    folds,
    load_matrix,
    packages,
    paired_agreement,
)
from vi_job import env_path, load_clips, log, out_dir, settings, write_jsonl

SEED = 11


# --- 1. shortest usable clip ------------------------------------------------------


def unit_rows(x, centre):
    import numpy as np

    y = x - centre
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def shortest_usable(x, meta, rows, crops, crop_ids, share):
    """Per crop length, how often a crop's nearest full clip (cosine, in the
    cleaned centred space) is its source or the source's paired take. The
    minimum usable length is the shortest crop reaching `share` of the 2 s
    rate (the longest crop measured)."""
    import numpy as np

    from confounds import take_base

    centre = x[rows].mean(axis=0)
    full = unit_rows(x[rows], centre)
    bases = [take_base(meta.ids[i]) for i in rows]
    by_len = {}
    for cid, v in zip(crop_ids, unit_rows(crops, centre)):
        src, n = cid.rsplit("@", 1)
        hit = bases[int(np.argmax(full @ v))] == take_base(src)
        by_len.setdefault(float(n), []).append(hit)
    rates = {n: float(np.mean(h)) for n, h in sorted(by_len.items())}
    if not rates:
        return {"measured": False, "minimum_s": 0.0}
    top = rates[max(rates)]
    minimum = next((n for n, r in rates.items() if r >= share * top), max(rates))
    return {"measured": True, "retrieval": {f"{n:g}": round(r, 4) for n, r in rates.items()},
            "crops_per_length": len(next(iter(by_len.values()))), "share_of_longest": share,
            "minimum_s": minimum}  # fmt: skip


# --- 2-4. PCA, GMM, merging ----------------------------------------------------------


def parallel_analysis(x, rng, iters=20, cap=32):
    """Components whose eigenvalue beats the 95th percentile of the same
    data with every column shuffled independently (Horn 1965)."""
    import numpy as np

    xs = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
    real = np.linalg.eigvalsh(np.cov(xs.T))[::-1]
    null = []
    for _ in range(iters):
        shuffled = np.column_stack([rng.permutation(c) for c in xs.T])
        null.append(np.linalg.eigvalsh(np.cov(shuffled.T))[::-1])
    bar = np.quantile(np.array(null), 0.95, axis=0)
    keep = int(np.argmax(real <= bar)) if (real <= bar).any() else len(real)
    return max(2, min(cap, keep)), {"eigenvalues": [round(float(v), 4) for v in real[:cap]],
                                    "null_p95": [round(float(v), 4) for v in bar[:cap]]}  # fmt: skip


def gmm(k, cov, seed=SEED):
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(n_components=k, covariance_type=cov, n_init=3, reg_covar=1e-4, random_state=seed,
                           max_iter=500)  # fmt: skip


def entropy(resp):
    import numpy as np

    r = np.clip(resp, 1e-300, 1)
    return float(-(resp * np.log(r)).sum())


def choose_gmm(z, groups, s):
    """k and covariance type by mean held-out log-likelihood per clip, folds
    grouped by conversation. BIC and ICL of the all-data fit are reported."""
    import numpy as np

    table = []
    best = None
    for cov in s.get("cov_types", ["full", "diag"]):
        for k in range(int(s.get("k_min", 2)), int(s.get("k_max", 30)) + 1):
            if k * 2 > len(z):
                break
            held = []
            for train, test in folds(groups, int(s.get("folds", 5))):
                if len(train) <= k * 2:
                    continue
                held.append(gmm(k, cov).fit(z[train]).score(z[test]))
            m = gmm(k, cov).fit(z)
            bic = float(m.bic(z))
            row = {"k": k, "cov": cov, "heldout_ll": round(float(np.mean(held)), 4) if held else None,
                   "bic": round(bic, 2), "icl": round(bic + 2 * entropy(m.predict_proba(z)), 2)}  # fmt: skip
            table.append(row)
            if row["heldout_ll"] is not None and (best is None or row["heldout_ll"] > best["heldout_ll"]):
                best = row
    return best, table


def merge_path(resp):
    """Baudry et al. (2010): merge the two components whose union lowers the
    entropy of the soft assignment most, until one is left. Returns, at
    every step from K classes down to 1, the groups of components, the
    entropy, and how many clips the merges so far have involved."""
    groups = [[c] for c in range(resp.shape[1])]
    cols = [resp[:, c] for c in range(resp.shape[1])]
    path = [([list(g) for g in groups], entropy_cols(cols), 0.0)]
    merged = 0.0
    while len(cols) > 1:
        best = None
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                gain = ent(cols[a]) + ent(cols[b]) - ent(cols[a] + cols[b])
                if best is None or gain > best[0]:
                    best = (gain, a, b)
        _, a, b = best
        merged += float(cols[a].sum() + cols[b].sum())
        cols[a] = cols[a] + cols[b]
        groups[a] = groups[a] + groups[b]
        del cols[b], groups[b]
        path.append(([list(g) for g in groups], entropy_cols(cols), merged))
    return path


def ent(col):
    import numpy as np

    c = np.clip(col, 1e-300, 1)
    return float(-(col * np.log(c)).sum())


def entropy_cols(cols):
    return sum(ent(c) for c in cols)


def stop_merging(path, min_slope):
    """Number of classes to keep: merge while a merge lowers the entropy by
    at least `min_slope` nats per clip it involves (the slope of Baudry et
    al.'s entropy plot against merged clips), and stop at the first merge
    that doesn't. Components that overlap have steep merges; separate
    classes have flat ones."""
    n = len(path[0][0])
    for (_, e0, m0), (_, e1, m1) in zip(path, path[1:]):
        if m1 - m0 <= 0 or (e0 - e1) / (m1 - m0) < min_slope:
            return n
        n -= 1
    return n


class Classes:
    """A fitted GMM and the merge of its components into classes."""

    def __init__(self, model, groups):
        self.model, self.groups = model, groups

    def responsibilities(self, z):
        import numpy as np

        r = self.model.predict_proba(z)
        return np.column_stack([r[:, g].sum(axis=1) for g in self.groups])

    def scores(self, z):
        """log p(x | class) - log p(x | not class), from the components."""
        import numpy as np
        from scipy.special import logsumexp

        logp = self.model._estimate_weighted_log_prob(z)  # log w_c + log N(x | c)
        w = self.model.weights_
        out = []
        for g in self.groups:
            rest = [c for c in range(len(w)) if c not in g]
            inside = logsumexp(logp[:, g], axis=1) - math.log(w[g].sum())
            outside = logsumexp(logp[:, rest], axis=1) - math.log(w[rest].sum()) if rest else np.zeros(len(z))
            out.append(inside - outside)
        return np.column_stack(out)


def fit_classes(z, k, cov, n_classes=None, seed=SEED, min_slope=0.05):
    model = gmm(k, cov, seed).fit(z)
    path = merge_path(model.predict_proba(z))
    n = n_classes or stop_merging(path, min_slope)
    groups = next(g for g, _, _ in path if len(g) == n)
    return Classes(model, groups), path, n


# --- 6-7. stability and audits --------------------------------------------------------


def jaccard_stability(z, groups, labels, k, cov, n_classes, resamples, share, rng):
    """Hennig's clusterboot on conversations: refit on ~share of the
    conversations (no replacement), k and covariance fixed, and score each
    class by its best Jaccard match among the refit's classes."""
    import numpy as np

    convs = np.array(sorted(set(groups)))
    scores = {c: [] for c in range(n_classes)}
    for r in range(resamples):
        pick = set(rng.choice(convs, max(2, int(round(share * len(convs)))), replace=False))
        rows = np.flatnonzero([g in pick for g in groups])
        if len(rows) <= 2 * k:
            continue
        fitted, _, _ = fit_classes(z[rows], k, cov, n_classes, seed=SEED + 1 + r)
        new = fitted.responsibilities(z[rows]).argmax(axis=1)
        old = labels[rows]
        for c in range(n_classes):
            a = old == c
            if not a.any():
                continue
            scores[c].append(max(float((a & (new == d)).sum() / (a | (new == d)).sum()) for d in range(n_classes)))
    return {c: (round(float(np.mean(v)), 4) if v else None) for c, v in scores.items()}, resamples


def band(j, s):
    if j is None or j < float(s.get("trust_jaccard", 0.6)):
        return "not trusted"
    return "class" if j >= float(s.get("keep_jaccard", 0.75)) else "pattern"


def effective_conversations(groups):
    """exp of the entropy of the members' conversations."""
    counts = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    n = sum(counts.values())
    return math.exp(-sum(c / n * math.log(c / n) for c in counts.values())) if n else 0.0


def session_check(meta, rows, labels, c, rng, draws=200):
    import numpy as np

    members = rows[labels == c]
    eff = effective_conversations(meta.group[members])
    base = [effective_conversations(meta.group[rng.choice(rows, len(members), replace=False)]) for _ in range(draws)]
    counts = {}
    for g in meta.group[members]:
        counts[g] = counts.get(g, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])
    top2 = sum(n for _, n in top[:2]) / max(1, len(members))
    return {"effective_conversations": round(eff, 2), "baseline": round(float(np.mean(base)), 2),
            "ratio": round(eff / max(float(np.mean(base)), 1e-9), 3), "biggest_conversation": top[0][0] if top else None,
            "top2_share": round(top2, 3)}  # fmt: skip


def split_half(z, members, centre, rng, sizes=(10, 20, 40, 80, 160), draws=50):
    """Split-half cosine of the class vector (class mean minus corpus mean)
    from n of its clips, stepped up with Spearman-Brown to n."""
    import numpy as np

    out = {}
    for n in sizes:
        if n > len(members):
            break
        vals = []
        for _ in range(draws):
            pick = rng.choice(members, n, replace=False)
            a, b = z[pick[: n // 2]].mean(0) - centre, z[pick[n // 2 :]].mean(0) - centre
            r = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            vals.append(2 * r / (1 + r) if r > -1 else -1.0)
        out[n] = round(float(np.median(vals)), 4)
    return out


def enrichment(meta, members, rows, min_count=5):
    """Tags over-represented in a class: share inside / share in all rows."""
    total = {}
    inside = {}
    for i in rows:
        for t in meta.tags[i]:
            total[t] = total.get(t, 0) + 1
    for i in members:
        for t in meta.tags[i]:
            inside[t] = inside.get(t, 0) + 1
    out = []
    for t, n in inside.items():
        if n >= min_count:
            ratio = (n / len(members)) / (total[t] / len(rows))
            out.append({"tag": t, "count": n, "ratio": round(ratio, 2)})
    return sorted(out, key=lambda r: -r["ratio"])[:8]


def nuisance_auc(meta, rows, labels, c, k):
    """How well game, text length and noise floor alone predict membership."""
    import numpy as np

    games = sorted(set(meta.game))
    x = np.column_stack([meta.game[rows] == g for g in games] +
                        [np.nan_to_num(meta.log_syll[rows], nan=float(np.nanmedian(meta.log_syll))),
                         np.nan_to_num(meta.floor_level[rows], nan=float(np.nanmedian(meta.floor_level)))
                         if np.isfinite(meta.floor_level).any() else np.zeros(len(rows))]).astype(float)  # fmt: skip
    return auc_probe(x, labels == c, meta.group[rows], k)


# --- the job --------------------------------------------------------------------------


def main():
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score

    s = settings()
    out = out_dir()
    conf, embed = env_path("VI_INPUT_CONFOUNDS"), env_path("VI_INPUT_EMBED")
    if conf is None or embed is None:
        raise SystemExit("cluster needs confounds and embed as inputs")
    report = json.loads((conf / "confounds.json").read_text())
    if report.get("status") != "chosen":
        raise SystemExit(f"confounds stopped ({report.get('stop_reason')}); nothing to cluster. See confounds.json.")
    rng = np.random.default_rng(SEED)
    x = np.load(conf / "cleaned.npy").astype(np.float64)
    ids = (conf / "cleaned.ids").read_text().split()
    by_id = {c.id: c for c in load_clips()}
    meta = Meta([by_id[i] for i in ids])
    k_folds = int(s.get("folds", 5))

    # 1. Shortest usable clip, on the clips that can form clusters.
    base_rows = np.flatnonzero(meta.unique & ~meta.excluded)
    shortest = {"measured": False, "minimum_s": 0.0}
    if (conf / "cleaned_crops.npy").exists():
        crops = np.load(conf / "cleaned_crops.npy").astype(np.float64)
        crop_ids = (conf / "cleaned_crops.ids").read_text().split()
        shortest = shortest_usable(x, meta, base_rows, crops, crop_ids, float(s.get("retrieval_share", 0.8)))
    dur = np.exp(meta.log_dur)
    rows = base_rows[dur[base_rows] >= shortest["minimum_s"]]
    bands = {"short (<2 s)": rows[dur[rows] < 2], "long (2 s+)": rows[dur[rows] >= 2]}
    shortest["paired_by_length"] = {b: paired_agreement(x, meta, r) for b, r in bands.items() if len(r) > 2}
    log(f"cluster: {len(rows)} clips form clusters (minimum {shortest['minimum_s']} s)")

    # 2-4.
    dims, pa = parallel_analysis(x[rows], rng, cap=int(s.get("max_dims", 32)))
    pca = PCA(n_components=dims, random_state=SEED).fit(x[rows])
    z_all = pca.transform(x)
    z = z_all[rows]
    best, table = choose_gmm(z, meta.group[rows], s)
    k, cov = best["k"], best["cov"]
    classes, path, n_classes = fit_classes(z, k, cov, min_slope=float(s.get("merge_min_slope", 0.05)))
    resp = classes.responsibilities(z)
    labels = resp.argmax(axis=1)
    # Classes in order of size.
    order = np.argsort(-np.bincount(labels, minlength=n_classes))
    classes.groups = [classes.groups[i] for i in order]
    resp = classes.responsibilities(z)
    labels = resp.argmax(axis=1)
    log(f"cluster: GMM k={k} {cov}, merged to {n_classes} classes")

    # 5. HDBSCAN.
    hd = HDBSCAN(min_cluster_size=int(s.get("hdbscan_min_size", 15))).fit(z)
    noise = float((hd.labels_ == -1).mean())
    core = hd.labels_ != -1
    hdbscan = {"clusters": int(hd.labels_.max() + 1), "noise_share": round(noise, 4),
               "ari_with_classes": round(float(adjusted_rand_score(labels[core], hd.labels_[core])), 4)
               if core.sum() > 1 else None}  # fmt: skip

    # 6. Stability.
    jac, draws = jaccard_stability(z, meta.group[rows], labels, k, cov, n_classes, int(s.get("resamples", 200)),
                                   float(s.get("subsample", 0.8)), rng)  # fmt: skip

    # 7. Audits per class.
    text = {}
    if (embed / "embeddings" / "minilm.npy").exists():
        tm, tids = load_matrix(embed, "minilm")
        text = dict(zip(tids, tm))
    centre = z.mean(axis=0)
    per = []
    min_size = None
    for c in range(n_classes):
        members = rows[labels == c]
        member_pos = np.flatnonzero(labels == c)
        audio_auc = auc_probe(z, labels == c, meta.group[rows], k_folds)
        worded = np.array([meta.ids[i] in text for i in rows])
        text_auc = None
        if worded.sum() > 20:
            tz = np.stack([text[meta.ids[i]] for i in rows[worded]])
            text_auc = auc_probe(tz, (labels == c)[worded], meta.group[rows][worded], k_folds)
        nuis = nuisance_auc(meta, rows, labels, c, k_folds)
        sizes = split_half(z, member_pos, centre, rng)
        enough = next((n for n, v in sizes.items() if v >= float(s.get("class_cosine", 0.9))), None)
        per.append({
            "class": f"c{c}",
            "size": int(len(members)),
            "components": [int(i) for i in classes.groups[c]],
            "jaccard": jac[c],
            "stability": band(jac[c], s),
            "nuisance_auc": nuis,
            "rejected_for_nuisance": nuis is not None and nuis > float(s.get("nuisance_max_auc", 0.7)),
            "audio_auc": audio_auc,
            "text_auc": text_auc,
            "topic_not_tone": text_auc is not None and audio_auc is not None and text_auc > audio_auc,
            "games": {g: int((meta.game[members] == g).sum()) for g in sorted(set(meta.game))},
            "barks": int(meta.bark[members].sum()),
            "enrichment": enrichment(meta, members, rows),
            "session": session_check(meta, rows, labels, c, rng),
            "split_half_cosine": {str(n): v for n, v in sizes.items()},
            "size_for_cosine": enough,
        })  # fmt: skip
    for p in per:
        sess = p["session"]
        p["session_flag"] = sess["ratio"] < 0.5 or sess["top2_share"] > 0.5
    reached = [p["size_for_cosine"] for p in per if p["size_for_cosine"]]
    min_size = max(reached) if reached else None
    for p in per:
        p["below_min_size"] = min_size is not None and p["size"] < min_size

    # Leave the biggest conversation out, per flagged-or-not class.
    for c, p in enumerate(per):
        big = p["session"]["biggest_conversation"]
        keep = meta.group[rows] != big
        if keep.sum() > 2 * k:
            refit, _, _ = fit_classes(z[keep], k, cov, n_classes, seed=SEED + 999)
            new = refit.responsibilities(z[keep]).argmax(axis=1)
            a = labels[keep] == c
            p["session"]["jaccard_without_biggest"] = round(max(
                float((a & (new == d)).sum() / max(1, (a | (new == d)).sum())) for d in range(n_classes)), 4)  # fmt: skip

    # Sensitivity: without the LE1 barks.
    sensitivity = None
    no_bark = ~meta.bark[rows]
    if meta.bark[rows].any() and no_bark.sum() > 2 * k:
        refit, _, _ = fit_classes(z[no_bark], k, cov, n_classes)
        new = refit.responsibilities(z[no_bark]).argmax(axis=1)
        sensitivity = {"without_barks_ari": round(float(adjusted_rand_score(labels[no_bark], new)), 4),
                       "barks_in_clustering": int(meta.bark[rows].sum())}  # fmt: skip

    # 8. Memberships for every clip.
    r_all = classes.responsibilities(z_all)
    s_all = classes.scores(z_all)
    formed = set(rows.tolist())
    twin = {}
    for i, h in enumerate(meta.sha):
        twin.setdefault(h, i)
    records = []
    hd_label = dict(zip(rows.tolist(), hd.labels_.tolist()))
    for i, cid in enumerate(ids):
        j = twin[meta.sha[i]]
        records.append({"id": cid, "cluster": f"c{int(r_all[j].argmax())}",
                        "responsibilities": {f"c{c}": round(float(r_all[j, c]), 4) for c in range(n_classes)},
                        "scores": {f"c{c}": round(float(s_all[j, c]), 3) for c in range(n_classes)},
                        "formed": i in formed, "hdbscan": hd_label.get(i)})  # fmt: skip
    write_jsonl(out / "clusters.jsonl", records)
    # Per component log w_c + log N(x | c), so classes can rescore merges
    # exactly without refitting.
    logp = classes.model._estimate_weighted_log_prob(z_all)
    twins = [twin[meta.sha[i]] for i in range(len(ids))]
    np.save(out / "components.npy", logp[twins].astype(np.float32))
    (out / "components.ids").write_text("".join(f"{i}\n" for i in ids))

    stable = sum(p["stability"] == "class" for p in per)
    stop_reasons = []
    if stable < 2:
        stop_reasons.append(f"{stable} class(es) at Jaccard >= {s.get('keep_jaccard', 0.75)}")
    if noise > float(s.get("max_noise", 0.6)):
        stop_reasons.append(f"HDBSCAN noise {noise:.0%}")
    result = {
        "status": "stopped" if stop_reasons else "ok",
        "stop_reason": "; ".join(stop_reasons) or None,
        "candidate": report["chosen"],
        "clips": len(ids),
        "formed_by": int(len(rows)),
        "shortest_usable": shortest,
        "pca": {"dims": dims, "parallel_analysis": pa},
        "gmm": {"k": k, "cov": cov, "table": table, "weights": [round(float(w), 6) for w in classes.model.weights_]},
        "merge": {"classes": n_classes, "entropy_path": [{"classes": len(g), "entropy": round(e, 2), "merged_clips": round(m, 1)}
                                                       for g, e, m in path]},
        "hdbscan": hdbscan,
        "stability": {"resamples": draws, "subsample": float(s.get("subsample", 0.8)), "by": "conversation"},
        "min_class_size": min_size,
        "classes": per,
        "sensitivity": sensitivity,
        "settings": {k_: s[k_] for k_ in sorted(s) if k_ != "backend"},
        "packages": packages(),
    }  # fmt: skip
    (out / "cluster.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    if stop_reasons:
        log(f"cluster: STOPPED for Brandon: {result['stop_reason']}")
    checks.finish(out, [
        checks.Check("classes at keep_jaccard", stable, 2, kind="min", note="stop and report to Brandon"),
        checks.Check("HDBSCAN noise share", round(noise, 3), float(s.get("max_noise", 0.6)), note="stop and report"),
    ], s.get("checks", "warn"))  # fmt: skip


if __name__ == "__main__":
    main()
