"""calibrate stage: fit the labels to Brandon's reviews.

Plan: docs/design/control-vector-datasets.md, step 5b. Reviews are the
ground truth; every signal keeps its weight on an axis only as far as it
agrees with them there.

Per axis:
  probe     a ridge regression from each clip's emotion2vec+ embedding (label's
            embeddings/emotion.npy) to the reviewed scores, so every axis gets
            an audio signal learned from her own voice, including axes no
            emotion2vec+ class covers. Its scores on reviewed clips are
            out-of-fold, so they're honest.
  weights   each signal's weight is its Spearman correlation with the reviews,
            squared (0 if it doesn't beat 0.1), scaled so the best is 1. They're
            fit on reviewed clips outside the calibration sample when there are
            enough, so the sample stays an honest test.
  accuracy  every signal's and the fused score's Spearman and mean absolute
            error against the reviews, on the calibration sample: the random
            clips the review page shows first (queue::calibration). This says
            whether an axis is labelled well enough to build a vector from.

Outputs: labels.jsonl (label's schema; each axis gains a `probe` signal and
is refused with the calibrated weights; reviewed clips take the reviewed
value with confidence 1 and `reviewed: true`; flagged clips get
`flagged: <reason>`), axes.json, and calibration.json.

Needs numpy and scikit-learn, except with backend "fake", where the probe
predicts each axis's mean review (for CI).

Settings: backend, calibration (sample size, 150), min_reviews (30),
reviews (path; default VI_CORPUS/reviews.jsonl, or files/reviews.jsonl).
"""

import hashlib
import json
import os
from pathlib import Path

import fusion
from vi_job import env_path, fake_models, log, out_dir, read_jsonl, settings, write_jsonl

PROBE_CONFIDENCE = 0.8
MIN_CORRELATION = 0.1


def main():
    s = settings()
    fake = fake_models()
    out = out_dir()
    label = env_path("VI_INPUT_LABEL")
    if label is None:
        raise SystemExit("calibrate needs the label stage as an input")
    labels = read_jsonl(label / "labels.jsonl")
    axes = json.loads((label / "axes.json").read_text())
    axis_ids = [a["id"] for a in axes]

    reviews = latest_reviews(reviews_path(s))
    kept = {i: r["scores"] for i, r in reviews.items() if r["verdict"] == "keep" and i in {l["id"] for l in labels}}
    flagged = {i: r.get("flag") or "flagged" for i, r in reviews.items() if r["verdict"] == "flag"}
    sample = set(calibration([l["id"] for l in labels], int(s.get("calibration", 150))))
    test_ids = [i for i in kept if i in sample]
    fit_ids = [i for i in kept if i not in sample]
    min_reviews = int(s.get("min_reviews", 30))
    if len(fit_ids) < min_reviews:
        fit_ids = list(kept)  # too few outside the sample; fit on everything, say so
    log(f"calibrate: {len(kept)} reviewed clips ({len(test_ids)} in the calibration sample), {len(flagged)} flagged")

    ids = [l["id"] for l in labels]
    by_id = {l["id"]: l for l in labels}
    vectors = None if fake else load_embeddings(label / "embeddings", "emotion")

    report = {"reviewed": len(kept), "calibration_sample": len(test_ids), "flagged": len(flagged), "axes": {}}
    weights = {}
    probe = {}
    for a in axis_ids:
        truth = {i: kept[i][a] for i in kept if a in kept[i]}
        if len(truth) < min_reviews:
            probe[a] = {}
            weights[a] = dict(fusion.DEFAULT_WEIGHTS)
            report["axes"][a] = {"reviewed": len(truth), "note": f"fewer than {min_reviews} reviews; defaults kept"}
            continue
        probe[a] = fit_probe(ids, truth, vectors, fake)
        cols = signal_columns(by_id, a, probe[a])
        weights[a] = {**fusion.DEFAULT_WEIGHTS, **fit_weights(cols, truth, [i for i in fit_ids if i in truth])}
        fused_before = {i: by_id[i]["scores"][a]["value"] for i in ids}
        fused_after = {i: fuse(cols, i, weights[a])["value"] for i in ids}
        test = [i for i in test_ids if i in truth]
        report["axes"][a] = {
            "reviewed": len(truth),
            "weights": weights[a],
            "weights_fit_on": "outside the calibration sample" if len(fit_ids) < len(kept) else "all reviews (optimistic)",
            "on_calibration_sample": {
                **{n: accuracy(values(c), truth, test) for n, c in cols.items()},
                "fused_default": accuracy(fused_before, truth, test),
                "fused_calibrated": accuracy(fused_after, truth, test),
            },
        }

    records = []
    for l in labels:
        i = l["id"]
        r = dict(l)
        r["scores"] = {}
        for a in axis_ids:
            old = l["scores"].get(a, {})
            conf = old.get("signal_confidence", {})
            sig = {n: None if v is None else (v, conf.get(n, fusion_conf(n))) for n, v in old.get("signals", {}).items()}
            p = probe.get(a, {}).get(i)
            sig["probe"] = None if p is None else (p, PROBE_CONFIDENCE)
            r["scores"][a] = fusion.fuse(sig, weights[a])
            if i in kept and a in kept[i]:
                r["scores"][a].update(value=round(kept[i][a], 4), confidence=1.0)
                r["scores"][a]["signals"]["review"] = round(kept[i][a], 4)
        if i in kept:
            r["reviewed"] = True
        if i in flagged:
            r["flagged"] = flagged[i]
        records.append(r)

    write_jsonl(out / "labels.jsonl", records)
    (out / "axes.json").write_text(json.dumps(axes, indent=1) + "\n")
    (out / "calibration.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log(f"calibrate: wrote {len(records)} labels")


# Confidence a signal gets when refused if label didn't record one
# (`signal_confidence`, written since 2026-10-13).
SIGNAL_CONFIDENCE = {"audio": 0.8, "text": 0.6, "prior": 0.6, "audio_llm": 0.6}


def fusion_conf(name):
    return SIGNAL_CONFIDENCE.get(name, 0.5)


def signal_columns(by_id, axis, probe):
    """name -> {id: (value, confidence)} for every signal that scored this
    axis anywhere, plus the probe."""
    cols = {}
    for i, l in by_id.items():
        score = l["scores"].get(axis, {})
        conf = score.get("signal_confidence", {})
        for n, v in score.get("signals", {}).items():
            if v is not None:
                cols.setdefault(n, {})[i] = (v, conf.get(n, fusion_conf(n)))
    if probe:
        cols["probe"] = {i: (v, PROBE_CONFIDENCE) for i, v in probe.items()}
    return cols


def values(col):
    return {i: v for i, (v, _) in col.items()}


def fuse(cols, i, weights):
    return fusion.fuse({n: c.get(i) for n, c in cols.items()}, weights)


def fit_weights(cols, truth, ids):
    """Each signal's Spearman with the reviews on `ids`, squared, 0 below
    MIN_CORRELATION, scaled so the best is 1. Signals with too few scored
    clips to judge keep half their default."""
    rs = {}
    for n, c in cols.items():
        pairs = [i for i in ids if i in c]
        r = fusion.spearman([c[i][0] for i in pairs], [truth[i] for i in pairs])
        rs[n] = r
    best = max((r for r in rs.values() if r is not None), default=None)
    out = {}
    for n, r in rs.items():
        if r is None or best is None or best <= 0:
            out[n] = round(fusion.DEFAULT_WEIGHTS.get(n, 0.5) / 2, 3)
        else:
            out[n] = round((max(0.0, r) ** 2) / best**2, 3) if r >= MIN_CORRELATION else 0.0
    return out


def accuracy(col, truth, ids):
    pairs = [i for i in ids if i in col and col[i] is not None]
    if not pairs:
        return {"n": 0}
    return {
        "n": len(pairs),
        "spearman": fusion.spearman([col[i] for i in pairs], [truth[i] for i in pairs]),
        "mae": round(sum(abs(col[i] - truth[i]) for i in pairs) / len(pairs), 3),
    }


def fit_probe(ids, truth, vectors, fake):
    """{id: predicted value} for every clip with an embedding; reviewed clips
    get out-of-fold predictions."""
    if fake:
        mean = sum(truth.values()) / len(truth)
        return {i: round(mean, 4) for i in ids}
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    have = [i for i in ids if i in vectors]
    x_all = np.array([vectors[i] for i in have], dtype=np.float32)
    train = [k for k, i in enumerate(have) if i in truth]
    y = np.array([truth[have[k]] for k in train], dtype=np.float32)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))
    folds = KFold(n_splits=min(5, len(train)), shuffle=True, random_state=0)
    oof = cross_val_predict(model, x_all[train], y, cv=folds)
    pred = model.fit(x_all[train], y).predict(x_all)
    pred[train] = oof
    return {i: round(float(min(1.0, max(0.0, p))), 4) for i, p in zip(have, pred)}


def load_embeddings(folder, name):
    """{id: vector} from write_embeddings' .npy and .ids."""
    import numpy as np

    x = np.load(folder / f"{name}.npy")
    ids = (folder / f"{name}.ids").read_text().split()
    return dict(zip(ids, x))


def reviews_path(s):
    if s.get("reviews"):
        return Path(s["reviews"])
    corpus = os.environ.get("VI_CORPUS")
    if corpus and (Path(corpus) / "reviews.jsonl").exists():
        return Path(corpus) / "reviews.jsonl"
    return Path("files/reviews.jsonl")


def latest_reviews(path):
    """id -> the last review record for it (the review page appends)."""
    if not path.exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        out[r["id"]] = r
    return {i: r for i, r in out.items() if r.get("verdict") != "skip"}


def calibration(ids, n):
    """The review page's calibration sample (crates/review/src/queue.rs):
    the n ids whose lowercase hex SHA-256 sorts first."""
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:n]


if __name__ == "__main__":
    main()
