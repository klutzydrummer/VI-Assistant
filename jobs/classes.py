"""classes: the corpus pipeline's final output, following
docs/design/corpus-pipeline.md step 6 and the build primer's step 7. Home,
CPU (numpy, scipy).

Takes the clusters, applies Brandon's purity-review decisions (name, merge,
split, reject) and writes the class table, every clip's memberships, the
Empathic-Insight-Voice backup labelling and a report with every acceptance
number and every check that failed.

- Merges and splits are rescored exactly from cluster's per-component
  log-probabilities: a merged class is the union of its clusters' GMM
  components, and a split separates a cluster back into its components
  (one with a single component can't be split here; that is reported).
- A rejected cluster's clips keep their responsibilities for the classes
  that remain; nothing is renormalised, so a clip mostly in a rejected
  cluster belongs weakly to everything.
- Purity is recomputed from purity.jsonl the same way the review page does:
  Wald's SPRT on member answers, corrected for the false-yes rate on foils.
- A cluster without a decision is kept as `unreviewed`, never dropped.

Inputs: cluster, embed (scores.jsonl), confounds (confounds.json), and the
corpus's purity.jsonl (listed as `corpus:purity.jsonl`, may be missing).
Outputs: classes.json, memberships.jsonl, eiv_labels.jsonl, report.md,
checks.json.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

import checks
from vi_job import env_path, log, out_dir, read_jsonl, settings, write_jsonl

P0, P1, ALPHA, BETA = 0.5, 0.8, 0.05, 0.05
MAX_MEMBERS = 40


# --- purity ---------------------------------------------------------------------------


def read_purity(path):
    if path is None or not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: {e}")
    return out


def purity_summary(records, members_available):
    """Per cluster: SPRT verdict, observed and foil-corrected purity; overall:
    false-yes rate and Cohen's kappa on hidden repeats. Mirrors the review
    page (crates/review/src/purity.rs)."""
    a, b = math.log((1 - BETA) / ALPHA), math.log(BETA / (1 - ALPHA))
    answers = [r for r in records if r.get("kind") == "answer"]
    foils = [r for r in answers if r.get("foil")]
    false_yes = sum(r["fits"] for r in foils) / len(foils) if foils else None
    per = {}
    for c, available in members_available.items():
        s = {"answers": 0, "members_yes": 0, "members_no": 0, "foils": 0, "llr": 0.0, "sprt": None,
             "skips": sum(r.get("kind") == "skip" and r.get("cluster") == c for r in records)}
        for r in answers:
            if r["cluster"] != c:
                continue
            s["answers"] += 1
            if r.get("foil"):
                s["foils"] += 1
            elif not r.get("repeat"):
                yes = bool(r["fits"])
                s["members_yes" if yes else "members_no"] += 1
                s["llr"] += math.log(P1 / P0) if yes else math.log((1 - P1) / (1 - P0))
                if s["sprt"] is None:
                    s["sprt"] = "pure" if s["llr"] >= a else "impure" if s["llr"] <= b else None
        heard = s["members_yes"] + s["members_no"]
        if s["sprt"] is None and heard and (heard >= available or heard >= MAX_MEMBERS):
            s["sprt"] = "undecided"
        s["observed"] = s["members_yes"] / heard if heard else None
        if heard and false_yes is not None and false_yes < 1:
            s["purity"] = min(1.0, max(0.0, (s["observed"] - false_yes) / (1 - false_yes)))
        else:
            s["purity"] = s["observed"]
        s["llr"] = round(s["llr"], 3)
        per[c] = s
    first, pairs = {}, []
    for r in answers:
        if r.get("foil"):
            continue
        if r.get("repeat"):
            if r["id"] in first:
                pairs.append((first[r["id"]], bool(r["fits"])))
        else:
            first.setdefault(r["id"], bool(r["fits"]))
    kappa = None
    if pairs:
        n = len(pairs)
        agree = sum(x == y for x, y in pairs) / n
        pa, pb = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
        chance = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - chance) / (1 - chance) if chance < 1 else None
    overall = {"false_yes": false_yes, "foils": len(foils), "kappa": kappa, "repeat_pairs": len(pairs),
               "answers": len(answers), "seconds": round(sum(r.get("seconds", 0) for r in answers), 1)}  # fmt: skip
    return per, overall


def decisions(records):
    """The last decision per cluster."""
    out = {}
    for r in records:
        if r.get("kind") == "decision":
            out[r["cluster"]] = r
    return out


# --- classes from decisions -----------------------------------------------------------------


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "class"


def resolve(clusters, decided):
    """Final classes: [{key, name, status, clusters, components}], plus the
    rejected clusters and any problems (merge cycles, merges into rejected
    clusters)."""
    problems = []

    def root(c, seen=()):
        d = decided.get(c)
        if d and d.get("action") == "merge":
            into = d.get("into")
            if into not in clusters:
                problems.append(f"{c} merges into unknown cluster {into}")
                return c
            if into in seen or into == c:
                problems.append(f"merge cycle through {c}")
                return c
            return root(into, (*seen, c))
        return c

    groups = {}
    for c in clusters:
        groups.setdefault(root(c), []).append(c)
    classes, rejected, used = [], [], set()

    def key_for(base):
        k, n = base, 2
        while k in used:
            k, n = f"{base}_{n}", n + 1
        used.add(k)
        return k

    for head, members in groups.items():
        d = decided.get(head)
        action = d.get("action") if d else None
        comps = sorted(i for c in members for i in clusters[c]["components"])
        if action == "reject":
            rejected.extend(members)
            if len(members) > 1:
                problems.append(f"{', '.join(m for m in members if m != head)} merged into rejected {head}")
            continue
        if action == "split" and len(comps) > 1:
            for n, comp in enumerate(comps):
                classes.append({"key": key_for(f"{head}.{n}"), "name": None, "status": "split_part",
                                "clusters": members, "components": [comp]})  # fmt: skip
            continue
        if action == "name" and d.get("name"):
            name, status = d["name"].strip(), "named"
        else:
            name = None
            status = {"split": "split_single_component", "merge": "unreviewed"}.get(action, "unreviewed")
        base = slug(name) if name else "+".join(members)
        classes.append({"key": key_for(base), "name": name, "status": status, "clusters": members,
                        "components": comps})  # fmt: skip
    return classes, rejected, problems


def memberships(logp, weights, classes):
    """Responsibilities and continuous scores (log p(x | class) - log p(x |
    not class)) for each class, from per-component weighted log-probs."""
    import numpy as np
    from scipy.special import logsumexp

    post = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    k = logp.shape[1]
    resp, score = [], []
    for c in classes:
        g = c["components"]
        rest = [i for i in range(k) if i not in g]
        resp.append(post[:, g].sum(axis=1))
        inside = logsumexp(logp[:, g], axis=1) - math.log(weights[g].sum())
        outside = logsumexp(logp[:, rest], axis=1) - math.log(weights[rest].sum()) if rest else 0.0
        score.append(inside - outside)
    return np.column_stack(resp), np.column_stack(score)


# --- the report -----------------------------------------------------------------------------


def fmt(v, digits=3):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def report_md(confounds, cluster, classes, rejected, overall, failed, s, unassigned=None):
    lines = ["# Corpus classes report", ""]
    lines += [f"Status: **{s['status']}**", ""]
    if failed:
        lines += ["## Checks that failed", ""] + [f"- {f}" for f in failed] + [""]
    else:
        lines += ["No check failed.", ""]

    lines += ["## Confounds (step 4)", ""]
    chosen = confounds.get("chosen")
    lines += [f"Chosen embedding: `{chosen}`. Comparable subset: {fmt(confounds.get('comparable'))} clips."]
    cand = (confounds.get("candidates") or {}).get(chosen) or {}
    rep = cand.get("report") or {}
    game = rep.get("game") or {}
    margin = (confounds.get("settings") or {}).get("game_margin", 0.05)
    if game:
        lines += ["", f"Game probe on the report split (chance {fmt(game.get('chance'))}, "
                      f"limit chance + {margin}):", "",
                  "| probe | balanced accuracy | upper bound | permutation p |", "|---|---|---|---|"]  # fmt: skip
        for probe in ("knn", "svm"):
            g = game.get(probe) or {}
            lines.append(f"| {probe} | {fmt(g.get('balanced_accuracy'))} | {fmt(g.get('upper'))} | "
                         f"{fmt(g.get('permutation_p'))} |")  # fmt: skip
    before, after = rep.get("before") or {}, rep.get("after") or {}
    keys = [k for k in before if not isinstance(before[k], dict)]
    if keys:
        lines += ["", "| measure | before | after |", "|---|---|---|"]
        lines += [f"| {k} | {fmt(before.get(k))} | {fmt(after.get(k))} |" for k in keys]
    tests = rep.get("tests") or {}
    if tests:
        lines += ["", "Tests on the report split: " + ", ".join(f"{k} {fmt(v)}" for k, v in tests.items()) + "."]
    paired = rep.get("paired")
    if paired:
        lines += [f"Paired-take agreement: top-1 {fmt(paired.get('top1'))} over {paired.get('pairs')} pairs "
                  f"(chance {fmt(paired.get('chance'), 4)})."]  # fmt: skip
    lines.append("")

    lines += ["## Clusters (step 5)", ""]
    su = cluster.get("shortest_usable") or {}
    lines += [f"Clips: {cluster.get('clips')}, forming clusters: {cluster.get('formed_by')}. "
              f"Shortest usable clip: {fmt(su.get('minimum_s'))} s "
              f"(retrieval by crop length {json.dumps(su.get('retrieval'))}).",
              f"PCA: {(cluster.get('pca') or {}).get('dims')} dimensions. "
              f"GMM: k = {(cluster.get('gmm') or {}).get('k')}, {(cluster.get('gmm') or {}).get('cov')}; "
              f"merged to {(cluster.get('merge') or {}).get('classes')} clusters.",
              f"HDBSCAN: {json.dumps(cluster.get('hdbscan'))}. Minimum class size: {fmt(cluster.get('min_class_size'))}.",
              f"Without LE1 barks: {json.dumps(cluster.get('sensitivity'))}.", ""]  # fmt: skip
    lines += ["| cluster | size | Jaccard | band | nuisance AUC | topic, not tone? | session flag | below min size |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for p in cluster.get("classes", []):
        lines.append(f"| {p['class']} | {p['size']} | {fmt(p.get('jaccard'))} | {p.get('stability')} | "
                     f"{fmt(p.get('nuisance_auc'))} | {fmt(p.get('topic_not_tone'))} | {fmt(p.get('session_flag'))} | "
                     f"{fmt(p.get('below_min_size'))} |")  # fmt: skip
    lines.append("")

    lines += ["## Purity review (step 6)", ""]
    lines += [f"{overall['answers']} answers, {overall['seconds'] / 60:.0f} minutes. False yes on foils: "
              f"{fmt(overall['false_yes'])} over {overall['foils']} foils. Consistency (Cohen's kappa): "
              f"{fmt(overall['kappa'])} on {overall['repeat_pairs']} of 50 repeats.", ""]  # fmt: skip

    lines += ["## Classes", ""]
    lines += ["| class | name | status | clusters | size | purity | SPRT | Jaccard |", "|---|---|---|---|---|---|---|---|"]
    for c in classes:
        lines.append(f"| {c['key']} | {c['name'] or '–'} | {c['status']} | {', '.join(c['clusters'])} | {c['size']} | "
                     f"{fmt(c['purity'].get('purity'))} | {', '.join(c['purity'].get('sprt') or ['–'])} | "
                     f"{fmt(c['stability'].get('jaccard'))} |")  # fmt: skip
    if rejected:
        lines += ["", f"Rejected: {', '.join(rejected)}."]
    if unassigned is not None:
        lines += ["", f"Clips in no class (top class under {s.get('min_membership', 0.5)} responsibility): {unassigned}."]
    lines.append("")
    return "\n".join(lines)


# --- the job --------------------------------------------------------------------------------


def main():
    import numpy as np

    s = settings()
    out = out_dir()
    cluster_dir = env_path("VI_INPUT_CLUSTER")
    if cluster_dir is None:
        sys.exit("classes needs the cluster stage as an input")
    cluster = json.loads((cluster_dir / "cluster.json").read_text())
    if cluster.get("status") != "ok":
        sys.exit(f"cluster stopped ({cluster.get('stop_reason')}): that goes to Brandon, not to classes")
    if not (cluster_dir / "components.npy").exists():
        sys.exit("cluster's output has no components.npy; rerun cluster")
    confounds_dir = env_path("VI_INPUT_CONFOUNDS")
    confounds = json.loads((confounds_dir / "confounds.json").read_text()) if confounds_dir else {}

    corpus = env_path("VI_CORPUS")
    purity_path = Path(s.get("purity", "purity.jsonl"))
    if not purity_path.is_absolute():
        purity_path = corpus / purity_path if corpus else None
    records = read_purity(purity_path)
    log(f"classes: {len(records)} purity records" + ("" if records else " (no review yet)"))

    rows = read_jsonl(cluster_dir / "clusters.jsonl")
    ids = (cluster_dir / "components.ids").read_text().split()
    logp = np.load(cluster_dir / "components.npy").astype(np.float64)
    weights = np.asarray(cluster["gmm"]["weights"], dtype=np.float64)
    formed = {r["id"]: r.get("formed", False) for r in rows}
    clusters = {p["class"]: p for p in cluster["classes"]}
    available = {c: sum(1 for r in rows if r["cluster"] == c and r.get("formed")) for c in clusters}
    per, overall = purity_summary(records, available)
    decided = decisions(records)
    classes, rejected, problems = resolve(clusters, decided)
    resp, score = memberships(logp, weights, classes)
    keys = [c["key"] for c in classes]
    is_formed = np.array([formed.get(i, False) for i in ids])
    top = resp.argmax(axis=1) if classes else np.zeros(len(ids), dtype=int)
    # A clip belongs to its top class only if that class holds most of it: a
    # clip of a rejected cluster sits weakly in everything, and the least bad
    # kept class isn't its class.
    min_membership = float(s.get("min_membership", 0.5))
    assigned = (resp[np.arange(len(ids)), top] >= min_membership) if classes else np.zeros(len(ids), dtype=bool)

    min_purity = float(s.get("min_purity", 0.6))
    keep_jaccard = float(cluster["settings"].get("keep_jaccard", 0.75))
    failed = list(problems)
    for j, c in enumerate(classes):
        c["size"] = int(((top == j) & assigned & is_formed).sum())
        c["size_all_clips"] = int(((top == j) & assigned).sum())
        audits = [clusters[m] for m in c["clusters"]]
        if len(c["clusters"]) == 1 and c["status"] != "split_part":
            a = audits[0]
            c["stability"] = {"jaccard": a.get("jaccard"), "band": a.get("stability")}
        else:
            c["stability"] = {"jaccard": None, "band": "not re-measured",
                              "clusters": {m: clusters[m].get("jaccard") for m in c["clusters"]}}  # fmt: skip
        heard = [per[m] for m in c["clusters"]]
        yes = sum(h["members_yes"] for h in heard)
        n = sum(h["members_yes"] + h["members_no"] for h in heard)
        observed = yes / n if n else None
        f = overall["false_yes"]
        corrected = (min(1.0, max(0.0, (observed - f) / (1 - f))) if observed is not None and f is not None and f < 1
                     else observed)  # fmt: skip
        c["purity"] = {"members_heard": n, "observed": observed, "purity": corrected,
                       "sprt": [h["sprt"] for h in heard if h["sprt"]]}  # fmt: skip
        c["flags"] = sorted({flag for a in audits for flag, on in [
            ("session", a.get("session_flag")), ("topic_not_tone", a.get("topic_not_tone")),
            ("below_min_size", a.get("below_min_size")), ("nuisance", a.get("rejected_for_nuisance")),
        ] if on})  # fmt: skip
        label = f"{c['key']}" + (f" ({c['name']})" if c["name"] else "")
        if c["status"] == "unreviewed":
            failed.append(f"{label}: not reviewed")
        if c["status"] == "split_part":
            failed.append(f"{label}: split from {c['clusters'][0]}, needs naming in a new review")
        if c["status"] == "split_single_component":
            failed.append(f"{label}: split asked for, but it is one GMM component; it stays whole")
        if corrected is not None and corrected < min_purity and c["status"] == "named":
            failed.append(f"{label}: purity {corrected:.2f} below {min_purity}, named anyway (Brandon's call)")
        jac = c["stability"]["jaccard"]
        if jac is not None and jac < keep_jaccard:
            failed.append(f"{label}: Jaccard {jac:.2f} below {keep_jaccard}")
        for flag in c["flags"]:
            failed.append(f"{label}: {flag.replace('_', ' ')} flag from cluster")
    kept = sum(c["status"] in ("named", "split_part") for c in classes)
    if kept < 2:
        failed.append(f"{kept} named class(es); at least 2 are needed")
    if overall["repeat_pairs"] < 50 and records:
        failed.append(f"consistency rests on {overall['repeat_pairs']} repeats, not 50")

    write_jsonl(out / "memberships.jsonl", [
        {"id": cid, "class": keys[top[i]] if assigned[i] else None,
         "responsibilities": {k: round(float(resp[i, j]), 4) for j, k in enumerate(keys)},
         "scores": {k: round(float(score[i, j]), 3) for j, k in enumerate(keys)},
         "formed": bool(is_formed[i])}
        for i, cid in enumerate(ids)])  # fmt: skip

    # The EIV backup labelling: every head, raw and as a z-score over the
    # corpus. Kept apart from the classes; never a fallback by itself.
    embed_dir = env_path("VI_INPUT_EMBED")
    eiv_rows = read_jsonl(embed_dir / "scores.jsonl") if embed_dir and (embed_dir / "scores.jsonl").exists() else []
    heads = sorted({h for r in eiv_rows for h in (r.get("eiv") or {})})
    stats = {}
    for h in heads:
        v = np.array([r["eiv"][h] for r in eiv_rows if h in (r.get("eiv") or {})], dtype=np.float64)
        stats[h] = (float(v.mean()), float(v.std()) or 1.0)
    eiv = []
    for r in eiv_rows:
        z = {h: round((x - stats[h][0]) / stats[h][1], 3) for h, x in (r.get("eiv") or {}).items()}
        eiv.append({"id": r["id"], "eiv": r.get("eiv") or {}, "eiv_z": z,
                    "quality": r.get("eiv_quality") or {},
                    "top": sorted(z, key=lambda h: -z[h])[:3]})  # fmt: skip
    write_jsonl(out / "eiv_labels.jsonl", eiv)

    reviewed = all(c["status"] in ("named", "split_part") for c in classes) and not problems
    s["status"] = "ok" if reviewed and kept >= 2 else "incomplete"
    result = {
        "status": s["status"],
        "classes": classes,
        "rejected": rejected,
        "unassigned": int((~assigned).sum()),
        "purity": {"clusters": per, **overall},
        "failed": failed,
        "cluster_candidate": cluster.get("candidate"),
        "settings": {k: v for k, v in s.items() if k not in ("status",)},
    }
    (out / "classes.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    (out / "report.md").write_text(report_md(confounds, cluster, classes, rejected, overall, failed, s, int((~assigned).sum())))
    log(f"classes: {kept} named, {len(rejected)} rejected, {len(failed)} check(s) failed; status {s['status']}")
    checks.finish(out, [
        checks.Check("named classes", kept, 2, kind="min", note="need at least 2"),
        checks.Check("failed checks in report.md", len(failed), 0),
    ], s.get("checks", "warn"))  # fmt: skip


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
