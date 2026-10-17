"""Carries Brandon's purity decisions over to a rebuilt cluster run.

A rebuild (a rerun of clean and embed, say) can renumber the clusters, so
"c2 is urgency" would land on whatever c2 is now. classes refuses decisions
made on another run (classes.fingerprint). This matches each decided old
cluster to the new cluster sharing most of its formed clips, and only when
the two are each other's best match at Jaccard >= --min-jaccard (0.6, cluster's trust bar); anything
else stays unreviewed for a new listen. Answers aren't carried over, since
they were heard as members of the old clusters.

    python3 jobs/remap_purity.py OLD_CLUSTER_OUT NEW_CLUSTER_OUT corpus/tali/purity.jsonl
    python3 jobs/remap_purity.py ... --write     # after Brandon has read the table

Without --write it only prints the proposed mapping. With it, it appends one
decision per carried cluster, stamped with the new run and a note saying
where it came from. Records without a `run` (written before the review page
stamped them) count as the old run's, and --write stamps them so, keeping
a copy of the file first. purity.jsonl is Brandon's file: write only on his
word.

    python3 jobs/remap_purity.py --stamp CLUSTER_OUT corpus/tali/purity.jsonl

stamps the unstamped records with that run and nothing else, so they can't
land on a rebuild's clusters before they're carried over.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from classes import decisions, fingerprint, for_run, read_purity
from vi_job import read_jsonl


def members(rows):
    out = {}
    for r in rows:
        if r.get("formed"):
            out.setdefault(r["cluster"], set()).add(r["id"])
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def propose(old_rows, new_rows, records, min_jaccard=0.6):
    """[(old cluster, decision, new cluster or None, Jaccard)], the new run,
    and the old run."""
    old_run, new_run = fingerprint(old_rows), fingerprint(new_rows)
    mine, _ = for_run(records, old_run, legacy_run=old_run)
    decided = decisions(mine)
    old, new = members(old_rows), members(new_rows)
    best_old = {n: max(old, key=lambda o: jaccard(old[o], new[n])) for n in new} if old else {}
    out = []
    for c, d in sorted(decided.items()):
        if c not in old or not new:
            out.append((c, d, None, 0.0))
            continue
        n = max(new, key=lambda n: jaccard(old[c], new[n]))
        j = jaccard(old[c], new[n])
        out.append((c, d, n if j >= min_jaccard and best_old[n] == c else None, round(j, 3)))
    return out, new_run, old_run


def remapped(proposal, new_run, now=None):
    """The decision records to append. A merge whose target wasn't carried
    over isn't either."""
    to_new = {c: n for c, _, n, _ in proposal if n}
    out = []
    for c, d, n, j in proposal:
        if n is None or (d.get("action") == "merge" and d.get("into") not in to_new):
            continue
        r = {**d, "cluster": n, "run": new_run, "at": int(now or time.time()),
             "note": f"remapped from {c} (Jaccard {j:.2f})" + (f"; {d['note']}" if d.get("note") else "")}  # fmt: skip
        if d.get("action") == "merge":
            r["into"] = to_new[d["into"]]
        out.append(r)
    return out


def stamp(path, run):
    """Writes `run` into every record of purity.jsonl that has none, after
    copying the file to purity.jsonl.bak-<unix seconds>. Returns how many."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out, n = [], 0
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if "run" not in r:
            r["run"], n = run, n + 1
        out.append(json.dumps(r, ensure_ascii=False, sort_keys=True))
    if n:
        backup = path.with_name(f"{path.name}.bak-{int(time.time())}")
        shutil.copy2(path, backup)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        tmp.replace(path)
    return n


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--stamp":
        cluster, path = Path(sys.argv[2]), Path(sys.argv[3])
        run = fingerprint(read_jsonl(cluster / "clusters.jsonl"))
        print(f"stamped {stamp(path, run)} record(s) of {path} with run {run}")
        return
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old", type=Path, help="the old cluster stage's out/ folder")
    ap.add_argument("new", type=Path, help="the rebuilt cluster stage's out/ folder")
    ap.add_argument("purity", type=Path, help="corpus/<character>/purity.jsonl")
    ap.add_argument("--min-jaccard", type=float, default=0.6)
    ap.add_argument("--write", action="store_true", help="append the carried decisions to purity.jsonl")
    a = ap.parse_args()
    old_rows, new_rows = read_jsonl(a.old / "clusters.jsonl"), read_jsonl(a.new / "clusters.jsonl")
    proposal, new_run, old_run = propose(old_rows, new_rows, read_purity(a.purity), a.min_jaccard)
    print(f"old run {old_run}, new run {new_run}")
    if old_run == new_run:
        print("the clusters didn't change; nothing to carry over")
        return
    for c, d, n, j in proposal:
        what = d.get("name") or d.get("action")
        print(f"{c} ({what}) -> {n or 'not carried: hear it again'}  Jaccard {j:.2f}")
    records = remapped(proposal, new_run)
    if not a.write:
        print(f"{len(records)} decision(s) would be appended; rerun with --write on Brandon's word")
        return
    stamp(a.purity, old_run)
    with open(a.purity, "a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"appended {len(records)} decision(s) to {a.purity}", file=sys.stderr)


if __name__ == "__main__":
    main()
