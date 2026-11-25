"""Run the jobs/ unittest suite in parallel, or only the tests a change touches.

A faster path for iterating, standard library only. It runs exactly the tests that
`python3 -m unittest discover -s jobs` runs, split across worker processes. The serial
command stays the one CI runs and the one to pass before a merge.

    python3 jobs/run_tests.py                  # the whole suite on up to 4 workers
    python3 jobs/run_tests.py -j 2             # two workers
    python3 jobs/run_tests.py --changed        # tests affected by changes against origin/main
    python3 jobs/run_tests.py --changed HEAD~3 # ... or against another ref
    python3 jobs/run_tests.py --changed --list # show what --changed would run
    python3 jobs/run_tests.py test_suit_fit test_paid_run.Base   # named modules, classes or tests

Each worker runs a share of the tests in one `python -m unittest`-style process. A class
(or module) with setUpClass/setUpModule stays in one worker; other tests are spread one by
one. Shares are balanced on the durations of the last run, kept in
$XDG_CACHE_HOME/vi-assistant/test-durations.json. Worker logs go to a temporary directory
that is printed at the end and kept when anything failed.

--changed maps changed files to test modules: a changed jobs/test_*.py runs itself, and a
changed jobs/ file runs every test module that names it, directly or through other jobs/
files that name it (suit_fit.py -> test_suit_fit.py; filter_probe.py -> test_filter_probe.py,
and test_suit_fit.py because suit_fit.py imports it). A change to CI, requirements or a file
nothing names falls back to the whole suite; docs and Rust changes run nothing here.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
REPO = JOBS.parent
CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "vi-assistant" / "test-durations.json"
# Changes here can affect any test: run everything.
WHOLE_SUITE = (".github/", "jobs/requirements-", "jobs/run_tests.py")
# Changes here run no Python test.
NO_PYTHON = ("docs/", "crates/", "Cargo.toml", "Cargo.lock", "README.md", "characters/", "models.toml")


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def discover():
    """Every test the serial discover command runs, in its order."""
    return list(flatten(unittest.defaultTestLoader.discover(str(JOBS), top_level_dir=str(JOBS))))


def unit_of(test):
    """Tests that must share a worker: a class with class fixtures, a module with module
    fixtures; anything else stands alone. Import failures (_FailedTest) stand alone too."""
    cls = type(test)
    module = sys.modules.get(cls.__module__)
    if module is not None and (hasattr(module, "setUpModule") or hasattr(module, "tearDownModule")):
        return cls.__module__
    if "setUpClass" in vars(cls) or "tearDownClass" in vars(cls):
        return f"{cls.__module__}.{cls.__qualname__}"
    return test.id()


def load_durations():
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def save_durations(new):
    old = load_durations()
    old.update(new)
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(old, indent=0, sort_keys=True))
        tmp.replace(CACHE)
    except OSError:
        pass


def plan(tests, workers, durations):
    """Longest-first greedy split of the units over the workers; each share keeps the
    discover order inside it so class and module fixtures run once."""
    units = {}
    for i, t in enumerate(tests):
        units.setdefault(unit_of(t), []).append((i, t.id()))
    cost = {u: sum(durations.get(tid, 1.0) for _, tid in items) for u, items in units.items()}
    shares = [[0.0, []] for _ in range(max(1, workers))]
    for u in sorted(units, key=lambda u: -cost[u]):
        share = min(shares, key=lambda s: s[0])
        share[0] += cost[u]
        share[1].extend(units[u])
    return [(est, [tid for _, tid in sorted(items)]) for est, items in shares if items]


# -- worker side -------------------------------------------------------------------------
def child(ids_file, out_file):
    """Run the ids in ids_file the way discover would, write a JSON summary to out_file."""
    sys.path.insert(0, str(JOBS))
    os.chdir(REPO)
    ids = Path(ids_file).read_text().split()
    suite = unittest.defaultTestLoader.loadTestsFromNames(ids)
    runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=2, durations=0)
    result = runner.run(suite)
    summary = {
        "run": result.testsRun,
        "skipped": len(result.skipped),
        "failures": [(t.id(), tb) for t, tb in result.failures],
        "errors": [(t.id() if hasattr(t, "id") else str(t), tb) for t, tb in result.errors],
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": [t.id() for t in result.unexpectedSuccesses],
        "durations": {name: secs for name, secs in getattr(result, "collectedDurations", [])},
    }
    Path(out_file).write_text(json.dumps(summary))
    return 0 if result.wasSuccessful() else 1


# -- --changed ---------------------------------------------------------------------------
def changed_files(ref):
    def git(*args):
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"run_tests.py: git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.split("\n")

    base = git("merge-base", ref, "HEAD")[0].strip()
    files = git("diff", "--name-only", base) + git("ls-files", "--others", "--exclude-standard")
    return sorted({f for f in files if f})


def affected(files):
    """(test module names, reason) for the changed files, or (None, reason) for all."""
    sources = {}
    for p in sorted(JOBS.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".toml", ".txt", "") \
                and "__pycache__" not in p.parts and p.stat().st_size < 2_000_000:
            try:
                sources[p] = p.read_text(errors="ignore")
            except OSError:
                pass
    tests = {p.stem for p in JOBS.glob("test*.py")}

    def mention(path):
        rel = path.relative_to(JOBS)
        return [re.compile(r"(?<![\w.-])" + re.escape(n) + r"(?![\w-])") for n in {rel.name, str(rel)}]

    def loads(path):
        """How code pulls this file in: a Python import, or a shell `source` / `.`."""
        if path.suffix == ".py":
            stem = re.escape(path.stem)
            return re.compile(rf"^\s*(from\s+{stem}\s+import|import\s+([\w.]+\s*,\s*)*{stem}\b)", re.M)
        return re.compile(r"^\s*(source|\.)\s+\S*" + re.escape(path.name) + r"\b", re.M)

    def tests_of(path):
        """Test modules that name path, or name code that imports or sources it (and
        so on). Other mentions, mostly comments between the shell scripts, do not
        chain: following them reaches nearly every test."""
        todo, seen, hits = [path], set(), set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            direct, load = mention(p), loads(p)
            for q, text in sources.items():
                if q == p:
                    continue
                if q.parent == JOBS and q.stem in tests:
                    if any(m.search(text) for m in direct) or load.search(text):
                        hits.add(q.stem)
                elif q.suffix in (".py", ".sh") and (load.search(text) or (
                        p == path and p.suffix not in (".py", ".sh") and any(m.search(text) for m in direct))):
                    todo.append(q)  # code that loads it; for a data file, code that names it
        return hits

    run, reasons = set(), []
    for f in files:
        if f.startswith(WHOLE_SUITE):
            return None, f"{f} can affect every test"
        if f.startswith(NO_PYTHON) or not f.startswith("jobs/"):
            reasons.append(f"{f}: no Python test")
            continue
        path = REPO / f
        if path.stem in tests and path.parent == JOBS:
            run.add(path.stem)
        if not path.exists():
            continue  # deleted: whatever named it changed too, or fails at import
        hits = tests_of(path)
        if not hits and path.stem not in tests:
            return None, f"{f}: nothing names it, so run everything"
        run |= hits
    return sorted(run), "; ".join(reasons)


# -- parent side -------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="test modules, classes or tests (default: all)")
    ap.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--changed", nargs="?", const="origin/main", metavar="REF",
                    help="only tests affected by changes since REF (default origin/main)")
    ap.add_argument("--list", action="store_true", help="print the tests and the split, run nothing")
    ap.add_argument("--child", nargs=2, metavar=("IDS", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(*a.child)

    sys.path.insert(0, str(JOBS))
    os.chdir(REPO)
    if a.changed:
        files = changed_files(a.changed)
        mods, why = affected(files)
        print(f"changed since {a.changed}: {len(files)} file(s)" + (f" ({why})" if why else ""))
        if mods is None:
            names = []
        elif not mods:
            print("no Python tests are affected")
            return 0
        else:
            names = mods
            print("affected test modules: " + " ".join(mods))
    else:
        names = a.names
    loader = unittest.defaultTestLoader
    tests = list(flatten(loader.loadTestsFromNames(names))) if names else discover()
    durations = load_durations()
    shares = plan(tests, a.jobs, durations)
    if a.list:
        for i, (est, ids) in enumerate(shares):
            print(f"worker {i}: {len(ids)} tests, about {est:.0f} s")
            for tid in ids:
                print(f"  {tid}")
        return 0

    logdir = Path(tempfile.mkdtemp(prefix="vi-tests-"))
    procs = []
    t0 = time.monotonic()
    for i, (est, ids) in enumerate(shares):
        (logdir / f"w{i}.ids").write_text("\n".join(ids) + "\n")
        log = open(logdir / f"w{i}.log", "w")
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child",
                              str(logdir / f"w{i}.ids"), str(logdir / f"w{i}.json")],
                             stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        procs.append((i, p, log, est, len(ids)))
    print(f"{len(tests)} tests on {len(procs)} worker(s); logs in {logdir}", flush=True)

    total = {"run": 0, "skipped": 0, "expected_failures": 0}
    failed, new_durations = [], {}
    for i, p, log, est, n in procs:
        rc = p.wait()
        log.close()
        took = time.monotonic() - t0
        try:
            s = json.loads((logdir / f"w{i}.json").read_text())
        except (OSError, ValueError):
            failed.append((f"worker {i}", f"exited {rc} without a summary; see {logdir}/w{i}.log"))
            print(f"worker {i}: CRASHED (exit {rc}) after {took:.0f} s", flush=True)
            continue
        for k in total:
            total[k] += s[k]
        failed += [(tid, tb) for tid, tb in s["failures"] + s["errors"]]
        failed += [(tid, "unexpected success") for tid in s["unexpected_successes"]]
        # collectedDurations names tests by str(test); map back to ids for the next plan.
        by_str = {str(t): t.id() for t in tests}
        new_durations.update({by_str.get(k, k): v for k, v in s["durations"].items()})
        print(f"worker {i}: {s['run']}/{n} tests, done at {took:.0f} s (planned about {est:.0f} s)", flush=True)
    wall = time.monotonic() - t0
    if new_durations:
        save_durations(new_durations)
    for tid, tb in failed:
        print(f"\n{'=' * 70}\nFAIL: {tid}\n{'-' * 70}\n{tb}")
    print(f"\nRan {total['run']} tests in {wall:.1f} s on {len(procs)} worker(s) "
          f"(skipped={total['skipped']}, expected failures={total['expected_failures']})")
    if total["run"] != len(tests):
        failed.append(("count", f"{len(tests)} tests were planned but {total['run']} ran"))
        print(f"MISMATCH: {len(tests)} tests were planned but {total['run']} ran")
    if failed:
        print(f"FAILED ({len(failed)}); logs kept in {logdir}")
        return 1
    print("OK")
    for f in logdir.iterdir():
        f.unlink()
    logdir.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
