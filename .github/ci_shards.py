"""Split the jobs/ unittest suite across CI runners and prove nothing was dropped.

Standard library only. Run from the repo root.

    python3 .github/ci_shards.py run SHARD    # run one shard's modules, one after another
    python3 .github/ci_shards.py check        # every jobs/test_*.py is in exactly one shard,
                                              # and the shards add up to EXPECTED_TESTS tests

Each shard runs its modules serially in one process, the same way
`python3 -m unittest discover -s jobs` does, so the timing-sensitive bash supervisor
tests never compete with another test for CPU. A shard fails if it runs a different
number of tests than it loaded, or a different number of skips than SKIPPED says.

To rebalance, move module names between the lists below. Durations are seconds of the
suite on a developer machine after the Part B speed-up (jobs/run_tests.py caches them).
"""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "jobs"

EXPECTED_TESTS = 864

SHARDS = {
    # 283 s on a CI runner
    "1": ["test_worker_return", "test_eq_check", "test_vast_limits", "test_steer"],
    # about 210 s (test_host_qual adds about 55 s)
    "2": ["test_omnivoice_preflight", "test_fx_detect", "test_confounds",
          "test_venv_build", "test_filter_probe", "test_prep", "test_eq_match",
          "test_textscore", "test_rehearsal_push", "test_refscreen", "test_host_qual"],
    # about 175 s
    "3": ["test_paid_run", "test_ttstest", "test_suit_fit", "test_classes", "test_worker_preflight_cuda",
          "test_checkpoints", "test_codectrip", "test_embed", "test_voicesteer",
          "test_decide", "test_remap_purity", "test_omnivoice_data", "test_replay_public"],
    # 130 s
    "4": ["test_debug_pass", "test_validity", "test_voicearms", "test_cluster", "test_worker_limits",
          "test_omnivoice_supervision", "test_calibrate", "test_delete_allowlist",
          "test_fusion", "test_progress", "test_ft_guards", "test_voicetune",
          "test_bundle_ids", "test_buildset", "test_checks", "test_qtcheck", "test_progress_counters",
          "test_deadman_heartbeat",
          "test_clean", "test_label", "test_push_probe", "test_bg_watch", "test_parallel_push",
          "test_push_progress"],
}

# Skips each shard reports on a GitHub ubuntu-latest runner (23 in total; the 21 seen on a
# developer machine differ by two environment-dependent skips).
SKIPPED = {"1": 0, "2": 0, "3": 0, "4": 23}


def on_disk():
    return sorted(p.stem for p in JOBS.glob("test_*.py"))


def assigned():
    return [m for mods in SHARDS.values() for m in mods]


def load(modules):
    sys.path.insert(0, str(JOBS))
    return unittest.defaultTestLoader.loadTestsFromNames(modules)


def check():
    names, disk = assigned(), on_disk()
    problems = []
    problems += [f"{m} is assigned to more than one shard" for m in sorted(set(names)) if names.count(m) > 1]
    problems += [f"{m} is not assigned to any shard" for m in disk if m not in names]
    problems += [f"{m} is assigned but jobs/{m}.py does not exist" for m in sorted(set(names)) if m not in disk]
    counts = {s: load(mods).countTestCases() for s, mods in SHARDS.items()}
    total = sum(counts.values())
    discovered = unittest.defaultTestLoader.discover(str(JOBS), top_level_dir=str(JOBS)).countTestCases()
    for s, n in counts.items():
        print(f"shard {s}: {len(SHARDS[s])} modules, {n} tests")
    print(f"shards total {total}, discover total {discovered}, expected {EXPECTED_TESTS}")
    if total != discovered:
        problems.append(f"shards hold {total} tests but discover finds {discovered}")
    if total != EXPECTED_TESTS:
        problems.append(f"shards hold {total} tests, expected {EXPECTED_TESTS} (update EXPECTED_TESTS if tests were added on purpose)")
    for p in problems:
        print(f"FAIL: {p}")
    return 1 if problems else 0


def run(shard):
    suite = load(SHARDS[shard])
    loaded = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    skipped = len(result.skipped)
    print(f"SHARD_RESULT shard={shard} modules={len(SHARDS[shard])} loaded={loaded} "
          f"ran={result.testsRun} skipped={skipped} failures={len(result.failures)} errors={len(result.errors)}")
    ok = result.wasSuccessful() and result.testsRun == loaded
    if result.testsRun != loaded:
        print(f"FAIL: shard {shard} loaded {loaded} tests but ran {result.testsRun}")
    expected_skips = SKIPPED[shard]
    if expected_skips is not None and skipped != expected_skips:
        print(f"FAIL: shard {shard} skipped {skipped} tests, expected {expected_skips}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "check":
        sys.exit(check())
    if len(sys.argv) == 3 and sys.argv[1] == "run" and sys.argv[2] in SHARDS:
        sys.exit(run(sys.argv[2]))
    sys.exit(__doc__)
