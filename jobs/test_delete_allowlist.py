"""Lint: every delete call in production code is on the reviewed allowlist.

R47 makes `~/.cache/vi/checkpoints` deletable only by `prune`, and only on a
VERIFIED run; `publish_epoch` also refuses a publish target inside that store.
This test keeps the rule structural: it finds every delete call in production
`jobs/` and `crates/*/src/` code and requires a reviewed allowlist entry naming
the file, the call and the target root it may remove.

Adding a delete call means adding an entry in the same change, after checking
that its target root is safe. An unlisted call fails here. Test code
(`jobs/test_*.py`, Rust `#[cfg(test)]` modules and `crates/*/tests/`) is out of
scope: it deletes only temp dirs it created.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "jobs"
CRATES = ROOT / "crates"

# (file, call pattern, target root the call may remove)
ALLOWLIST = [
    ("jobs/check-worker.sh", r'rm -rf "\$tmp"', "the mktemp -d scratch this check created"),
    ("jobs/checkpoints.py", r"shutil\.rmtree\(part\)", "a publish dir's previous-attempt .part"),
    ("jobs/checkpoints.py", r"shutil\.rmtree\(dst\)", "a publish dir's epoch being re-published"),
    ("jobs/checkpoints.py", r"p\.unlink\(\)", "the state dir's unverified marker for this exact run"),
    ("jobs/checkpoints.py", r"shutil\.rmtree\(p\)", "prune: an epoch dir inside a VERIFIED run dir"),
    ("jobs/fx_detect.py", r"raw\.unlink\(\)", "a raw model download under --weights"),
    ("jobs/on-worker.sh", r'rm -f "\$f\.part"', "a killed pack's debris in the bundle cache ($VI_BUNDLE_DIR)"),
    ("jobs/on-worker.sh", r'rm -f "\$path"', "the bundle cache ($VI_BUNDLE_DIR)"),
    ("jobs/on-worker.sh", r'rm -f "\$VI_OUT/\.return-token"', "the stage output dir"),
    ("jobs/on-worker.sh", r'rm -f "\$return_state/tunnel\.pid"', "the attempt's return state dir (its tunnel pid file only)"),
    ("jobs/on-worker.sh", r'rm -rf "\$return_state"', "the attempt's return state dir under the vi state dir"),
    ("jobs/on-worker.sh", r'rm -rf "\$return_scratch"', "the mktemp -d scratch this run created for the checkpoint module"),
    ("jobs/on-worker.sh", r'rm -f \$\(q "\$work/push\.pid"\)', "the worker job dir's push-loop pid file"),
    ("jobs/on-worker.sh", r'rm -rf \$\(q "\$work"\)', "the worker's job dir (/root/vi-jobs)"),
    ("jobs/on-worker.sh", r'rm \$\(q "\$work/bundles/', "the worker job's unpacked bundle copy"),
    ("jobs/refscreen.py", r"shutil\.rmtree\(root, ignore_errors=True\)", "the run's own out/shortlist dir"),
    ("jobs/voicearms.py", r"shutil\.rmtree\(root, ignore_errors=True\)", "the run's own out/listen dir"),
    ("jobs/voicetune_data.py", r"\.unlink\(missing_ok=True\)", "out/: the raw manifest replaced by the coded one"),
    ("jobs/voicetune-fresh.sh", r'rm -rf "\$probe_run/voice-epoch-0"', "the checkpoint run dir's voice-model copy"),
    ("jobs/voicetune-fresh.sh", r'rm -rf "\$run/voice-epoch-\$epoch"', "the checkpoint run dir's voice-model copy"),
    ("jobs/voicetune-fresh.sh", r'rm -f "\$work/repush/epoch-\$1"', "the worker job dir's repush marker and fail counter"),
    ("jobs/voicetune-fresh.sh", r'rm -f "\$rp" "\$f"', "the worker job dir's repush marker and fail counter"),
    ("jobs/voicetune-fresh.sh", r'rm -f "\$work/push\.pid"', "the worker job dir's push-loop pid file"),
    ("jobs/worker-job.sh", r'rm -f "\$p"', "the worker job dir's pending marker once its epoch is acked"),
    ("jobs/worker-return.sh", r'rm -f "\$state/rsyncd\.conf"', "the mktemp state dir holding the rsync daemon files"),
    ("jobs/worker-return.sh", r'rm -rf "\$tmp"', "the mktemp -d scratch the receiver_verify token push created"),
    ("jobs/worker-return.sh", r'rm -f "\$path/\.restart-token"', "the receiver module dirs: only the .restart-token this receiver itself pushed"),
    ("jobs/worker-return.sh", r'rm -f "\$state/rsyncd\.pid"', "the state dir's dead-receiver pid and per-module lock files on restart"),
    ("crates/cloud/src/state.rs", r"std::fs::remove_file\(path\)", "the vi state dir (markers and known_hosts)"),
    ("crates/review/src/hub.rs", r"fs::remove_file\(&path\)", "a task's TASK_DONE file under the hub dir"),
    ("crates/harvest/src/extract.rs", r"fs::remove_dir_all\(&tmp\)", "the extraction temp dir this run created"),
    ("crates/dataset/src/runner.rs", r"fs::remove_dir_all\(&tmp\)", "a stage's .tmp- partial dir under corpus/stages"),
    ("crates/dataset/src/runner.rs", r"fs::remove_dir_all\(&dest\)", "a stage's old run dir, replaced by rename"),
    ("crates/dataset/src/runner.rs", r"fs::remove_dir_all\(&checkpoint\)", "a stage's .checkpoint- scratch dir"),
    ("crates/dataset/src/runner.rs", r"fs::remove_file\(path\)", "an output file replaced by a link to the object store"),
    ("crates/dataset/src/runner.rs", r"fs::remove_dir_all\(entry\.path\(\)\)", "gc: a non-current run dir under corpus/stages"),
    ("crates/dataset/src/runner.rs", r"fs::remove_file\(entry\.path\(\)\)", "gc: an unreferenced object under corpus/objects"),
    ("crates/llm/src/selftest.rs", r"std::fs::remove_dir_all\(&store_dir\)", "the selftest's own scratch store dir"),
    ("crates/llm/src/capture.rs", r"fs::remove_dir_all\(&store\)", "the fit-check scratch store dir passed to reuse_fit_check"),
    ("crates/llm/src/fit.rs", r"fs::remove_file\(&p\)", "generated *.gguf vectors in the method output dir"),
]

PY_DELETE = re.compile(r"\.unlink\(|shutil\.rmtree\(|os\.remove\(")
SH_DELETE = re.compile(r"""(?:^|[\s;&|("'])rm\s+(?:-|\$|")""", re.MULTILINE)
RS_DELETE = re.compile(r"\b(?:remove_dir_all|remove_file|remove_dir)\s*\(")


def production_files():
    """Every production source file the lint covers."""
    for p in sorted(JOBS.glob("*.py")):
        if not p.name.startswith("test_"):
            yield p
    yield from sorted(JOBS.glob("*.sh"))
    yield from sorted(CRATES.glob("*/src/**/*.rs"))


def scan(paths):
    """[(relative path, line number, line)] for every delete call in `paths`."""
    calls = []
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        lines = list(enumerate(p.read_text(encoding="utf-8").splitlines(), 1))
        if p.suffix == ".rs":
            # Production code ends at the file's #[cfg(test)] module.
            for i, (_, line) in enumerate(lines):
                if line.strip() == "#[cfg(test)]":
                    lines = lines[:i]
                    break
            pattern = RS_DELETE
        elif p.suffix == ".py":
            pattern = PY_DELETE
        else:
            pattern = SH_DELETE
        for n, line in lines:
            if pattern.search(line):
                calls.append((rel, n, line))
    return calls


def unallowlisted(calls):
    """The calls with no allowlist entry for their file and call text."""
    bad = []
    for rel, n, line in calls:
        if not any(rel == f and re.search(pat, line) for f, pat, _ in ALLOWLIST):
            bad.append(f"{rel}:{n}: {line.strip()}")
    return bad


class DeleteAllowlistTest(unittest.TestCase):
    def test_delete_calls_match_allowlist(self):
        bad = unallowlisted(scan(production_files()))
        self.assertEqual(bad, [], "unreviewed delete calls:\n" + "\n".join(bad))

    def test_a_call_outside_the_allowlist_fails(self):
        # Prove the negative case: a fixture call not on the list is reported.
        fixture = JOBS / "_delete_lint_fixture.py"
        try:
            fixture.write_text("import shutil\nshutil.rmtree('/somewhere')\n")
            bad = unallowlisted(scan([fixture]))
            self.assertEqual(len(bad), 1, bad)
        finally:
            fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
