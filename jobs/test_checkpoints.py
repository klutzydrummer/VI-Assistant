"""Tests for checkpoints.py (standard library only, tiny files in temp dirs).

The omnivoice format is the default everywhere; the legacy epoch format stays
tested behind fmt="epoch" / --format epoch (jobs/voicetune.py still publishes it).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import checkpoints

CLI = [sys.executable, str(Path(__file__).resolve().parent / "checkpoints.py")]

# The pipeline's checkpoint contract: one file from each required category
# (jobs/omnivoice-ft/train_ft.py), listed in SHA256SUMS. Real ones also carry
# tokenizer_config.json; one tokenizer file satisfies the category.
CKPT_FILES = {
    "config.json": b"{}",
    "model.safetensors": b"weights",
    "optimizer.bin": b"opt",
    "scheduler.bin": b"sched",
    "random_states_0.pkl": b"rng",
    "train_config.json": b"{}",
    "ft_config.json": b'{"steps": 5000, "save_steps": 500}',
    "tokenizer.json": b"{}",
}


def _epoch(run, k, files):
    """Write epoch-k under `run` with a SHA256SUMS over `files` ({name: bytes})."""
    ep = run / f"epoch-{k}"
    ep.mkdir(parents=True)
    lines = []
    for name, data in files.items():
        (ep / name).write_bytes(data)
        lines.append(f"{checkpoints.sha256_file(ep / name)}  {name}\n")
    (ep / "SHA256SUMS").write_text("".join(lines))
    return ep


def _ckpt(run, step, files=None):
    """Write checkpoint-<step> under `run` with a SHA256SUMS over the pipeline's
    required files ({name: bytes}; `files` replaces the whole default set, so
    omitting a category tests the missing-file rule)."""
    ck = run / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    chosen = dict(CKPT_FILES if files is None else files)
    lines = []
    for name, data in sorted(chosen.items()):
        (ck / name).write_bytes(data)
        lines.append(f"{checkpoints.sha256_file(ck / name)}  {name}\n")
    (ck / "SHA256SUMS").write_text("".join(lines))
    return ck


def _verified_run(root, n):
    """A run dir under `root` with `n` verified epochs (0..n-1) and a VERIFIED file."""
    run = root / "run"
    run.mkdir()
    for k in range(n):
        _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
    checkpoints.verify(run, n, fmt="epoch")
    return run


def _verified_ckpt_run(root, steps):
    """A run dir under `root` with verified checkpoints at `steps` and a VERIFIED file."""
    run = root / "run"
    run.mkdir()
    for s in steps:
        _ckpt(run, s)
    checkpoints.verify(run, len(steps))
    return run


def _epoch_names(run):
    return sorted(p.name for p in run.glob("epoch-*") if not p.name.endswith(".part"))


def _ckpt_names(run):
    return sorted(p.name for p in run.glob("checkpoint-*"))


def _snapshot(root):
    """Every entry under `root` with its bytes, for byte-for-byte comparisons."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = b"<dir>" if p.is_dir() else p.read_bytes()
    return out


class PublishTest(unittest.TestCase):
    def _src(self, root, style=True):
        run = root / "src"
        run.mkdir()
        (run / "voice.json").write_text('{"speaker": "tali"}')
        (run / "xvectors.json").write_text("{}")
        (run / "train_log.jsonl").write_text("line\n")
        ep = run / "epoch-0"
        ep.mkdir()
        (ep / "talker.safetensors").write_bytes(b"weights")
        (ep / "tune.json").write_text('{"lr": 2e-6}')
        if style:
            (ep / "style.safetensors").write_bytes(b"style")
        return run, ep

    def test_publish_epoch_is_atomic_and_hard_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run, ep = self._src(root)
            publish = root / "pub"
            checkpoints.publish_epoch(ep, publish)
            self.assertFalse((publish / "epoch-0.part").exists())
            dest = publish / "epoch-0"
            self.assertTrue((dest / "SHA256SUMS").is_file())
            self.assertTrue((dest / "talker.safetensors").is_file())
            self.assertTrue((dest / "tune.json").is_file())
            self.assertTrue((dest / "style.safetensors").is_file())
            # Same filesystem: hard link, not a copy (same inode).
            self.assertEqual(os.stat(ep / "talker.safetensors").st_ino,
                             os.stat(dest / "talker.safetensors").st_ino)
            # Run-level files copied, overwriting whatever was there.
            self.assertEqual((publish / "voice.json").read_text(), '{"speaker": "tali"}')
            self.assertEqual((publish / "train_log.jsonl").read_text(), "line\n")

    def test_publish_epoch_skips_absent_style(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run, ep = self._src(root, style=False)
            publish = root / "pub"
            checkpoints.publish_epoch(ep, publish)
            sums = (publish / "epoch-0" / "SHA256SUMS").read_text()
            self.assertNotIn("style.safetensors", sums)
            self.assertFalse((publish / "epoch-0" / "style.safetensors").exists())

    def test_publish_epoch_refuses_target_under_home_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            home = root / "home"
            publish = home / ".cache" / "vi" / "checkpoints" / "stage" / "run"
            publish.mkdir(parents=True)
            _epoch(publish, 0, {"talker.safetensors": b"old", "tune.json": b"{}"})
            (publish / "notes.txt").write_text("keep me")
            before = _snapshot(publish)
            ep = root / "src" / "epoch-1"
            ep.mkdir(parents=True)
            (ep / "talker.safetensors").write_bytes(b"new")
            (ep / "tune.json").write_bytes(b"{}")
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                with self.assertRaises(SystemExit):
                    checkpoints.publish_epoch(ep, publish)
            # The guard fires before any mkdir or rmtree: nothing changed.
            self.assertEqual(_snapshot(publish), before)

    def test_publish_epoch_raises_on_missing_required_file(self):
        for missing in ("talker.safetensors", "tune.json"):
            with self.subTest(missing=missing):
                with tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    run, ep = self._src(root)
                    (ep / missing).unlink()
                    publish = root / "pub"
                    with self.assertRaises(FileNotFoundError):
                        checkpoints.publish_epoch(ep, publish)
                    self.assertFalse((publish / "epoch-0").exists())
                    self.assertFalse((publish / "epoch-0.part").exists())


class VerifyTest(unittest.TestCase):
    """The omnivoice format, which is the default."""

    def test_verify_passes_a_good_run(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            _ckpt(run, 1000)
            out = checkpoints.verify(run, 2)
            self.assertEqual(set(out), {"checkpoint-500", "checkpoint-1000"})
            v = (run / "VERIFIED").read_text()
            self.assertIn('"checkpoints"', v)
            self.assertIn(checkpoints.sha256_file(run / "checkpoint-500" / "model.safetensors"), v)

    def test_verify_fails_on_kill_during_save_no_manifest(self):
        # The pipeline writes SHA256SUMS last: a kill mid-save leaves the files but
        # no manifest, and that dir must never count as a checkpoint.
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            ck = _ckpt(run, 500)
            (ck / "SHA256SUMS").unlink()
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)
            self.assertFalse((run / "VERIFIED").exists())

    def test_verify_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            (run / "checkpoint-500" / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            (run / "checkpoint-500" / "model.safetensors").write_bytes(b"")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_when_a_required_category_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            files = dict(CKPT_FILES)
            del files["optimizer.bin"]
            _ckpt(run, 500, files)
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_unlisted_extra_file(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            (run / "checkpoint-500" / "sneaky.bin").write_bytes(b"x")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_self_listed_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            ck = run / "checkpoint-500"
            sums = ck / "SHA256SUMS"
            sums.write_text(sums.read_text() + f"{checkpoints.sha256_file(sums)}  {sums.name}\n")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_when_expect_is_wrong(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            _ckpt(run, 1000)
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 3)

    def test_verify_fails_on_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outside = root / "outside.txt"
            outside.write_text("not a checkpoint")
            run = root / "run"
            _ckpt(run, 500)
            (run / "checkpoint-500" / "model.safetensors").unlink()
            (run / "checkpoint-500" / "model.safetensors").symlink_to(outside)
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_leftover_part_dir(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            (run / "checkpoint-1000.part").mkdir()
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_parent_manifest_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"
            _ckpt(run, 500)
            ck = run / "checkpoint-500"
            (root / "x").write_text("outside")
            (ck / "SHA256SUMS").write_text(
                (ck / "SHA256SUMS").read_text()
                + f"{checkpoints.sha256_file(root / 'x')}  ../x\n")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_cli_verify_defaults_to_omnivoice(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            ok = subprocess.run(CLI + ["verify", str(run), "--expect", "1"],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("1 checkpoints ok", ok.stdout)


class EpochModeRegressionTest(unittest.TestCase):
    """The legacy epoch format still works behind fmt='epoch' / --format epoch."""

    def test_verify_epoch_contiguity(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            _epoch(run, 1, {"talker.safetensors": b"c", "tune.json": b"d"})
            out = checkpoints.verify(run, 2, fmt="epoch")
            self.assertEqual(set(out), {"epoch-0", "epoch-1"})
            # A gap fails.
            (run / "epoch-2").mkdir()
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 3, fmt="epoch")

    def test_verify_epoch_fails_on_leftover_part(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            (run / "epoch-1.part").mkdir()
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1, fmt="epoch")

    def test_cli_epoch_mode_verify_and_verify_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            ok = subprocess.run(CLI + ["--format", "epoch", "verify", str(run), "--expect", "1"],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("1 epochs ok", ok.stdout)
            one = subprocess.run(CLI + ["--format", "epoch", "verify-epoch", str(run), "0"],
                                 capture_output=True, text=True)
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertIn("epoch-0: ok (2 files)", one.stdout)

    def test_epoch_prune_requires_keep(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            bad = subprocess.run(CLI + ["--format", "epoch", "prune", str(run)],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("--keep is required", bad.stderr)
            checkpoints.prune(run, 1, yes=True, fmt="epoch")
            self.assertEqual(_epoch_names(run), ["epoch-1", "epoch-2"])


class VerifyEpochTest(unittest.TestCase):
    """The home supervisor checks one checkpoint at a time as it lands."""

    def test_verify_epoch_checks_one_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            _ckpt(run, 1000)
            # Corrupt checkpoint 1000 without changing its size; 500 must still pass.
            (run / "checkpoint-1000" / "model.safetensors").write_bytes(b"TAMPERED!")
            before = _snapshot(run)
            ok = subprocess.run(CLI + ["verify-epoch", str(run), "500"],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("checkpoint-500: ok (8 files)", ok.stdout)
            # A single checkpoint writes no run-level VERIFIED: a run is verified whole.
            self.assertFalse((run / "VERIFIED").exists())

            bad = subprocess.run(CLI + ["verify-epoch", str(run), "1000"],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("sha256 mismatch", bad.stderr)
            self.assertFalse((run / "VERIFIED").exists())
            # Checking a checkpoint never changes a file, marker, or directory.
            self.assertEqual(_snapshot(run), before)

            missing = subprocess.run(CLI + ["verify-epoch", str(run), "2000"],
                                     capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("is not a directory", missing.stderr)


class GateTest(unittest.TestCase):
    def test_gate_at_49_and_51_gb(self):
        with mock.patch("checkpoints.os.statvfs") as sv:
            sv.return_value = mock.Mock(f_bavail=51 * 2**18, f_frsize=4096)
            self.assertTrue(checkpoints.gate(Path("/run"), 50))
            sv.return_value = mock.Mock(f_bavail=49 * 2**18, f_frsize=4096)
            self.assertFalse(checkpoints.gate(Path("/run"), 50))

    def test_gate_reserve_refuses_when_projected_checkpoints_leave_too_little(self):
        with mock.patch("checkpoints.os.statvfs") as sv:
            # 60 free, 27 reserved, 50 needed: 33 left < 50 -> refuse.
            sv.return_value = mock.Mock(f_bavail=60 * 2**18, f_frsize=4096)
            self.assertFalse(checkpoints.gate(Path("/run"), 50, 27))
            # 80 free: 53 left >= 50 -> pass.
            sv.return_value = mock.Mock(f_bavail=80 * 2**18, f_frsize=4096)
            self.assertTrue(checkpoints.gate(Path("/run"), 50, 27))

    def test_gate_reserve_zero_keeps_current_behavior(self):
        with mock.patch("checkpoints.os.statvfs") as sv:
            sv.return_value = mock.Mock(f_bavail=51 * 2**18, f_frsize=4096)
            self.assertTrue(checkpoints.gate(Path("/run"), 50, 0))
            sv.return_value = mock.Mock(f_bavail=49 * 2**18, f_frsize=4096)
            self.assertFalse(checkpoints.gate(Path("/run"), 50, 0))

    def test_gate_counts_retention_gb(self):
        # 65 free, 15 retained at home, 50 needed: 50 >= 50 -> pass; 64 free -> refuse.
        with mock.patch("checkpoints.os.statvfs") as sv:
            sv.return_value = mock.Mock(f_bavail=65 * 2**18, f_frsize=4096)
            self.assertTrue(checkpoints.gate(Path("/run"), 50, 0, 15))
            sv.return_value = mock.Mock(f_bavail=64 * 2**18, f_frsize=4096)
            self.assertFalse(checkpoints.gate(Path("/run"), 50, 0, 15))


class PruneTest(unittest.TestCase):
    def test_prune_first_epoch_keeps_zero_and_one(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            checkpoints.prune(run, 0, yes=True, fmt="epoch")
            self.assertEqual(_epoch_names(run), ["epoch-0", "epoch-1"])

    def test_prune_middle_epoch_keeps_one_and_two(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            checkpoints.prune(run, 1, yes=True, fmt="epoch")
            self.assertEqual(_epoch_names(run), ["epoch-1", "epoch-2"])

    def test_prune_last_epoch_keeps_two_and_one(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            checkpoints.prune(run, 2, yes=True, fmt="epoch")
            self.assertEqual(_epoch_names(run), ["epoch-1", "epoch-2"])

    def test_prune_unverified_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            for k in range(3):
                _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            with self.assertRaises(SystemExit):
                checkpoints.prune(run, 0, yes=True, fmt="epoch")
            self.assertEqual(len(_epoch_names(run)), 3)

    def test_prune_refuses_unverified_and_leaves_every_file(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            for k in range(3):
                _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            (run / "logs").mkdir()
            (run / "logs" / "train.log").write_text("line\n")
            before = _snapshot(run)
            with self.assertRaises(SystemExit):
                checkpoints.prune(run, 0, yes=True, fmt="epoch")
            self.assertEqual(_snapshot(run), before)

    def test_prune_refuses_abandoned_dir(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            for k in range(3):
                _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            (run / "ABANDONED").write_text('{"reason": "gave up"}\n')
            before = _snapshot(run)
            with self.assertRaises(SystemExit):
                checkpoints.prune(run, 0, yes=True, fmt="epoch")
            self.assertEqual(_snapshot(run), before)

    def test_prune_dry_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            before = _epoch_names(run)
            checkpoints.prune(run, 1, fmt="epoch")
            self.assertEqual(_epoch_names(run), before)
            self.assertEqual(len(before), 3)

    def test_prune_never_touches_outside_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = _verified_run(root, 3)
            sibling = root / "other"
            (sibling / "epoch-7").mkdir(parents=True)
            (sibling / "epoch-7" / "talker.safetensors").write_bytes(b"keep")
            checkpoints.prune(run, 0, yes=True, fmt="epoch")
            self.assertTrue((sibling / "epoch-7" / "talker.safetensors").exists())


class PruneOmnivoiceTest(unittest.TestCase):
    """Decision Q1b: prune keeps the two newest, except steps 1000 and 2500."""

    def test_prune_keeps_newest_two_plus_exempt(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
            checkpoints.prune(run, yes=True)
            self.assertEqual(_ckpt_names(run),
                             ["checkpoint-1000", "checkpoint-2500", "checkpoint-4500", "checkpoint-5000"])

    def test_prune_deletes_nothing_when_every_checkpoint_is_kept(self):
        # Newest two plus exempt cover all three: nothing is deleted.
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [1000, 2500, 5000])
            before = _ckpt_names(run)
            checkpoints.prune(run, yes=True)
            self.assertEqual(_ckpt_names(run), before)

    def test_prune_deletes_old_checkpoints_below_the_limit(self):
        # 500 is neither among the newest two nor exempt: it goes.
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [500, 1000, 1500])
            checkpoints.prune(run, yes=True)
            self.assertEqual(_ckpt_names(run), ["checkpoint-1000", "checkpoint-1500"])

    def test_prune_custom_exempt_step(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [500, 1000, 1500, 2000])
            checkpoints.prune(run, yes=True, exempt=(1500,))
            self.assertEqual(_ckpt_names(run), ["checkpoint-1500", "checkpoint-2000"])

    def test_prune_dry_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [500, 1000, 1500, 2000])
            before = _ckpt_names(run)
            checkpoints.prune(run)
            self.assertEqual(_ckpt_names(run), before)

    def test_prune_refuses_unverified(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _ckpt(run, 500)
            _ckpt(run, 1000)
            _ckpt(run, 1500)
            before = _snapshot(run)
            with self.assertRaises(SystemExit):
                checkpoints.prune(run, yes=True)
            self.assertEqual(_snapshot(run), before)

    def test_cli_prune_rejects_keep_in_omnivoice_mode(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [500, 1000, 1500])
            bad = subprocess.run(CLI + ["prune", str(run), "--keep", "1", "--yes"],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("--keep is epoch-mode only", bad.stderr)

    def test_cli_prune_applies_the_exemption(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_ckpt_run(Path(d), [500, 1000, 1500, 2000, 2500, 3000])
            ok = subprocess.run(CLI + ["prune", str(run), "--yes"],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual(_ckpt_names(run), ["checkpoint-1000", "checkpoint-2500", "checkpoint-3000"])


class AuditTest(unittest.TestCase):
    def test_audit_lists_a_3_epoch_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"
            for k in range(3):
                _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            (run / "VERIFIED").write_text("{}")
            clean = root / "clean"
            for k in range(2):
                _epoch(clean, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            (clean / "VERIFIED").write_text("{}")
            self.assertEqual(checkpoints.audit(root, fmt="epoch"), [(run, 3)])

    def test_audit_counts_non_exempt_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            over = root / "over"
            for s in (500, 1000, 1500, 2000, 2500):
                _ckpt(over, s)
            (over / "VERIFIED").write_text("{}")
            # 3 non-exempt (500, 1500, 2000) -> flagged.
            fine = root / "fine"
            for s in (1000, 1500, 2500):
                _ckpt(fine, s)
            (fine / "VERIFIED").write_text("{}")
            # 1 non-exempt (1500) -> not flagged.
            self.assertEqual(checkpoints.audit(root), [(over, 3)])


class MarkerTest(unittest.TestCase):
    def _marker(self, state, stage, path):
        d = state / "unverified"
        d.mkdir(parents=True, exist_ok=True)
        (d / stage).write_text(f"{path}\n")

    def test_verify_clears_exact_marker_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = root / "state"
            run = root / "runs" / "checkpoint"
            unrelated = root / "other" / "checkpoint"
            self._marker(state, "stage-eq", run)          # recorded path == run
            self._marker(state, "stage-parent", run.parent)  # a parent, must stay
            self._marker(state, "stage-other", unrelated)    # unrelated, must stay
            _ckpt(run, 500)
            with mock.patch.dict(os.environ, {"VI_STATE_DIR": str(state)}):
                checkpoints.verify(run, 1)
            self.assertFalse((state / "unverified" / "stage-eq").exists())
            self.assertTrue((state / "unverified" / "stage-parent").exists())
            self.assertTrue((state / "unverified" / "stage-other").exists())

    def test_failed_verify_clears_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = root / "state"
            run = root / "runs" / "checkpoint"
            self._marker(state, "stage-eq", run)
            _ckpt(run, 500)
            (run / "checkpoint-500" / "model.safetensors").write_bytes(b"corrupt!!")  # corrupt
            with mock.patch.dict(os.environ, {"VI_STATE_DIR": str(state)}):
                with self.assertRaises(SystemExit):
                    checkpoints.verify(run, 1)
            self.assertTrue((state / "unverified" / "stage-eq").exists())


class AbandonTest(unittest.TestCase):
    def test_abandon_clears_exact_marker_and_keeps_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = root / "state"
            run = root / "runs" / "checkpoint"
            unrelated = root / "other" / "checkpoint"
            marker_dir = state / "unverified"
            marker_dir.mkdir(parents=True)
            (marker_dir / "stage-eq").write_text(f"{run}\n")
            (marker_dir / "stage-parent").write_text(f"{run.parent}\n")
            (marker_dir / "stage-other").write_text(f"{unrelated}\n")
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            (run / "notes.txt").write_text("keep me")
            before = _snapshot(run)
            with mock.patch.dict(os.environ, {"VI_STATE_DIR": str(state)}):
                checkpoints.abandon(run, "gave up on this attempt")
            after = _snapshot(run)
            for rel, data in before.items():
                self.assertEqual(after.get(rel), data, rel)
            record = json.loads((run / "ABANDONED").read_text())
            self.assertEqual(record["reason"], "gave up on this attempt")
            self.assertFalse((marker_dir / "stage-eq").exists())
            self.assertTrue((marker_dir / "stage-parent").exists())
            self.assertTrue((marker_dir / "stage-other").exists())


if __name__ == "__main__":
    unittest.main()
