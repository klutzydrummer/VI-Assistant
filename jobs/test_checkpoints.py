"""Tests for checkpoints.py (standard library only, tiny files in temp dirs)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import checkpoints


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


def _verified_run(root, n):
    """A run dir under `root` with `n` verified epochs (0..n-1) and a VERIFIED file."""
    run = root / "run"
    run.mkdir()
    for k in range(n):
        _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
    checkpoints.verify(run, n)
    return run


def _epoch_names(run):
    return sorted(p.name for p in run.glob("epoch-*") if not p.name.endswith(".part"))


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
    def test_verify_passes_a_good_run(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            _epoch(run, 1, {"talker.safetensors": b"c", "tune.json": b"d"})
            out = checkpoints.verify(run, 2)
            self.assertEqual(set(out), {"epoch-0", "epoch-1"})
            v = (run / "VERIFIED").read_text()
            self.assertIn("epoch-0", v)
            self.assertIn(checkpoints.sha256_file(run / "epoch-0" / "talker.safetensors"), v)

    def test_verify_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"aaaa", "tune.json": b"b"})
            (run / "epoch-0" / "talker.safetensors").write_bytes(b"bbbb")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"aaaa", "tune.json": b"b"})
            (run / "epoch-0" / "talker.safetensors").write_bytes(b"")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_when_expect_is_wrong(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            _epoch(run, 1, {"talker.safetensors": b"c", "tune.json": b"d"})
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 3)

    def test_verify_fails_on_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            outside = root / "outside.txt"
            outside.write_text("not a checkpoint")
            run = root / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            (run / "epoch-0" / "talker.safetensors").unlink()
            (run / "epoch-0" / "talker.safetensors").symlink_to(outside)
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_leftover_part(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            (run / "epoch-1.part").mkdir()
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_when_manifest_omits_talker(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            ep = run / "epoch-0"
            ep.mkdir(parents=True)
            (ep / "talker.safetensors").write_bytes(b"a")
            (ep / "tune.json").write_bytes(b"b")
            (ep / "SHA256SUMS").write_text(f"{checkpoints.sha256_file(ep / 'tune.json')}  tune.json\n")
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)

    def test_verify_fails_on_parent_manifest_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run"
            ep = run / "epoch-0"
            ep.mkdir(parents=True)
            (ep / "talker.safetensors").write_bytes(b"a")
            (ep / "tune.json").write_bytes(b"b")
            (root / "x").write_text("outside")
            lines = [
                f"{checkpoints.sha256_file(ep / 'talker.safetensors')}  talker.safetensors\n",
                f"{checkpoints.sha256_file(ep / 'tune.json')}  tune.json\n",
                f"{checkpoints.sha256_file(root / 'x')}  ../x\n",
            ]
            (ep / "SHA256SUMS").write_text("".join(lines))
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 1)


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


class PruneTest(unittest.TestCase):
    def test_prune_first_epoch_keeps_zero_and_one(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            checkpoints.prune(run, 0, yes=True)
            self.assertEqual(_epoch_names(run), ["epoch-0", "epoch-1"])

    def test_prune_middle_epoch_keeps_one_and_two(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            checkpoints.prune(run, 1, yes=True)
            self.assertEqual(_epoch_names(run), ["epoch-1", "epoch-2"])

    def test_prune_last_epoch_keeps_two_and_one(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            checkpoints.prune(run, 2, yes=True)
            self.assertEqual(_epoch_names(run), ["epoch-1", "epoch-2"])

    def test_prune_unverified_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            for k in range(3):
                _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            with self.assertRaises(SystemExit):
                checkpoints.prune(run, 0, yes=True)
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
                checkpoints.prune(run, 0, yes=True)
            self.assertEqual(_snapshot(run), before)

    def test_prune_refuses_abandoned_dir(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            for k in range(3):
                _epoch(run, k, {"talker.safetensors": f"w{k}".encode(), "tune.json": b"{}"})
            (run / "ABANDONED").write_text('{"reason": "gave up"}\n')
            before = _snapshot(run)
            with self.assertRaises(SystemExit):
                checkpoints.prune(run, 0, yes=True)
            self.assertEqual(_snapshot(run), before)

    def test_prune_dry_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            run = _verified_run(Path(d), 3)
            before = _epoch_names(run)
            checkpoints.prune(run, 1)
            self.assertEqual(_epoch_names(run), before)
            self.assertEqual(len(before), 3)

    def test_prune_never_touches_outside_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = _verified_run(root, 3)
            sibling = root / "other"
            (sibling / "epoch-7").mkdir(parents=True)
            (sibling / "epoch-7" / "talker.safetensors").write_bytes(b"keep")
            checkpoints.prune(run, 0, yes=True)
            self.assertTrue((sibling / "epoch-7" / "talker.safetensors").exists())


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
            self.assertEqual(checkpoints.audit(root), [(run, 3)])


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
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            with mock.patch.dict(os.environ, {"VI_STATE_DIR": str(state)}):
                checkpoints.verify(run, 1)
            self.assertFalse((state / "unverified" / "stage-eq").exists())
            self.assertTrue((state / "unverified" / "stage-parent").exists())
            self.assertTrue((state / "unverified" / "stage-other").exists())

    def test_verify_fails_when_epochs_are_not_contiguous(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            _epoch(run, 0, {"talker.safetensors": b"a", "tune.json": b"b"})
            _epoch(run, 2, {"talker.safetensors": b"c", "tune.json": b"d"})
            with self.assertRaises(SystemExit):
                checkpoints.verify(run, 2)

    def test_failed_verify_clears_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = root / "state"
            run = root / "runs" / "checkpoint"
            self._marker(state, "stage-eq", run)
            _epoch(run, 0, {"talker.safetensors": b"aaaa", "tune.json": b"b"})
            (run / "epoch-0" / "talker.safetensors").write_bytes(b"bbbb")  # corrupt
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
