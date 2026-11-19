"""Tests for jobs/bundle_ids.py and the id sanitising in the OmniVoice data build (round 8).
Standard library only; synthetic ids shaped like the real ones (`le2.s178234_m`). A test that uses
the real webdataset package skips cleanly when it is not installed."""

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))
sys.path.insert(0, str(JOBS / "omnivoice-ft"))
import bundle_ids
import data_manifest
import omnivoice_data


def jl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def folder(root, train, dev, style=None, replay=2):
    f = Path(root)
    jl(f / "train_manifest.jsonl", [{"id": i} for i in train])
    jl(f / "dev_manifest.jsonl", [{"id": i} for i in dev])
    jl(f / "style_manifest.jsonl", [{"id": i} for i in (dev if style is None else style)])
    jl(f / "replay_texts.jsonl", [{"text": f"t{i}"} for i in range(replay)])
    return f


class KeyRule(unittest.TestCase):
    def test_key_is_the_text_before_the_first_dot(self):
        self.assertEqual(bundle_ids.webdataset_key("le2.s178234_m.npy"), ("le2", "s178234_m.npy"))
        self.assertEqual(bundle_ids.webdataset_key("le2_s178234_m.npy"), ("le2_s178234_m", "npy"))
        self.assertEqual(bundle_ids.webdataset_key("a/b.c.npy"), ("a/b", "c.npy"))


class CheckIds(unittest.TestCase):
    def check(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            return bundle_ids.check_ids(folder(d, **kw))

    def test_clean_bundle_passes(self):
        ok, problems, _ = self.check(train=["le2_s1_m", "le2_s2_m"], dev=["le3_s9_m"])
        self.assertTrue(ok, problems)

    def test_dotted_ids_fail_with_count_and_first_ids(self):
        ok, problems, _ = self.check(train=["le2.s1_m", "le2.s2_m", "x_ok"], dev=["le3.s9_m"])
        self.assertFalse(ok)
        text = "\n".join(problems)
        self.assertIn("2 of 3 train id(s)", text)
        self.assertIn("1 of 1 dev id(s)", text)
        self.assertIn("'le2.s1_m'", text)

    def test_ids_repeated_across_train_and_dev_fail(self):
        ok, problems, _ = self.check(train=["a", "b"], dev=["b"])
        self.assertFalse(ok)
        self.assertIn("repeat across", "\n".join(problems))

    def test_style_ids_must_be_dev_ids(self):
        ok, problems, _ = self.check(train=["a"], dev=["b"], style=["zzz"])
        self.assertFalse(ok)
        self.assertIn("style id(s) are not dev ids", "\n".join(problems))

    def test_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as d:
            ok, problems, _ = bundle_ids.check_ids(d)
        self.assertFalse(ok)
        self.assertIn("is missing", problems[0])

    def test_shard_keys_without_labels_fail(self):
        with tempfile.TemporaryDirectory() as d:
            f = folder(d, train=["le2_s1_m"], dev=["le3_s9_m"])
            (f / "shards/train").mkdir(parents=True)
            tar_path, lab = f / "shards/train/a.tar", f / "shards/train/a.jsonl"
            with tarfile.open(tar_path, "w") as t:
                for name in ("le2.s1_m.npy",):
                    info = tarfile.TarInfo(name)
                    t.addfile(info)
            jl(lab, [{"id": "le2_s1_m"}])
            (f / "shards/train/data.lst").write_text(f"{tar_path} {lab} 1\n")
            ok, problems, _ = bundle_ids.check_ids(f)
        self.assertFalse(ok)
        self.assertIn("no label", "\n".join(problems))


class Sanitising(unittest.TestCase):
    def raw(self, d, name, ids):
        (Path(d) / "wav").mkdir(exist_ok=True)
        rows = []
        for i in ids:
            (Path(d) / "wav" / f"{i}.wav").write_bytes(b"RIFF")
            rows.append({"id": i, "audio": f"wav/{i}.wav", "text": "hello"})
        jl(Path(d) / name, rows)
        return str(Path(d) / name)

    def test_dots_become_underscores_in_ids_but_not_in_paths(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.raw(d, "r.jsonl", ["le2.s178234_m"])
            data_manifest.build_manifest(raw_jsonls=[raw], audio_root=d, out=str(Path(d) / "m.jsonl"))
            row = json.loads((Path(d) / "m.jsonl").read_text())
        self.assertEqual(row["id"], "le2_s178234_m")
        self.assertTrue(row["audio_path"].endswith("wav/le2.s178234_m.wav"))

    def test_colliding_ids_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.raw(d, "r.jsonl", ["a.b", "a_b"])
            with self.assertRaises(SystemExit) as cm:
                data_manifest.build_manifest(raw_jsonls=[raw], audio_root=d, out=str(Path(d) / "m.jsonl"))
        self.assertIn("duplicate id 'a_b'", str(cm.exception))

    def test_a_build_fails_if_a_dotted_id_survives(self):
        # Simulate a regression: a sanitiser that does nothing must be caught by build() itself.
        real = data_manifest.sanitise_id
        data_manifest.sanitise_id = lambda i: i
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "out"
                out.mkdir()
                self.raw(out, "train_raw.jsonl", ["le2.s1_m"])
                self.raw(out, "val_raw.jsonl", ["le3.s2_m"])
                voices = Path(d) / "v"
                voices.mkdir()
                (voices / "a.wav").write_bytes(b"RIFF")
                texts = Path(d) / "t.jsonl"
                jl(texts, [{"text": "x"}])
                with self.assertRaises(SystemExit) as cm:
                    omnivoice_data.build(out, None, None, voices, texts, None, raw_dir=out)
        finally:
            data_manifest.sanitise_id = real
        self.assertIn("contain a '.'", str(cm.exception))


class RealWebdataset(unittest.TestCase):
    def test_real_webdataset_agrees_with_the_pure_python_rule(self):
        try:
            from webdataset.tariterators import base_plus_ext
        except ImportError:
            self.skipTest("webdataset is not installed")
        for name in ("le2.s178234_m.npy", "le2_s178234_m.npy", "a/b.c.npy", "plain"):
            self.assertEqual(tuple(base_plus_ext(name)), bundle_ids.webdataset_key(name), name)


if __name__ == "__main__":
    unittest.main()
