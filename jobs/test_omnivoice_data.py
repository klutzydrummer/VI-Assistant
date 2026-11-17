"""Tests for jobs/omnivoice_data.py, the home-side builder of the OmniVoice data folder.
Standard library only; synthetic clips (empty files) and synthetic manifests, no game audio."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))
import omnivoice_data


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_pool(root, mood="calm"):
    """out/ holding buildset-shaped raw manifests and clip files. Every row carries label fields
    (mood, rating, an 'in old labelled set' flag) the builder must never look at."""
    out = Path(root) / "out"
    (out / "wav").mkdir(parents=True)
    rows = {"train": [], "val": []}
    for split, n in (("train", 40), ("val", 30)):
        for i in range(n):
            cid = f"{split}-{i:03d}"
            (out / "wav" / f"{cid}.wav").write_bytes(b"RIFF")
            rows[split].append({"id": cid, "audio": f"wav/{cid}.wav", "text": f"line [sfx] {i}  here",
                                "mood": mood if i % 2 else "angry", "rating": i % 5,
                                "in_old_labelled_set": i % 3 == 0})
        write_jsonl(out / f"{split}_raw.jsonl", rows[split])
    return out


def make_support(root):
    voices = Path(root) / "libri"
    voices.mkdir()
    for n in ("a.wav", "b.wav"):
        (voices / n).write_bytes(b"RIFF")
    texts = Path(root) / "replay_texts.jsonl"
    write_jsonl(texts, [{"text": "Public line one.", "audio_codes": [[1, 2]], "mood": "x"},
                        {"text": "Public line two."}])
    return voices, texts


def build(root, mood="calm", tokenizer=None, **kw):
    out = make_pool(root, mood)
    voices, texts = make_support(root)
    summary = omnivoice_data.build(out, None, None, voices, texts, tokenizer, raw_dir=out, **kw)
    return out, summary


class BuildTest(unittest.TestCase):
    def test_folder_has_everything_the_worker_reads(self):
        with tempfile.TemporaryDirectory() as d:
            out, summary = build(d)
            for name in ("train_manifest.jsonl", "dev_manifest.jsonl", "style_manifest.jsonl",
                         "data_config.json", "replay_texts.jsonl", "voices/a.wav", "voices/b.wav",
                         "audio_tokenizer/PLACEHOLDER", "omnivoice_data_summary.json"):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(summary["audio_tokenizer"], "placeholder")

    def test_every_clip_is_kept_with_only_schema_fields(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = build(d)
            train = read_jsonl(out / "train_manifest.jsonl")
            dev = read_jsonl(out / "dev_manifest.jsonl")
            self.assertEqual((len(train), len(dev)), (40, 30))  # nothing dropped for any label
            for row in train + dev:
                self.assertEqual(set(row), {"id", "audio_path", "text", "language_id"})
                self.assertEqual(row["text"].count("["), 0)

    def test_audio_paths_are_relative_and_resolve_inside_the_folder(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = build(d)
            for name in ("train_manifest.jsonl", "dev_manifest.jsonl", "style_manifest.jsonl"):
                for row in read_jsonl(out / name):
                    self.assertFalse(os.path.isabs(row["audio_path"]), row)
                    self.assertTrue((out / row["audio_path"]).is_file(), row)

    def test_changing_every_label_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a, _ = build(d1, mood="calm")
            b, _ = build(d2, mood="furious")
            for name in ("train_manifest.jsonl", "dev_manifest.jsonl", "style_manifest.jsonl"):
                self.assertEqual((a / name).read_text(), (b / name).read_text(), name)

    def test_style_rows_are_a_hash_ranked_sample_of_dev(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = build(d, style_n=5)
            dev = {r["id"]: r for r in read_jsonl(out / "dev_manifest.jsonl")}
            style = read_jsonl(out / "style_manifest.jsonl")
            self.assertEqual(len(style), 5)
            ranked = sorted(dev, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:5]
            self.assertEqual([r["id"] for r in style], ranked)
            for r in style:
                self.assertEqual(set(r), {"id", "audio_path", "text"})
                self.assertEqual(r["audio_path"], dev[r["id"]]["audio_path"])

    def test_data_config_names_the_shards_the_worker_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = build(d)
            cfg = json.loads((out / "data_config.json").read_text())
            self.assertEqual(cfg["train"][0]["manifest_path"],
                             ["shards/train/data.lst", "shards/replay/data.lst"])
            self.assertEqual(cfg["dev"][0]["manifest_path"], ["shards/dev/data.lst"])

    def test_replay_texts_keep_only_the_text_field(self):
        with tempfile.TemporaryDirectory() as d:
            out, summary = build(d)
            self.assertEqual(read_jsonl(out / "replay_texts.jsonl"),
                             [{"text": "Public line one."}, {"text": "Public line two."}])
            self.assertEqual(summary["replay_texts"], 2)

    def test_a_given_tokenizer_is_copied_in(self):
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "tok"
            tok.mkdir()
            (tok / "config.json").write_text("{}")
            out, summary = build(d, tokenizer=tok)
            self.assertEqual(summary["audio_tokenizer"], "copied")
            self.assertTrue((out / "audio_tokenizer" / "config.json").is_file())
            self.assertFalse((out / "audio_tokenizer" / "PLACEHOLDER").exists())

    def test_no_voices_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pool(d)
            empty = Path(d) / "novoices"
            empty.mkdir()
            _, texts = make_support(d)
            with self.assertRaises(SystemExit):
                omnivoice_data.build(out, None, None, empty, texts, None, raw_dir=out)

    def test_source_never_names_a_label_field(self):
        # The builder and the manifest tool it calls must not read label fields.
        text = (JOBS / "omnivoice_data.py").read_text()
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        for word in ("mood", "rating", "labelled_set", "moods.toml", "states.toml"):
            self.assertNotIn(f'["{word}"]', code)
            self.assertNotIn(f"get('{word}'", code)
            self.assertNotIn(f'get("{word}"', code)


if __name__ == "__main__":
    unittest.main()
