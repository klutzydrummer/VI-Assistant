"""Tests for omnivoice-ft/ft_guards.py (standard library only: no torch, no omnivoice)."""

import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "omnivoice-ft"))
import ft_guards as FG


class _Processor:
    """Raises on 30 % of the rows (every row whose index mod 10 is 0, 1 or 2)."""

    text_tokenizer = "tok"

    def __call__(self, sample):
        if sample % 10 < 3:
            raise RuntimeError(f"bad row {sample}")
        return sample


def _run_like_upstream(counter, n):
    """What StreamLengthGroupDataset does: call the processor, silently skip a raise."""
    kept = 0
    for i in range(n):
        try:
            counter(i)
            kept += 1
        except Exception:
            pass
    return kept


def _data_config(root, lines, repeat=1):
    lst = root / "data.lst"
    lst.write_text("".join(l + "\n" for l in lines))
    cfg = root / "data.json"
    cfg.write_text(json.dumps({"train": [{"manifest_path": [str(lst)], "repeat": repeat}]}))
    return str(cfg)


class GuardTests(unittest.TestCase):
    def test_ft_data_counts_skipped_samples(self):
        counter = FG.SampleCounter(_Processor())
        self.assertEqual(counter.text_tokenizer, "tok")  # collator attributes pass through
        self.assertEqual(_run_like_upstream(counter, 100), 70)
        self.assertEqual((counter.ok, counter.failed), (70, 30))
        self.assertEqual(len(counter.first_errors), 3)
        with self.assertRaises(ValueError) as cm:
            FG.check_sample_counts(counter, "probe pass")
        msg = str(cm.exception)
        self.assertIn("ft_data: 30 of 100 samples failed in the processor", msg)
        self.assertIn("first error: RuntimeError('bad row 0')", msg)

    def test_ft_data_counts_cross_process_workers(self):
        import multiprocessing as mp
        counter = FG.SampleCounter(_Processor())
        p = mp.get_context("fork").Process(target=_run_like_upstream, args=(counter, 20))
        p.start()
        p.join()
        self.assertEqual((counter.ok, counter.failed), (14, 6))

    def test_few_failures_are_tolerated(self):
        counter = FG.SampleCounter(lambda s: 1 / s)  # one ZeroDivisionError in 1000
        _run_like_upstream(counter, 1000)
        self.assertEqual(counter.failed, 1)
        FG.check_sample_counts(counter, "x")  # 0.1 %, 1 sample: fine

    def test_train_ft_too_few_batches(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _data_config(Path(d), ["a.tar a.jsonl 3 2.0"])
            with self.assertRaises(ValueError) as cm:
                FG.check_batches_per_epoch(cfg, 8192)
            self.assertIn("expected 1 batch per epoch", str(cm.exception))
            self.assertIn("need at least 50", str(cm.exception))

    def test_enough_batches_pass_and_halving_doubles_them(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _data_config(Path(d), ["a.tar a.jsonl 500 4000.0"])  # 100000 tokens
            self.assertEqual(FG.expected_batches_per_epoch(cfg, 8192), 13)
            self.assertEqual(FG.expected_batches_per_epoch(cfg, 2048), 49)
            self.assertEqual(FG.check_batches_per_epoch(cfg, 1024), 98)

    def test_missing_seconds_column_uses_target_frames(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _data_config(Path(d), ["a.tar a.jsonl 1000"])  # 250000 tokens
            self.assertEqual(FG.expected_batches_per_epoch(cfg, 5000), 50)

    def test_train_ft_resume_allows_equal_product(self):
        saved = {"batch_tokens": 8192, "gradient_accumulation_steps": 1}
        self.assertIsNone(FG.resume_product_mismatch(saved, 4096, 2))
        self.assertIsNone(FG.resume_product_mismatch(saved, 2048, 4))
        self.assertIsNone(FG.resume_product_mismatch({}, 1, 1))
        self.assertIn("8192 to 4096", FG.resume_product_mismatch(saved, 4096, 1))

    def test_oom_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = FG.write_oom_json(d, 3, 8192, 123)
            self.assertEqual(json.load(open(path)),
                             {"step": 3, "batch_tokens": 8192, "max_memory_allocated": 123})


    def test_r23_report_oom_prints_torch_text_and_memory(self):
        # base: the handler dropped the exception and printed only its own summary
        class FakeCuda:
            def is_available(self):
                return True

            def max_memory_allocated(self):
                return 20 * 2**30

            def max_memory_reserved(self):
                return 23 * 2**30

            def mem_get_info(self):
                return (512 * 2**20, 24 * 2**30)

        err = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
                           "capacity of 23.69 GiB\nmore detail on a second line")
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            rc = FG.report_oom(d, 0, 8192, err, FakeCuda(), out=buf)
            info = json.loads(Path(d, "oom.json").read_text())
        self.assertEqual(rc, 76)
        text = buf.getvalue()
        self.assertIn("train_ft: torch: CUDA out of memory. Tried to allocate 2.00 GiB", text)
        self.assertNotIn("second line", text)
        self.assertIn("free 0.50 of 24.00 GiB", text)
        self.assertIn("max reserved 23.00 GiB", text)
        self.assertIn("max allocated 20.00 GiB", text)
        self.assertIn("train_ft: GPU out of memory at micro-step 0, batch_tokens 8192", text)
        self.assertEqual(info["step"], 0)
        self.assertEqual(info["batch_tokens"], 8192)
        self.assertEqual(info["max_memory_allocated"], 20 * 2**30)
        self.assertEqual(info["max_memory_reserved"], 23 * 2**30)
        self.assertEqual(info["mem_free"], 512 * 2**20)
        self.assertEqual(info["mem_total"], 24 * 2**30)
        self.assertTrue(info["error"].startswith("CUDA out of memory. Tried"))

    def test_r23_report_oom_without_cuda_or_with_a_failing_query(self):
        class NoCuda:
            def is_available(self):
                return False

        class Broken(NoCuda):
            def is_available(self):
                return True

            def max_memory_allocated(self):
                return 7

            def max_memory_reserved(self):
                raise RuntimeError("driver gone")

            def mem_get_info(self):
                raise RuntimeError("driver gone")

        for cuda in (NoCuda(), Broken()):
            buf = io.StringIO()
            with tempfile.TemporaryDirectory() as d:
                rc = FG.report_oom(d, 3, 4096, RuntimeError(""), cuda, out=buf)
                info = json.loads(Path(d, "oom.json").read_text())
            self.assertEqual(rc, 76)
            self.assertIn("train_ft: GPU out of memory at micro-step 3, batch_tokens 4096", buf.getvalue())
            self.assertEqual(info["batch_tokens"], 4096)
            self.assertIsNone(info["mem_free"])


class _FakeCuda:
    """torch.cuda with fixed peaks: 20 GiB allocated, 23 GiB reserved."""

    def is_available(self):
        return True

    def max_memory_allocated(self):
        return 20 * 2**30

    def max_memory_reserved(self):
        return 23 * 2**30

    def mem_get_info(self):
        return (512 * 2**20, 24 * 2**30)


# Stand-ins named like the real call chain: our forward calls the KL reference, whose own
# forward runs out of memory; or our forward itself runs out.
def _kl_to_reference():
    raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")


def forward(kl):  # noqa: A001 - named like nn.Module.forward on purpose
    if kl:
        _kl_to_reference()
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class R25DiagnosticsTest(unittest.TestCase):
    def test_r25_defaults_keep_8192_tokens_per_optimizer_step(self):
        # base: there were no shared defaults (train_ft.py said 8192 x 1 in two places)
        self.assertEqual(FG.DEFAULT_BATCH_TOKENS, 2048)
        self.assertEqual(FG.DEFAULT_GRAD_ACCUM, 4)
        self.assertEqual(FG.DEFAULT_BATCH_TOKENS * FG.DEFAULT_GRAD_ACCUM, 8192)

    def _oom(self, kl):
        try:
            forward(kl)
        except RuntimeError as e:
            return e

    def test_r25_oom_json_keeps_the_traceback_and_says_where(self):
        # base: oom.json had only the message, so student and reference forward looked the same
        for kl, where in ((True, "kl reference forward"), (False, "student forward")):
            buf = io.StringIO()
            with tempfile.TemporaryDirectory() as d, \
                    unittest.mock.patch.object(FG, "_OUR_FILES", FG._OUR_FILES + ("test_ft_guards.py",)):
                FG.report_oom(d, 0, 2048, self._oom(kl), _FakeCuda(), out=buf)
                info = json.loads(Path(d, "oom.json").read_text())
            self.assertEqual(info["where"], where)
            self.assertTrue(any(f.endswith(" in forward") for f in info["traceback"]), info)
            self.assertEqual(any(f.endswith(" in _kl_to_reference") for f in info["traceback"]), kl)
            line = [l for l in buf.getvalue().splitlines() if l.startswith("train_ft: oom in ")]
            self.assertEqual(len(line), 1, buf.getvalue())
            self.assertTrue(line[0].startswith(f"train_ft: oom in {where}: test_ft_guards.py:"), line)
            self.assertLessEqual(len(line[0]), 300)

    def test_r25_oom_without_a_traceback(self):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            FG.report_oom(d, 0, 2048, RuntimeError("x"), _FakeCuda(), out=buf)
            info = json.loads(Path(d, "oom.json").read_text())
        self.assertEqual(info["where"], "unknown")
        self.assertEqual(info["traceback"], [])
        self.assertIn("train_ft: oom in unknown: no traceback", buf.getvalue())

    def test_r25_oom_where_classifies_backward_and_optimizer(self):
        self.assertEqual(FG.oom_where(["/x/torch/_tensor.py:1 in backward"]), "backward")
        self.assertEqual(FG.oom_where(["/x/torch/optim/adamw.py:1 in step"]), "optimizer step")
        self.assertEqual(FG.oom_where(["/x/lib.py:1 in forward"]), "other")

    def test_r25_vram_peak_line(self):
        line = FG.vram_peak_line(_FakeCuda(), "after the first optimizer step")
        self.assertEqual(line, "train_ft: vram peak: allocated 20.00 GiB reserved 23.00 GiB "
                               "(after the first optimizer step)")

    def test_r25_vram_peak_line_does_not_feed_the_vram_gate(self):
        # voicetune-fresh.sh's gate reads only "train_ft: vram peak X of Y GiB" lines.
        import re
        gate = re.compile(r"train_ft: vram peak [0-9.]+ of [0-9.]+ GiB")
        self.assertIsNone(gate.search(FG.vram_peak_line(_FakeCuda(), "end of training")))

    def test_r25_vram_peak_line_without_cuda(self):
        class NoCuda:
            def is_available(self):
                return False

        class Broken(_FakeCuda):
            def max_memory_reserved(self):
                raise RuntimeError("driver gone")

        self.assertIsNone(FG.vram_peak_line(NoCuda(), "end of training"))
        self.assertIsNone(FG.vram_peak_line(Broken(), "end of training"))

    def test_r25_sdpa_mask_warning_lines(self):
        import warnings
        with warnings.catch_warnings(record=True) as recs:
            warnings.simplefilter("always")
            warnings.warn("Memory efficient kernel not used because: attn_mask is a dense bool")
            warnings.warn("Memory efficient kernel not used because: attn_mask is a dense bool")
            warnings.warn("something about tokenizers")
        self.assertEqual(FG.sdpa_mask_warning_lines(recs), [
            "train_ft: sdpa-mask-warning: Memory efficient kernel not used because: attn_mask "
            "is a dense bool"])
        self.assertEqual(FG.sdpa_mask_warning_lines(recs[2:]), [
            "train_ft: sdpa-mask-warning: none (no attention-kernel warning in the first forward)"])


if __name__ == "__main__":
    unittest.main()
