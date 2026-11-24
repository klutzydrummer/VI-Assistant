"""Torch-free guards for the OmniVoice fine-tune (F-DATA-2, F-OOM-1).

Everything here uses only the standard library so it runs, and is tested, without torch or the
upstream ``omnivoice`` package. It only counts and compares: it never reads a label, never
chooses, weights or filters a clip.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

MIN_BATCHES_PER_EPOCH = 50
FRAMES_PER_SECOND = 25  # the Higgs audio tokenizer's frame rate (unverified here; see README)
MAX_FAILED_FRACTION = 0.01
MAX_FAILED_SAMPLES = 5
OOM_EXIT_CODE = 76
# The trainer's default micro-batch and accumulation (r25 B1): 2048 tokens x 4 micro-steps keeps the
# 8192 tokens per optimizer step the run always had, while a micro-step's saved activations
# (about 2.3 MiB per token, round 24's CPU measurement) drop from about 18.4 GiB to about 4.6 GiB.
# train_ft.py's config and CLI defaults, and worker-job.sh's OOM halving when the flags are absent,
# all read these two names; nothing else repeats the numbers.
DEFAULT_BATCH_TOKENS = 2048
DEFAULT_GRAD_ACCUM = 4
_ERR_SLOTS = 3
_ERR_BYTES = 240


class SampleCounter:
    """Wraps the sample processor, counting successes and failures (and the first 3 errors).

    The wrapped call still raises on failure, so the upstream ``StreamLengthGroupDataset`` skips
    the sample exactly as before; the counter only records what upstream would hide. The counts
    live in shared memory, so DataLoader worker processes (forked) and the parent see the same
    totals.
    """

    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn
        self._ok = multiprocessing.Value("l", 0)
        self._failed = multiprocessing.Value("l", 0)
        self._errs = multiprocessing.Array("c", _ERR_SLOTS * _ERR_BYTES)

    def __getattr__(self, name: str) -> Any:  # text_tokenizer, base, ... pass through
        if name in ("fn", "_ok", "_failed", "_errs"):
            raise AttributeError(name)
        return getattr(self.fn, name)

    def __call__(self, sample: Any) -> Any:
        try:
            out = self.fn(sample)
        except Exception as e:
            with self._failed.get_lock():
                idx = self._failed.value
                self._failed.value = idx + 1
                if idx < _ERR_SLOTS:
                    raw = repr(e).encode("utf-8", "replace")[: _ERR_BYTES - 1]
                    self._errs.raw = (
                        self._errs.raw[: idx * _ERR_BYTES]
                        + raw.ljust(_ERR_BYTES, b"\0")
                        + self._errs.raw[(idx + 1) * _ERR_BYTES :]
                    )
            raise
        with self._ok.get_lock():
            self._ok.value += 1
        return out

    @property
    def ok(self) -> int:
        return self._ok.value

    @property
    def failed(self) -> int:
        return self._failed.value

    @property
    def first_errors(self) -> List[str]:
        raw = self._errs.raw
        out = []
        for i in range(min(self.failed, _ERR_SLOTS)):
            chunk = raw[i * _ERR_BYTES : (i + 1) * _ERR_BYTES].split(b"\0", 1)[0]
            out.append(chunk.decode("utf-8", "replace"))
        return out


def check_sample_counts(counter: SampleCounter, where: str) -> None:
    """Fail when too many samples failed in the processor (> 1 % of the total, or > 5)."""
    failed, ok = counter.failed, counter.ok
    total = failed + ok
    if failed and (failed > MAX_FAILED_SAMPLES or failed > MAX_FAILED_FRACTION * total):
        errs = counter.first_errors
        raise ValueError(
            f"ft_data: {failed} of {total} samples failed in the processor ({where}) "
            f"(first error: {errs[0] if errs else 'unknown'}; first errors: {errs})"
        )


def _manifest_paths(data_config: str) -> List[Tuple[str, int]]:
    with open(data_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(data_config))
    out: List[Tuple[str, int]] = []
    for entry in cfg.get("train", []):
        paths = entry["manifest_path"]
        if isinstance(paths, str):
            paths = [paths]
        repeat = int(entry.get("repeat", 1))
        for p in paths:
            out.append((p if os.path.isabs(p) else os.path.join(base, p), repeat))
    return out


def expected_batches_per_epoch(
    data_config: str,
    batch_tokens: int,
    max_batch_size: int = 64,
    target_frames: int = 250,
) -> int:
    """Estimate batches per epoch from the data.lst columns alone (rows, seconds). No decode.

    Each data.lst line is ``tar jsonl count [seconds]``. Tokens are ``seconds * 25``; a line
    without the seconds column counts ``target_frames`` per row. The batch count is the larger of
    the token-packing bound and the per-batch sample cap.
    """
    rows = 0
    tokens = 0.0
    for lst, repeat in _manifest_paths(data_config):
        with open(lst, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                n = int(parts[2])
                secs = float(parts[3]) if len(parts) > 3 else None
                rows += n * repeat
                tokens += (secs * FRAMES_PER_SECOND if secs is not None else n * target_frames) * repeat
    if rows == 0:
        return 0
    return max(math.ceil(tokens / max(1, batch_tokens)), math.ceil(rows / max(1, max_batch_size)))


def check_batches_per_epoch(
    data_config: str,
    batch_tokens: int,
    min_batches: int = MIN_BATCHES_PER_EPOCH,
    max_batch_size: int = 64,
    target_frames: int = 250,
) -> int:
    n = expected_batches_per_epoch(data_config, batch_tokens, max_batch_size, target_frames)
    if n < min_batches:
        raise ValueError(
            f"ft_data: expected {n} batch{'' if n == 1 else 'es'} per epoch at batch_tokens "
            f"{batch_tokens}, need at least {min_batches}"
        )
    return n


def resume_product_mismatch(saved: Dict[str, Any], batch_tokens: int, grad_accum: int) -> Optional[str]:
    """Resume may change batch_tokens and grad-accum only when their product is unchanged."""
    if "batch_tokens" not in saved or "gradient_accumulation_steps" not in saved:
        return None
    old = int(saved["batch_tokens"]) * int(saved["gradient_accumulation_steps"])
    new = int(batch_tokens) * int(grad_accum)
    if old == new:
        return None
    return (
        f"batch_tokens x gradient_accumulation_steps changed from {old} to {new}; a resume may "
        "change them only when the product stays the same (same tokens per optimizer step)"
    )


def write_oom_json(output_dir: str, step: int, batch_tokens: int, max_memory_allocated: int,
                   **extra: Any) -> str:
    """$output_dir/oom.json: the micro-step, batch_tokens and peak allocation of a GPU OOM, plus any
    extra keys given (report_oom adds the reserved and free memory and torch's first line).
    worker-job.sh only checks that the file was written during the attempt."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "oom.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"step": step, "batch_tokens": batch_tokens,
                   "max_memory_allocated": max_memory_allocated, **extra}, f)
    return path


def _cuda_query(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # a dead or missing device must not hide the OOM report itself
        return None


def report_oom(output_dir: str, step: int, batch_tokens: int, err: BaseException, cuda: Any,
               out: Any = None) -> int:
    """train_ft.py's handler for torch.cuda.OutOfMemoryError (r23 A2). ``cuda`` is torch.cuda
    (passed in so this stays torch-free and testable). Prints torch's own first line (what was
    being allocated and how much the process held) and the device's free, total, peak-reserved and
    peak-allocated memory, writes the same figures into oom.json, and returns OOM_EXIT_CODE.
    r25: the traceback is kept too, every frame in oom.json and one "train_ft: oom in ..." line
    that says whether the student forward or the KL reference forward ran out (round 24 could
    not tell, because only the message was kept). Every CUDA query is guarded: on CPU, or with
    a failing driver, the figures are 0 or None."""
    out = sys.stderr if out is None else out
    on_cuda = bool(_cuda_query(cuda.is_available))
    peak = reserved = 0
    free = total = None
    if on_cuda:
        peak = int(_cuda_query(cuda.max_memory_allocated) or 0)
        reserved = int(_cuda_query(cuda.max_memory_reserved) or 0)
        info = _cuda_query(cuda.mem_get_info)
        if info:
            free, total = int(info[0]), int(info[1])
    lines = str(err).strip().splitlines()
    first = (lines[0] if lines else type(err).__name__)[:400]
    gib = 2.0 ** 30
    print(f"train_ft: torch: {first}", file=out)
    mem = (f"free {free / gib:.2f} of {total / gib:.2f} GiB" if free is not None and total is not None
           else "free/total unknown")
    print(f"train_ft: cuda memory at the OOM: {mem}; max reserved {reserved / gib:.2f} GiB; "
          f"max allocated {peak / gib:.2f} GiB" + ("" if on_cuda else " (no CUDA device)"), file=out)
    frames = oom_frames(err)
    where = oom_where(frames)
    print(f"train_ft: oom in {where}: {oom_frames_summary(frames)}"[:300], file=out)
    path = write_oom_json(output_dir, step, batch_tokens, peak, max_memory_reserved=reserved,
                          mem_free=free, mem_total=total, error=first, where=where,
                          traceback=frames)
    print(f"train_ft: GPU out of memory at micro-step {step}, batch_tokens {batch_tokens}; "
          f"wrote {path}; exit {OOM_EXIT_CODE}", file=out)
    out.flush()
    return OOM_EXIT_CODE


# ---------------------------------------------------------------------------
# r25 diagnostics: where an OOM happened, the peak memory, SDPA's mask warning
# ---------------------------------------------------------------------------

_OUR_FILES = ("style_omnivoice.py", "train_ft.py", "ft_data.py", "style_data.py", "style_encoder.py")


def oom_frames(err: BaseException) -> List[str]:
    """Every frame of err's traceback, outermost first, as "path:line in function"."""
    tb = getattr(err, "__traceback__", None)
    if tb is None:
        return []
    return [f"{f.filename}:{f.lineno} in {f.name}" for f in traceback.extract_tb(tb)]


def _is_ours(frame: str) -> bool:
    return os.path.basename(frame.split(":", 1)[0]) in _OUR_FILES


def oom_where(frames: List[str]) -> str:
    """Which part of the step ran out: the KL reference forward, the student forward, backward,
    the optimizer step, or "unknown" (no traceback)."""
    names = [f.rsplit(" in ", 1)[-1] for f in frames]
    if "_kl_to_reference" in names:
        return "kl reference forward"
    if any(n == "forward" and _is_ours(f) for f, n in zip(frames, names)):
        return "student forward"
    if any(n in ("backward", "_engine_run_backward") for n in names):
        return "backward"
    if any(n == "step" for n in names):
        return "optimizer step"
    return "unknown" if not frames else "other"


def oom_frames_summary(frames: List[str]) -> str:
    """The innermost frame of all (the library op that asked for the memory), then up to four of
    our own frames, innermost first; file basenames only, so it fits a 300-character key line."""
    def short(f: str) -> str:
        path, _, rest = f.partition(":")
        return f"{os.path.basename(path)}:{rest}"
    ours = [short(f) for f in reversed(frames) if _is_ours(f)]
    last = short(frames[-1]) if frames else "no traceback"
    if not ours:
        return last
    return f"{last} | ours: " + " < ".join(ours[:4])


def vram_peak_line(cuda: Any, when: str) -> Optional[str]:
    """"train_ft: vram peak: allocated X GiB reserved Y GiB (WHEN)" from the CUDA peak counters,
    or None without CUDA (or when the driver query fails). ``cuda`` is torch.cuda. This line is
    the diagnostic of r25; it is not the "train_ft: vram peak X of Y GiB" line that
    voicetune-fresh.sh's VRAM gate parses, and the gate's regex does not match it."""
    if not _cuda_query(cuda.is_available):
        return None
    alloc = _cuda_query(cuda.max_memory_allocated)
    reserved = _cuda_query(cuda.max_memory_reserved)
    if alloc is None or reserved is None:
        return None
    gib = 2.0 ** 30
    return f"train_ft: vram peak: allocated {alloc / gib:.2f} GiB reserved {reserved / gib:.2f} GiB ({when})"


_SDPA_WORDS = ("attention", "sdpa", "flash", "efficient", "kernel", "mask")


def sdpa_mask_warning_lines(records: List[Any]) -> List[str]:
    """B4 diagnostic: "train_ft: sdpa-mask-warning: ..." for each distinct warning recorded during
    the first forward that is about attention kernels or masks, or one "none" line. ``records``
    are warnings.WarningMessage objects; nothing here changes which kernel runs."""
    seen: List[str] = []
    for w in records:
        text = " ".join(str(getattr(w, "message", w)).split())
        if any(k in text.lower() for k in _SDPA_WORDS) and text not in seen:
            seen.append(text)
    if not seen:
        return ["train_ft: sdpa-mask-warning: none (no attention-kernel warning in the first forward)"]
    return [f"train_ft: sdpa-mask-warning: {t}"[:300] for t in seen]
