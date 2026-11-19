"""Home real-library check: run the worker's real data path on CPU before any GPU is rented.

The CPU tests use a tiny synthetic model and a stub trainer, so they never touch the code the
worker actually runs (upstream ``omnivoice`` tokeniser, WebDataset shards, ``train_ft._build_data``).
That is how omnivoice-ft-r6 failed on a rented GPU with an AttributeError a home check would have
shown. This script runs that path on a tiny slice of the real clips:

  1. take a few rows of the real train and dev manifests (and two train rows as replay rows),
  2. tokenise them with upstream ``extract_audio_tokens`` on CPU into scratch shards,
  3. build the real tokenizer and call ``StyleShardProcessor`` directly (no swallowed errors),
  4. run ``train_ft._build_data`` and pull collated batches from the real loader,
  5. with ``--forward`` only: load the real model, attach the KL reference, run one forward and
     backward step (no optimizer) and check the style encoder received gradients.

Run it with the home CPU venv (see jobs/requirements-omnivoice-home-cpu.txt):

    nice -n 19 ~/.cache/vi/home-cpu-venv-<hash>/bin/python home_realpath_check.py \\
        --data <omnivoice_data out folder> --base <HF snapshot of k2-fsa/OmniVoice> \\
        --scratch <dir> [--forward] [--json]

Copyrighted audio is read in place and tokenised into ``--scratch`` (under ~/.cache); nothing is
copied into the repo or sent anywhere. No mood or other legacy label is read: the rows come from
the manifests in file order.

Not proven: CUDA, bf16, GPU memory at the real batch size, DDP, throughput, GPU replay generation,
``accelerator.save_state`` with an optimizer, the network, or how the voice sounds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "omnistyle"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Exit codes of this script (paid_run maps a non-zero exit to its own exit 13).
EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 3

MIN_RAM_MB = 4096
MIN_RAM_FORWARD_MB = 12288
MIN_SCRATCH_GB = 2.0
MARKER = ".realpath-scratch"
BATCH_TOKENS = 1024
BATCH_KEYS = {"input_ids", "labels", "audio_mask", "attention_mask", "position_ids",
              "style_slot", "style_tokens", "style_valid", "kl_mask"}

# The same upstream arguments voicetune-fresh.sh `tokenise()` uses; a test compares the two.
TOKENISE_FLAGS = ["--samples_per_shard", "1000", "--skip_errors"]


def mem_available_mb(meminfo: str = "/proc/meminfo") -> int:
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def refusal(forward: bool, scratch: Path, ram_mb: int | None = None, free_gb: float | None = None):
    """Why the check must not start, or None. ``forward`` raises the RAM bar from 4 GB to 12 GB."""
    need = MIN_RAM_FORWARD_MB if forward else MIN_RAM_MB
    ram_mb = mem_available_mb() if ram_mb is None else ram_mb
    if ram_mb < need:
        return (f"only {ram_mb} MB RAM is available, the {'forward step' if forward else 'check'} "
                f"needs {need} MB")
    if free_gb is None:
        scratch.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(scratch).free / 1e9
    if free_gb < MIN_SCRATCH_GB:
        return f"only {free_gb:.1f} GB free under {scratch}, the check needs {MIN_SCRATCH_GB} GB"
    return None


def read_rows(path: Path, n: int):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def slice_manifests(data: Path, scratch: Path, n_train: int, n_dev: int, n_replay: int):
    """Write scratch manifests: absolute audio paths, WebDataset-safe ids, replay rows marked."""
    import data_manifest as DM

    def prep(rows, replay=False, tag=""):
        out = []
        for r in rows:
            audio = Path(r["audio_path"])
            audio = audio if audio.is_absolute() else data / audio
            row = {"id": DM.sanitise_id(r["id"]) + tag, "audio_path": str(audio),
                   "text": r["text"], "language_id": r.get("language_id", "en")}
            if replay:
                row["replay"] = True
            out.append(row)
        return out

    train_rows = read_rows(data / "train_manifest.jsonl", n_train + n_replay)
    dev_rows = read_rows(data / "dev_manifest.jsonl", n_dev)
    if len(train_rows) < n_train + n_replay or len(dev_rows) < 1:
        raise RuntimeError(f"{data}: too few manifest rows ({len(train_rows)} train, "
                           f"{len(dev_rows)} dev)")
    sets = {
        "train": prep(train_rows[:n_train]),
        "dev": prep(dev_rows),
        # replay shards need GPU generation; real train clips stand in, ids kept distinct
        "replay": prep(train_rows[n_train:], replay=True, tag="_replay"),
    }
    paths = {}
    for name, rows in sets.items():
        p = scratch / f"{name}_manifest.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        paths[name] = p
    return paths


def tokenise(name: str, manifest: Path, scratch: Path, audio_tokenizer: Path):
    out = scratch / "shards" / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
           "--input_jsonl", str(manifest),
           "--tar_output_pattern", str(out / "audios" / "shard-%06d.tar"),
           "--jsonl_output_pattern", str(out / "txts" / "shard-%06d.jsonl"),
           "--tokenizer_path", str(audio_tokenizer), *TOKENISE_FLAGS]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if r.returncode != 0 or not (out / "data.lst").is_file() or (out / "data.lst").stat().st_size == 0:
        raise RuntimeError(f"tokenise {name} failed (exit {r.returncode}):\n{(r.stdout + r.stderr)[-2000:]}")
    return out


def write_data_config(scratch: Path):
    cfg = {
        "train": [{"manifest_path": ["shards/train/data.lst", "shards/replay/data.lst"], "repeat": 1}],
        "dev": [{"manifest_path": ["shards/dev/data.lst"], "repeat": 1}],
    }
    p = scratch / "data_config.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


def run_check(args) -> dict:
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    scratch = Path(args.scratch)
    reuse = args.reuse_shards and (scratch / "shards" / "replay" / "data.lst").is_file()
    marker = scratch / MARKER
    if scratch.exists() and not reuse:
        if any(scratch.iterdir()) and not marker.is_file():
            raise RuntimeError(f"{scratch} exists and was not made by this check; refusing to wipe it")
        shutil.rmtree(scratch)  # only a folder this check made (it carries the marker file)
    scratch.mkdir(parents=True, exist_ok=True)
    marker.write_text("made by home_realpath_check.py; safe to delete\n")
    os.environ["TMPDIR"] = str(scratch)
    os.chdir(scratch)  # the worker runs from its data folder; data_config.json paths are relative
    t0 = time.time()
    data, base = Path(args.data), Path(args.base)
    result = {"ok": False, "forward": bool(args.forward), "steps": []}

    def step(msg):
        result["steps"].append(msg)
        print(f"real-path: {msg} ({time.time() - t0:.0f}s)", flush=True)

    paths = slice_manifests(data, scratch, args.rows, args.dev_rows, args.replay_rows)
    step(f"sliced {args.rows} train, {args.dev_rows} dev, {args.replay_rows} replay clips")
    audio_tok = base / "audio_tokenizer"
    if reuse:
        step("reusing the scratch shards from the last run (--reuse-shards)")
    else:
        for name in ("train", "dev", "replay"):
            tokenise(name, paths[name], scratch, audio_tok)
        step("tokenised on CPU into scratch shards")
    cfg_path = write_data_config(scratch)

    import ft_data as FD
    import train_ft as TF

    cfg = TF.FTTrainConfig(
        output_dir=str(scratch / "run"), init_from=str(base), data_config=str(cfg_path),
        batch_tokens=BATCH_TOKENS, num_workers=0, mixed_precision="no",
    )
    if args.forward:
        model, tokenizer = TF.load_style_model(cfg)
        step("real model loaded")
    else:
        tokenizer, _ = TF.load_text_tokenizer(cfg)
        model = None
        step("real text tokenizer loaded (model not loaded; no --forward)")

    # (a) the processor on a raw decoded sample, with no try/except around it: the stream dataset
    # would swallow an error here and report only "no batches".
    from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json

    manifests, _ = prepare_data_manifests_from_json(str(cfg_path))
    proc = FD.StyleShardProcessor(tokenizer, n_style=cfg.style_n_vectors,
                                  target_frames=cfg.target_frames, crop_frames=cfg.crop_frames)
    processed, replay_seen = [], False
    for raw in WebDatasetReader(manifests=manifests, evaluation=True):
        out = proc(raw)
        processed.append(out)
        replay_seen |= bool(out["kl"])
        if len(processed) >= 2 * (args.rows + args.replay_rows):
            break
    if not processed:
        raise RuntimeError("WebDatasetReader returned no samples from the scratch shards")
    if not replay_seen:
        raise RuntimeError("no replay row reached the processor with kl=True")
    collated = FD.SD.StylePaddingCollator(proc, BATCH_TOKENS)(processed[: max(2, len(processed))])
    assert bool(collated["kl_mask"].any()), "replay rows did not set kl_mask"
    step(f"processor called directly on {len(processed)} raw samples; replay rows set kl")

    # (b) the real _build_data and loader.
    bundle = TF._build_data(cfg, 0, tokenizer)
    if not bundle.probe_crops:
        raise RuntimeError("probe pass produced no crops")
    batches = []
    for batch in bundle.loader:
        batches.append(batch)
        if len(batches) >= 3:
            break
    if not batches:
        raise RuntimeError("the real loader produced no batches")
    for b in batches:
        missing = BATCH_KEYS - set(b)
        if missing:
            raise RuntimeError(f"batch is missing keys {sorted(missing)}; has {sorted(b)}")
        if b["input_ids"].dim() != 3 or b["input_ids"].size(1) != 8:
            raise RuntimeError(f"input_ids shape {tuple(b['input_ids'].shape)}, expected [B, 8, L]")
        for k in ("input_ids", "labels", "style_tokens", "position_ids"):
            if b[k].dtype != torch.long:
                raise RuntimeError(f"{k} dtype {b[k].dtype}, expected torch.long")
        for k in ("audio_mask", "style_slot", "style_valid", "kl_mask"):
            if b[k].dtype != torch.bool:
                raise RuntimeError(f"{k} dtype {b[k].dtype}, expected torch.bool")
    shapes = {k: tuple(v.shape) for k, v in batches[0].items() if hasattr(v, "shape")}
    result["batches"] = len(batches)
    result["shapes"] = {k: list(v) for k, v in shapes.items()}
    result["probe_crops"] = len(bundle.probe_crops)
    step(f"real loader gave {len(batches)} batch(es); first batch shapes {shapes}")

    # (c) forward + backward, opt-in.
    if args.forward:
        batch = next((b for b in batches if bool(b["style_slot"].any())), batches[0])
        model.attach_reference(TF.StyleOmniVoice.reference_from(model))
        model.train()
        out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is not finite: {loss}")
        loss.backward()
        if bool(batch["style_slot"].any()):
            grads = [p.grad for p in model.style_encoder.parameters() if p.requires_grad]
            if not grads or all(g is None for g in grads):
                raise RuntimeError("style encoder got no gradients")
        result["loss"] = float(loss)
        step(f"forward + backward ok, loss {float(loss):.4f}, style encoder has gradients")

    result["ok"] = True
    result["seconds"] = round(time.time() - t0, 1)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="omnivoice_data out folder (read only)")
    ap.add_argument("--base", required=True, help="local HF snapshot of k2-fsa/OmniVoice")
    ap.add_argument("--scratch", required=True, help="throwaway folder under ~/.cache (wiped first)")
    ap.add_argument("--forward", action="store_true", help="also load the model; one forward+backward")
    ap.add_argument("--json", action="store_true", help="print the result as one JSON line")
    ap.add_argument("--reuse-shards", action="store_true", help="debugging: keep shards from the last run")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--dev-rows", type=int, default=2)
    ap.add_argument("--replay-rows", type=int, default=2)
    ap.add_argument("--code-hash", default="", help="stored in result.json for the caller's cache")
    a = ap.parse_args(argv)

    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))  # heavy home job: lowest priority, whoever started it
    scratch = Path(a.scratch)
    why = refusal(a.forward, scratch.parent if not scratch.exists() else scratch)
    if why:
        print(f"real-path check: REFUSED ({why})", flush=True)
        return EXIT_REFUSED
    try:
        result = run_check(a)
    except BaseException:  # noqa: BLE001  (report any failure, including SystemExit from upstream)
        tb = traceback.format_exc()
        result = {"ok": False, "forward": bool(a.forward), "error": tb[-3000:]}
        print(tb, file=sys.stderr, flush=True)
    result["code_hash"] = a.code_hash
    try:
        (scratch / "result.json").write_text(json.dumps(result, indent=1))
    except OSError:
        pass
    if result["ok"]:
        print(f"real-path check: OK ({result['batches']} batches, keys {sorted(BATCH_KEYS)}, "
              f"forward {'ok' if a.forward else 'skipped'})", flush=True)
    else:
        print("real-path check: FAIL (see the traceback above)", flush=True)
    if a.json:
        print(json.dumps(result), flush=True)
    return EXIT_OK if result["ok"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
