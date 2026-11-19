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
  6. with ``--save`` only (after ``--forward`` when both are given): the real ``train_ft.run_training``
     (real optimizer, KL reference, checkpoint hook, fp32 on CPU) for 1 step, save checkpoint-1,
     verify it strictly; a fresh process resumes it for 1 more step and saves checkpoint-2; the
     checkpoint reloads on CPU. Needs a lot of RAM (about 13-15 GB, run only above 18000 MB
     available, otherwise it reports "skipped (RAM)" and the save path is NOT proven) and free disk
     of the calculated checkpoint size plus 20 GB (otherwise it refuses, exit 3).

Run it with the home CPU venv (see jobs/requirements-omnivoice-home-cpu.txt):

    nice -n 19 ~/.cache/vi/home-cpu-venv-<hash>/bin/python home_realpath_check.py \\
        --data <omnivoice_data out folder> --base <HF snapshot of k2-fsa/OmniVoice> \\
        --scratch <dir> [--forward] [--json]

Copyrighted audio is read in place and tokenised into ``--scratch`` (under ~/.cache); nothing is
copied into the repo or sent anywhere. No mood or other legacy label is read: the rows come from
the manifests in file order.

Not proven: CUDA, bf16, GPU memory at the real batch size, DDP, throughput, GPU replay generation,
the network, or how the voice sounds; ``accelerator.save_state`` with an optimizer is proven only
when ``--save`` ran (not "skipped (RAM)").
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
MIN_RAM_SAVE_MB = 18000
# Base model.safetensors, bytes (fp32). A full checkpoint is the weights plus the two AdamW
# states of the same size (optimizer.bin) plus small files: CALCULATED, not measured, unless a
# real --save ran and recorded checkpoint_bytes.
BASE_MODEL_BYTES = 2_450_344_112
SAVE_DISK_MARGIN_GB = 20.0
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


def calc_checkpoint_bytes(base_bytes: int = BASE_MODEL_BYTES) -> int:
    """Calculated size of one full checkpoint: weights + two AdamW states (about 7.35 GB)."""
    return 3 * base_bytes


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


class SaveRefused(RuntimeError):
    """The save step must not write: the disk would be left with too little free space."""


def check_save_disk(path: Path, ckpt_bytes: int | None = None, free: int | None = None,
                    margin_gb: float = SAVE_DISK_MARGIN_GB) -> None:
    """Refuse when the filesystem of ``path`` has less than (checkpoint size + margin) free.

    Two checkpoints are written (checkpoint-1 and checkpoint-2), so the need is counted twice."""
    ckpt_bytes = calc_checkpoint_bytes() if ckpt_bytes is None else ckpt_bytes
    path = Path(path)
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = free_bytes(probe) if free is None else free
    need = 2 * ckpt_bytes + int(margin_gb * 1e9)
    if free < need:
        raise SaveRefused(f"only {free / 1e9:.1f} GB free under {probe}, the save step needs "
                          f"{need / 1e9:.1f} GB (two checkpoints of {ckpt_bytes / 1e9:.1f} GB "
                          f"plus a {margin_gb:.0f} GB margin)")


def dir_bytes(path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def prove_save(out_dir: Path, make_cfg, resume, reload, log=print) -> dict:
    """The save / resume / reload proof (F-CKPT-1), independent of the model size.

    ``make_cfg(steps, resume_from)`` returns the FTTrainConfig; ``resume(ckpt1)`` runs the second
    step in a fresh process (or in-process in a test); ``reload(ckpt2)`` loads it back on CPU."""
    import train_ft as TF

    out_dir = Path(out_dir)
    TF.run_training(make_cfg(1, None))
    ck1 = out_dir / "checkpoint-1"
    rep = TF.verify_checkpoint(str(ck1), strict_files=True)
    if not rep["ok"]:
        raise RuntimeError(f"checkpoint-1 failed verify_checkpoint: {rep['problems']}")
    size = dir_bytes(ck1)
    log(f"save: checkpoint-1 saved and verified ({size} bytes, measured)")
    resume(ck1)
    ck2 = out_dir / "checkpoint-2"
    rep = TF.verify_checkpoint(str(ck2), strict_files=True)
    if not rep["ok"]:
        raise RuntimeError(f"checkpoint-2 failed verify_checkpoint: {rep['problems']}")
    log("save: a fresh process resumed checkpoint-1 and saved a verified checkpoint-2")
    reload(ck2)
    log("save: checkpoint-2 reloads on CPU")
    return {"checkpoint_bytes": size, "checkpoint_bytes_label": "measured"}


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

    if args.save:
        model = None  # free the forward step's model before the heavier save run
        import gc
        gc.collect()
        ram = mem_available_mb()
        result["checkpoint_bytes_calculated"] = calc_checkpoint_bytes()
        if ram <= MIN_RAM_SAVE_MB:
            result["save"] = "skipped (RAM)"
            step(f"save: skipped (RAM): {ram} MB available, needs more than {MIN_RAM_SAVE_MB} MB; "
                 f"NOT PROVEN. Calculated checkpoint size {calc_checkpoint_bytes()} bytes "
                 "(3 x Base weights; calculated, not measured)")
        else:
            result.update(run_save(args, base, cfg_path, scratch, step))
            result["save"] = "ok"
    result["ok"] = True
    result["seconds"] = round(time.time() - t0, 1)
    return result


def _save_cfg(base: Path, cfg_path: Path, out: Path, steps: int, resume_from):
    import train_ft as TF

    return TF.FTTrainConfig(
        output_dir=str(out), init_from=str(base), data_config=str(cfg_path),
        batch_tokens=BATCH_TOKENS, num_workers=0, mixed_precision="no", steps=steps, save_steps=1,
        logging_steps=1, guard=False, resume_from_checkpoint=str(resume_from) if resume_from else None,
    )


def run_save(args, base: Path, cfg_path: Path, scratch: Path, step) -> dict:
    """Real-Base save proof. Resume runs in a fresh process via ``--resume-phase``."""
    out = scratch / "save-run"

    def resume(ck1):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--data", args.data, "--base", str(base),
               "--scratch", str(scratch), "--resume-phase", str(ck1)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"resume process failed (exit {r.returncode}):\n{(r.stdout + r.stderr)[-2000:]}")

    def reload(ck2):
        import torch
        from style_omnivoice import StyleOmniVoice

        m = StyleOmniVoice.from_pretrained(str(ck2), train=True, dtype=torch.float32)
        if not any(True for _ in m.parameters()):
            raise RuntimeError("reloaded model has no parameters")

    res = prove_save(out, lambda steps, rf: _save_cfg(base, cfg_path, out, steps, rf), resume, reload, log=step)
    step(f"save: checkpoint size {res['checkpoint_bytes']} bytes (measured)")
    return res


def run_resume_phase(args) -> int:
    """Second process of the save proof: resume checkpoint-1 for one more step (steps=2)."""
    import train_ft as TF

    scratch = Path(args.scratch)
    os.environ["TMPDIR"] = str(scratch)
    os.chdir(scratch)
    cfg = _save_cfg(Path(args.base), scratch / "data_config.json", scratch / "save-run", 2, args.resume_phase)
    TF.run_training(cfg)
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="omnivoice_data out folder (read only)")
    ap.add_argument("--base", required=True, help="local HF snapshot of k2-fsa/OmniVoice")
    ap.add_argument("--scratch", required=True, help="throwaway folder under ~/.cache (wiped first)")
    ap.add_argument("--forward", action="store_true", help="also load the model; one forward+backward")
    ap.add_argument("--save", action="store_true",
                    help="also prove save/resume/reload of a real checkpoint (needs >18000 MB RAM)")
    ap.add_argument("--resume-phase", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true", help="print the result as one JSON line")
    ap.add_argument("--reuse-shards", action="store_true", help="debugging: keep shards from the last run")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--dev-rows", type=int, default=2)
    ap.add_argument("--replay-rows", type=int, default=2)
    ap.add_argument("--code-hash", default="", help="stored in result.json for the caller's cache")
    a = ap.parse_args(argv)

    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))  # heavy home job: lowest priority, whoever started it
    if a.resume_phase:
        return run_resume_phase(a)
    scratch = Path(a.scratch)
    why = refusal(a.forward, scratch.parent if not scratch.exists() else scratch)
    if why:
        print(f"real-path check: REFUSED ({why})", flush=True)
        return EXIT_REFUSED
    if a.save:
        try:
            check_save_disk(scratch)
        except SaveRefused as e:
            print(f"real-path check: REFUSED ({e})", flush=True)
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
              f"forward {'ok' if a.forward else 'skipped'}, "
              f"save {result.get('save', 'not requested')})", flush=True)
    else:
        print("real-path check: FAIL (see the traceback above)", flush=True)
    if a.json:
        print(json.dumps(result), flush=True)
    return EXIT_OK if result["ok"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
