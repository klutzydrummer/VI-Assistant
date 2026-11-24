"""Stand-ins for the GPU side of the OmniVoice fine-tune, for jobs/rehearsal/run.sh.

The rehearsal drives the REAL home and worker scripts (on-worker.sh, worker-preflight.sh,
worker-return.sh, worker-job.sh, voicetune-fresh.sh, fetch_base_model.py, checkpoints.py,
omnivoice_abort.py) on this machine. Only what needs a GPU, the network or real data is
replaced, and everything here is synthetic: sine tones and numbered sentences, never game audio
or text.

    python3 stubs.py wheelhouse DIR   stub wheels the real `pip install -r` installs offline
    python3 stubs.py fakebin DIR      nvidia-smi (24467 MiB by default) and an rsync fault shim
    python3 stubs.py data DIR         a synthetic data folder in the omnivoice_data layout

Installed into the worker venv, this same file is the `vi_rehearsal_stub` module. Its .pth hook
runs at interpreter start: when argv[0] is train_ft.py, replay_public.py or infer_listening.py the
stub runs instead of the real script (which imports torch). Each stub parses its arguments with
the option table read from the real script (ast, nothing imported), so a flag the shell scripts
pass but the real script lacks fails here too, and the trainer writes and verifies checkpoints
with train_ft.py's own manifest and verify functions.

Knobs (environment):
  VI_STUB_VRAM_MIB          nvidia-smi memory.total (default 24467, a 24 GB card)
  VI_STUB_CUDA              nvidia-smi CUDA Version (default 12.8)
  VI_STUB_HF_FAIL=1         the Base download raises
  VI_STUB_STEP_SECONDS      seconds per trainer step for long runs (default 0.02); runs of at
                            most 50 steps (probe, kill/resume rehearsal) use 0.25 s so the
                            rehearsal's kill lands between checkpoint-10 and checkpoint-20
  VI_STUB_WEIGHTS_KB        size of each checkpoint's model.safetensors (default 2048)
  VI_STUB_CRASH_AT_STEP     the trainer exits 1 once at that step (a marker keeps it to once)
  VI_STUB_ENOSPC_AT_STEP    the trainer prints "No space left on device" and exits 1 there
  VI_STUB_NAN_AT_STEP       the trainer logs a NaN loss there (the abort rule must stop it)
  VI_STUB_TRACE             file the trainer appends one start event to per launch (pid, step,
                            resume source); default stub-trace.jsonl in the run dir
  VI_REHEARSAL_FAULTS       directory of rsync fault triggers (see the fakebin rsync shim)
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import tarfile
import time
import wave
import zipfile
from typing import Any, Dict, List, Optional, Tuple

STUB_VERSION = "0.0.0+rehearsal"
INTERCEPT = {"train_ft.py", "replay_public.py", "infer_listening.py"}
SYNTH_NOTE = ("SYNTHETIC REHEARSAL STUB: these files are sine tones from a stub, not a voice. "
              "For a real run, human listening decides voice quality; there is no automatic score.")


# ---------------------------------------------------------------------------
# The real scripts, read without importing them
# ---------------------------------------------------------------------------


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def real_parser(script: str) -> argparse.ArgumentParser:
    """An ArgumentParser with the real script's options (names, actions, nargs, required,
    choices and int/float types), taken from its add_argument calls."""
    tree = ast.parse(open(script, encoding="utf-8").read(), script)
    ap = argparse.ArgumentParser(prog=os.path.basename(script))
    seen = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not names or names[0] in seen:
            continue
        seen.add(names[0])
        kw: Dict[str, Any] = {}
        for k in node.keywords:
            if k.arg in ("action", "nargs", "required", "choices", "default"):
                v = _literal(k.value)
                if v is not None or k.arg == "default":
                    kw[k.arg] = v
            elif k.arg == "type" and isinstance(k.value, ast.Name) and k.value.id in ("int", "float"):
                kw["type"] = {"int": int, "float": float}[k.value.id]
        if kw.get("action") in ("store_true", "store_false"):
            kw.pop("type", None)
            kw.pop("nargs", None)
            kw.pop("choices", None)
        ap.add_argument(*names, **kw)
    return ap


def real_checkpoint_contract(script: str) -> Dict[str, Any]:
    """train_ft.py's CHECKSUM_NAME, REQUIRED_CHECKPOINT_FILES, write_checkpoint_manifest and
    verify_checkpoint, executed from its own source (all stdlib)."""
    tree = ast.parse(open(script, encoding="utf-8").read(), script)
    want = {"CHECKSUM_NAME", "REQUIRED_CHECKPOINT_FILES", "_sha256", "_relative_files",
            "write_checkpoint_manifest", "verify_checkpoint"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in want for t in targets):
                body.append(node)
    ns: Dict[str, Any] = {"hashlib": hashlib, "os": os, "json": json,
                          "Dict": Dict, "List": List, "Any": Any, "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.Module(body=body, type_ignores=[]), script, "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise SystemExit(f"stub: {script} no longer defines {sorted(missing)}")
    return ns


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------


def write_tone(path: str, freq: float, seconds: float = 0.5, rate: int = 16000):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = int(seconds * rate)
    frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def _log(msg: str):
    print(f"stub: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# train_ft.py stand-in
# ---------------------------------------------------------------------------

ARCH_KEYS = ("style_n_vectors", "style_quantizer", "target_frames", "crop_frames", "init_from")


def stub_train(script: str, argv: List[str]) -> int:
    a = real_parser(script).parse_args(argv)
    c = real_checkpoint_contract(script)
    if a.verify_checkpoint:
        reports = [c["verify_checkpoint"](p) for p in a.verify_checkpoint]
        print(json.dumps(reports, indent=1))
        return 0 if all(r["ok"] for r in reports) else 1
    if not a.output_dir:
        raise SystemExit("--output-dir is required")
    if not a.data_config:
        raise SystemExit("a real run needs --data-config (or use --synthetic)")
    # What the real loader would open first: the Base snapshot and every data.lst the data
    # config names (paths relative to the working directory, the data folder).
    if not os.path.isfile(os.path.join(a.init_from, "config.json")):
        raise SystemExit(f"stub trainer: no Base model at {a.init_from} (config.json missing)")
    cfg = json.load(open(a.data_config))
    for split in ("train", "dev"):
        for entry in cfg.get(split, []):
            for lst in entry["manifest_path"]:
                if not (os.path.isfile(lst) and os.path.getsize(lst) > 0):
                    raise SystemExit(f"stub trainer: {split} shard list {os.path.abspath(lst)} is missing or empty; "
                                     "the shards were not built in this data folder")
    start = 0
    if a.resume_from_checkpoint:
        old = json.load(open(os.path.join(a.resume_from_checkpoint, "ft_config.json")))
        bad = {k: (old.get(k), getattr(a, k)) for k in ARCH_KEYS if old.get(k) != getattr(a, k)}
        if bad:
            raise SystemExit(f"resume config mismatch: {bad}")
        report = c["verify_checkpoint"](a.resume_from_checkpoint, strict_files=True)
        if not report["ok"]:
            raise SystemExit(f"refusing to resume from an unverifiable checkpoint: {report['problems']}")
        start = json.load(open(os.path.join(a.resume_from_checkpoint, "train_config.json")))["global_step"]
    os.makedirs(a.output_dir, exist_ok=True)
    ft_config = {k: getattr(a, k) for k in ARCH_KEYS}
    ft_config.update({"guard": not a.no_collapse_guard, "guard_patience": a.guard_patience,
                      "steps": a.steps, "save_steps": a.save_steps, "stub": True})
    with open(os.path.join(a.output_dir, "ft_config.json"), "w") as f:
        json.dump(ft_config, f, indent=1)
    guard_path = os.path.join(a.output_dir, "collapse_guard.jsonl")
    if os.path.exists(guard_path):
        os.remove(guard_path)
    trace = os.environ.get("VI_STUB_TRACE") or os.path.join(a.output_dir, "stub-trace.jsonl")
    with open(trace, "a") as f:
        f.write(json.dumps({"event": "start", "pid": os.getpid(), "start_step": start,
                            "resume": a.resume_from_checkpoint, "time": time.time()}) + "\n")
    _log(f"trainer pid {os.getpid()} starting at step {start}"
         + (f" (resumed from {a.resume_from_checkpoint})" if a.resume_from_checkpoint else ""))

    env = os.environ
    step_s = float(env.get("VI_STUB_STEP_SECONDS", "0.02")) if a.steps > 50 else 0.25
    crash_at = int(env.get("VI_STUB_CRASH_AT_STEP", "0") or 0)
    enospc_at = int(env.get("VI_STUB_ENOSPC_AT_STEP", "0") or 0)
    nan_at = int(env.get("VI_STUB_NAN_AT_STEP", "0") or 0)
    weights = int(env.get("VI_STUB_WEIGHTS_KB", "2048")) * 1024
    metrics = open(os.path.join(a.output_dir, "style_metrics.jsonl"), "a")
    guard = open(guard_path, "a")
    for step in range(start + 1, a.steps + 1):
        time.sleep(step_s)
        marker = os.path.join(a.output_dir, f".stub-crashed-{crash_at}")
        if crash_at and step == crash_at and not os.path.exists(marker):
            open(marker, "w").close()
            _log(f"injected trainer crash at step {step}")
            print("RuntimeError: stub: injected trainer crash", file=sys.stderr, flush=True)
            return 1
        if enospc_at and step == enospc_at:
            print("OSError: [Errno 28] No space left on device (stub: injected)", file=sys.stderr, flush=True)
            return 1
        loss = float("nan") if (nan_at and step >= nan_at) else 4.0 * math.exp(-step / 2000) + 0.5
        if step % max(1, a.logging_steps) == 0 or step == a.steps:
            metrics.write(json.dumps({"micro_step": step, "attempt": 0, "loss": loss, "lr": a.lr}) + "\n")
            metrics.flush()
        if not a.no_collapse_guard and step >= a.guard_warmup and step % a.guard_every == 0:
            guard.write(json.dumps({"attempt": 0, "seed": a.seed, "step": step, "sep_ratio": 1.2,
                                    "collapsed": False, "strikes": 0}) + "\n")
            guard.flush()
        if step % a.save_steps == 0 or step == a.steps:
            ck = os.path.join(a.output_dir, f"checkpoint-{step}")
            os.makedirs(ck, exist_ok=True)
            with open(os.path.join(ck, "model.safetensors"), "wb") as f:
                f.write(os.urandom(weights))
            for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                with open(os.path.join(ck, name), "wb") as f:
                    f.write(os.urandom(4096))
            json.dump({"stub": True}, open(os.path.join(ck, "config.json"), "w"))
            json.dump({"stub": True}, open(os.path.join(ck, "tokenizer_config.json"), "w"))
            json.dump({"global_step": step}, open(os.path.join(ck, "train_config.json"), "w"))
            json.dump(ft_config, open(os.path.join(ck, "ft_config.json"), "w"), indent=1)
            c["write_checkpoint_manifest"](ck)  # last, as the real checkpoint hook does
            _log(f"saved checkpoint-{step}")
    print(json.dumps({"output_dir": a.output_dir, "global_step": a.steps, "attempts": 1}))
    return 0


# ---------------------------------------------------------------------------
# replay_public.py and infer_listening.py stand-ins
# ---------------------------------------------------------------------------


def stub_replay(script: str, argv: List[str]) -> int:
    a = real_parser(script).parse_args(argv)
    if not os.path.isfile(os.path.join(a.base_model, "config.json")):
        raise SystemExit(f"stub replay: no Base model at {a.base_model}")
    if not os.path.isdir(a.voices):
        raise SystemExit(f"stub replay: no voices folder {a.voices}")
    texts = [json.loads(l)["text"] for l in open(a.texts) if l.strip()]
    os.makedirs(os.path.join(a.out_dir, "wav"), exist_ok=True)
    rows = []
    for i, text in enumerate(texts):
        wav = os.path.join(a.out_dir, "wav", f"replay-{i:04d}.wav")
        write_tone(wav, 330 + 20 * i)
        rows.append({"id": f"replay-{i:04d}", "audio_path": wav, "text": text,
                     "language_id": "en", "replay": True})
    with open(os.path.join(a.out_dir, "replay_public.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    _log(f"replay set: {len(rows)} synthetic rows on device {a.device}")
    return 0


def stub_infer(script: str, argv: List[str]) -> int:
    a = real_parser(script).parse_args(argv)
    c = real_checkpoint_contract(os.path.join(os.path.dirname(script), "train_ft.py"))
    report = c["verify_checkpoint"](a.checkpoint, strict_files=True)
    if not report["ok"]:
        raise SystemExit(f"refusing an unverifiable checkpoint: {report['problems']}")
    if a.base_model and not os.path.isfile(os.path.join(a.base_model, "config.json")):
        raise SystemExit(f"stub infer: no Base model at {a.base_model}")
    styles = [json.loads(l)["id"] for l in open(a.style_manifest) if l.strip()][: a.max_styles or None]
    lines = list(range(a.max_lines or 2))
    step = os.path.basename(a.checkpoint.rstrip("/")).rpartition("-")[2]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in styles:
        for n in lines:
            name = f"{s}__line{n}__s0__step{step}"
            write_tone(os.path.join(a.out, name + ".wav"), 440 + 10 * n, 0.2)
            rows.append({"name": name, "style": s, "line": n, "seed": 0, "step": step, "stub": True})
    with open(os.path.join(a.out, "listening.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(a.out, "README.txt"), "w") as f:
        f.write(SYNTH_NOTE + "\n")
    _log(f"listening set: {len(rows)} stub clips from checkpoint-{step} into {a.out}")
    return 0


def intercept():
    """The .pth hook: run the stub instead of a real GPU script, then exit."""
    script = sys.argv[0] if sys.argv else ""
    name = os.path.basename(script)
    if name not in INTERCEPT or not os.path.isfile(script):
        return
    fn = {"train_ft.py": stub_train, "replay_public.py": stub_replay, "infer_listening.py": stub_infer}[name]
    try:
        rc = fn(os.path.abspath(script), sys.argv[1:])
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            rc = 1
        else:
            rc = e.code or 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


# ---------------------------------------------------------------------------
# Package stand-ins: accelerate, omnivoice's extract_audio_tokens, huggingface_hub
# ---------------------------------------------------------------------------


def accelerate_main():
    """`accelerate launch [--opt VALUE]... SCRIPT ARGS`: run SCRIPT as a child of this interpreter,
    the way the real accelerate 1.16.0 launcher does (r23; measured on the home CPU venv):
      - the child's stdout passes through; its stderr is shown live and also kept;
      - a non-zero child exit raises CalledProcessError chained from RuntimeError(<child stderr>),
        so the child's stderr appears twice and the launcher exits 1 (a trainer's 76 becomes 1);
      - SIGTERM or SIGINT to the launcher is passed to the child, then the launcher dies of it;
        a SIGKILL of the launcher leaves the child running (it is not passed on).
    The rehearsal used to os.execv the script, which let exit 76 through unchanged and hid that
    worker-job.sh's OOM retry never fired behind the real launcher."""
    import signal
    import subprocess
    import threading

    args = sys.argv[1:]
    if not args or args[0] != "launch":
        raise SystemExit("stub accelerate: only `launch` is supported")
    args = args[1:]
    while args and args[0].startswith("--"):
        args = args[2:] if "=" not in args[0] else args[1:]
    if not args:
        raise SystemExit("stub accelerate: no script")
    cmd = [sys.executable] + args
    child = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    got = []

    def forward(signum, _frame):
        got.append(signum)
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)
    kept = []

    def pump():
        for line in iter(child.stderr.readline, b""):
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
            kept.append(line)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    rc = child.wait()
    t.join()
    if got:
        signal.signal(got[0], signal.SIG_DFL)
        os.kill(os.getpid(), got[0])
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd) from RuntimeError(
            b"".join(kept).decode("utf-8", "replace"))


# `python -m accelerate.commands.launch ...` in the stub accelerate wheel (and the tests' pylib).
ACCELERATE_LAUNCH_PY = ("import sys\nfrom vi_rehearsal_stub import accelerate_main\n"
                        "if __name__ == '__main__':\n    sys.argv[1:1] = ['launch']\n"
                        "    accelerate_main()\n")


def extract_audio_tokens_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--tar_output_pattern", required=True)
    ap.add_argument("--jsonl_output_pattern", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--samples_per_shard", type=int, default=1000)
    ap.add_argument("--skip_errors", action="store_true")
    ap.add_argument("--nj_per_gpu", type=int, default=1)
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.tokenizer_path, "config.json")):
        raise SystemExit(f"stub extract_audio_tokens: no tokenizer at {a.tokenizer_path}")
    rows = [json.loads(l) for l in open(a.input_jsonl) if l.strip()]
    lst = []
    for shard in range(0, len(rows), a.samples_per_shard):
        part = rows[shard: shard + a.samples_per_shard]
        tar_path = a.tar_output_pattern % (shard // a.samples_per_shard)
        jsonl_path = a.jsonl_output_pattern % (shard // a.samples_per_shard)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with tarfile.open(tar_path, "w") as tar:
            for r in part:
                if not os.path.isfile(r["audio_path"]):
                    if a.skip_errors:
                        continue
                    raise SystemExit(f"missing audio {r['audio_path']}")
                data = hashlib.sha256(open(r["audio_path"], "rb").read()).digest() * 8
                info = tarfile.TarInfo(f"{r['id']}.npy")  # as upstream's serialise_numpy writes it
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        with open(jsonl_path, "w") as f:
            f.writelines(json.dumps({"id": r["id"], "text": r["text"]}) + "\n" for r in part)
        readback_check(tar_path, {r["id"] for r in part})
        lst.append(f"{os.path.abspath(tar_path)} {os.path.abspath(jsonl_path)} {len(part)}")
    out_dir = os.path.dirname(os.path.dirname(a.tar_output_pattern))
    with open(os.path.join(out_dir, "data.lst"), "w") as f:
        f.writelines(l + "\n" for l in lst)
    _log(f"tokenised {len(rows)} rows into {out_dir}")
    # voicetune-fresh.sh reads upstream's summary line to learn the set's size.
    ok = sum(int(l.rsplit(" ", 1)[1]) for l in lst)
    print(f"INFO [extract_audio_tokens.py:607] Processing Complete - Successful: {ok}, "
          f"Failed: {len(rows) - ok}, Filtered/Skipped: 0, Shards written: {len(lst)}", flush=True)


_BASE_PLUS_EXT = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")  # webdataset 1.0.2 base_plus_ext


def readback_check(tar_path, label_ids):
    """What webdataset + omnivoice's LabelDataset do with the shard, reduced to the part that broke
    omnivoice-ft-r5: members group into samples by the text before the FIRST dot, and every sample
    needs an `npy` field and a label under that key. A dotted id fails here exactly as it did on
    the worker (KeyError on the cut key). Not emulated: tar I/O, shuffling, tensors."""
    samples = {}
    with tarfile.open(tar_path) as tar:
        for name in tar.getnames():
            m = _BASE_PLUS_EXT.match(name)
            key, suffix = (m.group(1), m.group(2)) if m else (name, "")
            samples.setdefault(key, {})[suffix] = name
    for key, fields in samples.items():
        if key not in label_ids:
            raise KeyError(f"stub webdataset: sample key {key!r} (members {sorted(fields.values())}) "
                           "has no label; ids must not contain '.'")
        if "npy" not in fields:
            raise KeyError(f"stub webdataset: sample {key!r} has no 'npy' field: {sorted(fields)}")
    if len(samples) != len(label_ids):
        raise KeyError(f"stub webdataset: {len(label_ids)} labels but {len(samples)} samples")


def snapshot_download(repo_id, revision=None, local_dir=None, **_):
    if os.environ.get("VI_STUB_HF_FAIL") == "1":
        raise OSError(f"stub huggingface_hub: {repo_id} is unreachable (injected)")
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k):
            raise SystemExit(f"stub huggingface_hub: {k} reached the download; it must be stripped")
    os.makedirs(os.path.join(local_dir, "audio_tokenizer"), exist_ok=True)
    json.dump({"stub": True}, open(os.path.join(local_dir, "audio_tokenizer", "config.json"), "w"))
    json.dump({"stub": True, "repo": repo_id, "revision": revision},
              open(os.path.join(local_dir, "config.json"), "w"))
    # fetch_base_model.py checks the pinned revision's weight sizes (SIZE_TABLE, TOTAL_BYTES).
    # Sparse files of exactly those sizes pass it while using no real disk; nothing reads them.
    for rel, size in (("model.safetensors", 2_450_344_112),
                      ("audio_tokenizer/model.safetensors", 805_665_628),
                      ("stub-padding.bin", 3_267_470_260 - 2_450_344_112 - 805_665_628 + 4096)):
        with open(os.path.join(local_dir, rel), "wb") as f:
            f.truncate(size)
    _log(f"snapshot_download {repo_id}@{revision} -> {local_dir}")
    return local_dir


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _norm(dist: str) -> str:
    """PEP 503 name normalisation (what pip compares), with `_` for a wheel file name."""
    return re.sub(r"[-_.]+", "-", dist).lower()


def worker_pins(path: Optional[str] = None) -> Dict[str, str]:
    """{normalised name: version} for every `name==version` line of the worker requirements
    (jobs/requirements-omnivoice-worker.txt), so the stub wheels satisfy the real pinned list."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "requirements-omnivoice-worker.txt")
    pins = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*==\s*([^\s;#]+)", line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def _wheel(out_dir: str, dist: str, files: Dict[str, str], requires=(), scripts=None,
           version: str = STUB_VERSION):
    norm = dist.replace("-", "_")
    info = f"{norm}-{version}.dist-info"
    meta = f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\nSummary: rehearsal stub\n"
    meta += "".join(f"Requires-Dist: {r}\n" for r in requires)
    files = dict(files)
    files[f"{info}/METADATA"] = meta
    files[f"{info}/WHEEL"] = "Wheel-Version: 1.0\nGenerator: vi-rehearsal\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    if scripts:
        files[f"{info}/entry_points.txt"] = "[console_scripts]\n" + "".join(f"{k} = {v}\n" for k, v in scripts.items())
    record = []
    for path, text in files.items():
        data = text.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{path},sha256={digest},{len(data)}")
    record.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = "\n".join(record) + "\n"
    with zipfile.ZipFile(os.path.join(out_dir, f"{norm}-{version}-py3-none-any.whl"), "w") as z:
        for path, text in files.items():
            z.writestr(path, text)


def build_wheelhouse(out_dir: str):
    """Offline wheels for the real `pip install -r requirements-omnivoice-worker.txt`. Each pinned
    package gets a wheel at its pinned version: the stubbed ones carry the stub code below, every
    other one is metadata only (installs, imports nothing; e.g. there is no torch stub)."""
    os.makedirs(out_dir, exist_ok=True)
    me = open(os.path.abspath(__file__), encoding="utf-8").read()
    stub = ["vi-rehearsal-stub"]
    pins = worker_pins()
    done = set()
    real_wheel = globals()["_wheel"]

    def _wheel(out_dir, dist, files, requires=(), scripts=None):  # noqa: F811 (pinned version)
        done.add(_norm(dist))
        real_wheel(out_dir, dist, files, requires, scripts, pins.get(_norm(dist), STUB_VERSION))
    _wheel(out_dir, "vi-rehearsal-stub", {
        "vi_rehearsal_stub.py": me,
        "vi_rehearsal_stub.pth": "import vi_rehearsal_stub; vi_rehearsal_stub.intercept()\n"})
    _wheel(out_dir, "accelerate", {
        "accelerate/__init__.py": "STUB = True\n",
        "accelerate/commands/__init__.py": "",
        # voicetune-fresh.sh runs `python -m accelerate.commands.launch ...` (no console script).
        "accelerate/commands/launch.py": ACCELERATE_LAUNCH_PY}, stub,
           {"accelerate": "vi_rehearsal_stub:accelerate_main"})
    _wheel(out_dir, "omnivoice", {
        "omnivoice/__init__.py": "STUB = True\n",
        "omnivoice/scripts/__init__.py": "",
        "omnivoice/scripts/extract_audio_tokens.py":
            "from vi_rehearsal_stub import extract_audio_tokens_main\n"
            "if __name__ == '__main__':\n    extract_audio_tokens_main()\n"},
        stub + ["huggingface_hub"])
    _wheel(out_dir, "huggingface_hub", {
        "huggingface_hub/__init__.py": "from vi_rehearsal_stub import snapshot_download\n"}, stub)
    for name in ("tensorboard", "webdataset", "soundfile"):
        _wheel(out_dir, name, {f"{name}/__init__.py": "STUB = True\n"})
    for name in sorted(set(pins) - done):
        _wheel(out_dir, name, {})


NVIDIA_SMI = r'''#!/usr/bin/env bash
# Rehearsal stand-in for nvidia-smi: one idle card with VI_STUB_VRAM_MIB MiB (default 24467,
# what a 24 GB card reports). Answers exactly the queries the worker scripts make.
mib=${VI_STUB_VRAM_MIB:-24467}
case "$*" in
  *--query-gpu=memory.total,memory.used*) echo "$mib, 0" ;;
  *--query-gpu=memory.total*) echo "$mib" ;;
  *--query-gpu=driver_version*) echo "550.54.15" ;;
  *--query-gpu=clocks.sm*)
    echo "clocks.sm [MHz], clocks.max.sm [MHz], power.draw [W], power.limit [W], clocks_throttle_reasons.active"
    echo "210 MHz, 1980 MHz, 21.50 W, 350.00 W, 0x0000000000000001" ;;
  "") printf '| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: %s |\n| stub card %s MiB |\n' "${VI_STUB_CUDA:-12.8}" "$mib" ;;
  *) echo "stub nvidia-smi: unsupported query: $*" >&2; exit 2 ;;
esac
'''

RSYNC_SHIM = r'''#!/usr/bin/env bash
# Rehearsal rsync shim: runs the real rsync ($VI_REAL_RSYNC) unless a fault trigger in
# $VI_REHEARSAL_FAULTS applies to a worker -> home push (an rsync:// URL):
#   receiver-down      every rsync:// push fails as if the port were closed (exit 10)
#   publish-partial    "checkpoint-N COUNT": a vi-publish push of a dir holding checkpoint-N
#                      sends it without model.safetensors and SHA256SUMS, then fails with
#                      exit 23 (partial transfer), COUNT times
real=${VI_REAL_RSYNC:?}
f=${VI_REHEARSAL_FAULTS:-}
url= src=
prev=
for a in "$@"; do
  case $a in rsync://*) url=$a; src=$prev ;; esac
  prev=$a
done
if [ -n "$f" ] && [ -n "$url" ]; then
  if [ -e "$f/receiver-down" ]; then
    echo "rsync: failed to connect to 127.0.0.1: Connection refused (stub: receiver-down)" >&2
    exit 10
  fi
  if [ -s "$f/publish-partial" ] && [[ $url == */vi-publish/ ]]; then
    exec 9>>"$f/.lock"; flock 9
    read -r ck left <"$f/publish-partial" || true
    if [ -n "$ck" ] && [ "${left:-0}" -gt 0 ] && [ -f "$src/$ck/SHA256SUMS" ]; then
      echo "$ck $((left - 1))" >"$f/publish-partial"
      echo "$(date +%s) publish-partial $ck" >>"$f/fired.log"
      "$real" "$@" --exclude="/$ck/model.safetensors" --exclude=SHA256SUMS || true
      echo "rsync error: some files could not be transferred (code 23) (stub: publish-partial $ck)" >&2
      exit 23
    fi
  fi
fi
exec "$real" "$@"
'''


def build_fakebin(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, body in (("nvidia-smi", NVIDIA_SMI), ("rsync", RSYNC_SHIM)):
        p = os.path.join(out_dir, name)
        with open(p, "w") as f:
            f.write(body)
        os.chmod(p, 0o755)


def build_data(out_dir: str, jobs_dir: str):
    """A synthetic folder in the jobs/omnivoice_data.py layout (its data_config() is reused)."""
    sys.path.insert(0, jobs_dir)
    sys.path.insert(0, os.path.join(jobs_dir, "omnivoice-ft"))
    import omnivoice_data  # noqa: E402  (stdlib + buildset/data_manifest; reads no corpus here)

    os.makedirs(out_dir, exist_ok=True)
    # Raw rows shaped like jobs/buildset.py's (real ids look like `le2.s178234_m`, dot included)
    # go through the real manifest builder, so a regression in its id sanitising reaches the
    # stub tokeniser's read-back check. The wav paths keep the dots, as the real ones do.
    import data_manifest  # noqa: E402

    def build(stem, game, n, base, first):
        raw = os.path.join(out_dir, f"{stem}_raw.jsonl")
        with open(raw, "w") as f:
            for i in range(n):
                cid = f"{game}.s{first + 37 * i}_m"
                write_tone(os.path.join(out_dir, f"wav/{cid}.wav"), base + 15 * i)
                f.write(json.dumps({"id": cid, "audio": f"wav/{cid}.wav",
                                    "text": f"Synthetic rehearsal sentence {stem} {i}."}) + "\n")
        name = f"{stem}_manifest.jsonl"
        data_manifest.build_manifest(raw_jsonls=[raw], audio_root=out_dir, out=os.path.join(out_dir, name))
        os.remove(raw)
        return omnivoice_data.relativise(os.path.join(out_dir, name), out_dir)
    train, dev = build("train", "le2", 12, 200, 178234), build("dev", "le3", 3, 600, 505116)
    with open(os.path.join(out_dir, "style_manifest.jsonl"), "w") as f:
        f.writelines(json.dumps({k: r[k] for k in ("id", "audio_path", "text")}) + "\n" for r in train[:6])
    for i in range(2):
        write_tone(os.path.join(out_dir, "voices", f"public-{i}.wav"), 300 + 50 * i)
    with open(os.path.join(out_dir, "replay_texts.jsonl"), "w") as f:
        f.writelines(json.dumps({"text": f"Synthetic replay line {i}."}) + "\n" for i in range(4))
    os.makedirs(os.path.join(out_dir, "audio_tokenizer"), exist_ok=True)
    open(os.path.join(out_dir, "audio_tokenizer", "PLACEHOLDER"), "w").write("synthetic rehearsal\n")
    with open(os.path.join(out_dir, "data_config.json"), "w") as f:
        json.dump(omnivoice_data.data_config(), f, indent=1)
    with open(os.path.join(out_dir, "SYNTHETIC.txt"), "w") as f:
        f.write("Synthetic rehearsal data: sine tones and numbered sentences. No game audio or text.\n")


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "wheelhouse":
        build_wheelhouse(out)
    elif cmd == "fakebin":
        build_fakebin(out)
    elif cmd == "data":
        build_data(out, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        raise SystemExit(__doc__)
