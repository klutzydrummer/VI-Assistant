"""decide stage: decision models answer, for every line of hers, one yes/no
question per mood in moods.toml and one rating per knob, as probabilities.
Runs only after Brandon has seen the exact command and said ok to that run.

Plan: docs/design/corpus-pipeline.md, "Seeded moods". Like textscore, the
scores only nominate lines for Brandon to hear; they are never labels, and the
bake-off against his gold lines runs at home.

A decision model reads a state (context's `prompt`: the scene, the lines
around hers, hers marked >>>) and typed questions, and returns a probability
per option in one forward pass, writing no text. Two, both on Qwen3.5-9B:

  kev    Kev-9B (jaredpalmer/kev-9b): a LoRA and a pointer head, run through
         kev's own in-process server (batched, state-prefix cache).
  clef   Clef-flash (Cloudflare/clef-flash): its release loader and
         `systemone`, one line at a time.

Both run on the worker image's own torch 2.8 / CUDA 12.8 (no venv, no second
torch). Qwen3.5's Gated DeltaNet layers need two kernel packages, which
transformers uses when they import (requirements-decide.txt): causal-conv1d
(a prebuilt wheel for torch 2.8 / CUDA 12) and flash-linear-attention's gated
delta rule (Triton, compiled at first use with the gcc the boot script
installs). Hub kernels would need an explicit kernelize(), and the Hub's fla
build isn't public. Without the packages transformers silently runs a slow
PyTorch reference, so on the GPU this job stops if transformers logs that
fallback (`strict_kernels`).
Kev runs its documented unfused path (fused=False, no CUDA graphs): its fused
kernels were measured against a newer Triton than torch 2.8 ships.

Each model runs in its own process, so its GPU memory is freed before the
next, and checkpoints every chunk.

Outputs: scores/<model>.jsonl ({id, scores}: each mood's p(yes), and
`knob:level` probabilities, the keys textscore uses), questions.json (exactly
what was asked), decide.json, checks.json.

Settings: backend ("fake" for tests), models (default kev, clef), kev / clef
({repo, revision}), kev_source (kev's code, an archive URL at a commit),
character, strict_kernels, chunk, device, moods, limit, checks.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import checks
from textscore import free, load_lines, load_moods, pinned
from vi_job import Checkpoint, ensure_packages, fake_models, log, out_dir, settings, unit_hash, write_jsonl

MODELS = {
    "kev": {"repo": "jaredpalmer/kev-9b", "revision": None},
    "clef": {"repo": "Cloudflare/clef-flash", "revision": None},
}
FALLBACK = "falling back to its reference PyTorch implementation"


def questions(moods, character):
    """The typed questions every line gets, keyed by mood or knob id."""
    q = {}
    for m in moods.moods:
        q[m["id"]] = {"type": "noul", "instructions": f"In the line marked >>>, is {character} {m['description']}?",
                      "criteria": {"true": "yes", "false": "no"}}  # fmt: skip
    for k in moods.knobs:
        q[k["id"]] = {"type": "score", "instructions": f"For the line marked >>>, rate {k['description']}.",
                      "criteria": [lv["description"] for lv in k["levels"]]}  # fmt: skip
    return q


def scores(moods, answers):
    """A model's answers -> {mood: p(yes), "knob:level": p}."""
    out = {m["id"]: round(float(answers[m["id"]]["noul"]), 4) for m in moods.moods}
    for k in moods.knobs:
        p = answers[k["id"]]["probabilities"]
        for i, lv in enumerate(k["levels"]):
            out[f"{k['id']}:{lv['id']}"] = round(float(p[str(i)]), 4)
    return out


class Fallbacks(logging.Handler):
    """Catches transformers' warning that a Qwen3.5 kernel fell back to PyTorch."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def emit(self, record):
        msg = record.getMessage()
        if FALLBACK in msg:
            self.seen.append(msg)


def watch_kernels():
    handler = Fallbacks()
    import transformers

    transformers.logging.get_logger().addHandler(handler)
    return handler


def check_kernels(handler, strict):
    if handler.seen and strict:
        sys.exit("decide: a Qwen3.5 kernel fell back to PyTorch, so this would crawl: " + " | ".join(handler.seen))


class Kev:
    def __init__(self, cfg, device):
        import torch
        from kev.checkpoint import Checkpoint as KevCheckpoint
        from kev.checkpoint import LoadOptions
        from kev.predictors import kernel_environment
        from kev.serve import Server

        run = cfg["repo"] + (f"@{cfg['revision']}" if cfg.get("revision") else "")
        ck = KevCheckpoint(run)
        opts = LoadOptions(dtype=torch.bfloat16 if device == "cuda" else torch.float32, fused=False, cuda_graphs=False)
        tok, model = ck.load(device, opts)
        self.server = Server(ck, tok, model, device)
        self.environment = kernel_environment(model, device)

    def __call__(self, states, qs):
        from kev.api import SystemOneRequest, to_record, to_answers

        jobs = []
        for state in states:
            rec, meta = to_record(SystemOneRequest.model_validate({"state": state, "questions": qs}))
            jobs.append((self.server.submit(rec), meta))  # queued together, so the model thread batches them
        return [to_answers(future.result()[0], meta) for future, meta in jobs]

    def close(self):
        self.server.close()


class Clef:
    def __init__(self, cfg, device):
        from huggingface_hub import snapshot_download

        path = snapshot_download(cfg["repo"], revision=cfg.get("revision"))
        sys.path.insert(0, path)
        import joint_schema_model

        self.lib = joint_schema_model
        self.model, self.processor = joint_schema_model.load_release_model(path, device=device)
        self.environment = {"device": device}

    def __call__(self, states, qs):
        return [self.lib.systemone(self.model, self.processor, {"model": "clef-flash", "state": s, "questions": qs})["answers"]
                for s in states]  # fmt: skip

    def close(self):
        pass


class Fake:
    """Deterministic answers in the models' shape, for the plumbing tests."""

    environment = {"device": "fake"}

    def __init__(self, name):
        self.name = name

    def __call__(self, states, qs):
        out = []
        for s in states:
            a = {}
            for qid, q in qs.items():
                if q["type"] == "noul":
                    a[qid] = {"type": "noul", "noul": round(unit_hash(self.name, s, qid), 4)}
                else:
                    w = [unit_hash(self.name, s, qid, str(i)) + 1e-3 for i in range(len(q["criteria"]))]
                    a[qid] = {"type": "score", "probabilities": {str(i): x / sum(w) for i, x in enumerate(w)}}
            out.append(a)
        return out

    def close(self):
        pass


def device():
    """The GPU, unless `device = "cpu"` (a small model, for checking the
    adapters at home)."""
    import torch

    want = settings().get("device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        sys.exit("decide: no CUDA GPU here; it would run on the CPU for hours")
    return want


def model_pass(name):
    """One model over every line, in its own process."""
    s = settings()
    out = out_dir()
    moods, lines = load_moods(), load_lines()
    qs = questions(moods, s.get("character", "Tali"))
    cfg = {**MODELS[name], **s.get(name, {})}
    handler = None if fake_models() else watch_kernels()
    strict = s.get("strict_kernels", True) and s.get("device", "cuda") == "cuda"
    started = time.time()
    model = Fake(name) if fake_models() else (Kev if name == "kev" else Clef)(cfg, device())
    loaded = time.time()
    saved = Checkpoint(name)
    first = [True]

    def compute(part):
        answers = model([ln.prompt for ln in part], qs)
        if first[0] and handler is not None:
            check_kernels(handler, strict)  # after the first chunk, which ran every kernel
            first[0] = False
        return [scores(moods, a) for a in answers]

    done = saved.run(lines, compute, chunk=int(s.get("chunk", 64)))
    model.close()
    (out / "scores").mkdir(exist_ok=True)
    write_jsonl(out / "scores" / f"{name}.jsonl", [{"id": ln.id, "scores": done[ln.id]} for ln in lines])
    report = {**pinned(cfg["repo"], {cfg["repo"]: cfg.get("revision")}, fake_models()),
              "lines": len(done), "load_s": round(loaded - started, 1), "score_s": round(time.time() - loaded, 1),
              "environment": model.environment, "kernel_fallbacks": handler.seen if handler else []}  # fmt: skip
    (out / f"{name}.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    del model
    free()


def install():
    """requirements-decide.txt on top of requirements-gpu.txt, then kev's code
    itself without its dependencies: kev declares Python 3.12 and numpy 2.5.3,
    but runs on the image's 3.11 (its inference code needs neither)."""
    ensure_packages()
    ensure_packages("requirements-decide.txt")
    if fake_models() or os.environ.get("VI_WORKER") != "1":
        return
    try:
        import kev  # noqa: F401
    except ImportError:
        src = settings()["kev_source"]
        log(f"installing kev from {src}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python", src],
                       check=True)  # fmt: skip


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--model":
        return model_pass(sys.argv[2])
    s = settings()
    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "60")  # a dead download fails in a minute, not an hour
    install()
    out = out_dir()
    moods, lines = load_moods(), load_lines()
    qs = questions(moods, s.get("character", "Tali"))
    (out / "questions.json").write_text(json.dumps(qs, indent=1) + "\n")
    chosen = s.get("models", list(MODELS))
    log(f"decide: {len(lines)} lines, {len(qs)} questions, models {', '.join(chosen)}")
    started = time.time()
    for name in chosen:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model", name], check=True)
    reports = {n: json.loads((out / f"{n}.json").read_text()) for n in chosen}
    rows = {n: {r["id"]: r["scores"] for r in map(json.loads, (out / "scores" / f"{n}.jsonl").read_text().splitlines())}
            for n in chosen}  # fmt: skip
    report = {"lines": len(lines), "questions": list(qs), "models": reports, "seconds": round(time.time() - started, 1)}
    (out / "decide.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    results = [checks.Check(f"lines {n} missed", len(lines) - len(rows[n]), 0) for n in chosen]
    results += [checks.Check(f"{n}: kernel fallbacks", len(reports[n]["kernel_fallbacks"]), 0) for n in chosen]
    checks.finish(out, results, s.get("checks", "warn"))
    log(f"decide: done in {report['seconds']} s")


if __name__ == "__main__":
    main()
