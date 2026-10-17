"""textscore stage: every text model scores every line of hers against every
mood in moods.toml, in one GPU pass. Runs only after Brandon has seen the
exact command and said ok to that run.

Plan: docs/design/corpus-pipeline.md, "Seeded moods". Format:
docs/design/corpus-format.md, textscore. The models only nominate: their
scores pick which lines Brandon hears for a mood, they are never labels.
Judging them against his gold lines happens at home afterwards, so this job
needs none of his labels and runs once.

Per line, from context's `prompt` (the line, the lines around it, the scene):

  goemotions   a RoBERTa classifier's 28 GoEmotions probabilities, on her
               line alone (it was trained on single comments)
  nli          a zero-shot NLI model's entailment probability for "In the
               marked line, Tali is <description>", per mood and knob level
  embedder     Qwen3-Embedding vectors of the line and of the line in
               context, and each one's cosine to every mood's description
  llm.<name>   each local LLM's moods (strength 1-3, with a short reason),
               and one level per knob, `samples` times, so home can measure
               how consistent it is. Constrained to moods.toml's ids by a JSON
               schema; thinking is off.

Each LLM runs in its own process, so its GPU memory is freed before the next.
The whole job runs in its own virtualenv (requirements-text.txt), since vLLM
pins its own torch and transformers.

Outputs: scores/<model>.jsonl ({id, scores}), llm/<name>.jsonl ({id,
samples}), embeddings/qwen3_{line,context,moods}.npy + .ids, textscore.json,
checks.json.

Settings: backend ("fake" for tests), models (passes to run, default all:
goemotions, nli, embedder, llm), goemotions, nli, embedder (repos), llms
({name: {repo, max_model_len, limit_mm_per_prompt}}), revisions, samples,
temperature, top_p, max_tokens, max_moods, batch, nli_batch, llm_chunk,
character, moods (path at home; on the worker files/moods.toml), limit.
"""

import json
import os
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import checks
from vi_job import (
    ArrayCheckpoint,
    Checkpoint,
    ensure_venv,
    env_path,
    fake_models,
    log,
    out_dir,
    read_jsonl,
    settings,
    spread,
    unit_hash,
    write_jsonl,
)

DEFAULTS = {
    "goemotions": "SamLowe/roberta-base-go_emotions",
    "nli": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "embedder": "Qwen/Qwen3-Embedding-4B",
}
LLMS = {
    # 8-bit weights, so each fits a 48 GB card with room for the KV cache.
    "gemma": {"repo": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "limit_mm_per_prompt": {"image": 0, "audio": 0}},
    "qwen": {"repo": "Qwen/Qwen3.6-27B-FP8", "limit_mm_per_prompt": {"image": 0, "video": 0}},
}
GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
    "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]  # fmt: skip
NLI_TEMPLATE = "In the marked line, {character} is {description}."
EMBED_TASK = "Given how a character feels, find lines of dialogue spoken that way"


@dataclass
class Line:
    id: str
    text: str
    prompt: str


@dataclass
class Moods:
    moods: list  # [{id, description, found}]
    knobs: list  # [{id, description, levels: [{id, description}]}]

    def labels(self):
        """(key, description) for every mood and knob level: what the NLI
        model and the embedder score. Knob levels are `knob:level`."""
        out = [(m["id"], m["description"]) for m in self.moods]
        out += [(f"{k['id']}:{lv['id']}", lv["description"]) for k in self.knobs for lv in k["levels"]]
        return out


def load_moods():
    s = settings()
    candidates = [Path("files/moods.toml")]
    if s.get("moods"):
        candidates.insert(0, Path(s["moods"]))
    for path in candidates:
        if path.exists():
            data = tomllib.loads(path.read_text())
            moods = Moods(data.get("mood", []), data.get("knob", []))
            ids = [m["id"] for m in moods.moods] + [k["id"] for k in moods.knobs]
            if len(set(ids)) != len(ids) or not moods.moods:
                sys.exit(f"{path}: mood and knob ids must be unique, and there must be moods")
            return moods
    sys.exit("no moods.toml: pass it with --put moods.toml")


def load_lines():
    folder = env_path("VI_INPUT_CONTEXT")
    if folder is None:
        sys.exit("textscore needs the context stage as an input")
    lines = [Line(r["id"], r["text"], r["prompt"]) for r in read_jsonl(folder / "context.jsonl")]
    n = settings().get("limit")
    return spread(lines, int(n)) if n else lines


# --- the LLM's question --------------------------------------------------------------------


def system_prompt(moods, character, max_moods):
    intro = (
        f"You read one line spoken by {character} in the Mass Effect games, with the conversation around it, "
        "and say which moods the line carries. Judge from the words and the situation only; you can't hear her. "
        f"A line can carry several moods, up to {max_moods}, or none. List only moods that are really there: "
        "strength 1 is a hint, 2 is clear, 3 is strong. Give each a reason of at most 20 words that points at the "
        "words or the situation."
    )
    lines = [intro, "", "Moods:"]
    lines += [f"- {m['id']}: {character} is {m['description']}" for m in moods.moods]
    lines += ["", "Pick one level for each of these:"]
    for k in moods.knobs:
        levels = "; ".join(f"{lv['id']} ({lv['description']})" for lv in k["levels"])
        lines.append(f"- {k['id']}, {k['description']}: {levels}")
    lines += ["", "Answer with JSON only."]
    return "\n".join(lines)


def user_prompt(line):
    return f"{line.prompt}\n\nJudge the marked line (>>>)."


def schema(moods, max_moods):
    props = {
        "moods": {
            "type": "array",
            "maxItems": max_moods,
            "items": {
                "type": "object",
                "properties": {
                    "mood": {"enum": [m["id"] for m in moods.moods]},
                    "strength": {"enum": [1, 2, 3]},
                    "reason": {"type": "string", "maxLength": 200},
                },
                "required": ["mood", "strength", "reason"],
                "additionalProperties": False,
            },
        }
    }
    for k in moods.knobs:
        props[k["id"]] = {"enum": [lv["id"] for lv in k["levels"]]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def parse(text, moods):
    """One sample's answer, checked against moods.toml, or None if it doesn't
    parse. A mood listed twice keeps its first entry."""
    try:
        r = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(r, dict) or not isinstance(r.get("moods"), list):
        return None
    known = {m["id"] for m in moods.moods}
    seen, out = set(), []
    for m in r["moods"]:
        if not isinstance(m, dict) or m.get("mood") not in known or m.get("strength") not in (1, 2, 3):
            return None
        if m["mood"] in seen:
            continue
        seen.add(m["mood"])
        out.append({"mood": m["mood"], "strength": m["strength"], "reason": str(m.get("reason", ""))[:200]})
    answer = {"moods": out}
    for k in moods.knobs:
        if r.get(k["id"]) not in {lv["id"] for lv in k["levels"]}:
            return None
        answer[k["id"]] = r[k["id"]]
    return answer


# --- models --------------------------------------------------------------------------------


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class GoEmotions:
    def __init__(self, repo, revision, batch):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.batch = torch, batch
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).to(device()).eval()
        self.names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def __call__(self, lines):
        torch, out = self.torch, []
        for start in range(0, len(lines), self.batch):
            part = [ln.text for ln in lines[start : start + self.batch]]
            enc = self.tok(part, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device())
            with torch.no_grad():
                probs = torch.sigmoid(self.model(**enc).logits).float().cpu().numpy()
            out += [{n: round(float(p), 4) for n, p in zip(self.names, row, strict=True)} for row in probs]
        return out


class NLI:
    def __init__(self, repo, revision, labels, character, batch):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.batch = torch, batch
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        dtype = torch.float16 if device() == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision, torch_dtype=dtype)
        self.model = self.model.to(device()).eval()
        label2id = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = label2id["entailment"]
        self.labels = labels
        self.hypotheses = [NLI_TEMPLATE.format(character=character, description=d) for _, d in labels]

    def __call__(self, lines):
        torch = self.torch
        pairs = [(ln.prompt, h) for ln in lines for h in self.hypotheses]
        probs = []
        for start in range(0, len(pairs), self.batch):
            part = pairs[start : start + self.batch]
            enc = self.tok([p for p, _ in part], [h for _, h in part], padding=True, truncation="only_first",
                           max_length=512, return_tensors="pt").to(device())  # fmt: skip
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.entail]
            probs += p.cpu().tolist()
        n = len(self.labels)
        return [{k: round(probs[i * n + j], 4) for j, (k, _) in enumerate(self.labels)} for i in range(len(lines))]


class Embedder:
    def __init__(self, repo, revision):
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs = {"torch_dtype": torch.bfloat16} if device() == "cuda" else {}
        self.model = SentenceTransformer(repo, revision=revision, device=device(), model_kwargs=kwargs)

    def documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def queries(self, texts):
        return self.model.encode([f"Instruct: {EMBED_TASK}\nQuery: {t}" for t in texts], normalize_embeddings=True)


class FakeGoEmotions:
    names = GOEMOTIONS

    def __init__(self, *args):
        pass

    def __call__(self, lines):
        return [{n: round(unit_hash("ge", ln.text, n), 4) for n in self.names} for ln in lines]


class FakeNLI:
    def __init__(self, repo, revision, labels, character, batch):
        self.labels = labels

    def __call__(self, lines):
        return [{k: round(unit_hash("nli", ln.prompt, k), 4) for k, _ in self.labels} for ln in lines]


class FakeEmbedder:
    def __init__(self, *args):
        pass

    def _vec(self, t):
        import numpy as np

        v = np.array([unit_hash("emb", t, str(i)) - 0.5 for i in range(8)], dtype=np.float32)
        return v / np.linalg.norm(v)

    def documents(self, texts):
        import numpy as np

        return np.stack([self._vec(t) for t in texts])

    def queries(self, texts):
        return self.documents(["q:" + t for t in texts])


# --- the LLM pass, in its own process ------------------------------------------------------


def llm_pass(name):
    """Runs one LLM over every line and writes llm/<name>.jsonl. Called as
    `textscore.py --llm <name>` by main()."""
    s = settings()
    fake = fake_models()
    out = out_dir()
    moods = load_moods()
    lines = load_lines()
    cfg = {**LLMS.get(name, {}), **s.get("llms", {}).get(name, {})}
    character = s.get("character", "Tali")
    max_moods = int(s.get("max_moods", 4))
    samples = int(s.get("samples", 2))
    system = system_prompt(moods, character, max_moods)
    shape = schema(moods, max_moods)
    generate = fake_generate(moods, samples) if fake else vllm_generate(cfg, s, shape, samples)

    def compute(part):
        texts = generate([[{"role": "system", "content": system}, {"role": "user", "content": user_prompt(ln)}]
                          for ln in part])  # fmt: skip
        results = []
        for row in texts:
            parsed = [parse(t, moods) for t in row]
            # The first unparseable answer, verbatim, to see what went wrong.
            results.append({"samples": parsed, "raw_failed": [t for t, p in zip(row, parsed) if p is None][:1]})
        return results

    done = Checkpoint(f"llm_{name}").run(lines, compute, chunk=int(s.get("llm_chunk", 256)))
    (out / "llm").mkdir(exist_ok=True)
    write_jsonl(out / "llm" / f"{name}.jsonl", [{"id": ln.id, **done[ln.id]} for ln in lines])
    (out / "llm" / f"{name}.prompt.txt").write_text(system + "\n")


def vllm_generate(cfg, s, shape, samples):
    from vllm import LLM, SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams

        constrained = {"structured_outputs": StructuredOutputsParams(json=shape)}
    except ImportError:  # vLLM before the rename
        from vllm.sampling_params import GuidedDecodingParams

        constrained = {"guided_decoding": GuidedDecodingParams(json=shape)}
    revision = s.get("revisions", {}).get(cfg["repo"])
    llm = LLM(model=cfg["repo"], revision=revision, max_model_len=int(cfg.get("max_model_len", 4096)),
              gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)),
              limit_mm_per_prompt=cfg.get("limit_mm_per_prompt"), seed=0)  # fmt: skip
    params = SamplingParams(n=samples, temperature=float(s.get("temperature", 0.7)), top_p=float(s.get("top_p", 0.95)),
                            max_tokens=int(s.get("max_tokens", 600)), seed=0, **constrained)  # fmt: skip

    def generate(conversations):
        outs = llm.chat(conversations, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        return [[c.text for c in o.outputs] for o in outs]

    return generate


def fake_generate(moods, samples):
    ids = [m["id"] for m in moods.moods]

    def generate(conversations):
        rows = []
        for conv in conversations:
            text = conv[-1]["content"]
            row = []
            for k in range(samples):
                pick = [ids[int(unit_hash(text, str(k), str(j)) * len(ids))] for j in range(2)]
                answer = {"moods": [{"mood": m, "strength": 2, "reason": "fake"} for m in dict.fromkeys(pick)]}
                for kn in moods.knobs:
                    answer[kn["id"]] = kn["levels"][int(unit_hash(text, kn["id"]) * len(kn["levels"]))]["id"]
                row.append(json.dumps(answer))
            rows.append(row)
        return rows

    return generate


# --- the job -------------------------------------------------------------------------------


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--llm":
        return llm_pass(sys.argv[2])
    s = settings()
    ensure_venv("requirements-text.txt")
    fake = fake_models()
    out = out_dir()
    moods = load_moods()
    lines = load_lines()
    character = s.get("character", "Tali")
    repos = {k: s.get(k, v) for k, v in DEFAULTS.items()}
    revisions = s.get("revisions", {})
    llms = {n: {**LLMS.get(n, {}), **c} for n, c in s.get("llms", {k: {} for k in LLMS}).items()}
    chosen = s.get("models", ["goemotions", "nli", "embedder", "llm"])
    batch = int(s.get("batch", 64))
    labels = moods.labels()
    log(f"textscore: {len(lines)} lines, {len(moods.moods)} moods, {len(moods.knobs)} knobs")
    started = time.time()
    (out / "scores").mkdir(exist_ok=True)
    counts = {}

    def classify(name, make, chunk):
        saved = Checkpoint(name)
        model = make() if saved.todo(lines) else None
        done = saved.run(lines, lambda part, m=model: m(part), chunk=chunk)
        write_jsonl(out / "scores" / f"{name}.jsonl", [{"id": ln.id, "scores": done[ln.id]} for ln in lines])
        counts[name] = len(done)
        del model
        free()

    if "goemotions" in chosen:
        repo = repos["goemotions"]
        classify("goemotions", lambda: FakeGoEmotions() if fake else GoEmotions(repo, revisions.get(repo), batch), 256)
    if "nli" in chosen:
        cls, repo, nli_batch = FakeNLI if fake else NLI, repos["nli"], int(s.get("nli_batch", 64))
        classify("nli", lambda: cls(repo, revisions.get(repo), labels, character, nli_batch), 64)

    if "embedder" in chosen:
        import numpy as np

        saved = ArrayCheckpoint("embedder")
        repo = repos["embedder"]
        model = FakeEmbedder() if fake else Embedder(repo, revisions.get(repo))

        def embed(part):
            a, b = model.documents([ln.text for ln in part]), model.documents([ln.prompt for ln in part])
            return [{"line": x, "context": y} for x, y in zip(a, b, strict=True)]

        vecs = saved.run(lines, embed, chunk=256)
        queries = np.asarray(model.queries([f"{character} is {d}" for _, d in labels]), dtype=np.float32)
        ids = [ln.id for ln in lines]
        for part in ("line", "context"):
            m = np.stack([np.asarray(vecs[i][part], dtype=np.float32) for i in ids])
            write_matrix(out / "embeddings", f"qwen3_{part}", ids, m)
        write_matrix(out / "embeddings", "qwen3_moods", [k for k, _ in labels], queries)
        ctx = np.stack([np.asarray(vecs[i]["context"], dtype=np.float32) for i in ids])
        sims = ctx @ queries.T
        write_jsonl(out / "scores" / "embedder.jsonl",
                    [{"id": i, "scores": {k: round(float(x), 4) for (k, _), x in zip(labels, row, strict=True)}}
                     for i, row in zip(ids, sims, strict=True)])  # fmt: skip
        counts["embedder"] = len(ids)
        del model
        free()

    llm_report = {}
    if "llm" in chosen:
        for name in llms:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--llm", name], check=True, env=os.environ)
            rows = read_jsonl(out / "llm" / f"{name}.jsonl")
            llm_report[name] = llm_summary(rows)
            counts[f"llm.{name}"] = len(rows)

    report = {
        "lines": len(lines),
        "labels": [k for k, _ in labels],
        "moods": moods.moods,
        "knobs": moods.knobs,
        "models": {k: pinned(repos[k], revisions, fake) for k in DEFAULTS if k in chosen},
        "llms": {n: {**c, **pinned(c["repo"], revisions, fake), **llm_report.get(n, {})} for n, c in llms.items()}
        if "llm" in chosen
        else {},
        "prompts": {"nli": NLI_TEMPLATE, "embedder": EMBED_TASK, "llm_system": "llm/<name>.prompt.txt"},
        "sampling": {
            "samples": int(s.get("samples", 2)),
            "temperature": float(s.get("temperature", 0.7)),
            "top_p": float(s.get("top_p", 0.95)),
            "thinking": False,
        },
        "counts": counts,
        "packages": packages(fake),
        "seconds": round(time.time() - started, 1),
    }
    (out / "textscore.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    results = [checks.Check(f"lines {k} missed", len(lines) - v, 0) for k, v in counts.items()]
    for name, r in llm_report.items():
        results.append(checks.Check(f"llm.{name}: share of samples that failed to parse", r["failed_share"], 0.02))
    checks.finish(out, results, s.get("checks", "warn"))
    log(f"textscore: done in {report['seconds']} s")


def llm_summary(rows):
    """Parse failures, and how often a line's samples agree on its mood set
    (a first look; home measures consistency per mood)."""
    total = sum(len(r["samples"]) for r in rows)
    failed = sum(s is None for r in rows for s in r["samples"])
    agree = [len({frozenset(m["mood"] for m in s["moods"]) for s in r["samples"]}) == 1
             for r in rows if len(r["samples"]) > 1 and all(s is not None for s in r["samples"])]  # fmt: skip
    return {"samples": total, "failed": failed, "failed_share": round(failed / total, 4) if total else 0.0,
            "same_moods_share": round(sum(agree) / len(agree), 3) if agree else None}  # fmt: skip


def write_matrix(folder, name, ids, m):
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{name}.npy", np.asarray(m, dtype=np.float16))
    (folder / f"{name}.ids").write_text("".join(f"{i}\n" for i in ids))


def pinned(repo, revisions, fake):
    """The repo, its pinned revision, and the commit that resolves to, so an
    unpinned run still records exactly what it used."""
    revision = revisions.get(repo)
    if fake:
        return {"repo": repo, "revision": revision, "resolved": None}
    from huggingface_hub import HfApi

    return {"repo": repo, "revision": revision, "resolved": HfApi().model_info(repo, revision=revision).sha}


def packages(fake):
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in ["numpy"] + (
        [] if fake else ["torch", "transformers", "vllm", "sentence-transformers", "huggingface-hub"]
    ):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def free():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
