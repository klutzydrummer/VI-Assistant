"""steer stage: does activation steering move the TTS model's delivery, in her
cloned voice, better than swapping in an emotional reference clip?

The riskiest assumption in the voice plan (docs/design/control-vector-datasets.md,
step 7) is that a steering vector added to the TTS model's language-model
backbone gives audible, graded control of one axis without losing her voice or
the words. This stage tests it cheaply, for a few axes (anger and intoxication
by default), before the labelling effort grows. It's manual: it runs only with
`vi dataset run characters/tali/dataset.toml steer`.

For each axis and each source of contrast clips:
  tali      her own clips: the top `refs` on that axis in label's output (or
            the ids listed in settings.refs), against her most neutral ones.
  designed  clips the VoiceDesign model speaks from the same sentence twice,
            once instructed with the axis and once neutral, so each pair
            differs only in delivery.
the Base model clones each clip (in-context, with its words) while saying a
few capture sentences, and the vector per backbone layer is the mean hidden
state over generated audio tokens with the axis clips minus with the neutral
ones. Then it speaks the test sentences from her most neutral clip with the
vector added at `strengths` times, and, as the baseline to beat, from her top
clip on that axis with no vector.

Every generated line is scored for the axis (LAION Empathic-Insight-Voice;
emotion2vec+ "angry" too for anger), for voice (ECAPA cosine to her neutral
clips) and for words (Whisper's word error rate against the sentence). The
verdict per axis and source says whether the axis score rises with strength,
and the highest strength that keeps her voice and the words.

Outputs: report.json, report.md (the tables and verdicts), listen.md (which
file is which, for listening), audio/<source>/<axis>/<condition>/<n>.wav, and
vectors/<source>-<axis>.npy (layers x hidden). No subtitle text is written.

The Qwen3-TTS package pins an older transformers than the other jobs, so on
the worker this job runs in its own virtualenv (jobs/requirements-steer.txt).
Generated audio and vectors go to VI_CHECKPOINT as they're made, so a rerun
after a lost worker skips them.
"""

import json
import math
import os
import shutil
from pathlib import Path

import clean
import fusion
from vi_job import (
    ensure_venv,
    env_path,
    fake_models,
    load_clips,
    log,
    out_dir,
    read_jsonl,
    settings,
    unit_hash,
)

# Neutral lines for capturing and testing, written for this test (not from
# the games), so nothing copyrighted lands in the outputs.
CAPTURE = [
    "The supply shuttle docks at the third bay before the next shift.",
    "I checked the coolant lines twice and wrote down every reading.",
    "We should reach the relay in about four hours if nothing changes.",
]
TEST = [
    "I left the toolkit next to the engine console this morning.",
    "The message came in while you were talking to the captain.",
    "There are three crates of medical supplies in the cargo hold.",
    "I need to recalibrate the drive core before we jump again.",
    "The market on the lower deck closes at the end of the cycle.",
    "Someone moved my notes from the table in the mess hall.",
    "The scanner picked up a small ship near the second moon.",
    "We can talk about it after I finish these repairs.",
]
DESIGN = [
    "Do you have any idea what time it is right now?",
    "I told you the panel would come loose if you kicked it.",
    "We made it back to the ship, and everyone is still here.",
    "Give me a moment, I want to look at this one more time.",
    "That is not where the spare parts are supposed to go.",
    "You really think this is going to work out for us?",
]

DEFAULT_AXES = {
    "anger": {
        "design": "Speaking with fierce, barely contained anger, sharp and loud.",
        "empathic": "Anger",
        "e2v": "angry",
    },
    "intoxication": {
        "design": "Speaking very drunk: slurred words, loose rhythm, giggling between phrases.",
        "empathic": "Intoxication_Altered_States_of_Consciousness",
    },
}
VOICE = "A young woman with a light, quick voice and a soft, slightly rounded accent."
NEUTRAL = "Speaking calmly in a plain, neutral tone."
# Clean flags that make a clip a poor reference.
BAD_FLAGS = {"far_voice", "subtitle_mismatch", "asr_only", "nonverbal", "silent"}


def main():
    s = settings()
    fake = fake_models()
    ensure_venv("requirements-steer.txt")
    out = out_dir()
    keep = Path(os.environ["VI_CHECKPOINT"]) if os.environ.get("VI_CHECKPOINT") else out / ".work"
    keep.mkdir(parents=True, exist_ok=True)

    axes = {**DEFAULT_AXES, **s.get("axes", {})}
    axes = {a: axes[a] for a in s.get("axis_ids", list(DEFAULT_AXES))}
    strengths = [float(x) for x in s.get("strengths", [0.5, 1, 2, 4])]
    n_refs = int(s.get("refs_per_side", 12))
    sources = s.get("sources", ["tali", "designed"])

    clips = {c.id: c for c in load_clips()}
    labels = label_values()
    refs = pick_refs(clips, labels, list(axes), n_refs, s.get("refs", {}))
    voice_ref = refs["neutral"][0]
    log(f"steer: steering from {voice_ref.id}; {n_refs} clips a side")

    designed = {}
    if "designed" in sources:
        designer = None if fake else Designer(s.get("designer", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
        designed = design_pairs(designer, axes, keep / "designed", int(s.get("design_pairs", 6)), fake)
        if designer is not None:
            designer.close()

    tts = (FakeTTS if fake else TTS)(s.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"), s.get("layers"))

    # Vectors, one per source and axis. The neutral side is shared.
    vectors = {}
    means = {}

    def state(refs):
        key = tuple(refs)
        if key not in means:
            means[key] = mean_state(tts, refs)
        return means[key]

    for source in sources:
        for a in axes:
            path = keep / "vectors" / f"{source}-{a}.json"
            if path.exists():
                vectors[(source, a)] = json.loads(path.read_text())
                continue
            if source == "tali":
                pos = [(str(c.audio), c.text) for c in refs[a]]
                neg = [(str(c.audio), c.text) for c in refs["neutral"]]
            else:
                pos = [(str(p), t) for p, t in designed[a]["pos"]]
                neg = [(str(p), t) for p, t in designed[a]["neg"]]
            log(f"steer: capturing {source} {a} ({len(pos)} vs {len(neg)} clips)")
            mp, mn = state(pos), state(neg)
            v = [[x - y for x, y in zip(lp, ln, strict=True)] for lp, ln in zip(mp, mn, strict=True)]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(v))
            vectors[(source, a)] = v

    # Generation: neutral, each vector at each strength, and the swap baseline.
    gens = [("none", "-", "neutral", 0.0, voice_ref)]
    for a in axes:
        gens.append(("swap", a, "swap", 0.0, refs[a][0]))
        for source in sources:
            for k in strengths:
                gens.append((source, a, f"x{k:g}", k, voice_ref))
    audio = keep / "audio"
    made = []
    for source, a, cond, k, ref in gens:
        for n, sentence in enumerate(TEST):
            path = audio / source / a / cond / f"{n}.wav"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                add = None if k == 0 else (vectors[(source, a)], k)
                tts.speak(str(ref.audio), ref.text, sentence, seed(n), add, path)
            made.append({"source": source, "axis": a, "cond": cond, "strength": k, "n": n, "path": path})
        log(f"steer: generated {source} {a} {cond}")
    tts.close()

    scores = score_all(made, axes, refs["neutral"], fake, s)
    report = summarize(made, scores, axes, sources, strengths, s)
    report["references"] = {
        "voice": voice_ref.id,
        **{a: [c.id for c in refs[a]] for a in axes},
        "neutral": [c.id for c in refs["neutral"]],
    }

    shutil.copytree(audio, out / "audio", dirs_exist_ok=True)
    (out / "vectors").mkdir(exist_ok=True)
    for (source, a), v in vectors.items():
        write_npy(out / "vectors" / f"{source}-{a}.npy", v)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(report))
    (out / "listen.md").write_text(listen_sheet(report, gens))
    for a, v in report["verdicts"].items():
        log(f"steer: {a}: {v['verdict']}")


# --- references ---------------------------------------------------------------


def label_values():
    """{id: {axis: value}} from label's labels.jsonl, if it's an input."""
    label = env_path("VI_INPUT_LABEL")
    if label is None or not (label / "labels.jsonl").exists():
        return {}
    return {
        r["id"]: {a: v.get("value", 0.0) for a, v in r.get("scores", {}).items()}
        for r in read_jsonl(label / "labels.jsonl")
    }


def usable(c):
    return 2.0 <= c.duration_s <= 12.0 and c.has_words and not (set(c.flags) & BAD_FLAGS)


def pick_refs(clips, labels, axes, n, chosen):
    """{axis: clips, "neutral": clips}. Ids in `chosen` win; otherwise the top
    n usable clips on the axis, and for neutral the n whose highest axis value
    is lowest, preferring 3 to 8 s so the first makes a good voice reference."""
    pool = [c for c in clips.values() if usable(c)]
    if len(pool) < 2:
        raise SystemExit("steer: too few usable clips (2 to 12 s, with words, unflagged)")
    out = {}
    for a in axes:
        if chosen.get(a):
            out[a] = [clips[i] for i in chosen[a] if i in clips]
        else:
            ranked = sorted(pool, key=lambda c: (-labels.get(c.id, {}).get(a, 0.0), c.id))
            out[a] = ranked[:n]
    if chosen.get("neutral"):
        out["neutral"] = [clips[i] for i in chosen["neutral"] if i in clips]
    else:
        def calm(c):
            v = labels.get(c.id, {})
            return (max((v.get(a, 0.0) for a in axes), default=0.0), not 3.0 <= c.duration_s <= 8.0, c.id)

        used = {c.id for a in axes for c in out[a]}
        out["neutral"] = sorted([c for c in pool if c.id not in used], key=calm)[:n]
    for k, v in out.items():
        if not v:
            raise SystemExit(f"steer: no clips for {k}")
    return out


def design_pairs(designer, axes, folder, n, fake):
    """{axis: {"pos": [(path, text)], "neg": [...]}}: the same sentence spoken
    with the axis and neutrally, in one designed voice. Neutral clips are
    shared across axes."""
    folder.mkdir(parents=True, exist_ok=True)
    sentences = DESIGN[:n]
    neg = []
    for i, t in enumerate(sentences):
        path = folder / f"neutral-{i}.wav"
        if not path.exists():
            speak_design(designer, t, f"{VOICE} {NEUTRAL}", seed(i), path, fake)
        neg.append((path, t))
    out = {}
    for a, cfg in axes.items():
        pos = []
        for i, t in enumerate(sentences):
            path = folder / f"{a}-{i}.wav"
            if not path.exists():
                speak_design(designer, t, f"{VOICE} {cfg['design']}", seed(i), path, fake)
            pos.append((path, t))
        out[a] = {"pos": pos, "neg": neg}
    return out


def speak_design(designer, text, instruct, s, path, fake):
    if fake:
        write_wav(path, fake_wave(text + instruct))
        return
    designer.speak(text, instruct, s, path)


def seed(n):
    return 1000 + n


# --- the TTS model, hooked ----------------------------------------------------


def mean_state(tts, refs):
    """Mean over refs of each layer's mean hidden state over generated audio
    tokens, while cloning each ref and saying the capture sentences."""
    total = None
    count = 0
    for audio, text in refs:
        for n, sentence in enumerate(CAPTURE):
            m = tts.capture(audio, text, sentence, seed(100 + n))
            total = m if total is None else [[x + y for x, y in zip(a, b)] for a, b in zip(total, m)]
            count += 1
    return [[x / count for x in layer] for layer in total]


class TTS:
    """Qwen3-TTS Base (voice clone), with forward hooks on its talker's decoder
    layers. Hidden states are read and changed only on single-token steps,
    i.e. while it generates audio, not while it reads the prompt."""

    def __init__(self, model_id, layers=None):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(
            model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        blocks = self.model.model.talker.model.layers
        n = len(blocks)
        # Middle layers by default: steering usually works best there.
        self.layers = list(layers) if layers else list(range(n // 3, 2 * n // 3))
        self.mode = None
        self.sums = {}
        self.counts = {}
        self.add = {}
        for i, block in enumerate(blocks):
            block.register_forward_hook(self._hook(i))
        log(f"steer: {n} talker layers, steering {self.layers[0]}..{self.layers[-1]}")

    def close(self):
        """Frees the GPU for the scoring models."""
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()

    def _hook(self, i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] != 1:
                return None
            if self.mode == "capture":
                self.sums[i] = self.sums.get(i, 0) + h[:, -1].float().sum(0)
                self.counts[i] = self.counts.get(i, 0) + h.shape[0]
            elif self.mode == "steer" and i in self.add:
                h = h + self.add[i].to(h.dtype)
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return None

        return hook

    def _generate(self, audio, text, sentence, s):
        self.torch.manual_seed(s)
        wavs, sr = self.model.generate_voice_clone(
            text=sentence, language="English", ref_audio=audio, ref_text=text
        )
        return wavs[0], sr

    def capture(self, audio, text, sentence, s):
        self.mode, self.sums, self.counts = "capture", {}, {}
        try:
            self._generate(audio, text, sentence, s)
        finally:
            self.mode = None
        return [(self.sums[i] / self.counts[i]).tolist() for i in sorted(self.sums)]

    def speak(self, audio, text, sentence, s, add, path):
        import soundfile as sf

        if add is not None:
            vector, k = add
            dev = next(self.model.model.parameters()).device
            self.add = {i: k * self.torch.tensor(vector[i], device=dev) for i in self.layers}
            self.mode = "steer"
        try:
            wav, sr = self._generate(audio, text, sentence, s)
        finally:
            self.mode, self.add = None, {}
        sf.write(str(path), wav, sr)


class Designer:
    """Qwen3-TTS VoiceDesign: speech from a written voice and style description."""

    def __init__(self, model_id):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(
            model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
        )

    def close(self):
        """Frees the GPU for the Base model."""
        del self.model
        self.torch.cuda.empty_cache()

    def speak(self, text, instruct, s, path):
        import soundfile as sf

        self.torch.manual_seed(s)
        wavs, sr = self.model.generate_voice_design(text=text, language="English", instruct=instruct)
        sf.write(str(path), wavs[0], sr)


class FakeTTS:
    """Deterministic stand-in for CI: hidden states from hashes, audio a tone."""

    LAYERS = 6
    HIDDEN = 4

    def __init__(self, model_id, layers=None):
        self.layers = list(layers) if layers else [2, 3]

    def close(self):
        pass

    def capture(self, audio, text, sentence, s):
        key = Path(audio).name + sentence
        return [[unit_hash("h", key, str(i), str(j)) for j in range(self.HIDDEN)] for i in range(self.LAYERS)]

    def speak(self, audio, text, sentence, s, add, path):
        k = 0.0 if add is None else add[1]
        write_wav(path, fake_wave(f"{Path(audio).name}{sentence}{k}"))


# --- scoring ------------------------------------------------------------------


def score_all(made, axes, neutral, fake, s):
    """[{empathic: {key: v}, e2v: {class: p}, similarity, wer}] in `made` order."""
    if fake:
        return [fake_score(m, axes) for m in made]
    import models

    keys = sorted({cfg["empathic"] for cfg in axes.values() if cfg.get("empathic")})
    empathic = Empathic(s.get("empathic", "laion/Empathic-Insight-Voice-Small"), keys)
    e2v = models.Emotion2Vec(s.get("emotion", "emotion2vec/emotion2vec_plus_large"))
    speaker = models.Speaker(s.get("speaker", "speechbrain/spkrec-ecapa-voxceleb"))
    asr = models.Whisper(s.get("asr", "openai/whisper-large-v3"))

    her = mean_vector([speaker(c) for c in neutral])
    out = []
    for m in made:
        clip = AudioFile(m["path"])
        probs, _ = e2v(clip)
        out.append(
            {
                "empathic": empathic(m["path"]),
                "e2v": probs,
                "similarity": round(cosine(speaker(clip), her), 4),
            }
        )
    heard = asr([AudioFile(m["path"]) for m in made])
    for r, m, words in zip(out, made, heard, strict=True):
        r["wer"] = round(wer(TEST[m["n"]], words), 4)
    return out


class AudioFile:
    """What models.py's classes need from a clip."""

    def __init__(self, path):
        self.audio = Path(path)
        self.id = Path(path).stem


class Empathic:
    """LAION Empathic-Insight-Voice (Small): a fine-tuned Whisper encoder and
    one small MLP head per dimension. Trained largely on synthetic speech, so
    its scores rank lines; their absolute values mean little."""

    SEQ = 1500
    DIM = 768

    def __init__(self, repo, keys):
        import torch
        from huggingface_hub import hf_hub_download
        from torch import nn
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        encoder_id = "mkrausio/EmoWhisper-AnS-Small-v0.1"
        self.processor = WhisperProcessor.from_pretrained(encoder_id)
        self.encoder = WhisperForConditionalGeneration.from_pretrained(encoder_id).to("cuda").eval().get_encoder()

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.proj = nn.Linear(Empathic.SEQ * Empathic.DIM, 64)
                layers = [nn.ReLU(), nn.Dropout(0.0)]
                dim = 64
                for h in (64, 32, 16):
                    layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.1)]
                    dim = h
                layers.append(nn.Linear(dim, 1))
                self.mlp = nn.Sequential(*layers)

            def forward(self, x):
                return self.mlp(self.proj(self.flatten(x)))

        self.heads = {}
        for k in keys:
            state = torch.load(hf_hub_download(repo, f"model_{k}_best.pth"), map_location="cpu")
            state = {n.replace("_orig_mod.", ""): v for n, v in state.items()}
            head = Head()
            head.load_state_dict(state)
            self.heads[k] = head.eval()

    def __call__(self, path):
        import models

        torch = self.torch
        wav = models.load_audio(path)[: 30 * 16_000]
        feats = self.processor(wav, sampling_rate=16_000, return_tensors="pt").input_features.to("cuda")
        with torch.no_grad():
            emb = self.encoder(input_features=feats).last_hidden_state[:, : self.SEQ].float().cpu()
            if emb.shape[1] < self.SEQ:
                emb = torch.nn.functional.pad(emb, (0, 0, 0, self.SEQ - emb.shape[1]))
            return {k: round(float(h(emb)), 4) for k, h in self.heads.items()}


def fake_score(m, axes):
    """Fake scores that rise with strength for the "designed" source only, so
    CI sees both verdicts."""
    key = f"{m['source']}{m['axis']}{m['cond']}{m['n']}"
    lift = 0.3 * m["strength"] if m["source"] == "designed" else 0.0
    return {
        "empathic": {cfg["empathic"]: round(unit_hash("e", key, a) + lift, 4) for a, cfg in axes.items()},
        "e2v": {"angry": round(min(1.0, unit_hash("a", key) / 2 + lift / 2), 4)},
        "similarity": round(0.8 - 0.02 * m["strength"], 4),
        "wer": round(0.01 * m["strength"], 4),
    }


def mean_vector(vs):
    return [sum(x) / len(vs) for x in zip(*vs)]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def wer(sentence, heard):
    """Word error rate of what Whisper heard against the sentence."""
    return clean.wer(clean.words(sentence), clean.words(heard))


# --- report -------------------------------------------------------------------


def target(score, axis, cfg):
    return score["empathic"].get(cfg["empathic"])


def summarize(made, scores, axes, sources, strengths, s):
    """Per axis: a table of mean target score, voice similarity and WER for
    each condition, and per source a verdict."""
    max_sim_drop = float(s.get("max_similarity_drop", 0.05))
    max_wer = float(s.get("max_wer", 0.15))
    rows = {}
    for m, sc in zip(made, scores, strict=True):
        rows.setdefault((m["source"], m["axis"], m["cond"]), []).append((m, sc))

    def agg(group, axis):
        cfg = axes[axis] if axis in axes else None
        out = {
            "similarity": mean([sc["similarity"] for _, sc in group]),
            "wer": mean([sc["wer"] for _, sc in group]),
            "n": len(group),
        }
        if cfg:
            out["target"] = mean([target(sc, axis, cfg) for _, sc in group])
            if cfg.get("e2v"):
                out["e2v_" + cfg["e2v"]] = mean([sc["e2v"].get(cfg["e2v"], 0.0) for _, sc in group])
        return out

    neutral = rows[("none", "-", "neutral")]
    report = {"axes": {}, "verdicts": {}, "settings": {"strengths": strengths, "max_similarity_drop": max_sim_drop, "max_wer": max_wer}}
    for a, cfg in axes.items():
        base = agg(neutral, a)
        swap = agg(rows[("swap", a, "swap")], a)
        table = {"neutral": base, "swap": swap}
        best = None
        for source in sources:
            series = [(0.0, base)] + [(k, agg(rows[(source, a, f"x{k:g}")], a)) for k in strengths]
            for k, r in series[1:]:
                table[f"{source} x{k:g}"] = r
            # Spearman of target vs strength over every generated line, neutral included.
            xs, ys = [], []
            for m, sc in neutral + [p for k in strengths for p in rows[(source, a, f"x{k:g}")]]:
                xs.append(m["strength"])
                ys.append(target(sc, a, cfg))
            rho = fusion.spearman(xs, ys)
            ok = [
                (k, r) for k, r in series[1:]
                if base["similarity"] - r["similarity"] <= max_sim_drop and r["wer"] <= max_wer
            ]
            top = max(ok, key=lambda kr: kr[1]["target"], default=None)
            steers = rho is not None and rho >= 0.3 and top is not None and top[1]["target"] > base["target"]
            beats = steers and top[1]["target"] >= swap["target"]
            v = {
                "spearman_strength_target": rho,
                "best_strength": None if top is None else top[0],
                "best_target": None if top is None else top[1]["target"],
                "steers": steers,
                "beats_swap": beats,
            }
            report["verdicts"].setdefault(a, {"sources": {}})["sources"][source] = v
            if steers and (best is None or v["best_target"] > best[1]["best_target"]):
                best = (source, v)
        report["axes"][a] = table
        if best is None:
            verdict = "no source steers it within the voice and word limits"
        elif best[1]["beats_swap"]:
            verdict = f"steers ({best[0]} vectors, x{best[1]['best_strength']:g}) and beats swapping the reference"
        else:
            verdict = f"steers ({best[0]} vectors, x{best[1]['best_strength']:g}) but swapping the reference does as well"
        report["verdicts"][a]["verdict"] = verdict
    return report


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def render(report):
    lines = ["# Steering test", ""]
    for a, table in report["axes"].items():
        lines += [f"## {a}", "", f"**Verdict:** {report['verdicts'][a]['verdict']}.", ""]
        extra = [k for k in next(iter(table.values())) if k.startswith("e2v_")]
        head = ["condition", "target", *extra, "voice similarity", "WER"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for cond, r in table.items():
            cells = [cond, fmt(r.get("target")), *(fmt(r.get(k)) for k in extra), fmt(r["similarity"]), fmt(r["wer"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        for source, v in report["verdicts"][a]["sources"].items():
            lines.append(
                f"- {source}: Spearman of score vs strength {fmt(v['spearman_strength_target'])}, "
                f"best strength {fmt(v['best_strength'])}, steers {v['steers']}, beats swap {v['beats_swap']}"
            )
        lines.append("")
    lines += [
        "Target is LAION Empathic-Insight-Voice's raw score for the axis: it ranks lines but its",
        "absolute values mean little. Listening decides; see listen.md.",
        "",
    ]
    return "\n".join(lines)


def listen_sheet(report, gens):
    refs = report["references"]
    lines = [
        "# Listening sheet",
        "",
        f"Every condition speaks the same {len(TEST)} sentences (audio/<source>/<axis>/<condition>/<n>.wav).",
        "",
        f"- Steered lines clone her voice from clip {refs['voice']}.",
        "- `swap` clones from her top clip on the axis instead, with no vector: the baseline to beat.",
        "- `tali xK` adds K times the vector from her own clips; `designed xK` the one from the VoiceDesign pairs.",
        "",
        "Suggested order: neutral, then each axis's swap, then its strengths from low to high.",
        "",
    ]
    for a in report["axes"]:
        lines.append(f"- {a} reference clips: {', '.join(refs[a])}")
    lines.append(f"- neutral reference clips: {', '.join(refs['neutral'])}")
    lines.append("")
    return "\n".join(lines)


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    return f"{x:.3g}" if isinstance(x, float) else str(x)


# --- files ----------------------------------------------------------------------


def fake_wave(key, rate=16_000, seconds=1.0):
    f = 200 + 300 * unit_hash("f", key)
    return rate, [0.2 * math.sin(2 * math.pi * f * i / rate) for i in range(int(rate * seconds))]


def write_wav(path, wave):
    import struct
    import wave as wav

    rate, samples = wave
    with wav.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, x)) * 32767)) for x in samples))


def write_npy(path, rows):
    """A 2-D float32 .npy with the standard library (as vi_job.write_embeddings)."""
    import struct

    n, d = len(rows), len(rows[0])
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {d}), }}"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode())
        for r in rows:
            f.write(struct.pack(f"<{d}f", *r))


if __name__ == "__main__":
    main()
