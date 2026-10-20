"""voicesteer stage: voice vectors for her moods on Qwen3-TTS, and test D's automatic half
(docs/design/step5-rental-plan.md, after-labels-plan.md step 5). GPU, manual.

Replaces steer.py's label-axis test, built on what the voice gate (after-labels step 3) passed:
  voice_sets.json   per mood, the pairs step 3 tested (her trusted positive clips against
                    same-game trusted zeros matched on measured energy and length, effect clips
                    already left out), and her neutral clips
  voice_probe.npz   per mood, step 3's probe refit on features any audio has: the raw
                    emotion2vec+ large utterance embedding and five prosody measures
Both come from corpus/tali/analysis/after_labels/voice_probe.py, through on-worker.sh --put.

For each mood: the vector per talker layer is the mean hidden state over generated audio tokens
while Qwen3-TTS Base clones her positive clips (in context, with their words) saying the capture
sentences, minus the same for their matched zeros (the first `refs_per_side` pairs, strongest
first). Then the test sentences are spoken from her first neutral clip:
  - neutral: no vector
  - swap: her top positive clip as the reference, no vector (the baseline to beat)
  - steered: the vector at strength 1 (unaltered). Brandon, 2026-10-20: start unaltered, and lower
    it only if it performs badly. So if her voice (ECAPA cosine to her neutral clips) drops more
    than max_similarity_drop below the neutral lines', or Whisper's word error rate goes over
    max_wer, it's retried at each of `lower` in turn, and the first that keeps both is kept.
Scores per line: the mood's probe probability, voice similarity, word error rate.
Verdict per mood: "steers" if the probe rises over neutral by min_lift on average and on at least
min_share of the sentences, at a strength that keeps voice and words; borderline moods are marked.
Brandon's blind A/B pairs come later, at home, from listen.md's files.

Outputs: report.json, report.md, listen.md, audio/<mood>/<condition>/<n>.wav and
vectors/<mood>.npy (layers x hidden, with vectors.json naming model, layers and strength). Audio
and vectors go to VI_CHECKPOINT as they're made, so a rerun on a new worker resumes.
"""

import json
import os
import shutil
from pathlib import Path

import steer
from steer import CAPTURE, TEST, cosine, mean_vector, seed, wer, write_wav
from vi_job import ensure_venv, env_path, fake_models, load_clips, log, out_dir, settings, unit_hash


def main():
    s = settings()
    fake = fake_models()
    if not fake:
        ensure_venv("requirements-steer.txt")
    out = out_dir()
    keep = Path(os.environ["VI_CHECKPOINT"]) if os.environ.get("VI_CHECKPOINT") else out / ".work"
    keep.mkdir(parents=True, exist_ok=True)
    files = Path(os.environ.get("VI_FILES", "files"))
    sets = json.loads((files / "voice_sets.json").read_text())
    probe = Probe(files / "voice_probe.npz")
    moods = s.get("moods") or list(sets["moods"])
    n_refs = int(s.get("refs_per_side", 32))
    n_capture = int(s.get("capture_sentences", 2))
    lower = [float(x) for x in s.get("lower", [0.5, 0.25])]
    max_drop = float(s.get("max_similarity_drop", 0.05))
    max_wer = float(s.get("max_wer", 0.15))

    clips = {c.id: c for c in load_clips()}
    neutral = [clips[i] for i in sets["neutral"] if i in clips]
    if not neutral:
        raise SystemExit("voicesteer: none of the neutral clips are in harvest")
    voice = neutral[0]
    log(f"voicesteer: {len(moods)} moods, speaking from {voice.id}, {n_refs} pairs a mood")

    tests = TEST[: int(s.get("test_sentences", len(TEST)))]  # fewer only for smoke tests
    tts = (steer.FakeTTS(s.get("model"), s.get("layers")) if fake else
           steer.TTS(s.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"), s.get("layers"), s.get("device", "cuda:0")))
    scorer = FakeScorer() if fake else Scorer(s)
    her = scorer.her(neutral)

    def speak(mood, cond, ref, add):
        made = []
        for n, sentence in enumerate(tests):
            path = keep / "audio" / mood / cond / f"{n}.wav"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tts.speak(str(ref.audio), ref.text, sentence, seed(n), add, path)
            made.append(path)
        return made

    def scored(mood, cond, paths):
        cache = keep / "scores" / mood / f"{cond}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        rows = [scorer(p, TEST[n], mood, probe, her) for n, p in enumerate(paths)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
        return rows

    report = {"model": s.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"), "layers": tts.layers, "voice": voice.id,
              "settings": {"refs_per_side": n_refs, "capture_sentences": n_capture, "lower": lower,
                           "max_similarity_drop": max_drop, "max_wer": max_wer}, "moods": {}}
    base = scored("_neutral", "neutral", speak("_neutral", "neutral", voice, None))
    base_sim = mean([r["similarity"] for r in base])
    vectors = {}
    for mood in moods:
        pairs = [p for p in sets["moods"][mood]["pairs"] if p[0] in clips and p[1] in clips][:n_refs]
        vpath = keep / "vectors" / f"{mood}.json"
        if vpath.exists():
            v = json.loads(vpath.read_text())
        else:
            pos = [(str(clips[p].audio), clips[p].text) for p, _ in pairs]
            neg = [(str(clips[z].audio), clips[z].text) for _, z in pairs]
            log(f"voicesteer: capturing {mood} ({len(pairs)} pairs)")
            mp, mn = mean_state(tts, pos, n_capture), mean_state(tts, neg, n_capture)
            v = [[x - y for x, y in zip(a, b, strict=True)] for a, b in zip(mp, mn, strict=True)]
            vpath.parent.mkdir(parents=True, exist_ok=True)
            vpath.write_text(json.dumps(v))
        vectors[mood] = v
        neutral_probe = [r["probe"][mood] for r in scored(mood, "neutral", speak("_neutral", "neutral", voice, None))]
        swap = scored(mood, "swap", speak(mood, "swap", clips[pairs[0][0]], None))
        tried, kept = [], None
        for k in [1.0, *lower]:
            rows = scored(mood, f"x{k:g}", speak(mood, f"x{k:g}", voice, (v, k)))
            sim, w = mean([r["similarity"] for r in rows]), mean([r["wer"] for r in rows])
            ok = sim >= base_sim - max_drop and w <= max_wer
            lifts = [r["probe"][mood] - b for r, b in zip(rows, neutral_probe, strict=True)]
            tried.append({"strength": k, "probe": mean([r["probe"][mood] for r in rows]), "lift": mean(lifts),
                          "lift_share": sum(x > 0 for x in lifts) / len(lifts), "similarity": sim, "wer": w, "keeps_voice_and_words": ok})
            if ok:
                kept = tried[-1]
                break
            log(f"voicesteer: {mood} at {k:g} loses voice or words (similarity {sim:.3f}, wer {w:.3f}); lowering")
        steers = bool(kept) and kept["lift"] >= float(s.get("min_lift", 0.1)) and kept["lift_share"] >= float(s.get("min_share", 0.75))
        report["moods"][mood] = {
            "route": sets["moods"][mood].get("route"), "borderline": sets["moods"][mood].get("borderline", False),
            "pairs": len(pairs), "neutral_probe": mean(neutral_probe),
            "swap": {"probe": mean([r["probe"][mood] for r in swap]), "similarity": mean([r["similarity"] for r in swap]),
                     "wer": mean([r["wer"] for r in swap])},
            "tried": tried, "strength": kept["strength"] if kept else None,
            "verdict": "steers" if steers else ("loses voice or words" if not kept else "doesn't steer"),
        }
        log(f"voicesteer: {mood}: {report['moods'][mood]['verdict']}")
    tts.close()

    shutil.copytree(keep / "audio", out / "audio", dirs_exist_ok=True)
    (out / "vectors").mkdir(exist_ok=True)
    for mood, v in vectors.items():
        steer.write_npy(out / "vectors" / f"{mood}.npy", v)
    (out / "vectors" / "vectors.json").write_text(json.dumps(
        {"model": report["model"], "layers": tts.layers,
         "strength": {m: r["strength"] for m, r in report["moods"].items()}}, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(report))
    (out / "listen.md").write_text(listen(report))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def mean_state(tts, refs, n_capture):
    """Mean over refs and capture sentences of each layer's mean hidden state over generated audio."""
    total, count = None, 0
    for audio, text in refs:
        for n, sentence in enumerate(CAPTURE[:n_capture]):
            m = tts.capture(audio, text, sentence, seed(100 + n))
            total = m if total is None else [[x + y for x, y in zip(a, b)] for a, b in zip(total, m)]
            count += 1
    return [[x / count for x in layer] for layer in total]


class Probe:
    """voice_probe.npz: per mood, standardise -> PCA -> logistic regression, in plain numpy."""

    def __init__(self, path):
        import numpy as np

        self.np = np
        z = np.load(path, allow_pickle=False)
        self.moods = [str(m) for m in z["moods"]]
        self.keys = [str(k) for k in z["prosody_keys"]]
        self.pm, self.ps = z["prosody_mean"], z["prosody_scale"]
        self.w = {m: {k: z[f"{m}.{k}"] for k in ("mean", "scale", "pca_mean", "pca_components", "coef", "intercept")}
                  for m in self.moods}

    def features(self, embedding, prosody):
        np = self.np
        p = np.array([prosody.get(k) if prosody.get(k) is not None else self.pm[i] for i, k in enumerate(self.keys)], float)
        return np.concatenate([np.asarray(embedding, float), (p - self.pm) / self.ps])

    def __call__(self, x):
        np = self.np
        out = {}
        for m, w in self.w.items():
            h = ((x - w["mean"]) / w["scale"] - w["pca_mean"]) @ w["pca_components"].T
            out[m] = round(float(1 / (1 + np.exp(-(h @ w["coef"] + w["intercept"][0])))), 4)
        return out


class Line:
    """What models.prosody needs from a generated line."""

    def __init__(self, path, text):
        import soundfile as sf

        info = sf.info(str(path))
        self.audio, self.id, self.text = Path(path), Path(path).stem, text
        self.duration_s, self.nonverbal = info.frames / info.samplerate, False


class Scorer:
    def __init__(self, s):
        import models

        self.models = models
        self.e2v = models.Emotion2VecEmbed(s.get("emotion", "emotion2vec/emotion2vec_plus_large"),
                                           s.get("emotion_revision", "6c303ba987b86b93193de93e34bb2b077a6bedc4"), "e2v")
        self.speaker = models.Speaker(s.get("speaker", "speechbrain/spkrec-ecapa-voxceleb"))
        self.asr = models.Whisper(s.get("asr", "openai/whisper-large-v3"))

    def her(self, clips):
        return mean_vector([self.speaker(c) for c in clips])

    def __call__(self, path, sentence, mood, probe, her):
        line = Line(path, sentence)
        emb = self.e2v(self.models.load_audio(path))["e2v"]
        x = probe.features(emb, self.models.prosody(line))
        heard = self.asr([steer.AudioFile(path)])[0]
        return {"probe": probe(x), "similarity": round(cosine(self.speaker(steer.AudioFile(path)), her), 4),
                "wer": round(wer(sentence, heard), 4)}


class FakeScorer:
    """Deterministic scores for CI: the probe rises with strength, voice falls a little."""

    def her(self, clips):
        return [1.0]

    def __call__(self, path, sentence, mood, probe, her):
        cond = Path(path).parent.name
        k = float(cond[1:]) if cond.startswith("x") else 0.0
        return {"probe": {m: round(0.3 + 0.3 * k * (m == mood) + 0.05 * unit_hash("p", str(path), m), 4) for m in probe.moods},
                "similarity": round(0.8 - 0.15 * max(0.0, k - 0.6), 4), "wer": round(0.02 * k, 4)}


def render(report):
    lines = ["# Voice vectors and test D (automatic half)", "",
             f"Model {report['model']}, layers {report['layers'][0]}..{report['layers'][-1]}, speaking from {report['voice']}.",
             "Probe: step 3's probe on raw emotion2vec+ and prosody (0-1, higher is more of the mood).", "",
             "| mood | verdict | strength | probe neutral -> steered (lift, share up) | swap probe | similarity | WER | |",
             "|---|---|---|---|---|---|---|---|"]
    for m, r in sorted(report["moods"].items()):
        t = next((x for x in r["tried"] if x["strength"] == r["strength"]), r["tried"][-1])
        lines.append(f"| {m} | {r['verdict']} | {r['strength']} | {r['neutral_probe']} -> {t['probe']} ({t['lift']:+.3f}, {t['lift_share']:.0%}) | "
                     f"{r['swap']['probe']} | {t['similarity']} | {t['wer']} | {'borderline' if r['borderline'] else ''} |")
    return "\n".join(lines) + "\n"


def listen(report):
    lines = ["# Listening", "", "audio/_neutral/neutral/<n>.wav is her neutral voice; audio/<mood>/x<strength>/<n>.wav is steered;",
             "audio/<mood>/swap/<n>.wav swaps in her top clip for that mood. Sentence n:", ""]
    lines += [f"{n}. {t}" for n, t in enumerate(TEST)]
    lines += ["", "Kept strength per mood:", ""] + [f"- {m}: x{r['strength']:g}" if r["strength"] else f"- {m}: none kept"
                                                   for m, r in sorted(report["moods"].items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
