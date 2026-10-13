"""clean stage: check every clip before anything is scored.

Plan: docs/design/control-vector-datasets.md, step 2. Format:
docs/design/corpus-format.md, clean. Per clip:

  audio      leading and trailing silence trimmed, written as FLAC. Loudness
             is left alone: how loud she is carries emotion.
  asr_text   Whisper's words, to check the subtitle (subtitle_wer). The
             subtitle stays the transcript, since Whisper misspells names
             like Keelah se'lai; a line with no subtitle (barks) gets
             Whisper's words as its transcript, flagged asr_only.
  speaker    ECAPA embedding and its cosine similarity to her median voice.
             Clips far below the rest are flagged far_voice: another voice,
             heavy effects, or a clip mapped to the wrong line.
  quality    DNSMOS scores, for picking TTS references later.

Nothing is dropped for a mismatch or a far voice yet; they're flagged so the
review page can show them and we can set thresholds from what they turn out
to be. Only clips that trim to nothing are dropped (`silent`).

Outputs: clips.jsonl, audio/<id>.flac, embeddings/speaker.npy, clean.json.

Settings: backend ("fake" for tests), asr, speaker, batch, mismatch_wer,
far_voice_mads, limit.
"""

import gc
import json
import os
import re
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor

import models
from vi_job import (
    Checkpoint,
    ensure_packages,
    fake_models,
    load_clips,
    log,
    out_dir,
    settings,
    write_embeddings,
    write_jsonl,
)

# Spoken clips at least this long set her median voice and are the only ones
# judged against it: on the smoke test every clip under 1.6 s and every sigh
# scored far from her, without being anyone else.
SPEAKER_REFERENCE_MIN_S = 1.5
# Each game gets its own median voice and threshold once it has this many
# reference clips: the speaker embedding hears each game's recording and mix
# (on the smoke test it told the games apart 72% of the time, 45% by chance).
# Smaller groups use the all-games baseline.
SPEAKER_GROUP_MIN = 20


def free():
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    s = settings()
    ensure_packages()
    fake = fake_models()
    out = out_dir()
    clips = load_clips()
    log(f"clean: {len(clips)} clips")

    # 1. Trim. Cheap, so redone on a resume rather than checkpointed.
    (out / "audio").mkdir(exist_ok=True)
    audio = {}
    for c in clips:
        if fake:
            dest = out / "audio" / f"{c.id}{c.audio.suffix}"
            shutil.copyfile(c.audio, dest)
            audio[c.id] = {"path": f"audio/{dest.name}", "duration_s": c.duration_s, "rms_dbfs": -20.0}
        else:
            dest = out / "audio" / f"{c.id}.flac"
            duration, rate, rms = models.trim(c, dest)
            audio[c.id] = {"path": f"audio/{dest.name}", "sample_rate": rate, "channels": 1,
                           "duration_s": round(duration, 4), "rms_dbfs": rms}  # fmt: skip
    silent = {c.id for c in clips if audio[c.id]["duration_s"] < 0.2}
    live = [c for c in clips if c.id not in silent]

    # 2. Speech recognition, on what's left after trimming. Nonverbal lines
    # ({Sighs}) have no words to check.
    asr_model = s.get("asr", "openai/whisper-large-v3")
    spoken = [c for c in live if not c.nonverbal]
    saved = Checkpoint("asr")
    asr = None
    if saved.todo(spoken):
        asr = (models.FakeWhisper if fake else models.Whisper)(asr_model, batch=int(s.get("batch", 16)))
    heard = saved.run(spoken, lambda part: asr(part), chunk=64)
    del asr
    free()

    # 3. Speaker embeddings and her median voice.
    speaker_model = s.get("speaker", "speechbrain/spkrec-ecapa-voxceleb")
    saved = Checkpoint("speaker")
    spk = None
    if saved.todo(live):
        spk = (models.FakeSpeaker if fake else models.Speaker)(speaker_model)
    voice = saved.run(live, lambda part: [spk(c) for c in part])
    del spk
    free()
    judged = {c.id for c in live if c.duration_s >= SPEAKER_REFERENCE_MIN_S and not c.nonverbal}
    mads = float(s.get("far_voice_mads", 4))
    baselines = {"all": baseline(voice, judged, list(voice), mads)}
    for game in sorted({c.game for c in live}):
        members = [c.id for c in live if c.game == game]
        if sum(cid in judged for cid in members) >= SPEAKER_GROUP_MIN:
            baselines[game] = baseline(voice, judged, members, mads)
    group = {c.id: c.game if c.game in baselines else "all" for c in live}
    similarity = {cid: round(dot(unit(v), baselines[group[cid]]["centre"]), 4) for cid, v in voice.items()}

    # 4. Quality.
    # DNSMOS runs on the CPU at about 1 s a clip, so spread it over threads
    # (onnxruntime releases the GIL).
    measure = models.fake_quality if fake else models.quality
    if live:
        measure(live[0])  # load the model once before the threads share it
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        mos = Checkpoint("quality").run(live, lambda part: list(pool.map(measure, part)), chunk=128)

    # 5. Records.
    mismatch_wer = float(s.get("mismatch_wer", 0.5))
    records = []
    for c in clips:
        r = {"id": c.id, "keep": c.id not in silent, "audio": audio[c.id], "flags": []}
        if c.id in silent:
            r["flags"].append("silent")
            records.append(r)
            continue
        if c.id in heard:
            r["asr_text"] = heard[c.id]
            if c.text.strip():
                ref = words(c.text)
                r["subtitle_wer"] = round(wer(ref, words(heard[c.id])), 3)
                if r["subtitle_wer"] > mismatch_wer and len(ref) >= 3:
                    r["flags"].append("subtitle_mismatch")
            elif words(heard[c.id]):
                r["transcript"] = heard[c.id]
                r["flags"].append("asr_only")
        r["speaker_similarity"] = similarity[c.id]
        if c.id in judged and similarity[c.id] < baselines[group[c.id]]["far_below"]:
            r["flags"].append("far_voice")
        r["quality"] = mos[c.id]
        records.append(r)

    write_jsonl(out / "clips.jsonl", records)
    embeddings = {"speaker": write_embeddings(out / "embeddings", "speaker", speaker_model, voice)}
    report = summary(records, embeddings, asr_model, baselines)
    (out / "clean.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log(f"clean: {report['kept']} kept, flags {report['flags']}")


def baseline(voice, judged, members, mads):
    """Her median voice among `members` (from the judged ones, or all of them
    if none is judged), and how far below it counts as a far voice."""
    reference = [voice[cid] for cid in members if cid in judged] or [voice[cid] for cid in members]
    centre = unit([statistics.median(col) for col in zip(*reference)]) if reference else []
    sims = [dot(unit(v), centre) for v in reference]
    med = statistics.median(sims) if sims else 0.0
    mad = statistics.median(abs(x - med) for x in sims) if sims else 0.0
    return {"centre": centre, "references": len(reference), "median": round(med, 4),
            "mad": round(mad, 4), "far_below": round(med - mads * max(mad, 1e-3), 4)}  # fmt: skip


def summary(records, embeddings, asr_model, baselines):
    kept = [r for r in records if r["keep"]]
    flags = {}
    for r in records:
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    wers = [r["subtitle_wer"] for r in kept if "subtitle_wer" in r]
    sims = sorted(kept, key=lambda r: r["speaker_similarity"])
    return {
        "clips": len(records),
        "kept": len(kept),
        "flags": flags,
        "asr_model": asr_model,
        "embeddings": embeddings,
        "subtitle_wer": quantiles(wers),
        "speaker_similarity": quantiles([r["speaker_similarity"] for r in kept]),
        "speaker_baselines": {g: {k: v for k, v in b.items() if k != "centre"} for g, b in baselines.items()},
        "least_like_her": [r["id"] for r in sims[:20]],
        "quality_ovr": quantiles([r["quality"]["ovr"] for r in kept]),
    }


def quantiles(xs):
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    at = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {"n": len(xs), "min": xs[0], "p05": at(0.05), "p25": at(0.25), "median": at(0.5),
            "p75": at(0.75), "p95": at(0.95), "max": xs[-1]}  # fmt: skip


def words(text):
    """Lowercase words without punctuation or {stage notes}, for comparing
    a subtitle with what Whisper heard."""
    text = re.sub(r"[{\[(][^}\])]*[}\])]", " ", text.lower())
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.replace("’", "'"))


def wer(ref, hyp):
    """Word error rate of hyp against ref (word-level edit distance / len(ref))."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else list(v)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


if __name__ == "__main__":
    main()
