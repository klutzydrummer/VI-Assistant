"""label stage: score every clip on the pack's variables.

Plan: docs/design/control-vector-datasets.md, step 4. Format:
docs/design/corpus-format.md, label. Signals per axis:

  audio      emotion2vec+ class probabilities, mapped to axes (fusion.py)
  text       Gemma 4 reading the line and its context, with a confidence
  prior      scene tags and LE3 stage directions
  audio_llm  discover's no-words scores, when discover is an input

Barks have no subtitle, so they get no text signal unless clean found words;
they're tagged combat_bark so contrast sets can leave them out. Nonverbal
lines ({Sighs}) go to the judge as a sound, not words, and are tagged
nonverbal.

Outputs: labels.jsonl, axes.json, fusion.json (how often each signal
scored each axis and how well the signals agree, to tune weights), and
embeddings/emotion.npy + .ids (emotion2vec+ per clip, for finding clips
like a reviewed one and training small heads on reviews).

Settings: backend ("fake" for tests), audio, dimensions, judge, batch,
quant, weights, audio_map, scene_priors, audio_llm_axes, limit.
"""

import gc
import json

import fusion
import models
from vi_job import (
    ensure_packages,
    env_path,
    fake_models,
    load_axes,
    load_clips,
    log,
    out_dir,
    read_jsonl,
    settings,
    write_axes,
    write_embeddings,
    write_jsonl,
)

AUDIO_LLM_CONFIDENCE = 0.6


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
    axes = load_axes()
    log(f"label: {len(clips)} clips, axes {', '.join(a.id for a in axes)}")

    # Acoustic models.
    audio_model = s.get("audio", "emotion2vec/emotion2vec_plus_large")
    e2v = (models.FakeEmotion2Vec if fake else models.Emotion2Vec)(audio_model)
    dims = (models.FakeDimensions if fake else models.Dimensions)(
        s.get("dimensions", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    )
    probs, adv, emb = {}, {}, {}
    for i, clip in enumerate(clips):
        probs[clip.id], emb[clip.id] = e2v(clip)
        adv[clip.id] = dims(clip)
        if i % 250 == 0:
            log(f"acoustic: {i}/{len(clips)}")
    del e2v, dims
    free()
    embeddings = {"emotion": write_embeddings(out / "embeddings", "emotion", audio_model, emb)}

    # Text judge.
    judge = models.Judge(
        (models.FakeGemma if fake else models.Gemma)(
            s.get("judge", "google/gemma-4-12B-it"),
            batch=int(s.get("batch", 8)),
            quant=s.get("quant", "auto"),
        )
    )
    readable = [c for c in clips if c.text.strip()]
    log(f"judging {len(readable)} lines with text")
    text = {}
    unparsed = 0
    for clip, r in zip(readable, judge.json([models.judge_prompt(c, axes) for c in readable])):
        if not isinstance(r, dict):
            unparsed += 1
            continue
        text[clip.id] = r
    del judge
    free()

    # The audio LLM's scores from discover.
    heard = {}
    discover = env_path("VI_INPUT_DISCOVER")
    if discover is not None and (discover / "captions.jsonl").exists():
        for c in read_jsonl(discover / "captions.jsonl"):
            heard[c["id"]] = c
    audio_llm_axes = set(s.get("audio_llm_axes", [a.id for a in axes]))

    weights = {**fusion.DEFAULT_WEIGHTS, **s.get("weights", {})}
    audio_map = s.get("audio_map", fusion.DEFAULT_AUDIO_MAP)
    scene_priors = s.get("scene_priors", fusion.DEFAULT_SCENE_PRIORS)

    labels = []
    for clip in clips:
        prior = fusion.priors(clip.scene_tags, clip.stage_direction, scene_priors, fusion.DEFAULT_DIRECTION_PRIORS)
        caption = heard.get(clip.id, {})
        scores = {}
        for a in axes:
            signals = {
                "audio": fusion.audio_signal(probs.get(clip.id), a.id, audio_map, clip.duration_s),
                "text": text_signal(text.get(clip.id), a.id),
                "prior": prior.get(a.id),
            }
            if heard:
                v = caption.get("scores", {}).get(a.id) if a.id in audio_llm_axes else None
                signals["audio_llm"] = None if v is None else (v, AUDIO_LLM_CONFIDENCE)
            scores[a.id] = fusion.fuse(signals, weights)
        labels.append(
            {
                "id": clip.id,
                "scores": scores,
                "tags": fusion.tags_for(clip.flags, caption.get("tags", []), clip.duration_s),
                "audio_classes": probs.get(clip.id),
                "dimensions": adv.get(clip.id),
            }
        )

    write_jsonl(out / "labels.jsonl", labels)
    write_axes(out, axes)
    summary = report(labels, axes, unparsed) | {"embeddings": embeddings}
    (out / "fusion.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    log(f"label: wrote {len(labels)} labels, {unparsed} unparsed judge replies")


def text_signal(reply, axis):
    if reply is None:
        return None
    entry = reply.get(axis)
    if isinstance(entry, dict):
        value = fusion.clamp01(entry.get("value"))
        confidence = fusion.clamp01(entry.get("confidence"))
    else:
        value, confidence = fusion.clamp01(entry), 0.5
    if value is None:
        return None
    return value, 0.5 if confidence is None else confidence


def report(labels, axes, unparsed):
    """Per axis: how many clips each signal scored, how well each pair of
    signals agrees (Spearman), and how the fused confidence is spread."""
    out = {"clips": len(labels), "unparsed_judge_replies": unparsed, "axes": {}}
    for a in axes:
        cols = {n: [l["scores"][a.id]["signals"].get(n) for l in labels] for n in fusion.SIGNALS}
        conf = sorted(l["scores"][a.id]["confidence"] for l in labels)
        agree = {}
        names = [n for n in fusion.SIGNALS if any(v is not None for v in cols[n])]
        for i, x in enumerate(names):
            for y in names[i + 1 :]:
                agree[f"{x}~{y}"] = fusion.spearman(cols[x], cols[y])
        out["axes"][a.id] = {
            "scored_by": {n: sum(v is not None for v in cols[n]) for n in fusion.SIGNALS},
            "agreement": agree,
            "confidence_median": conf[len(conf) // 2] if conf else None,
            "low_confidence": sum(c < 0.3 for c in conf),
        }
    return out


if __name__ == "__main__":
    main()
