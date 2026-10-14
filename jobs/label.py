"""label stage: score every clip on the pack's variables.

Plan: docs/design/control-vector-datasets.md, step 4. Format:
docs/design/corpus-format.md, label. Signals per axis:

  audio      emotion2vec+ class probabilities, mapped to axes (fusion.py)
  text       Gemma 4 reading the line and its context, with a confidence
  prior      scene tags and LE3 stage directions
  audio_llm  the audio LLM's 1-5 rating of how well the delivery fits the
             axis, asked only for a shortlist per axis: the clips the other
             signals rank highest, plus a sample of low ones to show whether
             it can hear the axis at all (fusion.json's `verification`)

Barks have no subtitle, so they get no text signal unless clean found words;
they're tagged combat_bark so contrast sets can leave them out. Nonverbal
lines ({Sighs}) go to the judge as a sound, not words, and are tagged
nonverbal.

Outputs: labels.jsonl, axes.json, fusion.json (how often each signal
scored each axis and how well the signals agree, to tune weights), and
embeddings/emotion.npy + .ids (emotion2vec+ per clip, for finding clips
like a reviewed one and training small heads on reviews).

Settings: backend ("fake" for tests), audio, dimensions, judge, batch,
quant, weights, audio_map, scene_priors, audio_llm_axes (axes to verify),
verify_top, verify_low, limit.
"""

import gc
import hashlib
import json
import tempfile
from pathlib import Path

import checks
import fusion
import models
from vi_job import (
    Checkpoint,
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
    acoustic = Checkpoint("acoustic")
    if acoustic.todo(clips):
        e2v = (models.FakeEmotion2Vec if fake else models.Emotion2Vec)(audio_model)
        dims = (models.FakeDimensions if fake else models.Dimensions)(
            s.get("dimensions", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        )
        acoustic.run(clips, lambda part: [[*e2v(c), dims(c)] for c in part])
        del e2v, dims
        free()
    results = acoustic.run(clips, None)
    probs = {i: r[0] for i, r in results.items()}
    emb = {i: r[1] for i, r in results.items()}
    adv = {i: r[2] for i, r in results.items()}
    embeddings = {"emotion": write_embeddings(out / "embeddings", "emotion", audio_model, emb)}

    # Gemma: the text judge, then the audio verification. Loaded once, only
    # if a checkpoint is missing something.
    gemma = None

    def llm():
        nonlocal gemma
        if gemma is None:
            gemma = models.Judge(
                (models.FakeGemma if fake else models.Gemma)(
                    s.get("judge", "google/gemma-4-12B-it"),
                    batch=int(s.get("batch", 8)),
                    quant=s.get("quant", "auto"),
                    batch_audio_s=float(s.get("batch_audio_s", 200)),
                )
            )
        return gemma

    # Shortest prompt first, so batches pad less.
    readable = sorted((c for c in clips if c.text.strip()), key=lambda c: len(models.judge_prompt(c, axes)[0]["content"][0]["text"]))
    log(f"judging {len(readable)} lines with text")
    judged = Checkpoint("judge")
    if judged.todo(readable):
        judged.run(readable, lambda part: llm().json([models.judge_prompt(c, axes) for c in part]), chunk=64)
    replies = judged.run(readable, None)
    text = {i: r for i, r in replies.items() if isinstance(r, dict)}
    unparsed = len(replies) - len(text)

    # Delivery tags from discover's captions (laughing, whispering, ...).
    # Its per-axis guesses aren't used: open-ended captions mostly echo the
    # words (prior-work.md, section 2), so the audio LLM gets a narrow
    # question below instead.
    heard = {}
    discover = env_path("VI_INPUT_DISCOVER")
    if discover is not None and (discover / "captions.jsonl").exists():
        for c in read_jsonl(discover / "captions.jsonl"):
            heard[c["id"]] = c

    weights = {**fusion.DEFAULT_WEIGHTS, **s.get("weights", {})}
    audio_map = s.get("audio_map", fusion.DEFAULT_AUDIO_MAP)
    scene_priors = s.get("scene_priors", fusion.DEFAULT_SCENE_PRIORS)
    priors = {
        c.id: fusion.priors(c.scene_tags, c.stage_direction, scene_priors, fusion.DEFAULT_DIRECTION_PRIORS)
        for c in clips
    }

    def signals(clip, axis):
        return {
            "audio": fusion.audio_signal(probs.get(clip.id), axis, audio_map, clip.duration_s),
            "text": text_signal(text.get(clip.id), axis),
            "prior": priors[clip.id].get(axis),
        }

    # Audio LLM verification of a shortlist per axis.
    verify_axes = [a for a in axes if a.id in set(s.get("audio_llm_axes", [a.id for a in axes]))]
    top, low = int(s.get("verify_top", 150)), int(s.get("verify_low", 50))
    hearable = [c for c in clips if c.duration_s <= models.AUDIO_LLM_MAX_S]
    ratings, shortlists = {}, {}
    pad_dir = tempfile.TemporaryDirectory()
    for a in verify_axes:
        prelim = {c.id: fusion.fuse(signals(c, a.id), weights)["value"] for c in hearable}
        shortlists[a.id] = shortlist(prelim, top, low)
        picked = sorted((c for c in hearable if c.id in shortlists[a.id]["all"]), key=lambda c: c.duration_s)
        saved = Checkpoint(f"verify_{a.id}")
        if saved.todo(picked):
            log(f"verifying {a.id}: {len(picked)} clips")
            audio = {c.id: c.audio if fake else models.padded(c, Path(pad_dir.name)) for c in picked}
            saved.run(
                picked,
                lambda part: llm().json(
                    [models.verify_prompt(c, a, audio[c.id]) for c in part],
                    16,
                    [max(c.duration_s, models.AUDIO_LLM_MIN_S) for c in part],
                ),
                chunk=64,
            )
        ratings[a.id] = {i: rating(r) for i, r in saved.run(picked, None).items()}
    gemma = None
    free()

    labels = []
    for clip in clips:
        caption = heard.get(clip.id, {})
        scores = {}
        for a in axes:
            sig = signals(clip, a.id)
            r = ratings.get(a.id, {}).get(clip.id)
            sig["audio_llm"] = None if r is None else ((r - 1) / 4, AUDIO_LLM_CONFIDENCE)
            scores[a.id] = fusion.fuse(sig, weights)
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
    summary = report(labels, axes, unparsed) | {
        "embeddings": embeddings,
        "verification": verification_report(ratings, shortlists),
    }
    summary["confounds"] = {
        a.id: {n: checks.confounds([l["scores"][a.id]["signals"].get(n) if n != "fused" else l["scores"][a.id]["value"]
                                    for l in labels], clips)
               for n in ("fused", *fusion.SIGNALS)}
        for a in axes
    }  # fmt: skip
    summary["data"] = data_counts(labels, clips, axes)
    (out / "fusion.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    log(f"label: wrote {len(labels)} labels, {unparsed} unparsed judge replies")

    all_zero = sum(1 for r in text.values() if all((text_signal(r, a.id) or (0.0,))[0] == 0 for a in axes))
    results = [
        checks.Check("judge: unparsed share", round(unparsed / max(1, len(readable)), 3), 0.1),
        checks.Check("judge: share of lines scored 0 on every axis", round(all_zero / max(1, len(text)), 3), 0.8,
                     note="the judge defaulting to neutral, as on the first smoke test"),  # fmt: skip
    ]
    for axis, v in summary["verification"].items():
        rated = v["top"] + v["low"]
        results.append(checks.Check(f"verify {axis}: unparsed share", round(v["unparsed"] / max(1, rated), 3), 0.25))
    for a in axes:
        results += checks.confound_checks(f"{a.id} fused", summary["confounds"][a.id]["fused"])
    checks.finish(out, results, s.get("checks", "warn"))


def shortlist(prelim, top, low):
    """The `top` clips with the highest preliminary score, and `low` clips
    picked by id hash from the lower half, as a control."""
    ranked = sorted(prelim, key=lambda i: (-prelim[i], i))
    high = ranked[:top]
    rest = ranked[max(top, len(ranked) // 2) :]
    control = sorted(rest, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:low]
    return {"top": high, "low": control, "all": set(high) | set(control)}


def rating(reply):
    """A 1-5 rating from the audio LLM's reply, or None."""
    if not isinstance(reply, dict):
        return None
    try:
        r = round(float(reply.get("rating")))
    except (TypeError, ValueError):
        return None
    return r if 1 <= r <= 5 else None


def verification_report(ratings, shortlists):
    """Per axis: mean rating on the top of the shortlist vs the low control.
    If the audio LLM can hear the axis, the top should rate clearly higher;
    if they rate the same, its ratings there are noise."""
    out = {}
    for axis, r in ratings.items():
        tops = [r[i] for i in shortlists[axis]["top"] if r.get(i) is not None]
        lows = [r[i] for i in shortlists[axis]["low"] if r.get(i) is not None]
        mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
        out[axis] = {
            "top": len(shortlists[axis]["top"]),
            "low": len(shortlists[axis]["low"]),
            "unparsed": sum(v is None for v in r.values()),
            "mean_rating_top": mean(tops),
            "mean_rating_low": mean(lows),
            "separation": None if not tops or not lows else round(mean(tops) - mean(lows), 2),
        }
    return out


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


def data_counts(labels, clips, axes):
    """Per axis, how many clips the fused score puts at 0.6+ and 0.4+, overall
    and by game, outside barks: whether the axis has enough candidates for a
    contrast set before anything is extracted."""
    game = {c.id: c.game for c in clips}
    bark = {c.id for c in clips if c.bark}
    out = {}
    for a in axes:
        rows = [(l["scores"][a.id]["value"], game[l["id"]]) for l in labels if l["id"] not in bark]
        by_game = {}
        for v, g in rows:
            if v >= 0.6:
                by_game[g] = by_game.get(g, 0) + 1
        out[a.id] = {"at_0.6": sum(v >= 0.6 for v, _ in rows), "at_0.4": sum(v >= 0.4 for v, _ in rows),
                     "at_0.6_by_game": by_game, "of": len(rows)}  # fmt: skip
    return out


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
