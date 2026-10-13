"""Turning each signal's opinion into one score per axis.

Standard library only: it runs in CI and is tested by test_fusion.py.

Signals (docs/design/corpus-format.md, label):
  audio      acoustic emotion models (emotion2vec+), which hear the delivery
  text       the LLM judge reading the line and its context
  prior      scene tags and LE3 stage directions
  audio_llm  the audio LLM's no-transcript scores from discover
Each gives, per axis, a value and its own confidence, or nothing.
"""

import math
import re

SIGNALS = ("audio", "text", "prior", "audio_llm")

DEFAULT_WEIGHTS = {"audio": 1.0, "text": 1.0, "prior": 0.5, "audio_llm": 0.5}

# emotion2vec+ classes that speak for each pack variable. Axes with no entry
# (playfulness, intoxication) get no audio signal: no acoustic model here
# hears teasing or slurring, so those come from the text judge, priors and
# the audio LLM.
DEFAULT_AUDIO_MAP = {
    "joy": {"happy": 1.0},
    "sadness": {"sad": 1.0},
    "anger": {"angry": 1.0, "disgusted": 0.5},
    "worry": {"fearful": 1.0},
}

DEFAULT_SCENE_PRIORS = {
    "citadel_party": {"intoxication": 0.5, "joy": 0.4},
    "combat": {"intoxication": 0.0},
}

# Words in an LE3 stage direction, and what they suggest.
DEFAULT_DIRECTION_PRIORS = [
    (r"\b(sad|sadly|grief|griev|mourn|tearful|crying|upset|somber)", {"sadness": 0.75}),
    (r"\b(angry|angrily|furious|annoyed|irritat|frustrat|snap)", {"anger": 0.75}),
    (r"\b(worr|concern|nervous|anxious|afraid|scared|fear|urgent|panic)", {"worry": 0.75}),
    (r"\b(laugh|amused|teas|joking|playful|wry|sarcas|flirt)", {"playfulness": 0.7, "joy": 0.5}),
    (r"\b(happy|happily|excited|pleased|delighted|cheer|warm)", {"joy": 0.7}),
    (r"\b(drunk|tipsy|slurr|intoxicated)", {"intoxication": 0.85}),
]

SCENE_PRIOR_CONFIDENCE = 0.5
DIRECTION_PRIOR_CONFIDENCE = 0.6


def audio_signal(probs, axis, audio_map, duration_s):
    """emotion2vec+ class probabilities to one axis: the weighted sum of the
    classes mapped to it. Clips under a second are too short to trust much."""
    mapping = audio_map.get(axis)
    if not mapping or probs is None:
        return None
    value = min(1.0, sum(w * probs.get(cls, 0.0) for cls, w in mapping.items()))
    confidence = 0.4 if duration_s < 1.0 else 0.8
    return value, confidence


def priors(scene_tags, stage_direction, scene_priors, direction_priors):
    """Axis -> (value, confidence) from scene tags and stage directions. A
    direction beats a scene tag for the same axis, being about this line."""
    out = {}
    for tag in scene_tags:
        for axis, value in scene_priors.get(tag, {}).items():
            out[axis] = (value, SCENE_PRIOR_CONFIDENCE)
    if stage_direction:
        text = stage_direction.lower()
        for pattern, axes in direction_priors:
            if re.search(pattern, text):
                for axis, value in axes.items():
                    out[axis] = (value, DIRECTION_PRIOR_CONFIDENCE)
    return out


def fuse(signals, weights):
    """signals: name -> (value, confidence) or None. Returns the record the
    label stage writes for one axis.

    value is the confidence-weighted mean. confidence is how much evidence
    there is (1 - exp(-total weight)) times how well the signals agree
    (1 - 2 x their weighted spread), so lines where audio and text disagree
    come first in review.
    """
    total = 0.0
    acc = 0.0
    present = []
    for name, sig in signals.items():
        if sig is None:
            continue
        v, c = sig
        w = weights.get(name, 0.0) * c
        if w <= 0:
            continue
        total += w
        acc += w * v
        present.append((v, w))
    out = {"signals": {n: (None if s is None else round(s[0], 4)) for n, s in signals.items()}}
    if total == 0:
        out.update(value=0.0, confidence=0.0)
        return out
    value = acc / total
    spread = math.sqrt(sum(w * (v - value) ** 2 for v, w in present) / total)
    agreement = 1.0 - min(1.0, 2.0 * spread)
    support = 1.0 - math.exp(-total)
    out.update(value=round(value, 4), confidence=round(support * agreement, 4))
    return out


def clamp01(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return min(1.0, max(0.0, x))


def tags_for(flags, discover_tags, duration_s):
    """Non-emotion labels, used as filters by contrast sets."""
    tags = set(discover_tags)
    if "bark" in flags:
        tags.add("combat_bark")
    if "nonverbal" in flags:
        tags.add("nonverbal")
    if duration_s < 1.0:
        tags.add("short")
    return sorted(tags)


def spearman(xs, ys):
    """Rank correlation, or None with fewer than 8 pairs."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 8:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None
    return round(cov / (vx * vy), 3)
