"""context stage: one record per line of hers with words, holding the line and
the conversation around it as every text model reads it. Home, CPU, standard
library only. Deterministic, so textscore's models all see the same input and
the GPU worker gets only this small file, not the audio.

Plan: docs/design/corpus-pipeline.md, "Seeded moods". Format:
docs/design/corpus-format.md, context.

From harvest per line: the text (clean's transcript isn't used, so a clean
rerun doesn't rerun the GPU text models), up to `before` lines of the canonical
path before it, up to `after` of the lines that can follow it, the listener,
the non-structural scene tags, the LE3 director's note and the conversation
kind. `prompt` renders them for the models. Lines without words (barks with
no subtitle, sighs) are left out.

Outputs: context.jsonl, context.json.
Settings: before (default 3), after (default 2), character (default "Tali").
"""

import json

from vi_job import load_clips, log, out_dir, settings, write_jsonl

# Structural tags say how the conversation is built, not where it happens.
STRUCTURAL = {"one_liner", "exchange", "dialogue_choices", "to_shepard", "to_squad", "to_npc"}


def name(speaker):
    """`admiral_raan` -> `Admiral Raan`."""
    return " ".join(w.capitalize() for w in str(speaker or "someone").replace("-", "_").split("_") if w)


def record(clip, before, after, character):
    line = clip.line
    source = line.get("source", {})
    prev = [{"speaker": name(c.get("speaker")), "text": c.get("text", "")} for c in clip.context if c.get("text")]
    prev = prev[-before:] if before else []
    nxt = [
        {"speaker": name(c.get("speaker")), "text": c.get("text", "")}
        for c in line.get("following", [])
        if c.get("text")
    ]
    nxt = nxt[:after]
    listener = source.get("listener")
    scenes = [t for t in clip.scene_tags if t not in STRUCTURAL]
    r = {
        "id": clip.id,
        "game": clip.game,
        "conversation": source.get("conversation"),
        "text": clip.text,
        "before": prev,
        "after": nxt,
        "listener": name(listener) if listener else None,
        "scene_tags": scenes,
        "stage_direction": clip.stage_direction,
        "conversation_kind": source.get("line_kind") or source.get("conversation_kind"),
        "bark": clip.bark,
    }
    r["prompt"] = render(r, character)
    return r


def render(r, character):
    """The one text every model reads: the situation, then the lines before,
    her line marked, then what can follow."""
    out = []
    if r["scene_tags"]:
        out.append("Scene: " + ", ".join(t.replace("_", " ") for t in r["scene_tags"]))
    if r["bark"]:
        out.append("A short line she calls out in combat or while exploring.")
    if r["stage_direction"]:
        out.append(f"Voice director's note: {r['stage_direction']}")
    if r["before"]:
        out.append("Before:")
        out += [f"{c['speaker']}: {c['text']}" for c in r["before"]]
    to = f" (to {r['listener']})" if r["listener"] else ""
    out.append(f">>> {character}{to}: {r['text']}")
    if r["after"]:
        out.append("After (one of):" if len(r["after"]) > 1 else "After:")
        out += [f"{c['speaker']}: {c['text']}" for c in r["after"]]
    return "\n".join(out)


def main():
    s = settings()
    out = out_dir()
    before, after = int(s.get("before", 3)), int(s.get("after", 2))
    character = s.get("character", "Tali")
    clips = load_clips()
    records = [record(c, before, after, character) for c in clips if c.has_words and "no_subtitle" not in c.flags]
    write_jsonl(out / "context.jsonl", records)
    report = {
        "lines": len(records),
        "left_out_no_words": len(clips) - len(records),
        "with_before": sum(bool(r["before"]) for r in records),
        "with_after": sum(bool(r["after"]) for r in records),
        "with_stage_direction": sum(bool(r["stage_direction"]) for r in records),
        "settings": {"before": before, "after": after, "character": character},
    }
    (out / "context.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log(f"context: {len(records)} lines, {report['with_before']} with lines before")


if __name__ == "__main__":
    main()
