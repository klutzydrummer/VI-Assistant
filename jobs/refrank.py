"""refrank stage: which of her own clips make the best example for each mood? (Brandon, 2026-10-29)

By ear, her real mood clip as the ICL example (voicearms' swap, S) carried mood about as well as
any steering, but each run only ever tried one clip per mood. Here every usable positive clip of
hers for a mood (3-15 s, effect clips already out of voice_sets, held-out conversations excluded)
is the example in turn, with its own speaker token as S has, and speaks the same few test
sentences. Each output is scored the way voicearms scores it: test D's probe lift over neutral
(not similarity to the example, which rewards copying), ECAPA voice similarity, Whisper WER and
the suit gap to her real mood clips.

The ranking: clips whose outputs keep her voice and words first, then by probe lift. A clip whose
outputs lose voice or words, or don't lift the probe, is marked excluded: a candidate to leave out
of the pairs A and D are built from. For a mood the probe can't hear (the best clip's lift under
min_lift) the ranking is only a shortlist, and his ears decide.

For his ears, two hub clip lists:
  shortlist/   one generated sentence from each of the top `shortlist` clips per mood (more for a
               probe-weak mood) and the bottom `controls`, blind, rank hidden: does it carry the mood?
  submoods/    her real top clips themselves, to tag a sub-mood or tone in the note (e.g. sarcastic
               indignation), so later vectors and examples aren't averaged across flavours.

Inputs (VI_FILES): voice_sets.json and voice_probe.npz, as voicearms'. Outputs: ranking.json,
ranking.md, audio/<mood>/<clip>/<n>.wav, shortlist/ and submoods/. Work goes to VI_CHECKPOINT as it's
made, so a rerun on a new worker resumes.
"""

import json
import os
import shutil
from pathlib import Path

import steer
import voicearms
import voicesteer
from voicearms import TEST, reference, suit_curve, suit_gap
from vi_job import ensure_venv, fake_models, load_clips, log, out_dir, settings, unit_hash
from voicesteer import Probe, mean


def main():
    import numpy as np

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
    moods = list(s["moods"]) if "moods" in s else list(sets["moods"])
    ref_len = (float(s.get("ref_min_s", 3.0)), float(s.get("ref_max_s", 15.0)))
    sentences = list(range(int(s.get("rank_sentences", 4))))
    most = int(s.get("max_candidates", 100))
    max_drop, max_wer = float(s.get("max_similarity_drop", 0.05)), float(s.get("max_wer", 0.15))
    min_lift = float(s.get("min_lift", 0.1))
    n_short, n_weak, n_ctrl = int(s.get("shortlist", 5)), int(s.get("shortlist_weak", 8)), int(s.get("controls", 2))
    own = str(s.get("example_xvec", "own"))
    if own not in ("own", "neutral"):
        raise SystemExit(f"refrank: example_xvec {own!r} is not own or neutral")

    clips = {c.id: c for c in load_clips()}
    neutral = [clips[i] for i in sets["neutral"] if i in clips]
    if not neutral:
        raise SystemExit("refrank: none of the neutral clips are in harvest")
    voice = reference(neutral, ref_len)
    tts = voicearms.FakeArmTTS() if fake else voicearms.ArmTTS(s.get("model", voicearms.MODEL), s.get("device", "cuda:0"))
    scorer = voicearms.FakeScorer() if fake else voicesteer.Scorer(s)
    her = scorer.her(neutral)

    def cached(path, make):
        if path.exists():
            return json.loads(path.read_text())
        v = make()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(v))
        return v

    def speak(mood, cond, spec):
        rows = []
        for n in sentences:
            path = keep / "audio" / mood / cond / f"{n}.wav"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tts.speak(spec, TEST[n], steer.seed(n), path)
            row = cached(keep / "scores" / mood / cond / f"{n}.json", lambda p=path, n=n: scorer(p, TEST[n], mood, probe, her))
            rows.append({**row, "suit": cached(keep / "suit" / mood / cond / f"{n}.json", lambda p=path: suit_curve(p))})
        return rows

    # Her neutral voice, as voicearms speaks it: the reference clip with the neutral clips' mean x-vector.
    xn = np.array([cached(keep / "xvec" / f"{c.id}.json", lambda c=c: tts.xvector(c)) for c in neutral])
    x_mean = xn.mean(0)
    x_voice = (x_mean * np.linalg.norm(xn, axis=1).mean() / max(np.linalg.norm(x_mean), 1e-12)).tolist()
    neutral_rows = speak("_neutral", "N-icl", {"ref": voice, "mode": "icl", "xvec": x_voice})
    base_sim = mean([r["similarity"] for r in neutral_rows])
    report = {"model": s.get("model", voicearms.MODEL), "voice": voice.id, "sentences": sentences,
              "example_xvec": own, "settings": {"max_similarity_drop": max_drop, "max_wer": max_wer, "min_lift": min_lift,
                                                "max_candidates": most, "ref_seconds": list(ref_len)},
              "neutral_similarity": base_sim, "moods": {}}

    for mood in moods:
        pairs = [p for p in sets["moods"][mood]["pairs"] if p[0] in clips and p[1] in clips]
        seen, cands = set(), []
        for p, _ in pairs:
            c = clips[p]
            if p in seen or voicearms.held_out(c) or not ref_len[0] <= c.duration_s <= ref_len[1]:
                continue
            seen.add(p)
            cands.append(c)
        cands = cands[:most]
        log(f"refrank: {mood}: {len(cands)} candidate clips")
        # The similarity allowance, as voicearms': her own mood clips' drop from her neutral voice, + 0.02.
        sim_p = [cached(keep / "simclip" / f"{p}.json", lambda c=clips[p]: voicearms.voicesteer_cos(scorer, c, her)) for p, _ in pairs[:32]]
        sim_z = [cached(keep / "simclip" / f"{z}.json", lambda c=clips[z]: voicearms.voicesteer_cos(scorer, c, her)) for _, z in pairs[:32]]
        allow = max(max_drop, round(mean(sim_z) - mean(sim_p), 4) + 0.02) if pairs else max_drop
        suit_mood = np.mean([v for v in (cached(keep / "suit" / "clips" / f"{c.id}.json", lambda c=c: suit_curve(c.audio))
                                         for c in cands[:32]) if v], 0).tolist() if cands else None
        rows_by = {}
        for c in cands:
            spec = {"ref": c, "mode": "icl"} if own == "own" else {"ref": c, "mode": "icl", "xvec": x_voice}
            rows = speak(mood, c.id, spec)
            lifts = [r["probe"][mood] - b["probe"][mood] for r, b in zip(rows, neutral_rows, strict=True)]
            sim, wer = mean([r["similarity"] for r in rows]), mean([r["wer"] for r in rows])
            holds = sim >= base_sim - allow and wer <= max_wer
            rows_by[c.id] = {"clip": c.id, "seconds": round(c.duration_s, 2), "lift": round(mean(lifts), 4),
                             "share": round(sum(x > 0 for x in lifts) / len(lifts), 3), "similarity": round(sim, 4),
                             "wer": round(wer, 4), "holds": holds,
                             "suit_gap_db": suit_gap(rows, suit_mood) if suit_mood is not None else None}
        ranked = sorted(rows_by.values(), key=lambda r: (not r["holds"], -r["lift"], r["clip"]))
        for k, r in enumerate(ranked, 1):
            r["rank"] = k
            r["excluded"] = not r["holds"] or r["lift"] <= 0
        weak = not ranked or ranked[0]["lift"] < min_lift
        report["moods"][mood] = {"candidates": len(ranked), "similarity_allowance": round(allow, 4), "probe_weak": weak,
                                 "excluded": sum(r["excluded"] for r in ranked), "ranking": ranked}
        log(f"refrank: {mood}: best {ranked[0]['clip'] if ranked else '-'}, {report['moods'][mood]['excluded']} excluded"
            f"{' (probe weak)' if weak else ''}")

    tts.close()
    shutil.copytree(keep / "audio", out / "audio", dirs_exist_ok=True)
    (out / "ranking.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (out / "ranking.md").write_text(render(report))
    clip_lists(report, out, n_short, n_weak, n_ctrl, s.get("asked_by", "Modeling Tali's suit effect"))


def picks(ranked, top, ctrl):
    """The top clips and, as a check that the ranking means anything, the bottom few that aren't in the top."""
    head = ranked[:top]
    tail = [r for r in ranked[top:][-ctrl:]] if ctrl else []
    return head, tail


def clip_lists(r, out, n_short, n_weak, n_ctrl, asked_by):
    """Two hub clip lists: generated outputs to judge blind, and her real top clips to tag a sub-mood."""
    short, sub = [], []
    for mood, m in sorted(r["moods"].items()):
        head, tail = picks(m["ranking"], n_weak if m["probe_weak"] else n_short, n_ctrl)
        for x in head + tail:
            short.append((mood, x, x in tail))
        sub += [(mood, x) for x in head]
    sentence = r["sentences"][0]
    root = out / "shortlist"
    shutil.rmtree(root, ignore_errors=True)
    (root / "wav").mkdir(parents=True)
    lines = ['title = "Best example clips per mood (refrank)"',
             'question = "Her voice with one of her own clips as the example. Does it carry the mood named under it?"',
             'answers = ["clearly", "a little", "no", "copies the example"]', "blind = true", f"asked_by = {json.dumps(asked_by)}", ""]
    order = sorted(short, key=lambda t: unit_hash("refrank", t[0], t[1]["clip"]))
    for k, (mood, x, control) in enumerate(order, 1):
        src = out / "audio" / mood / x["clip"] / f"{sentence}.wav"
        name = f"wav/{k:03d}.wav"
        shutil.copy(src, root / name)
        lines += ["[[clip]]", f'path = "{name}"',
                  f'label = "{mood}: rank {x["rank"]} of {r["moods"][mood]["candidates"]}{" (low-ranked check)" if control else ""}"',
                  f"text = {json.dumps(f'{mood}: ' + TEST[sentence])}", ""]
    (root / "clips.toml").write_text("\n".join(lines))
    root = out / "submoods"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    lines = ['title = "Sub-moods of her top example clips (refrank)"',
             'question = "Her real line. Is it the plain mood named under it, or a different flavour? Name the flavour in the note (e.g. sarcastic indignation)."',
             'answers = ["plain", "different flavour", "not this mood"]', "blind = false", f"asked_by = {json.dumps(asked_by)}", ""]
    for mood, x in sub:
        lines += ["[[clip]]", f'id = "{x["clip"]}"', f'label = "{mood}, rank {x["rank"]}"', ""]
    (root / "clips.toml").write_text("\n".join(lines))


def render(r):
    lines = ["# Her example clips, ranked per mood (refrank)", "",
             f"Each candidate is the ICL example ({'its own' if r['example_xvec'] == 'own' else 'the neutral'} speaker token) "
             f"for sentences {r['sentences']}. Ranked: keeps voice and words first, then probe lift over neutral. "
             "Excluded: loses voice or words, or doesn't lift the probe.", "",
             "| mood | candidates | excluded | best (lift, similarity, suit gap) | probe |", "|---|---|---|---|---|"]
    for m, x in sorted(r["moods"].items()):
        b = x["ranking"][0] if x["ranking"] else None
        best = f"{b['clip']} ({b['lift']:+.3f}, {b['similarity']}, {b['suit_gap_db']} dB)" if b else "-"
        lines.append(f"| {m} | {x['candidates']} | {x['excluded']} | {best} | {'weak: ears decide' if x['probe_weak'] else 'ok'} |")
    for m, x in sorted(r["moods"].items()):
        lines += ["", f"## {m}", "", "| rank | clip | s | lift | share | similarity | WER | suit gap dB | |", "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {c['rank']} | {c['clip']} | {c['seconds']} | {c['lift']:+.3f} | {c['share']} | {c['similarity']} | {c['wer']} | "
                  f"{c['suit_gap_db']} | {'excluded' if c['excluded'] else ''} |" for c in x["ranking"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
