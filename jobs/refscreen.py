"""refscreen stage: which of her own clips make the best voice-clone reference for each mood?
GPU, manual, one rental. Brandon, 2026-10-29: her own mood clip as the ICL example (voicearms' swap)
carried the mood as well as any steering arm, but voicearms only ever tried one clip per mood.

Candidates per mood: every positive of the mood's pairs in voice_sets.json (her trusted clips, effect
clips already out) that is 3-15 s long and not in a held-out conversation (voicearms' split, so its
fits stay honest). `exclude` (clip ids) drops lines that aren't normal talking, such as screams, pain
or combat callouts, from the candidates and from her neutral clips. Each speaks the first `screen_sentences` test sentences as the ICL example with its
own x-vector, exactly voicearms' swap; her neutral voice (voicearms' N-icl: the reference clip with the
mean neutral x-vector) speaks the same sentences as the baseline.

Scores per candidate, means over its sentences: the probe's lift for the mood over neutral, the share
of sentences lifted, voice similarity to her neutral clips, Whisper WER, and the suit gap against her
mood clips' spectrum. A candidate "holds" when similarity is within the mood's allowance of neutral
(voicearms' rule) and WER <= max_wer. Holding candidates rank by lift. Similarity to the reference clip
itself is deliberately not a score: it rewards copying the clip, not carrying the mood.

The ranking only shortlists; Brandon's ears decide. shortlist/<mood>/ is a clip list for the review
hub: the top `shortlist` holding candidates (`shortlist_weak` for a mood whose best candidate lifts
less than min_lift, where the probe can't rank), plus `controls` low-ranked holding ones as a check,
in a shuffled order and blind. Each candidate brings its reference clip (her real line, to tag the
sub-mood in the note) and `listen_sentences` of its outputs. Answers: how much of the mood, 0-3.

Outputs: report.json, report.md, audio/<mood>/<clip id>/<n>.wav, shortlist/<mood>/{clips.toml, wav/}.
Work goes to VI_CHECKPOINT as it's made, so a rerun on a new worker resumes.
"""

import json
import os
import random
import shutil
from pathlib import Path

import steer
import voicearms
import voicesteer
from voicesteer import Probe, mean
from vi_job import ensure_venv, fake_models, load_clips, log, out_dir, settings, unit_hash


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
    moods = s.get("moods") or list(sets["moods"])
    ref_len = (float(s.get("ref_min_s", 3.0)), float(s.get("ref_max_s", 15.0)))
    n_sent = int(s.get("screen_sentences", 4))
    tests = list(range(n_sent))
    max_drop, max_wer = float(s.get("max_similarity_drop", 0.05)), float(s.get("max_wer", 0.15))
    min_lift = float(s.get("min_lift", 0.1))
    n_refs = int(s.get("refs_per_side", 32))
    max_cands = int(s.get("max_candidates", 0)) or None
    excluded = set(s.get("exclude", []))

    clips = {c.id: c for c in load_clips()}
    neutral = [clips[i] for i in sets["neutral"] if i in clips and i not in excluded]
    if not neutral:
        raise SystemExit("refscreen: none of the neutral clips are in harvest")
    voice = voicearms.reference(neutral, ref_len)

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
        for n in tests:
            path = keep / "audio" / mood / cond / f"{n}.wav"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tts.speak(spec, voicearms.TEST[n], steer.seed(n), path)
            row = cached(keep / "scores" / mood / cond / f"{n}.json",
                         lambda p=path, n=n: scorer(p, voicearms.TEST[n], mood, probe, her))
            rows.append({**row, "suit": cached(keep / "suit" / mood / cond / f"{n}.json", lambda p=path: voicearms.suit_curve(p))})
        return rows

    xn = np.array([cached(keep / "xvec" / f"{c.id}.json", lambda c=c: tts.xvector(c)) for c in neutral])
    x_mean = xn.mean(0)
    x_voice = (x_mean * np.linalg.norm(xn, axis=1).mean() / max(np.linalg.norm(x_mean), 1e-12)).tolist()
    base = {"ref": voice, "mode": "icl", "xvec": x_voice}
    base_rows = speak("_neutral", "N-icl", base)
    base_sim = mean([r["similarity"] for r in base_rows])
    log(f"refscreen: {len(moods)} moods, {n_sent} sentences, neutral from {voice.id} (similarity {base_sim:.3f})")

    report = {"model": s.get("model", voicearms.MODEL), "voice": voice.id, "sentences": tests,
              "neutral_similarity": base_sim,
              "settings": {"ref_seconds": ref_len, "max_similarity_drop": max_drop, "max_wer": max_wer,
                           "min_lift": min_lift, "max_candidates": max_cands},
              "moods": {}}
    for mood in moods:
        pairs = [p for p in sets["moods"][mood]["pairs"] if p[0] in clips and p[1] in clips]
        train = [p for p in pairs if not voicearms.held_out(clips[p[0]])]
        cands = [p for p, _ in train if ref_len[0] <= clips[p].duration_s <= ref_len[1] and p not in excluded][:max_cands]
        # voicearms' similarity allowance: her own real mood clips' drop from her neutral voice, + 0.02
        sim_p = [cached(keep / "simclip" / f"{p}.json", lambda c=clips[p]: voicearms.voicesteer_cos(scorer, c, her)) for p, _ in train[:n_refs]]
        sim_z = [cached(keep / "simclip" / f"{z}.json", lambda c=clips[z]: voicearms.voicesteer_cos(scorer, c, her)) for _, z in train[:n_refs]]
        allow = max(max_drop, round(mean(sim_z) - mean(sim_p), 4) + 0.02) if train else max_drop
        curves = [cached(keep / "suit" / "clips" / f"{p}.json", lambda c=clips[p]: voicearms.suit_curve(c.audio)) for p, _ in train[:n_refs]]
        curves = [c for c in curves if c]
        suit_mood = np.mean(curves, 0).tolist() if curves else None
        log(f"refscreen: {mood}: {len(cands)} candidates of {len(train)} clips")

        rows = []
        for cid in cands:
            got = speak(mood, cid, {"ref": clips[cid], "mode": "icl"})
            lifts = [g["probe"][mood] - b["probe"][mood] for g, b in zip(got, base_rows, strict=True)]
            sim, wer = mean([g["similarity"] for g in got]), mean([g["wer"] for g in got])
            rows.append({"clip": cid, "seconds": round(clips[cid].duration_s, 2), "lift": round(mean(lifts), 4),
                         "share": sum(x > 0 for x in lifts) / len(lifts), "similarity": round(sim, 4), "wer": round(wer, 4),
                         "suit_gap_db": voicearms.suit_gap(got, suit_mood) if suit_mood else None,
                         "holds": sim >= base_sim - allow and wer <= max_wer})
        held = sorted([r for r in rows if r["holds"]], key=lambda r: (-r["lift"], r["clip"]))
        for k, r in enumerate(held, 1):
            r["rank"] = k
        weak = not held or held[0]["lift"] < min_lift
        report["moods"][mood] = {"candidates": len(rows), "holding": len(held), "similarity_allowance": allow,
                                 "probe_weak": weak, "ranked": held,
                                 "dropped": sorted([r for r in rows if not r["holds"]], key=lambda r: r["clip"])}
        log(f"refscreen: {mood}: {len(held)} hold; best {held[0]['clip'] + ' ' + format(held[0]['lift'], '+.3f') if held else '-'}"
            f"{' (probe weak)' if weak else ''}")

    tts.close()
    shutil.copytree(keep / "audio", out / "audio", dirs_exist_ok=True)
    shortlists(report, out, clips, s)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(report))


def shortlists(r, out, clips, s):
    """shortlist/<mood>/clips.toml for the review hub: top candidates and a few controls, blind."""
    top, top_weak, n_ctl = int(s.get("shortlist", 5)), int(s.get("shortlist_weak", 8)), int(s.get("controls", 2))
    listen = list(range(int(s.get("listen_sentences", 2))))
    root = out / "shortlist"
    shutil.rmtree(root, ignore_errors=True)
    for mood, x in sorted(r["moods"].items()):
        ranked = x["ranked"]
        n = top_weak if x["probe_weak"] else top
        picked = [(c, f"rank {c['rank']}") for c in ranked[:n]]
        picked += [(c, f"control, rank {c['rank']} of {len(ranked)}") for c in ranked[n:][-n_ctl:]]
        if not picked:
            continue
        random.Random(unit_hash("refscreen", mood)).shuffle(picked)
        d = root / mood
        (d / "wav").mkdir(parents=True)
        lines = [f'title = "Reference screening: {mood}"',
                 f'question = "How much {mood} do you hear (0 none, 3 clearly)? First her real line, then two lines '
                 f'spoken with it as the example. In the note on her real line, tag its sub-mood (e.g. sarcastic)."',
                 'answers = ["0", "1", "2", "3"]', "blind = true", 'asked_by = "TTS steering"', ""]
        for c, label in picked:
            lift = f"lift {c['lift']:+.3f}, similarity {c['similarity']:.3f}"
            lines += ["[[clip]]", f'id = "{c["clip"]}"', f'label = "{label}: her real line ({c["seconds"]} s; {lift})"', ""]
            for n in listen:
                name = f"{c['clip']}-{n}.wav"
                shutil.copy(out / "audio" / mood / c["clip"] / f"{n}.wav", d / "wav" / name)
                lines += ["[[clip]]", f'path = "wav/{name}"', f'label = "{label}: spoken with it"',
                          f'text = "{voicearms.TEST[n]}"', ""]
        (d / "clips.toml").write_text("\n".join(lines))


def render(r):
    lines = ["# Reference screening (refscreen)", "",
             f"Model {r['model']}. Neutral: {r['voice']} with the mean neutral x-vector, similarity "
             f"{r['neutral_similarity']:.3f}. Each candidate speaks sentences {r['sentences']} as the ICL example with its "
             "own x-vector. Lift: the mood's probe over neutral. Holding: similarity within the mood's allowance, "
             f"WER <= {r['settings']['max_wer']}. The ranking only shortlists: shortlist/<mood>/ is for Brandon's ears.", ""]
    for mood, x in sorted(r["moods"].items()):
        lines += [f"## {mood}{' (probe weak: the ranking means little)' if x['probe_weak'] else ''}", "",
                  f"{x['candidates']} candidates, {x['holding']} hold (allowance {x['similarity_allowance']}).", "",
                  "| rank | clip | s | lift | share | similarity | WER | suit gap dB |", "|---|---|---|---|---|---|---|---|"]
        for c in x["ranked"]:
            lines.append(f"| {c['rank']} | {c['clip']} | {c['seconds']} | {c['lift']:+.3f} | {c['share']:.2f} | "
                         f"{c['similarity']:.3f} | {c['wer']:.3f} | {c['suit_gap_db']} |")
        if x["dropped"]:
            lines += ["", "Dropped (voice or words lost): " + ", ".join(
                f"{c['clip']} (sim {c['similarity']:.3f}, WER {c['wer']:.3f})" for c in x["dropped"])]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
