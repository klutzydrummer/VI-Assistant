"""voicearms stage: which way of steering Qwen3-TTS moves her voice? The protocol in
/mnt/project-files/research/steering-qwen3-tts.md, section 5. GPU, manual, one rental with every
branch decided here in advance.

Three arms, each built only from her clips (step 3's pairs: her trusted positives against
same-game matched zeros, effect clips already out), all tested on the same neutral reference,
sentences and seeds:
  A  activation steering: voicesteer's vectors (vectors/<mood>.npy, all talker layers; captured
     here the same way if absent) at single layers and a late band instead of voicesteer's 9-17,
     added with the residual rescaled to its norm before the add. At the best setting, also with
     a per-frame decay and in x-vector-only mode, and a norm-matched random vector (the control).
  B  reference-token arithmetic: the speaker embedding (x-vector, one prompt token) of her neutral
     reference plus alpha * tau, tau = mean x-vector of her positives - mean of their matched zeros.
     In voice-clone (ICL) mode and in x-vector-only mode.
  C  fitted reference soft token: with every weight frozen, one 2048-wide speaker token is fitted
     by gradient to her positive clips (teacher-forced codec tokens, the talker's loss plus 0.3 x
     the code predictor's), and another to their matched zeros; delta = the difference, used as
     her neutral reference's x-vector + w * delta. Both modes.
  D  residual-matched soft token (Brandon, 2026-10-21): the target is the per-layer residual shift her
     positive clips share while the frozen model reproduces them (teacher-forced, each with its own
     x-vector), minus the same for their matched zeros; one speaker token is fitted so that, on her
     zero clips, it moves the residual by that shift. Used like C. The report says how shared the
     shift is (each pair's shift against the target). One token per target type (`match_targets`,
     Brandon 2026-10-21: try them all): "mean" (the mean pair shift; arms D-<mode>), "pca:k" (the
     mean projected onto the top k principal directions of the pair shifts, which drops what only a
     few pairs carry; arms Dpcak-<mode>). "pca:1" is the single dominant shared direction.
Baselines: N (neutral reference, nothing changed, in each mode) and S (swap: her top positive clip
as the reference, the positive control for the probe on generated speech). Controls: a random
vector of the same size per layer for A, and a random token shift of the same size for B and C.

Strength: each arm starts at its natural full strength (vector x1, alpha 1, w 1) and steps down
the ladder only while her voice or the words go; `above` (strengths over 1, Brandon's call) is
tried only when full strength holds but doesn't lift the probe. Strengths are chosen on the first
half of the test sentences and the verdict is read on the second half.
Scores per line: test D's probe, ECAPA voice similarity, Whisper WER. Each mood's similarity
allowance is max(max_similarity_drop, how far her own real mood clips sit from her neutral voice
+ 0.02). Judge-free: the held-out likelihood gain, how much more each setting favours her held-out
positive clips over their zeros than N does (pairs held out by conversation).
Verdict per mood and arm: "steers" if the lift and share rules pass on the confirm half, voice
and words hold, it beats the random vector, and the likelihood gain's 95% interval is above 0.
If S itself doesn't lift the probe, the mood is "probe can't hear it" and Brandon's ears decide.

For his ears (Brandon, 2026-10-21): listen/ holds a short blind set, per mood one confirm-half
sentence for neutral, swap, the random control and every arm at its kept strength (and each blend),
renamed to numbers in a shuffled order, with her real example clip beside them and sheet.csv to fill
in. `python3 jobs/voicearms.py --unblind <out folder>` joins his sheet to the hidden key and writes
listen/results.md.

Inputs (VI_FILES): voice_sets.json and voice_probe.npz as voicesteer's, and voicesteer's vectors
bundled by `voicelayers.py --npz` as voice_vectors.npz (or vectors/<mood>.npy); a mood without one
is captured here the way voicesteer does. Outputs: report.json, report.md, listen.md,
audio/<mood>/<condition>/<n>.wav, tokens/<mood>.json (base, tau, delta; derived numbers, no
audio or text). Work goes to VI_CHECKPOINT as it's made, so a rerun on a new worker resumes.
"""

import json
import os
import shutil
from pathlib import Path

import steer
import voicesteer
from steer import TEST, seed, write_wav
from voicesteer import Probe, mean
from vi_job import ensure_venv, fake_models, load_clips, log, out_dir, settings, unit_hash

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LADDER = [1.0, 0.5, 0.25, 0.1, 0.05]


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
    blends = [list(b) for b in s.get("blends", [])]
    ladder = [float(x) for x in s.get("ladder", LADDER)]
    above = [float(x) for x in s.get("above", [])]
    above_tau = [float(x) for x in s.get("above_tau", [])]
    max_drop = float(s.get("max_similarity_drop", 0.05))
    max_wer = float(s.get("max_wer", 0.15))
    min_lift, min_share = float(s.get("min_lift", 0.1)), float(s.get("min_share", 0.75))
    targets = match_targets(s)
    n_refs = int(s.get("refs_per_side", 32))
    decay = float(s.get("decay", 0.985))
    for b in blends:
        missing = [m for m in b if m not in moods]
        if len(b) != 2 or missing:
            raise SystemExit(f"voicearms: blend {b} must name two of the tested moods")

    clips = {c.id: c for c in load_clips()}
    neutral = [clips[i] for i in sets["neutral"] if i in clips]
    if not neutral:
        raise SystemExit("voicearms: none of the neutral clips are in harvest")
    ref_len = (float(s.get("ref_min_s", 3.0)), float(s.get("ref_max_s", 15.0)))
    voice = reference(neutral, ref_len)
    tests = list(range(int(s.get("test_sentences", len(TEST)))))
    half = len(tests) // 2
    pick, confirm = tests[:half], tests[half:]
    log(f"voicearms: {len(moods)} moods, {len(blends)} blends, speaking from {voice.id} ({voice.duration_s:.1f} s)")

    tts = FakeArmTTS() if fake else ArmTTS(s.get("model", MODEL), s.get("device", "cuda:0"))
    scorer = FakeScorer() if fake else voicesteer.Scorer(s)
    layer_sets = [parse_layers(x, tts.n_layers) for x in s.get("layer_sets", ["14", "18", "21", "24", "21-25"])]
    her = scorer.her(neutral)

    def cached(path, make):
        if path.exists():
            return json.loads(path.read_text())
        v = make()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(v))
        return v

    def speak(mood, cond, spec, idxs):
        """Generate (or reuse) the lines and return their scores, in idxs order."""
        rows = []
        for n in idxs:
            path = keep / "audio" / mood / cond / f"{n}.wav"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tts.speak(spec, TEST[n], seed(n), path)
            rows.append(cached(keep / "scores" / mood / cond / f"{n}.json",
                               lambda p=path, n=n: scorer(p, TEST[n], mood, probe, her)))
        return rows

    base = {"icl": {"ref": voice, "mode": "icl"}, "xvec": {"ref": voice, "mode": "xvec"}}
    tts.measure = True  # residual norms per layer, read while the baselines speak
    neutral_rows = {m: speak("_neutral", f"N-{m}", base[m], tests) for m in base}
    tts.measure = False
    norms = cached(keep / "residual_norms.json", tts.residual_norms)
    base_sim = {m: mean([r["similarity"] for r in rows]) for m, rows in neutral_rows.items()}
    x_voice = cached(keep / "xvec" / f"{voice.id}.json", lambda: tts.xvector(voice))

    report = {"model": s.get("model", MODEL), "voice": voice.id, "voice_seconds": voice.duration_s, "layers": tts.n_layers,
              "settings": {"layer_sets": [f"{a[0]}-{a[-1]}" if len(a) > 1 else str(a[0]) for a in layer_sets],
                           "ladder": ladder, "above": above, "above_tau": above_tau, "decay": decay,
                           "max_similarity_drop": max_drop, "max_wer": max_wer, "min_lift": min_lift,
                           "min_share": min_share, "pick": pick, "confirm": confirm, "match_targets": targets},
              "residual_norms": norms, "moods": {}, "blends": {}}
    kept = {}

    for mood in moods:
        mset = sets["moods"][mood]
        pairs = [p for p in mset["pairs"] if p[0] in clips and p[1] in clips]
        held = [p for p in pairs if held_out(clips[p[0]])]
        train = [p for p in pairs if not held_out(clips[p[0]])][:n_refs]
        if not held:  # every pair in one conversation bucket: hold out the last fifth instead
            held, train = pairs[-max(1, len(pairs) // 5):], pairs[: len(pairs) - max(1, len(pairs) // 5)][:n_refs]
        log(f"voicearms: {mood}: {len(train)} pairs to build from, {len(held)} held out")

        # Her own real mood clips: how far they sit from her neutral voice sets the similarity allowance.
        sim_p = [cached(keep / "simclip" / f"{p}.json", lambda c=clips[p]: voicesteer_cos(scorer, c, her)) for p, _ in train]
        sim_z = [cached(keep / "simclip" / f"{z}.json", lambda c=clips[z]: voicesteer_cos(scorer, c, her)) for _, z in train]
        natural = round(mean(sim_z) - mean(sim_p), 4)
        allow = max(max_drop, natural + 0.02)

        def holds(rows, mode):
            sim, w = mean([r["similarity"] for r in rows]), mean([r["wer"] for r in rows])
            return sim >= base_sim[mode] - allow and w <= max_wer, sim, w

        def lifted(rows, mode, idxs):
            ref = [neutral_rows[mode][tests.index(n)]["probe"][mood] for n in idxs]
            lifts = [r["probe"][mood] - b for r, b in zip(rows, ref, strict=True)]
            return mean(lifts), sum(x > 0 for x in lifts) / len(lifts)

        def ladder_run(arm, mode, spec_at, ups):
            """Full strength first, down only while voice or words go; up through `ups` only if
            full strength holds but doesn't lift. Returns (kept strength or None, tried)."""
            tried, chosen = [], None
            for k in ladder:
                rows = speak(mood, f"{arm}@{k:g}", spec_at(k), pick)
                ok, sim, w = holds(rows, mode)
                lift, share = lifted(rows, mode, pick)
                tried.append({"strength": k, "lift": lift, "share": share, "similarity": sim, "wer": w, "holds": ok})
                if ok:
                    chosen = k
                    break
            if chosen == ladder[0] and tried[-1]["lift"] < min_lift:
                for k in ups:
                    rows = speak(mood, f"{arm}@{k:g}", spec_at(k), pick)
                    ok, sim, w = holds(rows, mode)
                    lift, share = lifted(rows, mode, pick)
                    tried.append({"strength": k, "lift": lift, "share": share, "similarity": sim, "wer": w, "holds": ok})
                    if not ok:
                        break
                    chosen = k
            return chosen, tried

        def confirmed(arm, mode, spec, k):
            rows = speak(mood, f"{arm}@{k:g}", spec, confirm)
            ok, sim, w = holds(rows, mode)
            lift, share = lifted(rows, mode, confirm)
            return {"strength": k, "lift": lift, "share": share, "similarity": sim, "wer": w, "holds": ok,
                    "probe": mean([r["probe"][mood] for r in rows]), "audio": f"{arm}@{k:g}"}

        def token_arm(name, mode, d, k):
            """A token arm on the confirm half, with its own control: a random shift of the same size."""
            r = confirmed(name, mode, token_spec(base[mode], x_voice, d, k), k)
            rnd = random_like([d], [0], f"{mood}/{name}")[0]
            r["random_lift"] = confirmed(f"R-{name}", mode, token_spec(base[mode], x_voice, rnd, k), k)["lift"]
            return r

        # Swap: her top positive clip as the reference (does the probe hear mood in generated speech?).
        swap_ref = reference([clips[p] for p, _ in (train or pairs)], ref_len)
        swap_rows = speak(mood, "S", {"ref": swap_ref, "mode": "icl"}, tests)
        swap_lift, swap_share = lifted(swap_rows, "icl", tests)

        # --- A: activation steering
        vec = load_vectors(files, keep, mood, tts, clips, train, fake)
        a_res, a_best = {}, None
        for layers in layer_sets:
            name = f"A{label(layers)}"
            k, tried = ladder_run(name, "icl", lambda k, L=layers: add_spec(base["icl"], vec, L, k, True, 1.0), above)
            a_res[name] = {"layers": layers, "strength": k, "tried": tried}
            if k is not None:
                lift = next(t["lift"] for t in tried if t["strength"] == k)
                if a_best is None or lift > a_res[a_best]["lift"]:
                    a_best = name
                a_res[name]["lift"] = lift
        arms = {}
        if a_best:
            L, k = a_res[a_best]["layers"], a_res[a_best]["strength"]
            arms["A"] = confirmed(a_best, "icl", add_spec(base["icl"], vec, L, k, True, 1.0), k)
            arms["A"]["setting"] = a_best
            arms["A-decay"] = confirmed(f"{a_best}-decay", "icl", add_spec(base["icl"], vec, L, k, True, decay), k)
            arms["A-xvec"] = confirmed(f"{a_best}-xvec", "xvec", add_spec(base["xvec"], vec, L, k, True, 1.0), k)
            rnd = random_like(vec, L, mood)
            arms["R"] = confirmed(f"R{label(L)}", "icl", add_spec(base["icl"], rnd, L, k, True, 1.0), k)
            kept[mood] = {"A": (L, k)}

        # --- B: her x-vector shift
        xp = [cached(keep / "xvec" / f"{p}.json", lambda c=clips[p]: tts.xvector(c)) for p, _ in train]
        xz = [cached(keep / "xvec" / f"{z}.json", lambda c=clips[z]: tts.xvector(c)) for _, z in train]
        tau = (np.mean(xp, 0) - np.mean(xz, 0)).tolist()
        tau_split = split_half(np.array(xp), np.array(xz))
        b = {}
        for mode in ("icl", "xvec"):
            k, tried = ladder_run(f"B-{mode}", mode, lambda k, m=mode: token_spec(base[m], x_voice, tau, k), above_tau)
            b[mode] = {"strength": k, "tried": tried}
            if k is not None:
                arms[f"B-{mode}"] = token_arm(f"B-{mode}", mode, tau, k)
        kept.setdefault(mood, {})["B"] = b["icl"]["strength"]

        # --- C: fitted speaker tokens
        fit_p = cached(keep / "fits" / f"{mood}.pos.json",
                       lambda: tts.fit(examples(tts, keep, clips, [p for p, _ in train]),
                                       examples(tts, keep, clips, [p for p, _ in held[: int(s.get("held_max", 16))]]), np.mean(xz, 0).tolist(), s))
        fit_z = cached(keep / "fits" / f"{mood}.zero.json",
                       lambda: tts.fit(examples(tts, keep, clips, [z for _, z in train]),
                                       examples(tts, keep, clips, [z for _, z in held[: int(s.get("held_max", 16))]]), np.mean(xz, 0).tolist(), s))
        delta = (np.array(fit_p["token"]) - np.array(fit_z["token"])).tolist()
        c = {}
        for mode in ("xvec", "icl"):
            k, tried = ladder_run(f"C-{mode}", mode, lambda k, m=mode: token_spec(base[m], x_voice, delta, k), above_tau)
            c[mode] = {"strength": k, "tried": tried}
            if k is not None:
                arms[f"C-{mode}"] = token_arm(f"C-{mode}", mode, delta, k)
        kept[mood]["C"] = c["xvec"]["strength"]

        # --- D: residual-matched speaker token (Brandon, 2026-10-21). The target is the shift her positive
        # clips share in the talker's residual stream while it reproduces them (each with its own x-vector),
        # against their matched zeros; the token is fitted so that on her zero clips it moves the residual
        # by that shift. delta_match = fitted token - the zeros' mean x-vector it starts from. One per target type.
        match_layers = parse_layers(s.get("match_layers", f"{tts.n_layers // 4}-{tts.n_layers - 2}"), tts.n_layers - 1)
        held_n = int(s.get("held_max", 16))
        fits_m, deltas_m, d = {}, {}, {}
        for target in targets:
            tag = target_tag(target)
            fit_m = cached(keep / "fits" / f"{mood}.match{'.' + tag if tag else ''}.json", lambda tg=target: tts.fit_match(
                examples(tts, keep, clips, [p for p, _ in train]), xp, examples(tts, keep, clips, [z for _, z in train]), xz,
                examples(tts, keep, clips, [z for _, z in held[:held_n]]),
                [cached(keep / "xvec" / f"{z}.json", lambda c=clips[z]: tts.xvector(c)) for _, z in held[:held_n]],
                np.mean(xz, 0).tolist(), match_layers, s, tg))
            fits_m[target] = fit_m
            deltas_m[target] = (np.array(fit_m["token"]) - np.mean(xz, 0)).tolist()
            for mode in ("xvec", "icl"):
                name = f"D{tag}-{mode}"
                k, tried = ladder_run(name, mode, lambda k, m=mode, dv=deltas_m[target]: token_spec(base[m], x_voice, dv, k),
                                      above_tau)
                d[name] = {"strength": k, "tried": tried}
                if k is not None:
                    arms[name] = token_arm(name, mode, deltas_m[target], k)
            kept[mood][f"D{tag}"] = d[f"D{tag}-xvec"]["strength"]
        delta_m = deltas_m[targets[0]]
        (keep / "tokens").mkdir(parents=True, exist_ok=True)
        (keep / "tokens" / f"{mood}.json").write_text(json.dumps(
            {"base": x_voice, "tau": tau, "delta": delta, **{f"delta_match{target_tag(g)}": v for g, v in deltas_m.items()}}))

        # --- held-out likelihood: does each setting favour her held-out positives over their zeros?
        scored_held = held[: int(s.get("held_max", 16))]
        hp = examples(tts, keep, clips, [p for p, _ in scored_held])
        hz = examples(tts, keep, clips, [z for _, z in scored_held])

        def nll(cond, spk, add=None):
            return cached(keep / "nll" / mood / f"{cond}.json", lambda: {
                "pos": [tts.nll(e, spk, add) for e in hp], "zero": [tts.nll(e, spk, add) for e in hz]})

        n0 = nll("N", x_voice)
        lik = {}
        settings_lik = {}
        if a_best:
            L, k = a_res[a_best]["layers"], a_res[a_best]["strength"]
            settings_lik["A"] = (x_voice, scaled(vec, L, k))
            settings_lik["R"] = (x_voice, scaled(random_like(vec, L, mood), L, k))
        if b["icl"]["strength"] is not None:
            settings_lik["B-icl"] = (shifted(x_voice, tau, b["icl"]["strength"]), None)
        if b["xvec"]["strength"] is not None:
            settings_lik["B-xvec"] = (shifted(x_voice, tau, b["xvec"]["strength"]), None)
        for mode in ("xvec", "icl"):
            if c[mode]["strength"] is not None:
                settings_lik[f"C-{mode}"] = (shifted(x_voice, delta, c[mode]["strength"]), None)
            for target, dv in deltas_m.items():
                name = f"D{target_tag(target)}-{mode}"
                if d[name]["strength"] is not None:
                    settings_lik[name] = (shifted(x_voice, dv, d[name]["strength"]), None)
        for arm, (spk, add) in settings_lik.items():
            lik[arm] = gain(nll(arm, spk, add), n0, mood)

        verdicts = {}
        r_lift = arms["R"]["lift"] if "R" in arms else None
        for arm, r in arms.items():
            if arm == "R":
                continue
            lk = lik.get(arm.replace("A-decay", "A").replace("A-xvec", "A"))
            control = r.get("random_lift", r_lift)  # token arms: a random token shift; A: the random vector
            steers = (r["holds"] and r["lift"] is not None and r["lift"] >= min_lift and r["share"] >= min_share
                      and (control is None or r["lift"] > control) and lk is not None and lk["low"] > 0)
            if swap_lift is None or swap_lift < min_lift:
                verdicts[arm] = "probe can't hear it"
            elif not r["holds"]:
                verdicts[arm] = "loses voice or words"
            else:
                verdicts[arm] = "steers" if steers else "doesn't steer"
        for name in arm_names(targets):
            if name not in arms and name not in ("A-decay", "A-xvec"):
                verdicts[name] = "loses voice or words"

        report["moods"][mood] = {
            "route": mset.get("route"), "borderline": mset.get("borderline", False),
            "pairs": {"train": len(train), "held_out": len(held)},
            "natural_similarity_drop": natural, "similarity_allowance": round(allow, 4),
            "swap": {"lift": swap_lift, "share": swap_share, "clip": swap_ref.id, "seconds": swap_ref.duration_s},
            "A_layers": {n: {k: v for k, v in r.items() if k != "layers"} for n, r in a_res.items()},
            "A_norm_fraction": norm_fraction(vec, a_res[a_best]["layers"], a_res[a_best]["strength"], norms) if a_best else None,
            "tau": {"norm_ratio": ratio(tau, x_voice), "split_half_cosine": tau_split},
            "delta": {"norm_ratio": ratio(delta, x_voice), "cosine_with_tau": round(cos(delta, tau), 4),
                      "fit_pos": {k: v for k, v in fit_p.items() if k != "token"},
                      "fit_zero": {k: v for k, v in fit_z.items() if k != "token"}},
            "match": {target: {"norm_ratio": ratio(dv, x_voice), "cosine_with_tau": round(cos(dv, tau), 4),
                               "cosine_with_delta": round(cos(dv, delta), 4),
                               "cosine_with_first_target": round(cos(dv, delta_m), 4),
                               "fit": {k: v for k, v in fits_m[target].items() if k != "token"}}
                      for target, dv in deltas_m.items()},
            "B_tried": b, "C_tried": c, "D_tried": d, "arms": arms, "likelihood_gain": lik, "verdicts": verdicts,
            "tokens": {"tau": tau, "delta": delta, **{f"delta_match{target_tag(g)}": v for g, v in deltas_m.items()}},
        }
        log(f"voicearms: {mood}: " + ", ".join(f"{a} {v}" for a, v in sorted(verdicts.items())))

    # --- blends: half of each mood's kept setting, added together
    for m1, m2 in blends:
        r1, r2 = report["moods"][m1], report["moods"][m2]
        name, out_b = f"{m1}+{m2}", {}
        if "A" in kept.get(m1, {}) and "A" in kept.get(m2, {}):
            L, k1 = kept[m1]["A"]
            k2 = kept[m2]["A"][1]
            v1, v2 = load_vectors(files, keep, m1, tts, clips, [], fake), load_vectors(files, keep, m2, tts, clips, [], fake)
            mix = [[0.5 * k1 * a + 0.5 * k2 * b for a, b in zip(x, y, strict=True)] for x, y in zip(v1, v2, strict=True)]
            out_b["A"] = blend_scores(speak(name, "A", add_spec(base["icl"], mix, L, 1.0, True, 1.0), tests), m1, m2, neutral_rows["icl"])
        token_arms = [("B", "tau", "icl"), ("C", "delta", "xvec")]
        token_arms += [(f"D{target_tag(g)}", f"delta_match{target_tag(g)}", "xvec") for g in targets]
        for arm, key, mode in token_arms:
            k1, k2 = kept.get(m1, {}).get(arm), kept.get(m2, {}).get(arm)
            if k1 is None or k2 is None:
                continue
            t1, t2 = np.array(r1["tokens"][key]), np.array(r2["tokens"][key])
            mixed = (0.5 * k1 * t1 + 0.5 * k2 * t2).tolist()
            out_b[arm] = blend_scores(speak(name, arm, token_spec(base[mode], x_voice, mixed, 1.0), tests), m1, m2, neutral_rows[mode])
        report["blends"][name] = out_b

    tts.close()
    shutil.copytree(keep / "audio", out / "audio", dirs_exist_ok=True)
    listening_set(report, out, clips, confirm[: int(s.get("listen_sentences", 1))])
    if (keep / "tokens").exists():
        shutil.copytree(keep / "tokens", out / "tokens", dirs_exist_ok=True)
    for r in report["moods"].values():
        r.pop("tokens")
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (out / "report.md").write_text(render(report))
    (out / "listen.md").write_text(listen(report))


# --- helpers ----------------------------------------------------------------------


def parse_layers(spec, n_layers):
    spec = str(spec)
    a, _, b = spec.partition("-")
    layers = list(range(int(a), int(b or a) + 1))
    if not layers or layers[-1] >= n_layers:
        raise SystemExit(f"voicearms: layer set {spec} is outside the talker's {n_layers} layers")
    return layers


def match_targets(s):
    """D's target types, in order; the first is the one the report compares the others with."""
    out = [str(x) for x in s.get("match_targets", ["mean", "pca:1", "pca:4"])]
    for x in out:
        kind, _, k = x.partition(":")
        if not (x == "mean" or (kind == "pca" and k.isdigit() and int(k) > 0)):
            raise SystemExit(f"voicearms: match target {x!r} is not mean or pca:<k>")
    return out


def target_tag(target):
    return "" if target == "mean" else target.replace(":", "")


def arm_names(targets):
    return ["A", "A-decay", "A-xvec", "B-icl", "B-xvec", "C-xvec", "C-icl"] + [
        f"D{target_tag(g)}-{m}" for g in targets for m in ("xvec", "icl")]


def shared_target(pairs, target):
    """pairs: (pairs, layers, hidden) residual shifts, positive minus its zero. "mean": their mean.
    "pca:k": per layer, the mean projected onto the top k principal directions of the shifts (uncentred,
    so the first is the dominant shared direction), keeping the mean's sign and size along them. Each
    shift is scaled to unit length for the directions, so every pair gets one vote and one loud clip
    can't take the first direction. Also returns, per layer, the share of those unit shifts' energy
    in the k directions."""
    import torch

    mean = pairs.mean(0)
    if target == "mean":
        return mean, None
    k = int(target.split(":")[1])
    unit = pairs.transpose(0, 1).float()  # layers x pairs x hidden
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    _, sv, vh = torch.linalg.svd(unit, full_matrices=False)
    basis = vh[:, :k]  # layers x k x hidden
    proj = torch.einsum("lkh,lh->lk", basis, mean.float())
    energy = sv[:, :k].pow(2).sum(-1) / sv.pow(2).sum(-1).clamp_min(1e-12)
    return torch.einsum("lkh,lk->lh", basis, proj).to(mean.dtype), energy


def label(layers):
    return f"L{layers[0]}-{layers[-1]}" if len(layers) > 1 else f"L{layers[0]}"


def reference(cands, length):
    """The first clip (candidates come best first) whose length is in [lo, hi] seconds, else the longest:
    a clip of a second or two gives the speaker encoder and the in-context prompt too little of her."""
    lo, hi = length
    for c in cands:
        if lo <= c.duration_s <= hi:
            return c
    return max(cands, key=lambda c: c.duration_s)


def held_out(clip):
    """A quarter of the conversations, by hash, are held out for the likelihood check and the fits."""
    conv = clip.line.get("conversation") or clip.id
    return unit_hash("held", str(conv)) < 0.25


def add_spec(spec, vec, layers, k, renorm, decay):
    return {**spec, "add": scaled(vec, layers, k), "renorm": renorm, "decay": decay}


def scaled(vec, layers, k):
    return {i: [k * x for x in vec[i]] for i in layers}


def token_spec(spec, x, d, k):
    return {**spec, "xvec": shifted(x, d, k)}


def shifted(x, d, k):
    return [a + k * b for a, b in zip(x, d, strict=True)]


def random_like(vec, layers, mood):
    """Per layer, a fixed random direction with the vector's norm (the "any push this size" control)."""
    import numpy as np

    rng = np.random.default_rng(int(unit_hash("random", mood) * 2**32))
    out = [list(v) for v in vec]
    for i in layers:
        r = rng.normal(size=len(vec[i]))
        out[i] = (r / np.linalg.norm(r) * np.linalg.norm(vec[i])).tolist()
    return out


def load_vectors(files, keep, mood, tts, clips, train, fake):
    """voicesteer's vector for the mood (all talker layers), or a capture the same way if absent."""
    import numpy as np

    v = None
    bundle = files / "voice_vectors.npz"  # voicelayers.py --npz, from voicesteer's vectors/
    if bundle.exists():
        with np.load(bundle) as z:
            if mood in z.files:
                v = z[mood].tolist()
    for path in (files / "vectors" / f"{mood}.npy", keep / "vectors" / f"{mood}.npy"):
        if v is None and path.exists():
            v = np.load(path).tolist()
    if v is not None:
        if len(v) != tts.n_layers:
            raise SystemExit(f"voicearms: {mood}'s vector has {len(v)} layers, the talker has {tts.n_layers}")
        return v
    if not train:
        raise SystemExit(f"voicearms: no vector for {mood}")
    log(f"voicearms: no vector for {mood}, capturing ({len(train)} pairs)")
    pos = [(str(clips[p].audio), clips[p].text) for p, _ in train]
    neg = [(str(clips[z].audio), clips[z].text) for _, z in train]
    mp, mn = voicesteer.mean_state(tts, pos, 2), voicesteer.mean_state(tts, neg, 2)
    v = [[x - y for x, y in zip(a, b, strict=True)] for a, b in zip(mp, mn, strict=True)]
    (keep / "vectors").mkdir(parents=True, exist_ok=True)
    np.save(keep / "vectors" / f"{mood}.npy", np.array(v, dtype=np.float32))
    return v


def examples(tts, keep, clips, ids):
    """(text, codec tokens) per clip, the codes encoded once and kept."""
    out = []
    for i in ids:
        path = keep / "codes" / f"{i}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(tts.codes(clips[i])))
        out.append({"id": i, "text": clips[i].text, "codes": json.loads(path.read_text())})
    return out


def gain(nll_cond, nll_base, mood):
    """Per held-out pair: (zero - pos) under the setting minus the same under N. Positive means the
    setting makes her mood clips likelier relative to their matched zeros. Mean and a 95% bootstrap."""
    import numpy as np

    d = (np.array(nll_cond["zero"]) - np.array(nll_cond["pos"])) - (np.array(nll_base["zero"]) - np.array(nll_base["pos"]))
    if not len(d):
        return None
    rng = np.random.default_rng(int(unit_hash("boot", mood) * 2**32))
    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return {"mean": round(float(d.mean()), 4), "low": round(float(np.percentile(boots, 2.5)), 4),
            "high": round(float(np.percentile(boots, 97.5)), 4), "pairs": int(len(d))}


def split_half(xp, xz):
    """Cosine between tau built from two random halves of the pairs (as check B)."""
    import numpy as np

    if len(xp) < 4:
        return None
    rng = np.random.default_rng(0)
    cs = []
    for _ in range(20):
        idx = rng.permutation(len(xp))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        cs.append(cos(xp[a].mean(0) - xz[a].mean(0), xp[b].mean(0) - xz[b].mean(0)))
    return round(float(np.mean(cs)), 4)


def cos(a, b):
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def ratio(d, x):
    import numpy as np

    return round(float(np.linalg.norm(d) / np.linalg.norm(x)), 4)


def norm_fraction(vec, layers, k, norms):
    """|k v_L| / mean |h_L| per steered layer: the push as a share of the residual."""
    import numpy as np

    return {str(i): round(float(k * np.linalg.norm(vec[i]) / norms[str(i)]), 4) for i in layers if norms.get(str(i))}


def voicesteer_cos(scorer, clip, her):
    return round(steer.cosine(scorer.speaker(clip), her), 4)


def blend_scores(rows, m1, m2, base_rows):
    return {m: {"lift": mean([r["probe"][m] - b["probe"][m] for r, b in zip(rows, base_rows, strict=True)])} for m in (m1, m2)} | {
        "similarity": mean([r["similarity"] for r in rows]), "wer": mean([r["wer"] for r in rows])}


# --- the TTS model ------------------------------------------------------------------


class ArmTTS(steer.TTS):
    """Qwen3-TTS Base with hooks on every talker layer (capture and steer as steer.TTS), plus: a
    norm-preserving add, a per-frame decay, residual-norm measurement, steering of teacher-forced
    passes (for the likelihood check), the speaker token as an input, and the frozen-model fit."""

    def __init__(self, model_id, device):
        super().__init__(model_id, None, device)
        import torch

        self.m = self.model.model
        self.talker = self.m.talker
        self.n_layers = len(self.talker.model.layers)
        self.dev = next(self.m.parameters()).device
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.renorm, self.decay, self.step, self.t = False, 1.0, 0, 0
        self.measure, self.norm_sum, self.norm_n = False, {}, {}
        self.seq_from = None
        self.F = torch.nn.functional

    def _hook(self, i):
        torch = self.torch

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            single = h.shape[1] == 1
            if single and i == 0:
                self.t, self.step = self.step, self.step + 1
            if single and self.mode == "capture":
                self.sums[i] = self.sums.get(i, 0) + h[:, -1].float().sum(0)
                self.counts[i] = self.counts.get(i, 0) + h.shape[0]
            if single and self.measure:
                self.norm_sum[i] = self.norm_sum.get(i, 0.0) + float(h[:, -1].float().norm(dim=-1).sum())
                self.norm_n[i] = self.norm_n.get(i, 0) + h.shape[0]
            if self.mode != "steer" or i not in self.add or not (single or self.seq_from is not None):
                return None
            x = h.float()
            if single:
                w = torch.tensor([[self.decay ** self.t]], device=x.device)
                lo = x.shape[1] - 1
            else:
                lo = self.seq_from
                w = (self.decay ** torch.arange(x.shape[1] - lo, device=x.device, dtype=torch.float32))[None]
            part = x[:, lo:]
            new = part + w[..., None] * self.add[i]
            if self.renorm:
                new = new * (part.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            h = torch.cat([x[:, :lo], new], dim=1).to(h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h

        return hook

    def _steering(self, add, renorm=True, decay=1.0):
        if add:
            self.add = {i: self.torch.tensor(v, device=self.dev, dtype=self.torch.float32) for i, v in add.items()}
            self.mode, self.renorm, self.decay, self.step = "steer", renorm, decay, 0

    def _off(self):
        self.mode, self.add, self.seq_from = None, {}, None

    def speak(self, spec, sentence, s, path):
        import soundfile as sf

        torch = self.torch
        xvec_only = spec["mode"] == "xvec"
        ref = spec["ref"]
        items = self.model.create_voice_clone_prompt(ref_audio=str(ref.audio), ref_text=None if xvec_only else ref.text,
                                                     x_vector_only_mode=xvec_only)
        if spec.get("xvec") is not None:
            e = items[0].ref_spk_embedding
            items[0].ref_spk_embedding = torch.tensor(spec["xvec"], dtype=e.dtype, device=e.device)
        self._steering(spec.get("add"), spec.get("renorm", True), spec.get("decay", 1.0))
        torch.manual_seed(s)
        cap = int(12 * (0.8 * len(sentence.split()) + 4))
        try:
            wavs, sr = self.model.generate_voice_clone(text=sentence, language="English", voice_clone_prompt=items,
                                                       max_new_tokens=cap)
        finally:
            self._off()
        sf.write(str(path), wavs[0], sr)

    def capture(self, audio, text, sentence, s):
        self.step = 0
        return super().capture(audio, text, sentence, s)

    def residual_norms(self):
        return {str(i): self.norm_sum[i] / self.norm_n[i] for i in sorted(self.norm_sum)}

    def _wav24(self, clip):
        import librosa
        import numpy as np

        wav, _ = librosa.load(str(clip.audio), sr=24_000, mono=True)
        return wav.astype(np.float32)

    def xvector(self, clip):
        with self.torch.no_grad():
            return [float(x) for x in self.m.extract_speaker_embedding(self._wav24(clip), 24_000).float().cpu()]

    def codes(self, clip):
        with self.torch.no_grad():
            c = self.m.speech_tokenizer.encode(self._wav24(clip), sr=24_000).audio_codes[0]
        return c.long().cpu().tolist()

    def _layout(self, text, codes, spk):
        """The talker's input for a teacher-forced pass, laid out exactly as generate() lays out
        x-vector-only voice cloning in English (streaming text), with `spk` as the speaker token.
        Returns (embeds, prefill length, group-0 targets)."""
        torch, t, c = self.torch, self.talker, self.m.config
        tc = c.talker_config
        ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])[0].to(self.dev)
        emb = t.get_input_embeddings()

        def txt(x):
            return t.text_projection(t.get_text_embeddings()(x))

        tts_bos, tts_eos, tts_pad = txt(torch.tensor([[c.tts_bos_token_id, c.tts_eos_token_id, c.tts_pad_token_id]],
                                                     device=self.dev)).chunk(3, dim=1)
        lang = tc.codec_language_id["english"]
        c0 = emb(torch.tensor([[tc.codec_think_id, tc.codec_think_bos_id, lang, tc.codec_think_eos_id]], device=self.dev))
        c1 = emb(torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=self.dev))
        codec = torch.cat([c0, spk.view(1, 1, -1).to(c0.dtype), c1], dim=1)
        pre = torch.cat([txt(ids[:, :3]),
                         torch.cat([tts_pad.expand(-1, codec.shape[1] - 2, -1), tts_bos], dim=1) + codec[:, :-1],
                         txt(ids[:, 3:4]) + codec[:, -1:]], dim=1)
        trailing = torch.cat([txt(ids[:, 4:-5]), tts_eos], dim=1)[0]
        cd = torch.tensor(codes, device=self.dev, dtype=torch.long)
        frames = emb(cd[:, 0])
        for g in range(1, cd.shape[1]):
            frames = frames + t.code_predictor.get_input_embeddings()[g - 1](cd[:, g])
        n = cd.shape[0]
        if n > trailing.shape[0]:
            trailing = torch.cat([trailing, tts_pad[0].expand(n - trailing.shape[0], -1)], dim=0)
        seq = torch.cat([pre, (frames + trailing[:n])[None]], dim=1)
        targets = torch.cat([cd[:, 0], torch.tensor([tc.codec_eos_token_id], device=self.dev)])
        return seq, pre.shape[1], targets, cd

    def _loss(self, ex, spk):
        """Per-frame NLL: talker CE on group 0 (and the stop) + 0.3 x code predictor CE on groups 1-15."""
        torch, t = self.torch, self.talker
        seq, p, targets, cd = self._layout(ex["text"], ex["codes"], spk)
        out = t(inputs_embeds=seq, attention_mask=torch.ones(seq.shape[:2], device=self.dev, dtype=torch.long),
                output_hidden_states=True, use_cache=False)
        n = cd.shape[0]
        logits = out.logits[0, p - 1: p + n].float()
        loss0 = self.F.cross_entropy(logits, targets)
        hid = out.hidden_states[0][-1][0, p - 1: p - 1 + n]
        _, loss_sub = t.forward_sub_talker_finetune(cd, hid)
        return loss0 + 0.3 * loss_sub.float()

    def nll(self, ex, spk, add=None):
        torch = self.torch
        with torch.no_grad():
            seq_len = None
            if add:
                self._steering(add)
                seq_len = self._layout(ex["text"], ex["codes"], torch.tensor(spk, device=self.dev))[1]
                self.seq_from = seq_len  # steer the frames' positions, as generation does
            try:
                return round(float(self._loss(ex, torch.tensor(spk, device=self.dev))), 5)
            finally:
                self._off()

    def fit(self, train, held, init, s):
        """One speaker token fitted to `train` with every weight frozen; early-stopped on `held`."""
        import random

        torch = self.torch
        steps, lr, l2 = int(s.get("fit_steps", 200)), float(s.get("fit_lr", 2e-3)), float(s.get("fit_l2", 1e-2))
        every, batch = int(s.get("fit_eval_every", 25)), int(s.get("fit_batch", 8))
        x0 = torch.tensor(init, device=self.dev, dtype=torch.float32)
        tok = x0.clone().requires_grad_(True)
        opt = torch.optim.AdamW([tok], lr=lr, weight_decay=0.0)
        rng = random.Random(0)

        def held_loss():
            with torch.no_grad():
                return float(sum(self._loss(e, tok) for e in held) / len(held)) if held else None

        best, best_loss, curve = tok.detach().clone(), held_loss(), []
        curve.append({"step": 0, "held": best_loss})
        for step in range(1, steps + 1):
            opt.zero_grad()
            loss = sum(self._loss(e, tok) for e in rng.sample(train, min(batch, len(train)))) / min(batch, len(train))
            loss = loss + l2 * ((tok - x0).pow(2).sum() / x0.pow(2).sum())
            loss.backward()
            opt.step()
            if step % every == 0 or step == steps:
                h = held_loss()
                curve.append({"step": step, "train": round(loss.item(), 5), "held": h})
                if best_loss is None or (h is not None and h < best_loss):
                    best, best_loss = tok.detach().clone(), h
        return {"token": [float(x) for x in best.cpu()], "held_loss": best_loss, "start_held_loss": curve[0]["held"],
                "curve": curve, "train_clips": len(train), "held_clips": len(held)}


    def _states(self, ex, spk, layers):
        """Each layer's output, averaged over the frame positions (where generation steers), teacher-forced."""
        torch = self.torch
        seq, p, _, cd = self._layout(ex["text"], ex["codes"], spk)
        out = self.talker(inputs_embeds=seq, attention_mask=torch.ones(seq.shape[:2], device=self.dev, dtype=torch.long),
                          output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[0]  # hs[i + 1] is layer i's output (the last one after the final norm)
        n = cd.shape[0]
        return torch.stack([hs[i + 1][0, p: p + n].float().mean(0) for i in layers])

    def fit_match(self, pos, pos_x, zero, zero_x, held, held_x, init, layers, s, target="mean"):
        """One speaker token fitted, with every weight frozen, so that on her zero clips it shifts each
        layer's residual by the shift her positive clips share (mean over positives, each with its own
        x-vector, minus the same over their zeros), each layer weighted by that shift's size."""
        import random

        torch = self.torch
        dev = self.dev
        steps, lr, l2 = int(s.get("fit_steps", 200)), float(s.get("fit_lr", 2e-3)), float(s.get("fit_l2", 1e-2))
        every, batch = int(s.get("fit_eval_every", 25)), int(s.get("fit_batch", 8))
        with torch.no_grad():
            sp = [self._states(e, torch.tensor(x, device=dev), layers) for e, x in zip(pos, pos_x, strict=True)]
            sz = [self._states(e, torch.tensor(x, device=dev), layers) for e, x in zip(zero, zero_x, strict=True)]
            pairs = torch.stack(sp) - torch.stack(sz)  # pairs x layers x hidden
            target, energy = shared_target(pairs, target)
            scale = target.norm(dim=-1).pow(2).clamp_min(1e-12)
            kept_of_mean = target.norm(dim=-1) / pairs.mean(0).norm(dim=-1).clamp_min(1e-12)
            agreement = self.F.cosine_similarity(pairs, target[None], dim=-1).mean(0)  # per layer
            x0 = torch.tensor(init, device=dev, dtype=torch.float32)
            base_train = [self._states(e, x0, layers) for e in zero]
            base_held = [self._states(e, x0, layers) for e in held]

        def loss_of(tok, ex, base):
            got = self._states(ex, tok, layers) - base
            return ((got - target).pow(2).sum(-1) / scale).mean()

        tok = x0.clone().requires_grad_(True)
        opt = torch.optim.AdamW([tok], lr=lr, weight_decay=0.0)
        rng = random.Random(0)
        carriers = list(zip(zero, base_train))

        def held_loss():
            with torch.no_grad():
                return float(sum(loss_of(tok, e, b) for e, b in zip(held, base_held)) / len(held)) if held else None

        best, best_loss, curve = tok.detach().clone(), held_loss(), []
        curve.append({"step": 0, "held": best_loss})
        for step in range(1, steps + 1):
            opt.zero_grad()
            pick = rng.sample(carriers, min(batch, len(carriers)))
            loss = sum(loss_of(tok, e, b) for e, b in pick) / len(pick)
            loss = loss + l2 * ((tok - x0).pow(2).sum() / x0.pow(2).sum())
            loss.backward()
            opt.step()
            if step % every == 0 or step == steps:
                h = held_loss()
                curve.append({"step": step, "train": round(loss.item(), 5), "held": h})
                if best_loss is None or (h is not None and h < best_loss):
                    best, best_loss = tok.detach().clone(), h
        return {"token": [float(x) for x in best.cpu()], "held_loss": best_loss, "start_held_loss": curve[0]["held"],
                "curve": curve, "layers": layers, "agreement": {str(i): round(float(a), 4) for i, a in zip(layers, agreement)},
                "target_norm": {str(i): round(float(v), 4) for i, v in zip(layers, scale.sqrt())},
                "mean_kept": {str(i): round(float(v), 4) for i, v in zip(layers, kept_of_mean)},
                "pc_energy": None if energy is None else {str(i): round(float(v), 4) for i, v in zip(layers, energy)},
                "train_clips": len(zero), "held_clips": len(held)}


class FakeArmTTS:
    """Deterministic stand-in for CI: 6 layers, 4 wide; mood settings lift the fake probe by strength."""

    n_layers = 6
    HIDDEN = 4

    def __init__(self):
        self.measure = False

    def close(self):
        pass

    def speak(self, spec, sentence, s, path):
        write_wav(path, steer.fake_wave(f"{spec['ref'].id}{sentence}{json.dumps(spec.get('xvec'))}{sorted(spec.get('add') or {})}",
                                        seconds=0.05))

    def capture(self, audio, text, sentence, s):
        key = Path(audio).name + sentence
        return [[unit_hash("h", key, str(i), str(j)) for j in range(self.HIDDEN)] for i in range(self.n_layers)]

    def residual_norms(self):
        return {str(i): 2.0 for i in range(self.n_layers)}

    def xvector(self, clip):
        return [1.0 + unit_hash("x", clip.id, str(j)) for j in range(self.HIDDEN)]

    def codes(self, clip):
        return [[int(unit_hash("c", clip.id, str(t), str(g)) * 100) for g in range(16)] for t in range(3)]

    def nll(self, ex, spk, add=None):
        pull = sum(abs(x) for v in (add or {}).values() for x in v) + sum(spk)
        return round(2.0 + 0.1 * unit_hash("n", ex["id"]) - (0.01 * pull if "pos" in ex["id"] else 0.0), 5)

    def fit_match(self, pos, pos_x, zero, zero_x, held, held_x, init, layers, s, target="mean"):
        bump = 0.3 if target == "mean" else 0.2 + 0.01 * int(target.split(":")[1])
        return {"token": [x + bump for x in init], "held_loss": 0.4, "start_held_loss": 1.0, "curve": [], "layers": layers,
                "agreement": {str(i): 0.5 for i in layers}, "target_norm": {str(i): 1.0 for i in layers},
                "train_clips": len(zero), "held_clips": len(held)}

    def fit(self, train, held, init, s):
        bump = 0.5 if train and "pos" in train[0]["id"] else 0.0
        return {"token": [x + bump for x in init], "held_loss": 1.9, "start_held_loss": 2.0, "curve": [],
                "train_clips": len(train), "held_clips": len(held)}


class FakeScorer:
    """Fake scores: a condition named <arm>@<k> lifts its mood's probe by k (the random control by
    nothing); swap lifts it; voice similarity falls above strength 0.6."""

    def her(self, clips):
        return [1.0]

    def speaker(self, clip):
        return [1.0 if "pos" not in clip.id else 0.97, 0.1]

    def __call__(self, path, sentence, mood, probe, her):
        cond = Path(path).parent.name
        k = float(cond.split("@")[1]) if "@" in cond else (1.0 if cond in ("S", "A", "B", "C") else 0.0)
        if cond.startswith("R"):
            k = 0.0
        sim_k = float(cond.split("@")[1]) if "@" in cond else 0.0
        return {"probe": {m: round(0.3 + 0.3 * k * (m == mood or "+" in Path(path).parent.parent.name)
                                   + 0.02 * unit_hash("p", str(path), m), 4) for m in probe.moods},
                "similarity": round(0.8 - 0.15 * max(0.0, sim_k - 0.6), 4), "wer": round(0.02 * sim_k, 4)}


# --- report -----------------------------------------------------------------------


def render(r):
    ARMS = arm_names(r["settings"].get("match_targets", ["mean"]))
    lines = ["# Steering Qwen3-TTS: three arms (voicearms)", "",
             f"Model {r['model']}, {r['layers']} talker layers, speaking from {r['voice']}. Strength picked on sentences "
             f"{r['settings']['pick']}, verdicts on {r['settings']['confirm']}.", "",
             "Verdict per mood and arm (strength kept; probe lift on the confirm half; likelihood gain [95%]):", "",
             "| mood | swap lift | " + " | ".join(ARMS) + " | random lift |", "|---" * (len(ARMS) + 3) + "|"]
    for m, x in sorted(r["moods"].items()):
        cells = []
        for a in ARMS:
            v = x["verdicts"].get(a)
            arm = x["arms"].get(a)
            lk = x["likelihood_gain"].get(a.replace("A-decay", "A").replace("A-xvec", "A"))
            if arm is None:
                cells.append(v or "-")
                continue
            lk_s = f" [{lk['low']:+.3f}, {lk['high']:+.3f}]" if lk else ""
            cells.append(f"{v} x{arm['strength']:g} {fmt(arm['lift'])}{lk_s}")
        rl = x["arms"].get("R", {}).get("lift")
        lines.append(f"| {m}{' (borderline)' if x['borderline'] else ''} | {fmt(x['swap']['lift'])} | " + " | ".join(cells)
                     + f" | {fmt(rl)} |")
    lines += ["", "Per mood: A's layer settings (kept strength, pick-half lift), the push as a share of the residual, "
              "and her own x-vector and fitted-token shifts:", ""]
    for m, x in sorted(r["moods"].items()):
        a = ", ".join(f"{n} x{v['strength']}{' ' + fmt(v.get('lift')) if v.get('lift') is not None else ''}"
                      for n, v in x["A_layers"].items())
        lines.append(f"- **{m}**: A {a}; residual share {x['A_norm_fraction']}; natural similarity drop "
                     f"{x['natural_similarity_drop']} (allowance {x['similarity_allowance']}); tau/x {x['tau']['norm_ratio']}, "
                     f"split-half {x['tau']['split_half_cosine']}; delta/x {x['delta']['norm_ratio']}, cos(delta, tau) "
                     f"{x['delta']['cosine_with_tau']}; fit held-out loss {x['delta']['fit_pos']['start_held_loss']} -> "
                     f"{x['delta']['fit_pos']['held_loss']}")
        for g, mt in x["match"].items():
            f = mt["fit"]
            lines.append(f"  - matched token, target {g}: shift shared by her clips (mean cosine of each pair's shift to "
                         f"the target) {f.get('agreement')}, share of the mean kept {f.get('mean_kept')}, energy in those "
                         f"directions {f.get('pc_energy')}, match loss {f.get('start_held_loss')} -> {f.get('held_loss')}, "
                         f"cos with delta {mt['cosine_with_delta']}, with the first target's token {mt['cosine_with_first_target']}")
    if r["blends"]:
        lines += ["", "Blends (half of each mood's kept setting; probe lift for each mood over neutral):", ""]
        for name, arms in r["blends"].items():
            for arm, v in arms.items():
                lifts = ", ".join(f"{m} {fmt(s['lift'])}" for m, s in v.items() if isinstance(s, dict))
                lines.append(f"- {name} {arm}: {lifts}; similarity {v['similarity']}, WER {v['wer']}")
    return "\n".join(lines) + "\n"


def listening_set(r, out, clips, sentences):
    """listen/<group>/<nn>.wav, blind and shuffled, plus her real example clip, sheet.csv and the hidden key."""
    import csv

    root = out / "listen"
    shutil.rmtree(root, ignore_errors=True)
    key, rows = {}, []
    groups = [(m, [("neutral", "_neutral", "N-icl"), ("swap", m, "S")] +
               [(a, m, v["audio"]) for a, v in x["arms"].items() if "audio" in v])
              for m, x in sorted(r["moods"].items())]
    groups += [(b, [("neutral", "_neutral", "N-icl")] + [(a, b, a) for a in arms])
               for b, arms in sorted(r["blends"].items())]
    for group, conds in groups:
        (root / group).mkdir(parents=True)
        items = [(cond, out / "audio" / d / a, n) for cond, d, a in conds for n in sentences]
        items.sort(key=lambda i: unit_hash("listen", group, i[0], str(i[2])))
        for num, (cond, src, n) in enumerate(items, 1):
            wav = src / f"{n}.wav"
            if not wav.exists():
                continue
            name = f"{num:02d}"
            shutil.copy(wav, root / group / f"{name}.wav")
            key[f"{group}/{name}"] = {"condition": cond, "sentence": n}
            rows.append([group, name, TEST[n], "", "", "", ""])
        if group in r["moods"]:
            clip = clips.get(r["moods"][group]["swap"].get("clip"))
            if clip is not None:
                shutil.copy(clip.audio, root / group / f"her_example{Path(clip.audio).suffix}")
    with open(root / "sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "clip", "sentence", "still_her_voice_y_n", "mood_heard_0_3", "words_clear_y_n", "notes"])
        w.writerows(rows)
    (root / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    (root / "README.md").write_text(
        "# Blind listening set\n\nEach folder is a mood (or a blend, a+b). The numbered clips are the same sentence\n"
        "spoken different ways (neutral, her swapped-in mood clip, and each steering method), in a shuffled order.\n"
        "her_example is one of her real game clips in that mood. In sheet.csv, per clip: is it still her voice (y/n),\n"
        "how much of the folder's mood you hear (0 none - 3 clearly; for a blend, name which moods), are the\n"
        "words clear (y/n), and any notes. Don't open key.json until you're done.\n")


def unblind(folder):
    """His filled-in sheet.csv joined to key.json: listen/results.md."""
    import csv

    root = Path(folder) / "listen"
    key = json.loads((root / "key.json").read_text())
    by = {}
    with open(root / "sheet.csv", newline="") as f:
        for row in csv.DictReader(f):
            k = key.get(f"{row['group']}/{row['clip']}")
            if k:
                by.setdefault(row["group"], []).append((k["condition"], row))
    lines = ["# What Brandon heard", "", "| group | condition | her voice | mood heard | words clear | notes |",
             "|---|---|---|---|---|---|"]
    for group, items in sorted(by.items()):
        for cond, row in sorted(items, key=lambda i: i[0]):
            lines.append(f"| {group} | {cond} | {row['still_her_voice_y_n']} | {row['mood_heard_0_3']} | "
                         f"{row['words_clear_y_n']} | {row['notes']} |")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'results.md'}")


def listen(r):
    lines = ["# Listening", "", "audio/_neutral/N-icl/<n>.wav is her neutral voice (N-xvec: x-vector-only mode).",
             "audio/<mood>/S/<n>.wav swaps in her top clip for that mood. audio/<mood>/<arm>@<k>/<n>.wav is an arm at",
             "strength k: A<layers> activation steering (-decay, -xvec variants), R the random control, B-<mode> her",
             "x-vector shift, C-<mode> her fitted token, D-<mode> her residual-matched token (Dpcak-<mode> with the",
             "PCA-cleaned target), R-<arm> the token arms'",
             "random-shift controls.",
             "audio/<a>+<b>/<arm>/ are blends. Sentence n:", ""]
    lines += [f"{n}. {t}" for n, t in enumerate(TEST)]
    lines += ["", "Kept settings per mood:", ""]
    for m, x in sorted(r["moods"].items()):
        lines.append(f"- {m}: " + ", ".join(f"{a} x{v['strength']:g}" for a, v in x["arms"].items()))
    return "\n".join(lines) + "\n"


def fmt(x):
    return "-" if x is None else f"{x:+.3f}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--unblind":
        unblind(sys.argv[2])
    else:
        main()
