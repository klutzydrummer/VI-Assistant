# Methodology review after the smoke tests

> **Superseded (2026-10-14).** The corpus pipeline is now [corpus-pipeline.md](corpus-pipeline.md): classes are discovered by clustering Tali's own clip embeddings, not scored on fixed axes. This document is kept for history only. Don't build from it.

Status: review and proposals, 2026-10-13. Nothing here is decided until Brandon approves it; the approved parts move into [control-vector-datasets.md](control-vector-datasets.md).

Inputs:
- The 40-clip smoke runs on real models: discover, label, then clean.
- The embedding checks.
- The A40 benchmark.
- [prior-work.md](prior-work.md).
- The practices cited below.

The question: what did the smoke tests say about our choices, and what do standard practices in annotation, evaluation and speech emotion recognition say about the method as a whole?

## 1. What the smoke tests actually showed

**Every off-the-shelf model broke one of our assumptions on Tali's audio:**

| model | assumption | what happened |
|---|---|---|
| Gemma 4 (audio) | it listens to the clip | refused 14 of 40 clips when the audio came after the text; refuses most clips under 1.6 s; tagged all 29 it captioned `other_voice` |
| emotion2vec (base) | its embedding encodes emotion | it mostly encodes clip length (log-duration R² 0.81) |
| ECAPA speaker | it measures "is this her" | it also identifies the game, 72% vs 45% chance; recording chain is a confound |
| DNSMOS | it measures audio quality | median 2.5/5 on studio audio, likely reading the helmet filter as damage |
| text judge | it scores feelings | it put exactly 0 on most axes until the prompt stopped pushing it toward neutral |

None of this showed up in the fake-model tests. Those check plumbing, not validity. Each problem surfaced only because someone looked at real outputs and asked whether they made sense.

**Lesson:** a model is an untested measurement instrument until it passes a check on our own data. Domain shift is the norm here: a helmet filter, acted game dialogue, short barks, three recording eras.

**The signals agree weakly, and we can't yet tell who is right.** In the first smoke run, agreement between signals ran from −0.32 to 0.55 Spearman per axis. That's expected. Even humans reach only κ≈0.48 on IEMOCAP's categorical labels ([summary](https://arxiv.org/abs/2210.07642)). But it means every number from the pipeline is unvalidated until there's human ground truth.

**Positives are rare.** Fused scores of 0.6 or more counted per axis on the 40 clips:
- joy: 4
- anger: 1
- sadness: 1
- worry: 1
- playfulness: 0
- intoxication: 0

The corpus is mostly neutral dialogue, while contrast sets need *positives*.

## 2. What best practice says about the current method

### 2a. The evaluation sample is too small and the wrong shape

The plan reviews 150 random clips first and measures accuracy on them. A random sample is unbiased, which was the point. But with positives at roughly 0–10% per axis, 150 random clips hold only a handful of positives per axis, sometimes none. You can't estimate how well positives are found from 3 of them.

Even for overall correlation, n=150 is loose. Using Fisher's z with Spearman's standard error √(1.06/(n−3)) (Fieller, Hartley & Pearson 1957, Biometrika 44), a measured ρ of 0.5 has a 95% interval of:

| n | 95% interval for ρ = 0.5 |
|---|---|
| 50 | 0.25–0.69 |
| 150 | 0.37–0.61 |
| 300 | 0.41–0.58 |
| 600 | 0.44–0.56 |

**Best practice:** when labels are expensive and the interesting class is rare, sample in strata and weight back. Oversample clips the models rank high on an axis, sample the rest thinly, and weight each reviewed clip by 1 / its chance of being picked. The estimate stays unbiased and its variance drops a lot. This is the core of active testing ([Kossen et al. 2021](https://arxiv.org/abs/2103.05331)) and of classic stratified estimation.

### 2b. We measure the wrong thing

The pipeline scores every clip 0–1 on six axes, and calibration measures Spearman and MAE over all clips. But the product is contrast sets: for each axis, a set of clear positives and a set of clean negatives. What decides vector quality is:
- **Positive precision:** of the clips we'd pick as "sad", how many really are?
- **Negative purity:** how many "neutral" picks secretly carry the axis?

A signal can rank the 95% neutral bulk well and still pick bad positives. **Best practice is to evaluate the decision you'll make, not a proxy.**

### 2c. Rating six sliders per clip is slow and unreliable

Continuous ratings on many scales at once drift: within one rater over time, and against the anchors. [Kiritchenko & Mohammad (2017)](https://aclanthology.org/P17-2074/) showed that **best-worst scaling** (pick the most and least X out of 4 clips) gives significantly more reliable intensity scores than rating scales for the same number of judgements, and they used it for emotion intensity.

Simpler still, for contrast sets the question that matters is binary per axis: "is this a clear example of sad?" A binary call takes seconds; six sliders take many times longer.

### 2d. One reviewer means an unknown noise ceiling

With a single annotator there's no inter-rater agreement to report. The standard fallback is **intra-rater (test-retest) reliability**: re-show a small fraction of clips later, blind, and measure how consistent the answers are ([affect annotation reliability](https://arxiv.org/abs/2308.16029)). That number is the ceiling: no model can be shown to agree with Brandon better than Brandon agrees with himself.

### 2e. The calibrate stage's cross-validation leaks

`calibrate.py` splits reviewed clips into folds at random. Clips from the same conversation share context, scene, recording session and often mood. Lines repeated across games can be near-duplicates. Random folds put those in both training and test, which inflates the probe's measured accuracy.

In speech emotion recognition, splitting by speaker or session is standard because random splits are known to inflate results ([splitting criteria study](https://arxiv.org/abs/2210.14501); [IEMOCAP reality check](https://arxiv.org/abs/2304.00860)). We have one speaker, so the analogous grouping is **conversation**, and folds should be grouped by it.

### 2f. Evidence is counted twice

- **Stage directions:** since the smoke-test fixes, the text judge reads the LE3 stage direction, and the direction priors score the same direction again. Fusion treats them as independent evidence.
- **Gemma:** the text judge and the audio verification are the same model, so their errors are correlated.

Fusion's confidence formula assumes independent signals, so it overstates confidence exactly where these overlap.

### 2g. Confounds aren't checked as a matter of course

We found the length confound in emotion2vec base and the game confound in ECAPA only because we went looking. Contrast sets built on a confounded signal would teach the vector "long line" or "ME3 recording" instead of "sad". The plan matches negatives on length and pace, which is right, but nothing measures confounds systematically.

### 2h. Some axes may not have enough data

Intoxication had zero positives in 40 clips. Tali's drunk lines are probably a few dozen in the Citadel DLC party at most. For the LLM side that's fine, since minimal pairs are generated text. For TTS vectors you need real audio positives, maybe 30–100 per axis. An axis can fail for lack of data no matter how good the labels are. Better to know that after the full label run than after extraction.

## 3. How the stack looks with fresh eyes

**Solid; keep:**
- **The Rust stage runner:** content-addressed caching, per-run keys, checkpoints that survive a lost worker, rsync-only data movement. The smoke tests exercised all of it, including a destroyed worker mid-stage, and it held. Snakemake or DVC would do similar things, but there's no reason to switch now.
- **On-worker execution, the check script and the cost controls:** cheap and proven ($0.15–0.56 per test).
- **emotion2vec+ embeddings:** in a 39/1 leave-one-out test they predicted the arousal and valence model at r≈0.66, nearest neighbours shared the top emotion 68% of the time against 21% chance, and they weren't dominated by length or game. They're the right base for probes.
- **Whisper large-v3** for subtitle checks: median WER 0, worst 0.17. It just works.

**Weak; treat as provisional:**
- **Gemma 4 12B as audio judge.** LISTEN found audio LLMs near chance on tone alone, and our captions were near-identical. The shortlist rating with a control group (PR #18) measures whether it can hear each axis. If separation is near zero, drop it from that axis.
- **Gemma 4 12B as text judge.** Plausible, but unmeasured. It decides most of the signal for playfulness and intoxication, where no acoustic model helps.
  - Game text can't go to a hosted API (the copyright rule), so the alternatives are larger open models on the worker.
  - Run a judge bake-off on the reviewed clips before trusting it.
- **The fusion math.** The confidence formula was designed, not fitted. After reviews, fusion should be learned (per-axis logistic stacking with grouped folds), and confidence should mean "probability this clip is a positive".
- **DNSMOS.** Only useful for picking TTS references, and it costs the most CPU time in clean. Its score on the helmet voice is suspect. Worth deferring until TTS is unblocked.
- **The audeering arousal/valence/dominance model.** Its output is stored but unused. Arousal is a good extra input to the probes (the energy behind anger, joy and worry).

**Process gaps:**
- Model assumptions were tested after building on them.
- Two sessions built the same checkpointing in parallel.

The first is fixed by acceptance checks (below). The second needs the coordinator to hand out work before it starts.

## 4. Proposed changes

Ordered by what has to happen before the full run, before review, and after review.

### Before the full run (cheap, code only)
1. **Group calibrate's folds by conversation** (2e).
2. **Stop counting stage directions twice** (2f). Options:
   - (a) drop direction priors when the judge saw the direction (recommended; the judge reads it in context)
   - (b) hide the direction from the judge and keep the priors
3. **A confound report in every stage that makes a signal.** Per axis, how well clip length, game and bark-vs-conversation predict each signal (R², or AUC for the flags). A high number is a warning before anything is built on it (2g).
4. **Model acceptance checks** as part of the smoke pipeline: each model must pass on the smoke set. Examples:
   - Gemma answers rather than refuses.
   - Embeddings aren't mostly length.
   - The speaker check doesn't flag sighs.
   - The judge doesn't put zeros everywhere.

   Fail loudly, before the full run (section 1's lesson).
5. **Defer DNSMOS** behind a setting until TTS work starts (saves ~25% of clean's time).
6. **A data-sufficiency report after label.** Per axis, the number of clips whose fused score is ≥ 0.6 and ≥ 0.4. That shows early which axes can support TTS contrast sets (2h).

### Review design (replaces "150 random clips with six sliders")
7. **Per-axis binary passes, not six sliders per clip.** The page asks one question at a time, "Clear example of SAD? yes / no / unsure", over clips sampled for that axis. That's a few seconds per answer, so an hour covers roughly 600–900 judgements instead of 150–300 clips.
8. **Stratified sampling with weights** (2a). For each axis, sample from score bands:
   - all of the top band, up to a cap
   - a good share of the middle band
   - a thin random slice of the bulk

   Record each clip's selection probability so precision and recall come out unbiased. Aim for **≥ 30 confirmed positives per axis** where the data allows, which is what the precision estimates need. Grow beyond 150; ~100 judgements per axis is a reasonable first target.
9. **Test-retest:** about 10% of judgements come back later, blind, to measure Brandon's own consistency, the noise ceiling (2d).
10. **Best-worst scaling for intensity, only if needed.** If the vectors need graded intensity rather than yes/no, add BWS rounds on the confirmed positives (2c).

### After review
11. **Judge on the decision** (2b). Per axis, report positive precision at the size we'd use for the contrast set, negative purity, and their intervals, from the weighted sample.
12. **Learned fusion:** a per-axis logistic stack on the signals plus the probe, with grouped folds, replacing the hand-set confidence formula.
13. **Judge-model bake-off** on the reviewed clips (Gemma 12B against a larger open model) before committing to one.
14. **Gate each axis.** An axis goes to contrast sets only if its precision interval clears a bar (say 0.8) and it has enough positives. Otherwise it gets the prompt fallback the plan already has.

### Where calibrate runs
Calibrate fits ridge regressions on about 3,400 × 1,024 embeddings with a few hundred labels, which takes seconds even on an old 4-core machine. It stays at home through nix-shell. No GPU or rental needed.

## 5. Questions for Brandon
1. Stage directions (item 2): (a) drop the priors when the judge saw the direction, or (b) hide the direction from the judge?
2. Review format (items 7–10): switch to per-axis yes/no passes with stratified sampling and test-retest? It changes the review page and the calibrate stage.
3. DNSMOS (item 5): defer until TTS?
