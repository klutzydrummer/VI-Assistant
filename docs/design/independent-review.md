# Independent review: Tali dataset pipeline (labelling for mood control vectors)

Written 2026-10-13 by a fresh reviewer (a subagent with no prior context, read-only access, told to form its own view before reading earlier reviews). Saved unedited. It had no access to `methodology-review.md`, which was then only on an unmerged branch.

## Verdict

Gemma 4 12B is the wrong tool for both audio jobs (a and b). For the text judge (c) it is adequate, but the way scores are read out of it should change. The bigger problem is the overall labelling design, which solves a harder problem than the goal needs. Neither consumer needs a calibrated six-axis score on all 3,400 clips. `cvector-generator` needs text pairs, and the TTS steer stage needs about 12 clips per side per axis (`refs_per_side = 12`). That is a high-precision retrieval problem: find a few dozen clear exemplars per axis, plus matched neutrals.

Rebuild labelling around three steps:
1. Cheap rankers.
2. Separate content and delivery targets.
3. One hour of blind, per-axis yes/no review of the top candidates.

Drop discovery captions and clustering. Swap the audio-LLM verifier for an acoustic model that actually covers the hard axes.

## 1. Gemma 4 12B, job by job

Facts checked: Gemma 4 12B-it has 11.95B parameters, takes audio up to 30 s, and is Apache 2.0. Its model card gives only ASR and translation audio numbers, with no emotion or paralinguistic benchmark ([HF card](https://huggingface.co/google/gemma-4-12B-it)). LISTEN (EACL 2026) found that audio LLMs mostly use the words. On paralinguistic-only cues they scored 15.7–22.7% against 12.5% chance, and Qwen3-Omni-30B scored 21.0% ([arXiv 2510.10444](https://arxiv.org/html/2510.10444v2)). So moving to a Qwen omni model does not fix the underlying problem.

**(a) Audio captioning for discovery: drop.** Smoke-test evidence (current discover run, `2377…`):
- **Coverage and variety:** 36 of 40 clips captioned (4 failures). Emotion words cluster on "focused" (16), "serious" (14) and "calm" (11), and the `delivery` field was empty on all 36. The captions follow one template ("…slightly filtered…").
- **Axis scores:**
  - Joy was 0 on 31/36, playfulness 0 on 33/36 and intoxication 0 on 35/36.
  - On the 10 clips captioned with and without the words, joy, playfulness and intoxication changed by exactly 0.0.
  - Sadness kept Spearman 0.81 between the two runs. That is expected: Gemma hears the words in the audio either way, so "without transcript" is not word-blind. The `words_change_scores` test cannot detect lexical reliance at all.
- **Clusters:** both clusters are named after the helmet filter or general weariness. Nothing came out that would change the pack's axes.

If you want a captioner anyway, the purpose-built option is Qwen3-Omni-30B-A3B-Captioner (Apache 2.0, audio-only, no prompt, best under 30 s; [HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)). It is about 32B parameters in BF16, so it needs FP8 to fit 48 GB, and LISTEN's result still applies. I would not spend effort on it.

**(b) 1–5 "delivery fits axis" rating on a shortlist: replace. There is no evidence for it yet.**
- The current label run (`5fb5…`) has no `verify_top`/`verify_low` settings. Its `audio_llm` signal is discover's caption scores, so the verifier has never run on real audio.
- On a shortlist the words are audible. By LISTEN's result the rating will mostly repeat the text judge, which is the signal that built the shortlist, so it adds correlated noise.
- The smoke data is consistent with this: text~audio_llm agreement on sadness is 0.72, against 0.32 for audio~audio_llm.

The replacement is an acoustic model trained on these categories: LAION's Empathic-Insight-Voice. It is CC-BY 4.0, uses a Whisper encoder with per-emotion regression heads, and covers 40 categories including Intoxication, Teasing, Amusement, Anger and Sadness ([HF](https://huggingface.co/laion/Empathic-Insight-Voice-Small)). On the EmoNet-Voice expert benchmark it scored Spearman 0.48 on intoxication against Gemini's 0.27, and teasing was the most detectable category on average at 0.62 ([arXiv 2506.09827](https://arxiv.org/html/2506.09827v2)). Two caveats: that benchmark is synthetic voice acting, and the model is unproven on her helmet-filtered voice. Even so, it is the only component that addresses the two axes with no acoustic signal today, and `steer.py` already loads it.

If you want one last test of the audio LLM: run the verifier on content-masked audio, low-passed around 400 Hz. Low-pass filtering keeps prosody and removes intelligibility, a classic method in emotion-perception research ([Frontiers 2015](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01340/full)). If top-versus-control separation disappears on masked audio, drop the verifier.

**(c) Text judge with context: keep the role, change how it scores. Model size is a cheap A/B.**

Smoke evidence:
- 0 unparsed replies across 40 lines.
- Every value sits on the 0.2 grid the prompt anchors.
- Intoxication was 0 on 40/40. Worry was non-zero on 25/40, which looks inflated by the instruction that "any trace scores at least 0.2".
- Audio~text agreement: worry 0.00, anger 0.24, joy 0.25.

Changes:
- Read scores as probability-weighted values from the token logprobs instead of sampled numbers (G-Eval; [Liu et al. 2023](https://arxiv.org/abs/2303.16634)). That turns grid-snapped scores into continuous rankings, which is what retrieval needs.
- Remove the "at least 0.2" floor.
- Ask per axis "is this clearly X?" rather than for six values at once.
- On 48 GB, Gemma 4 31B dense is about 31 GB of weights at FP8 (Gemma 4 sizes: [Google](https://ai.google.dev/gemma/docs/core)). At about 1.2 s per line, 3,400 lines cost under $1 on the 12B, and a model 2–3 times larger still costs a few dollars. Keep whichever wins on the reviewed set. I have not verified that 31B beats 12B on this task, so measure it.
- The text must stay on the rented worker, as it does now.

## 2. Is the labelling design sound?

Partly. The parts that are right:
- treating Brandon's reviews as ground truth
- refusing to trust agreement between models
- the steer stage as an early check on the riskiest assumption
- the prompting fallback for axes that won't steer

Three structural problems:

1. **One fused number serves two consumers with different targets.** The LLM vector needs what she says, which is content in context. The TTS vector needs how she says it, which is delivery. Game lines are often emotional in text and restrained in delivery, and the reverse. Fusing text and acoustic signals into one score means:
   - TTS exemplars picked by `label`'s top clips may not sound emotional (the fused score leans on text);
   - LLM pairs may be filtered by audio that is irrelevant to them.

   Keep two scores per axis, `content` and `delivery`, and two review questions.

2. **The hour is spent on the wrong sample.**
   - The page asks six sliders per clip, and 150 random calibration clips come first. At a realistic 20–30 s per clip, the calibration sample alone uses most of the hour.
   - In the smoke test the fused score reached ≥0.6 on 4 joy clips, 1 sadness, 1 anger, 1 worry, 0 playfulness and 0 intoxication out of 40. A random 150 will hold a handful of positives on most axes and probably none for intoxication.
   - `calibrate`'s per-axis Spearman and ridge probe on mostly-zero targets will then measure the ability to rank neutrals.
   - The probe needs "a few hundred reviewed clips", which one hour will not produce for rare axes.

3. **Review anchoring inflates the calibration metric.** The page pre-fills each axis with the model's score, and Enter keeps it. Any axis Brandon doesn't touch is recorded as the reviewed value, so reviews will "agree" with the model by construction. Calibration accuracy, and the go/no-go per axis, will be over-optimistic.

**A better design for contrast sets:**
- **Rank:** per axis, rank clips by
  - text judge logprob scores (content);
  - Empathic-Insight-Voice plus emotion2vec+ (delivery);
  - stage-direction priors.
- **Review blind:** for each axis, show the top ~40 candidates plus ~10 random clips, one axis at a time. Ask one blind question with three answers: clear / weak / no, judged by ear. At 5–10 s each, 6 × 50 ≈ 300 judgments takes 30–50 minutes. Use the rest for matched neutrals: same game, similar duration and speaking rate, judged "no" on all axes.
- **Stop rule:** precision@k on the random-ten controls and the top list tells you directly whether an axis has enough clear exemplars. No fusion-weight fitting is needed.
- **LLM vectors:** build pairs mostly from persona prompts ("as Tali, slightly drunk" vs "as Tali, sober") over diverse suffixes, the repeng method ([vgel.me](https://vgel.me/posts/representation-engineering/)). Add her reviewed real lines and the minimal-pair rewrites. LLM vectors hardly need the audio labels.
- **Intoxication:** hand-pick it. The canon drunk scenes are few, harvest emits no `scene_tags` (0/40 clips; it writes `scene.prefix`), so the `citadel_party` prior never fires. Use the `steer.settings.refs` override and the VoiceDesign pairs.

## 3. Other components

| Component | Verdict | Evidence / reason |
|---|---|---|
| Whisper large-v3 | Keep | 0.35 s/clip; subtitle WER median 0 on 33 spoken clips; catches mis-mapped clips. |
| ECAPA speaker check | Keep; make the threshold duration-aware | Similarity correlates with duration (r = 0.42 over 40 clips); the lowest are short barks (0.26–0.42 at <1.6 s). Nonverbal clips (sim ≈ 0) are already excluded correctly. |
| DNSMOS | Keep, only for picking TTS refs, compared within duration bands | Correlates with duration r = 0.44, probably from looping short clips to fill its 9 s window. |
| emotion2vec+ large | Keep as a cheap delivery signal and embedding | Argmax "angry" on 13/40, uncorrelated with speaker similarity (r = 0.04). Covers only 4 axes ([HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large)); the class bias needs checking against blind reviews. |
| audeering A/V/D | Drop, or actually use it | Computed in `label.py` but never fused or calibrated: dead weight. CC BY-NC-SA, "research purpose only" ([HF](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)). Arousal could serve as a matching variable for negatives; otherwise remove it. |
| Empathic-Insight-Voice | Add to label | See 1(b). |
| Discover clustering (UMAP + HDBSCAN) | Drop as a stage; keep prosody features | 2 clusters, 23/40 noise; the clusters it finds are recording condition, game and bark-vs-talk, which are better handled as explicit filters. The axes are fixed by the pack and mixing table, so "discover new axes" has no consumer. Keep `prosody.jsonl` for matching negatives. |
| Stage-direction priors | Keep; fix the regexes | 6/40 clips have a direction and the prior fired on 1. This is the cleanest human signal you have for LE3; read the directions and widen the patterns, or have the text judge interpret them. |
| Fusion weights + `calibrate` | Simplify | Fine as a ranking heuristic. The regression calibration is low value given rare positives and anchoring (section 2). |
| Review page | Change | Blind per-axis question, no pre-filled model score, top-k plus controls, content and delivery asked separately. Re-rate ~20 clips at the end for test–retest reliability. |

## 4. Biggest risks the current approach doesn't address

1. **Too few genuine positives** for intoxication and playfulness, and possibly joy, in a mostly serious character. No amount of fusion creates exemplars that aren't there. Find out early by counting "clear" judgments per axis.
2. **Confounds leaking into vectors.** Clips differ by game, by recording condition (helmet, cutscene vs bark), and by line length. Contrast sets matched only on length and rate can produce an "anger" vector that is really "LE3 combat". Match negatives on game and condition too, and cap barks.
3. **Content–delivery confusion.** This is the TTS risk in section 2.1: references chosen for emotional words with flat delivery will produce weak or null TTS vectors. The steer stage will report that as "doesn't steer" when the cause is bad reference picks.
4. **The go/no-go metric is biased** by anchoring (section 2.3).
5. **LLM steering is input-dependent.** Some datasets steer close to half of inputs the wrong way ([Tan et al., NeurIPS 2024](https://arxiv.org/abs/2407.12404)). The plan's per-prompt sweep handles this, provided it actually reports per-prompt direction.
6. **A single annotator, with low human agreement on fine-grained emotion** (EmoNet-Voice expert Cronbach's α = 0.14). Use coarse three-level judgments and a test–retest subset.
7. **Generalising to Carl.** There are no stage directions, possibly no aligned subtitles or context, and a cartoon voice outside the acoustic models' training data. A design that depends on six calibrated fused signals will port poorly. A design of rank, blind top-k review and persona-prompt LLM pairs ports well.
8. **A simpler TTS route may beat steering.** Qwen3-TTS Base, the clone model, has no instruction control; only CustomVoice and VoiceDesign do ([HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)). Choosing a mood-matched reference clip is the obvious baseline, and it needs only the reviewed exemplars. The steer stage already compares against it, so keep that comparison decisive.

## 5. Prioritised changes

1. **Make review blind and per-axis, over top-k candidates plus random controls.** Record content and delivery separately and drop the pre-filled model scores. *Evidence:* the anchoring in the review design; base rates of 0–4 positives per 40 clips; downstream needs ~12 per side.
2. **Separate `content` and `delivery` scores per axis.** Choose TTS references by delivery only and LLM material by content only. *Evidence:* audio~text agreement of 0.00–0.46 across the four mapped axes; the different consumers.
3. **Run the steer test with hand-picked references first** (anger, plus hand-picked drunk lines), before any further labelling work. It already exists, and its result decides how much labelling the TTS side needs at all.
4. **Add Empathic-Insight-Voice as a delivery signal; drop the Gemma verifier** unless it survives a content-masked (low-pass) separation test. *Evidence:* LISTEN; EmoNet-Voice per-emotion results; the verifier has never run on real data.
5. **Score the text judge from logprobs, per axis, without the 0.2 floor.** A/B Gemma 4 12B against a ~31B model at FP8 on the reviewed set. *Evidence:* every value on the 0.2 grid; worry non-zero on 25/40 lines; G-Eval.
6. **Build LLM contrast pairs mainly from persona prompts** (repeng style) plus reviewed real lines and minimal pairs, and sweep on open-ended chat. *Evidence:* vgel.me; Tan et al.
7. **Delete the discover stage's captioning and clustering, and the unused audeering call.** Keep prosody and the embeddings. *Evidence:* template captions; 0.0 change when words are shown; 2 clusters, 23/40 noise; A/V/D never consumed.
8. **Fix the priors.** Emit `scene_tags` from harvest (`scene.prefix`/`dlc`), and widen the stage-direction patterns (1 hit in 6).
9. **Small hygiene fixes:**
   - make the ECAPA threshold duration-aware;
   - compare DNSMOS within duration bands;
   - match negatives on game and condition;
   - check the emotion2vec+ "angry" bias against blind reviews.

## Agreement/disagreement with the previous review

`docs/design/methodology-review.md` does not exist in this checkout, so I compared against `prior-work.md` only.

**Agree:**
- LISTEN means audio-LLM emotion labels can't be trusted.
- Calibrate to Brandon's reviews, not to agreement between models.
- A speaker embedding replaces the `other_voice` tag, with a median per condition.
- Barks are a weak spot.
- Steering vectors are unreliable, so sweep on open-ended prompts and keep a prompting fallback.
- Reference-clip selection is a serious alternative for TTS.
- Retrieval of real lines plus point-in-time tagging on the persona side.

**Differ:**
1. **The verifier.** It kept Gemma as a ParaSpeechCaps-style verifier. That cascade (per prior-work.md) used a stronger hosted model as the last filter after DVA and text filters. Here the shortlist is built mostly by the text judge, and the verifier hears the same words, so its agreement is close to circular. I would not keep it without a content-masked test.
2. **Captions.** It kept open captions "where being wrong is cheap". The cost is not GPU time but pipeline complexity, plus a stage whose output (axis discovery) has no consumer. Drop it.
3. **Review allocation.** It endorsed a random calibration sample plus an uncertainty queue, and per-axis probes after "a few hundred reviewed clips". With rare positives and one hour, I'd spend the review on top-k precision per axis instead. It also didn't flag the anchoring in the review page.
4. **A missed model.** It didn't note that Empathic-Insight-Voice covers intoxication and teasing, the two axes with no acoustic signal. The repo already uses it in `steer.py` but not in `label.py`.
5. **Separate targets.** It treats one fused label per axis as serving both LLM and TTS. I think content and delivery must be separate targets.

## Sources

- Gemma 4 12B-it model card: https://huggingface.co/google/gemma-4-12B-it
- Gemma 4 family: https://ai.google.dev/gemma/docs/core
- LISTEN (EACL 2026): https://arxiv.org/html/2510.10444v2
- Qwen3-Omni-30B-A3B-Captioner: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner
- Empathic-Insight-Voice: https://huggingface.co/laion/Empathic-Insight-Voice-Small
- EmoNet-Voice: https://arxiv.org/html/2506.09827v2
- emotion2vec+ large: https://huggingface.co/emotion2vec/emotion2vec_plus_large
- audeering A/V/D: https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
- Qwen3-TTS Base: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- G-Eval: https://arxiv.org/abs/2303.16634
- Tan et al. 2024, steering reliability: https://arxiv.org/abs/2407.12404
- repeng control vectors: https://vgel.me/posts/representation-engineering/
- Low-pass content masking: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01340/full
- llama.cpp cvector-generator (mean/PCA methods, layers above 10): https://github.com/ggml-org/llama.cpp/tree/master/tools/cvector-generator

No repo files were modified. The smoke-test numbers above come from local analysis of `corpus/tali-smoke/stages/{harvest,clean,discover,label}/<current>/out/`.
