# Tali: embeddings, clusters, and mixed steering vectors (plan, draft 6)

Written 2026-10-13/14 in the "step back" thread. Draft 6 adds the critical-review fixes Brandon accepted on 2026-10-14 ([review](corpus-plan-critical-review.md)). Nothing here is built or run. DNSMOS: kept (Brandon, 2026-10-14).

Source note: arXiv pages could not be opened from this sandbox. Claims taken from search summaries are marked **(summary)**. My own reasoning is marked **(inferred)**.

## Goal
Find Tali's emotion and state classes in her own voice clips, without picking a list beforehand. Build one steering vector per class for both the voice model and the chat model. At runtime, mix and interpolate those vectors to get blends of, and transitions between, emotions and states. The six placeholder axes are gone, and the classes we find define the pack's variables.

Added 2026-10-17 (Brandon): discovered classes alone are too limited, so the moods a friend and assistant needs are also looked for directly ("Seeded moods" below). Clustering stays, to find what nobody thought to ask for.

## The core analysis problem
3,516 lines (202 min, re-extracted 2026-10-14) from three games, each recorded and processed differently. The 40-clip test showed two problems. The plain emotion embeddings mostly encoded **clip length**. The speaker embeddings identified the **game** 72% of the time. So the main risk is clusters that sort clips by recording conditions instead of emotion. Everything below is built to rule that out before we trust any cluster.

## Scope
This work builds the **corpus**: clean audio, embeddings, confound removal, stable clusters, named classes and per-clip class memberships, plus the Empathic-Insight-Voice backup scores. Choosing the voice and chat models, building the vectors and mixing (steps 7–9) are later work. They are kept below only as notes on what the corpus must support.

## Two kinds of covariates
- **Nuisance** factors are recording or production artefacts that can't carry emotion. They are **erased** in step 4:
  - game (the per-game recording and processing chain)
  - **text length**: syllables or characters from the subtitle, *not* duration. Duration = text × speaking rate, and speaking rate carries arousal (Scherer 2003; Juslin & Laukka 2003).
  - the recording noise floor measured in step 1
- **Situation** factors are real context that legitimately goes with emotion. They are **measured and reported, never erased**:
  - scene tags and structural tags
  - bark vs dialogue
  - stage direction
  - text content
  - level and vocal effort after normalisation (spectral tilt is arousal)

  Erasing `citadel_party` would erase drunk.
- There is no "channel" field in the corpus. The helmet filter is constant for Tali, and "radio" exists only as a scene tag. So channel is not a nuisance variable here, and we don't invent a label for it.

## Pipeline

### 1. Audio prep (home, CPU)
- Trim leading and trailing silence with a voice-activity detector *before* embedding. **Before trimming, store the spectrum and level of the trimmed silence** as a per-clip noise-floor measure. It is pure recording chain with no voice in it, and it is the cleanest nuisance covariate we have.
- Level: apply one linear gain per clip, either two-pass `loudnorm` with `linear=true` or pyloudnorm, falling back to RMS under about 1 s. Integrated loudness is unreliable on very short clips because of R128's 400 ms gating. **Never use dynamic compression**, because dynamics carry vocal effort.
- Resample a **16 kHz copy for embedding only**. The 24 kHz originals stay as they are for the later voice-model work.
- Optional per-game EQ match to a common long-term spectrum, behind a setting. It is fitted on the comparable-clip subset (step 4), because a game with more high-arousal speech is brighter for real reasons. Built (prep setting `eq_match`, 2026-10-17): the target is the mean of the three games' curves. It is for analysis only. The voice model keeps her suit sound (Brandon, 2026-10-17); `jobs/eq_match.py` makes a page to hear it. It stays off until a subset check says it helps: `characters/tali/eq_check.toml` preps about 600 clips with it on, embeds only them (emotion2vec+ large and A/V/D, on the GPU), and `jobs/eq_check.py` compares them with the same clips from the main embed. It reports how well the game can still be told from emotion2vec+ and how much of each clip's readings the EQ kept.

### 2. Filters (GPU, built)
- Whisper large-v3 checks that the subtitle matches. It skips `no_subtitle` clips.
- ECAPA checks that the speaker is her, within the same game. Its threshold is checked per length band. It is used only as a filter, never for clustering.
- **DNSMOS: kept** (Brandon, 2026-10-14). The EIV-Plus quality heads are also recorded: they are free, but they are distilled copies of DNSMOS (overall r 0.90, background r 0.87, speech-quality r 0.52 on LAION's validation). Both are reported covariates, alongside the noise floor from step 1, and are later used to pick reference clips. Neither is ever a gate.
- Flags only. Nothing is deleted.

### 3. Embeddings and backup scores (one GPU job, your ok first)
Candidates are computed in one pass. Selection is in step 4.
- **WavLM-large**, layers 6–18. Middle layers carry the most paralinguistic information (WavLM layer studies). Pre-narrowing the layers keeps the candidate count down.
- **emotion2vec+ large.** It was the best in the 40-clip test. Caveat: it was fine-tuned on 9 categories (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown), which may pull clusters toward them.
- **BUD-E-Whisper encoder** (the EIV backbone). **Pool only over real audio frames**: Whisper pads every input to 30 s (1,500 frames), so use the first ceil(duration × 50) frames. A CI test with the fake backend checks that the same content at different padding gives the same vector. The same check applies to any model that pads.
- **emotion2vec base**, one more try now that silence is trimmed.
- **audeering A/V/D** is a *guard*, not the selector (it is itself a wav2vec2 model and favours embeddings like it).
- Prosody (pitch, energy, speaking rate) and MiniLM text embeddings, for reporting and naming.
- **Empathic-Insight-Voice on every clip:** the 54 emotion and attribute heads, plus the Plus quality heads. The "content enjoyment" head is skipped. These are stored as the backup labelling and used for naming. Store float16, but compute everything downstream in float32 or float64.

### 4. Confound removal (home, CPU)
For each candidate:
1. **Measure first.** Probes for every nuisance and situation factor, with folds grouped by conversation, give the "before" table.
2. **Reduce dimensions first.** Fit a PCA on the pooled comparable subset and keep 64–128 dimensions. Raw 1,024-d per-game covariances are singular at our sample sizes.
3. **Re-colour each game to the pooled covariance** (CORAL-style, not plain whitening):
   - shrink each game's covariance with Ledoit–Wolf
   - map each game as x → Σ_pooled^½ Σ_g^−½ (x − μ_g) + μ_pooled, fitted on the comparable subset and applied to all clips

   Plain per-game whitening would flatten the space so that clustering finds nothing. This step already equalises the game means, so a separate LEACE for game is redundant.
4. **LEACE for text length** (log syllable count) and the noise-floor measure. Afterwards, also probe log(duration) and its square, because mean pooling makes the length effect non-linear.
5. **Comparable-clip subset** (hard data only): `exchange` or `dialogue_choices` lines, not barks, `to_shepard`, length-matched across games. It is reported with its size per game.
6. **Acceptance test, per candidate (fixed before results):**
   - **Game is gone, non-linearly too.** A linear probe passes by construction after erasure, so the test uses k-NN (k ≈ 15) and an RBF-SVM or gradient boosting, all with conversation-grouped folds stratified by game. The whole transform is refitted inside each fold without the held-out conversations: fitted on every clip, Coral levels each game's mean with the test clips in it, which puts even pure noise well below chance and lets a game that is still there pass. It passes if the upper 95% bound of *balanced* accuracy is within 5 points of chance or, when higher, of the median score with game shuffled **across conversations** (not clips) and the transform refitted the same way.
   - The text-length probe's R² is ≈ 0, plus the log-duration check above.
   - A/V/D prediction doesn't drop by more than a set margin (guard).
   - **Paired-take agreement**: how often both takes of the same line land on the same nearest cluster or neighbour, compared with a permutation baseline. This is a label-free reliability score.
7. **Selection without fooling ourselves.** Choose the candidate on 30% of conversations held out for selection, then report every acceptance number on the other 70%. Selection uses analytic binomial bounds and a few permutations (`selection_permutations`), because the home machine has 4 cores. Candidates that pass selection go to the report split best first, and the first to pass there is chosen; its permutations run only once its other tests pass. (A top-two cut once ranked out the only candidate that passed.)

### 5. Clustering (home, CPU)
- PCA, with the number of components chosen by parallel analysis.
- **GMM:** report BIC *and* ICL (ICL targets clusters, not density components). Covariance type and k are chosen by held-out likelihood.
- **Components to classes** by an established merging method, entropy-based (Baudry et al. 2010) or Hennig's (2010) DEMP or ridgeline. That is what "merge up" means.
- **HDBSCAN** as a second opinion (scikit-learn ≥ 1.3).
- **Stability (Hennig clusterboot bands, for mean Jaccard):**
  - below 0.6: not trusted
  - 0.6–0.75: a pattern, not a class
  - 0.75 or more: kept as a class

  We use 200 resamples **by conversation**, subsampling about 80% of conversations without replacement, with k and covariance type fixed across resamples. Byte-identical takes (same `audio.sha256`) are de-duplicated first.
- **Confound audit, nuisance only.**
  - A cluster predictable from game, text length or noise floor is rejected.
  - Text and scene predictability go in the enrichment table.
  - A cluster that text predicts *better than audio* gets a "topic, not tone?" flag, not a rejection.
- **Recording-session check.** The effective number of conversations per cluster, against a baseline drawn at the same cluster size, plus leave-the-biggest-conversation-out stability. A cluster mostly from one or two conversations is **flagged for your call** (decided 2026-10-14).
- **Two membership outputs:** the GMM responsibilities (these saturate near 0 or 1) and a **continuous per-class score** (the log-likelihood ratio of the class against the rest). The second gives intensity for mixing.
- UMAP is used only for plots.

### 6. Naming and checking classes (home, your ears)
- The purity review plays clips stratified across scenes and conversations, with **about 1 foil per 4 clips**: a clip from another cluster at a similar length. You answer "does this fit?"
- **Stopping rule: Wald SPRT** for p₀ = 0.5 against p₁ = 0.8 with α = β = 0.05, about 10–20 clips per cluster on average. Purity is reported corrected for your false-yes rate on foils.
- **Only after you answer** does it show the line, its context, the following lines, scene tags, stage direction, the prosody summary, the top EIV heads and the Gemma text name suggestion. This avoids anchoring.
- For each cluster, you name it, merge it, split it, or reject it.
- Hidden repeats run at about 1 in 8, targeting 50.
- Time: roughly **50–95 minutes** for 15–25 clusters.

### When a check fails (written before results)
| Check fails | What happens |
|---|---|
| Game still recoverable | Try EQ match on, then another layer or model. If still failing, **stop and report to you**. The bound is never loosened. |
| Text-length or duration effect left | Add log-duration to the eraser, re-check, then report |
| A/V/D guard drops too far | Reject that candidate and use the next best |
| No candidate passes everything | Stop and report, with the table |
| **Fewer than 2 clusters at Jaccard ≥ 0.75, or more than 60% HDBSCAN noise** | **Stop and bring it to you** (decided 2026-10-14). No automatic switch to continuous axes or to EIV labels. |
| Session flag | Your call, per cluster |
| Purity below 0.6 | Merge with its nearest cluster or reject it, your call in the review |
| A class below the minimum size | Merge, keep as a soft component, or flag. Never silently dropped |

### Seeded moods (added 2026-10-17)
Brandon's call: the classes the clusters happen to find aren't enough, so the moods an everyday friend and assistant needs are sought directly, from her lines' words and context. The list is `characters/tali/moods.toml` (drafted in the project's design/must-have-moods.md, cut by Brandon). One flat list; a line may carry several moods; vectors are later fitted jointly so mixes add up. Two degrees are knobs, not moods: energy (calm to urgent) and drunk (sober to drunk).

1. **context** (home, CPU): one record per line with the lines around it, the scene and the director's note, rendered once so every model reads the same text.
2. **textscore** (GPU, manual, Brandon's ok for that run): GoEmotions, zero-shot NLI, Qwen3-Embedding and two local LLMs (Gemma 4 31B and Qwen 3.6 27B, 8-bit) score every line against every mood. Each LLM answers twice per line. It needs no labels, so it runs once.
3. **gold** (Brandon): about 150 lines labelled from text, stratified so every mood appears.
4. **bake-off** (home, CPU; scripts first): every model against gold per mood (macro-F1 and run-to-run consistency; SetFit trained on gold by conversation-grouped folds). The rule picking a model per mood is written before results.
5. **candidates** (home, CPU): the top lines per mood from its winner, re-ranked by audio-only tone (prosody, emotion2vec). Lines whose text and voice disagree go to a sarcasm list.
6. **listening** (Brandon): candidates go through the purity review as pseudo-clusters, with the same foils, repeats and decisions.
7. **classes**: discovered and confirmed seeded classes in one table. A seeded class whose split-half class-vector cosine reaches `class_cosine` (0.9) steers voice and chat; one that doesn't is chat only.

Model output is never stored as a label; it only picks what Brandon hears.

### 7–9. Voice vectors, chat vectors, mixing (later work, out of scope)
Notes on what the corpus must support:
- **Voice vectors.** Hidden states of the eventual voice model (Qwen3-TTS 1.7B is one example, chosen later by comparison), teacher-forced on her clips. Class vector = the membership-weighted class mean minus the corpus mean. Built from her clips only.
- **Chat vectors.** Contrast on each class's lines with context, loaded through llama.cpp control vectors.
- **Mixing.** The `[[mix]]` table sets weights from the continuous class scores. Interference between summed vectors is a known risk, with known fixes (orthogonalising, per-layer injection, conceptors) **(summary)**.

## Corpus v2 specifics
- `singing` (le3.s769911_m) and clips clean flags as another voice or a subtitle mismatch are excluded from forming clusters. They are kept in the output.
- **Tali's paired takes** (`_m` / `_f`, the same line recorded for a male and a female Shepard):
  - De-duplicate byte-identical files by sha256.
  - Keep the remaining takes in the same fold and bootstrap draw.
  - Use them for the paired-take agreement score and for within-class noise.

  (Single-take distance does *not* limit how close classes can be: class means average that noise away.)
- **LE1 combat barks (129):**
  - Kept out of the eraser fit, because they are all one game and one situation.
  - They **join clustering if the shortest-usable-clip test passes them**, like any short clip (decided 2026-10-14).
  - One sensitivity clustering with and without them is reported.
- Scene, structural tags, `context` and `following` are hard data, shown in the review and enrichment tables. EIV scores, cluster ids and class names stay in stage outputs, never in `lines.jsonl`.

## Sample sizes, derived
Every count follows from what it must measure. Thresholds are written before results.

1. **Shortest usable clip: a retrieval test.** Crop long clips to 0.5, 1, 1.5 and 2 s. In the cleaned, centred space, check how often a crop's nearest full clip is its own source (or that clip's paired take). The minimum length is where retrieval reaches 80% of the 2 s level. Cross-check: do short paired takes agree as well as long ones? Raw cosine isn't used, because pooled speech embeddings are anisotropic and raw cosine is high even between unrelated clips.
2. **Minimum class size.** The vector is a mean difference, measured in the reduced clustering space (small, known d):
   - Draw subsets of n = 10, 20, 40, 80 and 160 clips and compute the split-half cosine, stepped up with Spearman–Brown to the full n. **n is the class total.**
   - Threshold: cosine 0.9, meaning the vector's direction error is under about 26°.
   - Prior guess: 100 or more per class, so roughly 15–25 classes from about 3,000 usable clips.
   - Re-checked on model activations later.
3. **GMM size.** d(d+3)/2 parameters per full-covariance component: 44 at d = 8, 152 at d = 16 and 560 at d = 32. "5–10 clips per parameter" is a heuristic.
4. **Bootstrap.** 200 resamples gives an SE of about 0.011 on mean Jaccard.
5. **Purity listening.** SPRT as above, plus foils at 1 in 4. The 0.6 bar is chosen because with soft memberships, a 60%-pure class gives a vector that is about 60% signal. That is the lowest that still points the right way **(inferred)**.
6. **Your consistency.** Kappa: ±0.31 at 20 repeats, ±0.20 at 50 and ±0.14 at 100. Target 50, and also report your false-yes rate on foils.
7. **Steering tests (later).** 34 sentences per class per strength for d = 0.5, and about 25 blind A/B trials per class.

## Validation rules throughout
- Folds and resamples are grouped by conversation, and paired takes stay together.
- Selection and reporting use different conversations.
- Results are reported per game and per length band.
- Model revisions and packages are pinned. LEACE is a roughly 30-line numpy implementation (or EleutherAI `concept-erasure`), so there's no torch at home.

## Before anything runs
The only GPU step is the one `embed` job. It runs only after you've seen the exact command and said ok. Everything else runs on CPU at home, at low priority.

## Sources
- Critical review, 2026-10-14: [corpus-plan-critical-review.md](corpus-plan-critical-review.md) (full source list)
- LEACE: https://arxiv.org/abs/2306.03819 · Kernel concept erasure: https://arxiv.org/abs/2201.12191 · CORAL: https://arxiv.org/abs/1511.05547
- ICL (Biernacki, Celeux & Govaert 2000) · Component merging (Baudry et al. 2010; Hennig 2010) · clusterboot bands: https://search.r-project.org/CRAN/refmans/fpc/html/clusterboot.html
- SPRT (Wald 1945) · Optional stopping (Armitage et al. 1969) · Intrusion test (Chang et al. 2009)
- Speaking rate and arousal (Scherer 2003; Juslin & Laukka 2003) · Anisotropy (Ethayarajh 2019)
- Steering sample efficiency: https://arxiv.org/pdf/2505.03189 , https://arxiv.org/pdf/2604.14090 , https://arxiv.org/pdf/2605.15604
- EIV Small / Plus / BUD-E-Whisper: https://huggingface.co/laion/Empathic-Insight-Voice-Plus , https://huggingface.co/laion/BUD-E-Whisper
- emotion2vec+ large (9 classes): https://huggingface.co/emotion2vec/emotion2vec_plus_large
- WavLM layers: https://arxiv.org/html/2405.04485v1 · EmoBox: https://arxiv.org/pdf/2406.07162
- Steering composition: https://arxiv.org/pdf/2502.19649 , https://arxiv.org/pdf/2410.16314
