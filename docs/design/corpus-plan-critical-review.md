# Critical review: Tali corpus plan (draft 5) and build primer

> Saved from the project's shared folder. The plan it reviewed (draft 5) became [corpus-pipeline.md](corpus-pipeline.md) draft 6, with every finding Brandon accepted on 2026-10-14 folded in.

Reviewed 2026-10-14 for Brandon, at his request. Documents reviewed: `design/embedding-clustering-plan.md` (draft 5) and `design/corpus-build-primer.md`, both read in full, plus the "Rethinking the Tali dataset plan" thread, `docs/design/corpus-format.md`, `crates/harvest/src/scenes.toml`, `jobs/discover.py` and `jobs/checks.py` at main 604412c.

Everything here is a **proposal**. Nothing has been edited, built or run. The plan thread folds in only what Brandon approves.

I reviewed against the fixed design and did not propose replacing it:
- Classes come from clustering embeddings of Tali's own clips.
- Each class gets one vector for both the TTS and the chat LLM, built from her clips and lines.
- Vectors are mixed and interpolated at runtime.
- Empathic-Insight-Voice (EIV) is only a backup.
- No audio LLM is used as a listener.
- Metadata is hard game data.
- The scope is the corpus only.

Findings that would touch that intent are listed separately at the end, as Brandon's call.

Tags: **(verified)** means I checked it against the repo, a model card or arithmetic. **(source)** means published work, cited. **(inferred)** means my own reasoning.

---

## Top issues (in order)

1. **Per-game whitening, as written, flattens the space the clustering needs** (High, §4.3). Whitening each game sets its covariance to the identity. The pooled data is then close to isotropic, so the PCA, parallel analysis and GMM have little left to find. CORAL whitens and then **re-colours** to a target covariance. At 1,024 dimensions with a few hundred to ~1,500 clips per game, the raw per-game covariance is also singular.
2. **The main acceptance test passes by construction** (High, §4.6). LEACE guarantees that no *linear* probe can recover the erased concept on the data it was fitted on. "The linear game probe is at baseline after LEACE" therefore tests almost nothing. The test needs non-linear probes, and an equivalence bound instead of "not significantly better than chance".
3. **The confound audit rejects clusters that are predictable from text** (High, §5). This contradicts the plan's own rule that situation factors are reported and never removed. It would also reject exactly the classes whose lines differ, and those are the ones the chat vectors need.
4. **Clip length is treated as pure nuisance, but duration carries speaking rate, and rate carries arousal** (Medium-High, §4.4, Corpus v2 item 1). Erase text length (syllables or characters from the subtitle) instead of duration. Then speaking rate survives.
5. **The BUD-E-Whisper encoder needs pooling only over the real audio frames** (High, practical, primer stage 3). The Whisper encoder pads every input to 30 s. A plain mean over its 1,500 output frames would be mostly padding, and that is a strong length artefact. The same check applies to any pooled model.
6. **Cosine thresholds of 0.9 are unjustified, and they behave badly in raw embedding space** (Medium-High, Sample sizes items 1 and 2). Mean-pooled self-supervised speech embeddings are anisotropic, so raw cosine between unrelated clips is already high. Use centred embeddings and a retrieval or relative criterion.
7. **The purity review has no foils and stops sequentially without correction** (Medium-High, step 6, Sample sizes item 5). Asking "does this fit the name?" only about clips from that cluster invites yes-bias. Mix in clips from other clusters (an intruder test) and use a proper sequential rule (SPRT or a fixed stopping table).
8. **GMM plus BIC over-splits, and "merge up the hierarchy" has no defined method** (Medium, step 5). Use ICL alongside BIC, and use an established component-merging method (Baudry et al. 2010; Hennig 2010) to turn components into classes.
9. **"Classes closer than the take-to-take distance aren't separable" is statistically wrong** (Medium, Corpus v2 item 3). Class *centroids* are resolved far below single-clip noise. The paired takes are better used as a reliability test: do both takes land in the same cluster? Byte-identical `_m`/`_f` pairs must also be de-duplicated first.
10. **The plan has no written rule for what happens when an acceptance test fails, including the EIV fallback trigger** (Medium). It should be written down before any results, like the thresholds are.

---

## Findings by section

### Step 1: audio prep

**1.1 Loudness normalisation on short clips (Low-Medium, practical).**
- Evidence: single-pass ffmpeg `loudnorm` runs in dynamic mode, which compresses the signal. Integrated loudness is unreliable on clips under about 3 s, because of the 400 ms gating blocks (EBU R128 / ITU-R BS.1770) **(source: BS.1770 gating; inferred for the effect)**. About a third of the corpus was under 2 s.
- Fix: run two-pass `loudnorm` with `linear=true`, or apply a single gain from pyloudnorm, falling back to RMS for clips under ~1 s. Never apply dynamic compression, because the dynamics carry vocal effort.

**1.2 Keep the source-rate audio (Low).** Harvest writes 24 kHz audio (`corpus-format.md:49`) **(verified)**. The 16 kHz resample should be an embedding-only copy. The later TTS vector work needs the original audio.

**1.3 Use the trimmed silence as a pure channel measure (Medium, improvement).** The leading and trailing silence that VAD removes holds the recording chain's noise floor and spectrum, with no emotion in it. Store its spectrum per clip before trimming. It is a clean nuisance covariate for the game and session probes, and for checking whether the optional EQ match worked **(inferred)**.

**1.4 The EQ match has the same emotion-leak risk as erasure (Low).** If one game has more high-arousal speech, its long-term spectrum is brighter partly for real reasons. Fit the per-game EQ on the comparable-clip subset, as the plan already does for the eraser.

### Step 3: embeddings

**3.1 Whisper-encoder padding (High, practical).** Whisper's feature extractor pads or trims every input to 30 s, and the encoder always emits 1,500 frames **(source: Whisper / transformers `WhisperFeatureExtractor`)**. Pool BUD-E-Whisper over the first `ceil(duration × 50)` frames only. Add a CI test with the fake backend checking that two clips of different lengths with identical content give identical pooled vectors. Check the same thing for any model that pads.

**3.2 Model card fact (Low, verified).** emotion2vec+ large's card lists 9 categories (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown), not the 5 in the plan or the "about 4" said in the thread. The caveat still stands, but the plan should state it correctly.

**3.3 The yardstick may bias the choice of model (Medium).**
- Evidence: the audeering A/V/D model is itself a wav2vec2-large-robust model fine-tuned on MSP-Podcast. "Keeps the most A/V/D prediction" therefore favours embeddings that resemble it. A/V/D also cannot see the states the project cares about, such as drunk or teasing **(inferred)**.
- Fix: keep A/V/D as a guard (it must not fall much), not the selector. Select on a combination that includes take-pair cluster agreement (see 9) and nearest-neighbour listening.

**3.4 Selection over many candidates on the same data (Medium).**
- Evidence: 25 WavLM layers, plus 3 other models, times with and without EQ, gives about 56 candidates. They are picked by the same probes that later judge the result, a "garden of forking paths" risk.
- Fix: choose the candidate on a held-out set of conversations (for example 30%, grouped by conversation) and report the acceptance numbers on the rest. Alternatively, pre-narrow WavLM to the middle layers (about 6–18), where the cited work places paralinguistic information.

### Step 4: confound removal

**4.1 Per-game whitening destroys the cluster structure (High).**
- Evidence: whitening game g maps its covariance Σ_g to I. After per-game whitening and centring, the pooled covariance is close to I, so the PCA eigenvalues are flat and parallel analysis keeps nothing. Real structure survives only in higher moments **(inferred, straightforward linear algebra)**. CORAL (Sun et al. 2016, "Return of Frustratingly Easy Domain Adaptation") whitens the source and then re-colours it with the target covariance **(source)**. It does not leave data whitened.
- Second problem: with d = 1,024 (WavLM) and n per game in the hundreds to ~1,500, the sample covariance is rank-deficient. Whitening then divides by near-zero eigenvalues and amplifies noise.
- Fix:
  - First reduce dimensions with a PCA fitted on the pooled comparable subset (keep, say, 64–128 dimensions).
  - Estimate each game's covariance with Ledoit–Wolf shrinkage (`sklearn.covariance.LedoitWolf`).
  - Map each game as Σ_pooled^{1/2} Σ_g^{-1/2}(x − μ_g) + μ_pooled. That is, re-colour every game to the **pooled** comparable-subset covariance.
  - Fit on the comparable subset and apply to all clips.

**4.2 LEACE for game after per-game centring is redundant (Low).** Centring each game already equalises the class-conditional means, which is all LEACE removes for a categorical concept. Keep one or the other in the code, and say which. Order: re-colouring (4.1), then LEACE for length or text-length.

**4.3 The acceptance test is close to tautological (High).**
- Evidence: LEACE's theorem guarantees that no linear predictor beats the constant predictor for the erased concept, on the fitting distribution (Belrose et al. 2023) **(source)**. Linear erasure leaves information that non-linear probes recover (Ravfogel et al. 2022, kernel concept erasure; the plan's MANCE note) **(source)**. The GMM itself uses second moments, so it is a non-linear reader.
- Fix: the game acceptance test should use at least one non-linear probe (k-NN with k ≈ 15, plus an RBF-SVM or gradient boosting), with conversation-grouped folds. Also report the mutual information between the final clusters and the game (the confound audit partly does this).

**4.4 "No better than the permutation baseline" is the wrong kind of criterion (Medium-High).**
- Evidence: failing to reject the null is not evidence that the confound is gone. A noisy probe passes more easily, so a worse setup can look better.
- Fix: use an equivalence bound, for example "the upper 95% bound of balanced accuracy is within 5 points of chance". Use balanced accuracy rather than accuracy against the majority share.
- Permutation scheme: permute game labels **across conversations**, not across clips. Clip-level shuffles break the grouping the folds rely on and give too narrow a null.

**4.5 Length: erase text length, not duration (Medium-High).**
- Evidence: duration ≈ amount of text × speaking rate. Speaking rate is one of the most reliable acoustic correlates of arousal (Scherer 2003; Juslin & Laukka 2003) **(source)**. Erasing duration erases part of rate.
- Fix: the length concept should be log(syllable count) (or the subtitle character count) from the subtitle. Barks have no subtitle, but they are already outside fitting. Check that the leftover length effect is gone with the length probe after erasure. Also probe log(duration) and its square, because frame-mean pooling makes the length effect non-linear.

**4.6 Loudness is partly emotion too (Medium).** After R128 normalisation, the remaining "loudness" variable is ill-defined. Vocal effort shows up in spectral tilt and is real arousal. Normalise level once in prep, and do **not** erase a loudness direction. Probe it and report it only. If it must be treated as a nuisance, use the trimmed-silence noise floor (1.3), which is purely recording.

**4.7 "Channel (helmet, radio, clean)" has no hard-data source (Medium, intent-adjacent).**
- Evidence: the corpus has no channel field (`corpus-format.md` lists only `audio.channels`, the count of audio channels) **(verified)**. `radio` exists only as a *scene* tag from a level name (`scenes.toml:210`, e.g. `war40_radiochatter`), which is a situation factor. The helmet filter is constant for Tali.
- Fix: either drop channel from the nuisance list, or define it from hard data (the radio scene tag, used for reporting only) or from a measured signal kept in stage outputs, such as the noise-floor spectrum. Don't invent a channel label.

**4.8 The amount of CPU work at home is underestimated (Medium, practical).**
- Evidence: about 1,000 permutations × ~56 candidates × ~10 confounds × 5 folds is millions of probe fits, on 4 cores at nice 19 on a machine whose system disk is busy **(inferred)**.
- Fix: run permutations only for the shortlisted candidate or two. For screening, use analytic chance bounds (binomial on balanced accuracy). Cache the fold splits and write results as small files.

### Step 5: clustering

**5.1 BIC picks density components, not clusters (Medium).**
- Evidence: BIC on a GMM chooses the number of Gaussians needed to fit the density. Non-Gaussian clusters get several components each. ICL (Biernacki, Celeux & Govaert 2000) penalises overlapping components and targets clusters **(source)**.
- Fix: report BIC and ICL together. Turn components into classes with an established merging method: entropy-based merging (Baudry et al. 2010, "Combining mixture components for clustering") or Hennig's (2010) merging methods (DEMP, ridgeline). That also gives the plan's "merge up the hierarchy" a defined procedure, since a GMM has no hierarchy of its own. Ward linkage on component means is a cruder alternative.

**5.2 Bootstrap details (Medium).**
- Evidence: Hennig's guidance (fpc `clusterboot` documentation) is: a *mean* Jaccard below 0.6 means don't trust the cluster, 0.6–0.75 means patterns, and ≥ 0.75 means valid. The ≤ 0.5 figure is the per-resample "dissolved" cut-off, not a band for the mean **(source)**. The plan's bands mix the two and leave 0.5–0.6 undefined.
- Duplicates: a bootstrap repeats points. That inflates density for HDBSCAN (`min_samples`) and can make degenerate GMM components. Hennig recommends subsetting for such methods.
- Fix:
  - Use the documented bands.
  - Resample by conversation (a cluster bootstrap), or subsample about 80% of conversations without replacement.
  - Fix the model's k and covariance type across resamples, so stability measures the clusters and not the model selection.

**5.3 The text rule in the confound audit contradicts the plan (High).**
- Evidence: step 5 says a cluster predictable from "game, length, channel or text alone is rejected". Corpus v2 item 1 says situation factors (scenes, structure, context) are reported, never erased. Text is the most direct situation factor. Sad lines say sad things, and step 8 needs classes whose lines differ.
- Fix: reject only on nuisance factors (game, recording session as Brandon decides, length). Report text and scene predictability in the enrichment table. A cluster that text predicts *better than audio* is worth a flag ("this may be a topic, not a tone"), not an automatic rejection.

**5.4 GMM posteriors are not blend weights (Medium).**
- Evidence: in more than a few dimensions, GMM responsibilities saturate near 0 or 1. A mildly angry clip and a very angry one both get about 1.0 **(inferred, well-known behaviour)**. The plan says memberships "directly give mix weights".
- Fix: store responsibilities as they are, and also a continuous per-class score, such as the log-likelihood ratio of the class against the rest, or the projection onto (class mean − rest mean). Steering vectors need weights; runtime mixing will need intensity. This adds a column and changes nothing in the design.

**5.5 Barks: excluding them from the eraser is right, but excluding them from clustering is a separate choice (Medium, part intent).** The stated reason, that all barks are one game and one situation, applies to fitting the *eraser*, not to finding *classes*. Combat barks may be the main evidence for an alarm or urgency state. If they are only assigned afterwards, that state can't form. Fix: keep them out of the eraser fit, and let the shortest-usable-clip test (Sample sizes item 1) decide whether they join clustering, like any other short clip. Run one sensitivity clustering with them in and report the difference. *Whether combat barks may form their own class is Brandon's call.*

**5.6 The continuous fallback has no trigger (Medium).** "If HDBSCAN finds mostly noise" needs a number written down before the run, for example "fewer than k clusters pass Jaccard ≥ 0.75, or more than 60% noise". The same goes for the EIV fallback in step 6: "if the clusters don't give usable classes" should be a written rule. See also the intent section below.

### Corpus v2 checks

**6.1 Paired takes (Medium).**
- Evidence: the distance between two takes is single-clip noise. Class centroids built from n clips have noise that shrinks with 1/√n, so classes much closer than one take-to-take distance can still be separated **(inferred, standard statistics)**. The statement in the plan is wrong as written.
- Better uses:
  1. A **reliability test**: the rate at which both takes get the same top cluster, compared with a permutation baseline. This is a free, label-free score for choosing between candidate embeddings (3.3).
  2. Within-class noise variance, used for the effect size between classes.
- De-duplicate first: the `_m` and `_f` files can be byte-identical copies. `audio.sha256` is in the corpus (`corpus-format.md:31,48`) **(verified)**. Identical pairs count as one clip in clustering and bootstraps, or they inflate density.

**6.2 Recording-session check (Low).** Good idea. Normalise the effective number of conversations for cluster size (the expected value under random draws of that size), which the permutation baseline does if it is drawn at the same size. Say so explicitly. Also report a leave-one-conversation-out stability: does the cluster survive when its biggest conversation is removed?

### Sample sizes

**7.1 Item 1, the shortest usable clip (Medium-High).**
- Evidence: a 0.5 s crop from the middle of a long line isn't like a naturally short utterance, which has its own onset and offset prosody. Comparing a crop with the whole clip also penalises real changes of tone within a line. Raw cosine in mean-pooled self-supervised speech embeddings is high even between unrelated clips. Such representations are strongly anisotropic, as shown for text by Ethayarajh 2019 **(source for text; inferred for WavLM, checkable on the smoke set)**. So 0.9 may pass everything, or mean nothing.
- Fix:
  - Measure in the cleaned, centred space.
  - Use a retrieval criterion: for crops of length L, how often is the crop's nearest neighbour among full clips its own source clip (or that clip's paired take)? Compare that with longer crops.
  - Pick the threshold where retrieval reaches a set fraction (for example 80%) of the 2 s level.
  - Cross-check with short paired takes: do short `_m`/`_f` pairs agree as well as long ones?

**7.2 Item 2, minimum class size (Medium).**
- Evidence: the 0.9 split-half cosine has no source. Split-half cosine at total n measures vectors built from n/2 clips each, so the n the method reports is the size per half, not the class size. The cosine also depends on dimension: noise in d dimensions pulls it down roughly like 1/(1 + d·σ²/(n·‖δ‖²)).
- Fix:
  - Define the n as the class total, and apply a Spearman–Brown style step-up, or report both.
  - Compute it in the reduced space used for clustering, so d is small and known.
  - Write down why 0.9. One defensible anchor: the reliability at which the vector's direction error is below some angle, for example cos 0.9 ≈ 26°.
- The plan already says this is re-measured on model activations later. Keep that.

**7.3 Item 3, GMM parameters (verified, fine).** d(d+3)/2 gives 44, 152 and 560 at d = 8, 16 and 32. "5–10 clips per parameter" is a heuristic and is labelled as one. Fine.

**7.4 Item 4, bootstrap Monte Carlo error (verified, fine).** With a per-resample Jaccard SD of about 0.15, 200 resamples give an SE of about 0.011.

**7.5 Item 5, purity listening (Medium-High).**
- Wilson arithmetic verified: 8/10 → 0.49–0.94, 16/20 → 0.58–0.92, 20/25 → 0.61–0.91.
- Problems:
  1. **Optional stopping.** Checking the interval after every clip from 10 to 30 and stopping when it clears 0.6 raises the false-pass rate above the nominal 5% (Armitage, McPherson & Rowe 1969) **(source)**.
  2. **No foils.** The listener is told the cluster's name and hears only its clips, so a yes-bias inflates purity.
  3. The 0.6 purity bar and the 0.8 "true purity" in the worked example have no stated justification.
- Fix:
  - Use Wald's SPRT for p₀ = 0.5 against p₁ = 0.8 with α = β = 0.05, which needs about 10–20 clips on average, or a precomputed group-sequential table (looks at 10, 20 and 30 with Pocock or O'Brien–Fleming bounds).
  - Mix in about 1 foil per 4 clips, drawn from other clusters at similar length, as in the word-intrusion test of Chang et al. 2009, "Reading Tea Leaves" **(source)**.
  - Report purity corrected for the false-yes rate on foils.
  - Write down why 0.6, or choose by consequence: with soft memberships, a 60%-pure class makes a vector that is about 60% signal.
- Time: foils add about 25% listening, so the estimate becomes roughly 50–95 minutes, within Brandon's stated hour or a bit over. Say so up front.

**7.6 Item 6, rater consistency (verified, small fix).** The kappa intervals check out (SE ≈ √(p₀(1−p₀)/(n(1−pₑ)²)): ±0.31, ±0.20 and ±0.14). With foils from 7.5, also report his foil false-yes rate. That is a more useful number than test-retest alone.

**7.7 Item 7, steering tests (verified, out of scope).** Paired t-test sizes: d = 0.5 → 34, d = 0.8 → 15. Binomial 75% against 50% at 80% power needs about 23–26 trials. Fine as notes for later work.

### Step 6: naming

**8.1 Order of evidence (Low).** The primer says the Gemma text suggestion and prosody are shown *after* Brandon answers. The plan's step 6 doesn't say when. Put "after" in the plan too, and the same for EIV top heads. This avoids anchoring.

### Primer

**9.1 What happens when a check fails (Medium).** The primer lists acceptance numbers but not what follows a failure. Add a short table:
- The game probe stays above its bound → try re-colouring with EQ on, then another layer, then report it to Brandon. Never loosen the bound.
- No cluster passes stability → the continuous-axes question (intent section below).
- And so on for each check.

**9.2 Parts of the new plan aren't in the primer's build list (Low).** The primer's stage list has the plan's step-4 order, but not the paired-take de-duplication, the take-agreement check, the noise-floor capture (if adopted) or the conversation-level permutation scheme. Add the ones Brandon approves to stage 4 (`confounds`).

**9.3 Precision (Low).** Store float16, as the primer says, but compute LEACE, covariances and whitening in float32 or float64. Shrinkage covariance in float16 is unstable.

**9.4 Libraries (Low).** LEACE is available as EleutherAI's `concept-erasure` package, which uses PyTorch. A numpy implementation is about 30 lines, which avoids adding torch to the CPU nix-shell. HDBSCAN is in scikit-learn ≥ 1.3. Pin both.

---

## Drift check against the fixed design

| Fixed point | Plan | Primer | Note |
|---|---|---|---|
| Classes from clustering her own clips | ✓ | ✓ | Text-predictable rejection (5.3) and bark exclusion (5.5) could each quietly remove real classes |
| Per-class vectors for TTS and LLM from her clips and lines, mixed | ✓ (steps 7–9 as later notes) | ✓ out of scope | Fine |
| EIV backup only | ✓ | ✓ | Using EIV to name clusters and as a confound measure is fine. The trigger for falling back is unwritten (5.6) |
| No audio LLM as listener | ✓ | ✓ explicit | Gemma reads text only |
| No public or synthetic vector sources | ✓ | ✓ | Synthetic Qwen3-TTS pairs appear only as "comparison" in step 7, outside this scope |
| Hard-data metadata only | ✓ outputs kept out of `lines.jsonl` | ✓ | "Channel" has no hard-data source (4.7) |
| Corpus-only scope, no model choice | Mostly | ✓ | Step 7 still calls Qwen3-TTS 1.7B "the leading candidate". Brandon said the voice model is compared later. Suggest softening to "example" |
| Self-hosted, low budget, one GPU job with his ok | ✓ | ✓ | Fine |

No drift toward fixed axes, shortlists or public data.

## Would change intent: Brandon's call

These are listed so they're decided, not slipped in. None is recommended over the current design without his yes.

1. **The continuous-axes fallback.** If clustering finds no stable islands, the plan uses "GMM components or main directions of the cleaned space as continuous axes". Principal directions are not discovered *classes*. That is a different design, even though it is still built from her clips. Should it be allowed, and what triggers it (5.6)?
2. **Combat barks as possible class-formers** (5.5). Should they be allowed to form or join a class, or only be assigned after clustering?
3. **The recording-session flag.** A cluster drawn mostly from one or two conversations, like Rannoch, could be a real state tied to a scene or a mic session. The plan already routes it to him. Confirm that's what he wants, since it's where the "situation vs nuisance" line gets decided case by case.
4. **Dropping DNSMOS.** Brandon chose DNSMOS on 2026-10-13. The primer drops it in favour of EIV-Plus quality heads. I found no message where he approved dropping it. Confirm.

## Sources
- Belrose et al. 2023, LEACE: https://arxiv.org/abs/2306.03819
- Ravfogel et al. 2022, Adversarial concept erasure in kernel space: https://arxiv.org/abs/2201.12191
- Sun, Feng & Saenko 2016, CORAL: https://arxiv.org/abs/1511.05547
- Ledoit & Wolf 2004, shrinkage covariance; scikit-learn `LedoitWolf`
- Biernacki, Celeux & Govaert 2000, ICL (IEEE TPAMI 22(7))
- Baudry, Raftery, Celeux, Lo & Gottardo 2010, Combining mixture components for clustering (JCGS 19(2))
- Hennig 2010, Methods for merging Gaussian mixture components (ADAC 4); fpc `mergenormals`
- Hennig 2007, clusterboot; bands in fpc docs: https://search.r-project.org/CRAN/refmans/fpc/html/clusterboot.html
- Armitage, McPherson & Rowe 1969, Repeated significance tests on accumulating data (JRSS A 132)
- Wald 1945, Sequential tests of statistical hypotheses (SPRT)
- Chang et al. 2009, Reading tea leaves (word and topic intrusion)
- Scherer 2003, Vocal communication of emotion; Juslin & Laukka 2003 (speaking rate and arousal)
- Ethayarajh 2019, How contextual are contextualized word representations? (anisotropy)
- emotion2vec+ large card (9 categories): https://huggingface.co/emotion2vec/emotion2vec_plus_large
- Whisper 30 s input window: https://github.com/openai/whisper
