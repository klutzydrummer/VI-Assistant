# Primer: building Tali's emotion corpus (implementation hand-off)

For the thread that implements this. Written 2026-10-13 from the "step back" planning thread. **The spec is [corpus-pipeline.md](corpus-pipeline.md). Read it in full before writing code.** This primer covers what the spec doesn't: the rules, what already exists in the repo, the build order, and the mistakes not to repeat.

## 1. The goal in one paragraph
Tali's emotion and state classes are **discovered from her own voice clips**: embed every clip, remove recording confounds (game, text length, recording noise floor), and cluster what's left. Brandon then names and confirms each stable cluster by ear. The output is a corpus where every clip has soft memberships across those classes. Later work, **not this thread**, turns each class into a steering vector for both the voice model and the chat model, then mixes and interpolates them at runtime. The six axes in `characters/tali/states.toml` (joy, sadness, anger, worry, playfulness, intoxication) were placeholders. The classes found here replace them.

## 2. Scope
In scope:
- audio prep
- filters
- embeddings
- Empathic-Insight-Voice (EIV) backup scores on every clip
- confound measurement and removal
- clustering
- stability
- a purity review mode for Brandon
- the final class table and per-clip memberships

Out of scope:
- choosing, comparing or hooking a voice (TTS) or chat model
- building steering vectors
- mixing
- the `steer` stage

## 3. Rules (non-negotiable; each one was learned the hard way)
1. **No GPU job, rental or paid run without Brandon's explicit ok for that specific run**, after he has seen exactly what changed and the exact command. "I'm home" is not an ok. The whole of this scope needs one GPU job (`embed`).
2. **Never run on a stack already decided to change.** Build what's agreed first.
3. **Reviewer or subagent suggestions are proposals, never decisions.** Changes to classes-from-clusters, the vector sources (her own clips and lines), the fallback, or sample sizes need Brandon's yes. The last planning thread was shut down for drifting from this.
4. **Game audio and subtitles never leave home except by direct rsync to the rented worker.** No Hugging Face uploads, nothing in the repo, nothing in messages.
5. **GPU jobs checkpoint and sync home every minute** (`vi_job.Checkpoint`, `jobs/on-worker.sh`) so a lost worker or empty credit resumes.
6. **CI has no GPU.** Every job needs a `backend = "fake"` path covered by tests (see `crates/vi/tests/label.rs` and `jobs/test_*.py`). CI runs `cargo fmt`, `clippy -D warnings` and `cargo test`.
7. **Brandon's machine:** 4 cores, a slow game disk (sda), and a system disk (sdb) that is about 88% busy, with /home nearly full (21 GB free on 2026-10-13). Heavy local work runs with `nice -n 19`. Reads from the game disk use `ionice -c3`. Anything touching sdb uses `ionice -c2 -n7`, because idle class starves there. Keep writes to /home small: store embeddings as float16.
8. **Change the corpus spec first, then the code.** That spec is `docs/design/corpus-format.md`. New fields stay optional and additive.
9. **Keep Brandon's status checklist current** as steps change. He follows progress there.

Note: the old `clean → discover → label` rerun the harvest thread mentioned is replaced by this pipeline. Don't rerun `discover`/`label`.

## 4. What exists (main @ 604412c) and what to do with it
| Piece | Path | Verdict |
|---|---|---|
| Acceptance checks + confound reports (`checks.json`, eta² by game, Spearman with length) | `jobs/checks.py` (merged in #21) | **Reuse** in `confounds` and `cluster` |
| Old-plan docs merged with #19/#21: `docs/design/methodology-review.md`, `independent-review.md`, `control-vector-datasets.md` | repo | **Mark superseded** in the spec PR, pointing to this plan. They contain the drift (fixed axes, public datasets, shortlists) |
| Stage runner, content-addressed caching | `crates/dataset`, `vi dataset run` | **Reuse as is** |
| GPU worker, rsync, checkpointing, round-trip check | `crates/cloud`, `jobs/on-worker.sh`, `jobs/vi_job.py`, `jobs/check-worker.sh` | **Reuse as is** |
| Harvest (re-extracted 2026-10-14: 3,516 lines, 202 min, LE1–LE3; `vi corpus check` clean; PRs #23–#26 merged, `following` + structural tags in commit 6cc714e on `claude/le1-barks-singing` still to merge) | `crates/harvest` (scene map in `crates/harvest/src/scenes.toml`), corpus at `corpus/tali/harvest` on Brandon's machine | **Reuse.** How each new field is used is in the spec's "Corpus v2" section. Every line now has `scene_tags`, plus structural tags (`one_liner`/`exchange`/`dialogue_choices`, `to_shepard`/`to_squad`/`to_npc`), a `following` field, and 129 LE1 combat barks (`bark` + `no_subtitle`, tag `combat`). `le3.s769911_m` is flagged `singing`: exclude it from clustering. **Metadata is hard data only (Brandon's rule): no LLM-inferred situation tags, and empty means the game doesn't say.** Scene and structural tags are required inputs: they feed the confound probes (step 4), help with naming clusters (step 6), and show in the purity review. Barks without subtitles skip the Whisper check and the text-content probe |
| clean: Whisper large-v3 subtitle check, ECAPA per-game speaker check, DNSMOS | `jobs/clean.py` | **Keep Whisper and ECAPA as flag-only filters. Keep DNSMOS** (Brandon, 2026-10-14) as a reported covariate, never a gate. The EIV-Plus quality heads are also recorded |
| discover: emotion2vec+, MiniLM, prosody, UMAP/HDBSCAN, Gemma captions | `jobs/discover.py` | **Replace.** No Gemma listening anywhere. Embedding moves to the new `embed` stage. Clustering moves home to a CPU stage. Prosody and MiniLM code can be reused |
| label: fixed-axis fusion, Gemma judge, Gemma audio ratings | `jobs/label.py`, `jobs/fusion.py` | **Not in this pipeline.** Leave the files and remove the stages from the Tali pipeline. Gemma *text* naming help for clusters is the only Gemma use |
| calibrate (fixed axes) | `jobs/calibrate.py` | **Not used** here. Its grouped-fold and probe helpers may be reusable |
| steer | `jobs/steer.py` | **Out of scope.** Leave it, `manual = true` |
| Review page | `crates/review`, `vi review` | **Extend** with a cluster-purity mode (below) |

## 5. Proposed stages (adjust names to fit the runner)
1. **`prep`** (home, CPU). Store the trimmed-silence noise-floor spectrum and level, then VAD trim, then one linear gain (two-pass `loudnorm linear=true` or pyloudnorm, RMS under ~1 s; never dynamic compression), then a 16 kHz copy for embedding (keep the 24 kHz originals). Optional per-game EQ match fitted on the comparable subset, behind a setting.
2. **`clean`** (GPU, existing). Flags only. Whisper skips `no_subtitle`; ECAPA threshold checked per length band; DNSMOS kept, reported only.
3. **`embed`** (GPU, **the one paid job**). Per clip:
   - WavLM-large, layers 6–18, mean-pooled (float16)
   - emotion2vec+ large
   - emotion2vec base
   - BUD-E-Whisper encoder, pooled **only over real frames** (first ceil(duration×50) of 1,500), with a CI test that padding doesn't change the vector
   - EIV scores: all 54+ Small heads plus the Plus quality heads (skip content enjoyment)
   - audeering A/V/D

   Prosody and MiniLM go here or at home, whichever is simpler. Checkpointed. Revisions are pinned in the output.
4. **`confounds`** (home, CPU, numpy + scikit-learn in nix-shell; float32/64 math). Follow spec step 4 exactly:
   - de-duplicate byte-identical paired takes
   - before-table
   - PCA to 64–128 dimensions on the comparable subset
   - Ledoit–Wolf CORAL re-colouring of each game to the pooled covariance (not whitening)
   - LEACE for text length and noise floor
   - non-linear game probes with an equivalence bound and conversation-level permutations
   - paired-take agreement
   - candidate chosen on 30% of conversations, reported on 70%

   Reuse `jobs/checks.py`.
5. **`cluster`** (home, CPU).
   - parallel-analysis PCA
   - GMM with BIC + ICL
   - component merging (Baudry entropy or Hennig DEMP)
   - HDBSCAN second opinion
   - conversation-level 200-resample Jaccard stability (keep ≥ 0.75; below 0.6 not trusted)
   - nuisance-only confound audit, plus enrichment and "topic, not tone?" flags
   - session flag for Brandon
   - split-half class size
   - responsibilities *and* continuous per-class scores
   - one sensitivity run with and without LE1 barks
6. **`review`, purity mode** (Brandon).
   - Clips stratified across scenes and conversations, with 1 foil per 4 from other clusters.
   - Wald SPRT stopping (p₀ 0.5, p₁ 0.8, α = β = 0.05).
   - Hidden repeats about 1 in 8, targeting 50.
   - Context, following lines, tags, prosody, EIV top heads and the Gemma text suggestion are shown only *after* he answers.
   - Name, merge, split or reject.
7. **`classes`** (home). Final output:
   - `classes.json`: names, sizes and stability
   - `memberships.jsonl`: id, responsibility and continuous score per class
   - an EIV-only backup labelling
   - a report with every acceptance number and every check that failed

Use `characters/tali/smoke.toml` (40 clips) for end-to-end tests on Brandon's machine before the full run.

## 6. Acceptance numbers and failure handling
Both are in the spec (step 4, step 5, and "When a check fails"). Key points:
- Game is not recoverable by non-linear probes (balanced-accuracy upper bound within 5 points of chance).
- The text-length and duration effects are gone.
- The A/V/D guard holds.
- Cluster Jaccard is ≥ 0.75.
- Purity ≥ 0.6 by SPRT, corrected for foils.
- **If fewer than 2 clusters are stable or HDBSCAN noise is over 60%: stop and report to Brandon.** No automatic fallback to continuous axes or EIV labels.
- A bound is never loosened to pass.

## 7. Suggested build order (small PRs, each green in CI)
1. Update `docs/design/corpus-format.md` and the repo copy of the plan, the spec PR.
2. `prep` + `clean` changes (DNSMOS kept).
3. `embed` (fake backend in CI). Then stop: show Brandon the exact command and ask for the run.
4. `confounds` + `cluster`, tested on the smoke corpus.
5. Review purity mode.
6. `classes` + report.

Steps 4 and 5 can be built while waiting on the embed run.

## 8. Don'ts (what went wrong last time)
- Don't replace clustering with fixed axes, shortlists or "30–40 confirmed clips per axis".
- Don't build vectors from public emotional-speech datasets or generated speech. Her clips are the source.
- Don't use an audio LLM (Gemma or another) as a listener or labeller.
- Don't reuse sample counts from the old thread. Every count is in the spec's "Sample sizes, derived" section.
- Don't present something as "agreed" or "settled" unless Brandon said yes to it.
- Don't suggest running before the agreed changes are built.

## 9. Open with Brandon
- Where the build happens: this cloud thread opening draft PRs, or a session on his machine (which has the real corpus). Ask him.
- Whether to delete `label`, `calibrate` and `discover` from the repo or just remove them from the Tali pipeline. The default here is to remove them from the pipeline only.
