# Steering vector builder: design

Status: design draft 1, 2026-10-20, for critical review. Built on `steering-builder-research.md`.
Nothing here is implemented. Implementation starts only after the review and Brandon's go.

## Goal

Build the dialogue model's control vectors from **every** pair in step 4's 80 chat sets (77 moods plus
the energy, intoxication and hungover knobs; 8,986 pairs) with memory that doesn't grow with set size,
output the control-vector GGUF that llama.cpp and `crates/llm` already load, and make rebuilding for a
new model (Qwen 4, a fine-tune, a modified model) one command.

What stays as Brandon decided: vectors come from her own lines; lines are multi-label and vectors are
fitted jointly; the chat model writes plain speech and vectors are mixed per turn through
`llama_set_adapter_cvec`; vectors are stored by model hash; start at full strength and lower only on
evidence; no rental without his go.

## Shape: capture once, fit many times

Two stages with a small file between them.

1. **Capture** (needs the model; runs on the rental, or on CPU for the smoke). Decode each *unique*
   rendered prompt once, with the persona prefix decoded once and reused. While her line is decoded, an
   eval callback reads each layer's output and keeps per-line summaries. Writes a **summary store**.
2. **Fit** (CPU only, no model, seconds to minutes). Reads the store plus the sets and labels and writes
   the vectors and a report. Can be rerun at home for free with a different method, labels, sets or layer
   band, without renting again.

This is what removes the memory problem: capture holds one layer of one batch at a time; fit holds
running sums.

## Stage 1: capture

**Input.** The prompts `render_cvector.py --gguf` produces, but as records rather than paired text
files, one per unique line:
`{line_id, prefix, context, line, suffix}` where `prefix` is the persona system turn (identical across
records), `context` is the user turn she answers plus the assistant header, `line` is her line, and
`suffix` is the end-of-turn token(s). The renderer already knows these parts; it emits them separately
so the capture never has to search for her line inside a string. Records are keyed by `line_id` and a
hash of the rendered text, so two sets that use the same line share one record.

**Decoding.**
- Tokenize `prefix`, decode it once into sequence 0.
- For each record: `llama_memory_seq_cp(0 -> 1)`, decode `context + line + suffix` in sequence 1 as one
  batch (`n_batch = n_ubatch` large enough for the longest record, so every layer's output arrives as one
  tensor in token order), then `llama_memory_seq_rm(1)`.
- Fallback if the reuse test (below) fails on the hybrid model: snapshot sequence 0 with
  `llama_state_seq_get_data` and restore it per record. Last resort: decode the full prompt per record,
  which still costs ~5x less than cvector-generator because of deduplication.

**Callback.** `cb_eval` asks for tensors whose name is `<layer_tensor>-<il>`; `layer_tensor` is a
per-architecture setting, default `l_out` (`qwen35`, `llama`, `qwen3`). It refuses, with a clear error,
any tensor whose first dimension isn't `n_embd` or whose token count isn't the batch's, so an
architecture like `qwen4exp` (hyper-connections, `l_last`, no `build_cvec`) fails loudly. Layers
1..n_layer-1 are captured (llama.cpp can't steer layer 0, and the last layer is cropped).

**Summaries per record and layer** (F32 accumulation, stored as F16):
- `line_mean`: mean of the layer output over her line's tokens (default fitting input).
- `pre_line`: the output at the last context token, the state just before she starts speaking.
- `norm`: mean L2 norm of the layer output over her line's tokens, so the fit can report each vector's
  size relative to the residual stream at that layer.

**Summary store.** One directory per model:
`vectors/<character>/llm/<model-hash>/capture/` with `summaries.f16` (records x layers x 2 x n_embd,
memory-mapped), `records.jsonl` (line_id, text hash, token counts), and `capture.json` (model file
sha256, llama.cpp version, architecture, n_embd, n_layer, layer tensor name, renderer version, chat
template hash). Size for Qwen3.8-27B: about 4.5 GB for at most 3,516 lines; 2.3 GB if `pre_line` is
dropped. Written in record order with a checkpoint, so a lost worker resumes where it stopped. Game text
never goes in the store, only ids, hashes and numbers; it stays on the worker and Brandon's PC like the
rest of the derived data.

## Stage 2: fit

**Inputs.** The store, the 80 sets as lists of `(positive line_id, zero line_id)` pairs, and the label
table from the Claude labelling run: per line, its moods (multi-label), its knob levels, game and length.

**Methods** (all from running sums over the store; one pass per method):
- `joint` (**default**, matches "fitted jointly"): per layer, ridge least squares
  `h = μ + Σ_k y_k β_k + Σ_j c_j γ_j`, over every line that appears in any set. Columns: one per mood
  (0/1), one per knob (its level), and nuisance covariates (game one-hot, log line length) standing in
  for the matching the sets were built with. `β_k` at layer *L* is mood *k*'s vector there. λ is picked
  by 5-fold cross-validation grouped by conversation, on held-out prediction of `h` projected onto the
  `β` directions. Sufficient statistics: `CᵀC` and `CᵀH` per layer, about 260 MB in F64.
- `mean_diff`: per set and layer, mean over its positives minus mean over its matched zeros, each line
  counted once per set. The direct, unconfounded-by-construction comparison; also what cvector-generator
  meant to compute. About 2 x 80 x 63 x 5,120 F64 sums, 400 MB.
- `pca_center`: repeng's method with sign fixed so positives project higher, from a streamed
  `d x d` covariance per layer for one set at a time. Included only as a comparison, not proposed.

`joint` and `mean_diff` are both written; the chat check compares them on a few moods and blends before
one becomes the default for the rest.

**Assumption to review:** a mood absent from a line's labels counts as 0 for that line in `joint`. The
Claude run was open-ended multi-label, so absent should mean "not that mood"; if the critic thinks
otherwise, `joint` can weight each column only over lines that are trusted positives or trusted zeros
for it, at the cost of one regression per column (still running sums).

**Per vector and layer, the report gives:**
- held-out AUC of projecting onto the vector, positives against zeros, folds grouped by conversation;
- split-half cosine (two random halves of conversations; the same as voice check B);
- mean pairwise cosine of the set's per-pair differences (predicts steerability, arXiv 2505.22637);
- vector norm divided by the mean residual norm at that layer;
- cosine between moods (which vectors overlap, for blending).

**Recommended layers.** Per vector, the contiguous band in the middle 20-80% of depth whose held-out AUC
is within 0.02 of its best and whose split-half cosine is at least 0.8. Written to the vector folder's
`vectors.toml` as `layers = [a, b]`, which `crates/llm` already honours. A mood whose best AUC is near
0.5 or whose halves disagree is flagged in the report before any rental time is spent on it.

## Output

For each vector, a control-vector GGUF exactly like cvector-generator's (`general.architecture =
controlvector`, `controlvector.model_hint`, `controlvector.layer_count`, F32 `direction.<il>` per layer),
so `llama-server --control-vector-scaled` and `crates/llm::Vector::from_gguf` load it unchanged, with two
differences:
- `direction.il` is the vector measured at layer *il*'s output (cvector-generator writes it one layer
  late);
- vectors keep their natural size (not normalised), plus extra keys: method, model sha256, store hash,
  λ, recommended band.

Written to `vectors/<character>/llm/<model-hash>/<method>/<mood>.gguf` with `vectors.toml`, matching the
step 5 plan's layout, so `vi chat` keeps refusing vectors built for another model.

## What "full strength" means (needs Brandon's eye)

With natural-size vectors, strength 1.0 at one layer means "move her by the average gap between her mood
lines and her plain lines", which is the obvious meaning of unaltered. Applied across a band of *K*
layers, the same vector at every layer compounds, so full strength over a band is each layer at `1/K`
(the `vectors.toml` `scale`). The builder writes `scale = 1/K` for the recommended band. The chat check
starts there, per Brandon's rule, and lowers only if output degrades; it also tries the single best layer
at 1.0 and `1/√K` on a few moods, and the better scheme becomes the rule for the rest. This replaces the
unit-length vectors the step 5 plan assumed, whose "full strength" had no fixed meaning.

## Where it lives

- Rust, in `crates/llm` (`capture.rs`, `fit.rs`), driven by a `vi vectors` command:
  `vi vectors capture`, `vi vectors fit`, and `vi vectors build <model-id>` which runs both for a model
  from the registry. The rebuild-for-a-new-model command in the step 5 plan becomes `vi vectors build`
  followed by the chat check.
- No new heavy dependencies: sums and a ridge solve on a ~100 x 100 matrix (a small Cholesky in Rust, or
  `nalgebra`).
- Renderer change (`render_cvector.py`, on Brandon's PC): also emit the record format above. The paired
  text files stay, for comparing against cvector-generator in the smoke.

**Getting a CUDA build to the worker** (the worker image has no compiler; nothing compiles there).
Recommended: a GitHub Actions job, separate from CI, that builds `vi` with `--features cuda` in an
`nvidia/cuda:12.8.1-devel` container (no GPU needed to compile) and publishes the binary as a workflow
artifact; the worker downloads it. Alternative: build in the same container on Brandon's PC (his disk is
tight). Either way the binary carries the exact llama.cpp version `vi chat` runs.

## Tests

In CI (no GPU, no download), on `testmodel.rs`'s tiny random llama:
- prefix reuse gives the same per-line summaries as a full decode, within F32 tolerance;
- layer indexing: plant a known vector with `llama_set_adapter_cvec` at layer *k*, capture, and check the
  shift appears at layer *k*'s output and not *k-1*'s;
- fit recovers a planted direction: synthetic summaries with known `β` per mood, including two
  correlated moods, where `joint` separates them and `mean_diff` mixes them;
- `mean_diff` equals a brute-force mean on small data; F16 store round trip;
- output loads with `Vector::from_gguf` and `llama_set_adapter_cvec`.

At home, CPU, Qwen3.5-0.8B (same `qwen35` hybrid design), as part of step 5's one smoke:
- the reuse test on the hybrid model (recurrent state copy), plus the fallback if it fails;
- two real sets: capture, fit, measure peak RAM and tokens per second, and compare the `mean_diff`
  direction with cvector-generator's `mean` output on the same pairs (expect high cosine after fixing
  its one-layer shift; padding and length weighting explain the rest);
- one steered generation per method, to see that it loads and steers at all.

## Cost and memory (estimates; the smoke measures them)

| | cvector-generator, all pairs | this design |
|---|---|---|
| tokens decoded | ~16-23 M | ~0.35 M (plus one persona) |
| peak host RAM | ~51 GB for the largest set | model + under 1 GB |
| time on an L40S | ~2-3 h | well under 1 h, mostly model load |
| refit with new labels or method | another rental | free, at home, from the store |

That should take roughly 2 hours and about $1-1.50 off the step 5 rental, and remove the 64-128 GB RAM
filter on offers.

## Out of scope

- Voice vectors: the suit thread keeps them (`steer` on Qwen3-TTS). The capture could serve a TTS model
  that runs in llama.cpp (it now has `qwen3tts`), but only if Brandon asks.
- Run-time changes beyond honouring `layers` and `scale`, which `crates/llm` already does.
- Synthetic expansion of thin moods.

## Questions for the critical review

1. Is `joint` with "absent means 0" sound for these labels, and are game and length the right nuisance
   columns given the sets were matched on game, energy and length?
2. `line_mean` versus `pre_line`: is a teacher-forced mean over her line the right training signal for
   steering generation?
3. Is the band rule (AUC within 0.02, split-half ≥ 0.8, middle 20-80%) a sensible pre-registered default,
   and is `1/K` the right starting scale for "full strength"?
4. Does `llama_memory_seq_cp` on `qwen35`'s recurrent state give exact reuse, and is the fallback order
   right?
5. Anything in the store or the CUDA build path that breaks the "verify everything before renting" rule.
