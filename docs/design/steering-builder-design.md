# Steering vector builder: design

Status: design draft 2, 2026-10-20. Built on `steering-builder-research.md`; draft 1 was reviewed in
`steering-builder-critique.md` (thread "Critical review of steering builder") and every finding is
addressed here (see "Review findings and where they went" at the end). Nothing is implemented.
Implementation starts only after Brandon's go.

## Goal

Build the dialogue model's control vectors from **all** of Tali's scored lines (3,516, every one scored
0-3 on every mood family by the Claude run) and step 4's 80 chat sets (77 moods plus the energy,
intoxication and hungover knobs; 8,986 pairs), with memory that doesn't grow with the data, output the
control-vector GGUF that `crates/llm` loads, and make rebuilding for a new model (Qwen 4, a fine-tune, a
modified model) one command.

What stays as Brandon decided: vectors come from her own lines; lines are multi-label and vectors are
fitted jointly; the chat model writes plain speech and vectors are mixed per turn through
`llama_set_adapter_cvec`; vectors are stored by model hash; thin and flagged moods stay in and start at
full strength, lowered only on evidence; no rental without his go; verify everything before renting.

## Shape: capture once, fit many times

1. **Capture** (needs the model; on the rental, or on CPU for the smoke). Decode each of her lines once,
   in the exact prompt `vi chat` would build, with the persona decoded once and reused. An eval
   callback reads each layer's output and keeps per-line summaries in a **summary store**.
2. **Fit** (CPU only, no model). Reads the store and the label table, writes vectors and a report.
   Reruns at home for free with other labels, methods or layers.
3. **Check** (needs the model). The step 5 chat check, run through `crates/llm`'s own steering path so
   the strengths it settles mean the same thing in `vi chat`.

Capture holds one layer of one line at a time; fit holds running sums. Peak host RAM is the model plus
under 1 GB.

## Stage 1: capture

**Prompts, built the way `vi chat` builds them.** The capture renders each prompt in Rust with the same
`format_chat` (`llama_chat_apply_template`) that `vi chat` uses. The Python renderer only supplies
messages per line: `{line_id, system, user, assistant}` (persona, the line she answers, her line); the
Rust side owns the template. In the engine's llama.cpp that template is plain ChatML ending in
`<|im_start|>assistant\n`, with no think block (`src/llama-chat.cpp:99,256` in `llama-cpp-sys-2
0.1.159`), not the GGUF's jinja template that `render_cvector.py --gguf` used. Whether Qwen3.8 needs an
empty think block at run time is step 5's call; the capture follows whatever `format_chat` does, so it
changes together with `vi chat`.
- The whole prompt is tokenized once. Her line's token span comes from piece offsets.
- The persona prefix must tokenize identically on its own (prefix reuse depends on it); the capture
  refuses to run otherwise.
- A smoke check confirms that the capture prompt and `vi chat`'s prompt for the same exchange are
  byte-equal.

**Decoding.**
- Context: `n_seq_max = 2`, `kv_unified = true` (otherwise `n_ctx` is split per sequence and every
  `seq_cp` copies whole buffers), `n_batch = n_ubatch` large enough for the longest line, so each layer's
  output arrives as one tensor in token order.
- Decode the persona once into sequence 0. Per line: `llama_memory_seq_cp(0 -> 1)`, decode the rest of
  the prompt in sequence 1, `llama_memory_seq_rm(1)`. On `qwen35`, `seq_cp` shares the recurrent tail
  cell and `find_slot` makes a private copy when sequence 1 decodes, so the persona state is reused
  (checked in the vendored source by the review).
- Fallbacks if the reuse test fails: restore a `llama_state_seq_get_data` snapshot per line; last
  resort, decode each full prompt (still ~5x cheaper than cvector-generator thanks to deduplication).

**Callback and layer checks.** `cb_eval` asks for `<layer_tensor>-<il>`, with `layer_tensor` a
per-architecture setting (default `l_out`). It fails the run unless:
- every captured tensor is 2-D (`ggml_n_dims == 2`) with `ne[0] == n_embd` and `ne[1]` equal to the
  batch's token count (catches `qwen4exp`'s 3-D `l_last`);
- exactly one tensor arrives per expected layer per decode, layers `1..n_layer-2` (catches `l_last`
  matching nothing, and `bitnet`/`lfm2`, which emit `l_out` twice). Layer 0 can't be steered by
  llama.cpp, and layer `n_layer-1` is cropped to output rows, so neither is captured.

**Summaries per line and layer** (F32 accumulation, stored as F16):
- `line_mean`: mean over her line's tokens **plus** the last prompt token. Those are exactly the
  positions `vi chat` steers (it decodes the prompt unsteered, then steers the last prompt token and
  every generated token).
- `line_head`: the same over the last prompt token and her first 8 tokens, where a reply's tone is set.
  Cheap to keep now; it would otherwise need another rental.
- per line, not per layer: her line's mean token log-probability and the token count, from logits
  requested only for her line's tokens.
- per layer: running mean of the residual norm over those positions, so the report can give each
  vector's size against the stream.

**Preflight** (`vi vectors preflight <model-id>`), run on every real model before its capture: at home
on Qwen3.5-0.8B, and in the first minutes of the rental on the 27B, while the full download finishes if
it can.
1. **Layer test:** plant a known vector with `llama_set_adapter_cvec` at layer *k*, capture, and check
   that the shift appears at layer *k*'s `l_out` and not at *k-1*'s. Repeat for a few *k*.
2. **Reuse test:** a handful of lines with and without prefix reuse; per line, cosine ≥ 0.9999 and
   relative error ≤ 1e-3 at every layer (DeltaNet's chunking makes results close, not bit-exact).
3. **GPU against CPU** (rental only): the tiny test model captured on GPU matches its CPU capture.
Any failure stops the run before the main capture.

**Summary store.** `vectors/<character>/llm/<model-hash>/capture/`: `line_mean.f16` and
`line_head.f16` (lines x layers x n_embd, memory-mapped), `lines.jsonl` (line_id, text hash, token
counts, log-probability), `capture.json` (model sha256, llama.cpp version, architecture, n_embd,
n_layer, layers captured, layer tensor name, chat template hash, preflight results). For Qwen3.8-27B:
3,516 lines x 62 layers x 5,120 x 2 bytes ≈ 2.2 GB per summary, **≈ 4.5 GB in all**. Written in line
order with a checkpoint, so a lost worker resumes. No game text goes in the store, only ids, hashes and
numbers; it stays on the worker and Brandon's PC like the rest of the derived data.

**Where the store lands at home.** `/home` is nearly full, so the store goes to a disk with at least
6 GB free, chosen at the home smoke the same way the fx_detect weights were (`--store <dir>`). Fit reads
it once per pass, sequentially, at nice 19 (ionice idle if it's on the game disk).

## Stage 2: fit

**Inputs.** The store; the label table from the Claude run (every line's pass-2 score 0-3 on every
family, pass-1 agreement, knob levels); per-line game, conversation, length and measured audio energy;
the 80 sets as `(positive, zero)` pairs (for evaluation and `mean_diff`).

**`joint` (default; this is "fitted jointly").** Per layer, least squares over **all 3,516 lines**:
`h = μ + Σ_k s_k β_k + Σ_j c_j γ_j`, where `s_k` is the line's score for mood *k* divided by 3, the
knob columns are their levels, and the nuisance columns `c_j` are game (one level dropped as reference),
log line length, and measured audio energy (the sets were matched on it; without it mood vectors could
absorb energy). Each `β_k` is mood *k*'s shift with the other moods and the nuisances held fixed, so
blends add up. It needs `CᵀC` and `CᵀH` per layer, about 260 MB in F64, from one pass over the store.
- Regularisation is a **small fixed ridge on the mood columns only**, for numerical conditioning
  (λ = 1e-4 x the mean diagonal of `CᵀC`), with the intercept and nuisance columns unpenalised. It is
  not tuned: tuning toward 0 would lower strength before any evidence, thin moods most. The report
  shows each joint vector's norm against its `mean_diff` norm, so any shrinkage that remains is visible.
  If a column is still ill-conditioned (a very thin mood almost always co-occurring with another), the
  report flags it, and that mood's penalty shrinks toward its own `mean_diff` instead of toward 0.
- Score/3 treats a score of 2 as two thirds of a 3. That's an assumption; the report also fits the
  masked variant (scores 1-2 and pass disagreements left out per column) for comparison.

**`mean_diff` (comparison, also built).** Per set and layer, the mean of the set's pair differences
(positive minus its matched zero), so a zero matched to three positives counts three times and the
sets' game, energy and length matching is kept. Built for every set, since it is what step 4's sets were
made for and the simplest thing to trust.

**What the two mean, for the mixing table.** A joint β is "this mood, the others held fixed". A
`mean_diff` vector is "sound like her lines of this mood", including the moods that tend to come with
it (her gratitude usually comes with warmth). Joint vectors are meant to be added; `mean_diff` vectors
already carry their companions and double-count when added. Joint is the default; switching any mood to
`mean_diff` needs Brandon's yes, and the state-machine owner is told which is in use.

**Report, per vector and layer** (flags only; nothing is dropped on them):
- held-out AUC on the trusted sets, folds grouped by conversation, plus the number of conversations each
  mood's positives come from (some sit mostly in one, e.g. jealousy 75%, joy 51% at the party, where
  grouped AUC means little);
- a within-conversation diagnostic: the joint fit refitted with conversation fixed effects (~300 more
  columns, still a small solve), and the cosine between the two, to show how much of a vector is topic;
- split-half cosine (as voice check B; it depends on sample size, so it's a flag, not a gate);
- mean cosine among a set's pair differences (predicts steerability, arXiv 2505.22637);
- vector norm against the residual norm at that layer, and the joint norm against `mean_diff`;
- cosines between moods.

Thin and flagged moods are still built and still start at full strength (Brandon, PR #62).

## What "full strength" means (Brandon's call)

Vectors keep their natural size. A vector measured at layer *L* is the whole gap between her mood lines
and her plain lines at that point, already accumulated by the layers before. So:
- **Full strength = the vector's best single layer at 1.0.** That moves her by exactly the measured gap,
  once. This is the anchor the chat check starts from.
- **Band option, telescoping:** add the gap at the band's first layer, then at each later layer only
  the *extra* gap that layer adds (`d_L - d_(L-1)`). The stream then carries each layer's own measured
  gap, still at full strength. It fits the existing GGUF format (the builder writes the increments as
  that vector's `direction.L`) and needs nothing new from the store.
- `1/K` and `1/√K` across a band are only comparisons. `1/K` is below full strength over most of the
  band, and `1/√K` is above `1/K`, so neither counts as "unaltered".
Which layer is "best", and whether the band beats the single layer, is settled by the chat check on a
few moods and one two-mood blend, then used for the rest. If no layer stands out, the default is the
middle of the model's depth (layer 32 of 64), flagged.

## Stage 3: the chat check, on `vi chat`'s steering path

llama-server sets one control vector at start-up for every token, prompt included, with one layer range
for all vectors, and ignores `vectors.toml` (`common/common.cpp:1453`). `vi chat` steers only the last
prompt token and the generated tokens, with each vector's own `layers` and `scale`. Strengths tuned in
llama-server would mean something else in `vi chat`. So the step 5 chat check runs as
`vi vectors check <model-id>`, reusing `Llama::generate` and `VectorBank::blend`. Blends change per turn
without restarting the model, which also saves rental time. The judge (the same model unsteered) runs in
the same process. The check itself is step 5's (the suit thread owns the prompts, the judge and the
pass rule); this builder supplies the command and the steering path.

## Output

For each vector and method, a control-vector GGUF like cvector-generator's (`general.architecture =
controlvector`, `controlvector.model_hint`, `controlvector.layer_count`, F32 `direction.<il>`), loadable
by `crates/llm::Vector::from_gguf` unchanged, except that:
- `direction.il` is the vector measured at layer *il*'s output (cvector-generator writes it one layer
  late);
- vectors keep their natural size, with extra keys: method, model sha256, store hash, λ, layer choice.

Written to `vectors/<character>/llm/<model-hash>/<method>/<mood>.gguf`, plus `vectors.toml` with
`layers` and `scale` per vector. After the chat check settles the layer scheme, the builder rewrites
`vectors.toml` (single layer) or the band GGUFs (telescoping increments). `vi chat` keeps refusing
vectors whose model hash doesn't match.

## Commands and code

- Rust, in `crates/llm` (`capture.rs`, `fit.rs`, `check.rs`), behind `vi vectors preflight | capture |
  fit | check`, and `vi vectors build <model-id>` which runs preflight, capture and fit for a model in the
  registry. Rebuilding for a new model: add the registry entry, rent, run `vi vectors build` then
  `vi vectors check`.
- No heavy dependencies: running sums and a ~100-column (or ~400 with conversation effects) Cholesky
  solve, written directly or with `nalgebra`.
- `render_cvector.py` (on Brandon's PC) gains a messages-per-line output; the template moves to Rust.

## CUDA build for the worker

The worker image has no compiler and nothing compiles there. A GitHub Actions workflow, separate from
CI, builds `vi --features cuda`:
- in `nvidia/cuda:12.8.1-devel-ubuntu22.04` (the worker image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
  is Ubuntu 22.04, glibc 2.35, which the job checks; an Ubuntu 24.04 build may not start there);
- with `CUDAARCHS` set to the rental GPUs only (`86-real;89-real` for A6000/L40S-class), offers filtered
  to match; this keeps the build within a runner's time and disk;
- `build.rs` links `cudart_static` and `cublas_static`, so only the driver's `libcuda.so` is needed on
  the worker;
- then, in the same workflow, the artifact runs **inside the worker image** on `testmodel.rs`'s tiny
  model on CPU (capture and preflight), proving it starts, links and captures. If ggml-cuda doesn't fall
  back cleanly without a GPU, this job shows it, and the fix lands before any rental;
- on the rental, preflight step 3 (GPU against CPU on the tiny model) runs first.

## Tests

In CI (no GPU, no download), on `testmodel.rs`'s tiny random llama:
- prefix reuse matches a full decode within the preflight tolerance;
- the planted-vector layer test;
- the callback refuses 3-D tensors, wrong counts and missing layers;
- fit recovers planted directions on synthetic summaries, including two correlated moods, where `joint`
  separates them and per-set means mix them; nuisance columns remove a planted game offset;
- `mean_diff` matches a brute-force mean of pair differences, including a zero used in several pairs;
- F16 store round trip; output loads with `Vector::from_gguf` and `llama_set_adapter_cvec`;
- the capture prompt equals `format_chat`'s for the same messages.

At home, CPU, Qwen3.5-0.8B (same `qwen35` hybrid design), as part of step 5's one smoke:
- `vi vectors preflight` (layer and reuse tests on the hybrid model);
- the byte-equal prompt check against `vi chat`;
- two real sets end to end: capture, fit, measure peak RAM and tokens per second; compare `mean_diff`
  with cvector-generator's `mean` on the same pairs (high cosine expected after its layer shift);
- `vi vectors check` on one mood and one two-mood blend, single layer and telescoping band.

The fit runs at home after the rental, so a fit bug costs no rental money; what has to be right before
renting is that the store holds everything the fit will need, which is why `line_head` and the
log-probabilities are captured now.

## Cost and memory (estimates; the smoke measures them)

| | cvector-generator, all pairs | this design |
|---|---|---|
| tokens decoded | ~16-23 M | ~0.35 M plus one persona |
| peak host RAM | ~51 GB for the largest set | model plus under 1 GB |
| chat vectors on an L40S | ~2-3 h | well under 1 h, mostly model load and preflight |
| refit with new labels or method | another rental | free, at home, from the store |
| data used | capped at 200 pairs per set | all 3,516 scored lines, every pair |

That should take roughly 2 hours and $1-1.50 off the step 5 rental, and removes the 64-128 GB RAM filter
on offers. The chat check's cost is step 5's, and running it in-process removes a model restart per
vector.

## Out of scope

- Voice vectors: the suit thread keeps them (`steer` on Qwen3-TTS). llama.cpp now has a `qwen3tts`
  builder, so the capture could serve a TTS model that runs in llama.cpp, but only if Brandon asks.
- Synthetic expansion of thin moods.
- Run-time changes beyond `vectors.toml` and the band GGUFs.

**For the runtime owner (not this builder):** on `qwen35`, `vi chat`'s `truncate_cache` asks for a
partial `seq_rm`, which the recurrent cache refuses with the default `n_rs_seq = 0`; the code then
clears everything, so every turn re-reads the whole conversation. Replies are correct, just slower.

## Review findings and where they went

| Finding | Change |
|---|---|
| H1 complete 0-3 scores, use all lines | `joint` regresses on score/3 over all 3,516 lines; masked variant reported; trusted sets kept for evaluation |
| H2 ridge lowers strength | tiny fixed ridge on mood columns only, unpenalised intercept and nuisances, joint/mean_diff norm ratio reported, shrink-toward-mean_diff only for ill-conditioned columns |
| H3 template mismatch | Rust renders with `format_chat`, whole-prompt tokenization, prefix tokenization check, byte-equal smoke |
| H4 llama-server steers differently | `vi vectors check` on `crates/llm`'s path |
| H5 CUDA artifact untested | ubuntu22.04 base, `CUDAARCHS`, run in the worker image in CI, GPU-vs-CPU preflight on the rental |
| M1 1/K isn't full strength | full strength = best single layer at 1.0; telescoping band; 1/K and 1/√K only comparisons |
| M2 band metrics drop thin moods | metrics are flags only; flagged and thin moods built at full strength; default when no layer stands out |
| M3 mean_diff reverses "jointly" | joint stays default; switching needs Brandon's yes; meanings documented |
| M4 mean_diff weighting | mean of pair differences |
| M5 nuisances | measured energy added; conversation counts; conversation fixed-effects diagnostic |
| M6 callback guard | 2-D check, per-layer count check, layers 1..n_layer-2, preflight layer test on every real model |
| M7 reuse settings | `n_seq_max = 2`, `kv_unified`, relative tolerance, also on the 27B |
| L1 store at home | `--store` on a disk with 6 GB free; `line_head` and log-probabilities captured now |
| L2 pooling | `pre_line` dropped; last prompt token included in `line_mean` |
| L3 sources | design cites the vendored llama.cpp in `llama-cpp-sys-2 0.1.159` |
| L4 cvector `fabs` | noted in research; nothing reuses that code |
