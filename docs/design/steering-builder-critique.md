# Steering vector builder: critical review

Status: review of `steering-builder-design.md` and `steering-builder-research.md` (draft 1, PR #63),
2026-10-20. Written for Brandon's order: research, then design, then this critical review, then an
implementation thread after his go. Nothing here is implemented.

What I checked against: llama.cpp master 707616f (the commit the research cites) **and** the copy the
engine actually vendors (`llama-cpp-sys-2 0.1.159`, downloaded from crates.io), `crates/llm`,
the step 5 plan (PR #60), `labelling-run-plan.md`, `after-labels-plan.md` and `what-brandon-wants.md`.
I had no web access, so the 2026 arXiv citations were not checked.

**Verdict:** the shape (capture per-line summaries once through llama.cpp's eval callback, fit at home
from the store) is sound and is the right answer to the memory problem. The llama.cpp claims hold up.
The design is not ready to implement yet, though. Two of Brandon's rules are broken quietly: ridge
shrinkage and `1/K` both lower strength before there's any evidence, and the joint fit drops data he
wants used. Two mismatches would make the step 5 results not carry over to `vi chat`: the prompt
template, and where the steering is applied. And the CUDA binary is never run before the rental.
Each of these has a cheap fix, listed below.

## Brandon's constraints, checked

| Constraint | Where the design stands |
|---|---|
| Vectors from her own lines | Met |
| Multi-label, fitted jointly | Default is joint, but the chat check may choose `mean_diff` "for the rest", which would reverse his decision (M3). The joint fit also uses absent=0 on labels that actually have full scores (H1) |
| Use as much of our data as possible (his words, 10:49) | Not met: the joint fit uses only lines that appear in some set, not all 3,516 scored lines (H1) |
| Full strength first, lower only on evidence | Broken twice: ridge shrinks every vector, thin moods most (H2), and `1/K` per layer is below full strength over most of the band (M1) |
| Thin moods stay | At risk: split-half ≥ 0.8 will fail thin moods, and "flagged before rental time is spent" reads as skipping them (M2) |
| No paid run without his yes | Met |
| Verify the whole environment before renting | Not met: the CUDA artifact, the real-model reuse test and the layer preflight never run before the rental (H5, M6, M7). The chat check also tests a different runtime path (H4) |
| Rust preferred | Met |
| Game data stays on his PC or the worker | Met (the store holds ids, hashes and numbers; the CI artifact is only code) |

## High: fix before implementation

**H1. The labels aren't "open-ended, absent means 0". Use the complete scores, on every line.**
The design assumes the Claude run was open-ended multi-label. It wasn't: pass 2 scored **every line
0 to 3 on every family** (`labelling-run-plan.md` §1, §8). Trusted positive means a score of 3, or 2
with pass 1 agreeing. Trusted zero means 0 with pass 1 silent. Everything else is a middle ground
(scores of 1 or 2, or the two passes disagreeing). Using 0/1 trusted indicators with absent=0 puts
those middle lines in as zeros, which shrinks every β and leaks it into correlated moods. Fix:
- regress on `score/3` for every family (complete data, no missing values, one shared `CᵀC`, so it
  stays cheap); or, if indicators are kept, mask the middle per column (the design's own fallback);
- fit over **all 3,516 lines**, not only those in a set. Every line has scores, the capture already
  dedups to at most 3,516 lines, and Brandon asked today for as much data as possible;
- keep the trusted sets for evaluation (AUC, split-half), where trusted zeros are what you want.

**H2. Ridge shrinkage lowers strength before any evidence, and thin moods lose the most.**
With λ > 0, β is pulled toward 0. Thin moods (37–96 lines) are pulled hardest, so a joint vector at
1.0 is weaker than that mood's real gap. That breaks "start at full strength". It also makes `joint`
and `mean_diff` hard to compare at equal strength. Choosing λ by "held-out prediction of h projected
onto the β directions" is underspecified and circular, because the directions depend on λ. Choosing
it by full-width MSE would favour heavy shrinkage, since mood is a tiny share of the residual's
variance. Fix: shrink **toward each mood's own mean difference** (`‖β−β_md‖²` penalty) instead of
toward 0, or use a small fixed λ just for numerical conditioning. Leave the intercept and nuisance
columns unpenalised, and drop one reference game level. Report each vector's norm against its
`mean_diff` norm, so any shrinkage that remains is visible.

**H3. The capture prompt can differ from what `vi chat` feeds the model.**
`render_cvector.py --gguf` applies the GGUF's own jinja template. `vi chat` uses
`llama_chat_apply_template` (`crates/llm/src/llama.rs:193`), and in the vendored llama.cpp that call
recognises any `<|im_start|>` template as plain ChatML and ends with `<|im_start|>assistant\n`, with
no think block (`src/llama-chat.cpp:99,256`). Qwen3.x jinja templates add a `<think>` block to the
assistant header. So vectors could be measured in one context and applied in another. There's a
second trap: tokenizing `context`, `line` and `suffix` separately and joining them can give different
tokens from tokenizing the whole string, which is what runtime does. Fix:
- render capture prompts in Rust with the same `format_chat` that `vi chat` uses. The capture already
  lives in `crates/llm`, so this is one source of truth and the Python renderer only supplies messages;
- tokenize the whole prompt once, find her line's token span from piece offsets, and assert that the
  persona prefix tokenizes identically on its own (prefix reuse needs this);
- add a smoke check that the capture prompt and `vi chat`'s prompt for the same exchange are byte-equal.

Separately, step 5 has to settle thinking on or off for Qwen3.8 at runtime. Plain ChatML without an
empty think block may make a hybrid-thinking model emit `<think>`.

**H4. The step 5 chat check uses a different steering path from `vi chat`, so the strengths it finds
won't carry over.** PR #60 runs the check on `llama-server --control-vector-scaled`. llama-server sets
the control vector once at startup (`common/common.cpp:1453`), so it applies to **every token**,
including the persona and the user's turn, with one layer range for all vectors. It never reads
`vectors.toml` scales or bands. `vi chat` decodes the prompt unsteered and steers only the last prompt
token and the generated tokens (`llama.rs` `generate_inner`), with per-vector `scale` and `layers`.
A strength tuned under llama-server means something else in `vi chat`. Fix: run the chat check through
`crates/llm`, for example `vi vectors check`, which reuses `Llama::generate` and `VectorBank::blend`.
That also removes the model restart per vector, since blends change per turn, so it saves rental time.
If llama-server stays for anything, the builder should also write "baked" GGUFs that contain only the
band layers with the scale already applied.

**H5. The CUDA binary is never run before the rental.** The GitHub Actions idea is good. `build.rs`
links `cudart_static` and `cublas_static` on Linux, so only the driver's `libcuda.so` is needed on the
worker. But:
- `nvidia/cuda:12.8.1-devel` without an OS suffix means Ubuntu 24.04 (glibc 2.39). The worker runs
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, which I believe is Ubuntu 22.04 (glibc 2.35), so
  the binary may not start. Pin `-ubuntu22.04` and check the worker image's glibc;
- `GGML_NATIVE` is off, so ggml builds its default list of about 8 CUDA architectures, with flash
  attention templates for each. On a 4-core runner that's a long build and close to the disk limit.
  Set `CUDAARCHS` to the rental's GPUs (for example `86-real;89-real`) and filter offers to match;
- add a CI step that runs the built artifact **inside the exact worker image** on `testmodel.rs`'s
  tiny model on CPU, which proves it loads, links and captures. I haven't checked that ggml-cuda falls
  back cleanly with no device; the job will show it;
- the first minutes on the rental: a tiny model captured on GPU must match the CPU result, then
  the reuse test and the layer preflight (M6) run on the 27B before the full capture starts. They can
  run while the 29 GB download finishes.

## Medium

**M1. `1/K` isn't "full strength", and the design should define that term before Brandon is asked.**
The mean difference at layer L is already cumulative: it's the whole gap present at that point. Adding
`d_L/K` at every layer of the band gives about the full gap only at the band's last layer. Near the
start of the band it's a fraction of the gap. The one unambiguous "unaltered" setting is CAA's: the
best single layer at 1.0. Proposal for Brandon:
- full strength = single best layer at 1.0, used as the anchor;
- band option: **telescoping increments**. Inject `d_a` at the band's first layer, then
  `d_L − d_{L−1}` at each later layer, so the stream is meant to carry each layer's own measured gap.
  It's still additive per layer, fits the existing GGUF format, and costs nothing from the store;
- `1/K` and `1/√K` stay as comparisons. Note that `1/√K` is stronger than `1/K`, so testing it isn't
  "lowering".

**M2. The band rule picks layers on reading metrics, and it can quietly drop thin moods.** On
teacher-forced line tokens, mid-layer AUC tends to saturate near 1, so "within 0.02 of best" will
likely cover most of the 20–80% window (K ≈ 38). The research itself says reading isn't steering.
Split-half cosine ≥ 0.8 depends on sample size (noise in 5,120 dimensions falls only as 1/√n), so
thin moods will fail it at every layer. Fix: settle the default band (or single layer) from the chat
check on a few moods, and use AUC and split-half only as flags. **A flagged mood is still built and
starts at full strength** (Brandon: thin moods stay; flagged vectors start at full strength, PR #62).
Say what happens when no layer passes.

**M3. Picking `mean_diff` "for the rest" would reverse "fitted jointly"; that needs Brandon's yes.**
The two methods also mean different things. A joint β is "this mood with the others held fixed". A
`mean_diff` is "sound like her lines of this mood", which includes the moods that tend to come with
it. Her gratitude co-occurs with warmth, so joint gratitude alone may sound flat, and the mixing table
needs to know that. Keep both, document what each means, and tell the state-machine owner.

**M4. `mean_diff` "each line counted once per set" breaks the matching.** If one zero line was matched
to three positives, counting it once changes the zero side's game, energy and length balance. Use the
mean of the pair differences, which weights each line by how many pairs it's in. It's still a running
sum.

**M5. The nuisance columns don't match how the sets were matched.** The sets were matched on
**measured** (audio) energy. The joint fit has the energy knob level instead. Add measured energy as a
nuisance column if mood vectors mustn't absorb energy. Some moods sit mostly in one conversation
(jealousy 75% in one, joy 51% at the party). For those, conversation-grouped 5-fold CV leaves nearly
every positive in one fold, so AUC means little. Report the number of conversations per mood, and add
a within-conversation diagnostic (conversation fixed effects; about 300 columns is still a small
solve) to show how much of a vector is topic.

**M6. The layer callback guard has gaps, and the layer test should run on every real model.**
- `qwen4exp` names its layer output `l_last`, so with `layer_tensor = l_out` nothing matches and
  nothing is refused. The capture must fail when the number of captured layers isn't the expected
  count. Its `l_last` is 3-D (`[n_embd, hc, T]`), so a check on `ne[0] == n_embd` alone passes; also
  require `ggml_n_dims == 2`.
- The design says layers `1..n_layer-1` are captured and also that the last layer is cropped; both
  can't hold. `qwen35.cpp` crops layer `n_layer-1` to the output rows unless every token asks for
  output. Either capture `1..n_layer-2`, or request all outputs (that would also give per-token
  log-probabilities of her line for free).
- The planted-vector layer test (`llama_set_adapter_cvec` at layer k shows up at layer k's `l_out`)
  should run as a **preflight on each real model**: in the home smoke and at the start of the rental,
  not only on the tiny llama in CI. Architectures differ: `bitnet` and `lfm2` emit `l_out` twice per
  layer, while `deepseek4` and `qwen4exp` emit none.

**M7. Prefix reuse on the hybrid model works, but it needs settings and a realistic tolerance.**
I checked the source: `llama_memory_recurrent::seq_cp` shares the tail cell, and `find_slot` makes a
private copy (`src` set) when sequence 1 decodes, so the persona state really is reused. Needed:
- `n_seq_max ≥ 2`, so a free cell exists;
- `kv_unified = true`. Without it, `n_ctx` is split per sequence (`llama-context.cpp:382`) and every
  cross-stream `seq_cp` copies whole KV buffers;
- a relative tolerance (for example per-line cosine ≥ 0.9999, relative error ≤ 1e-3) rather than "F32
  tolerance". DeltaNet's chunking changes when the prefix sits in a separate batch, so results aren't
  bit-exact;
- the test also runs on the 27B on GPU (H5).

The fallback order is right. The new `n_rs_seq` rollback isn't a good substitute, because each
snapshot holds the whole recurrent state.

## Low

- **L1. The store comes home.** 2.3–4.5 GB lands on a PC whose `/home` was 99% full, and every fit
  pass reads it from a slow disk. Name where it goes. "Capture once" only covers the summaries
  actually captured. A mean over the first ~8 tokens of her line (where the reply's tone is set) and
  per-token log-probabilities are cheap to add now and would otherwise need another rental.
- **L2. Pooling.** I agree with `line_mean` as the default. `pre_line` mostly encodes the user's turn
  (what was said to her), not her mood, so it risks steering the model as if the user felt that way.
  Include the `pre_line` position in `line_mean`: `vi chat` steers the last prompt token too, so the
  steered positions then match exactly.
- **L3. Source versions.** The research reads master; the engine runs the vendored copy, which differs
  in places (for example, the recurrent memory and LoRA guards). I re-checked the key facts in the
  vendored copy and they hold. Cite the vendored copy in the design.
- **L4.** cvector-generator's `is_row_all_zeros` tests `> eps` without `fabs`. It doesn't matter here,
  and nothing reuses that code.

## The llama.cpp claims: verdict

| Claim | Verdict | Where |
|---|---|---|
| Each pair is decoded from scratch, everything held in RAM, even for `mean` | Confirmed | `cvector-generator.cpp` `get_hidden_layers` (`llama_memory_clear`), `build_v_diff` |
| Diffs are aligned by position with `" "` padding; zero rows dropped | Confirmed. The diff starts at the user turn, not her line, because each line answers a different line; the design's `line_mean` avoids that | `padding_seq`, `calc_diff`, `filter_nonzero_rows` |
| `mean` is normalised to length 1 per layer | Confirmed | `mean.hpp` |
| PCA sign is random, seeded from the clock | Confirmed | `pca.hpp:110` (`std::time(0)`), `:292` TODO "randomly inverted" |
| Off by one: `l_out-il` is written as `direction.(il+1)`, and `apply_to(il)` adds `direction.il` | Confirmed | `mean.hpp`/`pca.hpp` `direction.%zu, il+1`; `llama-adapter.cpp` `tensor_for(il)`; `qwen35.cpp` `build_cvec(cur, il)` then `cb("l_out", il)` |
| `qwen4exp` has no `build_cvec` | Confirmed in both master and vendored. Its only mention is a misleading comment. **PR #60's note that qwen4exp "applies control vectors" is wrong** | `qwen4exp.cpp:496` (master) |
| `seq_cp` on recurrent state gives reuse | Confirmed by reading; still needs the M7 settings and a test | `llama-memory-recurrent.cpp` `seq_cp`, `find_slot` |
| MTP layers aren't steered or counted | Confirmed. `n_layer()` excludes the nextn layers, so vector shapes are right | `llama-hparams.cpp:348` |

## Do the tests and the 0.8B smoke prove everything before a rental?

Not yet. Add these:
1. a CI job that runs the CUDA artifact in the worker image on the tiny model (H5);
2. a byte-equal prompt check between capture and `vi chat` (H3);
3. the chat check through `crates/llm` on 0.8B at home, including one two-mood blend (H4);
4. the layer preflight and the reuse test as a command that runs on any model, at home on 0.8B and in
   the first minutes of the rental on the 27B (M6, M7).

The fit itself runs at home after the rental, so a fit bug costs no rental money. What matters before
renting is that the store holds everything the fit will need (L1).

## Aside for the step 5 / runtime owner (not the builder)

On `qwen35`, `vi chat`'s `truncate_cache` asks for a partial `seq_rm`. The recurrent cache refuses it
when `n_rs_seq = 0` (the default), and the code then clears everything. So every turn re-reads the
whole conversation. Replies are still correct, just slower.
