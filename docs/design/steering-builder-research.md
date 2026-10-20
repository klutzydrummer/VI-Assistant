# Steering vector builder: research

Status: research, 2026-10-20. Step 1 of 4 in Brandon's order for our own chat steering vector builder:
research the maths and the existing tools, then design (`steering-builder-design.md`), then a critical
review in its own thread, then implementation in another thread after his go.

The question: can we build chat control vectors for the dialogue model (Qwen3.8-27B Q8_0, 64 layers,
width 5,120, `qwen35` hybrid architecture) from **all** of step 4's 80 chat contrast sets (8,986 pairs,
largest matter_of_fact at 665) without the ~51 GB of RAM llama.cpp's tool would need, and do it well?

## 1. What llama.cpp's `cvector-generator` actually does

Read from source (`tools/cvector-generator/`, llama.cpp master 707616f, 2026-10-20).

- **Each pair is decoded from scratch, twice.** For pair *i* it clears the KV cache, decodes the whole
  positive prompt, then the whole negative prompt (`get_hidden_layers`). The ~1,100-token persona is
  prefilled 17,972 times; every line that appears in several sets is redone each time.
- **Token-by-token diffs, aligned by position.** Both prompts are padded to the same length with the
  `" "` token, then `pos[t] - neg[t]` is taken at every position *t* and layer (`calc_diff`). Rows that
  are exactly zero are dropped, which removes the shared persona prefix (same tokens, same positions,
  same activations). Every position after the first differing token is kept, including positions where
  one prompt is only padding.
- **Everything is held in RAM** (`concat_diff_tmp`): every kept row, every layer, as F32, until all pairs
  are done. That's ~1.3 MB per kept token for this model, hence ~51 GB for the biggest set. The `mean`
  method stores everything too (the source has a TODO about it), although a mean needs only a running sum.
- **`mean` method:** the mean of all kept diff rows, then **normalised to length 1 per layer**
  (`mean.hpp`). The natural size of the shift is thrown away, and every layer gets the same length even
  though the residual stream's size grows several-fold with depth.
- **`pca` method:** power iteration on the uncentred diff matrix, from a random start seeded with the
  clock (`pca.hpp`). Nothing fixes the sign, so each layer's vector can independently point toward or
  away from the mood, and two runs can differ.
- **Off by one layer.** It captures `l_out-il` (the output of layer *il*, counted from 0) and writes it
  as `direction.(il+1)`. At run time llama.cpp adds `direction.k` to the output of layer *k*
  (`llama_adapter_cvec::apply_to`). So the vector found at a layer's output is added one layer later.
  Small, but wrong; our builder writes `direction.il`.
- **What it reads:** the eval callback asks for every tensor named `l_out*`, copies it to host memory
  with `ggml_backend_tensor_get`, and skips the last layer (it is cropped to output rows only).
- **Mixed-content diffs.** Positive and negative lines are different sentences, so position *t* compares
  unrelated words and, past the shorter line, a word with padding. For the mean, that sum still reduces to
  (sum over positive tokens) minus (sum over negative tokens), but the padding tokens enter as if they were
  part of the negative line, and long lines weigh more than short ones. For PCA the unaligned rows are
  mostly noise about which words happened to line up.

**Verdict:** the memory is a property of the tool, not of the maths. Its outputs also lose the natural
magnitude, can flip sign (PCA), sit one layer late, and are biased by padding and line length.

## 2. The maths: which direction to compute

Notation: for line *n* and layer *L*, `h_nL` is a summary of the residual stream over her line (section 3).

**Mean difference (CAA, Rimsky et al. 2023, arXiv 2312.06681; Marks and Tegmark's mass-mean; Arditi et
al.'s refusal direction).** `v_L = mean(h | positive) - mean(h | matched zero)`. Im and Li (arXiv
2502.02716) show it is the optimal steering vector under their mean-squared objective and that, unlike
PCA of differences, it doesn't latch onto the direction of most variance when that isn't the shift.
They report it beating PCA and classifier directions empirically. It needs only running sums: memory
is `2 x layers x 5,120` numbers per mood, whatever the set size. It keeps its natural size, so strength
1.0 means "shift by the average gap between her mood lines and her matched plain lines", a meaningful
"full strength".

**PCA of differences (RepE, Zou et al. 2023; repeng's `pca_diff`/`pca_center`).** Needs pairs and the
whole diff matrix, or a streaming covariance (`5,120 x 5,120` per layer, 105 MB at F32, ~6.6 GB for 63
layers). Sign must be fixed after the fact (repeng flips it so positives project higher). Gives unit
vectors. No evidence it beats mean difference for steering; evidence it can do worse.

**Classifier or whitened directions (logistic probe weights, LDA `Σ⁻¹Δμ`).** Better at *reading* a
concept, worse at *steering* it: ITI (Li et al. 2023) and Marks and Tegmark both found the mass-mean
direction steers better than probe weights. Useful as a diagnostic, not as the vector.

**Mean-centring (Jorgensen et al., arXiv 2312.03813).** Steering vector = mean over the target data
minus the mean over all data. Helped at the best layer, not at every layer. Our matched zeros already
play the role of the "all data" baseline, more tightly.

**Joint fit for multi-label lines.** Brandon's lines carry several moods at once (gratitude often with
warmth), and he OK'd "vectors fitted jointly" (2026-10-17). A per-set mean difference for warmth also
picks up part of gratitude, so `warmth + gratitude` double-counts when mixed. The standard fix is
multivariate least squares: `h_n = μ + Σ_k y_nk β_k + Σ_j c_nj γ_j + ε`, where `y_nk` is line *n*'s
label for mood *k* (0/1, or a level for the energy and drunk knobs) and `c_nj` are nuisance covariates
(game, line length). Each `β_k` is mood *k*'s own shift with the others held fixed, in the same units as
a mean difference, so blends add up by construction. With one mood and no covariates it reduces exactly
to the mean difference. It needs only the sufficient statistics `YᵀY` (K x K) and `YᵀH` (K x 5,120) per
layer, both running sums: about 260 MB in F64 for ~100 columns and 63 layers. Ridge regularisation
(`YᵀY + λI`) keeps thin moods stable; λ is chosen by cross-validation grouped by conversation.

**Alternatives looked at and not proposed:** conceptors (Postmus and Abreu, arXiv 2410.16314) need a
`d x d` matrix per mood per layer, too big for 80 moods and 63 layers, and llama.cpp can't apply them;
K-Steering (arXiv 2505.24535) and MSRS (arXiv 2508.10599) need gradient steps or non-additive runtime
code; steering tokens (arXiv 2601.05062) are trained. All of these would need engine changes beyond
`llama_set_adapter_cvec`.

## 3. Which tokens, and which layers

**Tokens.** repeng uses only the last token of each prompt; CAA uses the answer-letter token; style and
persona work usually averages over the response. At run time our vector is added while she *generates*
her reply, so the matching training signal is the residual stream over **her line's tokens**, with the
context teacher-forced. Averaging per line (not per token) gives each line one vote, so long lines don't
dominate, and drops the persona and the line she's answering, which are noise for her mood. The state at
the last token before her line (the "about to speak" state) is worth keeping alongside it as a cheap
second option.

**Layers.** CAA found early-to-middle layers work best on Llama 2; the llama.cpp README says above ~10;
a 2026 Jacobian study on Qwen3-8B found mid-early to mid-late layers most effective and very early ones
disruptive. Residual norms grow about 50x across layers in Qwen3-8B, so one fixed strength can't suit
every layer: natural-magnitude vectors (mean difference or joint fit) avoid that problem, unit vectors
(cvector-generator, repeng) have it.

**Adding at many layers compounds.** A vector added at layer *L* is carried forward by the residual
stream, so adding each layer's full mean difference at every layer of a band overshoots roughly in
proportion to the band. Published fixes: a single layer (CAA), or a band with each layer's vector
scaled by `1/K` or `1/√K` (arXiv 2608.08829, which also finds the best K layers together are not the K
best single layers). This is a run-time choice; the builder should write every layer and a recommended
band, and the chat check should settle the scaling.

**Choosing the band from data, without generating.** Per layer, how well does projecting onto the vector
separate her held-out mood lines from their zeros (AUC, grouped by conversation), and do two halves of
the data give the same direction (split-half cosine, the same as voice check B)? Higher cosine
similarity among a set's per-pair differences also predicts better steering (arXiv 2505.22637). All of
these come from per-line summaries, no generation needed.

**Streaming PCA, if it's ever wanted.** PCA needn't hold every row either: a streamed covariance
(`d x d` per layer, above), Oja's rule or incremental SVD (one pass, `d x k` memory), or a randomized
sketch (Halko et al.: multiply the rows by a random `d x (k+p)` matrix as they stream, then a small SVD)
all give the top components in bounded memory. Since mean difference is the better steering direction
(section 2), PCA is kept only as a comparison, and the streamed covariance for one set at a time is the
simplest exact route.

**Quantized GGUF versus BF16 activations.** Vectors are read from the same residual stream they're
added to, so building on the exact file and llama.cpp version used at chat time removes one source of
mismatch. Q8_0 is very close to BF16 (per-block 8-bit weights), so a Q8_0-built vector should transfer to
BF16 tools and to lower quants of the same model nearly unchanged. That's an expectation, not something
measured here; the step 5 chat check on the quant she runs is what confirms it. Libraries built on HF
transformers (repeng, EasySteer) would read BF16 activations of a model we then run as a GGUF.

## 4. Reliability limits worth knowing

Tan et al. (arXiv 2407.12404, NeurIPS 2024): steerability varies a lot across inputs and concepts, some
vectors push the wrong way on some inputs, and generalisation is brittle when prompts change. So the
per-vector chat check in step 5 stays essential, and a mood that fails falls back to prompt plus retrieved
lines, as already planned. Van der Weij et al. (arXiv 2403.05767) found merging several behaviour vectors
into one sum works poorly, while injecting different vectors at different layers works better. That
matters for mixing moods per turn: the joint fit reduces the double counting, and the chat check should
test a few two-mood blends, not only single moods.

**The hybrid architecture.** 48 of the model's 64 layers are Gated DeltaNet (linear attention); every
fourth is full attention. Each layer still writes to one residual stream of width 5,120, which is where
llama.cpp adds control vectors (`build_cvec`, then `l_out`, in `src/models/qwen35.cpp`). Published
steering on this family is thin: one 2026 preprint steered Qwen3.5-35B-A3B (same hybrid design) through
its residual stream successfully. The home smoke on Qwen3.5-0.8B and the step 5 chat check are the real
test.

## 5. Programmatic options

| Option | Fits our data | Notes |
|---|---|---|
| llama.cpp `cvector-generator` | no, ~51 GB for the largest set uncapped; caps throw data away | issues in section 1 |
| repeng (Python, HF transformers) | memory OK (last token only) | needs HF BF16 weights (~54 GB, an 80 GB card), PCA, unit vectors; activations from BF16, not the GGUF we run |
| EasySteer (vLLM, arXiv 2509.25175) | memory OK | fast, many methods, but HF weights and vLLM; same BF16 issue; another large stack to verify before renting |
| steering-vectors / nnsight / TransformerLens | memory OK | HF models only; same issue |
| **Our own capture via llama.cpp's eval callback** | yes, a few hundred MB | runs the exact GGUF and llama.cpp version `vi chat` uses; the engine already embeds it |

The last option is feasible with what we have:
- `crates/llm` embeds llama.cpp through `llama-cpp-sys-2 0.1.159`. Its bindings expose
  `llama_context_params.cb_eval` / `cb_eval_user_data` and every `ggml_*` function, including
  `ggml_backend_tensor_get` and `ggml_get_name` (bindgen allowlists `ggml_.*`, `gguf_.*`, `llama_.*`).
  The vendored llama.cpp has `qwen35` and names each layer's output `l_out`.
- llama.cpp's `imatrix` tool already does streaming statistics through the same callback, so the
  pattern is proven at this scale.
- `crates/llm` already reads and writes GGUF (`gguf.rs`), loads control vectors (`vectors.rs`), and has a
  tiny random llama model for tests that need no download (`testmodel.rs`).

**Prefix reuse.** Every prompt starts with the same persona (~1,100 tokens) and differs only in the line
she answers and her line (~100 tokens). Decoding the persona once and reusing it cuts the tokens to
decode by roughly 10x. For the full-attention layers that's an ordinary KV cache copy. For the Gated
DeltaNet layers, llama.cpp's recurrent memory supports `llama_memory_seq_cp` (the destination shares the
source's state cell; `src/llama-memory-recurrent.cpp`) but not trimming a sequence's tail, so each line
goes in its own sequence that is dropped afterwards. `llama_state_seq_get_data`/`set_data` are a fallback.
Either way it needs a test that a line's activations with reuse match a full decode.

**Deduplication.** The 17,972 prompts come from at most 3,516 corpus lines: a line that is a positive in
one set and a zero in others renders to the same prompt. Running each unique prompt once and storing a
per-line summary cuts the work again by about 5x and lets every set, and the joint fit, reuse it.

**Rough size of the work (estimates; the smoke measures them).** At most 3,516 lines x ~100 suffix
tokens is ~0.35 M tokens, against ~16-23 M for cvector-generator: minutes on an L40S rather than hours.
Per-line summaries for every layer: 3,516 x 63 x 5,120 x 2 bytes (F16) ≈ 2.3 GB on disk, 4.5 GB with the
"about to speak" state too. RAM during capture: one layer's activations at a time (~10 MB) plus the
running sums.

**Qwen 4 caveat (corrects an earlier note).** llama.cpp master's experimental `qwen4exp` builder uses
hyper-connections: the residual is `hc` streams of width 5,120, it names the layer output `l_last`, and it
does **not** call `build_cvec`. So llama.cpp can't apply control vectors to it yet, by any builder. Our
capture should read a configurable tensor name and refuse any shape other than `n_embd` per token, so a
new architecture fails loudly instead of producing wrong vectors.

**Building for the GPU.** The worker image has no compiler, and nothing is compiled on the worker. A
CUDA build of the capture tool can be made in GitHub Actions without a GPU (nvcc doesn't need one) and
downloaded by the worker, or built at home in a CUDA container. This is a design decision.

**Out of scope, noted:** llama.cpp now has a `qwen3tts` builder. If the voice side ever ran Qwen3-TTS
through llama.cpp, the same capture could build voice vectors. Not proposed without Brandon's yes.

## Sources

- llama.cpp source, master 707616f: `tools/cvector-generator/{cvector-generator.cpp,mean.hpp,pca.hpp}`,
  `src/llama-adapter.cpp`, `src/models/qwen35.cpp`, `src/models/qwen4exp.cpp`,
  `src/llama-memory-recurrent.cpp`, `tools/imatrix/imatrix.cpp`; `llama-cpp-sys-2 0.1.159` `build.rs`.
- repeng `repeng/extract.py` (github.com/vgel/repeng).
- Rimsky et al., Steering Llama 2 via Contrastive Activation Addition, arXiv 2312.06681.
- Im and Li, A Unified Understanding and Evaluation of Steering Methods, arXiv 2502.02716.
- Jorgensen et al., Improving Activation Steering with Mean-Centring, arXiv 2312.03813.
- Tan et al., Analysing the Generalisation and Reliability of Steering Vectors, arXiv 2407.12404.
- Understanding (Un)Reliability of Steering Vectors, arXiv 2505.22637.
- van der Weij et al., Extending Activation Steering to Broad Skills and Multiple Behaviours, arXiv 2403.05767.
- Deployable Per-Instance Multi-Layer Activation Steering, arXiv 2608.08829.
- Postmus and Abreu, Steering LLMs using Conceptors, arXiv 2410.16314; K-Steering, arXiv 2505.24535;
  MSRS, arXiv 2508.10599; Compositional steering tokens, arXiv 2601.05062; EasySteer, arXiv 2509.25175.
- Qwen3.5 hybrid steering preprint, arXiv 2603.16335; Jacobian layer study,
  alignmentforum.org/posts/EAE9u4YnE75eoiWDg (blog, not peer reviewed).
