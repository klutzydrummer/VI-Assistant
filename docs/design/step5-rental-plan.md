# Step 5: the one rental (chat vectors, voice vectors, test D)

Status: plan, 2026-10-20. Nothing is rented or built yet. Follows `after-labels-plan.md` step 5 and
the rule that a rental starts only after every runtime need is listed from the docs and the exact code
path has passed a cheap smoke test at home.

## What runs on the rental

1. **Chat vectors.** llama.cpp's `cvector-generator` on the dialogue model, one vector per set in
   `corpus/tali/analysis/after_labels/chat_sets/` (77 moods and 3 knobs, 8,986 pairs), after
   `render_cvector.py --gguf` applies that model's own chat template.
2. **Chat check.** Each vector at full strength (Brandon's rule: start unaltered, lower only if it
   performs badly) on a few open-ended chat prompts. A judge scores mood and gibberish; a vector that
   fails is retried at lower strength, and one that doesn't steer falls back to prompt plus retrieved
   example lines.
3. **Voice vectors.** Qwen3-TTS 1.7B Base, through the `steer` job (rebuilt on step 3's pairs, since
   it used to read label's axes): the mean backbone hidden state with her positive clips minus with
   her matched zeros. That covers the 11 voice-and-chat moods, the 4 borderline ones (playful, amused,
   romance, teasing), and apologetic (thin, but passing). Effect clips are left out.
4. **Test D.** Neutral test sentences spoken from her neutral clip with and without each vector, at
   full strength first. Each is scored automatically by step 3's probe (does it hear more of the
   mood?), ECAPA (is it still her?) and Whisper (are the words right?). Then, at home, a few blind A/B
   pairs per mood for Brandon on the phone page.

## Runtime needs (from the docs and sources)

| Need | Source | Status |
|---|---|---|
| CUDA llama.cpp with `llama-cvector-generator` and `llama-server` | `ghcr.io/ggml-org/llama.cpp:full-cuda`. Its Dockerfile copies all of `build/bin/*`, built with `GGML_CUDA=ON` on CUDA 12.8.1 | Verified in `.devops/cuda.Dockerfile`. No compiling on the worker: binaries and libs are pulled at home and sent over |
| Worker image and driver | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, offers filtered to CUDA ≥ 13.0 | As configured; no compiler in that image, so nothing gets built there |
| cvector prompt format | one prompt per line, positive and negative paired by line, equal counts, `\n` escapes (`string_process_escapes`) | Verified in source; `render_cvector.py` matches |
| cvector memory | it keeps every non-zero per-token diff row for every layer in host RAM. The persona prefix is the same in both prompts, so it should cancel out | Estimate for the largest set (matter_of_fact, 665 pairs × ~60 differing tokens × 4,096 dims × 35 layers × 4 bytes): ~22 GB. **The smoke measures the real row count.** Offers need ≥ 64 GB RAM |
| Voice model | `qwen-tts==0.1.1` in its own venv (it pins transformers 4.57), as `jobs/requirements-steer.txt` | Existing; the smoke runs it |
| Scorers | funasr emotion2vec+ large (probe features), speechbrain ECAPA, Whisper large-v3 | Existing in steer and clean |
| Disk | dialogue GGUF (~9 GB at Q8 for 8B), judge model if self-hosted (~20 GB), Qwen3-TTS Base (~4.5 GB), emotion2vec+ (~1.2 GB), Whisper (~3 GB), llama.cpp bundle (~1 GB) | 160 GB configured |

## Time and cost (estimate; the smoke refines it)

- **cvector prefill:** 17,972 prompts at about 1,300 tokens each (the persona is most of it) comes to
  about 23 M tokens. At a few thousand tokens per second on an L40S, that's about 1.5 to 2.5 hours.
  Capping pairs per set cuts this (decision 3).
- **Chat check:** 80 vectors × ~5 prompts × ~150 tokens, plus a model restart per vector, plus judging:
  about 1 hour.
- **Voice vectors and test D:** 16 moods. The old steer test took about an hour for 4 axis-source runs;
  this is about 2 hours.
- **Total:** about 5 hours on an L40S-class offer (≤ $0.70/h), so roughly **$3 to $4 of Vast credit**.
  Max hours goes to 6 for this one rental.

## Built before renting, in one change

- `steer` v2: reads step 3's voice pairs and the effect exclusion, and uses the full-strength-first
  rule.
- The chat check script (llama-server with `--control-vector-scaled`, the judge, the gibberish
  check).
- The probe for test D: refit at home on the features chosen in decision 4, saved as a small file the
  worker loads.
- A worker runner for the three parts, resumable, with outputs going to `VI_CHECKPOINT`.

## Smoke at home first (free)

1. llama.cpp on CPU (the same release) with a tiny GGUF (Qwen3 0.6B, about 0.6 GB download), on two
   rendered sets of 5 pairs. Checks: the format, the diff row count (that the persona cancels), and
   that the output loads in `crates/llm` (`Vector::from_gguf`).
2. `steer` v2 on fake models (unit tests), then one real Qwen3-TTS sentence on CPU to prove the venv.
3. The probe on one generated clip: features, then a score.
4. On the rental: `jobs/check-worker.sh`, then the real thing.

## The dialogue model: Qwen3.8-27B, and swapping it later (Brandon, 2026-10-20)

Brandon picked Qwen3.8-27B (vision, MTP), and wants it easy to swap: Qwen 4 is coming, and so may a
modified model or a fine-tune. Checked:
- **What it is:** a 27B dense model, 64 layers, hidden size 5,120. Its layers mix Gated DeltaNet (linear
  attention) with full attention, every fourth one. It has an MTP draft head and a separate vision
  projector (mmproj, ~0.9 GB).
- **llama.cpp runs it as the `qwen35` architecture.** That builder applies control vectors
  (`build_cvec`) and names each layer's output `l_out`, which is what cvector-generator reads. The
  engine's vendored llama.cpp (llama-cpp-sys-2 0.1.159) has `qwen35` too.
  - Note for Qwen 4: llama.cpp's current `qwen4exp` builder applies control vectors but has no `l_out`
    name, so cvector-generator couldn't capture it yet. That gets checked when it lands.
- **GGUF:** Qwen ships none; `unsloth/Qwen3.8-27B-GGUF` has Q8_0 (29.0 GB), UD-Q6_K_XL (25.3 GB),
  UD-Q4_K_M (16.5 GB) and others, plus mmproj. Vectors fit the exact file they're built on.
- **Vision and MTP only matter when talking to her**, not for building vectors (text prompts):
  `--mmproj` and `--spec-type draft-mtp` at inference. Speculative decoding doesn't change what the
  steered model says, only how fast. Whether llama-server runs MTP and control vectors together gets
  tested on the rental; it's not needed for the vectors. The engine itself doesn't do MTP yet.
- **Home smoke model:** Qwen3.5-0.8B (same `qwen35` architecture, hybrid layers, 0.5 GB), so the home
  test runs the same code path on CPU.

**Easy to swap.** Every model is an entry in one registry file: id, Hugging Face repo, file, sha256,
mmproj, a flag for MTP, or a local path for a fine-tune. Everything built from a model is stored
under that model's hash (`vectors/<character>/<llm|tts>/<model-hash>/`, as already planned). One
command per target redoes it all from step 4's chat sets, which don't depend on the model:
render with that model's chat template, generate, check. `vi chat` refuses to load vectors whose
hash doesn't match the model. A new model is then: add an entry, rent, run that command, and read
the check. Nothing else changes.

**Revised for 27B:**
- **cvector memory:** 5,120 dims × 63 layers × 4 bytes is about 1.3 MB per differing token. The
  largest set uncapped (matter_of_fact, ~40k rows) would need about 51 GB of RAM; capped at 200 pairs
  it's about 16 GB.
- **Prefill:** about 23 M tokens uncapped, about 16 M capped at 200 per set. At an estimated
  1,500-2,500 tokens/s on an L40S, that's about 2-3 h capped.
- **Judge:** the same 27B, unsteered (no second model to download or fit). Q8_0 (29 GB) fits a 48 GB
  card with room for context.
- **Total:** about 6-7 h, so roughly **$4-5 of Vast credit**, with max hours at 8 for this rental.

## Decisions for Brandon

1. **Model:** Qwen3.8-27B (decided). Still open: the quant, Q8_0 (29.0 GB, closest to the original),
   UD-Q6_K_XL (25.3 GB) or UD-Q4_K_M (16.5 GB). Vectors match the file, so pick what she'll run on.
2. **Judge for the chat check:** the same model, unsteered (recommended now that it's 27B), or an
   automatic gibberish check plus Brandon reading samples.
3. **Pairs per chat set:** capped at 200 (recommended for 27B: about 16 GB of RAM and 2-3 h), or all
   (needs 64+ GB of RAM free for one set and about 3-4 h).
4. **Probe features for test D:** refit on raw emotion2vec+ and prosody (recommended), or save the
   confounds transform.
5. **Voice vector source:** her clips only first (recommended), designed pairs only if a mood fails.
