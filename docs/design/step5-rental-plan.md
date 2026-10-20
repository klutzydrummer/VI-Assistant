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

## Decisions for Brandon

1. **Dialogue model and quant.** The README's example is Qwen3-8B. Which GGUF file? Vectors are tied to
   the exact model.
2. **Judge for the chat check:** (a) self-hosted on the rental, e.g. a ~32B instruct model at Q4 next to
   the 8B (no API cost, adds about 20 GB of download); (b) Claude API (about $1 per pass, and only $1.26
   is left there); (c) only an automatic gibberish check, plus Brandon reading a few samples per vector.
3. **Pairs per chat set:** all 8,986 (best estimate, about 2 h of prefill), or capped at 200 per set
   (about 6,100 pairs, about 1.4 h). The cap only affects the biggest sets.
4. **Probe features for test D:** (a) refit step 3's probe on raw emotion2vec+ and prosody (simple; pairs
   are within-game anyway), or (b) save the confounds stage's fitted transform and apply it to the
   generated audio (same features as step 3, more code).
5. **Voice vector source:** her clips only (the plan), or also the "designed" pairs `steer` supports
   (VoiceDesign speaks the same sentence with and without the mood).
