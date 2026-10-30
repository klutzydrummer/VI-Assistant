# Qwen3-TTS fine-tuning: what the official guide needs (2026-10-30)

Stage 5 step 1 of the Tali voice plan. I read the official `finetuning/` folder on QwenLM/Qwen3-TTS main (README, prepare_data.py, dataset.py, sft_12hz.py) and the `qwen-tts` 0.1.1 package it runs on. Nothing was rented or run. Earlier notes: `fan-tali-voice-recreations.md`, `suit-accent-options.md`.

## What it does

- **It's a full fine-tune of all weights, not a LoRA.** AdamW over every parameter in bf16, gradient accumulation 4. A LoRA (rank 16 via peft, as instavar did) would be a small change to the same loop. Either way it's one voice model for every mood.
- **It trains the no-reference mode.** The saved checkpoint is a `custom_voice` model with a named speaker, and `generate_custom_voice(text, speaker="tali")` speaks with no reference clip. That's the end state 1 path. It also takes an optional `instruct` text, which nobody has reported working on a fine-tuned speaker; it's worth one free test as a bonus mood lever.
- **Her identity is one frozen x-vector.** The speaker encoder turns `ref_audio` into one 2048-number vector at input slot 6. The script stores the vector from the *first training batch* as the speaker. The guide therefore says to use the same `ref_audio` for every line. Better for us: store her averaged x-vector (we already compute one, PR #85).
- **Slot 6 is where the mood projector plugs in.** The projector can add a mood offset to her x-vector at that slot. That matches the published finding that this x-vector carries emotion on our exact model (Brito 2026, in `steering-qwen3-tts.md`). So stage 5's projector is a small network from the app's mood state to that offset, trained with the voice model frozen.

## Data it needs

- A JSONL file, one line per clip: `audio` (wav path), `text` (transcript), `ref_audio`, plus optional `language`.
- **Audio must be 24 kHz.** It asserts on the reference mel and crashes otherwise. Mono wav.
- A prepare step turns each clip into codec codes with `Qwen3-TTS-Tokenizer-12Hz`. That's the same tokenizer the codec round trip already ran on your CPU, so this step can probably run at home at nice 19 (memory fit to be checked first).
- How much: 3.3 h is plenty. The issue #343 run used 21 min of one speaker. The guide supports one speaker only, which suits us.

## Bugs and traps to fix before any rental

1. **Double label shift.** `sft_12hz.py` on main still passes `labels=codec_0_labels[:, 1:]` to a model whose `transformers` loss shifts labels again. Each frame then learns the frame two ahead, which is the reported "speech speeds up every epoch" bug. The fix is `labels=codec_0_labels[:, :-1]`.
1b. **Code predictor pairing (my reading of the code, to confirm in the smoke test).** At generation, the code predictor gets the hidden state from the position *before* each frame, together with that frame's first code (`past_hidden` in the talker's `forward`). The official training loop instead gives it the hidden state *at* the frame. `jobs/voicelora.py` uses the one-position-back pairing. The smoke test checks it: speech speed and quality must hold after a few hundred steps.
2. **Learning rate.** Use 2e-6 (the main command). The one-click script's 2e-5 gives noise and endless output.
3. **flash-attention 2 is hard-coded.** It needs a flash-attn build matching the worker's torch and CUDA, a classic install failure. Switch to `sdpa` or pin a known wheel, and smoke-test it.
4. **Pinned versions:** `transformers==4.57.3`, `accelerate==1.12.0`. Plan the worker image around them.
5. **Disk:** each epoch saves a full copy of the model (about 4 GB, so about 40 GB for 10 epochs). Fine on the worker, but bring home only the chosen checkpoint.
6. **Short lines.** About 15% of short lines flipped timbre after a fine-tune (issue #343). Keep her short lines in, and test short lines specifically.
7. **Strength.** Community LoRAs sound best at scale 0.3 to 0.4, and 1.0 over-steers. That's one reason to prefer a LoRA over the full fine-tune: it gives a strength knob.

## GPU memory (my estimate, not measured)

A full fine-tune of 1.7B in bf16 with AdamW needs about 14 GB for weights, gradients and optimizer before activations. Batch 32 on a 24 GB card is tight; a 48 GB card is comfortable. A LoRA needs far less. The smoke test should log peak memory.

## Recommended shape

- **A LoRA, rank 16, on the official loop.** Apply the label fix, use sdpa, store the averaged x-vector, lr 2e-6, about 10 epochs, and save a checkpoint each epoch.
- **First pass mood-blind,** on her effect-free LE2/LE3 clips, with screams and pain lines excluded and the refscreen conversations held out. LE1 waits for our RVC.
- **Judge** without a reference clip, at scales 0.3, 0.5 and 1.0, on long and short lines, against the interim voice (her clip plus Collige01).
