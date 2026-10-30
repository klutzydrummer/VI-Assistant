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
1b. **Code predictor pairing (my reading of the code, to confirm in the smoke test).** At generation, the code predictor gets the hidden state from the position *before* each frame, together with that frame's first code (`past_hidden` in the talker's `forward`). The official training loop instead gives it the hidden state *at* the frame. `jobs/voicetune.py` uses the one-position-back pairing. The smoke test checks it: speech speed and quality must hold after a few hundred steps.
2. **Learning rate.** Use 2e-6 (the main command). The one-click script's 2e-5 gives noise and endless output.
3. **flash-attention 2 is hard-coded.** It needs a flash-attn build matching the worker's torch and CUDA, a classic install failure. Switch to `sdpa` or pin a known wheel, and smoke-test it.
4. **Pinned versions:** `transformers==4.57.3`, `accelerate==1.12.0`. Plan the worker image around them.
5. **Disk:** each epoch saves a full copy of the model (about 4 GB, so about 40 GB for 10 epochs). Fine on the worker, but bring home only the chosen checkpoint.
6. **Short lines.** About 15% of short lines flipped timbre after a fine-tune (issue #343). Keep her short lines in, and test short lines specifically.
7. **Strength (LoRA only, not used).** Community LoRAs sound best at scale 0.3 to 0.4. We do a full fine-tune with no strength knob.

## GPU memory (my estimate, not measured)

A full fine-tune of 1.7B in bf16 with AdamW needs about 14 GB for weights, gradients and optimizer before activations. Batch 32 on a 24 GB card is tight; a 48 GB card is comfortable. A LoRA needs far less. The smoke test should log peak memory.

## Decisions (Brandon, 2026-10-30 19:17)

- **A full fine-tune, no LoRA.**
- **No strength knob.** The app's state vector drives the voice through the projector. Knobs like energy and intoxication are just more inputs in that vector, so each training clip's state needs those values as well as its mood labels.

## Lessons from fine-tuning other models (researched 2026-10-30)

What usually goes wrong when a pretrained speech or language model is tuned on one small dataset:
- It forgets how to read new text: pronunciation, long and short lines, odd input.
- It narrows its expressive range toward the data's average delivery.
- For us, a third risk: **it can learn to ignore the speaker slot**, the input the mood projector needs later.

What helps, and how it applies here:

1. **KL to the frozen base model, aimed at replay data.** In LLM work, the KL between the tuned and base model tracks how much it forgets (RL's Razor, arXiv 2509.04259). Penalising it directly, by teacher-forcing on the base model's own samples, is an unbiased KL estimate and preserves general skills (Context-Free Synthetic Data, arXiv 2505.13811).
   - **The TTS catch:** a KL on HER clips would pull her voice back toward Base's clone of her, which is the very thing we're training away from.
   - **So it goes on replay lines:** generic sentences Base itself speaks in OTHER, public voices (LibriTTS-R, CC BY 4.0, downloaded to the worker). That protects reading skill and range while her clips move the voice.
   - **Cost:** a frozen copy of Base on the GPU (about 3.4 GB) and an extra forward pass. My estimate is 1.3 to 1.5 times the step time; the smoke test measures it. `voicetune.py train --replay … --kl 0.1`.
2. **Replay itself (prior preservation).** Stable-TTS (arXiv 2412.20155) keeps prosody by reusing the base model's own samples. Our replay lines do the same job.
3. **Each clip's own x-vector in the speaker slot, not one fixed vector.** Her x-vector moves with her mood (Brito 2026, on this exact model). With a constant input the model learns to ignore the slot. With per-clip vectors it learns that the slot steers delivery, which is what the projector will write to. The saved speaker is her mean x-vector. This is my reasoning, not a published result; the full run checks that changing the slot still changes delivery.
4. **Basics, which matter most:** low lr (2e-6), at most about 10 epochs, a checkpoint per epoch, held-out conversations, and the checkpoint picked by held-out loss plus your ears. The checks are word errors on unseen sentences, short lines, voice similarity, and pitch spread across moods.
5. **Not chosen:**
   - LoRA: less forgetting but less learning (Biderman et al. 2024, arXiv 2405.09673), and you chose full.
   - Weight interpolation after training (WiSE-FT, arXiv 2109.01903): that's a strength knob by another name. It stays a fallback only if the tuned model forgets and the KL can't fix it.
   - Partial fine-tuning of selected layers (CSP-FT, arXiv 2501.14273): the fallback if a full fine-tune forgets too much.

## Plan for the runs

- **Smoke test (one 48 GB rental, pennies):** `check`, then about 50 clips for 200 steps, once without KL and once with `--kl 0.1`. That proves both paths and measures memory and time. A few no-reference lines go on the hub.
- **Full runs:** her effect-free LE2/LE3 clips, refscreen conversations held out, per-clip x-vectors, 10 epochs. One run without KL and one with `--kl 0.1`, on the same rental if the smoke cost allows. Judge blind against the interim voice (her clip plus Collige01), on long and short lines.
