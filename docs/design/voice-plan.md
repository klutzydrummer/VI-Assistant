# Tali's voice: the plan from here

Status: plan, 2026-10-29. It replaces the voice parts of `after-labels-plan.md` and `step5-rental-plan.md`. Nothing new runs or spends until Brandon says go for that step. House rules hold throughout: heavy home jobs at nice 19, a GPU smoke before every full rental, game audio stays at home or on the rented worker, and his ears decide.

## Where we are

What the listening sets and runs showed (voicearms3, codectrip, rvctest, A/B; 2026-10-26 to 10-29):

- **Qwen3-TTS keeps her voice only in ICL mode.** That means prompted with one of her real clips. In x-vector-only mode, every method lost her voice.
- **Her own mood clip as the example carries the mood** about as well as the best steering. It scored 0.93 mood by ear, against D-icl 0.93, Dpca4-icl 1.00 and A 0.79; neutral scored 0.29.
- **Steering:** A steers indignation (A and A-decay) and alarm. D-icl steers alarm, and A-decay steers urgency. Warmth, amused and playful are "probe weak": the automatic probe can't hear them, so the verdict is by ear. B, C and the fitted token alone carried little or no mood, and are out.
- **The plain TTS loses her accent and suit effect in every mood, neutral included.** Blind: 0 of 19 TTS clips were "her, suit on".
- **The codec is not the main loss.** Through the 12 Hz tokenizer and back, 5 of her 8 strongest-suit clips kept the suit fully, 2 partly and 1 not. Both le1 clips lost some.
- **A fan RVC model after the TTS restores her voice and suit:** 17 of 19 clips became "her, suit on". The 2 failures were TTS clips that were already broken. RVC keeps the mood and the performance. In a blind A/B, Collige01/Tali beat AI-Wheelz/Tali_RVCv2 on 19 of 19 pairs, so it's the default.
- **RVC on the home CPU is 8.8 s of work per second of audio.** That's fine offline, far too slow for live replies.
- **Data:** 3,445 of her clips carry no extra effect (3.3 h): le1 726, le2 1,145, le3 1,574. Of those, 1,653 are 3–15 s long (2.5 h). There are also 155 gold-labelled lines.

## The target pipeline

```
app mood state ─► mood projector ─► Qwen3-TTS (ICL, her example clip) + one voice LoRA ─► RVC (Collige01) ─► audio
                         ▲                         ▲
                 (step 5; until then:      (step 4; until then:
                  refscreen's clip per mood  the base model)
                  plus steering A/D)
```

Qwen3-TTS carries the words and the mood. RVC carries her timbre and suit. The LoRA and the projector make the TTS part better and tie it to the app, and each one is kept only if it beats what it replaces, by ear, on held-out lines.

## Steps

### 1. refscreen: the best example clip per mood (ready; needs "go refscreen")
- **What:** every 3–15 s clip of hers in each mood, from her trusted clips with no effect and outside the held-out conversations, speaks 4 test sentences as the ICL example. That's 349 candidates over all 7 moods. They're scored on probe lift over neutral, voice similarity, WER and suit gap, and ranked. A hub list per mood gives the top 5 (8 where the probe is weak) plus 2 low-ranked controls, blind. Each shows her real line first, so it can get a sub-mood tag.
- **Cost:** about 1.5–2 h of GPU, about $1.50, smoke first.
- **Brandon:** about 175 clips, 45–60 min, mood 0–3 plus a sub-mood tag per real line. A lighter set is possible (about 90–116 clips).
- **Add to this rental:** time RVC (Collige01) on the worker's GPU, for step 3's latency number. That's a few minutes.
- **Gives:** the best clip per mood and sub-mood, an interim mood route needing no training, and labelled data for steps 4 and 5.

### 2. Finish the voicearms3 listening set (Brandon, any time)
- 35 of 128 done. It confirms by ear which steering arms (A, D-icl, Dpca4-icl) work with the longer fits, and which example clip (E1–E6) suits her best.

### 3. RVC in the pipeline
- **Interim voice:** TTS (ICL with the refscreen clip, steering where it works) → RVC Collige01. That's usable offline now.
- **For live use:** RVC has to run on the worker's GPU next to the TTS. Step 1's timing decides whether it fits the first-audio target (under 1.5 s, `mood-state-director.md`), and how lines must be chunked to stream.
- **Check:** a blind set of mood clips plain vs RVC, rated for mood 0–3. Brandon's first impression is that mood survives RVC; this confirms it per mood. About 30–40 clips, built at home.

### 4. One voice LoRA (Brandon, 2026-10-29: one adapter, never one per mood)
- **Data:** all her usable clips, every mood plus neutral, no effect clips. Each clip carries a short game tag, so the model learns the le1 and le2/3 suits apart and we ask for le2/3 at inference. RVC-converting le1 first is the fallback if mixed suits confuse it.
- **Training:** mood-blind first. The mood still comes from the example clip, so the LoRA only has to improve voice, accent and delivery. A mood tag in the text is the second option if the example clip alone doesn't carry mood reliably.
- **Before building:** read Qwen3-TTS's fine-tuning guide for the Base model: data format, how much data it needs, and memory. Then a smoke test on a few clips, then the run.
- **Check:** all 7 moods, against the untuned model, on the same held-out conversations and sentences, by ear and by refscreen-style scores. That includes the moods that failed zero-shot, e.g. playful and sadness. No mood is dropped before this comparison (Brandon, 2026-10-29). Also check that the LoRA still follows the example clip's mood and hasn't flattened every line to her average delivery.

### 5. A mood projector from the app's state
- **What:** a small network maps her mood variables from the app (`states.toml`: joy, sadness, anger, worry, playfulness…) to a few soft tokens for the talker. It's trained on top of the voice LoRA, with the TTS frozen. It replaces the hand-set mixing table for the voice, and handles blends ("a little sad, quite indignant") directly.
- **Data:** each clip paired with its mood vector, made automatically from the labelling run's tags, the gold labels and the probes. Brandon checks a sample, as with the gold set. refscreen's sub-mood tags feed in here.
- **Check:** on held-out lines, raising one mood input must audibly change the delivery toward that mood (by probe and by ear). A noisy-label projector learns to ignore its input, and this catches it.

### 6. LLM mood steering: test linearity before training a generator
- The app blends fixed LLM control vectors linearly. rental5 built 80 joint vectors and passed its chat check on 10-21, in that stage's checkpoint.
- **Test:** capture her lines' activation shifts against matched neutral lines, then fit mood vector → shift with a linear map and with a small MLP, and compare on held-out lines. One GPU capture plus minutes of CPU.
- Only if the MLP clearly wins: build a vector generator (mood state → per-layer vector) behind the same llama.cpp per-turn buffer, with a per-layer norm cap and a coherence check. If it doesn't win, fit the mixing table from data instead of setting it by hand.

### 7. Into the app
- A voice adapter in the engine: per sentence, the director's delivery and mood state choose the example clip (or projector input), the TTS + LoRA speak it on the worker, and RVC converts it there.
- Fallback clip bank per state, as `mood-state-director.md` describes.

## Decisions for Brandon
1. "go refscreen": full or lighter listening set; each candidate keeps its own x-vector (default).
2. Whether to fold the RVC GPU timing into the refscreen rental (default: yes).
3. After step 1: which interim mood route per mood (the refscreen clip alone, or plus steering where it steers).
4. Step 4 starts only after his go, with the cost estimate from the fine-tuning guide and the smoke test.

## Superseded (kept only for the record)
- x-vector-only mode, arms B and C, the fitted token alone (F-xvec), and the Dpca1 target: dropped after the 10-28 blind set.
- The likelihood "gate" (now a guard), fixed 200-step fits (now stop-on-loss), and the short (<3 s) reference clips.
- An EQ-only suit effects chain: le1.s129425 lost its suit by ear with a spectrum change of only 0.78 dB, so the suit isn't just EQ. RVC does it instead.
- AI-Wheelz/Tali_RVCv2: lost 19 of 19 in the A/B.
