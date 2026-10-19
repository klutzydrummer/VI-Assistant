# After the labels: from labelled lines to Tali talking in her voice

Status: draft 2026-10-19, planning only. Nothing here runs or spends until Brandon says go for
that step. Starts when pass 2 of the Claude labelling run is finished and its section 8 outputs
exist (score matrix, trusted positives and zeros, energy and intoxication per line).
Checked against `what-brandon-wants.md`; his concerns and where each is applied are at the end.

## The steps

Steps 1 to 4 are free and run at home on CPU (nice 19). Step 5 is the only rental, and it does
both the chat and the voice vectors, since an 8B model on his 4 cores would take days.

**0. Channel check (home, free; added by Brandon 2026-10-19).**
Some clips carry an extra effect on top of the suit filter (radio, comms, others). Six simple
per-clip measures (`corpus/tali/analysis/after_labels/channel.py`) turned out to follow her pitch
and loudness more than any effect, so Brandon marks about 40 clips by ear on his phone
(`vi review --channel corpus/tali`, appending to `corpus/tali/channel.jsonl`; the sample is
`analysis/after_labels/sample_channel.py`, and the page never shows why a clip was picked). A
research thread then tests AFx-Rep, Fx-Encoder++ and the six measures against his marks, and the
winner gives each clip an effect score. Step 3 matches pairs on it and checks that no mood's
direction is just the effect.

**1. Mood list and counts (home, free).**
Claude's families plus the must-have moods become one flat, multi-label list in `moods.toml`,
with energy and drunk (hungover as an option) as separate knobs. For each mood, count trusted
positives per game and scene, and flag moods that come mostly from one event (the LE3 party).
Nothing is cut yet: the counts and step 3 decide each mood's route.

**2. What clustering found that labels missed (home, free).**
Compare the audio clusters with the label families. A cluster no family covers becomes a
candidate mood and is shown to Brandon. This keeps clustering doing its job: finding moods
nobody asked for.

**3. Voice gate, parts A and B (home, free).** As decided in `voice-consistency-check.md`:
- A: an audio-only probe (prosody and emotion2vec, not the Whisper-based models) separates
  each mood's trusted positives from matched zeros, grouped by conversation, against shuffled
  labels.
- B: build each mood's vector from two random halves; the halves must point the same way.
- Energy and game are factored out first, so a mood vector doesn't learn the energy knob.
Each mood gets its route: **voice and chat** (passes), **chat only** (fails; the voice borrows
the nearest passing moods), **thin** (real but rare; blend or lower strength, flagged), or
**missing** (persona prompt only). Output: one table, mood by route, with counts.

**4. Chat contrast sets (home, free).**
For each mood and knob, her trusted positive lines against her matched trusted zeros, wrapped
in her persona prompt, written as the prompt files llama.cpp's `cvector-generator` takes. The
dialogue model is the GGUF the engine already embeds (`vi chat --model`).

**5. Chat and voice vectors, and the steering test, part D (one rental, his go).**
- Chat: build each vector from step 4's files, fitted together so blends add up. A strength
  sweep on open-ended chat (a judge for the mood, a check for gibberish) sets each maximum. A
  mood that doesn't steer falls back to prompt plus retrieved example lines.
- The TTS engine is chosen here. Default: Qwen3-TTS 1.7B first, because the `steer` stage
  (PR #20) already hooks it. A full bake-off of the other shortlisted engines runs only if
  Qwen3 doesn't steer.
- Before renting: list every runtime need against the worker image and smoke-test the exact
  code path on tiny inputs, all in one change.
- On the rental: build voice vectors from her clips for every mood that passed A and B, then
  generate neutral sentences with and without each vector. Automatic scoring: A's probe
  hears more of the mood, speaker similarity stays high, Whisper still gets the words.
- Brandon's part is light: a few blind A/B pairs per mood ("which sounds more X?").
- Estimated a few dollars of Vast credit.

**6. Wire it into the pack and run it (home).**
`states.toml` variables become the final moods and knobs; the mixing table gets one row per
vector per model, capped at the tested maximum; vectors are stored by model hash. The rules
director and persona retrieval (her lines indexed by mood) switch to the new list.
Milestone: `vi chat` with Tali replying in character, in her steered voice, to typed text.

**Later, not in this plan:** synthetic expansion of thin moods (see question 4), the small-model
director, speech input, tool agents, Carl.

## Decisions for Brandon

1. **Cut the mood list now or after step 3?** After (recommended): routes come from her data.
2. **TTS engine: Qwen3 first, or full bake-off first?** Qwen3 (recommended): already hooked, one rental.
3. **Chat vectors from her real lines, or from rewritten pairs?** Real (recommended): her own
   positives against her own matched zeros. Rewritten pairs (each line rewritten neutral by a
   model) only if real-line vectors turn out to steer the topic instead of the mood.
4. **Synthetic expansion of thin moods: plan it now or after step 3?** After (recommended):
   step 3 shows which moods are actually thin.

## His concerns and where each is applied

| Concern (Brandon) | Where |
|---|---|
| Moods come from her own voice and lines; vectors steer both voice and chat; blends | Steps 3 to 6; vectors built from her lines and clips only |
| Clustering stays to find moods nobody asked for | Step 2 |
| Cover the friend and assistant moods, not only discovered ones | Step 1 adds the must-have list; step 3 gives each a route |
| One flat multi-label list, plus energy and drunk knobs, hungover as an option | Step 1; energy factored out in step 3 |
| Thin moods stay in | Step 3's "thin" route; nothing is cut before counts |
| Claude labels are the main source; gold labels affect nothing | Steps 1 to 5 use only pass 1 and 2 labels; gold appears nowhere |
| Reduce reliance on his ears | His only part is a few blind A/B pairs in step 5 |
| Voice gate A, then B, then D | Steps 3 and 5 |
| Self-hosted; audio and game text stay home | Steps 1 to 4 and 6 at home; step 5 rsync only to the rental; no Claude API calls in this plan |
| No paid run without his yes; verify the whole setup before renting | Step 5, one rental, environment checked first |
| Heavy jobs on his PC at low priority; copy lines in code blocks | Steps 1 to 4; every go line |
| Prefer Rust, drop it where it limits | Pack and engine stay Rust; analysis stays Python like the other jobs |
| Short plans, no drift | This page; checked against `what-brandon-wants.md` |
