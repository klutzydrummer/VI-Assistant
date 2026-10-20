# Does a mood sound consistent in her voice? Options

Status: **decided 2026-10-19 by Brandon: A, then B, then D.** C and E are not used. Nothing is built yet. A and B run at home after the labelling run; D runs once on a rental, on his go, after a TTS engine is chosen.

The question: after Claude labels her lines from text, a mood gets a voice vector only if her
recordings of that mood share a sound. Otherwise the vector learns noise, or learns the words.
This is open question 1 in `what-brandon-wants.md`.

What we already have at home: every clip's audio embeddings (emotion2vec+, WavLM, prosody
features, EmoNet and EIV scores), the fair test from the cluster work (refit per fold, folds
grouped by conversation, compared against shuffled labels), and the steering-test stage (PR #20).

## Option A: an audio probe on the embeddings we already have (free, CPU, at home)

For each mood: can a simple classifier tell its trusted positive lines from matched trusted
zeros (same game, energy and length) using **audio only**? It's cross-validated by
conversation and compared with shuffled labels, exactly like the cluster test.
- **Pass:** clearly above the shuffled-label result, so the mood has a consistent sound.
- **Fail:** her voice doesn't carry it reliably, so it's a chat-only mood.
- **Guard against "it's the words":** use features that don't hear words (prosody,
  emotion2vec). Leave out EmoNet/EIV, which run on Whisper and partly hear the words.
- **Guard against energy and game:** matched negatives, plus a check that the mood's
  direction isn't just the energy or game direction.

## Option B: split-half stability of the vector itself (free, CPU)

A control vector is roughly "average positive minus average matched negative". Build it twice
from two random halves of the positives. If the two directions point the same way, the mood
has a stable sound; if they don't, the vector would be noise. This tests exactly what gets
built, and pairs well with A.

## Option C: does it overlap what clustering found? (free)

Moods that line up with the audio clusters (anger, warm, sad, surprise, calm focus, urgency)
already have independent audio support. This works only as a supporting signal: clusters
found only about six real classes.

## Option D: the steering test, the real proof (GPU rental, on his go)

Build the vector on the TTS, then generate the same neutral sentences with and without it.
- **Automatic:** the probe from A should score the steered audio as more of that mood.
- **Brandon (light):** a few blind A/B pairs per mood, asking "which sounds more X?"

This is the final gate for a voice vector. It needs a TTS engine chosen and a rental, so it
runs once for all the moods that passed A and B.

## Option E: an audio model as judge (not recommended)

A self-hosted audio LLM comparing two clips ("which sounds more teasing?"). Comparisons are
easier for models than ratings, but earlier tests with Gemma judging audio went badly. Only
worth it if A and B leave many moods unclear.

## What each result means (the routes in must-have-moods.md)

- **Passes A and B:** voice and chat vector, then D confirms.
- **Passes the text side only:** chat vector; the voice borrows the nearest passing moods.
- **Borderline:** a voice vector, flagged, and a candidate for synthetic expansion. It starts at full
  strength; test D lowers it only if the steered audio misbehaves (Brandon, 2026-10-20). Playful, amused,
  romance and teasing take this route.

## Recommendation

A plus B as the automatic gate, since both are free and run on data we already have. Then
D once for all moods that pass, with automatic scoring and a handful of blind A/B pairs from
Brandon. C is a supporting signal; E only if needed.
