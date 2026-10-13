# Control vector datasets

Status: plan approved 2026-10-12, revised 2026-10-13 after the 40-clip smoke test and the research in [prior-work.md](prior-work.md). Built: the stage runner, the GPU worker wrapper, harvest for LE1/LE2/LE3, and the discover and label jobs (`jobs/discover.py`, `jobs/label.py`, pipeline `characters/tali/dataset.toml`; signal fusion in `jobs/fusion.py`). Discover and label ran end to end on 40 real clips (`characters/tali/smoke.toml`).

Decided:
- Clips move only by direct rsync to the rented GPU, never through a third-party host (copyrighted material).
- Brandon will do about an hour of label review.
- Brandon's reviews are the ground truth (2026-10-13). Model-vs-model agreement is a health check only. Every signal keeps its weight on an axis only if it matches the reviews there.
- The audio LLM verifies, it doesn't label (2026-10-13). It rates how well a shortlisted clip fits one axis; its open captions are for discovery only.
- Review starts with a random calibration sample, then the uncertainty queue (2026-10-13).
- Finish Tali end to end before a second character (2026-10-13). The second will be Carl (Aqua Teen Hunger Force), a different kind of source, to test the method rather than the extractor.

Open:
- TTS emotion: steering vectors vs choosing an emotion-matched reference clip. Held until the LLM side is done.
- Persona timeline: one "latest" Tali, or a selectable point in the story.

## What changed on 2026-10-13
The smoke test and the research ([prior-work.md](prior-work.md)) agreed on five things:
1. Open-ended audio-LLM labelling is unreliable: audio LLMs judge emotion mostly from the words. On our clips Gemma tagged every one `other_voice`, wrote near-identical captions and refused short clips. So it moves from labeller to shortlist verifier (stage 4).
2. Agreement between models isn't accuracy. Even humans agree weakly on speech emotion. So review comes before tuning, and a calibrate stage (5b) fits everything to the reviews.
3. Voice checks need a speaker model, not an LLM. Clean gains a speaker-embedding distance from her typical voice (stage 2).
4. Steering vectors fail on some axes for some inputs. Each axis ships as a vector only if its strength sweep passes on open-ended chat; otherwise it falls back to prompting with retrieved example lines (stage 8).
5. Personas drift within a few turns and leak later-story knowledge. The runtime re-states persona and mood every turn, and lines and lore carry a point in the story.

Goal: from a character's own recorded lines, produce one control vector per mixing-table row (`tts`/`sad`, `llm`/`drunk`, ...) for the chosen models, with a tested maximum strength for each. Same pipeline for every character; Tali first.

## The axes come from the pack
The pack's `[[mix]]` rows list every vector we need, and its variables (joy, sadness, anger, worry, playfulness, intoxication) are the label axes. Labels are scored directly on those variables, so the dataset, the state machine and the mixing table all speak the same language. Adding an axis to a pack automatically adds it to the pipeline.

## Stages

```
harvest -> clean -> discover -> label -> review -> contrast sets -> extract -> evaluate
 (home)    (home)    (GPU)      (GPU)    (home)       (home)         (GPU)     (GPU)
```

Each stage writes content-addressed output and records the hashes of its inputs and settings, so rerunning only redoes stages whose inputs changed (a new label rule doesn't re-harvest). `vi dataset run` in Rust orchestrates (file formats and caching rules: [corpus-format.md](corpus-format.md)); GPU stages are Python jobs, since the models are PyTorch, sent to the same rented worker as batch work.

### 1. Harvest (home)
Extract every line the character speaks, plus the lines around it, from the game files:
- audio (24 kHz mono WAV) and its subtitle: each dialogue line in the game's conversation data points to a string id, and the string tables (`.tlk`) hold the subtitle text for that id. So every clip arrives with the game's own transcript and speaker, and nothing needs generating
- game (LE1/2/3), conversation and line ids
- context: the previous few lines and who said them
- scene tags from file names, such as the Citadel DLC party

Output: `corpus/<character>/lines.jsonl` plus `audio/`. Local only, gitignored. Expect a few thousand Tali lines across the trilogy.

### 2. Clean
- Trim silence; measure loudness. Flag rather than drop by length: 472 of Tali's 3,387 clips are under 1 s and 1,158 under 2 s, mostly barks, and every model here is weak on them.
- Run speech recognition (Whisper) only to verify the game subtitle. Small differences (an ad-lib, a trimmed word) mean we keep the clip and use Whisper's words as the exact transcript; a large mismatch means extraction mapped the wrong clip, so drop or flag it.
- **Speaker check**: a speaker embedding per clip, and its distance from her median voice. Far clips are another voice, heavy effects or a wrong mapping. This replaces the audio LLM's `other_voice` tag, which heard her helmet filter as a second voice. For characters whose voice changes by condition (helmet, radio), compute the median per condition.
- **Quality score** per clip (a DNSMOS-style estimate), used later to pick TTS references.
- Flag clips with music or effects mixed in (rare for game dialogue, but cutscenes happen).

Built as `jobs/clean.py` (decided 2026-10-13): it runs on the worker (Whisper large-v3, ECAPA-TDNN from SpeechBrain, DNSMOS). It flags and doesn't drop, until the thresholds are set from the flagged clips. One change from the list above: loudness is measured, not normalized, since how loud she is carries emotion. TTS references can be normalized when they're picked.

### 3. Discover (GPU)
Before scoring anything, let the data say what's there, so we find axes the pack is missing:
- Embed every clip three ways: an emotion representation (emotion2vec+'s utterance embedding, not just its class scores; plain emotion2vec was the first choice, but on the smoke test its embedding mostly encoded clip length, R² 0.81 against 0.35 for emotion2vec+), prosody (pitch, energy, speaking rate), and the text with its context.
- Cluster (UMAP, then HDBSCAN) and, for each cluster, pull exemplar clips and have the LLM summarize what its lines have in common.
- In parallel, the audio LLM captions every line with free-form emotion words, and we cluster that vocabulary too. These captions are for discovery only: they name clusters and suggest axes, and their per-axis guesses are not a label signal (stage 4).
- Compare against the pack's variables. Clusters no axis covers become candidate axes (flustered, awed, determined, tender...), and you decide which to add to `states.toml`.

**Audio language model.** An open-weight audio LLM (default candidate: Gemma 4 12B, reportedly Apache 2.0 with native audio input at 16 kHz, which fits a 24 GB card at 8-bit; Qwen3-Omni as the alternative) listens to each clip and writes a short free-form description: delivery, laughter, whispering, slurring, shouting, background music or effects, other voices. It handles the open-ended jobs the fixed models can't: discovery captions, filtering bad clips, and spotting things like drunk delivery. It does not replace the acoustic emotion models. A 2025 benchmark (LISTEN) found current audio LLMs mostly judge emotion from the words rather than the sound, so delivery scores still come from emotion2vec and the arousal/valence model. Gemma 4 12B can also be the text and context judge in the labeling stage, so one model covers both jobs. We check this on Tali's own data: caption a sample with and without the transcript visible, and trust the audio LLM only on what it hears without the words.

Clusters will also pick up things that aren't emotion: combat barks vs conversation, ME1 vs ME3 recording differences, line length. Those get labeled as such, and the useful ones become filters (combat barks out of most contrast sets) rather than axes.

### 4. Label (GPU)
A cascade, after ParaSpeechCaps (prior-work.md, section 1): cheap specialist signals score every clip, then the audio LLM checks only a shortlist.
1. **Audio emotion**: emotion2vec+ large (angry, happy, neutral, sad), plus audeering's arousal, valence and dominance model. These hear the delivery.
2. **Text and context judge**: an open-weight LLM on the worker reads the line with its context and the LE3 voice director's note, and scores each pack variable from 0 to 1, with a confidence. It catches what audio models can't, like playfulness, worry and intoxication.
3. **Priors**: scene tags (a Citadel party suggests intoxication) and stage-direction words.
4. **Audio LLM verification**: per axis, the clips the first three rank highest, plus a matched sample of low ones, go to the audio LLM with one narrow question: how well does this delivery fit *sad*, from 1 to 5. Only its rating is used, only on the shortlist, and only on axes where calibration (5b) shows it agrees with the reviews.

Fusion: a weighted average per axis. Lines where the signals disagree get a low confidence and go to review first. The weights start as defaults and are refit in 5b.

Both audio models carry non-commercial licenses (audeering is CC BY-NC-SA). That's fine for personal use; swap them if that ever changes.

### 5. Review (home)
A small local review page: plays a clip, shows its scores, and you accept or nudge them. About an hour, in two parts:
1. **Calibration sample**: about 150 clips drawn at random (stratified by game and by bark/conversation), so the accuracy numbers in 5b aren't biased toward hard cases.
2. **Uncertainty queue**: then clips ordered by low fused confidence and by how much they'd change a vector.

### 5b. Calibrate (home, CPU)
Fit everything to the reviews:
- **Per-axis probes**: a small classifier per axis on the saved embeddings, trained on reviewed clips. It gives every axis an audio signal, including playfulness and intoxication, which no emotion2vec+ class covers, and it learns her helmet-filtered voice instead of assuming the generic class mapping.
- **Fusion weights** refit per axis against the reviews.
- **Signal selection**: a signal that doesn't beat chance against the reviews on an axis gets weight 0 there.
- Output: a per-axis table of each signal's and the fused score's agreement with the reviews, on the calibration sample. This is the number that says whether an axis is labelled well enough to build a vector from.

### 6. Contrast sets (home)
For each axis, select:
- **Positives**: lines high on that axis and low on the others, so "sad" doesn't secretly encode "drunk".
- **Negatives**: neutral lines matched on length and speaking rate.

For the LLM, also build **minimal pairs**: the judge model rewrites each positive line into a neutral version with the same content, so a pair differs only on that axis. That's the cleanest steering signal, and it's text-only, so it's cheap.

Sets are capped and balanced per game, so ME3's larger script doesn't dominate.

### 7. Extract (GPU)
- **TTS**: for each axis, generate a fixed set of neutral test sentences with each positive clip as the reference, then each negative clip. Record hidden states at every layer of the TTS model's language-model backbone, averaged over generated audio tokens. The vector per layer is mean(positive) minus mean(negative); PCA is an alternative to compare. Then orthogonalize each axis against the others so blends add cleanly.
- **LLM**: llama.cpp's `cvector-generator` takes positive and negative prompt files (one prompt per line, chat template included) and writes a GGUF control vector by PCA or mean difference. Feed it the minimal pairs wrapped in Tali's persona prompt. Its own guidance is that vectors work best on layers above about 10.

Vectors are tied to the exact model, so each is stored under that model's hash: `vectors/<character>/<target>/<model-hash>/<axis>.{gguf,safetensors}`.

### 8. Evaluate (GPU)
For each vector, sweep strength from 0 upward and measure:
- **TTS**: does the emotion model hear more of that axis? Does speaker similarity to her real clips stay high? Does Whisper still understand every word?
- **LLM**: a judge scores how much the axis shows, and a coherence check catches gibberish.

The highest strength that passes all checks becomes the suggested `max` for that mixing-table row. Then check pairs (sad + drunk, playful + drunk) to confirm both still come through when blended. Output is a short report with audio samples to listen to.

Run the LLM sweep on open-ended chat prompts, not single sentences: steering works very unevenly across inputs, pushes some the wrong way, and costs coherence (prior-work.md, section 6). Report per axis how many prompts moved the right way, not only the average.

**Fallback.** An axis whose vector doesn't pass ships without one: the mood director expresses it by prompting, with a few of her real lines at that mood retrieved from the labelled corpus as examples. So every axis works at runtime; vectors make the ones that pass stronger and steadier.

### 9. Runtime notes (for the mood director)
- Re-state the persona and current mood every turn, not once in a system prompt; personas and speaking styles drift within a few turns (prior-work.md, section 7).
- Retrieve real lines by situation and mood as in-context examples; the labelled corpus is that index.
- Tag lines and wiki facts with a point in the story (game, and act where known), so a persona can be set to a moment without knowing later events. Whether we offer that or one "latest" Tali is open.

## Order of work
1. Finish the corpus (harvest, all three games).
2. Build clean, including the speaker check and quality score.
3. One full discover + label run. It also checks whether padding short clips with silence helps the audio LLM or makes it invent things.
4. Review (calibration sample, then the queue) and calibrate.
5. Shortlist verification pass with the audio LLM, kept per axis only if calibration supports it.
6. Contrast sets, LLM vectors, evaluation with the fallback.
7. TTS, after the open question above is decided.
8. Then Carl, with everything Tali-specific (helmet note, scene priors, stage-direction words) moved into her pack first.

### TTS bake-off
Harvest, clean and label first. The labeled clips feed the TTS bake-off, which narrows the field to one or two engines whose backbone we can hook. Only then extract and evaluate vectors, so we pay for extraction once per engine we'd actually use.

## Moving data
Clips go from home to the worker by rsync over the same SSH tunnel as everything else, roughly a few hundred MB as FLAC for Tali. Results (labels, vectors, reports) come back home. Nothing stays on the worker. Game audio never goes through a third-party host.

## Rough cost
On a rented RTX 3090 at about $0.14/hour, my estimates:
- labeling a few thousand lines (audio models plus a 14B judge): under an hour
- TTS extraction: roughly 30 minutes per axis per engine, 6 axes
- LLM vectors: minutes each
- evaluation sweeps: about an hour

Total is a few dollars for Tali, well inside the $25.

## Other characters
Harvest is the only character-specific stage. For characters without game files, a later source adapter can take recordings (videos, audiobooks), isolate the character with speaker diarization (which the speech input work needs anyway), and feed the same pipeline.
