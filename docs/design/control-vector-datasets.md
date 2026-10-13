# Control vector datasets

Status: plan approved 2026-10-12, revised 2026-10-13 after the 40-clip smoke test and a review of others' work ([primer](https://claude.ai/code/artifact/b314b4d6-516e-4fd2-a2b4-c8cda4158380)). Built: the stage runner, the GPU worker wrapper, harvest for LE2/LE3, and the discover and label jobs (`jobs/discover.py`, `jobs/label.py`, pipeline `characters/tali/dataset.toml`; signal fusion in `jobs/fusion.py`), which now save their embeddings. Smoke-tested on 40 real clips; the full run waits until the corpus is complete.

Decided: clips move only by direct rsync to the rented GPU, never through a third-party host (copyrighted material). Brandon will do about an hour of label review.

Revised 2026-10-13:
- **Reviews are the ground truth.** Agreement between our own signals is a health check, not accuracy. Each signal earns its weight per axis by matching Brandon's labels.
- **Specialist models measure; the audio LLM only checks.** Gemma names discover's clusters and rates a per-axis shortlist on tone; it no longer captions and scores every clip.
- **Channel is a first-class attribute.** Tali's helmet and radio filtering would otherwise dominate clustering and bias the acoustic models.
- **Steering is gated per axis.** An axis ships as a vector only if its strength sweep passes on open-ended chat; otherwise it falls back to prompting plus retrieved example lines.
- **Finish Tali end to end before a second character** (candidate: Carl from Aqua Teen Hunger Force, a different source, to test the method rather than the extractor).

Goal: from a character's own recorded lines, produce one control vector per mixing-table row (`tts`/`sad`, `llm`/`drunk`, ...) for the chosen models, with a tested maximum strength for each. Same pipeline for every character; Tali first.

## The axes come from the pack
The pack's `[[mix]]` rows list every vector we need, and its variables (joy, sadness, anger, worry, playfulness, intoxication) are the label axes. Labels are scored directly on those variables, so the dataset, the state machine and the mixing table all speak the same language. Adding an axis to a pack automatically adds it to the pipeline.

## Stages

```
harvest -> clean -> discover -> label -> review -> calibrate -> contrast sets -> extract -> evaluate
 (home)    (home)    (GPU)      (GPU)    (home)   (home + GPU)     (home)         (GPU)     (GPU)
```

Each stage writes content-addressed output and records the hashes of its inputs and settings, so rerunning only redoes stages whose inputs changed (a new label rule doesn't re-harvest). `vi dataset run` in Rust orchestrates (file formats and caching rules: [corpus-format.md](corpus-format.md)); GPU stages are Python jobs, since the models are PyTorch, sent to the same rented worker as batch work.

### 1. Harvest (home)
Extract every line the character speaks, plus the lines around it, from the game files:
- audio (24 kHz mono WAV) and its subtitle: each dialogue line in the game's conversation data points to a string id, and the string tables (`.tlk`) hold the subtitle text for that id. So every clip arrives with the game's own transcript and speaker, and nothing needs generating
- game (LE1/2/3), conversation and line ids
- context: the previous few lines and who said them
- scene tags from file names, such as the Citadel DLC party

Output: `corpus/<character>/lines.jsonl` plus `audio/`. Local only, gitignored. Expect a few thousand Tali lines across the trilogy.

### 2. Clean (home, CPU)
- Trim silence, normalize loudness, keep clips of 1 to 20 seconds.
- Run speech recognition (Whisper) only to verify the game subtitle. Small differences (an ad-lib, a trimmed word) mean we keep the clip and use Whisper's words as the exact transcript; a large mismatch means extraction mapped the wrong clip, so drop or flag it.
- Flag clips with music or effects mixed in (rare for game dialogue, but cutscenes happen).
- Tag each clip's **channel** (clean, helmet, radio). Later stages standardize and calibrate per channel, and TTS references and steering contrasts stay within one channel.
- **Speaker check**: a speaker-verification embedding per clip, and its distance from Tali's median voice per channel. Far-off clips are flagged as another voice. This replaces asking the audio LLM, which tagged another voice on every smoke-test clip.
- A **quality score** (DNSMOS or similar) as an attribute for choosing TTS references, not a filter: quality models rate filtered speech low and would delete her combat lines.
- Tag each line's **point in the story** (game, and act where known), for the persona.
- Deduplicate on game file id, then on transcript plus audio hash.

### 3. Discover (GPU)
Before scoring anything, let the data say what's there, so we find axes the pack is missing:
- Embed every clip three ways: an emotion representation (plain emotion2vec, which gives a rich embedding rather than four classes), prosody (pitch, energy, speaking rate), and the text with its context. The embeddings are saved for calibration and "more like this" in review.
- Standardize within each channel and factor out clip length, then cluster (UMAP, then HDBSCAN). Rerun with other seeds and keep only clusters that come back.
- For each cluster, pull exemplar clips and have the audio LLM summarize what they have in common. Cluster names are hints; reviews and fused labels decide what a cluster is.
- Compare against the pack's variables. Clusters no axis covers become candidate axes (flustered, awed, determined, tender...), and you decide which to add to `states.toml`.

**Audio language model.** Gemma 4 12B (Apache 2.0, native audio) names clusters here and rates shortlists in calibrate. It does not label every clip: studies (LISTEN, EmoVoice) found audio LLMs judge emotion mostly from the words and track people poorly per clip (Spearman about 0.2 to 0.26, against 0.40 for emotion2vec), and on the smoke test its captions were near-identical and it refused clips under about 1.6 s. Gemma 4 12B is also the text and context judge in label, where reading words is what we want.

Clusters will also pick up things that aren't emotion: combat barks vs conversation, ME1 vs ME3 recording differences, line length. Those get labeled as such, and the useful ones become filters (combat barks out of most contrast sets) rather than axes.

### 4. Label (GPU)
Independent signals per line, fused (plus the audio LLM's tags from discovery as a fourth, for the axes it proves it can hear):
1. **Audio emotion**: emotion2vec+ large (angry, happy, neutral, sad), plus audeering's arousal, valence and dominance model. These hear the delivery.
2. **Text and context judge**: an open-weight LLM on the worker reads the line with its context and scores each pack variable from 0 to 1, with a confidence, under constrained JSON output. It catches what audio models can't, like playfulness, worry and intoxication.
3. **Scene priors**: tags like "Citadel party" set a prior for intoxication, which no emotion model detects.

Fusion: a weighted average per axis, with different weights per axis (acoustics count most for anger and worry; the text judge and director notes for sadness, playfulness and intoxication). Lines where the signals disagree get a low confidence. Disagreement between script and voice (an angry line delivered flat) is kept as information, not averaged away.

Both audio models carry non-commercial licenses (audeering is CC BY-NC-SA). That's fine for personal use; swap them if that ever changes.

### 5. Review (home)
A small local review page: plays a clip, shows its scores, and you accept or nudge them. About 150 random clips come first, so calibration isn't biased toward hard cases; then the queue orders clips by uncertainty and by how much they'd change a vector, with "more like this" from the saved embeddings. Roughly an hour covers what matters. This stage makes the biggest quality difference.

### 5b. Calibrate (home, plus a short GPU pass)
- Fit the fusion weights per axis to the reviews, and train small per-axis probes on the saved embeddings (these also give an audio signal on axes no emotion2vec+ class maps to). Keep, per axis, only the signals that beat the others against the reviews, and publish a per-axis accuracy table.
- **Shortlist rating** (the ParaSpeechCaps cascade): per axis, take the top candidates by fused score, and have Gemma rate how well the delivery fits on a 1 to 5 scale, focusing on tone. Only top-rated clips become positives in contrast sets and TTS references.

### 6. Contrast sets (home)
For each axis, select:
- **Positives**: shortlisted lines high on that axis and low on the others, so "sad" doesn't secretly encode "drunk", from one channel so the vector doesn't learn "helmet".
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

LLM sweeps run on open-ended chat, not single sentences: steering effects vary a lot per input and some axes don't steer at all. An axis that fails ships without a vector and falls back to prompting plus retrieved example lines for that mood. People judge per-clip TTS results; emotion2vec labelled the data, so it can't be the only judge of the output.

## Order with the bake-off
Harvest, clean and label first. The labeled clips feed the TTS bake-off, which narrows the field to one or two engines whose backbone we can hook. Before building TTS extraction, test on one axis (sadness) whether steering beats simply choosing an emotion-matched reference clip from the labelled corpus; build extraction only if it does. Test with both matching and mismatched text, since the words override the reference's emotion in current models.

## Moving data
Clips go from home to the worker by rsync over the same SSH tunnel as everything else, roughly a few hundred MB as FLAC for Tali. Results (labels, vectors, reports) come back home. Nothing stays on the worker. Game audio never goes through a third-party host.

## Rough cost
On a rented RTX 3090 at about $0.14/hour, my estimates:
- labeling a few thousand lines (audio models plus a 14B judge): under an hour
- TTS extraction: roughly 30 minutes per axis per engine, 6 axes
- LLM vectors: minutes each
- evaluation sweeps: about an hour

Total is a few dollars for Tali, well inside the $25.

## Persona
The harvest doubles as the persona's memory: index real lines by situation and mood for retrieval at chat time, which is what captured character voice best in others' work. Start with one "latest" Tali; story-point tags keep a selectable point in the story possible later. The mood director restates persona and current mood every turn, since personas drift within a few turns.

## Other characters
Harvest is the only character-specific stage. For characters without game files, a later source adapter can take recordings (videos, audiobooks), isolate the character with speaker diarization (which the speech input work needs anyway), and feed the same pipeline.
