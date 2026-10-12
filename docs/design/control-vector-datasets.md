# Control vector datasets

Status: plan for review. Nothing here is built yet.

Goal: from a character's own recorded lines, produce one control vector per mixing-table row (`tts`/`sad`, `llm`/`drunk`, ...) for the chosen models, with a tested maximum strength for each. Same pipeline for every character; Tali first.

## The axes come from the pack
The pack's `[[mix]]` rows list every vector we need, and its variables (joy, sadness, anger, worry, playfulness, intoxication) are the label axes. Labels are scored directly on those variables, so the dataset, the state machine and the mixing table all speak the same language. Adding an axis to a pack automatically adds it to the pipeline.

## Stages

```
harvest -> clean -> discover -> label -> review -> contrast sets -> extract -> evaluate
 (home)    (home)    (GPU)      (GPU)    (home)       (home)         (GPU)     (GPU)
```

Each stage writes content-addressed output and records the hashes of its inputs and settings, so rerunning only redoes stages whose inputs changed (a new label rule doesn't re-harvest). A `vi dataset <stage>` command in Rust orchestrates; GPU stages are Python jobs, since the models are PyTorch, sent to the same rented worker as batch work.

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

### 3. Discover (GPU)
Before scoring anything, let the data say what's there, so we find axes the pack is missing:
- Embed every clip three ways: an emotion representation (plain emotion2vec, which gives a rich embedding rather than four classes), prosody (pitch, energy, speaking rate), and the text with its context.
- Cluster (UMAP, then HDBSCAN) and, for each cluster, pull exemplar clips and have the LLM summarize what its lines have in common.
- In parallel, the LLM tags every line with free-form emotion words, and we cluster that vocabulary too.
- Compare against the pack's variables. Clusters no axis covers become candidate axes (flustered, awed, determined, tender...), and you decide which to add to `states.toml`.

**Audio language model.** An open-weight audio LLM (Qwen3-Omni class, or Qwen2.5-Omni 7B if memory is tight) listens to each clip and writes a short free-form description: delivery, laughter, whispering, slurring, shouting, background music or effects, other voices. It handles the open-ended jobs the fixed models can't: discovery captions, filtering bad clips, and spotting things like drunk delivery. It does not replace the acoustic emotion models. A 2025 benchmark (LISTEN) found current audio LLMs mostly judge emotion from the words rather than the sound, so delivery scores still come from emotion2vec and the arousal/valence model. We check this on Tali's own data: caption a sample with and without the transcript visible, and trust the audio LLM only on what it hears without the words.

Clusters will also pick up things that aren't emotion: combat barks vs conversation, ME1 vs ME3 recording differences, line length. Those get labeled as such, and the useful ones become filters (combat barks out of most contrast sets) rather than axes.

### 4. Label (GPU)
Independent signals per line, fused (plus the audio LLM's tags from discovery as a fourth, for the axes it proves it can hear):
1. **Audio emotion**: emotion2vec+ large (angry, happy, neutral, sad), plus audeering's arousal, valence and dominance model. These hear the delivery.
2. **Text and context judge**: an open-weight LLM on the worker reads the line with its context and scores each pack variable from 0 to 1, with a confidence, under constrained JSON output. It catches what audio models can't, like playfulness, worry and intoxication.
3. **Scene priors**: tags like "Citadel party" set a prior for intoxication, which no emotion model detects.

Fusion: a weighted average per axis. Lines where the signals disagree get a low confidence and go to review first.

Both audio models carry non-commercial licenses (audeering is CC BY-NC-SA). That's fine for personal use; swap them if that ever changes.

### 5. Review (home)
A small local review page: plays a clip, shows its scores, and you accept or nudge them. Clips are queued by uncertainty and by how much they'd change a vector, so roughly an hour of review covers what matters. Your corrections also recalibrate the fusion weights. This stage is optional but makes the biggest quality difference.

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

## Order with the bake-off
Harvest, clean and label first. The labeled clips feed the TTS bake-off, which narrows the field to one or two engines whose backbone we can hook. Only then extract and evaluate vectors, so we pay for extraction once per engine we'd actually use.

## Moving data
Clips go from home to the worker by rsync over the same SSH tunnel as everything else, roughly a few hundred MB as FLAC for Tali. Results (labels, vectors, reports) come back home. Nothing stays on the worker. A private Hugging Face dataset is possible as a staging area, but it's a third party holding game audio, so the default is direct transfer.

## Rough cost
On a rented RTX 3090 at about $0.14/hour, my estimates:
- labeling a few thousand lines (audio models plus a 14B judge): under an hour
- TTS extraction: roughly 30 minutes per axis per engine, 6 axes
- LLM vectors: minutes each
- evaluation sweeps: about an hour

Total is a few dollars for Tali, well inside the $25.

## Other characters
Harvest is the only character-specific stage. For characters without game files, a later source adapter can take recordings (videos, audiobooks), isolate the character with speaker diarization (which the speech input work needs anyway), and feed the same pipeline.
