# Corpus format and stage runner

Status: format v1, implemented in `crates/dataset` (`vi dataset`, `vi corpus`). The extractor and the review page build against this file; change it here first.

The dataset pipeline (see `control-vector-datasets.md`) turns a character's game lines into labeled clips. Every stage reads files and writes files, so this document fixes two things: what those files look like, and how the runner decides which stages to rerun.

## Where things live

```
characters/tali/dataset.toml      pipeline definition, committed (no game data)
corpus/tali/                      the corpus, gitignored, home machine only
  objects/<sha256>                every output file, stored once by content
  stages/<stage>/<key>/
    manifest.json                 what produced this output and what it holds
    log.txt                       the job's stdout and stderr
    out/                          the stage's output files (hard links into objects/)
  stages/<stage>/current          one line: the key of the latest successful run
  stages/<stage>/.checkpoint-<key>/  a run's saved progress (VI_CHECKPOINT), kept while it hasn't succeeded
  reviews.jsonl                   Brandon's review decisions (written by the review page)
```

`corpus/` is in `.gitignore`. Nothing under it is ever committed or uploaded anywhere except by rsync to the rented worker.

To find a stage's current output, read `stages/<stage>/current` and open `stages/<stage>/<key>/out/`. That is all a reader (the review page, a notebook) needs to know about the runner.

## Conventions for every file
- Records are JSON Lines: one UTF-8 JSON object per line, `\n` endings, no blank lines.
- Every record carries `id`, the line id. Files from later stages are keyed by `id` and don't repeat earlier stages' fields, so a reader joins them by id.
- Unknown fields are allowed and must be preserved by tools that rewrite a file. New optional fields don't bump the format version.
- Paths inside records are relative to the `out/` folder of the stage that wrote the record.
- Hashes are lowercase hex SHA-256 of the file bytes (plain `hashlib.sha256` in Python).
- Scores are floats from 0 to 1.

### Line ids
`<game>.<rest>`, made of `[a-z0-9._-]`, at most 120 characters, unique within a corpus and stable across extractor reruns, since labels and reviews hang off them. `game` is `le1`, `le2` or `le3`. Since a line can appear in several packages, build the id from the string id and gender variant, not the package: `le2.s348112_f`, `le3.s612004` (no variant). If one string id and gender has genuinely different audio in two places, suffix the later one `.2`, `.3`. Ids double as file names.

## Stage outputs

### harvest: `lines.jsonl` + `audio/` + `conversations/` + `harvest.json`
Written by the extractor (see `harvester-plan.md`). One record per distinct line the character speaks; the same line found in several packages is one record with every place listed in `source.occurrences`.

```json
{
  "id": "le2.s348112_f",
  "speaker": "tali",
  "text": "Keelah se'lai.",
  "flags": [],
  "audio": {"path": "audio/le2.s348112_f.wav", "sha256": "9f2c...",
            "sample_rate": 24000, "channels": 1, "duration_s": 1.42,
            "source": {"container": "Wwise_QuaTlL_201.afc", "offset": 1048576, "size": 23110,
                       "codec": "wwise_vorbis", "sample_rate": 24000, "sha256": "41aa..."}},
  "source": {"game": "le2", "package": "Game/ME2/BioGame/CookedPCConsole/BioD_QuaTlL_201_LOC_INT.pcc",
             "conversation": "quatll_tali_trial_d", "node": {"kind": "entry", "index": 37},
             "string_id": 348112, "gender": "f", "tlk": "BIOGame_INT.tlk",
             "speaker_tag": "hench_tali", "speaker_name": "Tali'Zorah", "speaker_index": 2, "listener": "shepard",
             "occurrences": [{"package": "...", "conversation": "...", "node": {"kind": "entry", "index": 37}}]},
  "context": [
    {"id": "le2.s348110_f", "speaker": "shepard", "text": "...", "string_id": 348110},
    {"speaker": "admiral_raan", "text": "...", "string_id": 348111}
  ],
  "prev": ["le2.s348111_f"],
  "next": ["le2.s348115_f"],
  "scene_tags": ["flotilla", "loyalty"],
  "scene": {"dlc": null, "prefix": "BioD_QuaTlL"},
  "stage_direction": null
}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | line id, see above |
| `speaker` | yes | the character's pack id (`tali`) |
| `text` | yes | the game's subtitle, verbatim. May be empty only when `flags` has `no_subtitle`. |
| `flags` | no | `no_subtitle`, `nonverbal` (grunts, breaths), `bark` (no conversation node) |
| `audio.path` | yes | `audio/<id>.wav`, 24 kHz mono 16-bit PCM WAV |
| `audio.sha256` | yes | hash of that file |
| `audio.sample_rate`, `channels`, `duration_s` | yes | as written |
| `audio.source` | no | where the bytes came from: `container` (AFC file, or package for LE1), `offset`, `size`, `export`, `codec`, the original `sample_rate`, and `sha256` of the raw source bytes |
| `source.game` | yes | `le1`, `le2`, `le3` |
| `source.*` | no | `package` (relative to the install), `conversation`, `node` (`kind` entry/reply, `index`), `string_id`, `gender` (`m`, `f`, `none`), `tlk`, `speaker_tag`, `speaker_name`, `speaker_index` (-1 owner, -2 Shepard), `listener`, `occurrences` |
| `context` | no | the canonical preceding path, oldest first, up to 4 lines, each `{id?, speaker, text, string_id?}` |
| `prev`, `next` | no | ids of immediate predecessor and successor lines in the conversation graph (any speaker) |
| `scene_tags` | no | lowercase tags, like `citadel_party`, `normandy`, `loyalty`, `combat` |
| `scene` | no | `dlc`, `prefix` and anything else the tags were derived from |
| `stage_direction` | no | LE3 voice director's note for this line, verbatim. The label stage treats it as a prior. |

Everything else the harvester produces goes beside `lines.jsonl` in the same `out/`:
- `conversations/<conversation>.json`: the full graph of each conversation she appears in (nodes with speaker, text and string id; edges), so later stages can pick context differently without re-harvesting. It holds game text, so it lives in the corpus like everything else.
- `harvest.json`: the harvester's version, the game build, per-game line counts, and the coverage report (lines with text but no audio, audio with no text, merged duplicates). The runner already records the hashes of the game files the stage lists.

Harvest writes WAV because it's the simplest thing to check and decode. Clean writes FLAC, which is what travels to the worker, so rsync size isn't a reason to change harvest.

### clean: `clips.jsonl` + `audio/` + `embeddings/speaker.npy` + `clean.json`
Written by `jobs/clean.py`.
```json
{"id": "...", "keep": true,
 "audio": {"path": "audio/<id>.flac", "sample_rate": 24000, "channels": 1, "duration_s": 1.31, "rms_dbfs": -24.3},
 "asr_text": "Keelah selai.", "subtitle_wer": 1.0, "speaker_similarity": 0.83,
 "quality": {"p808": 3.6, "sig": 3.9, "bak": 4.1, "ovr": 3.4}, "flags": ["subtitle_mismatch"]}
```
- `audio`: leading and trailing silence trimmed, FLAC at the source rate. Loudness is not normalized, since how loud she is carries emotion; `rms_dbfs` records it.
- `asr_text`, `subtitle_wer`: Whisper's words and their word error rate against the subtitle (punctuation and `{stage notes}` ignored). Nonverbal lines aren't transcribed.
- `transcript`: only present when it differs from the harvest `text`, which is when the line had no subtitle (barks) and Whisper heard words (`asr_only`). Lines with a subtitle keep it, since Whisper misspells names. Later stages use `transcript` if present, else `text`.
- `speaker_similarity`: cosine similarity of the clip's ECAPA speaker embedding to her median voice (clips of 1.5 s or more set the median). The embeddings are in `embeddings/speaker.npy`.
- `quality`: DNSMOS estimates, roughly 1 to 5; `ovr` is overall quality.
- `flags`:
  - `silent`: trimmed to nothing; the only reason for `keep: false` so far.
  - `subtitle_mismatch`: WER above `mismatch_wer`, for subtitles of three or more words.
  - `far_voice`: similarity more than `far_voice_mads` median absolute deviations below her median. Only spoken clips of 1.5 s or more are judged; shorter clips and sighs give speaker embeddings too noisy to mean anything.
  - `asr_only`: the transcript came from Whisper.

Mismatches and far voices are flagged, not dropped, until their thresholds are set from what the flagged clips turn out to be. `clean.json` holds counts, the distributions of WER, similarity and quality, and the 20 clips least like her, for that.

### discover: `captions.jsonl`, `clusters.jsonl`, `prosody.jsonl`
Written by `jobs/discover.py`.
```json
{"id": "...", "caption": "quiet, breathy, slight laugh at the end", "emotions": ["tender", "amused"], "tags": ["laughing"],
 "scores": {"joy": 0.4, "sadness": 0.1}, "heard_without_text": true,
 "with_text": {"caption": "...", "emotions": [], "tags": [], "scores": {}}}
{"id": "...", "cluster": 7, "x": 1.23, "y": -0.4, "text_cluster": 3}
{"id": "...", "duration_s": 1.42, "f0_median_hz": 231.0, "f0_range_st": 6.2, "loudness_db": -24.1, "loudness_range_db": 5.3, "voiced_fraction": 0.71, "words_per_s": 2.8}
```
- `captions.jsonl`: the audio LLM's description of each clip from its sound alone (clips over 30 s are skipped). `tags` come from a fixed delivery list (`laughing`, `sighing`, `crying`, `whispering`, `shouting`, `breathy`, `strained`, `effort`, `pain`, `slurred`, `sarcastic`, `music`). `scores` are its per-axis guesses without the words. They're kept for discovery but not used as a label signal, since open captions mostly echo the words; label asks the audio LLM a narrower question instead. A sample also has `with_text`, the same prompt with the words shown, to measure how much the words sway it.
- `clusters.jsonl`: clusters by sound (emotion2vec embedding plus prosody); `cluster` -1 is unclustered, `x`/`y` a 2D map. `text_cluster` clusters the words, for lines that have them.
- `prosody.jsonl`: pitch, loudness and pace, for matching negatives to positives later.
- `embeddings/emotion.npy` and `embeddings/text.npy`: the emotion2vec vector of each clip's sound and the text-embedding vector of its words (lines with words only), the vectors the clusters were built from. Each is a float32 matrix with one row per clip; the matching `.ids` file lists the clip ids in row order. `discover.json` names the models under `embeddings`. Kept so later steps can train small per-axis probes on reviewed clips and find a clip's nearest neighbours without another GPU pass.

Plus `clusters.md` (what each cluster seems to be, examples, and the audio LLM's emotion vocabulary grouped by meaning) and `discover.json` (counts and the with/without-words comparison). Advisory only; nothing downstream depends on it automatically, except label reading `captions.jsonl` when discover is one of its inputs.

### label: `labels.jsonl` + `axes.json`
```json
{
  "id": "...",
  "scores": {
    "sadness": {"value": 0.71, "confidence": 0.42,
                "signals": {"audio": 0.80, "text": 0.60, "prior": null, "audio_llm": 0.75}},
    "intoxication": {"value": 0.05, "confidence": 0.90, "signals": {"audio": null, "text": 0.05, "prior": 0.0}}
  },
  "tags": ["combat_bark"],
  "audio_classes": {"angry": 0.01, "happy": 0.02, "neutral": 0.31, "sad": 0.64},
  "dimensions": {"arousal": 0.31, "dominance": 0.42, "valence": 0.28}
}
```
`scores` keys are the variable ids from the pack's `states.toml`. Signals are `audio` (acoustic emotion models), `text` (the LLM judge reading line and context), `prior` (scene tags and LE3 stage directions) and `audio_llm` (the audio LLM's 1 to 5 rating of how well the delivery fits the axis, scaled to 0–1). The audio LLM is asked only about a shortlist per axis: the `verify_top` clips the other signals rank highest, plus `verify_low` clips from the lower half as a control. Other clips have `audio_llm: null`; a signal that didn't score an axis is `null` or absent, and new signal names may appear. `tags` are non-emotion labels used as filters: discovery's delivery tags, plus `combat_bark` (harvest flag `bark`), `nonverbal`, and `short` (under a second). `audio_classes` (emotion2vec+ probabilities) and `dimensions` (arousal, dominance, valence) are the raw acoustic outputs, kept for matching and recalibration.

How `jobs/label.py` treats special lines: barks have no subtitle, so they get no `text` signal unless clean transcribed words for them. Nonverbal lines (`{Sighs}`) go to the judge described as a sound rather than words. Both still get the acoustic and audio LLM signals.

`fusion.json`'s `verification` gives, per axis, the audio LLM's mean rating on the shortlist's top and on the low control. If it can hear the axis, the top rates clearly higher (`separation`); if not, its ratings there are noise and the axis should leave `audio_llm_axes`.

Beside them, `fusion.json` reports per axis how many clips each signal scored, how well each pair of signals agrees (Spearman), and how many clips came out low-confidence, for tuning the fusion weights.

`embeddings/emotion.npy` + `.ids` hold the emotion2vec+ vector each clip's `audio_classes` came from, in the same layout as discover's; `fusion.json` names the model under `embeddings`. A small per-axis classifier trained on these and the reviewed clips can give every axis an audio signal, including axes no emotion2vec+ class maps to.

`axes.json` lists the axes in the pack's order with their descriptions, so readers don't need the pack:
```json
[{"id": "joy", "description": "Happiness and excitement"}, {"id": "sadness", "description": "Sadness and homesickness"}]
```

### review: `reviews.jsonl` (human input, not a stage)
The review page (`vi review`) appends to `corpus/<character>/reviews.jsonl`; it never rewrites earlier lines. The last record for an id wins.

```json
{"id": "...", "verdict": "keep", "scores": {"joy": 0.0, "sadness": 0.55}, "model": {"joy": 0.07, "sadness": 0.71},
 "flag": null, "seconds": 8.4, "at": 1791828624}
```

| field | meaning |
|---|---|
| `verdict` | `keep` (use `scores` as the truth for this clip), `flag` (bad clip, drop it), `skip` (no decision yet; treat as unreviewed) |
| `scores` | the reviewer's value for every axis, nudged or not |
| `model` | the label values shown at the time, so corrections stay measurable after labels are regenerated |
| `flag` | reason when flagged, like `wrong_clip`, `music`, `other_voice`, `bad_audio` |
| `seconds` | time spent on the clip |
| `at` | Unix time in seconds |

Stages that use reviews (contrast sets, fusion recalibration) list `corpus:reviews.jsonl` in `files`, so a new review reruns them and nothing upstream.

### What the review page reads
Stage names are fixed by convention: `harvest`, `clean`, `label`. For a corpus folder, the page reads:
- `lines.jsonl` from harvest's current output: `id`, `text`, `speaker`, `source.game`, `source.conversation`, `context`, `scene_tags`, `stage_direction`.
- `clips.jsonl` from clean, if it has run: play `audio.path` from there (FLAC) and skip `keep: false` clips. Otherwise play harvest's WAV.
- `labels.jsonl` and `axes.json` from label.
- `reviews.jsonl` at the corpus root.

Rust code can call `dataset::runner::current_out(corpus, stage)` for those folders.

### contrast: `<target>/<axis>/positives.jsonl`, `negatives.jsonl`, `pairs.jsonl`
Records are `{"id": ...}` plus, for pairs, `{"id": ..., "neutral_text": ...}`.

### extract, evaluate
Vectors go to `vectors/<axis>.{gguf,safetensors}` and the report to `report.md` plus `samples/`. The runner already keys these by the model through the stage settings; copying the chosen vectors into the pack's vector store is a separate step.

## Pipeline file

```toml
corpus = "../../corpus/tali"    # relative to this file

[stage.harvest]
command = ["vi", "extract", "--character", "tali", "/games/mass-effect-le"]
version = 1                     # bump when the extractor's behavior changes
files = ["/games/mass-effect-le/Game"]

[stage.clean]
command = ["python3", "../../jobs/clean.py"]
inputs = ["harvest"]
files = ["../../jobs/clean.py"]
settings = { min_seconds = 1.0, max_seconds = 20.0, whisper = "large-v3" }

[stage.label]
command = ["../../jobs/on-worker.sh", "--put", "states.toml", "python3", "jobs/label.py"]
inputs = ["clean"]
files = ["../../jobs/label.py", "states.toml"]
settings = { judge = "gemma-4-12b", audio = "emotion2vec_plus_large" }
```

- `command`: the program and arguments. Relative paths starting with `./` or `../` resolve against the pipeline file's folder; `vi` means the running `vi` binary; anything else is looked up on `PATH`. The job runs with the pipeline file's folder as its working directory.
- `inputs`: upstream stages. They run first.
- `files`: files or folders whose content counts as input: job scripts, the pack's `states.toml`, `reviews.jsonl`, the game folder. Relative to the pipeline file. A leading `corpus:` makes a path relative to the corpus instead, like `corpus:reviews.jsonl`; such a file may be missing (it counts as `missing` until it appears). Any other missing file is an error.
- `settings`: any TOML table. Passed to the job and part of the key.
- `version`: an integer to bump when code not listed in `files` (like `vi extract` itself) changes behavior.

## What a job sees
Environment variables:

| variable | value |
|---|---|
| `VI_STAGE` | stage name |
| `VI_OUT` | empty folder to write outputs into |
| `VI_INPUT_<NAME>` | each upstream stage's `out/` folder, name uppercased with `-` as `_` (read-only by convention) |
| `VI_SETTINGS` | path to a JSON file holding `settings` |
| `VI_CHECKPOINT` | a folder for work to resume from: it keeps whatever earlier failed runs of the same key left there, and is deleted when the run succeeds |
| `VI_CORPUS` | the corpus folder |
| `VI_BIN` | the running `vi` binary |
| `VI_CHECKPOINT` | a folder for saving progress; see below |

A job exits 0 on success. Anything else marks the run failed: nothing becomes current, and the partial output stays in `stages/<stage>/.tmp-<key>/` for inspection until the next attempt or `gc`. A job that saves per-item results in `VI_CHECKPOINT` (the GPU jobs use `vi_job.Checkpoint`) picks up where it stopped when rerun; changing the stage's settings, inputs or files gives a new key and starts over. `gc` deletes checkpoints too.

A GPU job is the same contract wrapped: `jobs/on-worker.sh` rsyncs the input folders, settings, `jobs/` and any `--put` files to the worker, runs the command there, copies `VI_CHECKPOINT` home every minute while it runs, rsyncs `VI_OUT` back, and deletes the worker's copy. The runner doesn't know or care where the work ran. Details in `docs/cloud-worker.md`.

## How caching works
Each run has a key:

```
key = sha256(canonical JSON of {
  stage, version, command, settings,
  inputs: {upstream stage: its output hash},
  files:  {path: content hash}
})
```

The output hash is the SHA-256 of the sorted `"<sha256>  <path>\n"` listing of every file in `out/`, the same text `sha256sum` prints. Two consequences:
- Changing a setting, a job script, a review, or an upstream output changes the key, so that stage and everything after it reruns.
- If an upstream stage reruns but produces byte-identical output (a refactor, a log change), downstream keys don't change and nothing else reruns.

If `stages/<stage>/<key>/` exists, the stage is cached and skipped. Old keys stay on disk, so reverting a setting is instant. `vi dataset gc` deletes every run that isn't `current` and every object no longer used.

Files are hashed once and remembered by path, size and modification time in `corpus/.hash-cache.json`, so listing the 100 GB game folder only costs a full read the first time.

Outputs are moved into `objects/<sha256>` and hard-linked back into `out/`, so a clip that passes through clean unchanged, or a rerun that produces the same files, takes no extra disk.

`manifest.json`:
```json
{"format": 1, "stage": "clean", "key": "...", "output_hash": "...", "version": 1,
 "command": ["python3", "../../jobs/clean.py"], "settings": {...},
 "inputs": {"harvest": "<output hash>"}, "files": {"../../jobs/clean.py": "<sha256>"},
 "outputs": {"clips.jsonl": "<sha256>", "audio/le2...flac": "<sha256>"},
 "started": "2026-10-12T19:00:00Z", "finished": "2026-10-12T19:04:12Z"}
```

## Commands
- `vi dataset run <pipeline.toml> [stage] [--force]`: runs that stage (default: all) and what it needs, skipping cached ones. `--force` reruns the named stage even if cached.
- `vi dataset status <pipeline.toml>`: each stage with `cached`, `stale` (will run) or `waiting` (an upstream stage runs first, so its key isn't known yet).
- `vi dataset gc <pipeline.toml>`: frees disk as above.
- `vi corpus check <harvest out folder>`: validates `lines.jsonl` against this document: required fields, unique ids, audio present with matching hash and WAV header.
- `vi corpus synth <folder> [--lines N]`: writes a synthetic harvest output (tones, invented lines) that passes `check`. For tests and for building the review page before real clips exist.
