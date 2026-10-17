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
| `source.*` | no | `package` (relative to the install), `conversation`, `node` (`kind` entry/reply, `index`), `string_id`, `gender` (`m`, `f`, `none`), `tlk`, `speaker_tag`, `speaker_name`, `speaker_index` (-1 owner, -2 Shepard), `listener`, `conversation_kind` (`one_liner`: one line; `dialogue_choices`: Shepard picks a reply somewhere; `exchange`: several lines, no choice), `line_kind` (`one_liner` if nothing comes before or after this line, else its conversation's kind), `bark_category` (LE1 barks: the raw category code from the stream name), `occurrences` |
| `context` | no | the canonical preceding path, oldest first, up to 4 lines, each `{id?, speaker, text, string_id?}` |
| `prev`, `next` | no | ids of immediate predecessor and successor lines in the conversation graph (any speaker) |
| `following` | no | every line directly after this one, all branches, one step: `{id, speaker, text, string_id}` each |
| `scene_tags` | no | lowercase tags for where and in what situation the line is spoken, like `citadel_party`, `normandy`, `loyalty`, `combat`; vocabulary and sources in `crates/harvest/src/scenes.toml`. Also structural facts from the conversation graph: `one_liner` / `exchange` / `dialogue_choices` (from `source.line_kind`) and `to_shepard` / `to_squad` / `to_npc` (from `source.listener`) |
| `scene` | no | `dlc`, `prefix` and anything else the tags were derived from |
| `stage_direction` | no | LE3 voice director's note for this line, verbatim. The label stage treats it as a prior. |

Everything else the harvester produces goes beside `lines.jsonl` in the same `out/`:
- `conversations/<conversation>.json`: the full graph of each conversation she appears in (nodes with speaker, text and string id; edges), so later stages can pick context differently without re-harvesting. It holds game text, so it lives in the corpus like everything else.
- `harvest.json`: the harvester's version, the game build, per-game line counts, and the coverage report (lines with text but no audio, audio with no text, merged duplicates). The runner already records the hashes of the game files the stage lists.

Harvest writes WAV because it's the simplest thing to check and decode. Clean writes FLAC, which is what travels to the worker, so rsync size isn't a reason to change harvest.

### prep: `prep.jsonl` + `audio16k/` + `prep.json`
Written by `jobs/prep.py`, at home on the CPU (numpy, scipy, soundfile). One trim and one level for every clip, so every later stage hears the same clip ([corpus-pipeline.md](corpus-pipeline.md), step 1).
```json
{"id": "le2.s348112_f", "trim": {"start_s": 0.212, "end_s": 1.632}, "duration_s": 1.42, "trimmed_s": 0.31,
 "level": {"method": "bs1770", "db": -27.4, "peak_dbfs": -6.1}, "gain_db": 4.4, "gain_capped": false,
 "noise_floor": {"level_db": -71.2, "seconds": 0.26, "spectrum": [-88.1, -90.4]},
 "audio16k": {"path": "audio16k/le2.s348112_f.flac", "sample_rate": 16000}, "flags": []}
```
- `trim`: where the kept clip starts and ends in harvest's WAV, in seconds. Frames more than `top_db` below the loudest frame count as silence (the same rule clean used before); `pad_s` of it is kept at each end. Clean cuts its FLAC at these bounds.
- `noise_floor`: the level (dBFS) and the per-band power spectrum (dB, bands in `prep.json`'s `band_edges_hz`, 50 Hz to 7.6 kHz) of the silence at the clip's two edges, measured before trimming. It is recording chain with no voice in it: a nuisance covariate for `confounds`. The edge silence is found with uncentred 20 ms frames (whole frames more than `top_db` below the loudest, from each end up to the first loud one), not the trim's 85 ms centred frames, which smear speech into the game's tightly cut edges. Pauses mid-line are never used: they can hold breath or reverb. `null` (flag `no_noise_floor`) when there is under 50 ms of edge silence; the share is reported in `prep.json`, not checked, and confounds fills missing floors with the median.
- `level`: the kept clip's loudness before the gain: ITU-R BS.1770 integrated loudness (`bs1770`, LUFS), or RMS dBFS (`rms`) for clips under `rms_below_s`, where the 400 ms gating is unreliable. A covariate, since how loud she is carries vocal effort.
- `gain_db`: the one linear gain to `target_lufs`, capped so the peak stays under `max_peak_dbfs` (`gain_capped`). Never dynamic compression.
- `audio16k`: the trimmed, levelled clip at 16 kHz as 16-bit FLAC, for the models only. Harvest's 24 kHz WAVs are untouched. Absent for `silent` clips (trimmed under 0.2 s), which later stages skip.

- `eq_matched`: whether the clip went through its game's EQ correction (setting `eq_match`, off by default). The correction is fitted first on the comparable clips (`eq_match.fit`: `confounds.Meta.comparable` with its seed; clean's flags aren't known yet, so its exclusions are left out) and applied to the whole clip before the level, from 100 Hz to 9 kHz, capped at ±12 dB, with each octave's average offset removed. Trim bounds and the noise floor are always measured on the audio as shipped.

`prep.json` holds the settings, band edges, package versions, flag counts and the distributions of gain, noise floor and trimmed silence. With `eq_match` on, it also holds `eq_match`: the clips each game was fitted on, the rule, and each game's correction in dB at the third-octave centres.

Jobs with prep as an input get each clip's 16 kHz copy and prep record from `vi_job.load_clips()` (`clip.audio16k`, `clip.prep`).

### clean: `clips.jsonl` + `audio/` + `embeddings/speaker.npy` + `clean.json`
Written by `jobs/clean.py`. With prep as an input, it cuts at prep's bounds and its models hear prep's 16 kHz copy; without it, it trims as described below.
```json
{"id": "...", "keep": true,
 "audio": {"path": "audio/<id>.flac", "sample_rate": 24000, "channels": 1, "duration_s": 1.31, "rms_dbfs": -24.3},
 "asr_text": "Keelah selai.", "subtitle_wer": 1.0, "speaker_similarity": 0.83,
 "quality": {"p808": 3.6, "sig": 3.9, "bak": 4.1, "ovr": 3.4}, "flags": ["subtitle_mismatch"]}
```
- `audio`: leading and trailing silence trimmed, FLAC at the source rate. Loudness is not normalized, since how loud she is carries emotion; `rms_dbfs` records it.
- `asr_text`, `subtitle_wer`: Whisper's words and their word error rate against the subtitle (punctuation and `{stage notes}` ignored). Nonverbal lines and lines without a subtitle (`no_subtitle`, the LE1 barks) aren't transcribed. Whisper is greedy and capped at `asr_max_tokens`; clips over 29.5 s, too long for a batch, are transcribed one at a time in 30 s pieces.
- `transcript`: only present when it differs from the harvest `text`, which is when the line had an empty subtitle without the `no_subtitle` flag and Whisper heard words (`asr_only`). Lines with a subtitle keep it, since Whisper misspells names. Later stages use `transcript` if present, else `text`.
- `speaker_similarity`: cosine similarity of the clip's ECAPA speaker embedding to her median voice (spoken clips of 1.5 s or more set the median). Each game with at least 20 such clips gets its own median and threshold, since the embedding hears each game's recording and mix; smaller groups use the all-games one. `clean.json`'s `speaker_baselines` lists them. The embeddings are in `embeddings/speaker.npy`.
- `quality`: DNSMOS estimates, roughly 1 to 5; `ovr` is overall quality. A reported covariate, never a gate.
- `flags`:
  - `silent`: trimmed to nothing; the only reason for `keep: false` so far.
  - `subtitle_mismatch`: WER above `mismatch_wer`, for subtitles of three or more words.
  - `far_voice`: similarity more than `far_voice_mads` median absolute deviations below her median. Only spoken clips of 1.5 s or more are judged; shorter clips and sighs give speaker embeddings too noisy to mean anything.
  - `asr_only`: the transcript came from Whisper.

`clean.json`'s `speaker_by_length` gives the similarity distribution and `far_voice` share per length band of the judged clips (1.5–3 s, 3–6 s, 6 s and up), to check the threshold treats short and long clips alike.

Mismatches and far voices are flagged, not dropped, until their thresholds are set from what the flagged clips turn out to be. `clean.json` holds counts, the distributions of WER, similarity and quality, and the 20 clips least like her, for that.

### embed: `embeddings/` + `scores.jsonl` + `prosody.jsonl` + `embed.json`
Written by `jobs/embed.py` on the rented GPU from prep's 16 kHz copies ([corpus-pipeline.md](corpus-pipeline.md), step 3). The one paid job of the corpus pipeline; it runs only on Brandon's ok for that run.
- `embeddings/<name>.npy` + `<name>.ids`: float16, one row per clip, ids in row order (`numpy.load`). Compute downstream in float32 or float64. Names:
  - `wavlm_L06` … `wavlm_L18`: WavLM-large hidden states, mean over frames, one file per layer (`hidden_states[k]`, where 0 is the input to the first transformer layer)
  - `emotion2vec_plus_large`, `emotion2vec_base`: FunASR utterance embeddings
  - `bude_whisper`: the BUD-E-Whisper encoder output averaged over the frames that hold audio, the first ceil(duration × 50) of 1,500. Whisper pads every clip to 30 s; pooling the padding too would make the vector mostly encode length.
  - `minilm`: text embeddings of lines with words (not `nonverbal` or `no_subtitle`), so fewer rows
- `crops/<name>.npy` + `<name>.ids`: the same embeddings for the centred 0.5, 1, 1.5 and 2 s of up to 300 non-bark clips of 4 s or more, ids `<id>@<seconds>`, for the shortest-usable-clip test.
- `scores.jsonl`, by name:
  ```json
  {"id": "...", "eiv": {"Anger": 0.41, "Teasing": 1.2}, "eiv_quality": {"score_overall_quality": 2.6},
   "avd": {"arousal": 0.52, "dominance": 0.48, "valence": 0.61}, "emotion2vec_plus_large_classes": {"angry": 0.02}}
  ```
  `eiv`: every Empathic-Insight-Voice-Small head (raw regression outputs, on LAION's scales), run on the full padded encoder output as the heads were trained. `eiv_quality`: the Plus quality heads except content enjoyment (mean+min+max+std pooled, as trained). Backup labelling and naming help only.
- `prosody.jsonl`: `duration_s`, `f0_median_hz`, `f0_range_st`, `loudness_db`, `loudness_range_db`, `voiced_fraction`, `words_per_s`, as discover wrote them.
- `embed.json`: per model the repo, the pinned revision and the commit it resolved to; the layers; the score names in order; how each embedding was pooled; package versions; run time.

### confounds: `confounds.json` + `cleaned.npy` + `cleaned.ids`
Written by `jobs/confounds.py` at home (numpy, scikit-learn), following [corpus-pipeline.md](corpus-pipeline.md) step 4. Inputs: harvest (game, conversation, text, tags, flags, audio hash), prep (noise floor, level, duration), clean (flags) and embed.
- `cleaned.npy` (float32) + `cleaned.ids`: the chosen candidate after PCA, per-game CORAL and LEACE, one row per clip, every clip (flagged ones too). Absent when nothing was chosen.
- `cleaned_crops.npy` + `.ids`: embed's crops of the chosen candidate through the same fitted transform.
- `confounds.json`:
  - `status`: `chosen` or `stopped`, with `chosen` (the candidate's name) or `stop_reason`
  - `comparable`: the comparable-clip rule and its size per game. Without structural tags in the harvest the rule says it fell back to every non-bark line.
  - `split`: conversations in the selection (30%) and report (70%) splits
  - `excluded` (singing, subtitle_mismatch, far_voice: cleaned and kept, never fitted), `duplicate_takes` (byte-identical, counted once)
  - `candidates.<name>`: `transform` (dimensions, variance kept, clips the eraser fitted, whether log duration had to be erased), `selection` and, for the candidates tried there (`shortlist`, in order), `report`, each with `before` and `after` probe tables (game k-NN balanced accuracy, text length, log duration and its square, noise floor, level, A/V/D as R², bark, stage direction and each tag of 20+ clips as AUC, text content as R² on MiniLM's top 5 components), `game` (k-NN and RBF-SVM balanced accuracy, the transform refitted in each fold, with the upper 95% bound: binomial on the selection split, conversation bootstrap on the report split; with permutations, game shuffled across conversations and the transform refitted the same way, the null's median and 95th percentile and a p-value), `paired` (how often a take's nearest neighbour is its own other take, against chance) and `tests` (each acceptance test, pass or fail)
  - `settings` and package versions

The acceptance limits are the stage settings, fixed before results. Folds are grouped by conversation everywhere.

### cluster: `clusters.jsonl` + `cluster.json`
Written by `jobs/cluster.py` at home (numpy, scipy, scikit-learn) from confounds' cleaned space, following [corpus-pipeline.md](corpus-pipeline.md) step 5. Refuses to run when confounds stopped.
```json
{"id": "...", "cluster": "c2", "responsibilities": {"c0": 0.01, "c1": 0.02, "c2": 0.97},
 "scores": {"c0": -8.1, "c1": -5.3, "c2": 4.2}, "formed": true, "hdbscan": 3}
```
- One record per clip, every clip. `formed: false` for clips that didn't form the clusters (flagged, duplicate takes, or shorter than the shortest usable length); they still get memberships. A duplicate take gets its twin's.
- `responsibilities`: the GMM's, summed over each class's components (they saturate near 0 or 1). `scores`: the continuous per-class score, log p(x | class) − log p(x | rest), for intensity when mixing. `hdbscan`: the second opinion's label (−1 noise, null if not formed).
- Class ids `c0`, `c1`, … go by size, largest first. Names come later, from the purity review.

`cluster.json`: `status` (`ok` or `stopped` with `stop_reason`: fewer than 2 classes at Jaccard ≥ `keep_jaccard`, or HDBSCAN noise over `max_noise`), `shortest_usable` (retrieval per crop length, the minimum length, paired-take agreement for short and long clips), `pca` (dimensions by parallel analysis), `gmm` (held-out likelihood, BIC and ICL for every k and covariance type tried; the chosen one), `merge` (the entropy path), `hdbscan` (clusters, noise share, agreement with the classes), `min_class_size` (from split-half cosines), and per class: size, components, Jaccard and its band (`class`, `pattern`, `not trusted`), nuisance AUC and `rejected_for_nuisance`, audio and text AUC with `topic_not_tone`, counts by game, barks, tag enrichment, the session check (effective conversations against a same-size baseline, top-two share, Jaccard without the biggest conversation) with `session_flag` for Brandon's call, split-half cosines and `below_min_size`; plus the sensitivity run without LE1 barks.

### review, purity mode: `purity.jsonl` (human input, not a stage)
`vi review --purity <corpus>` appends to `corpus/<character>/purity.jsonl`, following [corpus-pipeline.md](corpus-pipeline.md) step 6. It's its own file rather than records in `reviews.jsonl` (the earlier proposal), so the label review and the purity review never rerun each other's consumers. Three kinds of record:
```json
{"kind": "answer", "id": "le2.s4_f", "cluster": "c2", "fits": true, "foil": false, "repeat": false, "seconds": 3.1, "at": 1791979143}
{"kind": "decision", "cluster": "c2", "action": "name", "name": "teasing", "into": null, "note": null, "at": 1791979150}
{"kind": "skip", "id": "le2.s4_g", "cluster": "c2", "at": 1791979160}
```
- `answer`: one per clip played. `foil` is a clip from another cluster (responsibility for this one ≤ 0.1) at a similar length, every fifth clip; `repeat` is a hidden replay of a member already answered in this cluster, every eighth clip until 50 exist. Members come stratified: round-robin over scenes (the first non-structural scene tag), then over conversations, each in SHA-256 order. Only `formed` clips with audio are played.
- `decision`: `action` is `name` (with `name`), `merge` (with `into`, another cluster), `split` or `reject`; `note` is free text. A cluster is done once it has a decision. With no clusters picked on the page, they're reviewed in `cluster.json` order (largest first), one at a time; with some picked, clips rotate between the picked ones (one ready to decide first, else the one with fewest answers). The pick isn't saved.
- `skip`: a clip skipped unanswered, member, foil or repeat alike. It goes to the back of its queue (oldest skip first) and comes back later; a skipped foil or repeat slot is filled by another foil or repeat. Skips aren't answers. They're counted per cluster (`skips` in the classes purity summary), because skipping only the hard clips would inflate purity.
- The SPRT (p₀ 0.5 against p₁ 0.8, α = β = 0.05) counts member answers only; a cluster with no verdict after its members run out, or after 40, is `undecided`. Purity is the members' yes share, corrected for the false-yes rate on all foils so far: (y − f) / (1 − f). Consistency is Cohen's kappa of first answers against repeats. All of these are recomputed from the file, never stored, so nothing can drift.
- The server picks every clip from this file alone, so a restart resumes exactly. The page never sees a clip's id, line or audits before the answer (the id names the game).

The classes stage lists `corpus:purity.jsonl` in `files`.

### classes: `classes.json` + `memberships.jsonl` + `eiv_labels.jsonl` + `report.md`
Written by `jobs/classes.py` at home (numpy, scipy) from cluster, embed, confounds and `corpus:purity.jsonl`. The corpus pipeline's final output. Refuses to run when cluster stopped; runs without a review (every cluster `unreviewed`, status `incomplete`).
```json
{"id": "...", "class": "teasing", "responsibilities": {"calm": 0.02, "teasing": 0.97}, "scores": {"calm": -6.2, "teasing": 4.4}, "formed": true}
```
- `memberships.jsonl`: one record per clip, every clip. `class` is the top class only when its responsibility is at least `min_membership` (0.5), else null: a clip of a rejected cluster belongs weakly to everything, not to the least bad kept class. Keys are class keys: the slug of the name Brandon gave, or the cluster ids (`c4`, `c1+c4` for an unnamed merge, `c3.0` for a split part). `responsibilities` sum the GMM posteriors over the class's components; `scores` are log p(x | class) − log p(x | not class) for intensity when mixing. Both come from cluster's `components.npy`, so merges and splits are exact, not refitted. A rejected cluster's components stay in "not class" and nothing is renormalised.
- `classes.json`: `status` (`ok` when every class is named and at least 2 are; else `incomplete`), and per class: `key`, `name`, `status` (`named`, `split_part`, `unreviewed`, `split_single_component`), `clusters`, `components`, `size` (formed clips assigned to it) and `size_all_clips`, `stability` (the cluster's Jaccard and band; `not re-measured` for merges and splits, with each cluster's Jaccard), `purity` (members heard, observed, corrected, SPRT verdicts), `flags` carried from cluster. Also `rejected`, `unassigned` (clips in no class), the purity summary (per cluster and overall: false-yes rate, kappa, repeat pairs) and `failed`, the list of checks that failed.
- A split separates a cluster back into its GMM components; a single-component cluster can't be split this way and is kept whole and reported. Split parts are unnamed until another review.
- `eiv_labels.jsonl`: the Empathic-Insight-Voice backup labelling, `{id, eiv, eiv_z, quality, top}` (z-scores over the corpus, top three heads). Kept apart; never an automatic fallback.
- `report.md`: every acceptance number (confounds' probes before and after, the game bound, paired takes; cluster's shortest usable clip, GMM, HDBSCAN, per-cluster Jaccard and audits; purity, false-yes rate, kappa; the class table) with the failed checks first.

cluster also writes `components.npy` (float32, clips × GMM components, log w_c + log N(x | c), a duplicate take carrying its twin's) with `components.ids`, and the component weights in `cluster.json` under `gmm.weights`.

### context: `context.jsonl` + `context.json`
Written by `jobs/context.py` at home (standard library) from harvest, for the seeded moods ([corpus-pipeline.md](corpus-pipeline.md), "Seeded moods"). One record per line with words (not `nonverbal` or `no_subtitle`):
```json
{"id": "le2.s1_f", "game": "le2", "conversation": "...", "text": "Thank you, Shepard.",
 "before": [{"speaker": "Shepard", "text": "..."}], "after": [{"speaker": "Shepard", "text": "..."}],
 "listener": "Shepard", "scene_tags": ["normandy"], "stage_direction": null, "conversation_kind": "dialogue_choices",
 "bark": false, "prompt": "Scene: normandy\nBefore:\nShepard: ...\n>>> Tali (to Shepard): Thank you, Shepard.\nAfter:\nShepard: ..."}
```
- `before`: the last `before` (3) lines of harvest's `context`; `after`: the first `after` (2) of `following`, all branches. Speaker ids are written as names.
- `scene_tags`: without the structural tags. Harvest's `text` is used, not clean's transcript, so a clean rerun doesn't rerun textscore.
- `prompt`: what every text model reads; her line is marked `>>>`.
- `context.json`: counts (lines, left out, with lines before and after, with a director's note) and the settings.

### textscore: `scores/` + `llm/` + `embeddings/` + `textscore.json`
Written by `jobs/textscore.py` on the rented GPU from context and `characters/tali/moods.toml`. Manual; runs only on Brandon's ok for that run. Scores nominate lines for Brandon to hear; they are never labels.
- `scores/<model>.jsonl`, `{"id", "scores": {...}}`, one per context line:
  - `goemotions`: the 28 GoEmotions probabilities (sigmoid) of her line alone
  - `nli`: entailment probability of "In the marked line, Tali is <description>." given the prompt, per mood id and per knob level (`energy:urgent`)
  - `embedder`: cosine of the prompt's Qwen3-Embedding vector to each mood's and knob level's description, as a query
- `llm/<name>.jsonl` (`gemma`, `qwen`): `{"id", "samples": [answer or null, ...], "raw_failed": [first unparseable answer]}`, an answer being `{"moods": [{"mood", "strength" 1–3, "reason"}], "<knob>": "<level>", ...}`, constrained by a JSON schema to moods.toml's ids; `samples` (2) per line at `temperature` 0.7, thinking off. `llm/<name>.prompt.txt` is the system prompt as sent.
- `embeddings/qwen3_line.npy`, `qwen3_context.npy` (+ `.ids`): float16 vectors of her line and of the prompt; `qwen3_moods.npy` with the label keys as ids.
- `textscore.json`: lines, labels, the moods and knobs as read, per model the repo, pinned revision and resolved commit, per LLM the parse failures and the share of lines whose two samples name the same moods, the NLI template and embedding instruction, sampling, packages, run time. `checks.json`: no line missed per model, under 2% unparseable samples per LLM.

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

### calibrate: `labels.jsonl` + `axes.json` + `calibration.json`
Written by `jobs/calibrate.py` at home (CPU; numpy and scikit-learn through nix-shell), from label's output and `corpus:reviews.jsonl`.
- `labels.jsonl`: label's schema, refitted.
  - Each axis gains a `probe` signal: a ridge regression from the clip's emotion2vec+ embedding to the reviewed scores, out-of-fold on reviewed clips.
  - Each axis is refused with weights fitted to the reviews.
  - Reviewed clips (`reviewed: true`) take the reviewed value with confidence 1 and a `review` signal.
  - Flagged clips carry `flagged: <reason>`.

  Later stages (contrast sets) read this instead of label's.
- `calibration.json`: per axis, the fitted weights and each signal's and the fused score's Spearman and mean absolute error against the reviews on the calibration sample, before and after calibration. The sample is the random clips the review page shows first: the 150 ids whose lowercase hex SHA-256 sorts first. Weights are fit on reviewed clips outside the sample when there are at least `min_reviews` of them, so the sample stays an honest test.

A signal's weight on an axis is its Spearman with the reviews, squared, scaled so the best signal is 1, and 0 below 0.1. An axis with fewer than `min_reviews` reviews keeps the defaults.

### contrast: `<target>/<axis>/positives.jsonl`, `negatives.jsonl`, `pairs.jsonl`
Records are `{"id": ...}` plus, for pairs, `{"id": ..., "neutral_text": ...}`.

### extract, evaluate
Vectors go to `vectors/<axis>.{gguf,safetensors}` and the report to `report.md` plus `samples/`. The runner already keys these by the model through the stage settings; copying the chosen vectors into the pack's vector store is a separate step.

### Corpus pipeline stages
prep, clean, embed, confounds, cluster, the purity review and classes (with context and textscore for the seeded moods) replace discover, label and calibrate in Tali's pipeline (their jobs stay in `jobs/`, unused). Each has its section above. Outputs stay in stage folders, never in `lines.jsonl`.

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
- `manual`: `true` for a stage that runs only when named (`vi dataset run <pipeline> <stage>`), never in a whole-pipeline run: experiments that cost GPU time, like `steer`. A stage that isn't manual can't take a manual one as an input.

### Acceptance checks and confound reports (clean, discover, label)
Each of these stages writes `checks.json`: named checks with the measured value, the limit and whether it passed (`jobs/checks.py`). Examples:
- clean: share of `far_voice`, median subtitle WER
- discover: share of unparsed captions, share of captions carrying the most common tag, how well the emotion embedding predicts clip length
- label: the judge's unparsed share, its share of all-zero answers, verification parse rates, and the fused score's confounds

With the stage setting `checks = "fail"` (the smoke pipeline) a failed check fails the stage; with `"warn"` (the default) it's only logged. Replayed on the saved smoke-test outputs, the checks fail every run that had a known problem and pass the fixed ones.

Label's `fusion.json` also has `confounds`, per axis and signal: Spearman with log clip length, and the share of variance explained by game and by bark vs conversation. It also has `data`, per axis: how many non-bark clips the fused score puts at 0.6+ and 0.4+, overall and by game, so an axis without enough candidates shows up before contrast sets.

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
