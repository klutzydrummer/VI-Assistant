# Label review page

Status: built (`crates/review`, `vi review`), tested on synthetic clips only. Reads corpus format v1 (`corpus-format.md`).

Stage 5 of [control-vector-datasets.md](control-vector-datasets.md). You listen to a clip, see the label stage's scores, and keep, nudge, flag or skip it. About an hour of your time should cover the clips that matter.

## Running it

```
vi review <corpus-folder>            # serves http://127.0.0.1:8737/
vi review --demo <empty-folder>      # writes a synthetic corpus (harvest, clean, label) first
```

It binds to localhost only and the page loads nothing from the internet, so game audio stays on your machine. Every verdict is written as you go; stop with Ctrl+C and rerun to pick up where you left off.

## Keys

| Key | Does |
|---|---|
| Space | replay |
| 1–9, ↑ ↓ | pick an axis |
| ← → | nudge by 0.05 (Shift: 0.2) |
| 0 | reset the axis to the model's score |
| Enter | keep (with any nudges) |
| F | flag as a bad clip, with the reason picked in the dropdown |
| S | skip for now (goes to the back of the queue) |
| B | back to the previous clip, to change its verdict |

Each axis shows the model's score as a tick, its confidence, and each signal's own score (audio, text, prior). The shaded end of the track marks where a line becomes a positive for that axis's vector. The header tracks time against the hour and estimates how many more clips fit at your pace; time on one clip is capped at two minutes so breaks don't count.

## Queue order

First comes the **calibration sample**: 150 labelled clips picked at random, so the accuracy the calibrate stage measures on them isn't biased toward hard cases. "Random" is reproducible: the clips whose lowercase hex SHA-256 of their id sorts first, in that order (`queue::calibration`), so the calibrate stage can recompute the sample from ids alone. These clips say "calibration sample" in their reason.

After that, a clip's priority is its highest, over axes, of **uncertainty × stake**:
- **uncertainty**: the larger of (1 − confidence) and the spread between its signals, so a confident fusion of disagreeing signals still counts as unsure.
- **stake**: 0.4 + 0.6 × max(score, closeness to the positive threshold 0.6). Likely positives move a vector most, so they rank higher; neutral lines still count because they become the negatives.

Each clip shows which axis put it in the queue and why. Kept and flagged clips leave the queue; skipped ones go to the back.

## Data

The page follows "What the review page reads" in `corpus-format.md`; `crates/review/src/corpus.rs` is the only code that knows the layout.
- **harvest** (`stages/harvest/current`): `lines.jsonl` for the subtitle, speaker, `source.game`, `source.conversation`, context, `scene_tags` and `stage_direction`.
- **clean**, if it has run: `clips.jsonl`. Clips with `keep: false` are left out (the header counts them) and audio plays from clean's output. Without clean, harvest's WAV plays. A clean transcript that differs from the subtitle is shown as "Spoken".
- **label**: `labels.jsonl` for scores and non-emotion `tags`, and `axes.json` for axis order and descriptions (hover an axis name). Null signals are ignored.
- **reviews**: appended to `<corpus>/reviews.jsonl`, the last record per id wins.

A review records the verdict, your scores for every axis, the model scores you were shown, a `flag` reason (`wrong_clip`, `music`, `other_voice`, `bad_audio`, `other`), seconds spent and a timestamp:

```json
{"id":"le2.s348112_f","verdict":"keep","scores":{"joy":0.0,...},"model":{"joy":0.07,...},"flag":null,"seconds":8.4,"at":1791828624}
```

Contrast sets and fusion recalibration list `corpus:reviews.jsonl` as an input, so a new review reruns them and nothing upstream. The page shows the running bias per axis ("playfulness: +0.12 on average"), a first look at what recalibration will find; the recalibration itself belongs to the label stage.

## Purity mode

```
vi review --purity <corpus-folder>          # needs the cluster stage's output
vi review --purity --demo <empty-folder>    # synthetic corpus with four clusters by pitch
```

The corpus pipeline's listening step ([corpus-pipeline.md](corpus-pipeline.md) step 6). One cluster at a time, it plays a clip and asks whether it fits: <kbd>Y</kbd> or <kbd>N</kbd>, <kbd>Space</kbd> to replay. Only after you answer does it show the line, its context and following lines, scene tags, stage direction, prosody and the top three EIV heads (as z-scores against the corpus), then <kbd>Enter</kbd> plays the next.

About one clip in five is a foil from another cluster and one in eight a hidden repeat; the page doesn't say which, before or after, so your answers stay honest measures of purity and consistency. When the SPRT reaches a verdict (or the members run out), the cluster's purity and its audits from `cluster.json` appear (size, Jaccard, games, enriched tags, and flags for low purity, a session flag, "topic, not tone?", below the minimum size, nuisance), and you name, merge, split or reject it. "Decide this cluster now" ends listening early.

Answers and decisions go to `purity.jsonl` at the corpus root as you go (format in `corpus-format.md`). The Gemma text name suggestion from the spec isn't built yet.

## Not done
- Audio is served whole without range requests, which is fine for clips under 20 seconds.
- `current_out` is a few lines in `corpus.rs`; once `crates/dataset` merges, it can call `dataset::runner::current_out` instead.
