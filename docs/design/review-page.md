# Label review page

Status: built (`crates/review`, `vi review`), tested on synthetic clips only. Reads a provisional corpus layout until `corpus-format.md` lands.

Stage 5 of [control-vector-datasets.md](control-vector-datasets.md). You listen to a clip, see the label stage's scores, and keep, nudge, flag or skip it. About an hour of your time should cover the clips that matter.

## Running it

```
vi review <corpus-folder>            # serves http://127.0.0.1:8737/
vi review --demo <empty-folder>      # writes 120 synthetic clips first, then serves
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

A clip's priority is its highest, over axes, of **uncertainty × stake**:
- **uncertainty**: the larger of (1 − confidence) and the spread between its signals, so a confident fusion of disagreeing signals still counts as unsure.
- **stake**: 0.4 + 0.6 × max(score, closeness to the positive threshold 0.6). Likely positives move a vector most, so they rank higher; neutral lines still count because they become the negatives.

Each clip shows which axis put it in the queue and why. Kept and flagged clips leave the queue; skipped ones go to the back.

## Data

Provisional, isolated in `crates/review/src/corpus.rs` so the real corpus format only touches that file:

```
<corpus>/lines.jsonl    {"id","audio","text","speaker","game","conversation","context":[{"speaker","text"}],"tags":[]}
<corpus>/labels.jsonl   {"id","scores":{"<axis>":{"value","confidence","signals":{"audio","text","prior",...}}}}
<corpus>/reviews.jsonl  written by the page, append-only, latest per id wins
```

Axes are whatever keys the labels carry, so they follow the pack's variables automatically.

A review records the verdict, your scores for every axis, the model scores you were shown, a flag reason, seconds spent and a timestamp:

```json
{"id":"fx0022","verdict":"keep","scores":{"joy":0.0,...},"model":{"joy":0.07,...},"seconds":8.4,"at":1791828624}
```

Reviews never rewrite the label stage's output. The stage runner can treat `reviews.jsonl` as an input to contrast sets (your scores override the model's, flagged clips are dropped) and to recalibrating the fusion weights. Keeping the shown model scores means each correction stays measurable after labels are regenerated. The page shows the running bias per axis ("playfulness: +0.12 on average"), which is a first look at what recalibration will find; the recalibration itself belongs to the label stage.

## Open with the corpus format
- Real paths and field names for lines, labels and reviews, and where the review stage's output lives in the content-addressed layout.
- Whether the page should take axis order and descriptions from the pack rather than the labels (today it sorts them alphabetically).
- Audio is served whole without range requests, which is fine for clips under 20 seconds.
