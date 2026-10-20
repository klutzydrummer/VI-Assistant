# Effect detector: which measure hears the radio and other effects?

Status: built 2026-10-19 (`jobs/fx_detect.py`), not yet run on her clips. It runs at home on CPU, for free,
once Brandon's channel taps are saved in `corpus/tali/channel.jsonl`. Check A of the voice gate
(`voice-consistency-check.md`) uses the chosen score so a mood can't pass on the radio instead of her voice.

## Why

Some clips carry a radio (walkie-talkie) or other effect on top of the suit. The six channel measures
(`corpus/tali/analysis/after_labels/channel.py`: low and high cut, band edges, noise, compression) didn't
single out the 14 clips the game tags as radio, and the raw low-cut measure mostly tracked her shouting
(pitch, loudness and game explained R2 0.79). They can't see distortion, limiting or a squelch at all.

## What gets tested, against his marks

| Set | What it is | Size | Cost per clip (CPU) |
|---|---|---|---|
| channel | the six measures from channel.py's table (rebuilt here if no table) | 6 | ~0 |
| extras | crest factor, clipped share, flatness, periodicity, energy between harmonics, a fixed tone at either end | 6 | ~0 |
| fxpp | [Fx-Encoder++](https://github.com/SonyResearch/Fx-Encoder_PlusPlus) (Sony, 2025): effects encoder, contrastive on mixes | 128 | ~0.15 s |
| afx | AFx-Rep from [ST-ITO](https://github.com/csteinmetz1/st-ito) (Steinmetz et al., ISMIR 2024): Cnn14 trained to tell effect presets apart | 512 | ~0.3 s |

Both models are trained on music, not speech; the marks show whether they transfer. Only their effects
encoders are used, rebuilt in plain torch so the run needs no laion-clap, Lightning or torchlibrosa (their
pinned dependencies are what broke earlier installs). The rebuild was checked here against the authors' own
code: identical outputs (max difference 0.0) on the same weights. Weights come from Hugging Face at pinned
revisions with sha256 checks. Fx-Encoder++ is CC BY-NC 4.0 (fine for this personal project), AFx-Rep Apache 2.0.

## How each set is scored

Fixed before any result, so nothing is tuned on the ~40 marks:
- **Target:** effect (radio or other_effect) vs plain is the main one. Radio vs the rest and all three classes
  are reported too. cant_tell is left out.
- **Folds:** leave one conversation out, since radio clips come in runs from one scene.
- **Model:** standardise, PCA to 8 dims for the embeddings, L2 logistic regression with balanced classes.
- **Shuffled labels:** the whole fold loop rerun with labels shuffled (500 times) gives the p-value.
- **Confounds:** what loudness, pitch, duration and game do to each feature is fitted on the whole corpus
  (no labels needed, and nearly all of it is plain) and taken out. A set that only hears her shouting falls
  to chance. Fitting this on the marks instead would also remove a real radio effect whenever the marked
  radio clips happen to be shouted; the tests plant exactly that case. The confounds alone are scored as a
  baseline, and how much of each detector's corpus-wide score they explain is reported.
- **Pick:** highest residualised effect AUC with p < 0.05 and confound R2 < 0.5. Within 0.02 AUC the cheaper
  set wins. Its residualised per-clip score (`scores.csv`) is what check A uses.

## Running it (Brandon's PC, through the Remote Control thread)

From the repo root, inside
`nix-shell -p "python3.withPackages (p: [p.numpy p.scipy p.scikit-learn p.soundfile p.torch])"`, at low
priority (`nice -n 19`, plus `ionice -c3` if the harvest audio is on the game disk, else `ionice -c2 -n7`):

1. `python3 jobs/fx_detect.py fetch --weights DIR`: downloads 2.8 GB one model at a time, keeps only the
   encoders (about 0.65 GB) and deletes the rest. It stops before downloading if DIR lacks room, so point
   DIR at a disk with ~3 GB free if `/home` is still tight (default `~/.cache/vi-fx`). Can run before the taps.
2. `python3 jobs/fx_detect.py smoke corpus/tali --weights DIR`: every check in seconds, including both
   nets on two real clips, the marks and channel.py's table (it prints which columns it takes; pass
   `--channel FILE --channel-cols a,b,...` to choose). Prints the projected run time.
3. `python3 jobs/fx_detect.py run corpus/tali --weights DIR`: smoke again, then measures every clip from
   harvest's raw audio (resumable; clean's output may have removed the effect), then scores. Roughly an hour
   for 3,085 clips on 4 cores. Writes `corpus/tali/analysis/fx_detect/`: `report.md`, `summary.json`,
   `scores.csv` and `next_to_mark.csv` (unmarked clips at both ends and at the boundary, for a second round
   of taps if the intervals are wide).

Audio is read only on his machine and nothing is uploaded; the only network use is the weight download.

## Limits

- About 40 marks means wide intervals: the report shows conversation-level bootstrap bounds, and a second
  round of taps from `next_to_mark.csv` narrows them.
- If no set passes, nothing is picked and the report says so; how check A proceeds then is Brandon's call.
- The extras are guesses at what the effect is (distortion, limiting, a squelch); their weights in the report
  show which, if any, carries it.
