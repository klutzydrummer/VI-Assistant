# The fresh voice fine-tune: how to prepare and run it (2026-11-02)

Stage 5's one paid run, rebuilt after the two training bugs (QwenLM/Qwen3-TTS
issues #120 and #179) were fixed in `jobs/voicetune.py`. It fine-tunes
`Qwen3-TTS-12Hz-1.7B-Base` into a no-reference `tali` voice: 6 epochs from Base
with the KL-to-Base term on replay lines, each epoch pushed home as it lands,
the held-out-loss epoch picked by Brandon's ears. It is wired as three stages in
`characters/tali/voicetune.toml`:

```
vi dataset run characters/tali/voicetune.toml voicetune_data       # home, CPU
vi dataset run characters/tali/voicetune.toml voicetune_preflight  # GPU, manual
vi dataset run characters/tali/voicetune.toml voicetune_train      # GPU, manual
```

The two GPU stages run `jobs/voicetune-fresh.sh` through `jobs/on-worker.sh`;
the argument picks `preflight` (check `--generate`, gate VRAM/disk, a 5-step
probe, push the probe home) or `train` (the real 6-epoch run). They are separate
stages on purpose: separate `VI_STAGE` means separate home checkpoint dirs, so
each can push its own epoch-0 without the two colliding in the return path.

On the worker, `voicetune-fresh.sh` first fetches the official Qwen3-TTS
`finetuning/` folder (`dataset.py`) at the pinned commit
`022e286b98fbec7e1e916cb940cdf532cd9f488e` into `~/.cache/vi/`, and passes it to
both `voicetune.py train` calls via `--qwen-finetuning` so the training loader's
import of that folder resolves.

## The `voicetune_data` stage (home, CPU)

`jobs/voicetune_data.sh` (wrapping `jobs/voicetune_data.py`) builds one folder
that the two GPU stages read as their single input `VI_INPUT_VOICETUNE_DATA`. It
holds everything `voicetune-fresh.sh` needs, with every `audio`/`ref_audio` path
relative to the folder so the folder relocates and packs whole:

- `train_with_codes.jsonl` (1961 clips), `val_with_codes.jsonl` (222) and
  `val100_with_codes.jsonl` (100) — the label-blind split from `jobs/buildset.py`
  (whole conversations assigned by a seeded hash of the conversation id, ~10% per
  game; the 100-clip val sample is another seeded draw), codec-encoded by
  `voicetune.py prepare` on the home CPU. `val_with_codes.jsonl` is what the
  fresh run validates against; `val100_*` is the quick-smoke sample.
- `wav/` — her 24 kHz mono clips, hard-linked from an earlier build (never
  re-encoded, never duplicated).
- `replay_with_codes.jsonl` (300 rows) and `voices/` — the KL-to-Base replay
  lines and the 8 public LibriTTS-R reference voices (CC BY 4.0) they were spoken
  in.
- `summary.json` — the split and drop counts, plus the replay source note.

The split reads only game and conversation ids; no mood label, rating or
hand-labelled set is consulted, and the 6 delivery clips are later picked from
the train manifest by seed 0 alone.

## Replay: codes reused, no fabricated audio

Replay is Base speaking generic sentences in other, public voices. The replay
WAVs were generated on an old worker and never came home, but the codec codes
did: `~/.cache/vi/voicetune/fresh/data/replay_with_codes.jsonl` (300 rows,
already codec-encoded, label-free per audit) and the reference voices in
`~/.cache/vi/voicetune/abrun/libri/voices/` are at home. Base generation was
unaffected by the training bugs, so those codes are reused as-is.
`voicetune_data.py` copies the voices into `out/voices/` and rewrites each
replay row's `ref_audio` to `voices/<file>`; the `audio` field stays the
dangling `replay/wav/replay-*.wav` because the spoken clip does not exist and is
not fabricated. The training loader reads the precomputed `audio_codes`, and
`validate()` skips the sample-rate check for `audio` on rows that carry codes, so
the missing audio is accepted (the `ref_audio` voice is still checked, since the
x-vector encoder reads it).

`jobs/libri.py` is the committed fetch/selection script for the replay source
(public LibriTTS-R, CC BY 4.0): one 24 kHz 4-10 s clip per speaker and 300
transcripts, seeded, reading only speaker id and transcript — no labels.

## Run order

1. `python3 jobs/checkpoints.py gate /home --min-gb 50` — refuses to start
   under 50 GiB free on `/home` (the checkpoints come home before `cloud down`).
2. `vi dataset run characters/tali/voicetune.toml voicetune_data` — home, CPU,
   under `nice -n 19 ionice -c2 -n7`. See the note below about the home qwen-tts
   environment.
3. `vi cloud up --cap <usd> --until <time>` — the approved cap and deadline for
   the rental; `--until` is an RFC 3339 timestamp or `HH:MM` (today, local time).
4. `vi dataset run characters/tali/voicetune.toml voicetune_preflight`.
5. `vi dataset run characters/tali/voicetune.toml voicetune_train` — only after
   the preflight probe is verified and Brandon has okayed the paid run and its
   budget.
6. Before `vi cloud down`, verify every pushed checkpoint so it refuses to
   destroy the worker with an unverified one still at home:

   ```
   python3 jobs/checkpoints.py verify <preflight checkpoint dir> --expect 1
   python3 jobs/checkpoints.py verify <train checkpoint dir>      --expect 6
   ```

   Each checkpoint dir is the stage's `.checkpoint-<key>/` under
   `corpus/tali/stages/voicetune_<mode>/`; `on-worker.sh` also records the
   absolute path in `~/.local/state/vi/unverified/voicetune_<mode>`.
7. `vi cloud down`.

### The home qwen-tts environment (open item)

`voicetune_data` runs `prepare`, which needs `qwen-tts` and a CPU `torch`.
`jobs/voicetune_data.sh` builds the same `requirements-tune.txt` venv as
`voicetune-fresh.sh` (or honours `QWEN_TTS_PYTHON`). On this NixOS home the pip
wheel `torch` in that venv needs the C++ runtime that NixOS doesn't put on the
default path, so the venv's `numpy`/`torch` fail to import unless run through
`steam-run` (or an equivalent). The stage config itself stays clean; at home the
job is run as

```
QWEN_TTS_PYTHON="$(command -v steam-run) $HOME/.cache/vi/venv-d9da7e971e969c68/bin/python" \
  nice -n 19 ionice -c2 -n7 jobs/voicetune_data.sh
```

(that venv already holds `qwen-tts==0.1.1` + CPU `torch` + `soundfile`). This is
a NixOS runtime detail, not a stage-logic gap; see the report.

## Remaining gaps

- **The raw manifests are unaudited** at the time of writing. `jobs/buildset.py`
  produces the label-blind split deterministically, but the parallel data-rules
  audit had not yet signed off `train_raw.jsonl`/`val_raw.jsonl`; the stage
  names the folder, not the files, and must not be pointed at them until that
  audit passes.
- **No committed script re-encodes a missing wav** from the clean-stage FLACs;
  `jobs/buildset.py` does that via `soundfile` but the voicetune_data run
  hard-links from an earlier build, so the re-encode path is untested this round.
- **The LibriTTS-R parquet fetch is manual** (`jobs/libri.py` documents the
  `curl`), and `jobs/libri.py` needs `pyarrow` — a regeneration-time dependency,
  not part of the stage runtime.
