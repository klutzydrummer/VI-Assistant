# OmniVoice full fine-tune with the learned style prefix

This folder is the runnable pipeline around the style-prefix route validated in `jobs/omnistyle`:
the frozen OmniVoice Base gets a small style encoder whose continuous vectors occupy the prompt
slot, and both are trained together with the **unmodified** upstream `OmniTrainer`.

It reuses, without changing them:

* `jobs/omnistyle/style_omnivoice.py` — `StyleOmniVoice`, `StyleEncoder`, the 3-branch CFG decoder;
* `jobs/omnistyle/style_data.py` — the sample layout, dropouts and padding collator;
* `omnivoice.training.trainer.OmniTrainer` and `omnivoice.training.checkpoint` from the installed
  package.

New here:

| File | What it does |
| --- | --- |
| `data_manifest.py` | builds the upstream manifest (`id`/`audio_path`/`text`/`language_id`) from the project's clip pool, label-blind; optional public replay rows are marked `replay: true` |
| `ft_data.py` | real shard processor (`StyleShardProcessor`), synthetic/shards data bundles, and the guard probes |
| `train_ft.py` | entry script: model, own optimizer (separate encoder lr), guard, checkpoint contract, resume |
| `infer_listening.py` | style-slot inference with the 3-branch CFG and a labelled listening set |
| `test_omnivoice_ft.py` | CPU smoke + failure-injection tests |

## Shipped defaults (do not change without the parent's decision)

| Knob | Default | Why |
| --- | --- | --- |
| backbone lr | `5e-4` | pinned by the continuous-bottleneck evidence (`jobs/omnistyle`) |
| encoder lr | `1e-3` | its own, higher knob. 5e-3 collapsed the tiny continuous probe (sep_ratio 0.24 at step 50); 1e-3 stayed healthy (0.91 -> 1.82 over 150 steps) |
| `style_quantizer` | `"none"` | continuous bottleneck is the shipped route (`fsq`/`vq` remain selectable) |
| `kl_weight` | `1.0` | KL-to-Base on public replay rows only |
| `mask_ratio_range` | `[0.5, 1.0]` | biased upward so the prefix is used instead of read from the target |
| `prompt_ratio` | `0` | the learned prefix replaces the real audio prompt |
| `drop_cond_ratio` | `0.1` | upstream's value |
| `style_drop_ratio` | `0.2` | independent prefix dropout for the text-only CFG branch |
| `batch_tokens` | `2048` | per GPU micro-batch (r25; was 8192, which ran out of memory on 24 GB) |
| `gradient_accumulation_steps` | `4` | so an optimizer step still sees 8192 tokens; `steps` counts optimizer steps. Both defaults live in `ft_guards.py` alone |
| KL reference | bf16, frozen, no gradients | the KL itself is computed in fp32; the student stays fp32 under bf16 autocast. Since r26 its forward runs only on the replay rows of each micro-batch (same KL value) |
| `batch`/optim | bf16, TF32, weight decay 0.01, max grad norm 1.0, cosine schedule | primer's `train_config_finetune.json` |
| collapse guard | ON | probe, fire rule, restart with `seed + attempt`, max 3 attempts |

On a machine without CUDA, `--mixed-precision bf16` resolves to `no` and the CPU smoke runs fp32.

## 1. Build the manifest

The input is the project's existing label-blind raw manifests from `jobs/buildset.py`
(`train_raw.jsonl`, `val_raw.jsonl`, with `id`, `audio` and subtitle-derived `text`). Only those
three fields are read; mood, ratings and every other field are ignored. Clips are never dropped for
appearing in an old labelled set, and audio files are never copied — the manifest holds absolute
paths in place.

```bash
python jobs/omnivoice-ft/data_manifest.py \
    --raw-jsonl /path/buildset/train_raw.jsonl \
    --raw-jsonl /path/buildset/val_raw.jsonl \
    --audio-root /path/buildset \
    --out /path/work/train_manifest.jsonl
```

`--audio-root` is the folder the manifests' relative `wav/<id>.wav` paths resolve against.

For a bare folder of clips, use `--pool-dir <dir> --texts-jsonl <any id/text JSONL>`. Missing
transcripts are an error unless `--allow-empty-text` is passed (the clip is then kept with an empty
text span, and the summary reports how many).

Public-voice replay rows for the KL-to-Base guard go in separately:

```bash
python jobs/omnivoice-ft/data_manifest.py \
    --raw-jsonl /path/buildset/train_raw.jsonl --audio-root /path/buildset \
    --replay-jsonl /path/replay_public.jsonl \
    --out /path/work/train_manifest.jsonl
```

Each replay row must have `id`, `audio_path` (or `audio`) and `text`. Duplicate ids get a
`-replay` suffix. **Without replay rows the KL term has nothing to act on** (it stays zero); the
collapse guard still runs, but the anti-drift guard does not.

The builder writes `<out>.summary.json` next to the manifest with row counts, the input sources,
`fields_read`, `audio_copied: false` and the manifest's sha256.

For the real run, jobs/omnivoice_data.py (stage omnivoice_data in
characters/tali/omnivoice-ft.toml) does this step and builds the whole data folder at home:
manifests, style_manifest.jsonl, data_config.json, hard-linked clips, the public reference
voices and an audio_tokenizer/ placeholder. voicetune-fresh.sh preflight then runs the
tokenisation (section 2) and the replay generation on the GPU worker (steps 0a/0b) and rehearses
a kill and resume.

## 2. Tokenise into WebDataset shards

One upstream command does tokenisation, sharding and `data.lst`:

```bash
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl /path/work/train_manifest.jsonl \
    --tar_output_pattern /path/work/shards/audios/shard-%06d.tar \
    --jsonl_output_pattern /path/work/shards/txts/shard-%06d.jsonl \
    --tokenizer_path /path/to/higgs-audio-v2-tokenizer \
    --samples_per_shard 1000 --skip_errors
```

Outputs `/path/work/shards/data.lst` (one line per shard: tar, jsonl, count, seconds),
`errors.jsonl`, and the shards. Every manifest field survives into the shard metadata, so
`replay: true` reaches the sample processor as `sample["label"]["replay"]`.

Then a data config JSON:

```json
{
  "train": [{"manifest_path": ["/path/work/shards/data.lst"], "repeat": 1}],
  "dev":   [{"manifest_path": ["/path/work/shards/data.lst"], "repeat": 1}]
}
```

## 3. Train

CPU smoke on the tiny synthetic model (no data, no downloads, ~1 minute):

```bash
nice -n 19 python jobs/omnivoice-ft/train_ft.py --synthetic \
    --steps 20 --save-steps 10 --output-dir /tmp/ovft-smoke
```

Real run from the public Base:

```bash
accelerate launch --num_processes 1 jobs/omnivoice-ft/train_ft.py \
    --init-from k2-fsa/OmniVoice \
    --data-config /path/work/data_config.json \
    --output-dir /path/run \
    --steps 5000 --save-steps 500
```

Resume from a verified checkpoint (the run continues to `--steps`):

```bash
accelerate launch --num_processes 1 jobs/omnivoice-ft/train_ft.py \
    --data-config /path/work/data_config.json --output-dir /path/run \
    --steps 5000 --save-steps 500 \
    --resume-from-checkpoint /path/run/checkpoint-2000
```

The checkpoint's architecture knobs (`style_n_vectors`, `style_quantizer`, `target_frames`,
`crop_frames`, `init_from`) must match; the script refuses a mismatch instead of silently loading a
different model. Resume restores model, optimizer, scheduler and RNG state through
`accelerator.load_state`; the data position is not saved upstream, so the resumed epoch restarts
from the loader's beginning (documented upstream limitation).

Outputs in `--output-dir`:

* `checkpoint-N/` every `--save-steps` and at the final step;
* `style_metrics.jsonl` — one row per logged micro-step (`loss/ce`, `loss/aux`, `loss/kl`,
  `kl/rows_used`, `kl/rows_total`, `style/z_spread`, `style/out_rms`, `attempt`, ...);
* `collapse_guard.jsonl` — one row per probe (`sep_ratio` or `z_spread_min`, threshold, strikes);
* `ft_config.json` — the full pipeline configuration of the run;
* upstream's `train.log`, `initial_config.json` and TensorBoard events.

### Collapse guard

Every `--guard-every` micro-steps after `--guard-warmup`, the current style code is probed.

* Synthetic runs (tests/smoke): the validated labelled probe, `sep_ratio < --guard-min-sep` (0.5)
  for `--guard-patience` consecutive probes.
* Real runs: a **label-free** probe on fixed crops — the spread of the unit-RMS code across the
  probe batch. It fires below `max(--guard-min-z-spread, --guard-z-spread-frac * baseline)`, where
  the baseline is measured before training. This threshold has not been validated on real data;
  the validated labelled probe is not usable without labels.

On a fire, the attempt's checkpoints and metrics are removed, the run restarts from scratch with
`seed + attempt` (up to `--guard-attempts`), and the log keeps every attempt. A resume disables
retries (the same checkpoint would reproduce the same collapse) and raises instead.

## 4. Checkpoint and checksum contract

Every `checkpoint-N/` produced by `train_ft.py` contains:

```text
config.json                 model architecture (including the style fields)
model.safetensors           full model weights (Base + style encoder)
optimizer.bin               AdamW state, both parameter groups
scheduler.bin               cosine/constant schedule state
random_states_0.pkl         RNG state for a deterministic resume
train_config.json           upstream TrainingConfig copy
ft_config.json              this pipeline's full configuration
<tokenizer files>           the text tokenizer saved by the trainer
SHA256SUMS                  sha256 of every other file in the folder
```

The upstream trainer is untouched: after its own save, the instance's `save_checkpoint` is wrapped
to add `ft_config.json` and write `SHA256SUMS` (sha256sum-compatible, sorted relative paths, written
atomically). A checkpoint only exists once its manifest does.

Verify one or more checkpoints before training or inference:

```bash
python jobs/omnivoice-ft/train_ft.py --verify-checkpoint /path/run/checkpoint-500 /path/run/checkpoint-1000
```

It reports missing required files, checksum mismatches, listed-but-absent files and (strictly)
unlisted extra files, and exits non-zero on any problem. `--resume-from-checkpoint` and
`infer_listening.py` both refuse an unverifiable checkpoint. Checksumming a full 0.6B checkpoint
reads a few GB per save — that is the cost of the contract; it is not run in the CPU tests.

## 5. Listening set

```bash
python jobs/omnivoice-ft/infer_listening.py \
    --checkpoint /path/run/checkpoint-5000 \
    --style-manifest /path/work/train_manifest.jsonl --max-styles 6 \
    --base-model /path/to/k2-fsa-OmniVoice-snapshot \
    --out /path/listening-step5000
```

Style sources:

* `--style-manifest` — JSONL with `id`, `audio_path`, `text`. The style encoder reads a tail crop
  (`--style-crop-frames`) of the recording; the text is the reference transcript, so no ASR is
  needed. `--load-asr` allows transcribing rows that have no text.
* `--style-tokens` — `.pt` files holding `[C, T]` audio-token tensors (or `{"id","tokens"}`); the
  CPU test path, no audio tokenizer needed.

Text lines come from `jobs/ttstest/lines.json` by default (`--lines` overrides; JSON array or
JSONL, with `id`/`text`). Each clip is written to `wav/<style>__<line>__s<seed>__step<N>.wav` (or
`tokens/<label>.pt` with `--tokens-only`), and `listening.jsonl` records the label, style id, line
id, register, seed, checkpoint step, target length and guidance settings. `README.txt` in the
folder states the rule: **human listening decides voice quality; there is no automatic score.**

The three CFG branches are `full = text + prefix`, `text-only`, and `nothing`, combined as
`log_softmax(c + g_text (t - u) + g_style (c - t))`. Two prefix-strength knobs: `--style-scale`
(scales the prefix vectors) and `--guidance-style` (the CFG push away from the text-only branch).

Decoding needs the Higgs v2 audio tokenizer, read from `--audio-tokenizer` or
`<base-model>/audio_tokenizer` — no implicit download. If neither exists, use `--tokens-only`.
The tiny-model path (`--fake-tokenizer`) exists for tests only.

## 6. Tests

```bash
export LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/torch/lib  # NixOS venv only
nice -n 19 <venv>/bin/python -m pytest jobs/omnivoice-ft/ -q
```

The suite covers: label-blind manifest + replay flags + no audio copying; window choice and shard
processor layout; the checkpoint/checksum contract and tamper detection; the KL term on replay
rows; a simulated mid-run kill followed by resume (model/optimizer/scheduler restored exactly, and
a resumed run matches a control run bit-for-bit); the collapse-guard restart through this script's
own wiring; and style-slot inference with deterministic labelled tokens. It uses the tiny synthetic
model only — no game audio, no downloads, CPU only.

## Data rules (VI Assistant project)

* Learn from raw data, as unsupervised as possible. `data_manifest.py` reads only `id`, `text` and
  one audio-path field; it never reads, filters, weights or selects on mood or any legacy
  hand-labelled field. Do not add such a filter here.
* Do not remove clips because they were in an old labelled set; keep all suitable audio.
* Copyrighted game audio and text move only home <-> rented worker. Manifests reference paths; no
  audio or transcripts are bundled in this repo, and no dataset host is involved.
* Human listening, not an automatic score, decides voice quality.

## Known limits (verified vs not)

Verified on this host: the whole CPU test suite above; the real checkpoint layout of an upstream
tiny run (`config.json`, `model.safetensors`, `optimizer.bin`, `scheduler.bin`,
`random_states_0.pkl`, `train_config.json`, tokenizer); the actual collapse behaviour of encoder
lr 5e-3 vs 1e-3 vs 5e-4 on the tiny continuous probe.

Not run here (and therefore unverified): the real Base checkpoint on GPU, the real
`extract_audio_tokens` -> shard loader path, real-audio inference with the Higgs decoder, and the
real-data `z_spread` guard threshold. The `k2-fsa/OmniVoice` HF cache does exist on the host at
`/home/brandon/.cache/huggingface/hub/models--k2-fsa--OmniVoice`; this sandbox's HOME differs, so
it was not visible here and the real Base was not loaded (no downloads are allowed in-sandbox).
All model code paths were therefore exercised with the tiny synthetic model only. A future
environment can reuse the cache with `HF_HOME=/home/brandon/.cache/huggingface` and
`HF_HUB_OFFLINE=1`. On this NixOS host the venv needs the `LD_LIBRARY_PATH` above before
`numpy`/`omnivoice` can be imported.
