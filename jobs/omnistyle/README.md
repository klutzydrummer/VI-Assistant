# omnistyle: learned style-prefix smoke test for OmniVoice

A free, CPU-only check of a learned style-prefix route on OmniVoice. It uses synthetic tokens only: no audio,
no GPU, no rental. A style encoder reads a different crop of the same recording (no labels), squeezes it through
a tight bottleneck plus noise, and writes a few continuous vectors into the prompt slot (after the text, before
the masked target). These vectors replace the reference-audio prompt (`prompt_length=0`).

Nothing in upstream `omnivoice.training` is patched. `StyleOmniVoice`, the processor, the collator and
`train_style.py` live here and reuse the unmodified `OmniTrainer`.

## Files

| File | Purpose |
|---|---|
| `style_omnivoice.py` | `StyleOmniVoiceConfig`, `StyleEncoder` (FSQ / VQ / continuous), `StyleOmniVoice` (3-branch CFG, prefix scaling, KL-to-Base guard) |
| `style_data.py` | fake tokenizer, `StyleSampleProcessor`, `StylePaddingCollator`, synthetic clips with known style biases |
| `train_style.py` | entry script around the unmodified `OmniTrainer` |
| `smoke_style.py` | checks (a)-(g), the KL guard check, the collapse log sample; writes a JSON result |
| `test_omnistyle.py` | pytest version of the checks (a few minutes on CPU; (f) needs `OMNISTYLE_REAL=1`) |
| `bench_kl.py` | measures the cost of the KL guard (parameters, peak RSS, step time) |
| `sweep_init.py` | seed x output-init sweep used to diagnose the style merge (see "Known defect") |
| `sweep_fsq.py` | bound x levels x init sweep for the FSQ fix (checks (a) and (b), saturation metrics) |
| `requirements-omnistyle.txt` | pinned versions of the venv used |

## Running

```bash
python3 -m venv /tmp/ovsmoke-venv   # or uv venv; then install requirements-omnistyle.txt
nice -n 19 python smoke_style.py --out /tmp/ovsmoke-out/results.json          # all checks, includes (f)
nice -n 19 python smoke_style.py --skip-real --out /tmp/results.json           # without the real-model load
OMNISTYLE_REAL=1 nice -n 19 python -m pytest -q -p no:cacheprovider test_omnistyle.py
nice -n 19 python bench_kl.py --out /tmp/bench_kl.json
nice -n 19 python sweep_init.py --out /tmp/sweep_init.json
nice -n 19 python sweep_fsq.py --out /tmp/sweep_fsq.json          # the FSQ fix sweep, 5 seeds
nice -n 19 python sweep_fsq.py --mode none --out /tmp/fallback_none.json   # continuous-bottleneck fallback check
```

Check (f) downloads the public `k2-fsa/OmniVoice` weights (about 2.3 GB) from Hugging Face. It needs about 6 GB
of free RAM.

Environment notes: `torchaudio` must be pinned to the same version as `torch` (a newer CPU wheel fails to import
against `torch 2.8.0+cpu` with an undefined symbol). On NixOS the torch wheel does not find `libstdc++.so.6` from
the loader cache; copying `libstdc++.so.6` and `libgcc_s.so.1` from the gcc store into
`<venv>/lib/python3.12/site-packages/torch/lib/` (on torch's RUNPATH) fixes the import.

## Things that are easy to get wrong

* **Vocabulary.** The tiny Qwen3 vocabulary must be larger than the audio vocabulary of 1025 (mask id 1024).
  The smoke uses 1100. The audio ids index the same embedding table at the audio-token positions.
* **Pre-train the base first.** The tiny model starts from random weights, so the smoke first trains a plain
  model (no prefix) for 300 steps. Its loss floor is about 4.03. Prefix gains are measured relative to that.
* **Mask-ratio shortcut.** The style bias is readable from a clip's own unmasked tokens. With a low mask ratio
  the decoder reads it from the clip and ignores the prefix, so true and shuffled prefix give the same loss.
  The smoke therefore trains at mask ratio (0.8, 1.0) and evaluates at 1.0, with a small effective vocabulary
  (64) and only five texts that carry no style information. This is the "case 2b" shortcut and it was fixed
  first (see the report).
* **FSQ levels must be odd.** The implementation rounds the bounded `z`, which is only symmetric for odd
  `L`. The default `[7, 7, 5, 5]` gives 1225 codes. The bound is configurable (`tanh` / `clip` /
  `softsign`); `tanh` with `[7, 5, 5]` is the old configuration whose styles merged (see below).
* **The prefix-scale knob is effectively on/off.** Qwen3 layers start with an RMSNorm, so a positive rescale of the
  slot vectors is almost invisible. Only scale 0 (prefix removed) changes the output. The smoke sweeps
  (0 ... 4) and the CE is identical from 0.05 upward. A real strength control is the independent style dropout
  and the 3-branch CFG guidance weights (`g_text`, `g_style`).
* **transformers 5 loading.** Weights are loaded onto the meta device and `_init_weights` runs afterwards. Direct
  `.fill_` / `.zero_` on a loaded tensor would overwrite it, so `reset_parameters` skips tensors marked
  `_is_hf_initialized`. Non-persistent buffers are not restored, so the FSQ level tables are computed on use.

## Output-layer init

The encoder output is `LayerNorm(MLP(code)) * out_scale`. `out_scale` is calibrated once to the mean norm of a
real 8-codebook audio-embedding sum (buffer `scale_calibrated`). Two config knobs control the MLP init:

* `style_out_init_std` (default 0.02; `None` = fan-in scaling) and
* `style_out_ln_eps` (default 1e-10).

With std 0.02 and the usual LayerNorm eps of 1e-5, the MLP output is so small that eps dominates the variance
and the vector norm comes out at 0.48 of the target. Lowering eps to 1e-10 fixes the norm (matches within 5%)
and keeps the small init. Fan-in init also fixes the norm but starts with much larger pre-quantiser values.
The sweep shows no arm is better at keeping styles apart (see below), so the smaller change (eps) is the default.

## Known defect: FSQ styles can merge (check (b) is seed-fragile)

`sweep_init.py`, FSQ [7,5,5], 1000 steps, lr 1e-3, noise 0.1, 300 base steps, three seeds:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| OLD (std 0.02, eps 1e-5) | (a) pass, (b) pass | (a) fail, (b) fail, all styles on code 0 | (a) fail, (b) fail, all on code 34 |
| OLD_EPS (std 0.02, eps 1e-10; default) | pass / pass | fail / fail, all on code 168 | fail / fail, all on code 6 |
| NEW (fan-in) | (a) pass, (b) fail (2 codes) | (a) pass, (b) fail (3 codes) | pass / pass |

Across all nine runs, 92-100% of the code dimensions sit on the outermost FSQ level and the mean tanh slope is
0.004-0.09. The bound saturates, and once a style is stuck in a corner the straight-through estimator passes no
gradient through the tanh to move it. The smoke's (b) passes at seed 0 but the configuration is not robust.

### Fix attempt (2026-11-13)

`sweep_fsq.py` swept the approved fix space (more levels/dims, `clip`/`softsign` bounds, a small
pre-quantiser init) on the seeds the old configuration failed (1, 2, 4). The best arm was `[7, 7, 5, 5]`
with the tanh bound and fan-in init: it passed all three screened seeds (own-share 0.984-0.995). The shipped
defaults now use it: `style_fsq_levels=[7, 7, 5, 5]` (1225 codes), `style_fsq_bound="tanh"`,
`style_code_init_std=None`.

The five-seed confirmation with those defaults (seeds 0-4, no overrides) passes check (b) on seeds 1, 2 and 4
but fails on seeds 0 and 3: on seed 3 two styles share the same modal code (`b_distinct=3`), and on seed 0
the modal codes of styles 0 and 1 decode toward other styles. Modal-code purity on the failing seeds is
0.12-0.52. So the fix is not seed-robust; FSQ check (b) remains a 3/5 result (the old configuration passed
2/5). The continuous-bottleneck fallback (`sweep_fsq.py --mode none`) passes its tightness and leakage gates
on seeds 0-3 but collapses on seed 4 (centroid purity 0.586, check (a) fails), so it is not 5/5 either.
`workers/smoke/fsq-fix-report.md` has the per-seed tables, the purity details and the fallback numbers.
Whether to accept 3/5, switch the default quantiser or widen the fix space is the owner's call.
