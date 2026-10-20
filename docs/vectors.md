# Building chat control vectors (`vi vectors`)

How to run the builder designed in `docs/design/steering-builder-design.md` (PR #63) (approved by Brandon
2026-10-20: joint fit by default; full strength = each vector's best single layer at 1.0;
telescoping bands). Code: `crates/llm/src/{capture,fit,check,store,registry}.rs`, commands in
`crates/vi/src/vectors.rs`.

```
vi vectors fetch <model-id>                 download a models.toml entry, check its SHA-256
vi vectors preflight <model-id> --messages m.jsonl [--character characters/tali]
vi vectors capture <model-id> --messages m.jsonl --store <folder> [--no-think]
vi vectors fit --store <folder> --labels l.jsonl --pairs p.jsonl --out <folder>
vi vectors build <model-id> --messages m.jsonl --labels l.jsonl --pairs p.jsonl [--store <folder>]
vi vectors check <model-id> --vectors <out>/joint --plan plan.jsonl --results results.jsonl [--thinking on|off] [--steer reply|all]
vi vectors layers <out>/joint single:32 | band:24-40 [--only warmth,teasing]
vi vectors selftest
```

Model commands also take `--models <folder>` (default `$VI_MODELS`, else `models/`),
`--gpu-layers N` and `--threads N`. `build` writes the store to
`vectors/<name>/llm/<model-sha256>/capture/` and the vectors next to it unless `--store`/`--out`
say otherwise (`--name`, default `tali`). At home, put the store on a disk with 6 GB free and run
under `nice -n 19` (plus `ionice -c3` on the game disk, `ionice -c2 -n7` on the system disk).

## Thinking (Brandon, 2026-10-20: on for real use, off is fine for tests)

- **Capture** puts each of her lines behind an empty think block: `vi chat`'s prompt for the
  exchange (byte-equal through `<|im_start|>assistant\n`), then `<think>\n\n</think>\n\n`, then her
  line, exactly where a reply sits after thinking. `line_mean` pools over `</think>`, the blank line
  and her line; `line_head` over that lead-in and her first 8 tokens; the log-probability covers her
  line only. Nothing follows her line: with causal attention an end-of-turn token after it can't
  change what's pooled. `--no-think` (on `preflight`, `capture`, `build`) captures without the block,
  pooling from the last prompt token, for comparison only; the store's `capture.json` records which
  (`think`), and a store won't resume under the other setting.
- **Check** runs with thinking `on` and steering scope `reply` by default. `reply` leaves the prompt and
  the thinking unsteered and switches steering on when the model samples `</think>` (looked up by
  text; a model where it isn't one token is refused), so her reply is steered from its first token.
  `all` steers from the last prompt token on, thinking included, as `vi chat` does today. Thinking
  `off` ends the prompt with the empty think block, so both scopes start at the prompt's `</think>`
  and the last prompt token respectively. `max_tokens` covers the thinking (default 1000 with
  thinking on, 200 off). Each result records `think`, `scope`, and where steering started
  (`steer_start`: `prompt`, `reply` or `never`, with `steer_start_token`). The judge runs with the
  trial's thinking setting and only its text after `</think>` is kept.

## Swapping the model

Add an entry to `models.toml` (repo + file + sha256, or `path` for a local fine-tune), then
`vi vectors build <id>` and `vi vectors check <id>`. Everything lands under the new file's
SHA-256, and `vi chat` refuses vectors built for a different file.

## Inputs (JSON lines; ids are her line ids)

**messages.jsonl**, one per line to capture (from `render_cvector.py`'s messages output). Every
line must use the same persona; it's decoded once and reused. `user` may be left out for a line
that answers nothing. Templates are applied in Rust, exactly as `vi chat` does.
```json
{"line_id": "me2_tali_0412", "system": "<persona>", "user": "<the line she answers>", "assistant": "<her line>"}
```

**labels.jsonl**, every line's Claude scores (pass 2, 0-3, the same mood set on every line),
knob levels 0-1, pass-1 agreement where it disagreed, and the nuisance facts:
```json
{"line_id": "me2_tali_0412", "conversation": "me2_c0815", "game": "ME2", "audio_energy": -23.4,
 "scores": {"warmth": 3, "teasing": 1, "...": 0}, "knobs": {"energy": 0.6, "intoxication": 0, "hungover": 0},
 "pass1_agree": {"teasing": false}}
```

**pairs.jsonl**, step 4's contrast sets: one row per pair (positive and its matched zero). A zero
may appear in several pairs.
```json
{"set": "warmth", "pos": "me2_tali_0412", "zero": "me3_tali_1201"}
```

**plan.jsonl** for the chat check (step 5 writes the prompts, judge and pass rule). `scheme`:
`single:L`, `band:A-B` (telescoping), `band-k:A-B`, `band-sqrtk:A-B`; leave it out to use the
folder's own `vectors.toml`. `thinking` (`on`/`off`) and `scope` (`reply`/`all`) override the run's
`--thinking`/`--steer` for one trial. `{reply}` in the judge messages is replaced by the steered reply; the
judge is the same model, unsteered. Reruns skip ids already in the results.
```json
{"id": "warmth-single32-1", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "steer": {"warmth": 1.0}, "scheme": "single:32", "max_tokens": 1000, "thinking": "on", "scope": "reply",
 "judge": [{"role": "user", "content": "...{reply}..."}]}
```

## Outputs

```
<out>/joint/natural/<mood>.gguf    every captured layer's direction, natural size
<out>/joint/<mood>.gguf            the bank vi chat loads, for the current layer scheme
<out>/joint/vectors.toml           layers and scale (1.0) per vector
<out>/mean_diff/...                the same for the per-set mean of pair differences
<out>/report.md, report.json       per vector and layer: held-out AUC (folds by conversation),
                                   split-half, conversation fixed-effects and masked-score
                                   cosines, joint/mean_diff norm, norm vs residual, pair
                                   consistency, conversation spread, VIF; flags only
<out>/cosines.json                 joint cosines between vectors, per layer
```

`vectors.toml` starts every vector at the middle layer (single layer, 1.0) until the chat check
settles the scheme; then `vi vectors layers` rewrites the bank from `natural/`. Joint is the
default; using `mean_diff` for any mood needs Brandon's yes.

## Home smoke on Qwen3.5-0.8B (CPU)

Same `qwen35` architecture as Qwen3.8-27B. Run from the repo, in the nix-shell that builds `vi`.

1. `cargo build --release -p vi`, then `nice -n 19 target/release/vi vectors selftest`.
2. `target/release/vi vectors fetch qwen3.5-0.8b-q8 --models <disk with space>/models` (0.8 GB).
3. Make `messages.jsonl`, `labels.jsonl` and `pairs.jsonl` for two real sets (only the lines in
   those sets, or all lines if time allows).
4. `nice -n 19 target/release/vi vectors preflight qwen3.5-0.8b-q8 --models ... --messages ... --character characters/tali`
   (layer test, reuse test on the hybrid model, persona byte-equal to `vi chat`'s).
5. `nice -n 19 target/release/vi vectors build qwen3.5-0.8b-q8 --models ... --messages ... --labels ... --pairs ... --store <disk>/store --out <disk>/vectors`
   under a peak-RAM measure (`/usr/bin/time -v`); note the lines/s it prints.
6. A small plan for `vi vectors check`: one mood and one two-mood blend, `single:<mid>` and a
   telescoping `band:`, plus the unsteered reply, with thinking on in both scopes (`reply`, `all`)
   and once with thinking off.
7. Optional: compare `mean_diff` with `llama-cvector-generator --method mean` on the same pairs,
   if a CPU llama.cpp build is at hand (expect a high cosine after its one-layer shift).
