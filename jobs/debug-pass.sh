#!/usr/bin/env bash
# Debug pass on the worker: stages 3 to 11 of `jobs/paid_run.py --debug-pass`, run one at a time
# by jobs/on-worker.sh (the stage omnivoice_debug in characters/tali/omnivoice-ft.toml). Every
# stage runs in its own process under its own time limit, and a failure never stops the next
# stage: a stage whose prerequisite failed is reported SKIP-dependent instead of being run.
#
# The stages reuse the real worker code. The functions below are cut by name out of
# jobs/voicetune-fresh.sh (the same awk cut jobs/rehearsal/vast-limits.sh uses), so a debug pass
# tests exactly what the paid run will run:
#    3 upload       the token handshake and the input upload already worked (on-worker.sh only
#                   starts this script after both), so this records them
#    4 venv         ovft_python builds the worker venv; then every entry point is imported, the
#                   accelerate launcher is run as a module on a probe script, and the tokeniser
#                   module answers --help
#    5 base-model   fetch_base_model.py at the pinned revision (its size table is the check)
#    6 tokenise     the real tokenise() on the small train and dev slice
#    7 cuda-load    gpu_gate, then replay_public.py loads Base on cuda and writes the replay set,
#                   which is tokenised: this is what the KL reference reads
#    8 train-steps  5 real steps through the accelerate launcher at the trainer's default
#                   batch_tokens with the KL reference on, then the VRAM peak
#    9 ckpt-push    publish checkpoint-5 and push it home through the receiver (home verifies it)
#   10 listening    infer_listening.py writes a small labelled listening set into the stage
#                   output, which comes home with it; for a person to listen to, never scored
#   11 resume       2 more steps resumed from checkpoint-5, then --verify-checkpoint
#
# Status lines (read by paid_run.py), each followed by up to VI_DEBUG_EXCERPT_LINES log lines:
#   debug-pass: stage N NAME: PASS (S s)
#   debug-pass: stage N NAME: FAIL (exit RC, S s)          RC 124 means it hit its time limit
#   debug-pass: stage N NAME: SKIP-dependent (needs stage M NAME, which did not pass)
# Each stage's full log is $VI_OUT/debug/NN-NAME.log, and $VI_OUT/debug/summary.json holds every
# result. The script exits 0 whatever happened, so on-worker.sh always brings the output home.
#
# Knobs: VI_DEBUG_STAGES (default "3 4 5 6 7 8 9 10 11"), VI_DEBUG_TIMEOUT_<N> (seconds),
# VI_DEBUG_IMPORTS (the modules stage 4 imports), VI_DEBUG_NO_CUDA=1 (the home rehearsal: stage 7
# skips gpu_gate and runs replay on the CPU stubs), VI_DEBUG_PREFLIGHT_RC (the worker-preflight
# exit code on-worker.sh saw; a non-zero one is noted in stage 3, not fatal, in a debug pass).

jobs_dir=$(cd "$(dirname "$0")" && pwd)
self="$jobs_dir/$(basename "$0")"
: "${VI_OUT:?debug-pass.sh runs through jobs/on-worker.sh}"
debug="$VI_OUT/debug"
mkdir -p "$debug"

# Same scratch layout as voicetune-fresh.sh, beside VI_OUT (not under it), so checkpoint weights
# do not come back through vi-out.
work="$(cd "$(dirname "$VI_OUT")" && pwd)"
scratch="$work/voicetune-fresh"
probe_run="$scratch/probe-run"
probe_publish="$scratch/probe-publish"
publish="$probe_publish"
resume_run="$scratch/resume-run"
state="$scratch/debug-state"
mkdir -p "$scratch" "$state"
data_dir=${VI_INPUT_OMNIVOICE_DATA:-${VI_INPUT_VOICETUNE_DATA:-}}

NAMES=([3]=upload [4]=venv [5]=base-model [6]=tokenise [7]=cuda-load [8]=train-steps
       [9]=ckpt-push [10]=listening [11]=resume)
# The stages each one needs (all of them must pass first).
NEEDS=([3]="" [4]="3" [5]="4" [6]="4 5" [7]="4 5" [8]="6 7" [9]="8" [10]="8" [11]="8")
# Seconds. Stage 4 builds torch into a venv, which is the slow one on a fresh box.
LIMITS=([3]=60 [4]=1200 [5]=900 [6]=600 [7]=600 [8]=900 [9]=600 [10]=600 [11]=600)

# ----- the real worker functions ----------------------------------------------------------
cut_functions() {
  local src="$jobs_dir/voicetune-fresh.sh" fn
  for fn in "$@"; do
    awk -v n="$fn" '$0 ~ "^" n "\\(\\) \\{" { p = 1 } p { print } p && /^\}/ { exit }' "$src"
  done
}
load_worker_code() {
  local code
  code=$(cut_functions publish_sync acked_checkpoints mark_pushed mark_bad push_repairs \
    push_publish weights_first push_two_part ovft_python audio_tokenizer_dir tokenise \
    tokenise_count_guard gpu_gate vram_peak_gate)
  FULL_ONLY_RE='  (optimizer\.bin|scheduler\.bin|random_states_0\.pkl)$'
  eval "$code"
  # shellcheck source=jobs/worker-limits.sh
  . "$jobs_dir/worker-limits.sh"
}

py_or_die() {
  [ -s "$state/py" ] || { echo "no venv python recorded by stage 4" >&2; exit 1; }
  PY=$(cat "$state/py")
  ACCELERATE_LAUNCH=("$PY" -m accelerate.commands.launch)
}
base_or_die() {
  [ -s "$state/base" ] || { echo "no Base model recorded by stage 5" >&2; exit 1; }
  VI_BASE_MODEL=$(cat "$state/base")
  export VI_BASE_MODEL
}

# ----- one stage (runs in its own process: bash debug-pass.sh --stage N) ------------------
stage_3() {
  echo "work dir: $work"
  echo "inputs: $(find "$work/inputs" -mindepth 1 -maxdepth 1 -printf '%f ' 2>/dev/null)"
  [ -n "$data_dir" ] && [ -f "$data_dir/data_config.json" ] || { echo "the data folder did not arrive (VI_INPUT_OMNIVOICE_DATA=$data_dir)" >&2; exit 1; }
  echo "data folder: $(find "$data_dir" -type f | wc -l) files, $(du -sh "$data_dir" | cut -f1)"
  for m in train_manifest dev_manifest style_manifest replay_texts; do
    [ -f "$data_dir/$m.jsonl" ] && echo "$m.jsonl: $(wc -l <"$data_dir/$m.jsonl") rows"
  done
  echo "return path: the token handshake passed (on-worker.sh starts this script only after it)"
  if [ "${VI_DEBUG_PREFLIGHT_RC:-0}" != 0 ]; then
    echo "note: worker-preflight.sh exited ${VI_DEBUG_PREFLIGHT_RC}; see stages 1-2 and the job log"
  fi
}

stage_4() {
  local mods
  PY=$(ovft_python)
  printf '%s\n' "$PY" >"$state/py"
  echo "venv python: $PY ($("$PY" --version 2>&1))"
  read -r -a mods <<<"${VI_DEBUG_IMPORTS:-torch accelerate transformers omnivoice webdataset soundfile huggingface_hub tensorboard}"
  "$PY" - "${mods[@]}" <<'PY'
import importlib, importlib.metadata as md, sys
bad = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
        try:
            v = md.version(name)
        except md.PackageNotFoundError:
            v = "?"
        print(f"import {name}: ok ({v})")
    except Exception as e:  # noqa: BLE001 - report every module, then fail
        bad.append(name)
        print(f"import {name}: FAILED {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
PY
  local probe="$state/launch_probe.py"
  printf '%s\n' 'import os, sys' \
    'print("launch probe: python", sys.executable, "rank", os.environ.get("RANK", "?"))' >"$probe"
  "$PY" -m accelerate.commands.launch --num_processes 1 "$probe"
  echo "accelerate launcher (python -m accelerate.commands.launch): ok"
  "$PY" -m omnivoice.scripts.extract_audio_tokens --help >/dev/null
  echo "tokeniser entry point (python -m omnivoice.scripts.extract_audio_tokens --help): ok"
  for f in train_ft.py ft_data.py infer_listening.py replay_public.py fetch_base_model.py; do
    "$PY" -m py_compile "$jobs_dir/omnivoice-ft/$f"
  done
  echo "worker scripts compile: ok"
}

stage_5() {
  local base
  py_or_die
  base=$(env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HF_HUB_OFFLINE \
    "$PY" "$jobs_dir/omnivoice-ft/fetch_base_model.py" \
    "${VI_BASE_MODEL_REPO:-k2-fsa/OmniVoice}" \
    "${VI_BASE_MODEL_REVISION:-c5fdb5ccb189668d56333f77ba2629f4cd7535f4}" \
    "$HOME/.cache/vi/base-model")
  [ -n "$base" ] && [ -f "$base/config.json" ] || { echo "fetch_base_model.py returned no usable folder" >&2; exit 1; }
  printf '%s\n' "$base" >"$state/base"
  echo "Base model at $base (revision ${VI_BASE_MODEL_REVISION:-c5fdb5ccb189668d56333f77ba2629f4cd7535f4}, size table checked)"
  echo "size: $(du -sh "$base" | cut -f1)"
}

stage_6() {
  py_or_die; base_or_die
  cd "$data_dir"
  VI_RUN_TIMEOUT_SECONDS=${stage_limit:-600} tokenise train "$data_dir/train_manifest.jsonl"
  VI_RUN_TIMEOUT_SECONDS=${stage_limit:-600} tokenise dev "$data_dir/dev_manifest.jsonl"
  echo "shards: train $(wc -l <"$data_dir/shards/train/data.lst") lines, dev $(wc -l <"$data_dir/shards/dev/data.lst") lines"
}

stage_7() {
  py_or_die; base_or_die
  cd "$data_dir"
  local dev=cuda
  if [ "${VI_DEBUG_NO_CUDA:-0}" = 1 ]; then
    echo "VI_DEBUG_NO_CUDA=1 (home rehearsal): gpu_gate skipped, replay on the CPU stubs"
    dev=cpu
  else
    gpu_gate
  fi
  if [ ! -s "$data_dir/shards/replay/data.lst" ]; then
    "$PY" "$jobs_dir/omnivoice-ft/replay_public.py" \
      --out-dir "$data_dir/replay" --base-model "$VI_BASE_MODEL" \
      --voices "$data_dir/voices" --texts "$data_dir/replay_texts.jsonl" --device "$dev"
  fi
  VI_RUN_TIMEOUT_SECONDS=${stage_limit:-600} tokenise replay "$data_dir/replay/replay_public.jsonl"
  echo "replay set: $(wc -l <"$data_dir/replay/replay_public.jsonl") rows (Base loaded on $dev)"
}

stage_8() {
  py_or_die; base_or_die
  cd "$data_dir"
  rm -rf "$probe_run"
  mkdir -p "$probe_run"
  local rc=0
  # The trainer's default batch_tokens and KL weight (the real run's), 5 steps. The minimum
  # batches-per-epoch guard is off only because the debug slice is deliberately small.
  "${ACCELERATE_LAUNCH[@]}" --num_processes 1 "$jobs_dir/omnivoice-ft/train_ft.py" \
    --init-from "$VI_BASE_MODEL" --data-config "$data_dir/data_config.json" \
    --output-dir "$probe_run" --steps 5 --save-steps 5 --logging-steps 1 \
    --min-batches-per-epoch 0 --num-workers "${VI_NUM_WORKERS:-2}" >"$probe_run/train.log" 2>&1 || rc=$?
  cat "$probe_run/train.log"
  [ "$rc" = 0 ] || { echo "train_ft.py exited $rc" >&2; exit "$rc"; }
  [ -f "$probe_run/checkpoint-5/SHA256SUMS" ] || { echo "no checkpoint-5 with SHA256SUMS" >&2; exit 1; }
  grep -E 'kl|KL|replay' "$probe_run/train.log" | tail -n 3 || true
  vram_peak_gate "$probe_run/train.log"
}

stage_9() {
  py_or_die
  publish_sync "$probe_run" "$probe_publish"
  [ -d "$probe_publish/checkpoint-5" ] || { echo "checkpoint-5 was not published" >&2; exit 1; }
  local bytes t0 t1
  bytes=$(du -sb "$probe_publish" | cut -f1)
  t0=$(date +%s)
  push_publish "$probe_publish"
  t1=$(date +%s)
  echo "pushed checkpoint-5 home: $bytes bytes in $((t1 - t0 + 1)) s (home verifies it after the stage)"
}

stage_10() {
  py_or_die; base_or_die
  cd "$data_dir"
  local out="$debug/listening"
  rm -rf "$out"
  mkdir -p "$out"
  "$PY" "$jobs_dir/omnivoice-ft/infer_listening.py" \
    --checkpoint "$probe_run/checkpoint-5" --out "$out" \
    --base-model "$VI_BASE_MODEL" \
    ${VI_AUDIO_TOKENIZER:+--audio-tokenizer "$VI_AUDIO_TOKENIZER"} \
    --style-manifest "$data_dir/style_manifest.jsonl" \
    --max-styles 1 --max-lines 2 --device "${VI_INFER_DEVICE:-cuda}"
  cat >"$out/LISTEN-README.txt" <<'TXT'
Debug pass listening set: a 5-step probe checkpoint (checkpoint-5) speaking two lines in one
style. It exists so a person can hear that the generate path works end to end.
It is for human listening only; nothing scores it, and it says nothing about voice quality
after real training.
TXT
  echo "listening set: $(find "$out" -name '*.wav' | wc -l) wav files in $out (comes home with the stage output)"
}

stage_11() {
  py_or_die; base_or_die
  cd "$data_dir"
  [ -f "$probe_run/checkpoint-5/SHA256SUMS" ] || { echo "no checkpoint-5 to resume from" >&2; exit 1; }
  rm -rf "$resume_run"
  mkdir -p "$resume_run"
  cp -al "$probe_run/checkpoint-5" "$resume_run/checkpoint-5"
  "${ACCELERATE_LAUNCH[@]}" --num_processes 1 "$jobs_dir/omnivoice-ft/train_ft.py" \
    --init-from "$VI_BASE_MODEL" --data-config "$data_dir/data_config.json" \
    --output-dir "$resume_run" --steps 7 --save-steps 7 --logging-steps 1 \
    --min-batches-per-epoch 0 --num-workers "${VI_NUM_WORKERS:-2}" \
    --resume-from-checkpoint "$resume_run/checkpoint-5"
  "$PY" "$jobs_dir/omnivoice-ft/train_ft.py" --verify-checkpoint "$resume_run/checkpoint-5" "$resume_run/checkpoint-7"
  echo "resumed from checkpoint-5 and verified checkpoint-7"
}

if [ "${1:-}" = --stage ]; then
  set -euo pipefail
  n=$2
  stage_limit=${3:-}
  load_worker_code
  "stage_$n"
  exit 0
fi

# ----- the driver -------------------------------------------------------------------------
excerpt_lines=${VI_DEBUG_EXCERPT_LINES:-12}
read -r -a order <<<"${VI_DEBUG_STAGES:-3 4 5 6 7 8 9 10 11}"
declare -A result=() seconds=() rcs=() why=()
load_worker_code
# Open files: soft limit up to the hard limit (at most 65536), inherited by every stage below.
vi_raise_nofile
echo "debug-pass: open files: soft $vi_lim_nofile_soft, hard $vi_lim_nofile_hard (raised to the hard limit, at most 65536)"
# Thread caps for every Python the stages start (the real stage does the same first). A box with
# too few task slots is reported, not fatal: stage 2 at home logs the limits too.
if ! worker_limits; then
  echo "debug-pass: note: worker_limits refused this box (see the line above); carrying on"
fi
echo "debug-pass: worker stages ${order[*]} start at $(date -u +%FT%TZ)"

for n in "${order[@]}"; do
  name=${NAMES[$n]:-}
  [ -n "$name" ] || { echo "debug-pass: unknown stage $n; skipped"; continue; }
  blocked=
  for m in ${NEEDS[$n]}; do
    if [ "${result[$m]:-}" != PASS ]; then blocked=$m; break; fi
  done
  log="$debug/$(printf '%02d' "$n")-$name.log"
  if [ -n "$blocked" ]; then
    result[$n]=SKIP-dependent; seconds[$n]=0; rcs[$n]=; why[$n]="needs stage $blocked ${NAMES[$blocked]}, which did not pass"
    echo "debug-pass: stage $n $name: SKIP-dependent (${why[$n]})"
    continue
  fi
  limit=${LIMITS[$n]}
  over="VI_DEBUG_TIMEOUT_$n"
  [ -z "${!over:-}" ] || limit=${!over}
  echo "debug-pass: stage $n $name: running (limit ${limit} s; log $log)"
  t0=$(date +%s)
  # GNU timeout signals the whole process group of the stage, so its Python children go too.
  timeout -k 15 "$limit" bash "$self" --stage "$n" "$limit" >"$log" 2>&1 </dev/null &
  pid=$!
  last=$t0
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${VI_DEBUG_POLL_SECONDS:-5}"
    now=$(date +%s)
    # A line now and then keeps on-worker.sh's stall watch (job.log must change) quiet.
    if [ $((now - last)) -ge "${VI_DEBUG_HEARTBEAT_SECONDS:-60}" ]; then
      echo "debug-pass: stage $n $name: still running ($((now - t0)) s; $(tail -n 1 "$log" 2>/dev/null | cut -c1-160))"
      last=$now
    fi
  done
  rc=0
  wait "$pid" || rc=$?
  seconds[$n]=$(( $(date +%s) - t0 ))
  rcs[$n]=$rc
  if [ "$rc" = 0 ]; then
    result[$n]=PASS; why[$n]=
    echo "debug-pass: stage $n $name: PASS (${seconds[$n]} s)"
  else
    result[$n]=FAIL
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
      why[$n]="hit its ${limit} s limit"
      echo "debug-pass: stage $n $name: FAIL (exit $rc, ${seconds[$n]} s; ${why[$n]})"
    else
      why[$n]="exit $rc"
      echo "debug-pass: stage $n $name: FAIL (exit $rc, ${seconds[$n]} s)"
    fi
  fi
  tail -n "$excerpt_lines" "$log" 2>/dev/null | cut -c1-300 | sed -e 's/^/debug-pass:   | /'
done

{
  for n in "${order[@]}"; do
    [ -n "${NAMES[$n]:-}" ] || continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "${NAMES[$n]}" "${result[$n]:-}" "${rcs[$n]:-}" "${seconds[$n]:-0}" "${why[$n]:-}"
  done
} | python3 -I -c '
import json, sys
rows = []
for line in sys.stdin:
    n, name, res, rc, secs, why = line.rstrip("\n").split("\t")
    rows.append({"stage": int(n), "name": name, "result": res, "rc": int(rc) if rc else None,
                 "seconds": int(secs), "why": why, "log": f"debug/{int(n):02d}-{name}.log"})
json.dump({"stages": rows, "listening_set": "debug/listening (for human listening only; never scored)"},
          open(sys.argv[1], "w"), indent=1)
' "$debug/summary.json"
echo "debug-pass: worker stages done; summary $debug/summary.json"
exit 0
