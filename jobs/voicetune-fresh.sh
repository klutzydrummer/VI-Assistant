#!/usr/bin/env bash
# Fresh voice-quality fine-tune (stage 5): one preflight job, then one train job, run on the GPU
# worker by jobs/on-worker.sh. Preflight proves the card, disk, generate path and a 5-step probe
# and pushes the probe checkpoint home; train runs 6 epochs from Base, pushes each finished epoch
# home as it lands, aborts on bad loss/memory/time, then builds the unblinded listening set and
# pushes it plus the run files home.
#
# Expected inputs (jobs/on-worker.sh maps stage `inputs` to VI_INPUT_<NAME>):
#   VI_INPUT_VOICETUNE_DATA/                 one prepared folder, built at home by the
#                                           voicetune_data stage, holding
#     train_with_codes.jsonl                 prepared train manifest (id, audio, text, ref_audio, audio_codes)
#     val_with_codes.jsonl                   held-out manifest, same shape
#     replay_with_codes.jsonl                replay lines for the KL term, same shape
#     sentences.txt                          the 3 fixed listening lines (optional; defaults below)
#     wav/ voices/                           the clips' and replay voices' audio, beside the manifests
# Every audio/ref_audio path is relative to that folder, so this script runs with it as the working
# directory. The 6 delivery clips are picked from the train manifest by seed 0 alone — never a label.
set -euo pipefail

jobs_dir=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-}

usage() {
  echo "usage: voicetune-fresh.sh preflight|train" >&2
  echo "  preflight  check --generate, gate VRAM/disk, run a 5-step probe, push it home, speak one line" >&2
  echo "  train      6 epochs from Base, push each epoch home, abort on bad loss/mem/time, build the platter" >&2
}

case "$MODE" in
  preflight|train) ;;
  *) usage; exit 2 ;;
esac

: "${VI_OUT:?voicetune-fresh.sh runs through jobs/on-worker.sh}" \
  "${VI_RETURN_PORT:?voicetune-fresh.sh needs the worker return tunnel}" \
  "${VI_INPUT_VOICETUNE_DATA:?missing VI_INPUT_VOICETUNE_DATA}"

# The prepared data folder; every manifest path inside it is relative to it, so
# run with it as the working directory for the job (all other paths are absolute).
data_dir="$VI_INPUT_VOICETUNE_DATA"
cd "$data_dir"

# qwen-tts pins an older transformers, so voicetune runs in its own venv — the same one
# vi_job.ensure_venv("requirements-tune.txt") builds (sha256[:16] of the requirements file).
# QWEN_TTS_PYTHON overrides for a pre-built environment.
tune_python() {
  if [ -n "${QWEN_TTS_PYTHON:-}" ]; then
    printf '%s\n' "$QWEN_TTS_PYTHON"
    return
  fi
  local req="$jobs_dir/requirements-tune.txt" venv
  venv="$HOME/.cache/vi/venv-$(sha256sum "$req" | cut -c1-16)"
  if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv --system-site-packages "$venv"
    "$venv/bin/python" -m pip install -q -r "$req"
  fi
  printf '%s\n' "$venv/bin/python"
}
PY=$(tune_python)

# The official recipe's finetuning/ folder (dataset.py), pinned to the commit the guide was
# read at (docs/design/qwen3tts-finetune-guide.md). voicetune.py's loader imports it.
qwen_rev=022e286b98fbec7e1e916cb940cdf532cd9f488e
qwen_ft="$HOME/.cache/vi/Qwen3-TTS-$qwen_rev/finetuning"
if [ ! -f "$qwen_ft/dataset.py" ]; then
  mkdir -p "$HOME/.cache/vi"
  curl -fsSL "https://codeload.github.com/QwenLM/Qwen3-TTS/tar.gz/$qwen_rev" | tar xz -C "$HOME/.cache/vi"
fi

# The 3 fixed listening lines: plain neutral sentences, not from any labelled set. The same three
# every epoch, so a line is generated the same across checkpoints and only the checkpoint differs.
if [ -s "$data_dir/sentences.txt" ]; then
  lines="$data_dir/sentences.txt"
else
  lines="$VI_OUT/sentences.txt"
  cat >"$lines" <<'LINES'
Let me check the readings.
The system is holding steady.
I'll be ready in a moment.
LINES
fi

# Worker-side scratch inside on-worker.sh's work dir (deleted with the job): run/publish/probe live
# here, not under VI_OUT, so epoch weights aren't re-sent over vi-out at the end.
scratch="$(cd "$(dirname "$VI_OUT")" && pwd)/voicetune-fresh"
run="$scratch/run"
publish="$scratch/publish"
probe_run="$scratch/probe-run"
probe_publish="$scratch/probe-publish"
mkdir -p "$scratch"

train_manifest="train_with_codes.jsonl"
val_manifest="val_with_codes.jsonl"
replay_manifest="replay_with_codes.jsonl"

title="Fine-tuned voice: is this Tali, with no reference clip?"
question="Is this Tali with her suit and accent? Note short-line glitches or speed-ups."

# push_publish DIR: push a publish dir's finished epochs (never a .part) home via
# the vi-publish module, which on-worker.sh points at this attempt's durable dir.
push_publish() {
  [ -d "$1" ] || return 0
  "$jobs_dir/worker-return.sh" push "$1" "${VI_PUBLISH:-vi-publish}" --exclude='*.part'
}

# abort_reason LOG ELAPSED_S -> the check_abort reason, or empty. Pure Python, no qwen-tts import.
abort_reason() {
  [ -f "$1" ] || return 0
  PYTHONPATH="$jobs_dir" "$PY" -c '
import json, sys, voicetune
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(voicetune.check_abort(recs, float(sys.argv[2])) or "")
' "$1" "$2"
}

# check_and_gate [--generate]: run check (writing check.json under VI_OUT), then gate VRAM and disk.
check_and_gate() {
  local gen=()
  [ "${1:-}" = "--generate" ] && gen=(--generate)
  "$PY" "$jobs_dir/voicetune.py" check --out "$VI_OUT/check" --work-dir "$scratch" "${gen[@]}"
  "$PY" - "$VI_OUT/check/check.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
vram, disk = c.get("vram_gb"), c.get("free_disk_gb")
if vram is None or vram < 44:
    print(f"check: VRAM {vram} GB < 44 GB", file=sys.stderr)
    sys.exit(1)
if disk is None or disk < 60:
    print(f"check: worker disk {disk} GB free < 60 GB", file=sys.stderr)
    sys.exit(1)
print(f"check: VRAM {vram} GB, disk {disk} GB free")
PY
}

preflight() {
  check_and_gate --generate

  # Five optimizer steps at the full batch/accum/kl, published into the probe dir, to measure the
  # peak memory of the real run before the paid one.
  "$PY" "$jobs_dir/voicetune.py" train --qwen-finetuning "$qwen_ft" \
    --data "$train_manifest" --val "$val_manifest" --replay "$replay_manifest" \
    --kl 0.1 --batch 8 --accum 4 --epochs 1 --max-steps 5 --seed 0 --speaker clip \
    --no-resume-file --publish "$probe_publish" --out "$probe_run"

  "$PY" - "$probe_run/train_log.jsonl" <<'PY'
import json, sys
peak = max((r.get("peak_gb", 0) or 0) for r in (json.loads(l) for l in open(sys.argv[1]) if l.strip()))
print(f"probe peak memory: {peak} GB")
if peak > 44:
    print(f"probe peak memory {peak} GB > 44 GB", file=sys.stderr)
    sys.exit(1)
PY

  # Push the probe checkpoint home, timed; record bytes/seconds in preflight.json, fail under 3 MB/s (Brandon lowered it from 8 on 2026-11-03: home measured 4 MB/s).
  local bytes seconds mb_s t0 t1
  bytes=$(du -sb "$probe_publish" | cut -f1)
  t0=$(date +%s%N)
  push_publish "$probe_publish"
  t1=$(date +%s%N)
  seconds=$(( (t1 - t0) / 1000000000 ))
  [ "$seconds" -gt 0 ] || seconds=1
  mb_s=$(( bytes / seconds / 1048576 ))
  "$PY" - "$VI_OUT/preflight.json" "$bytes" "$seconds" "$mb_s" <<'PY'
import json, sys
json.dump({"probe_bytes": int(sys.argv[2]), "seconds": int(sys.argv[3]), "mb_s": int(sys.argv[4])},
          open(sys.argv[1], "w"), indent=1)
PY
  if [ "$mb_s" -lt 3 ]; then
    echo "probe push was $mb_s MB/s < 3 MB/s" >&2
    exit 1
  fi

  # Speak one line from the probe epoch, so a real checkpoint's generate path is exercised too.
  head -n1 "$lines" >"$scratch/one-line.txt"
  "$PY" "$jobs_dir/voicetune.py" speak --run "$probe_run" --epoch 0 \
    --sentences "$scratch/one-line.txt" --seed 0 --out "$VI_OUT/probe-speak"
  rm -rf "$probe_run/voice-epoch-0"

  echo "preflight: ok"
}

train() {
  # Preflight already proved generate, so this gates VRAM/disk without the extra model load.
  check_and_gate

  local start reason elapsed rc
  start=$(date +%s)
  "$PY" "$jobs_dir/voicetune.py" train --qwen-finetuning "$qwen_ft" \
    --data "$train_manifest" --val "$val_manifest" --replay "$replay_manifest" \
    --epochs 6 --lr 2e-6 --kl 0.1 --batch 8 --accum 4 --seed 0 --speaker clip \
    --no-resume-file --publish "$publish" --out "$run" &
  train_pid=$!
  # If this script exits while train still runs (a push failure, an abort, or a
  # signal), kill it so the GPU job never keeps burning money orphaned.
  trap 'kill "$train_pid" 2>/dev/null || true' EXIT

  # Watchdog: push each finished epoch home as it appears, and stop the run on the abort rules.
  reason=""
  while kill -0 "$train_pid" 2>/dev/null; do
    # A loop push failure must not orphan the running train, so it is non-fatal
    # here; the final push below is the one that decides success.
    push_publish "$publish" || echo "voicetune-fresh: publish push failed; will retry" >&2
    elapsed=$(( $(date +%s) - start ))
    reason=$(abort_reason "$run/train_log.jsonl" "$elapsed")
    if [ -n "$reason" ]; then
      echo "train abort: $reason" >&2
      kill "$train_pid" 2>/dev/null || true
      break
    fi
    sleep 15
  done
  set +e
  wait "$train_pid"
  rc=$?
  set -e
  trap - EXIT
  push_publish "$publish"  # push what exists (the final epochs, or the ones before an abort)
  if [ -n "$reason" ]; then
    echo "train aborted: $reason" >&2
    exit 1
  fi
  [ "$rc" -eq 0 ] || { echo "train failed: exit $rc" >&2; exit "$rc"; }

  # The 6 delivery clips, chosen from the train manifest by seed 0 alone (never a label/mood tag).
  mapfile -t deliveries < <(PYTHONPATH="$jobs_dir" "$PY" -c '
import json, sys, voicetune
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for cid, path in voicetune.delivery_clips(rows, 0, 6):
    print(cid + "=" + path)
' "$train_manifest")
  [ "${#deliveries[@]}" -eq 6 ] || { echo "train: expected 6 delivery clips, got ${#deliveries[@]}" >&2; exit 1; }

  # Unblinded listening set: 3 fixed lines at every epoch (18 clips), plus 3 delivery clips at each
  # of the last two epochs, each speaking line 0 with that clip's own x-vector in the slot (6 clips).
  local platter="$VI_OUT/platter" epoch one delivery_args dv
  mkdir -p "$platter"
  for epoch in 0 1 2 3 4 5; do
    "$PY" "$jobs_dir/voicetune.py" speak --run "$run" --epoch "$epoch" \
      --sentences "$lines" --seed 0 --unblind --title "$title" --question "$question" \
      --out "$platter"
    if [ "$epoch" -ge 4 ]; then
      one="$scratch/line0.txt"
      head -n1 "$lines" >"$one"
      delivery_args=()
      case "$epoch" in
        4) for dv in "${deliveries[0]}" "${deliveries[1]}" "${deliveries[2]}"; do delivery_args+=(--delivery-from "$dv"); done ;;
        5) for dv in "${deliveries[3]}" "${deliveries[4]}" "${deliveries[5]}"; do delivery_args+=(--delivery-from "$dv"); done ;;
      esac
      "$PY" "$jobs_dir/voicetune.py" speak --run "$run" --epoch "$epoch" \
        --sentences "$one" --seed 0 --unblind --title "$title" --question "$question" \
        "${delivery_args[@]}" --out "$platter"
    fi
    rm -rf "$run/voice-epoch-$epoch"  # the ~3.4 GB voice-model copy, deleted after speaking
  done

  # Run files for the record, beside the platter (check.json is already under VI_OUT/check).
  cp "$run/voice.json" "$run/xvectors.json" "$run/train_log.jsonl" "$VI_OUT/"
  cp "$VI_OUT/check/check.json" "$VI_OUT/" 2>/dev/null || true

  echo "train: done; listening set in $platter"
}

case "$MODE" in
  preflight) preflight ;;
  train) train ;;
esac
