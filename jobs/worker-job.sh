#!/usr/bin/env bash
# Runs one long worker job (training) away from the stage session.
#
# The wrapper re-executes itself in a new session (setsid -f) and runs the
# command in its own session too, so losing the ssh session does not stop the
# job. Its output goes to $work/job.log. The wrapper holds $work/job.lock for
# its lifetime, writes the trainer's pid to $work/train.pid, and writes the
# final exit code to $work/rc, so the watchdog can tell a live job from a dead
# one even while the owning ssh session is gone.
#
# It also copies the checkpoint home every VI_SYNC_SECONDS (default 60) while
# the trainer runs, and clears a $work/pending/epoch-<k> marker once the home
# supervisor writes $work/acked/epoch-<k>. Both live here, not in the stage
# supervisor: this runner outlives a killed stage session.
#
# A failed attempt is restarted with --resume-from-checkpoint from the newest
# checkpoint-<step> with its SHA256SUMS (a checkpoint is complete only once its
# manifest exists), up to VI_TRAIN_RETRIES (default 2). It is never restarted after
# an abort-rule stop ($run/abort.json, written by voicetune-fresh.sh before it
# kills the trainer), on ENOSPC, or on a lock collision (exit 75). Exit code
# 137 alone is not taken as OOM: the cgroup v2 oom_kill counter must have risen
# between the start and the end of the attempt, and only then does the retry
# run with VI_NUM_WORKERS=0. Images without cgroup v2 memory.events retry
# unchanged once.
#
# A GPU out-of-memory is exit 76 from train_ft.py, which also writes $run/oom.json. The
# accelerate launcher turns any non-zero child exit into its own exit 1 (r23), so an attempt
# also counts as an OOM when it failed and $run/oom.json was written during it (its
# modification time changed since the attempt started); the trainer's own file, not
# accelerate's message format, is the evidence. An OOM retry halves --batch-tokens and
# doubles --gradient-accumulation-steps (the same tokens per optimizer step) and resumes;
# below a batch of 2048 tokens it stops, and the job's rc is then 76 whatever the launcher
# returned. A command without the flags runs the trainer's defaults, read from
# omnivoice-ft/ft_guards.py (r25: 2048 x 4, already the floor, so its first OOM stops). A retry that fails with the same last traceback line as the attempt before it is
# deterministic and is not repeated (an OOM is exempt: its batch changed).
# Every retry decision is appended to $work/retries.jsonl.
#
# The rc file outlives a session that died. Relaunching a stopped job belongs
# to the home supervisor, a later task; this wrapper only records enough to
# make that relaunch safe.
#
#   worker-job.sh --work DIR --run DIR [--publish DIR] [--retries N] --
#     python -m accelerate.commands.launch --num_processes 1 jobs/omnivoice-ft/train_ft.py ...
set -euo pipefail

work= run= publish= retries=${VI_TRAIN_RETRIES:-2}
jobs_dir=$(cd "$(dirname "$0")" && pwd)
detached=
if [ "${1-}" = --detached ]; then
  detached=1
  shift
fi
while [ $# -gt 0 ]; do
  case $1 in
    --work) work=${2:?--work needs a directory}; shift 2 ;;
    --run) run=${2:?--run needs a directory}; shift 2 ;;
    --publish) publish=${2:?--publish needs a directory}; shift 2 ;;
    --retries) retries=${2:?--retries needs a number}; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [ -z "$work" ] || [ -z "$run" ] || [ $# -eq 0 ]; then
  echo "usage: worker-job.sh --work DIR --run DIR [--publish DIR] [--retries N] -- COMMAND [ARG]..." >&2
  exit 2
fi

# Row 30: the trainer's DataLoader forks workers, so Hugging Face's tokenizer
# would otherwise warn about (and try to manage) parallelism in each of them.
export TOKENIZERS_PARALLELISM=false

if [ -z "$detached" ]; then
  # The foreground half starts the detached runner and waits for its rc file;
  # a stage-session signal can kill this half without touching the job.
  mkdir -p "$work" "$run"
  args=(--detached --work "$work" --run "$run" --retries "$retries")
  [ -z "$publish" ] || args+=(--publish "$publish")
  setsid -f bash "$0" "${args[@]}" -- "$@" >>"$work/job.log" 2>&1 </dev/null
  for _ in {1..10}; do
    [ -s "$work/worker.pid" ] && break
    sleep 1
  done
  [ -s "$work/worker.pid" ] || { echo "worker-job.sh: the detached runner did not start" >&2; echo 1 >"$work/rc"; exit 1; }
  while [ ! -s "$work/rc" ]; do
    if ! kill -0 "$(cat "$work/worker.pid")" 2>/dev/null; then
      sleep 1
      [ -s "$work/rc" ] || { echo "worker-job.sh: the detached runner died without writing $work/rc" >&2; echo 1 >"$work/rc"; exit 1; }
      break
    fi
    sleep 1
  done
  exit "$(cat "$work/rc")"
fi

# The detached runner.
echo $$ >"$work/worker.pid"
mkdir -p "$run"
exec 9>>"$work/job.lock"
if ! flock -n -E 75 9; then
  echo "worker-job.sh: $work/job.lock is held by another job" >&2
  echo 75 >"$work/rc"
  exit 75
fi

# cgroup v2; a container that hides it (or an old cgroup v1 host) has no file.
oom_file=${VI_MEMORY_EVENTS:-/sys/fs/cgroup/memory.events}
has_oom=0
[ -r "$oom_file" ] && has_oom=1
oom_kill() { awk '$1 == "oom_kill" { print $2; exit }' "$oom_file" 2>/dev/null || true; }

# A published epoch is un-acked work: mark it for the watchdog. The home
# supervisor acks each epoch after it verifies the copy home; the marker goes
# away then, and an already-acked epoch never gets a fresh marker.
watch= sync=
if [ -n "$publish" ]; then
  mkdir -p "$work/pending"
  (
    while :; do
      for e in "$publish"/epoch-*; do
        [ -d "$e" ] || continue
        case ${e##*/} in *.part) continue ;; esac
        n=${e##*/epoch-}
        [ -e "$work/acked/epoch-$n" ] && continue
        [ -e "$work/pending/epoch-$n" ] || : >"$work/pending/epoch-$n"
      done
      for p in "$work/pending"/epoch-*; do
        [ -e "$p" ] || continue
        n=${p##*epoch-}
        [ -e "$work/acked/epoch-$n" ] && rm -f "$p"
      done
      sleep 5
    done
  ) &
  watch=$!
fi

# The checkpoint only grows, so --append sends the new tail instead of
# rereading the whole file on home's busy disk. epoch-<k> weights are published
# separately and skipped here. on-worker.sh still takes the final copy after
# the stage ends; this loop covers a home supervisor that is paused or killed.
sync_src=${VI_CHECKPOINT:-$work/checkpoint}
if [ -d "$sync_src" ] && { [ -e "$work/.return-port" ] || [ -n "${VI_RETURN_PORT:-}" ]; }; then
  (
    while :; do
      sleep "${VI_SYNC_SECONDS:-60}" || exit 0
      "$jobs_dir/worker-return.sh" push "$sync_src/" vi-checkpoint \
        --append --exclude='epoch-*/' >/dev/null 2>&1 ||
        echo "worker-job.sh: could not copy the checkpoint home; will retry" >&2
    done
  ) &
  sync=$!
fi
trap 'kill "${watch-}" "${sync-}" 2>/dev/null || true' EXIT

base=("$@")
cmd=("${base[@]}")
attempt=0 rc=0
min_batch_tokens=${VI_MIN_BATCH_TOKENS:-2048}
# The trainer's default batch (r25: 2048 tokens x 4 accumulation), read from ft_guards.py, its
# single source, for a command that does not pass the flag. Empty when it cannot be read.
ft_default() {
  python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import ft_guards as F; print(int(getattr(F, sys.argv[2])))' \
    "$jobs_dir/omnivoice-ft" "$1" 2>/dev/null || true
}
retries_log=$work/retries.jsonl

json_str() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | tr -d '\n\r'; }
# log_retry ATTEMPT RC REASON SIG: one JSON line per decision, for the diagnostics.
log_retry() {
  printf '{"attempt":%s,"rc":%s,"reason":"%s","signature":"%s","time":"%s"}\n' \
    "$1" "$2" "$(json_str "$3")" "$(json_str "$4")" "$(date -u +%FT%TZ)" >>"$retries_log" || true
}
# last_frame OFFSET: the last traceback frame line this attempt wrote to job.log.
last_frame() {
  tail -c +"$(($1 + 1))" "$work/job.log" 2>/dev/null |
    grep -E '^ +File ".*", line [0-9]+, in ' | tail -n1 || true
}
# get_flag NAME DEFAULT: the value of --NAME in base (either "--NAME V" or "--NAME=V").
get_flag() {
  local i
  for ((i = 0; i < ${#base[@]}; i++)); do
    case ${base[i]} in
      "$1") echo "${base[i + 1]}"; return ;;
      "$1"=*) echo "${base[i]#*=}"; return ;;
    esac
  done
  echo "$2"
}
# set_flag NAME VALUE: replace --NAME's value in base, or append the flag.
set_flag() {
  local i
  for ((i = 0; i < ${#base[@]}; i++)); do
    case ${base[i]} in
      "$1") base[i + 1]=$2; return ;;
      "$1"=*) base[i]="$1=$2"; return ;;
    esac
  done
  base+=("$1" "$2")
}
# oom_stamp: $run/oom.json's modification time to the nanosecond, or "none".
oom_stamp() { stat -c %y "$run/oom.json" 2>/dev/null || echo none; }
prev_sig=
while :; do
  log_off=$(stat -c %s "$work/job.log" 2>/dev/null || echo 0)
  oom_before=$(oom_stamp)
  before=$(oom_kill)
  before=${before:-0}
  # The trainer gets its own session, so it survives this wrapper too; it
  # writes train.pid itself (same pid after exec) for the watchdog and the
  # abort kill in voicetune-fresh.sh.
  setsid -w bash -c 'echo $$ >"$1"; shift; exec "$@"' _ "$work/train.pid" "${cmd[@]}" &
  rc=0
  wait $! || rc=$?
  [ $rc -eq 0 ] && break
  if [ -e "$run/abort.json" ]; then
    echo "worker-job.sh: abort.json exists; not retrying" >&2
    break
  fi
  if grep -q 'No space left on device' "$work/job.log" 2>/dev/null; then
    echo "worker-job.sh: ENOSPC; not retrying" >&2
    break
  fi
  [ $rc -eq 75 ] && break
  # Through accelerate the trainer's 76 arrives as 1; oom.json written in this attempt says OOM.
  if [ $rc -ne 76 ] && [ "$(oom_stamp)" != none ] && [ "$(oom_stamp)" != "$oom_before" ]; then
    echo "worker-job.sh: exit $rc with $run/oom.json written in this attempt: GPU out of memory (exit 76)" >&2
    rc=76
  fi
  sig=$(last_frame "$log_off")
  if [ $rc -ne 76 ] && [ -n "$sig" ] && [ "$sig" = "$prev_sig" ]; then
    log_retry "$attempt" "$rc" "same-traceback-stop" "$sig"
    echo "worker-job.sh: the same traceback line repeated; not retrying" >&2
    break
  fi
  prev_sig=$sig
  if [ $attempt -ge $retries ]; then
    log_retry "$attempt" "$rc" "retries-exhausted" "$sig"
    break
  fi
  if [ $rc -eq 76 ]; then
    bt=$(get_flag --batch-tokens "$(ft_default DEFAULT_BATCH_TOKENS)")
    ga=$(get_flag --gradient-accumulation-steps "$(ft_default DEFAULT_GRAD_ACCUM)")
    if ! [[ "$bt" =~ ^[0-9]+$ && "$ga" =~ ^[0-9]+$ ]]; then
      log_retry "$attempt" "$rc" "oom-unknown-batch" "batch_tokens=$bt grad_accum=$ga"
      echo "worker-job.sh: GPU out of memory, but the batch in use could not be read (ft_guards.py); stopping" >&2
      break
    fi
    if [ $((bt / 2)) -lt "$min_batch_tokens" ]; then
      log_retry "$attempt" "$rc" "oom-at-floor" "batch_tokens=$bt"
      echo "worker-job.sh: GPU out of memory at batch_tokens $bt; below the $min_batch_tokens floor, stopping" >&2
      break
    fi
    set_flag --batch-tokens $((bt / 2))
    set_flag --gradient-accumulation-steps $((ga * 2))
    log_retry "$((attempt + 1))" "$rc" "oom-halve" "batch_tokens=$((bt / 2)) grad_accum=$((ga * 2))"
    echo "worker-job.sh: GPU out of memory; batch_tokens $bt -> $((bt / 2)), grad accumulation $ga -> $((ga * 2))" >&2
  elif [ $has_oom = 0 ]; then
    # No cgroup v2 memory.events (old container or cgroup v1): retry once,
    # unchanged, rather than guess that the failure was an OOM.
    if [ $attempt -ge 1 ]; then
      log_retry "$attempt" "$rc" "no-cgroup-retry-once" "$sig"
      break
    fi
    log_retry "$((attempt + 1))" "$rc" "crash" "$sig"
  else
    after=$(oom_kill)
    after=${after:-0}
    if [ "$after" -gt "$before" ]; then
      log_retry "$((attempt + 1))" "$rc" "host-oom" "oom_kill $before -> $after"
      export VI_NUM_WORKERS=0
      echo "worker-job.sh: oom_kill $before -> $after; retrying with VI_NUM_WORKERS=0" >&2
    else
      log_retry "$((attempt + 1))" "$rc" "crash" "$sig"
    fi
  fi
  attempt=$((attempt + 1))
  # Every retry resumes from the newest checkpoint-<step> whose SHA256SUMS exists
  # (the manifest is written last, so a dir without one is a partial save), not the
  # one the first retry used. Every loop body ends with a zero status: under
  # set -euo pipefail a failing substitution would kill this runner before it
  # writes rc.
  resume_ckpt=$({
    for c in "$run"/checkpoint-*; do
      if [ -d "$c" ] && [ -f "$c/SHA256SUMS" ]; then
        printf '%s\n' "$c"
      fi
    done
  } | sort -V | tail -n1) || resume_ckpt=""
  cmd=("${base[@]}")
  [ -z "$resume_ckpt" ] || cmd+=(--resume-from-checkpoint "$resume_ckpt")
  echo "worker-job.sh: retry $attempt/$retries" >&2
done
echo "$rc" >"$work/rc"
exit "$rc"
