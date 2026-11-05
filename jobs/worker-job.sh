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
# A failed attempt is restarted with --resume from the last finished epoch
# (resume.pt), up to VI_TRAIN_RETRIES (default 2). It is never restarted after
# an abort-rule stop ($run/abort.json, written by voicetune-fresh.sh before it
# kills the trainer), on ENOSPC, or on a lock collision (exit 75). Exit code
# 137 alone is not taken as OOM: the cgroup v2 oom_kill counter must have risen
# between the start and the end of the attempt, and only then does the retry
# run with VI_NUM_WORKERS=0. Images without cgroup v2 memory.events retry
# unchanged once.
#
# The rc file outlives a session that died. Relaunching a stopped job belongs
# to the home supervisor, a later task; this wrapper only records enough to
# make that relaunch safe.
#
#   worker-job.sh --work DIR --run DIR [--publish DIR] [--retries N] --
#     python3 jobs/voicetune.py train ...
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

cmd=("$@")
attempt=0 resumed=0 rc=0
while :; do
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
  [ $attempt -ge $retries ] && break
  if [ $has_oom = 0 ]; then
    # No cgroup v2 memory.events (old container or cgroup v1): retry once,
    # unchanged, rather than guess that the failure was an OOM.
    [ $attempt -ge 1 ] && break
  else
    after=$(oom_kill)
    after=${after:-0}
    if [ "$after" -gt "$before" ]; then
      export VI_NUM_WORKERS=0
      echo "worker-job.sh: oom_kill $before -> $after; retrying with VI_NUM_WORKERS=0" >&2
    fi
  fi
  attempt=$((attempt + 1))
  if [ $resumed = 0 ] && [ -s "$run/resume.pt" ]; then
    cmd+=(--resume)
    resumed=1
  fi
  echo "worker-job.sh: retry $attempt/$retries" >&2
done
echo "$rc" >"$work/rc"
exit "$rc"
