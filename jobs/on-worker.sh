#!/usr/bin/env bash
# Runs one dataset stage on the rented GPU worker instead of here.
#
#   [stage.label]
#   command = ["../../jobs/on-worker.sh", "--put", "states.toml", "python3", "jobs/label.py"]
#
# It copies this jobs/ folder, the stage's settings, each VI_INPUT_* folder and
# any --put files to a fresh folder on the worker, runs the command there with
# the same VI_* variables pointing at the copies, copies VI_OUT back, and deletes
# the worker's folder whether the job worked or not. The job sees:
#   jobs/            this folder
#   files/<name>     each --put file, by its base name
#   VI_STAGE, VI_OUT, VI_SETTINGS, VI_INPUT_<NAME>, VI_CHECKPOINT, VI_WORKER=1
#
# VI_CHECKPOINT goes up with whatever an earlier, failed run of the same stage
# left in it, and comes home every VI_SYNC_SECONDS (default 60) while the job
# runs and once more when it ends. So if the worker dies or the credit runs
# out, a rerun of `vi dataset run` on a new worker resumes from the last copy.
# VI_CORPUS isn't sent: a GPU job gets only what its inputs and --put name.
#
# The worker comes from `vi cloud env` (run `vi cloud up` first) unless
# VI_WORKER_HOST is already set. VI_WORKER_HOST=local runs the "worker" in a
# folder on this machine (VI_WORKER_ROOT), for tests.
set -euo pipefail

jobs_dir=$(cd "$(dirname "$0")" && pwd)
puts=()
while [ "${1-}" = --put ]; do
  [ $# -ge 2 ] || { echo "on-worker.sh: --put needs a file" >&2; exit 2; }
  puts+=("$2")
  shift 2
done
if [ $# -eq 0 ]; then
  echo "usage: on-worker.sh [--put FILE]... COMMAND [ARG]..." >&2
  exit 2
fi
: "${VI_STAGE:?on-worker.sh runs from \`vi dataset run\`}" "${VI_OUT:?}" "${VI_SETTINGS:?}"
vi=${VI_BIN:-vi}

if [ -z "${VI_WORKER_HOST-}" ]; then
  vars=$("$vi" cloud env)
  eval "$vars"
fi
root=${VI_WORKER_ROOT:-/root/vi-jobs}
work="$root/$VI_STAGE.$(date +%s).$$"
q() { printf '%q' "$1"; }

if [ "$VI_WORKER_HOST" = local ]; then
  remote() { bash -c "$1"; }
  push() { rsync -a "$1" "$2"; }
  pull() { rsync -a "${@:3}" "$1" "$2"; }
else
  ssh_opts=(-p "$VI_WORKER_PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new
    -o "UserKnownHostsFile=$VI_WORKER_KNOWN_HOSTS" -o ServerAliveInterval=30)
  target="root@$VI_WORKER_HOST"
  remote() { ssh "${ssh_opts[@]}" "$target" "$1"; }
  push() { rsync -a -e "ssh ${ssh_opts[*]}" "$1" "$target:$2"; }
  pull() { rsync -a -e "ssh ${ssh_opts[*]}" "${@:3}" "$target:$1" "$2"; }
  # Keep home's idle clock running while the job does.
  (while sleep 60 >/dev/null 2>&1; do "$vi" cloud touch >/dev/null 2>&1 || true; done) &
  heartbeat=$!
fi

cleanup() {
  [ -n "${heartbeat-}" ] && kill "$heartbeat" 2>/dev/null
  [ -n "${syncer-}" ] && kill "$syncer" 2>/dev/null
  remote "rm -rf $(q "$work")" ||
    echo "on-worker.sh: could not delete $work on the worker; \`vi cloud down\` removes it with the instance" >&2
}
trap cleanup EXIT

remote "mkdir -p $(q "$work")/inputs $(q "$work")/out $(q "$work")/files && touch $(q "$root/.active")"
push "$jobs_dir/" "$work/jobs/"
push "$VI_SETTINGS" "$work/settings.json"
checkpoint=${VI_CHECKPOINT:-}
if [ -n "$checkpoint" ]; then
  mkdir -p "$checkpoint"
  # A copy taken mid-write can end in half a line; drop it before resuming.
  for f in "$checkpoint"/*.jsonl; do
    if [ -s "$f" ] && [ -n "$(tail -c1 "$f")" ]; then
      truncate -s $(($(wc -c <"$f") - $(tail -n1 "$f" | wc -c))) "$f"
    fi
  done
  push "$checkpoint/" "$work/checkpoint/"
fi
for f in "${puts[@]}"; do
  push "$f" "$work/files/"
done
exports="VI_STAGE=$(q "$VI_STAGE") VI_OUT=$(q "$work/out") VI_SETTINGS=$(q "$work/settings.json") VI_WORKER=1"
[ -n "$checkpoint" ] && exports+=" VI_CHECKPOINT=$(q "$work/checkpoint")"
for var in $(compgen -e | grep '^VI_INPUT_' || true); do
  name=${var#VI_INPUT_}
  push "${!var}/" "$work/inputs/$name/"
  exports+=" $var=$(q "$work/inputs/$name")"
done

cmd=""
for arg in "$@"; do
  cmd+=" $(q "$arg")"
done
active=$(q "$root/.active")
# The worker's watchdog releases it after idle_minutes without this file
# changing, so touch it while the job runs.
script="cd $(q "$work") && export $exports && {
  (while sleep 60; do touch $active; done) >/dev/null 2>&1 &
  hb=\$!
  $cmd
  rc=\$?
  kill \$hb
  touch $active
  exit \$rc
}"

# Checkpoint files only grow, so --append sends just the new lines instead of
# rereading whole files on home's busy disk every minute. That needs home's
# copy to be a prefix of the worker's, which the trimming above ensures.
sync_home() { [ -z "$checkpoint" ] || pull "$work/checkpoint/" "$checkpoint/" --append; }
if [ -n "$checkpoint" ]; then
  (while sleep "${VI_SYNC_SECONDS:-60}" >/dev/null 2>&1; do
    sync_home >/dev/null 2>&1 || echo "on-worker.sh: could not copy the checkpoint home; will retry" >&2
  done) &
  syncer=$!
fi

set +e
remote "$script"
rc=$?
set -e
kill "${syncer-}" 2>/dev/null || true
syncer=
sync_home || echo "on-worker.sh: could not copy the final checkpoint home; the last periodic copy is kept" >&2
if [ $rc -eq 0 ]; then
  pull "$work/out/" "$VI_OUT/"
else
  echo "on-worker.sh: $1 exited with $rc on the worker" >&2
fi
exit $rc
