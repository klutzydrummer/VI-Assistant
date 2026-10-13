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
#   VI_STAGE, VI_OUT, VI_SETTINGS, VI_INPUT_<NAME>, VI_CHECKPOINT, and VI_WORKER=1
# VI_CORPUS isn't sent: a GPU job gets only what its inputs and --put name.
#
# Checkpoints: the runner's VI_CHECKPOINT folder goes up with the job and comes
# back home every VI_CHECKPOINT_EVERY seconds (default 120) while it runs, and
# once more when it ends, worked or not. If the worker dies mid-job (credit runs
# out, the machine is lost), the next run of the stage starts from the last copy
# that reached home; the jobs skip what their checkpoint already has.
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
  pull() { rsync -a "$1" "$2"; }
else
  ssh_opts=(-p "$VI_WORKER_PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new
    -o "UserKnownHostsFile=$VI_WORKER_KNOWN_HOSTS" -o ServerAliveInterval=30)
  target="root@$VI_WORKER_HOST"
  remote() { ssh "${ssh_opts[@]}" "$target" "$1"; }
  push() { rsync -a -e "ssh ${ssh_opts[*]}" "$1" "$target:$2"; }
  pull() { rsync -a -e "ssh ${ssh_opts[*]}" "$target:$1" "$2"; }
  # Keep home's idle clock running while the job does.
  # Background loops get no stdout/stderr: a leftover `sleep` holding the
  # runner's pipes open would keep the stage from finishing.
  (while sleep 60; do "$vi" cloud touch || true; done) >/dev/null 2>&1 &
  heartbeat=$!
fi

cleanup() {
  [ -n "${heartbeat-}" ] && kill "$heartbeat" 2>/dev/null
  if [ -n "${saver-}" ]; then
    kill "$saver" 2>/dev/null || true
    wait "$saver" 2>/dev/null || true
    save_checkpoint ||
      echo "on-worker.sh: could not copy the checkpoint home; the last periodic copy stands" >&2
  fi
  remote "rm -rf $(q "$work")" ||
    echo "on-worker.sh: could not delete $work on the worker; \`vi cloud down\` removes it with the instance" >&2
}
trap cleanup EXIT

remote "mkdir -p $(q "$work")/inputs $(q "$work")/out $(q "$work")/files && touch $(q "$root/.active")"
push "$jobs_dir/" "$work/jobs/"
push "$VI_SETTINGS" "$work/settings.json"
for f in "${puts[@]}"; do
  push "$f" "$work/files/"
done
exports="VI_STAGE=$(q "$VI_STAGE") VI_OUT=$(q "$work/out") VI_SETTINGS=$(q "$work/settings.json") VI_WORKER=1"
if [ -n "${VI_CHECKPOINT-}" ]; then
  mkdir -p "$VI_CHECKPOINT"
  push "$VI_CHECKPOINT/" "$work/checkpoint/"
  exports+=" VI_CHECKPOINT=$(q "$work/checkpoint")"
  save_checkpoint() { pull "$work/checkpoint/" "$VI_CHECKPOINT/"; }
  (while sleep "${VI_CHECKPOINT_EVERY:-120}"; do save_checkpoint || true; done) >/dev/null 2>&1 &
  saver=$!
fi
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

set +e
remote "$script"
rc=$?
set -e
if [ $rc -eq 0 ]; then
  pull "$work/out/" "$VI_OUT/"
else
  echo "on-worker.sh: $1 exited with $rc on the worker" >&2
fi
exit $rc
