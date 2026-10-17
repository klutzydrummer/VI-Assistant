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
# Inputs over VI_PACK_MIN_KB (default 1024) are packed at home first, before
# the worker is even looked up, into one zstd'd tar each, kept in
# VI_BUNDLE_DIR (default ~/.cache/vi/bundles, at most VI_BUNDLE_CACHE_MB,
# default 4000) for reruns. Then a timed 8 MB upload estimates how long the
# inputs will take, and the stage stops before sending anything if that's over
# VI_MAX_UPLOAD_MINUTES (default 30) unless VI_UPLOAD_ANYWAY=1.
#
# The worker comes from `vi cloud env` (run `vi cloud up` first) unless
# VI_WORKER_HOST is already set; VI_WAIT_FOR_WORKER=MINUTES waits that long
# for one, so the packing can run before anything is rented.
# VI_WORKER_HOST=local runs the "worker" in a folder on this machine
# (VI_WORKER_ROOT), for tests.
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

# Pack each large input into one zstd'd tar at home before touching the
# worker. Home's disk is slow (spinning, encrypted, busy) and a stage input
# is thousands of small files, so reading them is the slow part of an upload;
# done first, it happens while no GPU is billed, and the upload is then one
# sequential file per input. Stage outputs never change, so a bundle is kept
# in VI_BUNDLE_DIR, keyed by the folder's path and file listing, and a rerun
# (or the run after `vi cloud up`) reuses it.
bundle_dir=${VI_BUNDLE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/vi/bundles}
pack_min_kb=${VI_PACK_MIN_KB:-1024}
declare -A bundle=()
packed_bytes=0
if command -v zstd >/dev/null 2>&1; then
  for var in $(compgen -e | grep '^VI_INPUT_' || true); do
    dir=${!var}
    kb=$(du -sk "$dir" | cut -f1)
    [ "$kb" -ge "$pack_min_kb" ] || continue
    key=$({ cd "$dir" && pwd -P && find . -printf '%P %s %T@\n' | LC_ALL=C sort; } | sha256sum | cut -c1-16)
    f="$bundle_dir/$key.tar.zst"
    if [ -s "$f" ]; then
      touch "$f"
      echo "on-worker.sh: ${var#VI_INPUT_} already packed ($(($(stat -c %s "$f") / 1048576)) MB)" >&2
    else
      mkdir -p "$bundle_dir"
      t0=$(date +%s%N)
      tar -C "$dir" -cf - . | zstd -q -T0 -3 -o "$f.part"
      mv "$f.part" "$f"
      ms=$((($(date +%s%N) - t0) / 1000000 + 1))
      echo "on-worker.sh: packed ${var#VI_INPUT_}: $((kb / 1024)) MB read in $((ms / 1000)) s ($((kb * 1000 / ms / 1024)) MB/s from disk), $(($(stat -c %s "$f") / 1048576)) MB to send" >&2
    fi
    bundle[$var]=$f
    packed_bytes=$((packed_bytes + $(stat -c %s "$f")))
  done
  # Keep the cache under VI_BUNDLE_CACHE_MB, dropping the least recently used
  # bundles that this run doesn't need.
  limit=$((${VI_BUNDLE_CACHE_MB:-4000} * 1048576))
  in_use=" ${bundle[*]} "
  total=0
  while read -r _ size path; do
    total=$((total + size))
    if [ "$total" -gt "$limit" ] && [[ $in_use != *" $path "* ]]; then
      rm -f "$path"
    fi
  done < <(find "$bundle_dir" -maxdepth 1 -name '*.tar.zst' -printf '%T@ %s %p\n' 2>/dev/null | sort -rn)
fi

if [ -z "${VI_WORKER_HOST-}" ]; then
  # VI_WAIT_FOR_WORKER=MINUTES waits that long for `vi cloud up`, so a run can
  # be started (and its inputs packed) before anything is rented.
  deadline=$(($(date +%s) + ${VI_WAIT_FOR_WORKER:-0} * 60))
  until vars=$("$vi" cloud env 2>/dev/null); do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      [ ${#bundle[@]} -eq 0 ] || echo "on-worker.sh: inputs are packed and kept; run \`vi cloud up\` and rerun the stage" >&2
      "$vi" cloud env >/dev/null
      exit 1
    fi
    echo "on-worker.sh: waiting for a worker (run \`vi cloud up\`)" >&2
    sleep 30
  done
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
# Preflight: time a small upload and stop before sending the inputs if they'd
# take longer than VI_MAX_UPLOAD_MINUTES (default 30), unless
# VI_UPLOAD_ANYWAY=1. A slow uplink otherwise bills the GPU for hours of
# waiting.
if [ "$VI_WORKER_HOST" != local ] && [ "$packed_bytes" -gt 0 ]; then
  probe=8000000
  t0=$(date +%s%N)
  remote true
  t1=$(date +%s%N)
  head -c "$probe" /dev/urandom | remote "cat >/dev/null"
  t2=$(date +%s%N)
  ms=$((((t2 - t1) - (t1 - t0)) / 1000000))
  [ "$ms" -gt 0 ] || ms=1
  rate=$((probe * 1000 / ms))
  eta=$((packed_bytes / rate / 60))
  echo "on-worker.sh: $((packed_bytes / 1048576)) MB to upload at about $((rate / 1024)) KB/s: about $eta min" >&2
  if [ "$eta" -gt "${VI_MAX_UPLOAD_MINUTES:-30}" ] && [ "${VI_UPLOAD_ANYWAY:-0}" != 1 ]; then
    echo "on-worker.sh: that's over ${VI_MAX_UPLOAD_MINUTES:-30} min, so nothing was sent; free up home's uplink, or set VI_UPLOAD_ANYWAY=1" >&2
    exit 1
  fi
fi

remote_zstd=0
[ ${#bundle[@]} -eq 0 ] || ! remote "command -v zstd >/dev/null" || remote_zstd=1
n=0
for var in $(compgen -e | grep '^VI_INPUT_' || true); do
  name=${var#VI_INPUT_}
  dest="$work/inputs/$name"
  f=${bundle[$var]-}
  if [ -z "$f" ]; then
    push "${!var}/" "$dest/"
  elif [ "$remote_zstd" = 1 ]; then
    remote "mkdir -p $(q "$work/bundles")"
    push "$f" "$work/bundles/"
    remote "mkdir -p $(q "$dest") && zstd -dcq $(q "$work/bundles/${f##*/}") | tar -xf - -C $(q "$dest") && rm $(q "$work/bundles/${f##*/}")"
  else
    # No zstd on the worker: still one stream, decompressed here.
    remote "mkdir -p $(q "$dest")"
    zstd -dcq "$f" | remote "tar -xf - -C $(q "$dest")"
  fi
  n=$((n + 1))
  echo "upload: $n/$(compgen -e | grep -c '^VI_INPUT_')" >&2
  exports+=" $var=$(q "$dest")"
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
