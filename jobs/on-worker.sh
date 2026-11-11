#!/usr/bin/env bash
# Runs one dataset stage on the rented GPU worker instead of here.
#
#   [stage.label]
#   command = ["../../jobs/on-worker.sh", "--put", "states.toml", "python3", "jobs/label.py"]
#
# It copies this jobs/ folder, the stage's settings, each VI_INPUT_* folder and
# any --put files to a folder on the worker, launches the command there in a
# detached session, then supervises it from home over short ssh polls until it
# ends. The worker pushes VI_OUT home through a home-created reverse SSH tunnel
# and restricted rsync receiver. The worker's folder is deleted once the result
# is home and (when the stage asks for it) verified; otherwise it is kept for a
# re-push or a reattach. The job sees:
#   jobs/            this folder
#   files/<name>     each --put file, by its base name
#   VI_STAGE, VI_OUT, VI_SETTINGS, VI_INPUT_<NAME>, VI_CHECKPOINT, VI_PUBLISH,
#   VI_WORKER=1, VI_RETURN_PORT
#
# A stage that publishes finished epochs gets a durable per-attempt dir under
# ~/.cache/vi/checkpoints/<stage>/ (the `vi-publish` return module) that
# nothing deletes. The home supervisor verifies each epoch there as it lands,
# writes $work/acked/epoch-<k> on the worker so it is never re-sent, and asks
# for a re-push ($work/repush/epoch-<k>) when a copy does not verify. Three
# failed repairs mark it $work/bad/epoch-<k> and the stage fails loudly.
#
# A rerun attaches to the same durable attempt instead of starting over:
# a live work dir is chosen; else the dir named by the newest unverified
# marker. Two or more live candidates refuse instead of picking silently.
# Reattaching restarts the receiver and tunnel, proves a fresh token, and
# sends no inputs.
#
# VI_CHECKPOINT goes up with whatever an earlier, failed run of the same stage
# left in it, and the detached worker-job.sh runner copies it home every
# VI_SYNC_SECONDS (default 60) while a training job runs; on-worker.sh takes a
# final copy when the stage ends. So if the worker dies or the credit runs out,
# a rerun on a new worker resumes from the last copy.
#
# Inputs over VI_PACK_MIN_KB (default 1024) are packed at home first, before
# the worker is even looked up, into one zstd'd tar each, kept in
# VI_BUNDLE_DIR (default ~/.cache/vi/bundles, at most VI_BUNDLE_CACHE_MB,
# default 4000) for reruns. Then a timed 8 MB upload estimates how long the
# inputs will take, and the stage stops before sending anything if that's over
# the stage's max_upload_minutes setting (VI_MAX_UPLOAD_MINUTES overrides;
# default 30) unless VI_UPLOAD_ANYWAY=1.
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

# A stage may name a home-disk floor (min_home_gb), a reserve for the checkpoint
# bytes it will publish home (home_gate_reserve_gb), the epoch count to verify
# before success (expect_epochs), and its upload-time limit
# (max_upload_minutes). Read them from the settings JSON once; the worker never
# sees any of them.
min_home_gb=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("min_home_gb",""))' "$VI_SETTINGS")
reserve_gb=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("home_gate_reserve_gb",""))' "$VI_SETTINGS")
expect_epochs=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("expect_epochs",""))' "$VI_SETTINGS")
stage_upload=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("max_upload_minutes",""))' "$VI_SETTINGS")
case $stage_upload in ''|*[!0-9]*) stage_upload=30 ;; esac
upload_limit=${VI_MAX_UPLOAD_MINUTES:-$stage_upload}
case $upload_limit in ''|*[!0-9]*) upload_limit=30 ;; esac

poll_seconds=${VI_POLL_SECONDS:-30}
lost_limit=$(( ${VI_LOST_MINUTES:-30} * 60 ))
stall_minutes=${VI_STALL_MINUTES:-30}
stall_seconds=$((stall_minutes * 60))
pending_hold=$(( ${VI_PENDING_HOLD:-120} * 60 ))
watchdog_script=${VI_WATCHDOG_SCRIPT:-/root/vi-watchdog.sh}
watchdog_log=${VI_WATCHDOG_LOG:-/var/log/vi-watchdog.log}

gate_ok() {
  [ -z "$min_home_gb" ] && return 0
  python3 "$jobs_dir/checkpoints.py" gate "$HOME" --min-gb "$min_home_gb" --reserve-gb "${reserve_gb:-0}" >/dev/null 2>&1
}

# Refuse to start under the floor plus the reserve before anything is packed or
# a worker looked up, so the GPU is never billed with too little home disk to
# hold the result.
if [ -n "$min_home_gb" ]; then
  python3 "$jobs_dir/checkpoints.py" gate "$HOME" --min-gb "$min_home_gb" --reserve-gb "${reserve_gb:-0}"
fi

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
      # One packer at a time per bundle dir. A killed pack leaves `$f.part`
      # behind and zstd refuses to overwrite it, so take the lock, drop the
      # debris, and let zstd replace it.
      (
        flock 9
        rm -f "$f.part"
        nice -n 19 ionice -c2 -n7 tar -C "$dir" -cf - . | nice -n 19 ionice -c2 -n7 zstd -q -f -T0 -3 -o "$f.part"
        mv "$f.part" "$f"
      ) 9>"$bundle_dir/.lock"
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
checkpoint=${VI_CHECKPOINT:-}
[ -z "$checkpoint" ] || mkdir -p "$checkpoint"
# The same state dir `vi` uses. The home supervisor keeps its log cursor here
# and one stable receiver directory per attempt, so a reattach can restart the
# exact same restricted receiver.
state_dir=${VI_STATE_DIR:-}
[ -n "$state_dir" ] || state_dir=${XDG_STATE_HOME:-${HOME:-.}/.local/state}/vi
hs_dir="$state_dir/hs"
return_base="$state_dir/return"
mkdir -p "$state_dir/unverified" "$hs_dir" "$return_base"
# Backstop for a runner that was SIGKILLed before its receiver could notice:
# the sweep spares any receiver whose owner.lock is still held.
"$jobs_dir/worker-return.sh" sweep "$return_base" 2>/dev/null || true

q() { printf '%q' "$1"; }

# The ssh options are shared by every short call, every rsync transport and
# the tunnel. ConnectTimeout keeps one dead connection from stalling a poll.
ssh_opts=()
rebuild_ssh_opts() {
  ssh_opts=(-p "$VI_WORKER_PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new
    -o "UserKnownHostsFile=$VI_WORKER_KNOWN_HOSTS" -o ServerAliveInterval=30
    -o ConnectTimeout=20 -o ForwardAgent=no -o ForwardX11=no)
}
if [ "$VI_WORKER_HOST" = local ]; then
  remote() { bash -c "$1"; }
  push() { local src=$1 dst=$2; shift 2; rsync -a --timeout=300 "$@" "$src" "$dst"; }
else
  rebuild_ssh_opts
  target="root@$VI_WORKER_HOST"
  remote() { ssh "${ssh_opts[@]}" "$target" "$1"; }
  push() { local src=$1 dst=$2; shift 2; rsync -a --timeout=300 -e "ssh ${ssh_opts[*]}" "$@" "$src" "$target:$dst"; }
  # Keep home's idle clock running while the job does.
  (while sleep 60 >/dev/null 2>&1; do "$vi" cloud touch >/dev/null 2>&1 || true; done) &
  heartbeat=$!
fi

# Re-learn host/port after `vi cloud refresh`; a restarted instance can come
# back on a different ssh port.
reload_worker_env() {
  [ "$VI_WORKER_HOST" = local ] && return 1
  local vars
  vars=$("$vi" cloud env 2>/dev/null) || return 1
  case $vars in *VI_WORKER_HOST=*) ;; *) return 1 ;; esac
  eval "$vars"
  rebuild_ssh_opts
  target="root@$VI_WORKER_HOST"
  echo "on-worker.sh: worker endpoint is now $VI_WORKER_HOST:$VI_WORKER_PORT" >&2
  return 0
}

key=
[ -z "$checkpoint" ] || { key=$(basename "$checkpoint"); key=${key#.checkpoint-}; }
attempt=
durable=
reattach=0
work=
return_state=
return_port=
worker_port=
return_checkpoint=
return_scratch=
tunnel_loop=
keep_work=
final_ok=0
final_rc=1

# The durable dir a given attempt publishes into. Its marker is authoritative
# when one exists (a previous run recorded the exact path), else it is derived
# the same way that run did.
durable_for_attempt() {  # ATTEMPT
  local a=$1 m b
  for m in "$state_dir/unverified/$VI_STAGE"-*; do
    [ -f "$m" ] && [ ! -L "$m" ] || continue
    b=$(basename "$(cat "$m" 2>/dev/null)" 2>/dev/null || true)
    [ "$b" = "$key-$a" ] && { cat "$m"; return 0; }
  done
  printf '%s\n' "$HOME/.cache/vi/checkpoints/$VI_STAGE/$key-$a"
}

# Pick a worker dir to reattach to, or a fresh attempt timestamp. LIVE is
# recomputed on the worker: a stage session (ssh) OR (job lock OR trainer OR
# un-acked pending) AND progress in the job dir within STALL minutes. A
# pending-only hold after the job wrote rc is capped by PENDING_HOLD.
pick_attempt() {
  local raw line all=() live=() failed=() d m t best_t=-1 best_marker= base a candidate=
  # The trailing `true` keeps the last attempt's "not live" status from
  # discarding the whole status block; the glob order is arbitrary.
  raw=$(remote "now=\$(date +%s)
    for d in $(q "$root/$VI_STAGE.$key").*; do
      [ -d \"\$d\" ] || continue
      echo \"VI-DIR \$d\"
      l=0; t=0; s=0; p=0; g=0; rc=0; live=0
      [ -e \"\$d/job.lock\" ] && ! flock -n \"\$d/job.lock\" true 2>/dev/null && l=1
      if [ -s \"\$d/train.pid\" ]; then
        tp=\$(cat \"\$d/train.pid\" 2>/dev/null) || tp=
        kill -0 \"\$tp\" 2>/dev/null && tr '\\0' ' ' <\"/proc/\$tp/cmdline\" 2>/dev/null | grep -q voicetune.py && t=1
      fi
      [ -s \"\$d/stage.pid\" ] && kill -0 \"\$(cat \"\$d/stage.pid\" 2>/dev/null)\" 2>/dev/null && s=1
      ls \"\$d\"/pending/* >/dev/null 2>&1 && p=1
      find \"\$d\" -mindepth 1 -newermt \"@\$((now - $stall_seconds))\" -print -quit 2>/dev/null | grep -q . && g=1
      [ -s \"\$d/rc\" ] && rc=\$(stat -c %Y \"\$d/rc\")
      [ \$((l + t + s)) -gt 0 ] && [ \$g = 1 ] && live=1
      if [ \$p = 1 ]; then
        if [ \$rc -gt 0 ]; then
          [ \$((now - rc)) -lt $pending_hold ] && live=1
        elif [ \$((l + t + s)) -gt 0 ] || [ \$g = 1 ]; then
          live=1
        fi
      fi
      [ \$live = 1 ] && echo \"VI-LIVE \$d\"
      # A finished, failed stage with nothing un-acked has nothing to resume.
      src=\$(cat \"\$d/stage.rc\" 2>/dev/null) || src=
      [ -n \"\$src\" ] && [ \"\$src\" != 0 ] && [ \$p = 0 ] && echo \"VI-FAILED \$d\"
    done
    true" 2>/dev/null) || raw=
  while IFS= read -r line; do
    case $line in
      "VI-DIR "*) all+=("${line#VI-DIR }") ;;
      "VI-LIVE "*) live+=("${line#VI-LIVE }") ;;
      "VI-FAILED "*) failed+=("${line#VI-FAILED }") ;;
    esac
  done <<<"$raw"

  if [ ${#live[@]} -gt 1 ]; then
    echo "on-worker.sh: refusing to choose between ${#live[@]} live attempts for stage $VI_STAGE:" >&2
    for d in "${live[@]}"; do
      a=${d##*.}
      echo "  $d (attempt $a)" >&2
      m="$state_dir/unverified/$VI_STAGE-$key-$a"
      if [ -f "$m" ]; then
        echo "    abandon this run if it is the stale one: python3 $(q "$jobs_dir/checkpoints.py") abandon $(q "$(cat "$m")") --reason \"duplicate live attempt\"" >&2
      fi
    done
    echo "on-worker.sh: nothing was chosen or deleted; keep one attempt and abandon the others, then rerun" >&2
    exit 2
  fi
  if [ ${#live[@]} -eq 1 ]; then
    attempt=${live[0]##*.}
    reattach=1
    durable=$(durable_for_attempt "$attempt")
    return 0
  fi

  for m in "$state_dir/unverified/$VI_STAGE"-*; do
    [ -f "$m" ] && [ ! -L "$m" ] || continue
    t=$(stat -c %Y "$m" 2>/dev/null || echo 0)
    if [ "$t" -gt "$best_t" ]; then best_t=$t; best_marker=$m; fi
  done
  if [ -n "$best_marker" ]; then
    d=$(cat "$best_marker" 2>/dev/null || true)
    base=$(basename "$d" 2>/dev/null || true)
    a=${base##*-}
    if [[ $a =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
      candidate="$root/$VI_STAGE.$key.$a"
      for line in "${failed[@]}"; do
        # Failed and nothing pending: start fresh so the home checkpoint is
        # pushed. The marker stays, so the old work remains unverified (R47).
        [ "$line" = "$candidate" ] && { attempt=$(date -u +%Y%m%dT%H%M%SZ); return 0; }
      done
      for line in "${all[@]}"; do
        if [ "$line" = "$candidate" ]; then
          attempt=$a
          durable=$d
          reattach=1
          return 0
        fi
      done
    fi
  fi
  attempt=$(date -u +%Y%m%dT%H%M%SZ)
}

if [ -n "$checkpoint" ]; then
  pick_attempt
else
  attempt=$(date -u +%Y%m%dT%H%M%SZ)
fi
work="$root/$VI_STAGE.$key.$attempt"
return_state="$return_base/$VI_STAGE.$key.$attempt"
cursor="$hs_dir/$attempt.offset"
if [ -n "$checkpoint" ]; then
  [ -n "$durable" ] || durable="$HOME/.cache/vi/checkpoints/$VI_STAGE/$key-$attempt"
else
  durable=
fi
return_checkpoint=$checkpoint
# A run with no checkpoint still needs a directory for the checkpoint module,
# but it must live outside return_state: the receiver refuses a non-empty
# state directory (it holds rsyncd.conf and the lock files).
[ -n "$return_checkpoint" ] || { return_scratch=$(mktemp -d); return_checkpoint=$return_scratch; }

if [ "$reattach" = 1 ]; then
  keep_work=1
  echo "on-worker.sh: reattaching to $work (attempt $attempt); no inputs will be sent" >&2
fi

receiver_alive() {
  [ -s "$return_state/rsyncd.pid" ] &&
    kill -0 "$(cat "$return_state/rsyncd.pid" 2>/dev/null)" 2>/dev/null
}

start_receiver() {
  mkdir -p "$return_state"
  local port
  if [ -s "$return_state/rsyncd.conf" ]; then
    # Reattach: the stored config is byte-identical, so the receiver keeps the
    # same local port and module paths; only the dead daemon's files are reset.
    port=$("$jobs_dir/worker-return.sh" restart "$return_state")
    echo "on-worker.sh: restarted the return receiver on loopback port $port" >&2
  else
    port=$("$jobs_dir/worker-return.sh" start "$VI_OUT" "$return_checkpoint" "$return_state" "$durable" "${VI_RETURN_PORT:-}")
    echo "on-worker.sh: return receiver listening on loopback port $port" >&2
  fi
  return_port=$port
}

free_remote_port() {  # PREVIOUS
  local prev=${1:-} p n
  for n in {1..20}; do
    p=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    [ "$p" != "$prev" ] && { printf '%s\n' "$p"; return 0; }
  done
  printf '%s\n' "$p"
}

# The reverse tunnel, kept up by a loop that picks a FRESH remote port on every
# reconnect (the local receiver port stays fixed). The worker learns the port
# from $work/.return-port; a lingering listener on the old port cannot block
# the new one. Every connect/disconnect is timestamped in tunnel.log (-E lets
# ssh append its own diagnostics there too).
start_tunnel_loop() {
  [ "$VI_WORKER_HOST" = local ] && return 0
  (
    set +e
    last=
    sshpid=
    # The tunnel dies with this runner even when it is SIGKILLed: the watchdog
    # below notices the owner is gone and stops this loop, whose trap stops ssh.
    # setpriv adds a second guarantee for the ssh process itself.
    # $$ is this script's pid even in a subshell (BASHPID would be our own).
    # $PPID is not: bash keeps the value from startup, so it would name our
    # parent's parent and the watchdog would outlive the supervisor.
    owner=$$
    self=$BASHPID
    ( while kill -0 "$owner" 2>/dev/null; do sleep 1; done
      kill -TERM "$self" 2>/dev/null ) &
    owner_watch=$!
    trap '[ -n "${sshpid-}" ] && kill "$sshpid" 2>/dev/null; kill "$owner_watch" 2>/dev/null; rm -f "$return_state/tunnel.pid"' EXIT
    while :; do
      rp=$(free_remote_port "$last")
      last=$rp
      printf '%s\n' "$rp" >"$return_state/remote-port"
      printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) tunnel connecting: remote 127.0.0.1:$rp -> home 127.0.0.1:$return_port" >>"$return_state/tunnel.log"
      setpriv --pdeathsig TERM ssh "${ssh_opts[@]}" -E "$return_state/tunnel.log" -N -T \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -R "127.0.0.1:$rp:127.0.0.1:$return_port" "$target" &
      sshpid=$!
      printf '%s\n' "$sshpid" >"$return_state/tunnel.pid"
      sleep 1
      if kill -0 "$sshpid" 2>/dev/null; then
        remote "printf '%s\n' $rp > $(q "$work/.return-port")" 2>>"$return_state/tunnel.log" &&
          printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) tunnel up: remote loopback port $rp" >>"$return_state/tunnel.log"
      fi
      wait "$sshpid"
      rc=$?
      sshpid=
      printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) tunnel down (ssh exit $rc); reconnecting in 10s" >>"$return_state/tunnel.log"
      rm -f "$return_state/tunnel.pid"
      sleep 10
    done
  ) >>"$return_state/tunnel.log" 2>&1 &
  tunnel_loop=$!
}

stop_tunnel_loop() {
  [ -n "${tunnel_loop-}" ] || return 0
  kill "$tunnel_loop" 2>/dev/null || true
  wait "$tunnel_loop" 2>/dev/null || true
  if [ -s "$return_state/tunnel.pid" ]; then
    kill "$(cat "$return_state/tunnel.pid" 2>/dev/null)" 2>/dev/null || true
  fi
  rm -f "$return_state/tunnel.pid"
  tunnel_loop=
}

# Prove the return path with a fresh random token: write it on the worker,
# push it through the receiver, and require the exact bytes home. BUDGET is the
# whole attempt: 5 s on first start, the lost-poll budget on reconnect and
# reattach. The push itself waits only a moment so a dead path retries here
# with backoff instead of blocking inside rsync.
return_prove() {  # BUDGET
  local budget=$1 deadline delays=(0.25 0.5 1 2 4 5) i=0 token p ok=0
  deadline=$(( $(date +%s) + budget ))
  token=$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')
  while :; do
    if [ "$VI_WORKER_HOST" = local ]; then
      p=$return_port
    else
      p=$(head -n1 "$return_state/remote-port" 2>/dev/null || true)
    fi
    case $p in ''|*[!0-9]*) p= ;; esac
    if [ -n "$p" ] &&
       remote "printf '%s\n' $(q "$p") > $(q "$work/.return-port") && printf '%s\n' $(q "$token") > $(q "$work/.return-token") && VI_OUT=$(q "$work/out") VI_RETURN_PORT=$(q "$p") VI_PUSH_WAIT=${VI_PROVE_WAIT:-5} $(q "$work/jobs/worker-return.sh") push $(q "$work/.return-token") vi-out" >/dev/null 2>>"$return_state/prove.err" &&
       [ -f "$VI_OUT/.return-token" ] && [ "$(cat "$VI_OUT/.return-token")" = "$token" ]; then
      ok=1
      worker_port=$p
      rm -f "$VI_OUT/.return-token"
      break
    fi
    rm -f "$VI_OUT/.return-token" 2>/dev/null || true
    [ "$(date +%s)" -lt "$deadline" ] || break
    sleep "${delays[$i]}"
    [ $i -ge 5 ] || i=$((i + 1))
  done
  if [ "$ok" != 1 ]; then
    echo "on-worker.sh: worker could not push through the reverse SSH return tunnel" >&2
    [ -f "$return_state/tunnel.log" ] && tail -n 5 "$return_state/tunnel.log" >&2 || true
    return 1
  fi
  return 0
}

# Reattach/restart entry point: receiver, tunnel, then the fresh-token proof.
return_start() {  # BUDGET
  start_receiver || return 1
  start_tunnel_loop || true
  return_prove "$1"
}

worker_sync() {
  local src=$1 module=$2 arg args=""
  shift 2
  for arg in "$@"; do args+=" $(q "$arg")"; done
  remote "VI_RETURN_PORT=$(q "$worker_port") $(q "$work/jobs/worker-return.sh") push $(q "$src") $(q "$module")$args"
}

# Stop the publish push loop this run started, if it is still running. The
# recorded pid is used only when its command line still names this attempt's
# voicetune-fresh.sh, so a reused pid is never killed. A loop started by an
# earlier, killed home process is left alone: a reattach takes the same lock.
stop_push_loop() {
  [ -n "${work-}" ] || return 0
  remote "p=\$(cat $(q "$work/push.pid") 2>/dev/null)
case \$p in ''|*[!0-9]*) exit 0 ;; esac
if tr '\\0' ' ' </proc/\$p/cmdline 2>/dev/null | grep -qF $(q "$work/jobs/voicetune-fresh.sh"); then
  kill \$p 2>/dev/null || true
fi
rm -f $(q "$work/push.pid")" || true
}

return_stop() {
  stop_tunnel_loop
  if [ -n "${return_state-}" ]; then
    "$jobs_dir/worker-return.sh" stop "$return_state" || true
    rm -rf "$return_state"
    return_state=
  fi
  if [ -n "${return_scratch-}" ]; then
    rmdir "$return_scratch" 2>/dev/null || rm -rf "$return_scratch"
    return_scratch=
  fi
}

cleanup() {
  [ -n "${heartbeat-}" ] && kill "$heartbeat" 2>/dev/null
  stop_tunnel_loop
  stop_push_loop
  # Delete the worker's job dir only once the result is safely home and
  # verified; otherwise keep it so a reattach or re-push is possible
  # (`vi cloud down` removes it with the instance).
  if [ -n "${work-}" ]; then
    if [ -z "${keep_work-}" ]; then
      remote "rm -rf $(q "$work")" ||
        echo "on-worker.sh: could not delete $work on the worker; \`vi cloud down\` removes it with the instance" >&2
    else
      echo "on-worker.sh: keeping $work on the worker; \`vi cloud down\` removes it with the instance" >&2
    fi
  fi
  return_stop
}
# Stop the receiver and tunnel on every exit path: normal exit, Ctrl-C (INT)
# and kill/TERM. The signal traps turn the signal into `exit`, which runs the
# EXIT trap, so cleanup runs exactly once and the recorded PIDs are used.
trap cleanup EXIT
# The INT/TERM traps only record the exit code. Bash runs a trap inside the
# redirection context of the command it interrupted (poll_worker redirects a
# shell function's stderr), so a message printed there can land in a poll's
# error file instead of the job log. The loop exits from a clean context.
interrupted=0
trap 'interrupted=130' INT
trap 'interrupted=143' TERM

if [ "$reattach" = 0 ]; then
  remote "mkdir -p $(q "$work")/inputs $(q "$work")/out $(q "$work")/files && touch $(q "$root/.active")"
  push "$jobs_dir/" "$work/jobs/"
  push "$VI_SETTINGS" "$work/settings.json"
fi
# A first start gets a short budget (the path was just proven moments ago and
# a slow one should fail before billing much GPU time); a reattach gets the
# same lost-poll budget the HS uses, because its path may come back slowly.
start_budget=5
[ "$reattach" = 1 ] && start_budget=$lost_limit
if ! return_start "$start_budget"; then
  echo "on-worker.sh: the return path did not come up; nothing will be sent" >&2
  [ "$reattach" = 1 ] || keep_work=
  exit 1
fi
keep_work=1

if [ -n "$checkpoint" ]; then
  mkdir -p "$durable"
fi

# Cheap checks on the worker (setsid/flock, VRAM, disk, cores, RAM, GPU facts,
# a disk probe and a raw 100 MiB push home) before the 8 MB upload estimate or
# any input. A reattach sends no inputs and is recovery of an already-checked
# worker, so it skips the checks. The dummy's home copy is deleted at once,
# before the runner ever looks at the stage output; the worker's own copy is
# deleted by the preflight script.
if [ "$reattach" = 0 ]; then
  pre_rc=0
  remote "VI_OUT=$(q "$work/out") VI_RETURN_PORT=$(q "$worker_port") VI_SETTINGS=$(q "$work/settings.json") VI_FAKE_PUSH_SECONDS=$(q "${VI_FAKE_PUSH_SECONDS:-}") bash $(q "$work/jobs/worker-preflight.sh")" || pre_rc=$?
fi
# The dummy is pushed home only to be timed. Delete it before the runner ever
# looks at the stage output, and on a reattach too, in case an earlier start
# was killed between its push and this line.
rm -f "$VI_OUT/.speedtest"
if [ "$reattach" = 0 ] && [ "$pre_rc" != 0 ]; then
  echo "on-worker.sh: the worker preflight failed; nothing was uploaded" >&2
  exit 1
fi

if [ "$reattach" = 0 ]; then
  if [ -n "$checkpoint" ]; then
    # A copy taken mid-write can end in half a line; drop it before resuming.
    for f in "$checkpoint"/*.jsonl; do
      if [ -s "$f" ] && [ -n "$(tail -c1 "$f")" ]; then
        truncate -s $(($(wc -c <"$f") - $(tail -n1 "$f" | wc -c))) "$f"
      fi
    done
    # Model epoch-<k> weights (tens of GB) live in the checkpoint too, but they
    # are published separately; skip them here so a rerun doesn't re-send them.
    push "$checkpoint/" "$work/checkpoint/" --exclude='epoch-*/'
  fi
  for f in "${puts[@]}"; do
    push "$f" "$work/files/"
  done
fi

exports="VI_STAGE=$(q "$VI_STAGE") VI_OUT=$(q "$work/out") VI_SETTINGS=$(q "$work/settings.json") VI_WORKER=1 VI_RETURN_PORT=$(q "$worker_port")"
[ -n "$checkpoint" ] && exports+=" VI_CHECKPOINT=$(q "$work/checkpoint")"
[ -n "$durable" ] && exports+=" VI_PUBLISH=vi-publish"

if [ "$reattach" = 0 ]; then
  # Preflight: time a small upload and stop before sending the inputs if they'd
  # take longer than the stage's max_upload_minutes (VI_MAX_UPLOAD_MINUTES
  # overrides; default 30), unless VI_UPLOAD_ANYWAY=1. A slow uplink otherwise
  # bills the GPU for hours of waiting.
  if [ "$VI_WORKER_HOST" != local ] && [ "$packed_bytes" -gt 0 ] && [ "${VI_UPLOAD_ANYWAY:-0}" != 1 ]; then
    if [ "$upload_limit" -eq 0 ]; then
      echo "on-worker.sh: max_upload_minutes is 0, so nothing was sent; raise it or set VI_UPLOAD_ANYWAY=1" >&2
      exit 1
    fi
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
    if [ "$eta" -gt "$upload_limit" ]; then
      echo "on-worker.sh: that's over ${upload_limit} min, so nothing was sent; free up home's uplink, or set VI_UPLOAD_ANYWAY=1" >&2
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
      push "$f" "$work/bundles/" --partial
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
fi

cmd=""
for arg in "$@"; do
  cmd+=" $(q "$arg")"
done
# The durable dir stays unverified until `checkpoints.py verify` clears its
# per-attempt marker, so `vi cloud down` refuses while it exists. Write the
# marker only once every input is on the worker: a rerun picks the newest
# marker's attempt, and it must never reattach to an attempt whose upload
# never finished (a reattach deliberately sends no inputs).
if [ -n "$checkpoint" ]; then
  marker="$state_dir/unverified/$VI_STAGE-$(basename "$durable")"
  [ -f "$marker" ] || printf '%s\n' "$durable" >"$marker"
fi
# Only a first start launches the command. A reattach supervises the stage the
# killed home process left running; launching it again would run the job twice
# and overwrite the surviving stage's pid and exit code.
if [ "$reattach" = 0 ]; then
  active=$(q "$root/.active")
  stage_script="echo \$\$ > $(q "$work/stage.pid")
cd $(q "$work") || exit 70
export $exports
$cmd
rc=\$?
touch $active
printf '%s\n' \"\$rc\" > $(q "$work/stage.rc")
exit \$rc"
  if ! remote "setsid -f bash -c $(q "$stage_script") >>$(q "$work/job.log") 2>&1 </dev/null"; then
    echo "on-worker.sh: could not launch the job on the worker" >&2
    final_ok=0
    final_rc=1
    exit 1
  fi
fi
# Until the run is verified home, a Ctrl-C or kill must not delete the worker's
# copy: the stage is detached and keeps running after this process is gone.
keep_work=1

# The poll's fixed beginning. Everything with a backslash runs on the worker;
# the values above are this run's paths, quoted at home. It prints key/value
# lines, a VI-LOG marker, then the new part of job.log.
status_head="cd $(q "$work") 2>/dev/null || { echo 'VI-STATUS no-work-dir'; exit 0; }
echo VI-NOW \$(date +%s)
echo VI-STAGE \$( [ -s stage.pid ] && kill -0 \"\$(cat stage.pid 2>/dev/null)\" 2>/dev/null && echo 1 || echo 0 )
echo VI-STAGE-RC \"\$(cat stage.rc 2>/dev/null)\"
echo VI-JOB-RC \"\$(cat rc 2>/dev/null)\"
echo VI-TRAIN \$( [ -s train.pid ] && kill -0 \"\$(cat train.pid 2>/dev/null)\" 2>/dev/null && echo 1 || echo 0 )
echo VI-LOCK \$( [ -e job.lock ] && ! flock -n job.lock true 2>/dev/null && echo 1 || echo 0 )
echo VI-PUSH \$( [ -s push.pid ] && kill -0 \"\$(cat push.pid 2>/dev/null)\" 2>/dev/null && echo 1 || echo 0 )
echo VI-ACKED \"\$(ls acked 2>/dev/null | tr '\n' ' ')\"
echo VI-PENDING \"\$(ls pending 2>/dev/null | tr '\n' ' ')\"
echo VI-REPUSH \"\$(ls repush 2>/dev/null | tr '\n' ' ')\"
echo VI-BAD \"\$(ls bad 2>/dev/null | tr '\n' ' ')\"
echo VI-WATCHDOG \$(pgrep -f vi-watchdog.sh >/dev/null 2>&1 && echo 1 || echo 0)
echo VI-PROGRESS \$(find . -mindepth 1 -newermt \"@\$(( \$(date +%s) - $stall_seconds ))\" -print -quit 2>/dev/null | grep -q . && echo 1 || echo 0)
echo VI-SIZE \$(stat -c %s job.log 2>/dev/null || echo 0)
echo VI-LOG"

poll_worker() {  # OFFSET
  local off=$1 raw="$return_state/status.raw"
  remote "$status_head
tail -c +$((off + 1)) job.log 2>/dev/null | head -c 65536" >"$raw" 2>"$return_state/poll.err" || return 1
  python3 - "$raw" "$return_state/status" "$return_state/chunk" <<'PY' || return 1
import sys
raw = open(sys.argv[1], "rb").read()
mark = b"\nVI-LOG\n"
i = raw.find(mark)
if i < 0:
    sys.exit(1)
open(sys.argv[2], "wb").write(raw[:i + 1])
open(sys.argv[3], "wb").write(raw[i + len(mark):])
PY
  [ "$(sv VI-STATUS)" != error ]
}

sv() { awk -v k="$1" '$1 == k { $1 = ""; sub(/^ /, ""); print; exit }' "$return_state/status" 2>/dev/null || true; }

# The cursor only decides what to print; offset 0 (as after the file is
# deleted) reprints the log from the start and changes no action.
render_log() {
  local size have chunk="$return_state/chunk" new
  size=$(sv VI-SIZE)
  case $size in ''|*[!0-9]*) size=0 ;; esac
  have=$(cat "$cursor" 2>/dev/null || true)
  case $have in ''|*[!0-9]*) have=0 ;; esac
  [ -s "$chunk" ] && cat "$chunk" >&2
  new=$((have + $(stat -c %s "$chunk" 2>/dev/null || echo 0)))
  [ "$new" -le "$size" ] || new=$size
  printf '%s\n' "$new" >"$cursor"
}

# Verify each epoch home found on the worker's publish path (SHA256SUMS is
# pushed last, so its presence means the epoch is complete) and ack it, or ask
# the worker to re-send it with --checksum.
ack_epochs() {  # ACKED REPUSH BAD
  local acked=" $1 " repush=" $2 " bad=" $3 " ep n
  [ -n "$durable" ] || return 0
  for ep in "$durable"/epoch-*; do
    [ -d "$ep" ] || continue
    case ${ep##*/} in *'.part') continue ;; esac
    n=${ep##*/epoch-}
    case $n in ''|*[!0-9]*) continue ;; esac
    case $acked in *" epoch-$n "*) continue ;; esac
    case $repush in *" epoch-$n "*) continue ;; esac
    case $bad in *" epoch-$n "*) continue ;; esac
    [ -f "$ep/SHA256SUMS" ] || continue
    if python3 "$jobs_dir/checkpoints.py" verify-epoch "$durable" "$n" >/dev/null 2>&1; then
      if remote "mkdir -p $(q "$work/acked") && : > $(q "$work/acked/epoch-$n")"; then
        echo "on-worker.sh: epoch $n verified home and acked" >&2
      else
        echo "on-worker.sh: ALERT: epoch $n verified home but could not be acked; it will be checked again" >&2
      fi
    else
      if remote "mkdir -p $(q "$work/repush") && : > $(q "$work/repush/epoch-$n")"; then
        echo "on-worker.sh: ALERT: epoch $n failed verification; asked the worker to re-send it" >&2
      else
        echo "on-worker.sh: ALERT: epoch $n failed verification and the re-send request could not be sent" >&2
      fi
    fi
  done
}

# Count the epochs that still need pushing home (published, not acked).
unacked_epochs() {  # ACKED
  local acked=" $1 " ep n out=0
  [ -n "$durable" ] || { printf '0\n'; return 0; }
  for ep in "$durable"/epoch-*; do
    [ -d "$ep" ] || continue
    case ${ep##*/} in *'.part') continue ;; esac
    n=${ep##*/epoch-}
    case $n in ''|*[!0-9]*) continue ;; esac
    case $acked in *" epoch-$n "*) continue ;; esac
    out=$((out + 1))
  done
  printf '%s\n' "$out"
}

# Start the detached push loop when the stage supervisor that would normally
# push is gone. The worker takes the same push.lock the train mode holds, so a
# live supervisor means the flock below fails and nothing is started.
start_push_if_needed() {  # TRAIN PENDING PUSH_ALIVE ACKED
  local train=$1 pending=$2 push_alive=$3 acked=$4 need=0
  [ "$push_alive" = 1 ] && return 0
  [ "$train" = 1 ] && need=1
  [ -n "$pending" ] && need=1
  { [ -n "$(sv VI-REPUSH)" ] || [ "$(unacked_epochs "$acked")" != 0 ]; } && need=1
  [ "$need" = 1 ] || return 0
  if remote "flock -n $(q "$work/push.lock") true" 2>/dev/null; then
    if remote "VI_OUT=$(q "$work/out") VI_RETURN_PORT=$(q "$worker_port") setsid -f bash $(q "$work/jobs/voicetune-fresh.sh") push >>$(q "$work/job.log") 2>&1 </dev/null"; then
      echo "on-worker.sh: started the publish push loop on the worker" >&2
    else
      echo "on-worker.sh: ALERT: could not start the publish push loop" >&2
    fi
  fi
}

# A changed worker host key is checked through the authenticated Vast API
# (`vi cloud hostkey`), never accepted blindly. A refusal halts pushes. The
# proof/need state lives at script scope because this runs from hs_loop.
maybe_rotate() {
  [ "$VI_WORKER_HOST" = local ] && return 1
  grep -q "REMOTE HOST IDENTIFICATION HAS CHANGED" \
    "$return_state/poll.err" "$return_state/tunnel.log" 2>/dev/null || return 1
  echo "on-worker.sh: ALERT: the worker's ssh host key changed; checking the new key against the Vast API" >&2
  if "$vi" cloud hostkey; then
    rotations=$((rotations + 1))
    echo "on-worker.sh: ALERT: accepted host-key rotation $rotations for this worker; only a reattach continues, no inputs are sent" >&2
    reload_worker_env || true
    stop_tunnel_loop
    start_tunnel_loop
    proven_port=
    need_proof=1
    return 0
  fi
  echo "on-worker.sh: ALERT: refused the worker's new host key; pushes stay halted" >&2
  return 2
}

stage_finished() {  # RC
  final_rc=$1
  [ "$1" -eq 0 ] && final_ok=1 || final_ok=0
}

# Set by maybe_rotate and read by hs_loop.
need_proof=1
proven_port=
rotations=0

hs_loop() {
  local offset now failed rp elapsed last_wd=0 last_refresh=0 lost_since=0
  local gate_paused=0 alerted_bad= rot_rc=0
  while :; do
    [ "$interrupted" = 0 ] || exit "$interrupted"
    now=$(date +%s)
    failed=0

    if ! receiver_alive; then
      echo "on-worker.sh: the return receiver is down; restarting it" >&2
      if "$jobs_dir/worker-return.sh" restart "$return_state" >/dev/null 2>&1; then
        need_proof=1
      else
        echo "on-worker.sh: ALERT: could not restart the return receiver" >&2
        failed=1
      fi
    fi
    if [ "$VI_WORKER_HOST" != local ] &&
       { [ -z "${tunnel_loop-}" ] || ! kill -0 "${tunnel_loop:-0}" 2>/dev/null; }; then
      echo "on-worker.sh: the tunnel loop died; restarting it" >&2
      start_tunnel_loop
      need_proof=1
    fi

    if [ "$failed" = 0 ]; then
      if [ "$VI_WORKER_HOST" = local ]; then
        rp=$return_port
      else
        rp=$(head -n1 "$return_state/remote-port" 2>/dev/null || true)
      fi
      if [ "$need_proof" = 1 ] || { [ -n "$rp" ] && [ "$rp" != "$proven_port" ]; }; then
        if return_prove "$lost_limit"; then
          if [ "$VI_WORKER_HOST" = local ]; then
            proven_port=$return_port
          else
            proven_port=$(head -n1 "$return_state/remote-port" 2>/dev/null || true)
          fi
          need_proof=0
          echo "on-worker.sh: return path re-proved through remote port $proven_port" >&2
        else
          rot_rc=0
          maybe_rotate || rot_rc=$?
          if [ "$rot_rc" = 2 ]; then
            echo "on-worker.sh: ALERT: halting; the worker copy stays for a human" >&2
            stage_finished 1
            return
          fi
          echo "on-worker.sh: ALERT: could not re-prove the return path" >&2
          failed=1
        fi
      fi
    fi

    if [ "$failed" = 0 ]; then
      offset=$(cat "$cursor" 2>/dev/null || true)
      case $offset in ''|*[!0-9]*) offset=0 ;; esac
      if poll_worker "$offset"; then
        lost_since=0
        render_log

        if [ "$VI_WORKER_HOST" != local ] && [ "$(sv VI-WATCHDOG)" = 0 ] && [ $((now - last_wd)) -ge 60 ]; then
          remote "nohup bash $(q "$watchdog_script") >>$(q "$watchdog_log") 2>&1 </dev/null &" >/dev/null 2>&1 || true
          echo "on-worker.sh: ALERT: the worker watchdog was missing; restarted $watchdog_script" >&2
          last_wd=$now
        fi

        if gate_ok; then
          if [ "$gate_paused" = 1 ]; then
            echo "on-worker.sh: home disk is above the floor again; resuming acks and pushes" >&2
          fi
          gate_paused=0
        else
          if [ "$gate_paused" = 0 ]; then
            echo "on-worker.sh: ALERT: home disk is below the floor; training continues, acks and pushes are paused" >&2
          fi
          gate_paused=1
        fi

        local bad stage_alive stage_rc train lock pending acked
        bad=$(sv VI-BAD)
        if [ -n "$bad" ] && [ "$bad" != "$alerted_bad" ]; then
          echo "on-worker.sh: ALERT: the worker gave up on $bad after three repair pushes; this run cannot succeed" >&2
          alerted_bad=$bad
        fi

        if [ "$gate_paused" = 0 ]; then
          ack_epochs "$(sv VI-ACKED)" "$(sv VI-REPUSH)" "$bad"
          start_push_if_needed "$(sv VI-TRAIN)" "$(sv VI-PENDING)" "$(sv VI-PUSH)" "$(sv VI-ACKED)"
        fi

        stage_alive=$(sv VI-STAGE)
        stage_rc=$(sv VI-STAGE-RC)
        train=$(sv VI-TRAIN)
        lock=$(sv VI-LOCK)
        pending=$(sv VI-PENDING)
        acked=$(sv VI-ACKED)

        if [ "$stage_alive" = 0 ] && [ -n "$stage_rc" ]; then
          if [ "$stage_rc" -eq 0 ]; then
            if [ -n "$bad" ]; then
              echo "on-worker.sh: ALERT: the stage finished but $bad never arrived intact; keeping every copy" >&2
              stage_finished 1
              return
            fi
            if [ "$(unacked_epochs "$acked")" = 0 ] && [ -z "$(sv VI-REPUSH)" ]; then
              stage_finished 0
              return
            fi
            # Epochs are still going home; keep polling and let the push loop
            # finish (it exits by itself once everything is acked).
          else
            echo "on-worker.sh: $VI_STAGE exited with $stage_rc on the worker" >&2
            stage_finished "$stage_rc"
            return
          fi
        elif [ "$stage_alive" = 0 ] && [ "$lock$train" = "00" ] && [ -z "$pending" ]; then
          if [ -n "$(sv VI-JOB-RC)" ]; then
            echo "on-worker.sh: ALERT: the stage supervisor ended without stage.rc while the trainer finished (exit $(sv VI-JOB-RC)); keeping the worker copy for inspection" >&2
          else
            echo "on-worker.sh: ALERT: the stage supervisor and trainer are gone with no exit code; keeping the worker copy" >&2
          fi
          stage_finished 1
          return
        fi

        [ "$interrupted" = 0 ] || exit "$interrupted"
        sleep "$poll_seconds"
        continue
      fi
      rot_rc=0
      maybe_rotate || rot_rc=$?
      if [ "$rot_rc" = 2 ]; then
        echo "on-worker.sh: ALERT: halting; the worker copy stays for a human" >&2
        stage_finished 1
        return
      fi
    fi

    [ "$lost_since" -eq 0 ] && lost_since=$now
    elapsed=$(( $(date +%s) - lost_since ))
    echo "on-worker.sh: poll failed; will retry (lost for ${elapsed}s)" >&2
    if [ "$elapsed" -ge "$lost_limit" ]; then
      echo "on-worker.sh: giving up after $((elapsed / 60)) min without a successful poll; keeping $work on the worker for a reattach" >&2
      keep_work=1
      stage_finished 1
      return
    fi
    if [ "$VI_WORKER_HOST" != local ] &&
       [ $((lost_limit / 3)) -gt 0 ] && [ "$elapsed" -ge $((lost_limit / 3)) ] &&
       [ $(( $(date +%s) - last_refresh )) -ge 60 ]; then
      echo "on-worker.sh: refreshing the worker endpoint from the Vast API" >&2
      if "$vi" cloud refresh >/dev/null 2>&1 && reload_worker_env; then
        stop_tunnel_loop
        start_tunnel_loop
        need_proof=1
      else
        echo "on-worker.sh: ALERT: the endpoint refresh failed" >&2
      fi
      last_refresh=$(date +%s)
    fi
    [ "$interrupted" = 0 ] || exit "$interrupted"
    sleep "$poll_seconds"
  done
}

hs_loop
[ "$interrupted" = 0 ] || exit "$interrupted"

# Final checkpoint copy, then (only on success) the output push and whole-run
# verification. Any failure keeps the worker's copy for a reattach/re-push.
if [ -n "$checkpoint" ]; then
  worker_sync "$work/checkpoint/" vi-checkpoint --append --exclude='epoch-*/' ||
    echo "on-worker.sh: could not copy the final checkpoint home; the last periodic copy is kept" >&2
fi

if [ "$final_ok" = 1 ]; then
  if ! worker_sync "$work/out/" vi-out; then
    echo "on-worker.sh: could not bring the job output home; keeping the worker copy" >&2
    exit 1
  fi
  # A stage that publishes a durable checkpoint dir MUST say how many epochs
  # to expect. Without that, anything it did publish was never verified home,
  # and deleting the worker copy would destroy the only re-push source. Fail
  # closed and keep every copy. The runner gives every stage a checkpoint dir,
  # so an empty durable dir just means this stage publishes nothing: expect 0
  # and let verify clear its marker.
  if [ -n "$durable" ] && [ -z "$expect_epochs" ]; then
    if compgen -G "$durable/epoch-*" >/dev/null; then
      echo "on-worker.sh: $VI_STAGE published epochs but sets no expect_epochs; keeping the worker copy and failing rather than deleting an unverified result" >&2
      exit 1
    fi
    expect_epochs=0
  fi
  if [ -n "$durable" ]; then
    if ! nice -n 19 ionice -c2 -n7 python3 "$jobs_dir/checkpoints.py" verify "$durable" --expect "$expect_epochs"; then
      echo "on-worker.sh: verify failed for $durable" >&2
      exit 1
    fi
  fi
  keep_work=
  exit 0
fi

keep_work=1
echo "on-worker.sh: $VI_STAGE did not complete successfully; keeping the worker copy" >&2
exit "${final_rc:-1}"
