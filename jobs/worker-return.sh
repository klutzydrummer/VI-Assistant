#!/usr/bin/env bash
# Worker return path. Home runs `start`, `restart` and `sweep`; the worker runs `push`.
# The unencrypted rsync daemon is safe only with both listeners on 127.0.0.1 and the SSH tunnel up.
set -euo pipefail

cmd=${1:?usage: worker-return.sh start OUT CHECKPOINT STATE [PUBLISH] [PORT] | restart STATE | stop STATE | sweep [BASE] | push SOURCE MODULE [RSYNC_OPTS...]}
shift

free_port() {
  python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

# Start the receiver in the background, tied to CALLER for its lifetime: it
# keeps running after start/restart return, but is stopped when the home
# process that asked for it dies (even by SIGKILL). The subshell holds
# owner.lock for exactly the receiver's lifetime, so a live state directory
# always has a locked owner.lock and `sweep` can spare it. rsync runs with
# --no-detach so the subshell can reap it; setpriv makes it die with the
# subshell, and a watcher stops it if only CALLER dies.
receiver_start() {  # STATE CALLER
  local state=$1 caller=$2
  (
    exec 9>>"$state/owner.lock"
    # A live receiver holds this lock; wait briefly for a just-killed
    # predecessor to release it before giving up loudly.
    flock -w 3 9 || { echo "worker-return: state directory is owned by a live receiver: $state" >&2; exit 1; }
    setpriv --pdeathsig TERM rsync --daemon --no-detach --config="$state/rsyncd.conf" &
    daemon=$!
    (
      exec 9>&-
      while kill -0 "$caller" 2>/dev/null; do sleep 1; done
      kill "$daemon" 2>/dev/null || true
    ) &
    watcher=$!
    wait "$daemon" 2>/dev/null || true
    kill "$watcher" 2>/dev/null || true
  ) </dev/null >>"$state/rsyncd.log" 2>&1 &
}

receiver_ready() {  # STATE
  local state=$1 pid
  for _ in {1..50}; do
    [ -s "$state/rsyncd.pid" ] && break
    sleep 0.1
  done
  [ -s "$state/rsyncd.pid" ] || { cat "$state/rsyncd.log" >&2 || true; exit 1; }
  pid=$(cat "$state/rsyncd.pid")
  kill -0 "$pid" 2>/dev/null || { cat "$state/rsyncd.log" >&2 || true; exit 1; }
}

# Prove every stored module accepts a write after a restart, and fail loudly
# if one refuses (for example because a pushed rsyncd-munged dir wedged it).
receiver_verify() {  # STATE PORT
  local state=$1 port=$2 module path token tmp n=0 ok
  token=$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')
  tmp=$(mktemp -d)
  printf '%s\n' "$token" >"$tmp/.restart-token"
  while IFS=$'\t' read -r module path; do
    n=$((n + 1))
    ok=
    for _ in {1..20}; do
      if rsync -a "$tmp/.restart-token" "rsync://127.0.0.1:$port/$module/"; then ok=1; break; fi
      sleep 0.1
    done
    if [ -z "$ok" ]; then
      echo "worker-return: receiver module $module refused a token after restart" >&2
      rm -rf "$tmp"
      exit 1
    fi
    rm -f "$path/.restart-token"
  done < <(awk '/^\[/ { m=substr($0,2,length($0)-2); next }
                 /^path[ \t]*=/ { p=$0; sub(/^path[ \t]*=[ \t]*/,"",p); print m "\t" p }' "$state/rsyncd.conf")
  rm -rf "$tmp"
  [ "$n" -gt 0 ] || { echo "worker-return: no modules in $state/rsyncd.conf" >&2; exit 1; }
}

case "$cmd" in
  start)
    out=${1:?output directory}; checkpoint=${2:?checkpoint directory}; state=${3:?state directory}
    publish=${4:-}
    port=${5:-${VI_RETURN_PORT:-}}
    [ -n "$port" ] || port=$(free_port)
    mkdir -p "$out" "$checkpoint" "$state"
    [ -n "$publish" ] && mkdir -p "$publish"
    # The daemon continues after `start` returns and does not keep this cwd, so
    # resolve every path to an absolute one before writing it into the config; a
    # relative path here would point the daemon at the wrong place (or nowhere).
    out=$(cd "$out" && pwd -P)
    checkpoint=$(cd "$checkpoint" && pwd -P)
    [ -n "$publish" ] && publish=$(cd "$publish" && pwd -P)
    state=$(cd "$state" && pwd -P)
    [ -z "$(find "$state" -mindepth 1 -maxdepth 1 -print -quit)" ] || {
      echo "worker-return: state directory is not empty: $state" >&2; exit 1;
    }
    # Per-module lock files and max connections give each module its own single
    # slot, so a long publish never starves the checkpoint or output sync. The
    # module default for max connections is 0 (unlimited), so both must be set
    # in every module; there is deliberately no global lock or connection limit.
    # exclude stops a worker-pushed rsyncd-munged dir from wedging a restart.
    cat >"$state/rsyncd.conf" <<EOF
address = 127.0.0.1
port = $port
pid file = $state/rsyncd.pid
log file = $state/rsyncd.log
use chroot = false
munge symlinks = true

[vi-out]
path = $out
read only = false
write only = true
refuse options = delete
list = false
hosts allow = 127.0.0.1
max connections = 1
lock file = $state/vi-out.lock
exclude = /rsyncd-munged/

[vi-checkpoint]
path = $checkpoint
read only = false
write only = true
refuse options = delete
list = false
hosts allow = 127.0.0.1
max connections = 1
lock file = $state/vi-checkpoint.lock
exclude = /rsyncd-munged/
EOF
    if [ -n "$publish" ]; then
      cat >>"$state/rsyncd.conf" <<EOF

[vi-publish]
path = $publish
read only = false
write only = true
refuse options = delete
list = false
hosts allow = 127.0.0.1
max connections = 1
lock file = $state/vi-publish.lock
exclude = /rsyncd-munged/
EOF
    fi
    printf '%s\n' "$port" >"$state/port"
    receiver_start "$state" "$PPID"
    receiver_ready "$state"
    printf '%s\n' "$port"
    ;;
  restart)
    state=${1:?state directory}
    [ -s "$state/rsyncd.conf" ] || { echo "worker-return: no stored receiver config in $state" >&2; exit 1; }
    port=$(awk -F' *= *' '$1 == "port" { print $2; exit }' "$state/rsyncd.conf")
    [ -n "$port" ] || port=$(cat "$state/port" 2>/dev/null || true)
    [ -n "$port" ] || { echo "worker-return: cannot read the receiver port from $state" >&2; exit 1; }
    # A restart reuses the stored config byte-for-byte. Only the dead
    # receiver's pid and per-module connection-lock files are cleared; the
    # owner lock is left for the new receiver to take over.
    rm -f "$state/rsyncd.pid" "$state/rsyncd.lock" "$state"/vi-*.lock
    receiver_start "$state" "$PPID"
    receiver_ready "$state"
    receiver_verify "$state" "$port"
    printf '%s\n' "$port"
    ;;
  stop)
    state=${1:?state directory}
    if [ -s "$state/rsyncd.pid" ]; then
      pid=$(cat "$state/rsyncd.pid")
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
      # Only escalate if it is still alive: the PID may have been reused.
      if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
    fi
    rm -f "$state/rsyncd.conf" "$state/rsyncd.pid" "$state/rsyncd.lock" "$state/rsyncd.log" "$state/port" "$state/ssh.log" "$state/owner.lock" "$state"/vi-*.lock
    rmdir "$state" 2>/dev/null || true
    ;;
  sweep)
    # Kill receivers left behind by a dead home process. A state directory is
    # abandoned only when its owner.lock can be taken and its pid file names a
    # process actually running that directory's config; a live run is spared.
    base=${1:-${VI_RETURN_STATE_DIR:-}}
    if [ -z "$base" ]; then
      base=${VI_STATE_DIR:-}
      [ -n "$base" ] || base=${XDG_STATE_HOME:-${HOME:-.}/.local/state}/vi
      base=$base/return
    fi
    [ -d "$base" ] || exit 0
    while IFS= read -r pidfile; do
      (
        dir=$(dirname "$pidfile")
        pid=$(cat "$pidfile" 2>/dev/null) || exit 0
        case "$pid" in ''|*[!0-9]*) exit 0 ;; esac
        exec 9>>"$dir/owner.lock"
        flock -n 9 || exit 0
        cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null) || exit 0
        case "$cmdline" in *"$dir/rsyncd.conf"*) ;; *) exit 0 ;; esac
        kill "$pid" 2>/dev/null || exit 0
        echo "worker-return: swept stale receiver pid $pid in $dir" >&2
      )
    done < <(find "$base" -maxdepth 2 -type f -name '*.pid' 2>/dev/null)
    ;;
  push)
    src=${1:?source}; module=${2:?module vi-out, vi-checkpoint or vi-publish}; shift 2
    case "$module" in vi-out|vi-checkpoint|vi-publish) ;; *) echo "worker-return: unknown module $module" >&2; exit 2 ;; esac
    # Home writes the remote tunnel port to $work/.return-port on every
    # reconnect; the worker learns the work dir from VI_OUT. A missing or
    # half-written file is ignored, so a flap keeps the last port it read.
    port_file=${VI_RETURN_PORT_FILE:-}
    [ -n "$port_file" ] || [ -z "${VI_OUT:-}" ] || port_file=${VI_OUT%/*}/.return-port
    last_port=${VI_RETURN_PORT:-}
    [ -n "$last_port" ] || [ -n "$port_file" ] || {
      echo "worker-return: push needs VI_RETURN_PORT or a .return-port file" >&2; exit 2;
    }
    wait_s=${VI_PUSH_WAIT:-300}
    case "$wait_s" in ''|*[!0-9]*) wait_s=300 ;; esac
    is_dir=; [ -d "$src" ] && is_dir=1
    [ -z "$is_dir" ] || [[ "$src" == */ ]] || src="$src/"
    defaults=()
    # Reuse the bytes already home after an interrupted push instead of
    # restarting the file from zero; rsync still checksums the whole file.
    case "$module" in vi-out|vi-publish) defaults+=(--partial-dir=.rsync-partial) ;; esac

    read_port() {
      [ -n "$port_file" ] || return 0
      local p
      p=$(head -n1 "$port_file" 2>/dev/null | tr -d ' \t\r' || true)
      case "$p" in ''|*[!0-9]*) return 0 ;; esac
      last_port=$p
    }
    push_once() {
      read_port
      [ -n "$last_port" ] || { echo "worker-return: no usable return port yet" >&2; return 10; }
      rsync -a "$@" "$src" "rsync://127.0.0.1:$last_port/$module/"
    }
    push_retry() {
      local deadline=$(( $(date +%s) + wait_s )) delay=1 rc
      while :; do
        if push_once "$@"; then return 0; else rc=$?; fi
        # rsync(1): 5 protocol start, 10 socket I/O, 12 data stream,
        # 30 data timeout, 35 daemon-connect timeout. Everything else is real.
        case " 5 10 12 30 35 " in *" $rc "*) ;; *) return "$rc" ;; esac
        [ "$(date +%s)" -lt "$deadline" ] || {
          echo "worker-return: giving up after rsync exit $rc (waited ${wait_s}s)" >&2
          return "$rc"
        }
        echo "worker-return: rsync exit $rc; retrying in ${delay}s" >&2
        sleep "$delay"
        [ "$delay" -ge 8 ] || delay=$((delay * 2))
      done
    }
    if [ -n "$is_dir" ] && [ "$module" = vi-publish ]; then
      # Push SHA256SUMS last, so a half-pushed epoch is never mistaken for a
      # complete one by home's verification.
      push_retry "${defaults[@]}" --exclude=SHA256SUMS "$@"
      push_retry "${defaults[@]}" "$@"
    else
      push_retry "${defaults[@]}" "$@"
    fi
    ;;
  *) echo "worker-return: unknown command $cmd" >&2; exit 2 ;;
esac
