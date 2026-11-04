#!/usr/bin/env bash
# Worker return path. Home runs `start` and opens a reverse SSH tunnel; the worker runs `push`.
# The unencrypted rsync daemon is safe only with both listeners on 127.0.0.1 and the SSH tunnel up.
set -euo pipefail

cmd=${1:?usage: worker-return.sh start OUT CHECKPOINT STATE [PUBLISH] [PORT] | stop STATE | push SOURCE MODULE [RSYNC_OPTS...]}
shift

free_port() {
  python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

case "$cmd" in
  start)
    out=${1:?output directory}; checkpoint=${2:?checkpoint directory}; state=${3:?state directory}
    publish=${4:-}
    port=${5:-${VI_RETURN_PORT:-}}
    [ -n "$port" ] || port=$(free_port)
    mkdir -p "$out" "$checkpoint" "$state"
    [ -n "$publish" ] && mkdir -p "$publish"
    # The daemon detaches and may not keep this cwd, so resolve every path to an
    # absolute one before writing it into the config; a relative path here would
    # point the daemon at the wrong place (or nowhere) after it forks.
    out=$(cd "$out" && pwd -P)
    checkpoint=$(cd "$checkpoint" && pwd -P)
    [ -n "$publish" ] && publish=$(cd "$publish" && pwd -P)
    state=$(cd "$state" && pwd -P)
    [ -z "$(find "$state" -mindepth 1 -maxdepth 1 -print -quit)" ] || {
      echo "worker-return: state directory is not empty: $state" >&2; exit 1;
    }
    cat >"$state/rsyncd.conf" <<EOF
address = 127.0.0.1
port = $port
pid file = $state/rsyncd.pid
lock file = $state/rsyncd.lock
log file = $state/rsyncd.log
use chroot = false
munge symlinks = true
max connections = 1

[vi-out]
path = $out
read only = false
write only = true
refuse options = delete
list = false
hosts allow = 127.0.0.1

[vi-checkpoint]
path = $checkpoint
read only = false
write only = true
refuse options = delete
list = false
hosts allow = 127.0.0.1
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
EOF
    fi
    printf '%s\n' "$port" >"$state/port"
    rsync --daemon --config="$state/rsyncd.conf"
    for _ in {1..50}; do
      [ -s "$state/rsyncd.pid" ] && break
      sleep 0.1
    done
    [ -s "$state/rsyncd.pid" ] || { cat "$state/rsyncd.log" >&2 || true; exit 1; }
    kill -0 "$(cat "$state/rsyncd.pid")"
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
    rm -f "$state/rsyncd.conf" "$state/rsyncd.pid" "$state/rsyncd.lock" "$state/rsyncd.log" "$state/port" "$state/ssh.log"
    rmdir "$state" 2>/dev/null || true
    ;;
  push)
    src=${1:?source}; module=${2:?module vi-out, vi-checkpoint or vi-publish}; shift 2
    : "${VI_RETURN_PORT:?worker-return push needs the tunnel port}"
    case "$module" in vi-out|vi-checkpoint|vi-publish) ;; *) echo "worker-return: unknown module $module" >&2; exit 2 ;; esac
    if [ -d "$src" ] && [[ "$src" != */ ]]; then src="$src/"; fi
    for _ in {1..20}; do
      if rsync -a "$@" "$src" "rsync://127.0.0.1:$VI_RETURN_PORT/$module/"; then exit 0; else rc=$?; fi
      [ "$rc" -eq 5 ] || exit "$rc"
      sleep 0.25
    done
    exit "$rc"
    ;;
  *) echo "worker-return: unknown command $cmd" >&2; exit 2 ;;
esac
