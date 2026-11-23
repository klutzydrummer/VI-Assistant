# shellcheck shell=bash
# Thread and process limits for the worker's Python jobs, and a watched launcher for the
# tokeniser. Sourced by jobs/voicetune-fresh.sh before any Python starts, by
# jobs/worker-preflight.sh for its "tasks" row, and by jobs/rehearsal/vast-limits.sh so the home
# rehearsal runs exactly this code.
#
# Why: r7 died on a rented box that showed 48 CPUs but allowed only a few thousand tasks. A cgroup
# pids.max counts threads, not just processes, and OpenBLAS sizes its thread pool from the CPUs it
# can see (48), not from the CPU quota. Every tokeniser process started a 48-thread pool, a pool
# thread could not be created, OpenBLAS raised SIGINT, the process pool broke, and the stage then
# hung forever because an orphaned Manager process kept the `| tee` pipe open.
#
# The rule (round13 diagnosis, fork F1):
#   cpus        = min(nproc, cpu.max quota/period, CPUs in cpuset.cpus.effective), at least 1
#   tasks_free  = the smallest of pids.max - pids.current (every cgroup level up to the root),
#                 ulimit -u and kernel threads-max, each minus the tasks already running
#   nj          = min(2, cpus)               tokeniser worker processes
#   procs       = 3 + nj                     main, Manager, resource tracker, workers
#   threads     = max(1, min(cpus / nj, tasks_free / 2 / procs / 3)), at most 2
# and the job is refused (a retryable machine gate) when tasks_free < procs * 8.
#
# Test hooks: CGROUP_ROOT names one cgroup folder to read instead of this process's own cgroup and
# its parents; VI_LIMITS_PROC replaces /proc; VI_LIMITS_NPROC replaces `nproc`.

# The tokeniser log lines that mean a worker or the pool died. Kept in one place so the watcher,
# paid_run.py and the rehearsal judge agree.
VI_FATAL_RE="BrokenProcessPool|pthread_create failed|can't start new thread|Traceback"

# A large stand-in for "no limit", so the arithmetic below needs no special cases.
vi_unlimited=1000000000

# vi_raise_nofile: raise this shell's open-file soft limit to its hard limit, capped at 65536 when
# the hard limit is larger or unlimited; a soft limit already at or above that target is left
# alone. Children (the tokeniser pool, DataLoader workers, accelerate) inherit the raised limit.
# Round 18's box had soft 1024, hard 1048576: the job ran with 1024 although it could have
# asked for more. It sets vi_lim_nofile_soft/vi_lim_nofile_hard to what is in force afterwards.
vi_raise_nofile() {
  local hard soft target
  hard=$(ulimit -Hn 2>/dev/null) || hard=
  soft=$(ulimit -Sn 2>/dev/null) || soft=
  target=$hard
  case $target in unlimited) target=65536 ;; ''|*[!0-9]*) target= ;; esac
  [ -z "$target" ] || [ "$target" -le 65536 ] || target=65536
  if [ -n "$target" ]; then
    case $soft in
      unlimited) ;;
      ''|*[!0-9]*) ulimit -Sn "$target" 2>/dev/null || true ;;
      *) [ "$soft" -ge "$target" ] || ulimit -Sn "$target" 2>/dev/null || true ;;
    esac
  fi
  vi_lim_nofile_soft=$(ulimit -Sn 2>/dev/null || echo '?')
  vi_lim_nofile_hard=$(ulimit -Hn 2>/dev/null || echo '?')
}

# vi_cpuset_count LIST: number of CPUs in a cpuset list such as "0-3,8,10-11".
vi_cpuset_count() {
  local list=$1 part n=0 a b
  [ -n "$list" ] || { echo 0; return; }
  local IFS=,
  for part in $list; do
    case $part in
      *-*) a=${part%-*}; b=${part#*-}; n=$((n + b - a + 1)) ;;
      '') ;;
      *) n=$((n + 1)) ;;
    esac
  done
  echo "$n"
}

# vi_cgroup_dirs: this process's cgroup folder and each parent up to the root, one per line. With
# CGROUP_ROOT set, only that folder.
vi_cgroup_dirs() {
  if [ -n "${CGROUP_ROOT:-}" ]; then
    echo "$CGROUP_ROOT"
    return
  fi
  local rel dir
  rel=$(awk -F: '$1 == "0" { print $3; exit }' "${VI_LIMITS_PROC:-/proc}/self/cgroup" 2>/dev/null)
  dir=/sys/fs/cgroup${rel%/}
  while :; do
    echo "$dir"
    [ "$dir" = /sys/fs/cgroup ] && break
    dir=${dir%/*}
    [ -n "$dir" ] && [ "$dir" != /sys/fs ] || break
  done
}

# worker_limits_measure: sets the vi_lim_* variables below from the machine, prints nothing.
worker_limits_measure() {
  local proc=${VI_LIMITS_PROC:-/proc} dir v q p free total tmax ulim
  # nproc obeys OMP_NUM_THREADS and OMP_THREAD_LIMIT, which an image may preset; read it without.
  vi_lim_nproc=${VI_LIMITS_NPROC:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 1)}
  vi_lim_quota=max vi_lim_cpuset=unknown vi_lim_pids_max=max vi_lim_pids_current=unknown
  local cpus=$vi_lim_nproc
  vi_lim_tasks_free=$vi_unlimited
  while read -r dir; do
    [ -d "$dir" ] || continue
    if [ -r "$dir/cpu.max" ]; then
      read -r q p <"$dir/cpu.max" || true
      if [ "${q:-max}" != max ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
        v=$((q / p)); [ "$v" -ge 1 ] || v=1
        [ "$vi_lim_quota" = max ] || [ "$v" -lt "$vi_lim_quota" ] || v=$vi_lim_quota
        vi_lim_quota=$v
      fi
    fi
    if [ "$vi_lim_cpuset" = unknown ] && [ -r "$dir/cpuset.cpus.effective" ]; then
      v=$(vi_cpuset_count "$(cat "$dir/cpuset.cpus.effective")")
      [ "$v" -gt 0 ] && vi_lim_cpuset=$v
    fi
    if [ -r "$dir/pids.max" ]; then
      v=$(cat "$dir/pids.max")
      if [ "$v" != max ]; then
        p=$(cat "$dir/pids.current" 2>/dev/null || echo 0)
        free=$((v - p))
        if [ "$free" -lt "$vi_lim_tasks_free" ]; then
          vi_lim_tasks_free=$free vi_lim_pids_max=$v vi_lim_pids_current=$p
        fi
      elif [ "$vi_lim_pids_current" = unknown ] && [ -r "$dir/pids.current" ]; then
        vi_lim_pids_current=$(cat "$dir/pids.current")
      fi
    fi
  done < <(vi_cgroup_dirs)
  [ "$vi_lim_quota" = max ] || [ "$vi_lim_quota" -ge "$cpus" ] || cpus=$vi_lim_quota
  [ "$vi_lim_cpuset" = unknown ] || [ "$vi_lim_cpuset" -ge "$cpus" ] || cpus=$vi_lim_cpuset
  [ "$cpus" -ge 1 ] 2>/dev/null || cpus=1
  vi_lim_cpus=$cpus

  # The fourth /proc/loadavg field is running/total tasks on the machine; it stands in for the
  # tasks already counted against threads-max and ulimit -u (an overcount, which only errs safe).
  total=$(awk '{ split($4, a, "/"); print a[2] + 0 }' "$proc/loadavg" 2>/dev/null)
  total=${total:-0}
  tmax=$(cat "$proc/sys/kernel/threads-max" 2>/dev/null || echo "")
  case $tmax in ''|*[!0-9]*) vi_lim_threads_max=max ;; *)
    vi_lim_threads_max=$tmax
    free=$((tmax - total)); [ "$free" -ge "$vi_lim_tasks_free" ] || vi_lim_tasks_free=$free ;;
  esac
  ulim=$(ulimit -u 2>/dev/null || echo unlimited)
  case $ulim in ''|*[!0-9]*) vi_lim_ulimit_u=unlimited ;; *)
    vi_lim_ulimit_u=$ulim
    free=$((ulim - total)); [ "$free" -ge "$vi_lim_tasks_free" ] || vi_lim_tasks_free=$free ;;
  esac
  [ "$vi_lim_tasks_free" -ge 0 ] || vi_lim_tasks_free=0

  vi_lim_nj=2; [ "$cpus" -ge 2 ] || vi_lim_nj=1
  vi_lim_procs=$((3 + vi_lim_nj))
  local by_cpu=$((cpus / vi_lim_nj)) by_tasks=$((vi_lim_tasks_free / 2 / vi_lim_procs / 3)) t
  t=$by_cpu; [ "$by_tasks" -ge "$t" ] || t=$by_tasks
  [ "$t" -ge 1 ] || t=1
  [ "$t" -le 2 ] || t=2
  vi_lim_threads=$t
  vi_lim_need=$((vi_lim_procs * 8))
}

# worker_limits_line: the one printed summary (no fatal words in it, so log watchers ignore it).
worker_limits_line() {
  local free=$vi_lim_tasks_free
  [ "$free" -lt "$vi_unlimited" ] || free=unlimited
  echo "limits: nproc=$vi_lim_nproc cpus=$vi_lim_cpus quota=$vi_lim_quota cpuset=$vi_lim_cpuset pids.max=$vi_lim_pids_max current=$vi_lim_pids_current ulimit-u=$vi_lim_ulimit_u threads-max=$vi_lim_threads_max tasks_free=$free nofile-soft=${vi_lim_nofile_soft:-$(ulimit -Sn)} nofile-hard=${vi_lim_nofile_hard:-$(ulimit -Hn)} -> threads=$vi_lim_threads nj=$vi_lim_nj"
}

# worker_limits: raise the open-file soft limit (vi_raise_nofile), measure, print the limits line, export the thread caps and VI_LIMIT_NJ. Returns 1
# after a "worker-preflight: tasks ..." line (which makes the run retry on another machine) when
# the box has too few free task slots for even the capped job.
worker_limits() {
  vi_raise_nofile
  worker_limits_measure
  worker_limits_line
  if [ "$vi_lim_tasks_free" -lt "$vi_lim_need" ]; then
    echo "worker-preflight: tasks: only $vi_lim_tasks_free free task slots (pids.max=$vi_lim_pids_max current=$vi_lim_pids_current ulimit-u=$vi_lim_ulimit_u threads-max=$vi_lim_threads_max), need at least $vi_lim_need for $vi_lim_procs processes" >&2
    return 1
  fi
  export OMP_NUM_THREADS=$vi_lim_threads OPENBLAS_NUM_THREADS=$vi_lim_threads \
    MKL_NUM_THREADS=$vi_lim_threads NUMEXPR_NUM_THREADS=$vi_lim_threads VI_LIMIT_NJ=$vi_lim_nj
}

# vi_run_watched LOG COMMAND [ARG]...: run COMMAND in its own session (so its process group holds
# the Manager and every pool worker), with its output in LOG and shown live, and return its exit
# code. Three things end it early:
#   - a fatal line (VI_FATAL_RE) followed by VI_FATAL_QUIET_SECONDS (120) with no new output:
#     the whole group is killed and the return code is 75;
#   - VI_RUN_TIMEOUT_SECONDS (3 h) in total: timeout kills the group, then SIGKILL 30 s later;
#   - after COMMAND exits, any process still in its group (an orphaned Manager) is killed, so
#     nothing is left holding the log or the stage's output open.
# The output goes to a file rather than through `| tee`, so an orphan can never keep a pipe open.
vi_run_watched() {
  local log=$1; shift
  local quiet=${VI_FATAL_QUIET_SECONDS:-120} poll=${VI_WATCH_POLL_SECONDS:-2}
  local limit=${VI_RUN_TIMEOUT_SECONDS:-10800} pid tail_pid='' rc fatal_at='' size=0 cur now killed=0
  : >"$log"
  setsid timeout --kill-after=30 "$limit" "$@" >>"$log" 2>&1 </dev/null &
  pid=$!
  # VI_WATCH_SHOW=0 keeps the output in LOG only (the tasks smoke, whose crash lines must not
  # reach the job log).
  if [ "${VI_WATCH_SHOW:-1}" = 1 ]; then
    tail -n +1 -f --pid="$pid" "$log" 2>/dev/null &
    tail_pid=$!
  fi
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$poll"
    now=$(date +%s)
    cur=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ -z "$fatal_at" ]; then
      if grep -Eq "$VI_FATAL_RE" "$log" 2>/dev/null; then fatal_at=$now size=$cur; fi
    elif [ "$cur" != "$size" ]; then
      fatal_at=$now size=$cur
    elif [ $((now - fatal_at)) -ge "$quiet" ]; then
      echo "vi_run_watched: a fatal line in $log and no new output for ${quiet}s; stopping the job's process group" >&2
      kill -TERM -- "-$pid" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-$pid" 2>/dev/null || true
      killed=1
      break
    fi
  done
  rc=0
  wait "$pid" || rc=$?
  # The command is gone; clear whatever it left in its group (the r7 orphaned Manager).
  kill -KILL -- "-$pid" 2>/dev/null || true
  [ -z "$tail_pid" ] || wait "$tail_pid" 2>/dev/null || true
  [ "$killed" = 0 ] || rc=75
  return "$rc"
}
