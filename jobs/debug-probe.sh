#!/usr/bin/env bash
# Debug pass, stages 1 and 2: read-only facts about a freshly rented worker, before anything is
# uploaded. jobs/paid_run.py --debug-pass sends this file's text through `vi cloud ssh`, with the
# stage's floors set in front of it as environment variables; the home rehearsal
# (jobs/rehearsal/debug-pass.sh) runs it locally against its stub nvidia-smi.
#
# It prints one status line per stage, then the log lines that back it:
#   debug-pass: stage 1 host-cuda: PASS|FAIL <reason>
#   debug-pass:   | <log line>
# and always exits 0, so a failed stage never hides the other one. Floors (all optional):
#   DEBUG_MIN_VRAM_GB, DEBUG_MIN_CUDA, DEBUG_MIN_DISK_GB, DEBUG_MIN_RAM_GB, DEBUG_MIN_CORES,
#   DEBUG_MIN_NOFILE (default 4096; judged after raising the soft limit to the hard one), DEBUG_MIN_SHM_GB (default 1), DEBUG_DISK_PATH (default $HOME).
# Nothing here writes outside /tmp, installs anything, or reads a credential.

excerpt() { sed -e 's/^/debug-pass:   | /'; }

# Version compare: ver_ge A B is true when A >= B (dotted numbers).
ver_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

# ---- stage 1: host, driver, CUDA -----------------------------------------------------------
s1_log=$(mktemp)
s1_fail=
{
  echo "host: $(uname -n 2>/dev/null) kernel $(uname -r 2>/dev/null)"
  [ -r /etc/os-release ] && echo "os: $(. /etc/os-release; echo "${PRETTY_NAME:-?}")"
} >>"$s1_log"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  s1_fail="nvidia-smi not found"
else
  # One field per query: the rehearsal's stub nvidia-smi answers exactly these.
  nvidia-smi -L >>"$s1_log" 2>&1 || true
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1)
  rc=$?
  q=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1) || rc=$?
  echo "driver: $drv" >>"$s1_log"
  echo "memory.total (MiB): $q" >>"$s1_log"
  if [ "$rc" != 0 ] || [ -z "$q" ]; then
    s1_fail="nvidia-smi failed (exit $rc)"
  else
    cuda=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version: *([0-9.]+).*/\1/p' | head -n 1)
    [ -n "$cuda" ] || cuda=$(nvidia-smi -q 2>/dev/null | sed -nE 's/.*CUDA (UMD )?Version *: *([0-9.]+).*/\2/p' | head -n 1)
    echo "cuda (driver): ${cuda:-unknown}" >>"$s1_log"
    vram_mib=$(printf '%s\n' "$q" | head -n 1 | tr -dc '0-9')
    vram_gb=$(awk -v m="${vram_mib:-0}" 'BEGIN { printf "%.1f", m / 1024 }')
    echo "vram: $vram_gb GiB" >>"$s1_log"
    if [ -z "$cuda" ]; then
      s1_fail="CUDA version not reported by the driver"
    elif [ -n "${DEBUG_MIN_CUDA:-}" ] && ! ver_ge "$cuda" "$DEBUG_MIN_CUDA"; then
      s1_fail="driver CUDA $cuda is under $DEBUG_MIN_CUDA"
    elif [ -n "${DEBUG_MIN_VRAM_GB:-}" ] && awk -v x="$vram_gb" -v y="$DEBUG_MIN_VRAM_GB" 'BEGIN { exit !(x < y) }'; then
      s1_fail="VRAM $vram_gb GiB is under $DEBUG_MIN_VRAM_GB GiB"
    fi
  fi
fi
if [ -n "$s1_fail" ]; then
  echo "debug-pass: stage 1 host-cuda: FAIL $s1_fail"
else
  echo "debug-pass: stage 1 host-cuda: PASS CUDA $cuda, ${vram_gb} GiB"
fi
excerpt <"$s1_log"
rm -f "$s1_log"

# ---- stage 2: disk and limits --------------------------------------------------------------
s2_log=$(mktemp)
s2_fail=()
path=${DEBUG_DISK_PATH:-$HOME}
disk_gb=$(df -Pk "$path" 2>/dev/null | awk 'NR == 2 { printf "%.0f", $4 / 1048576 }')
echo "disk free at $path: ${disk_gb:-?} GB" >>"$s2_log"
# The worker job scripts raise the open-file soft limit to the hard limit (at most 65536) before
# they start (jobs/worker-limits.sh vi_raise_nofile), so the gate judges the RAISED value, never
# the login shell's soft one: round 18's box had soft 1024, hard 1048576, which is fine.
nofile_soft0=$(ulimit -Sn)
nofile_hard=$(ulimit -Hn)
nofile_target=$nofile_hard
case $nofile_target in unlimited) nofile_target=65536 ;; ''|*[!0-9]*) nofile_target= ;; esac
[ -z "$nofile_target" ] || [ "$nofile_target" -le 65536 ] || nofile_target=65536
if [ -n "$nofile_target" ] && [ "$nofile_soft0" != unlimited ] && [ "$nofile_soft0" -lt "$nofile_target" ]; then
  ulimit -Sn "$nofile_target" 2>/dev/null || true
fi
nofile=$(ulimit -Sn)
echo "open files (ulimit -n): soft $nofile_soft0, hard $nofile_hard; raised soft $nofile" >>"$s2_log"
echo "user processes (ulimit -u): $(ulimit -u)" >>"$s2_log"
shm_kb=$(df -Pk /dev/shm 2>/dev/null | awk 'NR == 2 { print $2 }')
shm_gb=$(awk -v k="${shm_kb:-0}" 'BEGIN { printf "%.1f", k / 1048576 }')
if [ -n "$shm_kb" ]; then echo "/dev/shm size: $shm_gb GiB" >>"$s2_log"; else echo "/dev/shm: not mounted" >>"$s2_log"; fi
cores=$(nproc 2>/dev/null || echo 0)
echo "cpus (nproc): $cores" >>"$s2_log"
ram_gb=$(awk '/^MemTotal:/ { printf "%.0f", $2 / 1048576 }' /proc/meminfo 2>/dev/null)
echo "ram: ${ram_gb:-?} GB" >>"$s2_log"
for f in /sys/fs/cgroup/pids.max /sys/fs/cgroup/pids/pids.max; do
  [ -r "$f" ] && echo "cgroup $f: $(cat "$f")" >>"$s2_log"
done
[ -r /sys/fs/cgroup/cpu.max ] && echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)" >>"$s2_log"
[ -r /proc/sys/kernel/threads-max ] && echo "kernel threads-max: $(cat /proc/sys/kernel/threads-max)" >>"$s2_log"
for tool in python3 rsync zstd setsid flock timeout; do
  command -v "$tool" >/dev/null 2>&1 || s2_fail+=("$tool missing")
done
echo "python3: $(python3 --version 2>&1)" >>"$s2_log"
if [ -n "${DEBUG_MIN_DISK_GB:-}" ] && [ "${disk_gb:-0}" -lt "${DEBUG_MIN_DISK_GB%.*}" ]; then
  s2_fail+=("disk ${disk_gb:-?} GB under $DEBUG_MIN_DISK_GB GB")
fi
if [ -n "${DEBUG_MIN_RAM_GB:-}" ] && [ "${ram_gb:-0}" -lt "${DEBUG_MIN_RAM_GB%.*}" ]; then
  s2_fail+=("RAM ${ram_gb:-?} GB under $DEBUG_MIN_RAM_GB GB")
fi
if [ -n "${DEBUG_MIN_CORES:-}" ] && [ "$cores" -lt "${DEBUG_MIN_CORES%.*}" ]; then
  s2_fail+=("$cores cpus under $DEBUG_MIN_CORES")
fi
case $nofile in
  unlimited) ;;
  *) [ "$nofile" -ge "${DEBUG_MIN_NOFILE:-4096}" ] ||
       s2_fail+=("open-file limit: hard limit $nofile_hard (soft raised to $nofile) is under ${DEBUG_MIN_NOFILE:-4096}; the jobs cannot raise it further") ;;
esac
if [ -z "$shm_kb" ] || awk -v x="$shm_gb" -v y="${DEBUG_MIN_SHM_GB:-1}" 'BEGIN { exit !(x < y) }'; then
  s2_fail+=("/dev/shm ${shm_gb} GiB (0 means missing), under ${DEBUG_MIN_SHM_GB:-1} GiB (DataLoader workers share memory there)")
fi
if [ ${#s2_fail[@]} -gt 0 ]; then
  echo "debug-pass: stage 2 disk-limits: FAIL $(IFS=';'; echo "${s2_fail[*]}" | sed 's/;/; /g')"
else
  echo "debug-pass: stage 2 disk-limits: PASS ${disk_gb} GB disk, nofile $nofile, shm ${shm_gb} GiB, $cores cpus, ${ram_gb} GB RAM"
fi
excerpt <"$s2_log"
rm -f "$s2_log"
exit 0
