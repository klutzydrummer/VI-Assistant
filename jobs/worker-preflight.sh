#!/usr/bin/env bash
# Cheap checks on a fresh worker, run by jobs/on-worker.sh right after the
# return path is proved and BEFORE any input is uploaded. Every check is a
# couple of seconds, so a bad card, disk or line is refused while nothing but
# the rental is billed. On success it leaves $VI_OUT/preflight.json for the
# stage (jobs/voicetune-fresh.sh merges the probe push and loadbench into it).
#
# The gates come from the stage settings JSON (the home side's $VI_SETTINGS):
#   min_vram_gb       card memory, GiB (nvidia-smi MiB / 1024; a "24 GB" card reads 23.9, so use 23.5)
#   min_disk_mb_s     sequential disk read AND write, MB/s (decimal, dd)
#   min_disk_gb       free space on the job disk, GiB (statvfs / 2**30, the unit check_and_gate uses)
#   min_ram_gb        MemTotal, GiB (an advertised 30 GB host reads 28.6, so use 28)
#   min_cuda_version  CUDA version nvidia-smi reports (the driver's), compared as a number
#   min_cpu_cores     effective cores (nproc capped by a cgroup cpu.max quota)
#   min_push_mb_s     worker -> home rate measured by a raw 100 MiB dummy push
# One gate needs no setting: the tasks row refuses a box whose free task slots (cgroup pids.max,
# ulimit -u, threads-max) are under what the capped tokeniser needs (jobs/worker-limits.sh).
# A setting that is absent or empty turns that gate off; the facts are still
# recorded. nvidia-smi is optional unless min_vram_gb asks for it, so the same
# script runs on a CPU-only test worker.
#
# VI_FAKE_PUSH_SECONDS is a test hook: it replaces the measured time of the
# dummy push, so a slow line can be simulated without waiting for one.
set -euo pipefail

jobs_dir=$(cd "$(dirname "$0")" && pwd)
: "${VI_OUT:?worker-preflight.sh runs through jobs/on-worker.sh}" \
  "${VI_SETTINGS:?missing VI_SETTINGS}" \
  "${VI_RETURN_PORT:?missing VI_RETURN_PORT}"
work=$(cd "$(dirname "$VI_OUT")" && pwd)
report=$VI_OUT/preflight.json
export VI_PUSH_WAIT=${VI_PUSH_WAIT:-120}

setting() {
  python3 -c 'import json, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2], ""); print("" if v is None else v)' \
    "$VI_SETTINGS" "$1"
}
min_vram=$(setting min_vram_gb)
min_disk=$(setting min_disk_mb_s)
min_cores=$(setting min_cpu_cores)
min_disk_gb=$(setting min_disk_gb)
min_ram=$(setting min_ram_gb)
min_cuda=$(setting min_cuda_version)
min_push=$(setting min_push_mb_s)

fail() {
  echo "worker-preflight: $*" >&2
  exit 1
}

# at_least VALUE MIN LABEL: fail unless the measured number clears the setting.
at_least() {
  python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$1" "$2" ||
    fail "$3 is $1, under the required $2"
}

# version_at_least VALUE MIN LABEL: dotted versions compared part by part, so 12.10 beats 12.8.
version_at_least() {
  python3 -c 'import sys
def v(x): return tuple(int(p) for p in x.split("."))
sys.exit(0 if v(sys.argv[1]) >= v(sys.argv[2]) else 1)' "$1" "$2" 2>/dev/null ||
    fail "$3 is $1, under the required $2"
}

rate_mb_s() {  # BYTES MILLISECONDS -> decimal MB/s
  python3 -c 'import sys; b, ms = int(sys.argv[1]), max(1, int(sys.argv[2])); print(round(b / (ms / 1000) / 1e6, 2))' "$1" "$2"
}

# The tools the runner and the detached stage need before anything is sent.
command -v setsid >/dev/null 2>&1 || fail "setsid is missing"
command -v flock >/dev/null 2>&1 || fail "flock is missing"
command -v python3 >/dev/null 2>&1 || fail "python3 is missing"

# Row 34: RAM and effective cores. A container's cpu.max quota can be far
# below nproc on a shared host, and it is the quota that decides step time.
ram_gb=$(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
cpu_cores=$(nproc)
cpu_eff=$cpu_cores
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r quota period < /sys/fs/cgroup/cpu.max || true
  case "$quota" in
    ''|max) ;;
    *) cpu_eff=$(python3 -c 'import sys; print(round(int(sys.argv[1]) / int(sys.argv[2]), 2))' "$quota" "$period") ;;
  esac
fi
[ -z "$min_cores" ] || at_least "$cpu_eff" "$min_cores" "effective CPU cores"

# Tasks row (r7): a cgroup pids.max counts threads as well as processes, and r7's box showed 48
# CPUs but allowed only a few thousand tasks, so the tokeniser's thread pools could not start.
# jobs/worker-limits.sh reads pids.max/pids.current at every cgroup level, nproc, cpu.max, the
# cpuset, ulimit -u and kernel threads-max, and works out the capped thread and worker counts the
# stage will use; a box with too little room even for those is refused here, before any upload.
# (The stage's preflight then runs an 8-clip real tokeniser smoke with the same caps; see
# tokenise_smoke in jobs/voicetune-fresh.sh.)
# shellcheck source=jobs/worker-limits.sh
. "$jobs_dir/worker-limits.sh"
worker_limits_measure
worker_limits_line >&2
if [ "$vi_lim_tasks_free" -lt "$vi_lim_need" ]; then
  fail "tasks: only $vi_lim_tasks_free free task slots (pids.max=$vi_lim_pids_max current=$vi_lim_pids_current ulimit-u=$vi_lim_ulimit_u threads-max=$vi_lim_threads_max), need at least $vi_lim_need for $vi_lim_procs processes"
fi
tasks_free=$vi_lim_tasks_free
[ "$tasks_free" -lt "$vi_unlimited" ] || tasks_free=null

# Free space on the job disk, already net of any kept attempts below $work.
free_disk_gb=$(df -Pk "$work" | awk 'NR == 2 {printf "%.1f", $4 / 1048576}')

# Gates that need only /proc and statvfs: RAM and free disk. Every one of them
# runs before the 100 MiB push and before anything is uploaded.
[ -z "$min_ram" ] || at_least "$ram_gb" "$min_ram" "RAM (GiB)"
[ -z "$min_disk_gb" ] || at_least "$free_disk_gb" "$min_disk_gb" "free disk (GiB)"

# VRAM and the static card facts. The under-load clock query is row 31's job
# and runs inside the row-35 loadbench; this is the idle, pre-venv snapshot.
vram_gb= driver_version= cuda_version= cuda_header= nvidia_facts='[]'
if command -v nvidia-smi >/dev/null 2>&1; then
  vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null |
    head -n1 | python3 -c 'import sys; v = sys.stdin.read().strip()
try:
    print(round(float(v) / 1024, 1))
except ValueError:
    print("")' || true)
  driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '\r' || true)
  smi_text=$(nvidia-smi 2>&1 || true)
  # Older drivers print `CUDA Version: 12.8` in the header; driver 615 prints
  # `KMD Version: 615.71.09  CUDA UMD Version: 13.4` instead (omnivoice-ft-r8 attempt 1). Both count.
  cuda_version=$(printf '%s\n' "$smi_text" | sed -En 's/.*CUDA (UMD )?Version: *([0-9.]+).*/\2/p' | head -n1 || true)
  # What the failure message shows when no version parses: the raw header line, so a host that
  # prints `CUDA Version: N/A` is told apart from one that prints no such line at all.
  cuda_header=$(printf '%s\n' "$smi_text" | grep -m1 -E 'CUDA (UMD )?Version' || true)
  [ -n "$cuda_header" ] || cuda_header="no 'CUDA Version' or 'CUDA UMD Version' line; nvidia-smi began: $(printf '%s\n' "$smi_text" | head -n5 | tr '\n' '|')"
  clock_csv=$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_throttle_reasons.active \
    --format=csv 2>/dev/null || true)
  nvidia_facts=$(PYTHONPATH="$jobs_dir" python3 - "$clock_csv" <<'PY'
import json, sys, voicetune
print(json.dumps(voicetune.parse_nvidia_smi_csv(sys.argv[1])))
PY
)
fi
if [ -n "$min_vram" ]; then
  [ -n "$vram_gb" ] || fail "min_vram_gb is set but nvidia-smi reported no memory"
  at_least "$vram_gb" "$min_vram" "VRAM (GiB)"
fi
if [ -n "$min_cuda" ]; then
  if [ -z "$cuda_version" ]; then
    [ -n "$cuda_header" ] || cuda_header="nvidia-smi is not installed on this worker"
    case "$cuda_header" in
      *N/A*) na=" (it reads N/A: the host driver is not usable from this container)" ;;
      *) na= ;;
    esac
    fail "min_cuda_version is set but nvidia-smi reported no CUDA version; raw header: ${cuda_header}${na}"
  fi
  version_at_least "$cuda_version" "$min_cuda" "CUDA version"
fi

# Sequential disk probe. O_DIRECT when the filesystem takes it, buffered
# otherwise (a tmpfs in a test run); the fallback is recorded so a number from
# page cache is never mistaken for a disk speed.
disk_probe=$work/.preflight-disk
disk_mb=128
disk_direct=false
disk_write_mb_s= disk_read_mb_s=
t0=$(date +%s%N)
t1=$t0
if dd if=/dev/zero of="$disk_probe" bs=1M count=$disk_mb oflag=direct 2>/dev/null; then
  disk_direct=true
  t1=$(date +%s%N)
elif t0=$(date +%s%N); dd if=/dev/zero of="$disk_probe" bs=1M count=$disk_mb 2>/dev/null; then
  t1=$(date +%s%N)
fi
size=$(stat -c %s "$disk_probe" 2>/dev/null || echo 0)
if [ "$size" -gt 0 ]; then
  disk_write_mb_s=$(rate_mb_s "$size" $(((t1 - t0) / 1000000)))
  if [ "$disk_direct" = true ]; then
    t0=$(date +%s%N)
    dd if="$disk_probe" of=/dev/null bs=1M iflag=direct 2>/dev/null || true
    t1=$(date +%s%N)
  else
    t0=$(date +%s%N)
    dd if="$disk_probe" of=/dev/null bs=1M 2>/dev/null || true
    t1=$(date +%s%N)
  fi
  disk_read_mb_s=$(rate_mb_s "$size" $(((t1 - t0) / 1000000)))
fi
rm -f "$disk_probe"
if [ -n "$min_disk" ]; then
  [ -n "$disk_write_mb_s" ] || fail "min_disk_mb_s is set but the disk probe failed"
  at_least "$disk_write_mb_s" "$min_disk" "disk write (MB/s)"
  at_least "$disk_read_mb_s" "$min_disk" "disk read (MB/s)"
fi

# Raw worker -> home dummy push: the rate the epochs will actually get, with
# no stage output mixed in. It lands in the vi-out module as .speedtest; home
# (on-worker.sh) deletes that copy as soon as this returns, and the worker's
# own copy is deleted here. The 100 MiB file is random so no compression or
# sparse-file shortcut hides a slow line.
speed_file=$work/.speedtest
push_bytes=$((100 * 1024 * 1024))
head -c "$push_bytes" /dev/urandom >"$speed_file"
t0=$(date +%s%N)
VI_OUT="$VI_OUT" VI_RETURN_PORT="$VI_RETURN_PORT" "$jobs_dir/worker-return.sh" push "$speed_file" vi-out
t1=$(date +%s%N)
rm -f "$speed_file"
ms=$(((t1 - t0) / 1000000))
[ "$ms" -gt 0 ] || ms=1
if [ -n "${VI_FAKE_PUSH_SECONDS:-}" ]; then
  ms=$(python3 -c 'import sys; print(max(1, int(float(sys.argv[1]) * 1000)))' "$VI_FAKE_PUSH_SECONDS") ||
    fail "VI_FAKE_PUSH_SECONDS is not a number"
fi
push_mb_s=$(rate_mb_s "$push_bytes" "$ms")
push_seconds=$(python3 -c 'import sys; print(round(int(sys.argv[1]) / 1000, 3))' "$ms")
if [ -n "$min_push" ]; then
  at_least "$push_mb_s" "$min_push" "worker push (MB/s)"
fi

# Written only after every gate passes, so a present report means "checked".
python3 - "$report" \
  "ok=true" "vram_gb=$vram_gb" "free_disk_gb=$free_disk_gb" \
  "cpu_cores=$cpu_cores" "cpu_cores_effective=$cpu_eff" "ram_gb=$ram_gb" \
  "driver_version=$driver_version" "cuda_version=$cuda_version" "nvidia=$nvidia_facts" \
  "disk_write_mb_s=$disk_write_mb_s" "disk_read_mb_s=$disk_read_mb_s" "disk_direct=$disk_direct" \
  "push_bytes=$push_bytes" "push_seconds=$push_seconds" "push_mb_s=$push_mb_s" \
  "pids_max=$vi_lim_pids_max" "pids_current=$vi_lim_pids_current" "tasks_free=$tasks_free" \
  "ulimit_u=$vi_lim_ulimit_u" "threads_max=$vi_lim_threads_max" \
  "limit_threads=$vi_lim_threads" "limit_nj=$vi_lim_nj" \
  "checked_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" <<'PY'
import json, sys
d = {}
for item in sys.argv[2:]:
    key, _, value = item.partition("=")
    try:
        d[key] = json.loads(value)
    except ValueError:
        d[key] = value
json.dump(d, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
echo "worker-preflight: ok (VRAM ${vram_gb:-?} GiB, disk ${disk_write_mb_s:-?}/${disk_read_mb_s:-?} MB/s, push $push_mb_s MB/s, $cpu_eff cores, $ram_gb GB RAM)" >&2
