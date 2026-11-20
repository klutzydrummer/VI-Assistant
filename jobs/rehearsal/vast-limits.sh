#!/usr/bin/env bash
# Home rehearsal of the Vast container limits that broke the r7 preflight. Opt-in, one command.
#
#   jobs/rehearsal/vast-limits.sh [--expect red|green] [options]
#
# Why: r7 died on a rented box that showed 48 CPUs but allowed only a few thousand tasks
# (processes plus threads). Every Python process started a 48-thread OpenBLAS pool, a pool thread
# could not be created, OpenBLAS raised SIGINT, the process pool broke, and the stage then hung
# forever behind `| tee`. The stub rehearsal (run.sh) never loads real numpy/scipy, and the home
# real-path check runs on 4 CPUs with a huge task limit, so neither could see it.
#
# What it does: runs the REAL `tokenise` function from jobs/voicetune-fresh.sh (its real tee
# pipeline, its real count guard) on N_CLIPS real home clips with the real upstream
# omnivoice.scripts.extract_audio_tokens, on CPU, inside
# `systemd-run --user --scope -p TasksMax=N` (a cgroup pids.max, like a container's pids limit),
# with an LD_PRELOAD shim (jobs/rehearsal/fakenproc.c) that makes the process see FAKE_NPROC CPUs.
# voicetune-fresh.sh is not edited: its three tokenise functions are cut out of the file by name
# and run by this script's stage mode, so a later fix to them is rehearsed automatically.
#
# Verdicts:
#   --expect red    passes only if the r7 failure shows: `pthread_create failed ... of <CPUS>`
#                   together with KeyboardInterrupt or BrokenProcessPool, or the stage hangs until
#                   BUDGET kills it (exit 124/137). Use it to prove the gate catches the bug on a
#                   base without the fixes.
#   --expect green  (default) passes only if every check holds: exit 0, exactly one
#                   "Processing Complete" line with Successful == N_CLIPS, cgroup pids.events max
#                   0, no fatal line, no process left in the scope after the stage returns.
# Both need: no process left after teardown. Exit status 0 only when the verdict matches.
#
# Options (flag, or the environment variable in brackets; the flag wins):
#   --fake-nproc N   [FAKE_NPROC=48]   CPUs the shim reports
#   --tasks-max N    [TASKS_MAX]       scope TasksMax. Default 2800 for red (the narrow band just
#                                      under the ~2840-task peak where the exact r7 hang shows;
#                                      anything below ~2840 fails), 512 for green (a small sandbox)
#   --cpu-quota Q    [CPU_QUOTA=1152%] scope CPUQuota (r7's 11.52 cores; no effect on 4 CPUs,
#                                      kept so the limits line matches the r7 box)
#   --clips N        [N_CLIPS=8]       clips tokenised (the first N rows of MANIFEST, file order)
#   --budget S       [BUDGET=300]      seconds before the stage is killed (red hang ends here)
#   --venv DIR       [VENV]            home CPU venv (default ~/.cache/vi/home-cpu-venv-3750779e6152a9c4)
#   --manifest FILE  [MANIFEST]        8-row home manifest (default the realpath-check scratch one)
#   --threads N      [THREADS]         export OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=N (rehearses a fix)
#   --nj N           [NJ]              add --nj_per_gpu N to the upstream call, via a python wrapper
#   Test hooks: VL_MEM_AVAIL_MB and VL_DISK_FREE_GB override the measured RAM and disk.
#
# Safety: CPU only (CUDA_VISIBLE_DEVICES=""), nice -n 19, ionice -c2 -n7, refuses under 4 GB free
# RAM or under 20 GB free /home. Clips are read in place; derived token shards go to a scratch
# folder under ~/.cache/vi-rehearsal that is deleted at the end. Nothing leaves this machine, no
# mood labels are read, and the log kept in ~/.cache/vi-rehearsal/vast-limits-logs has lines
# that carry clip text removed. Only the scope this run started is stopped.
set -uo pipefail

self=$(readlink -f "${BASH_SOURCE[0]}")
repo=$(cd "$(dirname "$self")/../.." && pwd)
base=${VI_REHEARSAL_DIR:-$HOME/.cache/vi-rehearsal}

# ---------- verdict (pure function over facts; also reachable as --judge for the tests) ----------
# judge EXPECT LOG RC EVENTS ORPHANS_STAGE ORPHANS_POST CLIPS FAKE_NPROC
# Prints one "key=value" facts line to stdout and the reasons to stderr; returns 0 on a match.
judge() {
  local expect=$1 log=$2 rc=$3 events=$4 o_stage=$5 o_post=$6 clips=$7 fake=$8
  local n_complete succ fatal sig=no hang=no reasons=() ok=1
  n_complete=$(grep -c 'Processing Complete' "$log" 2>/dev/null)
  succ=$(grep -o 'Processing Complete - Successful: [0-9]*' "$log" 2>/dev/null | tail -n 1 | grep -o '[0-9]*$')
  succ=${succ:-0}
  fatal=$(grep -Ec "pthread_create failed|BrokenProcessPool|can't start new thread|Traceback" "$log" 2>/dev/null)
  case $rc in 124|137) hang=yes ;; esac
  if grep -Eq "pthread_create failed for thread [0-9]+ of $fake" "$log" 2>/dev/null &&
     grep -Eq 'KeyboardInterrupt|BrokenProcessPool' "$log" 2>/dev/null; then
    sig=yes
  fi
  echo "complete=$succ/$clips lines=$n_complete fatal_lines=$fatal r7_signature=$sig budget_hang=$hang"
  if [ "$o_post" != 0 ]; then ok=0; reasons+=("$o_post process(es) still alive after teardown"); fi
  if [ "$expect" = red ]; then
    if [ "$sig" != yes ] && [ "$hang" != yes ]; then
      ok=0; reasons+=("no r7 signature (pthread_create of $fake + KeyboardInterrupt/BrokenProcessPool) and no budget hang: the base did NOT fail the way r7 did")
    fi
  else
    [ "$rc" = 0 ] || { ok=0; reasons+=("stage exit code $rc, wanted 0"); }
    [ "$n_complete" = 1 ] || { ok=0; reasons+=("$n_complete 'Processing Complete' lines, wanted 1"); }
    [ "$succ" = "$clips" ] || { ok=0; reasons+=("Successful: $succ, wanted $clips"); }
    [ "$events" = 0 ] || { ok=0; reasons+=("pids.events max = $events, wanted 0"); }
    [ "$fatal" = 0 ] || { ok=0; reasons+=("$fatal fatal log line(s)"); }
    [ "$o_stage" = 0 ] || { ok=0; reasons+=("$o_stage process(es) left in the scope after the stage returned"); }
  fi
  if [ "$ok" = 1 ]; then return 0; fi
  printf '  %s\n' "${reasons[@]}" >&2
  return 1
}

die() { echo "vast-limits: $*" >&2; exit 2; }

# ---------- argument parsing ----------
expect=green
mode=main
fake=${FAKE_NPROC:-48}
tasks_max=${TASKS_MAX:-}
cpu_quota=${CPU_QUOTA:-1152%}
clips=${N_CLIPS:-8}
budget=${BUDGET:-300}
venv=${VENV:-$HOME/.cache/vi/home-cpu-venv-3750779e6152a9c4}
manifest=${MANIFEST:-$HOME/.cache/vi/realpath-check/scratch/train_manifest.jsonl}
threads=${THREADS:-}
nj=${NJ:-}
j_log='' j_rc='' j_events='' j_o_stage=0 j_o_post=0

need_value() { [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"; }
while [ $# -gt 0 ]; do
  case $1 in
    --expect) need_value "$@"; expect=$2; shift 2 ;;
    --fake-nproc) need_value "$@"; fake=$2; shift 2 ;;
    --tasks-max) need_value "$@"; tasks_max=$2; shift 2 ;;
    --cpu-quota) need_value "$@"; cpu_quota=$2; shift 2 ;;
    --clips) need_value "$@"; clips=$2; shift 2 ;;
    --budget) need_value "$@"; budget=$2; shift 2 ;;
    --venv) need_value "$@"; venv=$2; shift 2 ;;
    --manifest) need_value "$@"; manifest=$2; shift 2 ;;
    --threads) need_value "$@"; threads=$2; shift 2 ;;
    --nj) need_value "$@"; nj=$2; shift 2 ;;
    --judge) mode=judge; shift ;;
    --log) need_value "$@"; j_log=$2; shift 2 ;;
    --rc) need_value "$@"; j_rc=$2; shift 2 ;;
    --events) need_value "$@"; j_events=$2; shift 2 ;;
    --orphans-stage) need_value "$@"; j_o_stage=$2; shift 2 ;;
    --orphans-post) need_value "$@"; j_o_post=$2; shift 2 ;;
    --inner-scope) mode=inner-scope; shift ;;
    --inner-stage) mode=inner-stage; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "$self" | sed '$d;s/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

case $expect in red|green) ;; *) die "--expect must be red or green, not '$expect'" ;; esac
is_pos_int() { case $1 in ''|*[!0-9]*|0) return 1 ;; *) return 0 ;; esac; }
for pair in "fake-nproc:$fake" "clips:$clips" "budget:$budget"; do
  is_pos_int "${pair#*:}" || die "--${pair%%:*} must be a positive whole number, not '${pair#*:}'"
done
[ -z "$tasks_max" ] && { [ "$expect" = red ] && tasks_max=2800 || tasks_max=512; }
is_pos_int "$tasks_max" || die "--tasks-max must be a positive whole number, not '$tasks_max'"
[ -z "$threads" ] || is_pos_int "$threads" || die "--threads must be a positive whole number"
[ -z "$nj" ] || is_pos_int "$nj" || die "--nj must be a positive whole number"

# ---------- judge-only mode (canned logs) ----------
if [ "$mode" = judge ]; then
  [ -f "$j_log" ] || die "--judge needs --log FILE"
  [ -n "$j_rc" ] || die "--judge needs --rc N"
  [ -n "$j_events" ] || j_events=0
  facts=$(judge "$expect" "$j_log" "$j_rc" "$j_events" "$j_o_stage" "$j_o_post" "$clips" "$fake")
  jrc=$?
  echo "$facts"
  [ "$jrc" = 0 ] && echo "verdict: PASS (expect $expect)" || echo "verdict: FAIL (expect $expect)"
  exit "$jrc"
fi

# ---------- stage mode: runs inside the scope, under nice/ionice/timeout ----------
if [ "$mode" = inner-stage ]; then
  : "${VL_R:?}" "${VL_SHIM:?}" "${VL_VENV:?}" "${VL_MANIFEST:?}" "${VL_TOKENIZER:?}"
  export LD_PRELOAD=$VL_SHIM FAKE_NPROC=$fake CUDA_VISIBLE_DEVICES=""
  if [ -n "$threads" ]; then
    export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads NUMEXPR_NUM_THREADS=$threads
  fi
  # shellcheck disable=SC2034  # PY, data_dir and friends are read by the eval'd tokenise functions
  PY="$VL_VENV/bin/python"
  if [ -n "$nj" ]; then
    # tokenise passes a fixed flag list, so --nj_per_gpu is appended by this wrapper, only here.
    printf '#!/usr/bin/env bash\nexec %q "$@" --nj_per_gpu %q\n' "$VL_VENV/bin/python" "$nj" >"$VL_R/py-wrapper.sh"
    chmod +x "$VL_R/py-wrapper.sh"
    PY="$VL_R/py-wrapper.sh"
  fi
  data_dir="$VL_R/data"
  mkdir -p "$data_dir"
  export VI_AUDIO_TOKENIZER=$VL_TOKENIZER
  src="$repo/jobs/voicetune-fresh.sh"
  defs=
  for fn in audio_tokenizer_dir tokenise tokenise_count_guard ${EXTRA_FUNCS:-}; do
    def=$(awk -v n="$fn" '$0 ~ "^" n "\\(\\) \\{" { p = 1 } p { print } p && /^\}/ { exit }' "$src")
    [ -n "$def" ] || { echo "vast-limits: function $fn not found in $src" >&2; exit 2; }
    defs+="$def"$'\n'
  done
  eval "$defs"
  echo "vast-limits: limits fake_nproc=$fake tasks_max=$tasks_max threads=${threads:-unset} nj=${nj:-default}"
  set -euo pipefail
  tokenise rehearsal "$VL_MANIFEST"
  exit $?
fi

# ---------- scope wrapper: runs inside the scope, reads the cgroup before it ends ----------
if [ "$mode" = inner-scope ]; then
  : "${VL_R:?}"
  cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
  SECONDS=0
  nice -n 19 ionice -c2 -n7 timeout -k 5 "$budget" bash "$self" --inner-stage
  rc=$?
  secs=$SECONDS
  # Give a clean stage a few seconds to finish draining, then count what is left (not this shell).
  left=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    mapfile -t procs <"$cg/cgroup.procs"
    left=0
    for p in "${procs[@]}"; do [ "$p" = "$$" ] || left=$((left + 1)); done
    [ "$left" = 0 ] && break
    sleep 0.5
  done
  ev=0
  while read -r k v; do [ "$k" = max ] && ev=$v; done <"$cg/pids.events"
  {
    echo "rc=$rc"
    echo "stage_secs=$secs"
    echo "pids_max=$(cat "$cg/pids.max")"
    echo "pids_peak=$(cat "$cg/pids.peak" 2>/dev/null || echo ?)"
    echo "events_max=$ev"
    echo "orphans_stage=$left"
  } >"$VL_R/stats"
  exit "$rc"
fi

# ---------- main: gates, shim, scope, verdict, teardown ----------
[ -x "$venv/bin/python" ] || die "no CPU venv at $venv (build it from jobs/requirements-omnivoice-home-cpu.txt, or pass --venv)"
[ -f "$manifest" ] || die "manifest $manifest not found (pass --manifest)"
rows=$(grep -c . "$manifest")
[ "$clips" -le "$rows" ] || die "--clips $clips is more than the $rows rows in $manifest"

mem_mb=${VL_MEM_AVAIL_MB:-$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)}
[ "$mem_mb" -ge 4096 ] || die "only $mem_mb MB RAM available, need 4096; not starting"
mkdir -p "$base"
disk_gb=${VL_DISK_FREE_GB:-$(df -BG --output=avail "$base" | tail -n 1 | tr -dc 0-9)}
[ "$disk_gb" -ge 20 ] || die "only $disk_gb GB free under $base, need 20; not starting"

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
if ! [ -S "$XDG_RUNTIME_DIR/bus" ] || ! systemctl --user show -p Version >/dev/null 2>&1; then
  die "no systemd user manager reachable at $XDG_RUNTIME_DIR/bus; systemd-run --user --scope is needed for the TasksMax cap"
fi

tok=${VL_TOKENIZER:-$HOME/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/audio_tokenizer}
[ -d "$tok" ] || die "audio tokenizer snapshot not found in the Hugging Face cache (set VL_TOKENIZER)"

# Shim: build once, cached by source hash, outside the repo.
shim_src="$repo/jobs/rehearsal/fakenproc.c"
shim="$base/fakenproc-$(sha256sum "$shim_src" | cut -c1-12).so"
if [ ! -f "$shim" ]; then
  tmp="$shim.tmp-$$"
  if command -v cc >/dev/null 2>&1; then cc_cmd=(cc)
  elif command -v gcc >/dev/null 2>&1; then cc_cmd=(gcc)
  else cc_cmd=(nix shell nixpkgs#gcc -c gcc); fi
  "${cc_cmd[@]}" -O2 -shared -fPIC -o "$tmp" "$shim_src" -ldl || { rm -f "$tmp"; die "could not build the fake-CPU shim"; }
  mv "$tmp" "$shim"
fi

R=$(mktemp -d "$base/vast-limits.XXXXXX")
unit="vi-vast-limits-$$"
logdir="$base/vast-limits-logs"
mkdir -p "$logdir"
pgid=
teardown() {
  # Stop only the scope this run started, then its process group, then the scratch folder.
  systemctl --user stop "$unit.scope" >/dev/null 2>&1
  if [ -n "$pgid" ]; then
    local left; left=$(ps -o pid= -g "$pgid" 2>/dev/null | grep -c .)
    [ "$left" = 0 ] || kill -KILL -- "-$pgid" 2>/dev/null
  fi
  rm -rf "$R"
}
trap teardown EXIT
trap 'echo "vast-limits: interrupted" >&2; exit 130' INT TERM

if [ "$clips" -lt "$rows" ]; then
  head -n "$clips" "$manifest" >"$R/manifest.jsonl"
  used_manifest="$R/manifest.jsonl"
else
  used_manifest=$manifest
fi

echo "vast-limits: expect=$expect fake_nproc=$fake tasks_max=$tasks_max cpu_quota=$cpu_quota clips=$clips budget=${budget}s threads=${threads:-unset} nj=${nj:-default}"
SECONDS=0
export VL_R=$R VL_SHIM=$shim VL_VENV=$venv VL_MANIFEST=$used_manifest VL_TOKENIZER=$tok
export FAKE_NPROC=$fake TASKS_MAX=$tasks_max CPU_QUOTA=$cpu_quota N_CLIPS=$clips BUDGET=$budget THREADS=$threads NJ=$nj
# The scope runs under setsid so it has its own process group we can check and, if needed, kill.
setsid systemd-run --user --scope --quiet --unit="$unit" \
  -p "TasksMax=$tasks_max" -p "CPUQuota=$cpu_quota" \
  bash "$self" --inner-scope >"$R/stage.out" 2>&1 &
pgid=$!
wait "$pgid"
scope_rc=$?
total_secs=$SECONDS

getstat() { sed -n "s/^$1=//p" "$R/stats" 2>/dev/null | head -n 1; }
if [ ! -f "$R/stats" ]; then
  echo "vast-limits: the scope did not report (systemd-run exit $scope_rc); output tail:" >&2
  tail -n 15 "$R/stage.out" | cut -c1-300 >&2
  exit 3
fi
rc=$(getstat rc); events=$(getstat events_max); o_stage=$(getstat orphans_stage)
peak=$(getstat pids_peak); pmax=$(getstat pids_max); stage_secs=$(getstat stage_secs)

# Teardown now (not only at exit) so the post-teardown check below is real.
systemctl --user stop "$unit.scope" >/dev/null 2>&1
sleep 1
o_post=$(ps -o pid= -g "$pgid" 2>/dev/null | grep -c .)
cg_gone=yes
systemctl --user is-active --quiet "$unit.scope" 2>/dev/null && cg_gone=no
[ "$cg_gone" = yes ] || o_post=$((o_post + 1))

# Keep a log with clip-text lines removed (progress bars are collapsed).
stamp=$(date +%Y%m%dT%H%M%S)
keep="$logdir/vast-limits-$stamp-$expect.log"
tr '\r' '\n' <"$R/stage.out" | grep -v '"text"' | grep -v 'Loading weights\|Extracting Audio Tokens' | cut -c1-400 >"$keep"

facts=$(judge "$expect" "$R/stage.out" "$rc" "$events" "$o_stage" "$o_post" "$clips" "$fake")
jrc=$?
verdict=FAIL; [ "$jrc" = 0 ] && verdict=PASS
echo "vast-limits: $verdict expect=$expect rc=$rc stage_secs=$stage_secs total_secs=$total_secs tasks_max=$tasks_max fake_nproc=$fake pids_max=$pmax pids_peak=$peak events_max=$events orphans_stage=$o_stage orphans_post=$o_post $facts log=$keep"
exit "$jrc"
