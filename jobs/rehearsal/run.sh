#!/usr/bin/env bash
# Local rehearsal of the OmniVoice fine-tune worker flow: no GPU, no rental, no network.
#
# Drives the real home and worker scripts (on-worker.sh with VI_WORKER_HOST=local,
# worker-preflight.sh, worker-return.sh, worker-job.sh, voicetune-fresh.sh, checkpoints.py,
# omnivoice_abort.py, fetch_base_model.py) end to end. jobs/rehearsal/stubs.py supplies what
# needs a GPU, the network or real data: a fake nvidia-smi, offline stub wheels for the real
# `pip install -r`, a stub Base download, stub trainer/replay/listening scripts that write the
# real checkpoint layout, an rsync fault shim and a synthetic data folder (sine tones and
# numbered sentences; never game audio or text).
#
#   jobs/rehearsal/run.sh [SCENARIO...]
#
# With no arguments every scenario runs in order. Scenarios:
#   gate-home      home disk gate refuses before anything is sent
#   gate-vram      a 16 GB card fails the worker preflight VRAM gate
#   gate-push      a 2.37 MB/s push fails the 2.5 MB/s floor
#   receiver-down  the return path never proves; nothing is sent
#   receiver       the restricted receiver itself: token push, munged symlinks, write-only,
#                  no delete
#   base-fail      the real venv install runs, then the Base download fails
#   disk-full      the worker disk gate refuses
#   preflight      the full preflight stage (tokenise, replay, probe, push, speak, kill/resume)
#   train          the full 5000-step train stage with a trainer crash, a partial checkpoint
#                  push, the receiver killed, the trainer SIGKILLed, the home runner killed and
#                  rerun (reattach), then the final pushes, prune and listening sets
#   enospc         a "No space left on device" trainer failure is not retried
#   nan            a NaN loss aborts the run and is not retried
#
# Everything lives under ${VI_REHEARSAL_DIR:-~/.cache/vi-rehearsal}/run.XXXXXX (not /tmp, which
# is small here). Listeners bind 127.0.0.1 only; each scenario checks that no receiver it
# started is still running. Exit status 0 only when every check passed.
set -uo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
stubs="$repo/jobs/rehearsal/stubs.py"
base=${VI_REHEARSAL_DIR:-$HOME/.cache/vi-rehearsal}
mkdir -p "$base"
R=$(mktemp -d "$base/run.XXXXXX")
echo "rehearsal: working in $R"

all=(gate-home gate-vram gate-push receiver-down receiver base-fail disk-full preflight train enospc nan)
[ $# -gt 0 ] && scenarios=("$@") || scenarios=("${all[@]}")

avail_mb=$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)
echo "rehearsal: $avail_mb MB RAM available"
[ "$avail_mb" -ge 2048 ] || { echo "rehearsal: under 2 GB RAM free; not starting" >&2; exit 1; }

pass=0
fail=0
ok() { pass=$((pass + 1)); echo "PASS [$cur] $*"; }
bad() { fail=$((fail + 1)); echo "FAIL [$cur] $*"; }
check() {  # DESCRIPTION COMMAND...
  local d=$1; shift
  if "$@"; then ok "$d"; else bad "$d"; fi
}
has() { grep -qF -- "$2" "$1"; }   # FILE TEXT
hasnt() { ! grep -qF -- "$2" "$1"; }
count() { grep -cF -- "$2" "$1" || true; }

real_rsync=$(command -v rsync)
python3 "$stubs" wheelhouse "$R/wheels" >/dev/null
python3 "$stubs" fakebin "$R/fakebin" >/dev/null
python3 "$stubs" data "$R/data" >/dev/null
printf 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n' >"$R/memory.events"
mkdir -p "$R/home" "$R/settings"

# Stage settings straight from the stage file the real runner reads, so the gates rehearsed are
# the committed ones. Extra KEY=VALUE pairs override a copy (numbers stay numbers).
settings() {  # STAGE OUT [KEY=VALUE...]
  python3 - "$repo/characters/tali/omnivoice-ft.toml" "$@" <<'PY'
import json, sys, tomllib
toml, stage, out, *kv = sys.argv[1:]
s = dict(tomllib.load(open(toml, "rb"))["stage"][stage].get("settings", {}))
for item in kv:
    k, v = item.split("=", 1)
    s[k] = json.loads(v)
json.dump(s, open(out, "w"))
PY
}

# run_stage NAME STAGE MODE [ENV=VALUE...]: one `vi dataset run` of a GPU stage, the way
# crates/vi runs it, against the local worker. The log is $R/s/NAME/log; the exit status is
# returned. Set BG=1 to start it in the background (its pid in $stage_pid).
run_stage() {
  local name=$1 stage=$2 mode=$3; shift 3
  local sd="$R/s/$name"
  mkdir -p "$sd/out" "$R/faults/$name"
  [ -s "$sd/settings.json" ] || settings "$stage" "$sd/settings.json"
  local -a envs=(
    -u XDG_STATE_HOME -u XDG_CACHE_HOME -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN
    -u HUGGINGFACE_HUB_TOKEN -u VI_BASE_MODEL -u OVFT_PYTHON -u VI_RETURN_PORT -u VI_WORKER
    HOME="$R/home" VI_STAGE="$stage" VI_OUT="$sd/out" VI_SETTINGS="$sd/settings.json"
    VI_CHECKPOINT="$sd/.checkpoint-$name" VI_INPUT_OMNIVOICE_DATA="$R/data"
    VI_STATE_DIR="$R/state/$name" VI_WORKER_HOST=local VI_WORKER_ROOT="$R/worker/$name"
    VI_POLL_SECONDS=1 VI_BIN=false VI_MEMORY_EVENTS="$R/memory.events"
    PATH="$R/fakebin:$PATH" VI_REAL_RSYNC="$real_rsync" VI_REHEARSAL_FAULTS="$R/faults/$name"
    PIP_NO_INDEX=1 PIP_FIND_LINKS="$R/wheels" PIP_CONFIG_FILE=/dev/null
    PIP_DISABLE_PIP_VERSION_CHECK=1 VI_FAKE_PUSH_SECONDS=22 VI_STUB_TRACE="$sd/trace.jsonl"
    "$@"
  )
  if [ "${BG:-}" = 1 ]; then
    (cd "$repo/characters/tali" && exec env "${envs[@]}" nice -n 19 ionice -c2 -n7 \
      ../../jobs/on-worker.sh bash jobs/voicetune-fresh.sh "$mode") >>"$sd/log" 2>&1 &
    stage_pid=$!
    return 0
  fi
  (cd "$repo/characters/tali" && exec env "${envs[@]}" nice -n 19 ionice -c2 -n7 \
    ../../jobs/on-worker.sh bash jobs/voicetune-fresh.sh "$mode") >>"$sd/log" 2>&1
}

show_log() {  # NAME
  sed "s/^/  [$1] /" "$R/s/$1/log"
}

# No receiver from this rehearsal may outlive its scenario: any process whose command line
# names this run's directory and is an rsync daemon is a leak.
no_leftover_receiver() {
  local pid cmd leaks=0
  for pid in $(pgrep -f -- "--daemon" 2>/dev/null); do
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$cmd" in *rsync*"--daemon"*"$R/"*) echo "  leftover receiver pid $pid: $cmd"; leaks=1 ;; esac
  done
  [ "$leaks" = 0 ]
}

durable_dir() {  # NAME STAGE -> the durable checkpoint dir home kept for that scenario
  find "$R/home/.cache/vi/checkpoints/$2" -maxdepth 1 -type d -name "$1-*" -print -quit 2>/dev/null
}
work_dir() {  # NAME STAGE
  find "$R/worker/$1" -maxdepth 1 -type d -name "$2.*" -print -quit 2>/dev/null
}
last_step() {  # METRICS -> newest micro_step logged, or 0
  tail -n1 "$1" 2>/dev/null | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read())["micro_step"])
except Exception: print(0)'
}
venv_dir() {
  echo "$R/home/.cache/vi/venv-$(sha256sum "$repo/jobs/requirements-omnivoice-worker.txt" | cut -c1-16)"
}

sc_gate_home() {
  settings omnivoice_preflight "$R/s/gate-home/settings.json" min_home_gb=100000
  run_stage gate-home omnivoice_preflight preflight; local rc=$?
  show_log gate-home
  check "home gate refuses (exit $rc)" test "$rc" -ne 0
  check "nothing reached the worker" test -z "$(work_dir gate-home omnivoice_preflight)"
}

sc_gate_vram() {
  run_stage gate-vram omnivoice_preflight preflight VI_STUB_VRAM_MIB=16000; local rc=$?
  show_log gate-vram
  local L="$R/s/gate-vram/log"
  check "a 16 GB card fails (exit $rc)" test "$rc" -ne 0
  check "the VRAM gate names its floor" has "$L" "under the required 23.5"
  check "preflight failure stops before inputs" has "$L" "the worker preflight failed; nothing was uploaded"
  check "no data folder was sent" test -z "$(find "$R/worker/gate-vram" -path '*/inputs/*' -print -quit)"
}

sc_gate_push() {
  run_stage gate-push omnivoice_preflight preflight VI_FAKE_PUSH_SECONDS=44.2; local rc=$?
  show_log gate-push
  local L="$R/s/gate-push/log"
  check "a 2.37 MB/s worker fails (exit $rc)" test "$rc" -ne 0
  check "the push gate names the 2.5 floor" has "$L" "worker push (MB/s) is 2.37, under the required 2.5"
  check "the 24467 MiB card passed the VRAM gate" hasnt "$L" "VRAM (GiB) is"
}

sc_receiver_down() {
  mkdir -p "$R/faults/receiver-down"
  : >"$R/faults/receiver-down/receiver-down"
  run_stage receiver-down omnivoice_preflight preflight VI_PROVE_WAIT=3; local rc=$?
  show_log receiver-down
  local L="$R/s/receiver-down/log"
  check "an unreachable receiver fails the run (exit $rc)" test "$rc" -ne 0
  check "the return path is proved before anything is sent" has "$L" "the return path did not come up; nothing will be sent"
  check "worker preflight never ran" hasnt "$L" "worker-preflight"
  check "no data folder was sent" test -z "$(find "$R/worker/receiver-down" -path '*/inputs/*' -print -quit)"
}

sc_receiver() {
  local d="$R/s/receiver" port st=$R/s/receiver/state
  mkdir -p "$d/src" "$d/home-out" "$d/home-ck" "$d/home-pub"
  port=$(bash "$repo/jobs/worker-return.sh" start "$d/home-out" "$d/home-ck" "$st" "$d/home-pub")
  local pid; pid=$(cat "$st/rsyncd.pid")
  check "receiver listens on loopback only" bash -c "ss -ltnp 2>/dev/null | grep -F ':$port ' | grep -qF '127.0.0.1:$port'"
  check "receiver does not listen on any other address" bash -c "! ss -ltn 2>/dev/null | grep -F ':$port ' | grep -vqF '127.0.0.1:$port'"
  head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' >"$d/token"
  check "a random token push succeeds" env VI_RETURN_PORT="$port" VI_PUSH_WAIT=5 \
    bash "$repo/jobs/worker-return.sh" push "$d/token" vi-out
  check "the token arrived home intact" cmp -s "$d/token" "$d/home-out/token"
  ln -s /etc/hostname "$d/src/escape"
  echo synthetic >"$d/src/file.txt"
  env VI_RETURN_PORT="$port" VI_PUSH_WAIT=5 bash "$repo/jobs/worker-return.sh" push "$d/src" vi-out >/dev/null 2>&1
  local target; target=$(readlink "$d/home-out/escape" 2>/dev/null || true)
  check "a pushed symlink is munged ($target)" bash -c "[[ '$target' == /rsyncd-munged/* ]]"
  check "the module cannot be listed or read back" bash -c "! '$real_rsync' rsync://127.0.0.1:$port/vi-out/ >/dev/null 2>&1"
  check "the module cannot be downloaded from" bash -c "! '$real_rsync' rsync://127.0.0.1:$port/vi-out/file.txt '$d/readback' >/dev/null 2>&1"
  rm -f "$d/src/file.txt"
  check "a --delete push is refused" bash -c "! '$real_rsync' -r --delete '$d/src/' rsync://127.0.0.1:$port/vi-out/ >/dev/null 2>&1"
  check "home kept the file a --delete would remove" test -s "$d/home-out/file.txt"
  bash "$repo/jobs/worker-return.sh" stop "$st"
  check "the receiver pid $pid is gone after stop" bash -c "! kill -0 $pid 2>/dev/null"
}

sc_base_fail() {
  local venv; venv=$(venv_dir)
  check "no venv before the run" test ! -e "$venv"
  run_stage base-fail omnivoice_preflight preflight VI_STUB_HF_FAIL=1; local rc=$?
  show_log base-fail
  local L="$R/s/base-fail/log"
  check "a failed Base download fails the stage (exit $rc)" test "$rc" -ne 0
  check "it says so" has "$L" "could not fetch the Base model"
  check "the real venv install built the venv" test -x "$venv/bin/python"
  check "pip installed every requirement offline" bash -c "'$venv/bin/python' -m pip list 2>/dev/null | grep -cE '^(accelerate|omnivoice|tensorboard|webdataset|soundfile|huggingface.hub) ' | grep -qx 6"
  check "no tokenise ran without a Base model" hasnt "$L" "preflight 0a"
  check "the worker copy is kept for diagnosis" has "$L" "keeping the worker copy"
}

sc_disk_full() {
  run_stage disk-full omnivoice_preflight preflight VI_MIN_DISK_GB=100000; local rc=$?
  show_log disk-full
  local L="$R/s/disk-full/log"
  check "the disk gate fails the stage (exit $rc)" test "$rc" -ne 0
  check "it names the free and required space" has "$L" "check: worker disk"
  check "no tokenise ran on a full disk" hasnt "$L" "preflight 0a"
}

sc_preflight() {
  run_stage preflight omnivoice_preflight preflight; local rc=$?
  show_log preflight
  local L="$R/s/preflight/log" dur
  dur=$(durable_dir preflight omnivoice_preflight)
  check "preflight stage exit 0 (got $rc)" test "$rc" -eq 0
  check "worker preflight gates passed" has "$L" "worker-preflight: ok"
  check "the Base model was fetched (stub download)" has "$L" "Base model at"
  check "step 0a tokenise ran" has "$L" "preflight 0a: tokenise train and dev clips"
  check "step 0b replay ran" has "$L" "preflight 0b: generate and tokenise the replay set"
  check "the probe pushed and timed" has "$L" "probe push"
  check "checkpoint 5 was verified home and acked" has "$L" "checkpoint 5 verified home and acked"
  check "the kill landed mid-run" has "$L" "rehearsal: SIGKILL at micro-step"
  check "resume from checkpoint-10 verified checkpoint-20" has "$L" "rehearsal: resumed from checkpoint-10 and verified checkpoint-20"
  check "preflight: ok" has "$L" "preflight: ok"
  check "durable checkpoint-5 verifies home" python3 "$repo/jobs/checkpoints.py" --format omnivoice verify "$dur" --expect 1
  check "the probe listening set is labelled synthetic" has "$R/s/preflight/out/probe-speak/README.txt" "SYNTHETIC"
  check "the trace shows the rehearsal resumed from checkpoint-10" has "$R/s/preflight/trace.jsonl" "checkpoint-10"
}

# The train run is ~5000 stub steps at 0.04 s. Faults, in order:
#   step 1200          the trainer crashes once; worker-job.sh retry 1 resumes from checkpoint-1000
#   checkpoint-2000    its first two vi-publish pushes stop part way (no manifest, no weights)
#   step >= 1700       the home receiver is killed; the home loop must restart and re-prove it
#   step >= 2700       the trainer is SIGKILLed; retry 2 must resume from checkpoint-2500
#   step >= 3600       the home runner is killed (TERM), then rerun: it must reattach, send no
#                      inputs, and finish the run
sc_train() {
  local name=train L="$R/s/train/log" work metrics rs pid cmd t0
  mkdir -p "$R/faults/$name"
  echo "checkpoint-2000 2" >"$R/faults/$name/publish-partial"
  BG=1 run_stage "$name" omnivoice_train train VI_STUB_STEP_SECONDS=0.04 VI_STUB_CRASH_AT_STEP=1200
  local home_pid=$stage_pid killed_rx= killed_tr= killed_home=
  t0=$(date +%s)
  while kill -0 "$home_pid" 2>/dev/null; do
    sleep 0.5
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { bad "train phase 1 timed out"; kill "$home_pid"; break; }
    work=$(work_dir "$name" omnivoice_train)
    [ -n "$work" ] || continue
    metrics="$work/voicetune-fresh/run/style_metrics.jsonl"
    local step; step=$(last_step "$metrics")
    if [ -z "$killed_rx" ] && [ "$step" -ge 1700 ]; then
      rs=$(find "$R/state/$name/return" -name rsyncd.pid -print -quit 2>/dev/null)
      pid=$(cat "$rs" 2>/dev/null || true)
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
      if [[ "$cmd" == *--daemon*"$R/"* ]]; then
        kill "$pid" && killed_rx=$step && echo "rehearsal: killed home receiver pid $pid at step $step"
      fi
    fi
    if [ -z "$killed_tr" ] && [ "$step" -ge 2700 ] \
       && [ -s "$work/voicetune-fresh/run/checkpoint-2500/SHA256SUMS" ]; then
      pid=$(cat "$work/train.pid" 2>/dev/null || true)
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
      if [[ "$cmd" == *train_ft.py*"$R/"* ]]; then
        kill -9 "$pid" && killed_tr=$step && echo "rehearsal: SIGKILLed trainer pid $pid at step $step"
      fi
    fi
    if [ -z "$killed_home" ] && [ "$step" -ge 3600 ]; then
      cmd=$(tr '\0' ' ' <"/proc/$home_pid/cmdline" 2>/dev/null || true)
      if [[ "$cmd" == *on-worker.sh* ]]; then
        kill -TERM "$home_pid" && killed_home=$step && echo "rehearsal: TERMed home runner pid $home_pid at step $step"
      fi
    fi
  done
  wait "$home_pid"; local rc1=$?
  echo "rehearsal: first home run exited $rc1"
  check "the receiver was killed mid-run (step ${killed_rx:-never})" test -n "$killed_rx"
  check "the trainer was SIGKILLed after checkpoint-2500 (step ${killed_tr:-never})" test -n "$killed_tr"
  check "the home runner was killed mid-run (step ${killed_home:-never})" test -n "$killed_home"
  check "the killed home runner exited nonzero ($rc1)" test "$rc1" -ne 0
  check "no receiver outlived the killed home runner" no_leftover_receiver
  echo "rehearsal: rerunning the home side"
  echo "----- rerun -----" >>"$L"
  run_stage "$name" omnivoice_train train VI_STUB_STEP_SECONDS=0.04 VI_STUB_CRASH_AT_STEP=1200; local rc=$?
  show_log "$name"
  local dur trace="$R/s/$name/trace.jsonl"
  dur=$(durable_dir "$name" omnivoice_train)
  check "the rerun finished the stage (exit $rc)" test "$rc" -eq 0
  check "the rerun reattached and sent no inputs" has "$L" "no inputs will be sent"
  check "the injected crash happened" has "$L" "injected trainer crash at step 1200"
  check "worker-job.sh retried the crash" has "$L" "worker-job.sh: retry 1/2"
  check "worker-job.sh retried the SIGKILL" has "$L" "worker-job.sh: retry 2/2"
  check "retry 1 resumed from checkpoint-1000" has "$trace" "checkpoint-1000\""
  check "retry 2 resumed from checkpoint-2500 (the newest)" has "$trace" "checkpoint-2500\""
  check "the trainer started exactly three times" test "$(wc -l <"$trace")" -eq 3
  check "the partial checkpoint-2000 pushes fired twice" test "$(count "$R/faults/$name/fired.log" "publish-partial checkpoint-2000")" -eq 2
  check "the failed pushes were retried" has "$L" "publish push failed; will retry"
  check "checkpoint 2000 was verified home after the partial pushes" has "$L" "checkpoint 2000 verified home and acked"
  check "no checkpoint failed verification" hasnt "$L" "ALERT"
  check "the home loop restarted the killed receiver" has "$L" "the return receiver is down; restarting it"
  check "the restarted receiver was re-proved" has "$L" "re-proved"
  check "home kept exactly 1000, 2500, 4500, 5000" test "$(cd "$dur" 2>/dev/null && ls -d checkpoint-* | sort -t- -k2n | tr '\n' ' ')" = "checkpoint-1000 checkpoint-2500 checkpoint-4500 checkpoint-5000 "
  check "the durable checkpoints verify (expect 4)" python3 "$repo/jobs/checkpoints.py" --format omnivoice verify "$dur" --expect 4
  local step
  for step in 1000 2500 5000; do
    check "listening set step-$step is labelled synthetic" has "$R/s/$name/out/platter/step-$step/README.txt" "SYNTHETIC"
  done
  check "the train stage said done" has "$L" "train: done"
}

sc_enospc() {
  run_stage enospc omnivoice_train train VI_TRAIN_STEPS=400 VI_SAVE_STEPS=100 VI_STUB_ENOSPC_AT_STEP=150; local rc=$?
  show_log enospc
  local L="$R/s/enospc/log"
  check "a full worker disk fails the stage (exit $rc)" test "$rc" -ne 0
  check "worker-job.sh did not retry ENOSPC" has "$L" "ENOSPC; not retrying"
  check "the checkpoint before the failure still came home" has "$L" "checkpoint 100 verified home and acked"
}

sc_nan() {
  run_stage nan omnivoice_train train VI_TRAIN_STEPS=400 VI_SAVE_STEPS=100 VI_STUB_STEP_SECONDS=0.1 VI_STUB_NAN_AT_STEP=150; local rc=$?
  show_log nan
  local L="$R/s/nan/log"
  check "a NaN loss fails the stage (exit $rc)" test "$rc" -ne 0
  check "the abort rule fired" has "$L" "train abort:"
  check "worker-job.sh did not retry the abort" has "$L" "abort.json exists; not retrying"
}

for cur in "${scenarios[@]}"; do
  fn="sc_${cur//-/_}"
  declare -F "$fn" >/dev/null || { echo "rehearsal: unknown scenario $cur" >&2; exit 2; }
  echo "===== $cur ====="
  mkdir -p "$R/s/$cur"
  "$fn"
  check "no receiver left running" no_leftover_receiver
done
cur=all
echo "rehearsal: $pass passed, $fail failed ($R)"
[ "$fail" -eq 0 ]
