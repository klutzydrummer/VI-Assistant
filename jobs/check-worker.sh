#!/usr/bin/env bash
# Checks the whole round trip a GPU stage makes, without running one:
#
#   jobs/check-worker.sh
#
# A random token goes up as a stage input through jobs/on-worker.sh, exactly as
# a real stage's inputs do. A tiny job on the worker copies it into its output
# and its checkpoint and records what the worker has (GPU, disk, Python, torch).
# The check passes only if both copies come back home matching what went up.
# So it proves home can reach the worker, send files, run code there, and get
# results and checkpoints back. Nothing on the worker connects to home; every
# connection is home's ssh, as for real stages.
#
# The worker comes from `vi cloud env` (run `vi cloud up` first), or
# VI_WORKER_HOST=local with VI_WORKER_ROOT for tests. Prints what the worker
# reported and exits 0, or says which step failed and exits 1.
set -euo pipefail

jobs_dir=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
token=$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')
mkdir -p "$tmp/ping" "$tmp/out" "$tmp/checkpoint"
printf '%s\n' "$token" >"$tmp/ping/token"
printf '{}\n' >"$tmp/settings.json"

job='set -e
cp "$VI_INPUT_PING/token" "$VI_OUT/token"
cp "$VI_INPUT_PING/token" "$VI_CHECKPOINT/token"
{
  echo "host: $(hostname)"
  gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  echo "gpu: ${gpu:-none}"
  echo "disk free: $(df -h . | awk "NR==2 {print \$4}")"
  echo "gcc: $(command -v gcc >/dev/null && echo yes || echo "missing (Triton jobs such as textscore will fail; the boot script may still be installing it)")"
  python3 -c "import sys; print(\"python:\", sys.version.split()[0])" 2>/dev/null || echo "python: missing"
  python3 -c "import torch; print(\"torch:\", torch.__version__, \"cuda\" if torch.cuda.is_available() else \"no cuda\")" 2>/dev/null || echo "torch: missing"
} >"$VI_OUT/worker.txt"'

start=$(date +%s)
if ! VI_STAGE=check VI_OUT="$tmp/out" VI_SETTINGS="$tmp/settings.json" \
  VI_INPUT_PING="$tmp/ping" VI_CHECKPOINT="$tmp/checkpoint" \
  "$jobs_dir/on-worker.sh" bash -c "$job"; then
  echo "check-worker: FAILED: the job didn't run on the worker (see on-worker.sh's message above)" >&2
  exit 1
fi
took=$(($(date +%s) - start))

fail() {
  echo "check-worker: FAILED: $1" >&2
  exit 1
}
[ -f "$tmp/out/token" ] || fail "the job ran but its output didn't come home"
[ "$(cat "$tmp/out/token")" = "$token" ] || fail "the output that came home doesn't match what was sent"
[ -f "$tmp/checkpoint/token" ] || fail "the output came home but the checkpoint didn't"
[ "$(cat "$tmp/checkpoint/token")" = "$token" ] || fail "the checkpoint that came home doesn't match what was sent"

sed 's/^/  /' "$tmp/out/worker.txt"
echo "check-worker: OK in ${took}s: sent a token up, ran a job on the worker, and got it back in both the output and the checkpoint"
