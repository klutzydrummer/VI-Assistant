#!/usr/bin/env bash
# Build the voicetune_data stage's out/ folder at home (see jobs/voicetune_data.py).
# Runs the job under the qwen-tts virtualenv, the same way voicetune-fresh.sh's
# tune_python does, so prepare has qwen-tts + torch + soundfile.
#
#   vi dataset run characters/tali/voicetune.toml voicetune_data
#
# At home the pip wheel torch in that venv needs the NixOS C++ runtime; run
# QWEN_TTS_PYTHON="steam-run <venv>/bin/python" when invoking directly (it is
# split into an argv array below, so a multi-word command works). The runner
# sets VI_OUT and VI_CORPUS; see jobs/voicetune_data.py for the args.
set -euo pipefail

jobs_dir=$(cd "$(dirname "$0")" && pwd)
: "${VI_OUT:?voicetune_data runs from \`vi dataset run\`}" "${VI_SETTINGS:?}"

# Refuse to start when home lacks the RAM the CPU prepare peak needs (16 GiB),
# before the venv is built or any corpus file is read.
mem_avail_kb=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
if [ -n "$mem_avail_kb" ] && [ "$mem_avail_kb" -lt $((16 * 1024 * 1024)) ]; then
  echo "voicetune_data: only $((mem_avail_kb / 1024)) MiB RAM available; need at least 16 GiB" >&2
  exit 1
fi

# The stage names a home-disk floor (min_home_gb) and a reserve for the
# checkpoint bytes the GPU stages will publish home (home_gate_reserve_gb), so
# the prepare refuses before it starts when landing them would underrun the
# floor. Both come from the settings JSON the runner exports as VI_SETTINGS,
# read the same way on-worker.sh reads them.
min_home_gb=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("min_home_gb",""))' "$VI_SETTINGS")
home_gate_reserve_gb=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("home_gate_reserve_gb",""))' "$VI_SETTINGS")
if [ -n "$min_home_gb" ]; then
  python3 "$jobs_dir/checkpoints.py" gate "$HOME" --min-gb "$min_home_gb" --reserve-gb "${home_gate_reserve_gb:-0}"
fi

# qwen-tts pins an older transformers, so voicetune runs in its own venv — the
# same one voicetune-fresh.sh builds from requirements-tune.txt. QWEN_TTS_PYTHON
# may be a multi-word command (e.g. "steam-run <venv>/bin/python"), so it is
# read into an argv array rather than exec'd as one word.
if [ -n "${QWEN_TTS_PYTHON:-}" ]; then
  read -r -a py <<<"$QWEN_TTS_PYTHON"
else
  req="$jobs_dir/requirements-tune.txt"
  venv="$HOME/.cache/vi/venv-$(sha256sum "$req" | cut -c1-16)"
  if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv --system-site-packages "$venv"
    "$venv/bin/python" -m pip install -q -r "$req"
  fi
  py=("$venv/bin/python")
fi

exec "${py[@]}" "$jobs_dir/voicetune_data.py" "$@"
