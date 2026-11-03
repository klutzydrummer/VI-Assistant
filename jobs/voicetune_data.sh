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
: "${VI_OUT:?voicetune_data runs from \`vi dataset run\`}"

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
