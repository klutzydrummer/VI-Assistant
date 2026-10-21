#!/usr/bin/env bash
# Build qwentts.cpp (a ggml port of Qwen3-TTS, MIT) with Vulkan, at a pinned commit with
# jobs/qwentts/vi.patch (the streaming text layout voicearms uses for x-vector cloning, and arm A's
# residual steering), then make its GGUFs from the Qwen3-TTS-12Hz-1.7B-Base checkpoint: F32 and
# Q8_0 talkers, F32 tokenizer. Safe to rerun: every step skips what is already there.
#
#   jobs/qwentts-setup.sh <folder>     # about 10 GB: sources, build, checkpoint links, models
#
# On Brandon's NixOS machine, inside a shell with the build tools:
#   nix-shell -p cmake gcc gnumake pkg-config git vulkan-headers vulkan-loader shaderc.bin spirv-headers \
#     python3 --run 'nice -n 19 jobs/qwentts-setup.sh <folder>'
# The checkpoint comes from the Hugging Face cache (the voicearms smoke already downloaded it), or
# from QWEN_TTS_CKPT, a folder holding it;
# nothing of hers is read or written here. Then the check: python3 jobs/qtcheck.py, see its doc.
set -euo pipefail

DIR=${1:?usage: jobs/qwentts-setup.sh <folder>}
COMMIT=5916bf7c79dbb4e66fc059cdc507a79a23a28bb4
REPO=$(cd "$(dirname "$0")/.." && pwd)
PATCH="$REPO/jobs/qwentts/vi.patch"
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
SRC="$DIR/qwentts.cpp"

if [ ! -d "$SRC/.git" ]; then
    git clone https://github.com/ServeurpersoCom/qwentts.cpp.git "$SRC"
fi
cd "$SRC"
if [ "$(git rev-parse HEAD)" != "$COMMIT" ]; then
    git fetch -q origin
    git checkout -q "$COMMIT"
fi
git submodule update -q --init --recursive
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[setup] vi.patch already applied"
else
    git apply "$PATCH"
    echo "[setup] applied vi.patch"
fi

if [ ! -x build/qwen-tts ] || [ "$PATCH" -nt build/qwen-tts ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
    cmake --build build -j "$(nproc)"
fi
usage=$(build/qwen-tts --help 2>&1 || true)
case "$usage" in *--steer*) ;; *) echo "[setup] build lacks the vi patch"; exit 1 ;; esac

# Python for convert.py, which needs torch: voicearms' virtualenv (vi_job.ensure_venv, built by the
# voicearms smoke), else a venv here with the CPU build of torch; QWEN_TTS_PYTHON overrides.
STEER_VENV="$HOME/.cache/vi/venv-$(sha256sum "$REPO/jobs/requirements-steer.txt" | cut -c1-16)"
PY=${QWEN_TTS_PYTHON:-}
if [ -z "$PY" ] && [ -x "$STEER_VENV/bin/python" ]; then
    PY="$STEER_VENV/bin/python"
elif [ -z "$PY" ]; then
    [ -x "$DIR/venv/bin/python" ] || python3 -m venv "$DIR/venv"
    "$DIR/venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
    PY="$DIR/venv/bin/python"
fi
"$PY" -m pip install -q gguf safetensors numpy huggingface_hub

# Checkpoints as links into the Hugging Face cache, under the names convert.py expects. The Base
# repo's speech_tokenizer/ is the 12 Hz tokenizer itself (same config as Qwen3-TTS-Tokenizer-12Hz).
mkdir -p checkpoints models
BASE=${QWEN_TTS_CKPT:-$("$PY" -c \
    'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))')}
ln -sfn "$BASE" checkpoints/Qwen3-TTS-12Hz-1.7B-Base
ln -sfn "$BASE/speech_tokenizer" checkpoints/Qwen3-TTS-Tokenizer-12Hz

"$PY" -I convert.py
[ -f models/qwen-talker-1.7b-base-Q8_0.gguf ] ||
    build/quantize models/qwen-talker-1.7b-base-F32.gguf models/qwen-talker-1.7b-base-Q8_0.gguf Q8_0
ls -la models
echo "[setup] done: $SRC/build/qwen-tts, models in $SRC/models"
