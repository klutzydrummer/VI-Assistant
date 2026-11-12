#!/usr/bin/env bash
# Pre-rental gate for the moss15 run. Run it at home, free, before asking for approval to rent:
#
#   nice -n 19 ionice -c2 -n7 jobs/ttstest/smoke-gate.sh
#
# It FAILS (non-zero, loud) unless smoke_moss15_local.py ends with "MOSS15 CHECK OK" AND FFmpeg shared
# libraries (libavcodec, libavutil, libswresample) were reachable. The check is not weakened: torchcodec
# needs those libraries, exactly as on the worker, and without them the smoke cannot honestly pass.
#
# Where the libraries come from:
#   * already reachable: the system linker cache (`ldconfig -p`) or a directory in LD_LIBRARY_PATH;
#   * else, on NixOS, this script asks nix for `ffmpeg.lib` (from <nixpkgs>, no substituters; a download is
#     only attempted with TTSTEST_GATE_ALLOW_NIX_FETCH=1) and prepends its lib dir to LD_LIBRARY_PATH;
#   * else it stops with exit 3 and says what to do (set LD_LIBRARY_PATH to a dir holding the three libs,
#     e.g. `export LD_LIBRARY_PATH=$(nix-build '<nixpkgs>' -A ffmpeg.lib --no-out-link)/lib`).
# Exit codes: 0 gate passed; 1 smoke ran and failed; 2 smoke script missing/uv missing (smoke's own 2);
# 3 no FFmpeg libs found (smoke not started). Tests only: SMOKE_CMD replaces the smoke command,
# TTSTEST_GATE_NO_NIX=1 skips the nix lookup.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIBS=(libavcodec.so libavutil.so libswresample.so)

has_libs() {
  local cache l d ok
  cache=$(ldconfig -p 2>/dev/null || true)
  ok=1
  for l in "${LIBS[@]}"; do
    case $cache in *"$l"*) continue ;; esac
    local found=0
    IFS=: read -r -a dirs <<<"${LD_LIBRARY_PATH:-}"
    for d in "${dirs[@]}"; do
      [ -n "$d" ] || continue
      compgen -G "$d/$l*" >/dev/null && { found=1; break; }
    done
    [ $found = 1 ] || { ok=0; break; }
  done
  [ $ok = 1 ]
}

if ! has_libs && [ -z "${TTSTEST_GATE_NO_NIX:-}" ] && command -v nix-build >/dev/null 2>&1; then
  # No substituters = nothing is downloaded: it only succeeds when ffmpeg.lib is already in the store.
  subs=(--option substituters "")
  [ "${TTSTEST_GATE_ALLOW_NIX_FETCH:-0}" = 1 ] && subs=()
  out=$(nix-build '<nixpkgs>' -A ffmpeg.lib --no-out-link "${subs[@]}" 2>/dev/null | tail -1)
  if [ -n "$out" ] && [ -d "$out/lib" ]; then
    export LD_LIBRARY_PATH="$out/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "smoke-gate: using FFmpeg libs from $out/lib" >&2
  fi
fi

if ! has_libs; then
  printf '\a!!! smoke-gate FAILED: no FFmpeg shared libraries (libavcodec, libavutil, libswresample) are reachable. !!!\n' >&2
  echo "The moss15 smoke cannot honestly pass here, so DO NOT RENT. Make them reachable, then rerun, e.g.:" >&2
  echo "  export LD_LIBRARY_PATH=\$(nix-build '<nixpkgs>' -A ffmpeg.lib --no-out-link)/lib   # NixOS (may download ~100 MB)" >&2
  echo "  or TTSTEST_GATE_ALLOW_NIX_FETCH=1 jobs/ttstest/smoke-gate.sh" >&2
  exit 3
fi

if [ -n "${SMOKE_CMD:-}" ]; then
  out=$(bash -c "$SMOKE_CMD" 2>&1); rc=$?
else
  out=$(python3 -I "$HERE/smoke_moss15_local.py" 2>&1); rc=$?
fi
printf '%s\n' "$out" | tail -25
if [ $rc -ne 0 ] || ! printf '%s\n' "$out" | grep -q '^MOSS15 CHECK OK'; then
  printf '\a!!! smoke-gate FAILED (smoke exit %s, no "MOSS15 CHECK OK" line). DO NOT RENT. !!!\n' "$rc" >&2
  [ $rc -ne 0 ] && exit "$rc"
  exit 1
fi
echo "smoke-gate: PASSED" >&2
