#!/usr/bin/env bash
# Run the TTS test stage and, when the job says there is nothing worth waiting for, destroy the rented worker
# at once instead of letting it bill until the idle limit.
#
#   jobs/ttstest/run-or-release.sh nice -n 19 "$VI_BIN" dataset run "$N/pipeline.toml" tts
#
# The job (worker_run.py) exits 5 when the timed download was too slow (nothing installed, nothing fetched)
# and 6 when every model failed setup or smoke (nothing to generate). On exactly those two codes this script:
#   1. clears the empty placeholder checkpoint with `checkpoints.py verify --expect 0` (the same step as
#      run-commands step 9; `vi cloud down` refuses while an unverified marker exists), then
#   2. runs `vi cloud down` (the Vast instance is destroyed by home, via the `vi` binary).
# Any other exit code (0, 2, 3, 4, ...) is passed through untouched and the worker is NOT destroyed:
# results may still need to be repushed or read (see README). Env: VI_BIN (default `vi`), VI_STATE_DIR, TTSTEST_STAGE (stage name, default tts: only its markers are touched).
# Exit code: the job's own code; 70 if the job asked for release and the release itself failed (loud).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
VI=${VI_BIN:-vi}
state_dir=${VI_STATE_DIR:-${XDG_STATE_HOME:-${HOME:-.}/.local/state}/vi}

"$@"
rc=$?
case $rc in
  5|6) ;;
  *) exit "$rc" ;;
esac

echo "run-or-release: the job exited $rc (nothing to wait for); destroying the worker now" >&2
failed=0
for m in "$state_dir/unverified/${TTSTEST_STAGE:-tts}"-*; do
  [ -f "$m" ] && [ ! -L "$m" ] || continue
  d=$(cat "$m" 2>/dev/null) || continue
  [ -n "$d" ] || continue
  python3 -I "$HERE/../checkpoints.py" verify "$d" --expect 0 >&2 || failed=1
done
if ! "$VI" cloud down >&2; then failed=1; fi
if [ "$failed" = 1 ]; then
  printf '\a!!! run-or-release: releasing the worker FAILED. Run `vi cloud status`, then `vi cloud down` by hand. !!!\n' >&2
  exit 70
fi
exit "$rc"
