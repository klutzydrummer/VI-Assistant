#!/usr/bin/env bash
# Home safety loop for the TTS listening test: `vi cloud reap` every 300 s, as a backstop to the
# worker's own idle/max-hours watchdog. Run it in its own terminal for the whole rental:
#   VI_CLOUD_CONFIG=jobs/ttstest/cloud-tts-test.toml jobs/ttstest/reap-loop.sh
# Stop it with Ctrl-C only after `vi cloud status` shows no worker. If it exits for ANY reason
# it says so loudly (terminal bell + message), because the backstop is then gone.
set -u
INTERVAL="${REAP_INTERVAL:-300}"

loud() {
  printf '\a\n!!! reap-loop.sh STOPPED (%s). The home backstop is OFF. Check `vi cloud status` NOW. !!!\n\n' "$1" >&2
}
trap 'loud "signal"; exit 130' INT TERM
trap 'rc=$?; [ "$rc" -eq 130 ] || loud "exit $rc"' EXIT

while true; do
  date -u +"%Y-%m-%dT%H:%M:%SZ reaping"
  if ! vi cloud reap; then
    printf '\a!!! `vi cloud reap` FAILED at %s; will retry in %s s. Check `vi cloud status`.\n' "$(date -u +%H:%M:%SZ)" "$INTERVAL" >&2
  fi
  sleep "$INTERVAL"
done
