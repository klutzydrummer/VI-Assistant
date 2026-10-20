#!/bin/bash
# vi-worker boot script. `vi cloud up` fills in the limits and hands it to
# Vast.ai, which runs it when the instance starts. It installs what
# jobs/on-worker.sh needs and a watchdog that destroys this instance after
# @IDLE_MINUTES@ minutes with no job or ssh session, or @MAX_MINUTES@ minutes
# after boot (0 = no limit), so a forgotten worker can't drain the credit
# while the home machine is asleep.
mkdir -p /root/vi-jobs && rm -rf /root/vi-jobs/* && touch /root/vi-jobs/.active
# gcc: Triton (vLLM, and transformers models that use it) compiles a small
# helper at run time, and the pytorch runtime image ships no compiler.
# sox: qwen-tts calls it (voicesteer).
for t in rsync curl pgrep gcc sox; do command -v $t >/dev/null || missing=1; done
[ -n "$missing" ] && { apt-get update -qq; apt-get install -y -qq rsync curl procps gcc sox; } >/var/log/vi-onstart.log 2>&1
cat >/root/vi-watchdog.sh <<'EOF'
IDLE=$((@IDLE_MINUTES@ * 60)) MAX=$((@MAX_MINUTES@ * 60)) BOOT=$(date +%s)
API=https://console.vast.ai/api/v0/instances/$CONTAINER_ID/
release() {
  echo "$(date -u) releasing: $1" >>/var/log/vi-watchdog.log
  for _ in 1 2 3; do
    curl -fsS -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" "$API" && exit 0
    sleep 30
  done
  # Stopping at least ends the GPU charge; disk is billed until destroyed.
  curl -fsS -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" \
    -H 'Content-Type: application/json' -d '{"state":"stopped"}' "$API"
}
while sleep 60; do
  now=$(date +%s)
  pgrep -f 'sshd: [^ ]+@' >/dev/null && touch /root/vi-jobs/.active
  [ $((now - $(stat -c %Y /root/vi-jobs/.active))) -ge $IDLE ] && release idle
  [ $MAX -gt 0 ] && [ $((now - BOOT)) -ge $MAX ] && release max-hours
done
EOF
nohup bash /root/vi-watchdog.sh >/dev/null 2>&1 &
