#!/bin/bash
# vi-worker boot: job tools, host keys, release watchdog.
export R=/root/vi-jobs
mkdir -p $R && touch $R/.active
[ -s $R/.boot ] || date +%s >$R/.boot
date +%s >$R/.this-boot
for t in rsync curl pgrep gcc sox setsid flock; do command -v $t >/dev/null || missing=1; done
[ -n "$missing" ] && { apt-get update -qq; apt-get install -y -qq rsync curl procps gcc sox util-linux; } >/var/log/vi-onstart.log 2>&1
# No ssh port: home gets it from the authenticated instance API.
for f in /etc/ssh/ssh_host_*_key.pub; do [ -e "$f" ] || continue; echo "VI-HOSTKEY $(date +%s) $(ssh-keygen -lf "$f")"; done | tee -a /var/log/vi-hostkey.log
cat >/root/vi-watchdog.sh <<'EOF'
IDLE=$((@IDLE_MINUTES@ * 60)) MAX=$((@MAX_MINUTES@ * 60)) BOOT=$(cat $R/.boot)
PH=$((${PENDING_HOLD:-120} * 60)) RM=${RECOVERY_MINUTES:-30}
API=https://console.vast.ai/api/v0/instances/$CONTAINER_ID/
A="Authorization: Bearer $CONTAINER_API_KEY"
log(){ echo "$(date -u) $*" >>/var/log/vi-watchdog.log; }
put(){ curl -fsS -X PUT -H "$A" -H 'Content-Type: application/json' -d '{"state":"stopped"}' "$API"; }
del(){ curl -fsS -X DELETE -H "$A" "$API"; }
release(){
  log "release: $1"
  if ls $R/*/pending/* >/dev/null 2>&1; then
    for _ in 1 2 3; do put&&{ touch $R/.stopped-at; log stopped; exit; }; sleep 30; done
    log stop-failed; del
  else
    for _ in 1 2 3; do del&&{ log destroyed; exit; }; sleep 30; done
    log destroy-failed; put
  fi
  exit
}
log start
while sleep 60; do
  now=$(date +%s); ssh=0; l=0; t=0; p=0; rc=0; g=0
  pgrep -f 'sshd: [^ ]+@' >/dev/null&&ssh=1
  for d in $R/*/; do
    [ -d "$d" ]||continue
    [ -e "$d/job.lock" ]&&! flock -n "$d/job.lock" true 2>/dev/null&&l=1
    [ -s "$d/train.pid" ]&&kill -0 "$(cat "$d/train.pid")" 2>/dev/null&&t=1
    ls "$d"/pending/* >/dev/null 2>&1&&p=1
    [ -s "$d/rc" ]&&{ rc=1; rct=$(stat -c %Y "$d/rc"); }
  done
  find $R -mindepth 2 -newermt "@$((now-IDLE))" -print -quit 2>/dev/null|grep -q .&&g=1
  live=$ssh
  [ $live = 0 ]&&[ $((l+t+p)) -gt 0 ]&&[ $g = 1 ]&&live=1
  if [ $live = 1 ];then touch $R/.active
  elif [ $l = 0 ]&&[ $t = 0 ]&&[ $p = 1 ]&&[ $rc = 1 ];then
    [ $((now-rct)) -ge $PH ]&&release pending
  elif [ $((now-$(stat -c %Y $R/.active))) -ge $IDLE ];then release idle;fi
  if [ -e $R/.stopped-at ];then S=$(cat $R/.this-boot 2>/dev/null||echo $BOOT); M=$((RM*60)); else S=$BOOT; M=$MAX; fi
  [ $M -gt 0 ]&&[ $((now-S)) -ge $M ]&&release max-hours
done
EOF
nohup bash /root/vi-watchdog.sh >/dev/null 2>&1 &
