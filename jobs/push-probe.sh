#!/usr/bin/env bash
# Worker transfer-speed probe (round 30), sourced by jobs/on-worker.sh. It runs ONCE, at home,
# right after jobs/worker-preflight.sh passed (its 100 MiB push gate included), and only in a
# debug pass unless VI_PUSH_PROBE=1 asks for it (VI_PUSH_PROBE=0 turns it off). It measures where
# a slow checkpoint push comes from, three ways, with random synthetic bytes only (no audio, no
# text, no bucket):
#   a tunnel-rsync  200 MiB pushed by the worker through the existing reverse tunnel and receiver
#                   with the current flags (jobs/worker-return.sh push ... vi-out), into the
#                   throwaway folder vi-out/.push-probe, which home deletes afterwards. The bytes
#                   are counted at home, so a push cut off by the time limit still gives a rate.
#   b 4-streams     4 x 50 MiB at once, each over its own new ssh connection from home to the
#                   worker (`head -c N /dev/urandom` on the worker, counted and dropped at home).
#                   Not through the receiver: every receiver module allows one connection only
#                   (CLAUDE.md), and channels inside the one tunnel would share its single TCP
#                   connection anyway, so separate connections are what tests "more streams".
#   c public-curl   curl on the worker: download one public 100 MB test file (the first that
#                   answers of VI_PUSH_PROBE_URLS), then upload 50 MiB of random bytes to a public
#                   sink (VI_PUSH_PROBE_UP_URL). No home link is involved, so this is the worker's
#                   own line. Neither endpoint is officially documented; both answered from home
#                   in round 29. An empty VI_PUSH_PROBE_UP_URL means download only.
# The whole probe is capped at VI_PUSH_PROBE_SECONDS (default 300); every leg has its own time
# limit, cut to what is left of that cap, and a leg that would get under 10 s is skipped. Each
# home-side call is killed at its leg's limit (plus 5 s grace), and the worker-side command inside
# it stops 5 s sooner, so the probe never runs much past the cap. It is
# fail-soft: a failed, cut-off or skipped leg is reported and never stops the run or changes the
# gate. Output, on stderr: one `push-probe: leg ...` line per leg, then one `push-probe: json {...}`
# line, which jobs/paid_run.py copies into the debug summary.
#
# The caller sets: probe_ssh (the command that runs one shell string on the worker: `bash -c` for
# a local worker, else ssh with its own new connection), work (the worker's work folder), VI_OUT
# (home's stage output, where vi-out lands) and worker_port (the worker's end of the tunnel).

PUSH_PROBE_URLS_DEFAULT="https://fsn1-speed.hetzner.com/100MB.bin https://ash-speed.hetzner.com/100MB.bin https://nbg1-speed.hetzner.com/100MB.bin"
PUSH_PROBE_UP_URL_DEFAULT="https://speed.cloudflare.com/__up"

# probe_left: whole seconds left of the probe's cap.
probe_left() { echo $((probe_deadline - $(date +%s))); }

# probe_cap LEG_SECONDS: the leg's limit cut to the cap left, or 0 when under 10 s would remain.
probe_cap() {
  local left
  left=$(probe_left)
  [ "$left" -gt "$1" ] && left=$1
  [ "$left" -ge 10 ] || left=0
  echo "$left"
}

probe_ms() { echo $(( $(date +%s%N) / 1000000 )); }

# probe_rate BYTES MS -> decimal MB/s (2 decimals), or "null" with no bytes.
probe_rate() {
  python3 -c 'import sys; b, ms = int(sys.argv[1] or 0), max(1, int(sys.argv[2] or 1))
print(round(b / (ms / 1000) / 1e6, 2) if b > 0 else "null")' "$1" "$2"
}

# probe_record KEY=VALUE...: one leg's facts, collected for the JSON line.
probe_record() { printf '%s\n' "$*" >>"$probe_facts"; }

probe_leg_a() {
  local bytes=$1 cap rc=0 t0 t1 ms got status line q_work
  cap=$(probe_cap "${VI_PUSH_PROBE_LEG_SECONDS:-90}")
  if [ "$cap" -eq 0 ]; then
    probe_record "a name=tunnel-rsync status=skip detail=no_time_left"
    echo "push-probe: leg a tunnel-rsync: SKIP (no time left of the ${probe_total} s cap)" >&2
    return 0
  fi
  q_work=$(printf '%q' "$work")
  rm -rf "$VI_OUT/.push-probe"
  # The worker writes the file first (not timed), then times only the push. VI_PUSH_WAIT keeps
  # worker-return.sh's retry loop inside the leg; timeout -k stops the push at the leg's limit.
  t0=$(probe_ms)
  line=$(timeout -k 5 "$cap" "${probe_ssh[@]}" "set -u; cd $q_work || exit 0
rm -rf .push-probe-src; mkdir -p .push-probe-src/.push-probe
head -c $bytes /dev/urandom >.push-probe-src/.push-probe/probe.bin || { echo 'push-probe-result 98 0'; exit 0; }
t0=\$(date +%s%N); rc=0
VI_OUT=$q_work/out VI_RETURN_PORT=$(printf '%q' "$worker_port") VI_PUSH_WAIT=20 timeout -k 2 $((cap - 5)) bash $q_work/jobs/worker-return.sh push .push-probe-src vi-out >/dev/null 2>&1 || rc=\$?
t1=\$(date +%s%N); rm -rf .push-probe-src
echo \"push-probe-result \$rc \$(((t1 - t0) / 1000000))\"" 2>/dev/null | grep -a '^push-probe-result' | tail -n1) || rc=$?
t1=$(probe_ms)
# What reached home: the file, or its partial copy when the push was cut off.
got=$(du -sb "$VI_OUT/.push-probe" 2>/dev/null | cut -f1)
got=${got:-0}
rm -rf "$VI_OUT/.push-probe"
read -r _ rc ms <<<"${line:-x 255 0}"
case $ms in ''|*[!0-9]*|0) ms=$((t1 - t0)) ;; esac
if [ "$rc" = 0 ] && [ "$got" -ge "$bytes" ]; then status=ok
elif [ "$rc" = 124 ] || [ "$rc" = 137 ]; then status=timeout
else status=fail
fi
probe_record "a name=tunnel-rsync status=$status bytes=$got planned=$bytes ms=$ms rc=$rc cap_s=$cap"
echo "push-probe: leg a tunnel-rsync: $(probe_rate "$got" "$ms") MB/s ($got of $bytes bytes in $((ms / 1000)) s, $status, worker push exit $rc, limit $cap s)" >&2
}

probe_leg_b() {
  local bytes=$1 streams=$2 cap i each tmp t0 t1 got=0 slow=0 status=ok per=
  cap=$(probe_cap "${VI_PUSH_PROBE_LEG_SECONDS:-90}")
  if [ "$cap" -eq 0 ]; then
    probe_record "b name=4-streams status=skip detail=no_time_left"
    echo "push-probe: leg b $streams-streams: SKIP (no time left of the ${probe_total} s cap)" >&2
    return 0
  fi
  each=$((bytes / streams))
  tmp=$(mktemp -d)
  t0=$(probe_ms)
  for i in $(seq "$streams"); do
    (
      s=$(probe_ms)
      n=$(timeout -k 5 "$cap" "${probe_ssh[@]}" "head -c $each /dev/urandom" 2>/dev/null | wc -c)
      rc=${PIPESTATUS[0]}
      printf '%s %s %s\n' "${n:-0}" "$(( $(probe_ms) - s ))" "$rc" >"$tmp/$i"
    ) &
  done
  wait
  t1=$(probe_ms)
  for i in $(seq "$streams"); do
    read -r n ms rc <"$tmp/$i" 2>/dev/null || { n=0 ms=$((t1 - t0)) rc=255; }
    got=$((got + n))
    per+="${per:+ }$(probe_rate "$n" "$ms")"
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then slow=1
    elif [ "$rc" != 0 ] || [ "$n" -lt "$each" ]; then status=fail
    fi
  done
  rm -rf "$tmp"
  [ "$status" = ok ] && [ "$slow" = 1 ] && status=timeout
  probe_record "b name=4-streams status=$status bytes=$got planned=$((each * streams)) ms=$((t1 - t0)) streams=$streams per_stream_mb_s=${per// /,} cap_s=$cap"
  echo "push-probe: leg b $streams-streams: $(probe_rate "$got" $((t1 - t0))) MB/s in all ($got of $((each * streams)) bytes in $(((t1 - t0) / 1000)) s, $status; per stream ${per// /, } MB/s; limit $cap s)" >&2
}

probe_leg_c() {
  local up_bytes=$1 cap_d cap_u urls up_url q_work out rc down_line up_line
  urls=${VI_PUSH_PROBE_URLS-$PUSH_PROBE_URLS_DEFAULT}
  up_url=${VI_PUSH_PROBE_UP_URL-$PUSH_PROBE_UP_URL_DEFAULT}
  cap_d=$(probe_cap "${VI_PUSH_PROBE_CURL_SECONDS:-45}")
  if [ "$cap_d" -eq 0 ]; then
    probe_record "c name=public-curl status=skip detail=no_time_left"
    echo "push-probe: leg c public-curl: SKIP (no time left of the ${probe_total} s cap)" >&2
    return 0
  fi
  q_work=$(printf '%q' "$work")
  # Download: each URL in turn until one gives at least 50 MB (all inside cap_d together).
  rc=0
  out=$(timeout -k 5 "$cap_d" "${probe_ssh[@]}" "command -v curl >/dev/null 2>&1 || { echo 'push-probe-down none 0 0 0 no-curl'; exit 0; }
end=\$((\$(date +%s) + $cap_d - 5)); tried=
for u in $urls; do
  left=\$((end - \$(date +%s))); [ \$left -ge 5 ] || { [ -n \"\$tried\" ] || echo \"push-probe-down \$u 0 0 0 no-time\"; break; }; tried=1
  r=\$(curl -sS -L -o /dev/null --max-time \$left -w '%{http_code} %{size_download} %{time_total}' \"\$u\" 2>/dev/null); rc=\$?; [ -n \"\$r\" ] || r='000 0 0'
  echo \"push-probe-down \$u \$r \$rc\"
  set -- \$r; [ \"\${1:-0}\" = 200 ] && [ \"\${2:-0}\" -ge 50000000 ] && break
done" 2>/dev/null) || rc=$?
  down_line=$(printf '%s\n' "$out" | grep -a '^push-probe-down' | tail -n1)
  # Killed at the leg's limit before curl reported: curl's own timeout code (28) stands for it.
  if [ -z "$down_line" ] && { [ "$rc" = 124 ] || [ "$rc" = 137 ]; }; then
    down_line="push-probe-down ${urls%% *} 0 0 0 28"
  fi
  [ -n "$down_line" ] || down_line="push-probe-down none 0 0 0 no-answer"
  up_line=
  if [ -n "$up_url" ]; then
    cap_u=$(probe_cap "${VI_PUSH_PROBE_CURL_SECONDS:-45}")
    if [ "$cap_u" -gt 0 ]; then
      rc=0
      out=$(timeout -k 5 "$cap_u" "${probe_ssh[@]}" "cd $q_work || exit 0
command -v curl >/dev/null 2>&1 || { echo 'push-probe-up 0 0 0 no-curl'; exit 0; }
head -c $up_bytes /dev/urandom >.push-probe-up.bin
r=\$(curl -sS -o /dev/null --max-time $((cap_u - 5)) -w '%{http_code} %{size_upload} %{time_total}' --data-binary @.push-probe-up.bin $(printf '%q' "$up_url") 2>/dev/null); rc=\$?; [ -n \"\$r\" ] || r='000 0 0'
rm -f .push-probe-up.bin
echo \"push-probe-up \$r \$rc\"" 2>/dev/null) || rc=$?
      up_line=$(printf '%s\n' "$out" | grep -a '^push-probe-up' | tail -n1)
      if [ -z "$up_line" ] && { [ "$rc" = 124 ] || [ "$rc" = 137 ]; }; then
        up_line="push-probe-up 0 0 0 28"
      fi
      [ -n "$up_line" ] || up_line="push-probe-up 0 0 0 no-answer"
    else
      up_line="push-probe-up 0 0 0 no-time"
    fi
  fi
  python3 - "$down_line" "$up_line" "$up_url" "$probe_facts" <<'PY'
import sys
down, up, up_url, facts = sys.argv[1:5]

def leg(parts, url, kind, need):
    # parts: http_code size seconds curl_rc (or a reason word in place of curl_rc)
    code, size, secs, rc = (parts + ["0", "0", "0", "?"])[:4]
    try:
        size, secs = int(float(size)), float(secs)
    except ValueError:
        size, secs = 0, 0.0
    rate = round(size / secs / 1e6, 2) if size > 0 and secs > 0 else None
    if code == "200" and size >= need and rc == "0":
        status = "ok"
    elif rc == "28":
        status = "timeout"  # curl: operation timed out
    else:
        status = "fail"
    return dict(url=url, status=status, http=code, bytes=size, seconds=round(secs, 2),
                mb_s=rate, curl_rc=rc)

d = down.split()[1:]
dl = leg(d[1:], d[0] if d else "none", "download", 50_000_000)
ul = None
if up:
    ul = leg(up.split()[1:], up_url, "upload", 1)
st = dl["status"] if ul is None else ("ok" if dl["status"] == ul["status"] == "ok" else
                                      "partial" if "ok" in (dl["status"], ul["status"]) else "fail")
with open(facts, "a") as f:
    f.write("c name=public-curl status=%s down_url=%s down_status=%s down_bytes=%d down_s=%s "
            "down_mb_s=%s down_http=%s up_url=%s up_status=%s up_bytes=%s up_s=%s up_mb_s=%s "
            "up_http=%s\n" % (st, dl["url"], dl["status"], dl["bytes"], dl["seconds"], dl["mb_s"],
                               dl["http"], ul["url"] if ul else "", ul["status"] if ul else "none",
                               ul["bytes"] if ul else 0, ul["seconds"] if ul else 0,
                               ul["mb_s"] if ul else None, ul["http"] if ul else ""))
fmt = lambda x: "?" if x["mb_s"] is None else f"{x['mb_s']:g}"
text = (f"download {fmt(dl)} MB/s ({dl['bytes']} bytes in {dl['seconds']:g} s from {dl['url']}, "
        f"{dl['status']}, http {dl['http']}, curl {dl['curl_rc']})")
text += ("; upload not measured (no sink set)" if ul is None else
         f"; upload {fmt(ul)} MB/s ({ul['bytes']} bytes in {ul['seconds']:g} s to {ul['url']}, "
         f"{ul['status']}, http {ul['http']}, curl {ul['curl_rc']})")
print(f"push-probe: leg c public-curl: {text}", file=sys.stderr)
PY
}

# push_probe: run the three legs and print the JSON line. Never fails (fail-soft).
push_probe() {
  (
    set +e
    probe_total=${VI_PUSH_PROBE_SECONDS:-300}
    case $probe_total in ''|*[!0-9]*) probe_total=300 ;; esac
    probe_start=$(date +%s)
    probe_deadline=$((probe_start + probe_total))
    probe_facts=$(mktemp)
    bytes=${VI_PUSH_PROBE_BYTES:-$((200 * 1024 * 1024))}
    case $bytes in ''|*[!0-9]*|0) bytes=$((200 * 1024 * 1024)) ;; esac
    up_bytes=${VI_PUSH_PROBE_UP_BYTES:-$((50 * 1024 * 1024))}
    case $up_bytes in ''|*[!0-9]*|0) up_bytes=$((50 * 1024 * 1024)) ;; esac
    echo "push-probe: start (random bytes only; at most $probe_total s; a failed leg never stops the run)" >&2
    probe_leg_a "$bytes"
    probe_leg_b "$bytes" 4
    probe_leg_c "$up_bytes"
    python3 - "$probe_facts" "$probe_total" "$(( $(date +%s) - probe_start ))" <<'PY'
import json, sys
facts, cap, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
legs = {}
for line in open(facts):
    key, *items = line.split()
    d = {}
    for it in items:
        k, _, v = it.partition("=")
        if v in ("", "None", "null"):
            d[k] = None
            continue
        try:
            d[k] = int(v)
        except ValueError:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    if key in ("a", "b") and d.get("ms") is not None:
        b = d.get("bytes") or 0
        d["seconds"] = round(d["ms"] / 1000, 2)
        d["mb_s"] = round(b / max(d["ms"], 1) * 1000 / 1e6, 2) if b > 0 else None
    if key == "b" and isinstance(d.get("per_stream_mb_s"), str):
        d["per_stream_mb_s"] = [None if x == "null" else float(x)
                                for x in d["per_stream_mb_s"].split(",")]
    elif key == "b" and isinstance(d.get("per_stream_mb_s"), float):
        d["per_stream_mb_s"] = [d["per_stream_mb_s"]]
    legs[key] = d
for key, name in (("a", "tunnel-rsync"), ("b", "4-streams"), ("c", "public-curl")):
    legs.setdefault(key, dict(name=name, status="fail", detail="the leg left no result"))
doc = dict(ran=True, cap_seconds=cap, seconds=secs, legs=legs,
           note="random synthetic bytes only; MB/s is decimal (1e6 bytes/s)")
print("push-probe: json " + json.dumps(doc, sort_keys=True), file=sys.stderr)
PY
    rm -f "$probe_facts"
  ) || echo "push-probe: the probe itself failed (exit $?); carrying on" >&2
  return 0
}
