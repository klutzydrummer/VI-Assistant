#!/usr/bin/env bash
# OmniVoice full fine-tune with the learned style prefix (stage 5): one preflight job, then one
# train job, run on the GPU worker by jobs/on-worker.sh. Preflight proves the card, disk and
# generate path with a 5-step probe and pushes the probe checkpoint home; train runs 5000 steps
# from the public Base, pushes each finished checkpoint-<step> home as it lands, aborts on NaN
# loss or a fired collapse guard, then builds the labelled listening sets at steps 1000/2500/5000
# (before home can prune them away) and pushes them plus the run files home.
#
# `push` runs only the publish-push loop. The train mode holds $work/push.lock while it runs, so
# when a killed session leaves the detached trainer alive, the home supervisor starts `push`
# under the same lock: finished checkpoints keep going home, and checkpoints home found corrupt
# are re-sent with --checksum while the receiver moves the old copy aside (nothing is deleted).
#
# A checkpoint is published only once its SHA256SUMS exists: train_ft.py writes the manifest
# last and atomically, so "a checkpoint only exists once its manifest does". The collapse guard
# removes its attempt's checkpoint dirs when it restarts with a new seed, so the publish mirror
# tracks the run dir (hard links; a replaced checkpoint is re-copied and re-pushed).
#
# Home acks (jobs/on-worker.sh ack_epochs) still speak epoch-<k>, so the push loop serves both
# layouts: epoch-<k> dirs are acked and repaired through home's acks, while checkpoint-<step>
# dirs keep a local $work/pushed/checkpoint-<step> marker instead of waiting for acks (and still
# honor $work/acked/checkpoint-<step> if a future on-worker.sh writes them).
#
# Preflight also prepares the data folder (step 0a: extract_audio_tokens into shards/, step 0b:
# replay_public.py --device cuda into replay/ and shards/replay/; both need CUDA) and rehearses a
# real kill and resume: a 20-step run saving every 10 steps is SIGKILLed after checkpoint-10 and
# resumed from it.
# The train mode prepares the data folder the same way: on-worker.sh sends the inputs into a fresh
# work dir for every stage, so the shards preflight built are not there (a reattach keeps them,
# and prepare_data then skips every piece).
#
# Expected inputs (jobs/on-worker.sh maps stage `inputs` to VI_INPUT_<NAME>):
#   VI_INPUT_OMNIVOICE_DATA/                 one prepared folder, built at home, holding
#                                            (the omnivoice_data stage; the older voicetune.toml
#                                            stages name theirs VI_INPUT_VOICETUNE_DATA, which is
#                                            read when the first is unset)
#     data_config.json                       OmniVoice data config (manifest_path -> data.lst)
#     style_manifest.jsonl                   style-source rows (id, audio_path, text) for the
#                                            listening sets and the probe line
#     audio_tokenizer/                       the Higgs v2 tokenizer (optional when the Base
#                                            snapshot below already carries it)
#     train_manifest.jsonl, dev_manifest.jsonl, replay_texts.jsonl, voices/
#                                            what steps 0a/0b read (jobs/omnivoice_data.py)
#     shards/{train,dev,replay}/             the shards named by data_config.json, written by
#                                            steps 0a/0b into the same folder
#   VI_BASE_MODEL                            local snapshot of k2-fsa/OmniVoice, used as
#                                            --init-from and --base-model. Unset on a rented
#                                            worker: the script then downloads the public Base
#                                            (VI_BASE_MODEL_REPO at VI_BASE_MODEL_REVISION) into
#                                            ~/.cache/vi/base-model and sets it. The Base is a
#                                            public model, not game data, so it comes from the
#                                            hub, not from home.
# Every manifest path inside the data folder is relative to it, so this script runs with it as
# the working directory. Listening lines come from jobs/ttstest/lines.json (infer_listening's
# default) — the hub's established lines, the same every checkpoint.
set -euo pipefail

jobs_dir=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-}

usage() {
  echo "usage: voicetune-fresh.sh preflight|train|push" >&2
  echo "  preflight  gate VRAM/disk, tokenise the clips, generate the replay set, run a 5-step probe, push it home, speak one line, rehearse kill and resume" >&2
  echo "  train      5000 steps from Base, push each checkpoint home, abort on NaN/guard fire, build the listening sets" >&2
  echo "  push       only push finished checkpoints and repairs home (started by the home supervisor)" >&2
}

case "$MODE" in
  preflight|train|push) ;;
  *) usage; exit 2 ;;
esac

: "${VI_OUT:?voicetune-fresh.sh runs through jobs/on-worker.sh}"

# Worker-side scratch inside on-worker.sh's work dir (deleted with the job): run/publish/probe live
# here, not under VI_OUT, so checkpoint weights aren't re-sent over vi-out at the end.
scratch="$(cd "$(dirname "$VI_OUT")" && pwd)/voicetune-fresh"
run="$scratch/run"
publish="$scratch/publish"
probe_run="$scratch/probe-run"
probe_publish="$scratch/probe-publish"
mkdir -p "$scratch"
work="$(cd "$(dirname "$VI_OUT")" && pwd)"

# publish_sync SRC DST: mirror every finished checkpoint from SRC into DST as hard links (so the
# copy costs no disk), and drop published checkpoints the trainer removed again. Only a
# checkpoint with its SHA256SUMS counts; a replaced same-name checkpoint (collapse guard
# restart) is detected by its manifest and re-copied, and its pushed marker cleared so it is
# re-sent home.
publish_sync() {
  local src=$1 dst=$2 ck d n
  for ck in "$src"/checkpoint-*; do
    [ -d "$ck" ] || continue
    [ -f "$ck/SHA256SUMS" ] || continue
    d="$dst/${ck##*/}"
    if [ -e "$d" ]; then
      if [ -f "$d/SHA256SUMS" ] && cmp -s "$ck/SHA256SUMS" "$d/SHA256SUMS"; then
        continue
      fi
      n=${ck##*/checkpoint-}
      case $n in ''|*[!0-9]*) ;; *) rm -f "$work/pushed/checkpoint-$n" "$work/wpushed/checkpoint-$n" ;; esac
      rm -rf "$d"
    fi
    mkdir -p "$dst"
    cp -al "$ck" "$d"
  done
  for d in "$dst"/checkpoint-*; do
    [ -d "$d" ] || continue
    [ -d "$src/${d##*/}" ] || rm -rf "$d"
  done
}

# The publish dir holds checkpoint-<step> dirs (the omnivoice layout, train_ft.py) and, for a
# stage on the legacy epoch layout, epoch-<k> dirs. Home's acks and repair requests use the
# same names ($work/acked/<name>, $work/repush/<name>), so every loop below walks both. Only
# checkpoint-<step> also gets this script's own pushed markers; an epoch is re-sent every pass
# until home acks it.
#
# Which finished checkpoints still need to go home. Home acks ($work/acked/<name>) or, for a
# checkpoint-<step>, this script's own pushed markers ($work/pushed/checkpoint-<step>) exclude
# a checkpoint from the whole-directory push; everything else is re-sent (rsync skips
# unchanged files) every pass.
acked_checkpoints() {
  local c name n acked=()
  for c in "$publish"/checkpoint-* "$publish"/epoch-*; do
    [ -d "$c" ] || continue
    name=${c##*/}
    case $name in *.part) continue ;; esac
    n=${name#*-}
    case $n in ''|*[!0-9]*) continue ;; esac
    if [ -e "$work/acked/$name" ] || { [ "${name%%-*}" = checkpoint ] && [ -e "$work/pushed/$name" ]; }; then
      acked+=("--exclude=/$name")
    fi
  done
  [ ${#acked[@]} -eq 0 ] || printf '%s\n' "${acked[@]}"
}

# A whole-directory push only marks checkpoints pushed when it succeeds, so a failed pass is
# retried next loop; a fresh checkpoint that appears after the last successful push is unmarked
# and goes out with the next pass.
mark_pushed() {
  local c n
  mkdir -p "$work/pushed"
  for c in "$publish"/checkpoint-*; do
    [ -d "$c" ] || continue
    n=${c##*/checkpoint-}
    case $n in ''|*[!0-9]*) continue ;; esac
    : >"$work/pushed/checkpoint-$n"
  done
}

mark_bad() {  # NAME (checkpoint-<step> or epoch-<k>)
  mkdir -p "$work/bad"
  : >"$work/bad/$1"
  rm -f "$work/repush/$1" "$work/repush/$1.fails" "$work/pushed/$1"
  echo "voicetune-fresh: ALERT: ${1%%-*} ${1#*-} failed three repair pushes; marked $work/bad/$1 and no longer retried (the corrupt copy stays home beside the repaired files)" >&2
}

# push_repairs: re-send checkpoints home found corrupt. --checksum ignores size and
# mtime (a flipped byte keeps both), --backup moves the bad copy into
# .rsync-replaced-<attempt>/ instead of overwriting it, and --relative keeps
# the checkpoint-<step>/ layout the whole-directory push uses. After three failed
# pushes the checkpoint is marked bad, loudly, and never retried.
push_repairs() {
  local rp name kind n fails f
  for rp in "$work/repush"/checkpoint-* "$work/repush"/epoch-*; do
    [ -e "$rp" ] || continue
    name=${rp##*/}
    case $name in *.fails) continue ;; esac
    kind=${name%%-*}
    n=${name#*-}
    case $n in ''|*[!0-9]*) continue ;; esac
    f="$work/repush/$name.fails"
    fails=$(cat "$f" 2>/dev/null || true)
    case $fails in ''|*[!0-9]*) fails=0 ;; esac
    if [ "$fails" -ge 3 ]; then mark_bad "$name"; continue; fi
    if [ ! -d "$publish/$name" ]; then
      printf '%s\n' "$((fails + 1))" >"$f"
      echo "voicetune-fresh: $kind $n not found in $publish; cannot repair it" >&2
      [ "$((fails + 1))" -lt 3 ] || mark_bad "$name"
      continue
    fi
    if "$jobs_dir/worker-return.sh" push "$publish/./$name" "${VI_PUBLISH:-vi-publish}" \
         --relative --checksum --backup --backup-dir=".rsync-replaced-$((fails + 1))" --exclude='*.part'; then
      rm -f "$rp" "$f"
      if [ "$kind" = checkpoint ]; then
        mkdir -p "$work/pushed"
        : >"$work/pushed/$name"
      fi
      echo "voicetune-fresh: $kind $n re-sent; the copy it replaced is under .rsync-replaced-$((fails + 1))/" >&2
    else
      fails=$((fails + 1))
      printf '%s\n' "$fails" >"$f"
      echo "voicetune-fresh: $kind $n repair push failed ($fails/3)" >&2
      [ "$fails" -lt 3 ] || mark_bad "$name"
    fi
  done
}

# push_publish DIR: push a publish dir's finished checkpoints (never a .part) home
# via the vi-publish module, which on-worker.sh points at this attempt's
# durable dir. Repairs go first, then the ordinary push.
push_publish() {
  local dir=$1 acked=()
  [ -d "$dir" ] || return 0
  push_repairs
  if weights_first; then
    push_two_part "$dir"
    return
  fi
  mapfile -t acked < <(acked_checkpoints)
  if "$jobs_dir/worker-return.sh" push "$dir" "${VI_PUBLISH:-vi-publish}" --exclude='*.part' "${acked[@]}"; then
    mark_pushed
  else
    return 1
  fi
}

# weights_first: on unless switched off. VI_WEIGHTS_FIRST (1 or 0) wins; else the stage setting
# weights_first from the $VI_SETTINGS JSON; else on. Home's verify in jobs/on-worker.sh passes
# --weights-only / --weights-only-older, so it accepts an older checkpoint held as weights only.
weights_first() {
  case "${VI_WEIGHTS_FIRST:-}" in 1) return 0 ;; 0) return 1 ;; esac
  [ -n "${VI_SETTINGS:-}" ] && [ -r "$VI_SETTINGS" ] || return 0
  python3 - "$VI_SETTINGS" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1])).get("weights_first", True)
except (OSError, ValueError):
    value = True
sys.exit(0 if value not in (False, 0, "0", "false", "False") else 1)
PY
}

# Weights-only publish (F-PUSH-6 c). Per pass:
#   1. weights part: every checkpoint without optimizer.bin, scheduler.bin, random_states_0.pkl
#      and without SHA256SUMS, but with its own SHA256SUMS.weights (the full manifest minus those
#      three files), so the weights are checkable the moment they land.
#   2. full part, for the newest checkpoint only (the one training resumes from).
#   3. an older checkpoint whose full part is not home is not sent it once a newer checkpoint's
#      full part is; home gets the weights-only manifest as its SHA256SUMS instead (staged
#      beside the publish dir, pushed after the weights), which checkpoints.py verify
#      --weights-only-older accepts for every checkpoint but the newest.
FULL_ONLY_RE='  (optimizer\.bin|scheduler\.bin|random_states_0\.pkl)$'
push_two_part() {
  local dir=$1 c name n newest=-1 acked=() skip=() part1=() stage="$scratch/weights-stage"
  local mod=${VI_PUBLISH:-vi-publish}
  for c in "$dir"/checkpoint-*; do
    [ -f "$c/SHA256SUMS" ] || continue
    n=${c##*/checkpoint-}
    case $n in ''|*[!0-9]*) continue ;; esac
    [ "$n" -le "$newest" ] || newest=$n
    grep -vE "$FULL_ONLY_RE" "$c/SHA256SUMS" >"$c/SHA256SUMS.weights" || true
  done
  mapfile -t acked < <(acked_checkpoints)
  part1=(--exclude='*.part' --exclude=SHA256SUMS --exclude=optimizer.bin --exclude=scheduler.bin
         --exclude=random_states_0.pkl --exclude=SHA256SUMS.weights)
  "$jobs_dir/worker-return.sh" push "$dir" "$mod" "${part1[@]}" "${acked[@]}" || return 1
  mkdir -p "$work/wpushed"
  for c in "$dir"/checkpoint-*; do
    [ -f "$c/SHA256SUMS.weights" ] || continue
    name=${c##*/}; n=${name#checkpoint-}
    : >"$work/wpushed/$name"
    if [ "$n" -lt "$newest" ] && [ ! -e "$work/pushed/$name" ] && [ ! -e "$work/acked/$name" ]; then
      skip+=("$name" "--exclude=/$name")
    fi
  done
  local skipargs=() i
  for ((i = 1; i < ${#skip[@]}; i += 2)); do skipargs+=("${skip[i]}"); done
  "$jobs_dir/worker-return.sh" push "$dir" "$mod" --exclude='*.part' --exclude=SHA256SUMS.weights "${skipargs[@]}" "${acked[@]}" || return 1
  for ((i = 0; i < ${#skip[@]}; i += 2)); do
    name=${skip[i]}
    mkdir -p "$stage/$name"
    cp "$dir/$name/SHA256SUMS.weights" "$stage/$name/SHA256SUMS"
    "$jobs_dir/worker-return.sh" push "$stage/./$name" "$mod" --relative --exclude='*.part' || return 1
    echo "voicetune-fresh: ${name} is home as weights only (its optimizer state was skipped; a newer checkpoint is home in full)" >&2
  done
  mark_pushed
}

# push_final DIR: the last push after the trainer exits. A receiver restarting at that moment must
# not fail a finished run, so it gets 3 tries; VI_FINAL_PUSH_SLEEP (default 20 s) is the pause.
push_final() {
  local try
  for try in 1 2 3; do
    push_publish "$1" && return 0
    [ "$try" -lt 3 ] || break
    echo "voicetune-fresh: final push failed (try $try of 3); retrying" >&2
    sleep "${VI_FINAL_PUSH_SLEEP:-20}"
  done
  return 1
}

# backlog_check: F-TRN-3. Checkpoints finished here but not yet home pile up on the worker's disk
# when pushes keep failing. Warn when free space is under 3 x one checkpoint's size with any
# un-acked checkpoint; return 1 (the caller writes abort.json "push backlog") at six or more.
# VI_FREE_GB overrides the measured free space.
backlog_check() {
  local c name n unacked=0 size=0 s free
  for c in "$publish"/checkpoint-* "$publish"/epoch-*; do
    [ -d "$c" ] || continue
    name=${c##*/}
    case $name in *.part) continue ;; esac
    n=${name#*-}
    case $n in ''|*[!0-9]*) continue ;; esac
    [ ! -e "$work/acked/$name" ] || continue
    if [ "${name%%-*}" = checkpoint ] && [ -e "$work/pushed/$name" ]; then continue; fi
    unacked=$((unacked + 1))
    s=$(du -sbL "$c" | cut -f1)
    [ "$s" -le "$size" ] || size=$s
  done
  [ "$unacked" -gt 0 ] || return 0
  free=${VI_FREE_GB:-$(df -P -k "$work" | awk 'NR==2 {printf "%.1f", $4 / 1048576}')}
  if awk -v f="$free" -v s="$size" 'BEGIN {exit !(f * 1073741824 < 3 * s)}'; then
    echo "voicetune-fresh: WARNING: $free GB free is under 3 x the checkpoint size ($size bytes) with $unacked checkpoints not yet home" >&2
  fi
  if [ "$unacked" -ge "${VI_BACKLOG_MAX:-6}" ]; then
    echo "voicetune-fresh: push backlog: $unacked checkpoints are not home" >&2
    return 1
  fi
}

push_loop() {
  local c name n rp pending
  while :; do
    publish_sync "$run" "$publish"
    push_publish "$publish" || echo "voicetune-fresh: publish push failed; will retry" >&2
    if ! backlog_check; then
      mkdir -p "$run"
      python3 -c 'import json; print(json.dumps({"reason": "push backlog"}))' >"$run/abort.json"
      break
    fi
    # A live trainer can still publish another checkpoint: keep going.
    if [ -s "$work/train.pid" ] && kill -0 "$(cat "$work/train.pid" 2>/dev/null)" 2>/dev/null; then
      sleep "${VI_PUSH_SECONDS:-15}"
      continue
    fi
    pending=0
    for c in "$publish"/checkpoint-* "$publish"/epoch-*; do
      [ -d "$c" ] || continue
      name=${c##*/}
      case $name in *.part) continue ;; esac
      n=${name#*-}
      case $n in ''|*[!0-9]*) continue ;; esac
      if [ ! -e "$work/acked/$name" ]; then
        if [ "${name%%-*}" = epoch ] || [ ! -e "$work/pushed/$name" ]; then
          pending=1
        fi
      fi
    done
    for rp in "$work/repush"/checkpoint-* "$work/repush"/epoch-*; do
      [ -e "$rp" ] || continue
      case ${rp##*/} in *.fails) continue ;; esac
      pending=1
    done
    [ "$pending" = 1 ] || break
    sleep "${VI_PUSH_SECONDS:-15}"
  done
}

if [ "$MODE" = push ]; then
  # One push loop per job. The train mode holds the same lock for its
  # lifetime, so this exits quietly if a live supervisor is already pushing.
  exec 8>>"$work/push.lock"
  if ! flock -n 8; then
    echo "voicetune-fresh: a push loop already holds $work/push.lock; exiting" >&2
    exit 0
  fi
  echo $$ >"$work/push.pid"
  trap 'rm -f "$work/push.pid"' EXIT
  push_loop
  exit 0
fi

: "${VI_RETURN_PORT:?voicetune-fresh.sh needs the worker return tunnel}"

# The prepared data folder; every manifest path inside it is relative to it, so
# run with it as the working directory for the job (all other paths are absolute).
# on-worker.sh names the variable after the stage input: omnivoice_data -> OMNIVOICE_DATA.
data_dir=${VI_INPUT_OMNIVOICE_DATA:-${VI_INPUT_VOICETUNE_DATA:-}}
[ -n "$data_dir" ] || { echo "voicetune-fresh: missing VI_INPUT_OMNIVOICE_DATA (or VI_INPUT_VOICETUNE_DATA)" >&2; exit 1; }
cd "$data_dir"

# The OmniVoice venv: the image's CUDA torch is reused through --system-site-packages, only the
# training deps are installed — the same shape jobs/ttstest/worker_run.py uses for omnivoice.
# The venv is named after the requirements file's sha256, so a changed list gets a fresh venv.
# OVFT_PYTHON overrides for a pre-built environment.
ovft_python() {
  if [ -n "${OVFT_PYTHON:-}" ]; then
    printf '%s\n' "$OVFT_PYTHON"
    return
  fi
  local req="$jobs_dir/requirements-omnivoice-worker.txt" venv
  venv="$HOME/.cache/vi/venv-$(sha256sum "$req" | cut -c1-16)"
  if [ ! -x "$venv/bin/python" ]; then
    # Build into venv.tmp-$$ and rename only on success, so a half-built venv is never reused.
    # 3 tries, sleeping 30 s then 90 s (VI_PIP_BACKOFF="30 90" overrides, for tests).
    local tmp="$venv.tmp-$$" try delays rc=1
    read -r -a delays <<<"${VI_PIP_BACKOFF:-30 90}"
    for try in 0 1 2; do
      rm -rf "$tmp"
      if python3 -m venv --system-site-packages "$tmp" &&
         "$tmp/bin/python" -m pip install -q --retries 10 --timeout 60 -r "$req"; then
        rc=0
        break
      fi
      [ "$try" -lt 2 ] && sleep "${delays[$try]:-30}"
    done
    if [ "$rc" != 0 ]; then
      rm -rf "$tmp"
      echo "worker-preflight: pip install failed (3 tries) for $req" >&2
      exit 1
    fi
    mv "$tmp" "$venv"
  fi
  printf '%s\n' "$venv/bin/python"
}
# Thread caps and the tokeniser worker count for this box (jobs/worker-limits.sh), exported before
# any Python starts so the tokeniser, replay_public.py and train_ft.py all inherit them. Too few
# free task slots prints a "worker-preflight: tasks ..." line, which retries on another machine.
# shellcheck source=jobs/worker-limits.sh
. "$jobs_dir/worker-limits.sh"
worker_limits || exit 1
PY=$(ovft_python)
ACCELERATE="$(dirname "$PY")/accelerate"

# Base model: use VI_BASE_MODEL when given, otherwise fetch the public k2-fsa/OmniVoice snapshot
# at a pinned revision (the one replay_public.py's home default points at). The folder is kept
# under ~/.cache/vi so a reattach or the train mode does not download it again; HF tokens are
# never passed on.
if [ -z "${VI_BASE_MODEL:-}" ]; then
  VI_BASE_MODEL=$(env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HF_HUB_OFFLINE \
    "$PY" "$jobs_dir/omnivoice-ft/fetch_base_model.py" \
    "${VI_BASE_MODEL_REPO:-k2-fsa/OmniVoice}" \
    "${VI_BASE_MODEL_REVISION:-c5fdb5ccb189668d56333f77ba2629f4cd7535f4}" \
    "$HOME/.cache/vi/base-model") || { echo "voicetune-fresh: could not fetch the Base model" >&2; exit 1; }
  echo "voicetune-fresh: Base model at $VI_BASE_MODEL"
fi
export VI_BASE_MODEL

# abort_reason METRICS_JSONL GUARD_JSONL FT_CONFIG_JSON -> the abort reason, or empty.
# Pure stdlib python (jobs/omnivoice_abort.py): NaN in style_metrics.jsonl, or a collapse-guard
# fire in the latest attempt. `-I` keeps the data folder's files off sys.path.
abort_reason() {
  python3 -I "$jobs_dir/omnivoice_abort.py" "$1" "$2" "$3"
}

# check_and_gate: report VRAM and disk (check.json under VI_OUT), then gate them against the
# stage's floors (min_vram_gb from the settings JSON; VI_MIN_VRAM_GB/VI_MIN_DISK_GB
# override). Pure stdlib python, no venv.
check_and_gate() {
  mkdir -p "$VI_OUT/check"
  python3 - "$VI_OUT/check/check.json" <<'PY'
import json, os, subprocess, sys
# GiB, the unit nvidia-smi's MiB gives (a "24 GB" card reads 23.9). The floor comes from
# VI_MIN_VRAM_GB, else the stage's min_vram_gb in the settings JSON, else 23.5.
settings = {}
if os.environ.get("VI_SETTINGS"):
    try:
        settings = json.load(open(os.environ["VI_SETTINGS"]))
    except (OSError, ValueError):
        pass
min_vram = float(os.environ.get("VI_MIN_VRAM_GB") or settings.get("min_vram_gb") or 23.5)
min_disk = float(os.environ.get("VI_MIN_DISK_GB", "100"))
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip()
    total, used = (float(x) for x in out.split(",")[:2])
except Exception as e:
    print(f"check: nvidia-smi failed: {e}", file=sys.stderr)
    sys.exit(1)
st = os.statvfs(os.getcwd())
disk = st.f_bavail * st.f_frsize / 2**30
json.dump({"vram_total_gb": round(total / 1024, 1), "vram_free_gb": round((total - used) / 1024, 1),
           "free_disk_gb": round(disk, 1)}, open(sys.argv[1], "w"), indent=1, sort_keys=True)
if total / 1024 < min_vram:
    print(f"check: VRAM {total / 1024:.1f} GiB < {min_vram:.1f} GiB", file=sys.stderr)
    sys.exit(1)
if disk < min_disk:
    print(f"check: worker disk {disk:.0f} GB free < {min_disk:.0f} GB", file=sys.stderr)
    sys.exit(1)
print(f"check: VRAM {total / 1024:.1f} GiB, disk {disk:.0f} GB free")
PY
}

# audio_tokenizer_dir: the Higgs v2 tokenizer folder, in order of preference: $VI_AUDIO_TOKENIZER,
# the data folder's audio_tokenizer/ (when jobs/omnivoice_data.py bundled a real one; a
# PLACEHOLDER-only folder does not count), then the Base snapshot's own audio_tokenizer/.
audio_tokenizer_dir() {
  if [ -n "${VI_AUDIO_TOKENIZER:-}" ]; then
    printf '%s\n' "$VI_AUDIO_TOKENIZER"
  elif [ -f "$data_dir/audio_tokenizer/config.json" ]; then
    printf '%s\n' "$data_dir/audio_tokenizer"
  elif [ -d "$VI_BASE_MODEL/audio_tokenizer" ]; then
    printf '%s\n' "$VI_BASE_MODEL/audio_tokenizer"
  else
    echo "no audio tokenizer: set VI_AUDIO_TOKENIZER, bundle one in the data folder, or use a Base snapshot that carries audio_tokenizer/" >&2
    return 1
  fi
}

# tokenise NAME MANIFEST: upstream extract_audio_tokens turns the manifest into WebDataset shards
# under shards/NAME/ (tars, jsonl metadata and data.lst, the file data_config.json names). A
# shard set that already has a non-empty data.lst is kept, so a rerun after a reattach or the
# train mode does not redo GPU work.
tokenise() {
  local name=$1 manifest=$2 tok lst="$data_dir/shards/$1/data.lst"
  if [ -s "$lst" ]; then
    echo "tokenise: shards/$name already built; keeping them"
    return 0
  fi
  tok=$(audio_tokenizer_dir) || return 1
  [ -s "$manifest" ] || { echo "tokenise: $manifest missing or empty" >&2; return 1; }
  mkdir -p "$data_dir/shards/$name"
  local log="$data_dir/shards/$name/extract.log" rc=0
  # Watched, in its own process group, with the worker count from jobs/worker-limits.sh: a fatal
  # pool crash stops it within VI_FATAL_QUIET_SECONDS instead of hanging, and no orphan is left.
  vi_run_watched "$log" "$PY" -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl "$manifest" \
    --tar_output_pattern "$data_dir/shards/$name/audios/shard-%06d.tar" \
    --jsonl_output_pattern "$data_dir/shards/$name/txts/shard-%06d.jsonl" \
    --tokenizer_path "$tok" --samples_per_shard 1000 --skip_errors \
    --nj_per_gpu "${VI_LIMIT_NJ:-1}" || rc=$?
  [ "$rc" = 0 ] || { echo "tokenise: extract_audio_tokens exited $rc for $name" >&2; return 1; }
  [ -s "$lst" ] || { echo "tokenise: $lst was not written for $name" >&2; return 1; }
  tokenise_count_guard "$name" "$manifest" "$log" || { rm -f "$lst"; return 1; }
}

# tokenise_count_guard NAME MANIFEST LOG: refuse a silently shrunk set. Reads upstream's
# "Processing Complete - Successful: N, Failed: M, Filtered/Skipped: K" line and compares it with
# the manifest row count R: fail when N < R, or when M+K > max(2, 1% of R). It only counts; it
# never chooses, weights or filters clips, and reads no labels.
tokenise_count_guard() {
  python3 -I - "$1" "$2" "$3" <<'PY'
import re, sys
name, manifest, log = sys.argv[1:4]
rows = sum(1 for line in open(manifest, encoding="utf-8", errors="replace") if line.strip())
text = open(log, encoding="utf-8", errors="replace").read()
m = None
for m in re.finditer(r"Processing Complete\s*-\s*Successful:\s*(\d+),\s*Failed:\s*(\d+),\s*Filtered/Skipped:\s*(\d+)", text):
    pass
if m is None:
    print(f"voicetune-fresh: tokenise {name}: no 'Processing Complete' line in {log}; "
          "refusing to train on a set whose size is unknown", file=sys.stderr)
    sys.exit(1)
n, f, k = (int(x) for x in m.groups())
if n < rows or f + k > max(2, rows // 100):
    print(f"voicetune-fresh: tokenise {name}: {n} of {rows} rows tokenised ({f} failed, {k} skipped); "
          "refusing to train on a silently shrunk set", file=sys.stderr)
    sys.exit(1)
PY
}

# prepare_data: steps 0a and 0b, both need CUDA and both write into the data folder.
#   0a  tokenise the train and dev manifests into shards/train and shards/dev
#   0b  generate the public replay set with Base (replay_public.py --device cuda) into replay/,
#       then tokenise it into shards/replay. Without replay rows the KL-to-Base guard has nothing
#       to act on (omnivoice-ft/README.md), so a missing replay set fails here, not at step 4000.
# Every piece is skipped when its output already exists (replay_public.py also resumes per line).
prepare_data() {
  echo "preflight 0a: tokenise train and dev clips"
  tokenise train "$data_dir/train_manifest.jsonl"
  tokenise dev "$data_dir/dev_manifest.jsonl"
  echo "preflight 0b: generate and tokenise the replay set"
  if [ ! -s "$data_dir/shards/replay/data.lst" ]; then
    "$PY" "$jobs_dir/omnivoice-ft/replay_public.py" \
      --out-dir "$data_dir/replay" --base-model "$VI_BASE_MODEL" \
      --voices "$data_dir/voices" --texts "$data_dir/replay_texts.jsonl" --device cuda
  fi
  tokenise replay "$data_dir/replay/replay_public.jsonl"
}

# gpu_gate: the first CUDA touch, then the bf16 check, then the pinned library versions in the
# log. A CUDA init failure or a pre-Ampere card is host-specific, so each prints a
# "worker-preflight: ..." line that makes the run retry on a new machine; a Python traceback
# after CUDA works stays non-retryable (a code bug repeats on every host).
gpu_gate() {
  local msg
  if ! msg=$("$PY" -c 'import torch; torch.zeros(1).cuda(); torch.cuda.synchronize()' 2>&1); then
    echo "worker-preflight: cuda init failed: $(printf '%s' "$msg" | tail -n 1)" >&2
    exit 1
  fi
  if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    echo "worker-preflight: bf16 unsupported on this GPU" >&2
    exit 1
  fi
  "$PY" -c '
import importlib.metadata as m
out = []
for n in ("omnivoice", "torch", "transformers", "accelerate", "webdataset"):
    try:
        out.append(f"{n}={m.version(n)}")
    except m.PackageNotFoundError:
        out.append(f"{n}=missing")
print("versions: " + " ".join(out))' || echo "versions: could not read" >&2
}

# vram_peak_gate LOG: F-OOM-1 (d). train_ft prints "train_ft: vram peak X of Y GiB" at the end of
# a CUDA run. Above 90 % of the card the real run is one longer batch from an out-of-memory stop
# hours in, so print the retryable worker-preflight line (a new worker with a higher VRAM floor
# is tried) and exit 1. No such line (a non-CUDA run or an older trainer) means no gate, only a warning.
vram_peak_gate() {
  local line x y
  line=$(grep -E 'train_ft: vram peak [0-9.]+ of [0-9.]+ GiB' "$1" 2>/dev/null | tail -n 1 || true)
  if [ -z "$line" ]; then
    echo "voicetune-fresh: WARNING: no 'train_ft: vram peak' line in $1; the VRAM headroom gate was skipped" >&2
    return 0
  fi
  x=$(printf '%s' "$line" | sed -E 's/.*vram peak ([0-9.]+) of ([0-9.]+) GiB.*/\1/')
  y=$(printf '%s' "$line" | sed -E 's/.*vram peak ([0-9.]+) of ([0-9.]+) GiB.*/\2/')
  if awk -v x="$x" -v y="$y" 'BEGIN {exit !(x > 0.90 * y)}'; then
    echo "worker-preflight: vram peak $x of $y GiB" >&2
    exit 1
  fi
  echo "preflight: vram peak $x of $y GiB (limit 90 %)"
}

# eta_check STEP ELAPSED_S: F-TRN-5. From step 100 on, extrapolate the time the remaining steps
# need at the pace so far. Over hours_left - 0.5 h (VI_HOURS_LEFT is the budget left when train
# started, 8 if unset), warn once and push now so a stop at the cap loses as little as possible.
# It only warns: the step count never changes.
eta_check() {
  local step=$1 elapsed=$2 total=${VI_TRAIN_STEPS:-5000} left_h out
  [ "$step" -ge "${VI_ETA_MIN_STEP:-100}" ] || return 0
  [ ! -e "$work/eta.warned" ] || return 0
  out=$(awk -v st="$step" -v el="$elapsed" -v tot="$total" -v hl="${VI_HOURS_LEFT:-8}" 'BEGIN {
    proj = (tot - st) * el / st / 3600; left = hl - el / 3600
    if (proj > left - 0.5) printf "%.1f %.1f", proj, left }')
  [ -n "$out" ] || return 0
  : >"$work/eta.warned"
  echo "voicetune-fresh: projected ${out%% *} h > ${out##* } h left" >&2
  publish_sync "$run" "$publish"
  push_publish "$publish" || echo "voicetune-fresh: publish push failed; will retry" >&2
}

# rehearse_kill_resume: the real-GPU kill/resume proof. A 20-step run saving every 10 steps is
# SIGKILLed a few micro-steps after checkpoint-10 lands (about step 15), then resumed from
# checkpoint-10; both checkpoints must pass --verify-checkpoint. A run that finishes before the
# kill, or never reaches checkpoint-10, fails the rehearsal rather than passing it vacuously.
rehearse_kill_resume() {
  local rr="$scratch/rehearse-run-$(date +%s)" pidfile pgid bg waited=0 m0 m
  local limit=${VI_REHEARSE_TIMEOUT:-900} trainer=("$jobs_dir/omnivoice-ft/train_ft.py")
  local args=(--init-from "$VI_BASE_MODEL" --data-config "$data_dir/data_config.json"
              --output-dir "$rr" --steps 20 --save-steps 10 --logging-steps 1
              --num-workers "${VI_NUM_WORKERS:-2}")
  mkdir -p "$rr"
  pidfile="$rr/trainer.pgid"
  # setsid makes the launcher its own process group, so one kill -9 takes the accelerate
  # process and every dataloader worker with it.
  setsid bash -c 'echo $$ >"$1"; shift; exec "$@"' _ "$pidfile" \
    "$ACCELERATE" launch --num_processes 1 "${trainer[@]}" "${args[@]}" &
  bg=$!
  while [ ! -s "$pidfile" ]; do sleep 0.2; done
  pgid=$(cat "$pidfile")
  last_micro() {
    python3 -c '
import json, sys
last = 0
for line in open(sys.argv[1], errors="replace"):
    try:
        last = max(last, int(json.loads(line)["micro_step"]))
    except (ValueError, KeyError, TypeError):
        pass
print(last)' "$rr/style_metrics.jsonl" 2>/dev/null || echo 0
  }
  while [ ! -f "$rr/checkpoint-10/SHA256SUMS" ]; do
    kill -0 "$bg" 2>/dev/null || { echo "rehearsal: trainer exited before checkpoint-10" >&2; return 1; }
    [ "$waited" -lt "$((limit * 5))" ] || { kill -9 -- "-$pgid" 2>/dev/null || true; echo "rehearsal: no checkpoint-10 in ${limit}s" >&2; return 1; }
    sleep 0.2; waited=$((waited + 1))
  done
  m0=$(last_micro)
  while :; do
    if [ -d "$rr/checkpoint-20" ]; then
      kill -9 -- "-$pgid" 2>/dev/null || true
      echo "rehearsal: the run finished before it could be killed; no kill was proven" >&2
      return 1
    fi
    m=$(last_micro)
    [ "$m" -ge "$((m0 + ${VI_REHEARSE_KILL_AFTER:-3}))" ] && break
    kill -0 "$bg" 2>/dev/null || { echo "rehearsal: trainer exited before the kill point" >&2; return 1; }
    sleep 0.2
  done
  kill -9 -- "-$pgid" 2>/dev/null || true
  wait "$bg" 2>/dev/null || true
  echo "rehearsal: SIGKILL at micro-step $m (checkpoint-10 was written at $m0)"
  # Give the GPU memory back before the resume starts.
  for _ in $(seq 1 60); do
    kill -0 -- "-$pgid" 2>/dev/null || break
    sleep 0.5
  done
  [ ! -d "$rr/checkpoint-20" ] || { echo "rehearsal: checkpoint-20 exists after the kill; the kill proved nothing" >&2; return 1; }
  "$ACCELERATE" launch --num_processes 1 "${trainer[@]}" "${args[@]}" \
    --resume-from-checkpoint "$rr/checkpoint-10"
  "$PY" "${trainer[@]}" --verify-checkpoint "$rr/checkpoint-10" "$rr/checkpoint-20" \
    || { echo "rehearsal: resumed run failed checkpoint verification" >&2; return 1; }
  python3 - "$VI_OUT/preflight.json" "$m" <<'PY'
import json, sys
path, killed = sys.argv[1], int(sys.argv[2])
try:
    report = json.load(open(path))
except (OSError, ValueError):
    report = {}
report["kill_resume"] = {"killed_at_micro_step": killed, "resumed_from": "checkpoint-10", "verified": ["checkpoint-10", "checkpoint-20"]}
json.dump(report, open(path, "w"), indent=1, sort_keys=True)
PY
  echo "rehearsal: resumed from checkpoint-10 and verified checkpoint-20"
}

preflight() {
  check_and_gate
  gpu_gate
  prepare_data

  # Five real steps at the full batch (--steps 5 --save-steps 5 writes checkpoint-5), published
  # and pushed home as an end-to-end integrity proof, timed for the record. The Qwen3 loadbench
  # and push-rate gate are gone: they were voicetune.py internals with no OmniVoice counterpart.
  # The probe uses the trainer's default batch, the same one train() runs, so it is the largest
  # batch the real run will see; its log carries the VRAM peak line the gate reads.
  mkdir -p "$probe_run"
  "$ACCELERATE" launch --num_processes 1 "$jobs_dir/omnivoice-ft/train_ft.py" \
    --init-from "$VI_BASE_MODEL" --data-config "$data_dir/data_config.json" \
    --output-dir "$probe_run" --steps 5 --save-steps 5 \
    --num-workers "${VI_NUM_WORKERS:-2}" 2>&1 | tee "$probe_run/train.log"
  [ -d "$probe_run/checkpoint-5" ] || { echo "preflight: no $probe_run/checkpoint-5 produced" >&2; exit 1; }
  vram_peak_gate "$probe_run/train.log"
  publish_sync "$probe_run" "$probe_publish"

  local bytes seconds mb_s t0 t1
  bytes=$(du -sb "$probe_publish" | cut -f1)
  t0=$(date +%s%N)
  push_publish "$probe_publish"
  t1=$(date +%s%N)
  seconds=$(( (t1 - t0) / 1000000000 ))
  [ "$seconds" -gt 0 ] || seconds=1
  mb_s=$(python3 -c 'import sys; print(round(int(sys.argv[1]) / int(sys.argv[2]) / 1e6, 2))' "$bytes" "$seconds")
  python3 - "$VI_OUT/preflight.json" "$bytes" "$seconds" "$mb_s" <<'PY'
import json, sys
path, probe_bytes, seconds, mb_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
try:
    report = json.load(open(path))
except (OSError, ValueError):
    report = {}
report.update({"probe_bytes": probe_bytes, "seconds": seconds, "mb_s": mb_s})
json.dump(report, open(path, "w"), indent=1, sort_keys=True)
print(f"probe push {mb_s} MB/s ({probe_bytes} bytes in {seconds} s)")
PY

  # Speak one line from the probe checkpoint, so a real checkpoint's generate path is exercised
  # too (infer_listening refuses an unverifiable checkpoint, so this also re-checks the contract).
  mkdir -p "$VI_OUT/probe-speak"
  "$PY" "$jobs_dir/omnivoice-ft/infer_listening.py" \
    --checkpoint "$probe_run/checkpoint-5" --out "$VI_OUT/probe-speak" \
    --base-model "$VI_BASE_MODEL" \
    ${VI_AUDIO_TOKENIZER:+--audio-tokenizer "$VI_AUDIO_TOKENIZER"} \
    --style-manifest "$data_dir/style_manifest.jsonl" \
    --max-styles 1 --max-lines 1 --device "${VI_INFER_DEVICE:-cuda}"

  rehearse_kill_resume

  echo "preflight: ok"
}

train() {
  # Hold the push lock for this supervisor's lifetime. A detached push loop
  # started after a session kill holds it instead; this run then refuses
  # rather than push twice.
  exec 8>>"$work/push.lock"
  if ! flock -n 8; then
    echo "voicetune-fresh: another push loop holds $work/push.lock" >&2
    exit 1
  fi
  # Preflight already proved generate, so this gates VRAM/disk without the extra model load.
  check_and_gate
  prepare_data

  local start reason elapsed rc wj_pid rc_file pid_file
  start=$(date +%s)
  rc_file="$(dirname "$VI_OUT")/rc"
  # worker-job.sh writes the trainer's pid beside its own files (the job dir,
  # which the watchdog scans), not under --run.
  pid_file="$(dirname "$VI_OUT")/train.pid"
  # worker-job.sh detaches the trainer from this stage session, holds the job
  # lock, writes train.pid and restarts a crash with --resume-from-checkpoint
  # (its resume flag change is part of this task). This script stays the
  # supervisor: it pushes finished checkpoints home and stops the run on the
  # abort rules. 8>&- keeps the push lock from leaking into the detached
  # trainer, so a killed supervisor really does release it.
  "$jobs_dir/worker-job.sh" --work "$(dirname "$VI_OUT")" --run "$run" \
    --publish "$publish" -- "$ACCELERATE" launch --num_processes 1 \
    "$jobs_dir/omnivoice-ft/train_ft.py" \
    --init-from "$VI_BASE_MODEL" --data-config "$data_dir/data_config.json" \
    --output-dir "$run" --steps "${VI_TRAIN_STEPS:-5000}" --save-steps "${VI_SAVE_STEPS:-500}" \
    --num-workers "${VI_NUM_WORKERS:-2}" 8>&- &
  wj_pid=$!
  # If this script exits on its own (an abort, or a push failure), kill the
  # trainer. A signal that kills only the stage session must not: the detached
  # wrapper keeps running and writes rc, so the attempt can be recovered.
  # Exit 75 is a lost job lock: another attempt holds it and wrote train.pid,
  # so that pid is not this script's trainer to kill.
  dying=0
  trap 'dying=1; exit 143' HUP TERM
  trap '[ "$dying" = 1 ] || [ "${rc:-0}" -eq 75 ] || kill "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null || true' EXIT

  # Watchdog: push each finished checkpoint home as it appears, and stop the run on the abort
  # rules (NaN loss; collapse guard fired in the latest attempt).
  reason=""
  local step
  while [ ! -s "$rc_file" ]; do
    publish_sync "$run" "$publish"
    # A loop push failure must not orphan the running train, so it is non-fatal
    # here; the final push below is the one that decides success.
    push_publish "$publish" || echo "voicetune-fresh: publish push failed; will retry" >&2
    reason=$(abort_reason "$run/style_metrics.jsonl" "$run/collapse_guard.jsonl" "$run/ft_config.json")
    [ -n "$reason" ] || backlog_check || reason="push backlog"
    if [ -z "$reason" ]; then
      step=$(python3 -c '
import json, sys
last = 0
for line in open(sys.argv[1], errors="replace"):
    try:
        last = max(last, int(json.loads(line)["micro_step"]))
    except (ValueError, KeyError, TypeError):
        pass
print(last)' "$run/style_metrics.jsonl" 2>/dev/null || echo 0)
      eta_check "$step" "$(( $(date +%s) - start ))"
    fi
    if [ -n "$reason" ]; then
      echo "train abort: $reason" >&2
      # Written before the kill: worker-job.sh sees it and never restarts an
      # aborted run.
      python3 -c 'import json, sys; print(json.dumps({"reason": sys.argv[1]}))' "$reason" >"$run/abort.json"
      kill "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null || true
      break
    fi
    sleep "${VI_WATCH_SECONDS:-15}"
  done
  set +e
  wait "$wj_pid"
  rc=$?
  set -e
  trap - EXIT HUP TERM
  publish_sync "$run" "$publish"
  push_final "$publish"  # push what exists (the final checkpoints, or the ones before an abort)
  if [ -n "$reason" ]; then
    echo "train aborted: $reason" >&2
    exit 1
  fi
  [ "$rc" -eq 0 ] || { echo "train failed: exit $rc" >&2; exit "$rc"; }

  # Labelled listening sets at steps 1000/2500/5000 (decision Q1b), built from those checkpoints
  # BEFORE home can prune them away: prune keeps only the two newest plus the exempt 1000/2500,
  # so the sets must exist by the time the run ends. infer_listening writes listening.jsonl and a
  # README.txt stating the rule — human listening decides voice quality; there is no automatic
  # score. The sets land in the platter, which the hub presents as the one listening place.
  local platter="$VI_OUT/platter" step
  mkdir -p "$platter"
  for step in 1000 2500 5000; do
    if [ ! -d "$run/checkpoint-$step" ]; then
      echo "train: checkpoint-$step missing; skipping its listening set" >&2
      continue
    fi
    "$PY" "$jobs_dir/omnivoice-ft/infer_listening.py" \
      --checkpoint "$run/checkpoint-$step" --out "$platter/step-$step" \
      --base-model "$VI_BASE_MODEL" \
      ${VI_AUDIO_TOKENIZER:+--audio-tokenizer "$VI_AUDIO_TOKENIZER"} \
      --style-manifest "$data_dir/style_manifest.jsonl" \
      --max-styles "${VI_MAX_STYLES:-6}" --device "${VI_INFER_DEVICE:-cuda}"
  done

  # Run files for the record, beside the platter (check.json is already under VI_OUT/check).
  cp "$run/ft_config.json" "$run/style_metrics.jsonl" "$run/collapse_guard.jsonl" "$VI_OUT/"
  cp "$VI_OUT/check/check.json" "$VI_OUT/" 2>/dev/null || true

  echo "train: done; listening sets in $platter"
}

case "$MODE" in
  preflight) preflight ;;
  train) train ;;
esac
