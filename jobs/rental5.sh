#!/usr/bin/env bash
# Step 5's one rental (docs/design/step5-rental-plan.md), run by on-worker.sh as the `rental5` stage.
# Every fork has a rule decided in advance (Brandon, 2026-10-20), so the worker takes each branch
# itself and writes which one and why to decisions.md; nothing waits on anyone, and only choices
# that are explicitly Brandon's (e.g. mean_diff vectors instead of joint) are left for him.
#
#   1. fetch Qwen3.8-27B (sha256-checked) and run preflight on it, with the reuse gates
#      (copy test and fit-level check, docs/vectors.md). Pass: capture reusing the persona.
#      Fail: capture with --no-reuse (about 1-1.5 h more). Either way it's logged.
#   2. build: capture every labelled line, fit (joint is the default), write the vector bank.
#   3. chat check (vi vectors check): thinking on, steering reply-only and all tokens, every vector
#      at strength 1. Vectors that go off are retried at 0.5, then 0.25, and only those.
#   4. voice: voicesteer (her clips, full strength first, lowered only if her voice or the words go).
#
# Each step's output lands in VI_CHECKPOINT as it's made, so a lost worker resumes where it was.
# Files sent with --put (files/): vi (the CUDA build from .github/workflows/cuda-build.yml),
# models.toml, persona-checked inputs from corpus/tali/analysis/after_labels/builder_inputs.py
# (messages.jsonl, labels.jsonl, pairs.jsonl), check_plan.jsonl and check_plan.py, and
# voice_sets.json and voice_probe.npz.
set -euo pipefail
: "${VI_OUT:?}" "${VI_CHECKPOINT:?}"
f=files
ck=$VI_CHECKPOINT
model=${VI_RENTAL_MODEL:-qwen3.8-27b-q8}   # the home smoke runs this same script on qwen3.5-0.8b-q8
models=${VI_MODELS:-$HOME/.cache/vi/models}
vi=$PWD/$f/vi
chmod +x "$vi"
log="$VI_OUT/decisions.md"
mkdir -p "$ck" "$VI_OUT"
[ -f "$ck/decisions.md" ] && cp "$ck/decisions.md" "$log"
note() { echo "- $(date -u +%H:%M) $*" | tee -a "$log"; cp "$log" "$ck/decisions.md"; }
common=(--models "$models" --registry "$f/models.toml")

note "worker: $(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "no GPU")"
"$vi" vectors selftest --dir "$ck/selftest" >"$VI_OUT/selftest.log" 2>&1 || { note "selftest failed; stopping (see selftest.log)"; exit 1; }
"$vi" vectors fetch "$model" "${common[@]}"

# 1. Preflight on the real model decides reuse. (The persona's byte-check against vi chat ran at
# home on these exact messages; the character pack isn't sent.) A failure only in the reuse gates
# (copy-free fit-level check) means capture without reuse; any other failure stops the run.
if [ ! -f "$ck/reuse.choice" ]; then
  if "$vi" vectors preflight "$model" "${common[@]}" --messages "$f/messages.jsonl" \
       --labels "$f/labels.jsonl" --pairs "$f/pairs.jsonl" >"$VI_OUT/preflight.log" 2>&1; then
    echo reuse >"$ck/reuse.choice"; note "preflight passed (copy test and fit-level check): capturing with the persona reused"
  else
    failed=$(grep -i "failed" "$VI_OUT/preflight.log" | grep -vi "preflight failed" || true)
    if [ -n "$failed" ] && ! echo "$failed" | grep -qiv "fit-level\|fit check\|reuse"; then
      echo no-reuse >"$ck/reuse.choice"; note "preflight: only the fit-level reuse check failed: capturing with --no-reuse"
    else
      note "preflight failed outside the reuse gate; stopping (see preflight.log)"; exit 1
    fi
  fi
fi
reuse=()
[ "$(cat "$ck/reuse.choice")" = no-reuse ] && reuse=(--no-reuse)

# 2. Build: capture (resumable store) and fit.
"$vi" vectors build "$model" "${common[@]}" "${reuse[@]}" --messages "$f/messages.jsonl" --labels "$f/labels.jsonl" \
  --pairs "$f/pairs.jsonl" --store "$ck/store" --out "$ck/vectors" >"$VI_OUT/build.log" 2>&1
note "built $(ls "$ck/vectors/joint"/*.gguf | wc -l) joint vectors"

# 3. Chat check, then lower only the vectors that went off, at most twice.
plans=("$f/check_plan.jsonl")
results=()
for round in 1 2 3; do
  res="$ck/check-$round.jsonl"
  results+=("$res")
  "$vi" vectors check "$model" "${common[@]}" --vectors "$ck/vectors/joint" --plan "${plans[-1]}" --results "$res" \
    >"$VI_OUT/check-$round.log" 2>&1
  # Scored over every round so far: round 1's baselines stay the comparison, and only a vector's
  # lowest strength tried is lowered again.
  python3 "$f/check_plan.py" score "${results[@]}" --plan "${plans[@]}" >"$VI_OUT/check.md"
  next="$ck/next_plan.jsonl"
  [ -s "$next" ] || { note "chat check round $round: nothing went off; done"; break; }
  [ "$round" = 3 ] && { note "chat check: $(grep -c . "$next") trials still off at the lowest strength"; break; }
  note "chat check round $round: $(grep -c . "$next") trials went off; retrying them one step lower"
  mv "$next" "$ck/plan-$((round + 1)).jsonl"
  plans+=("$ck/plan-$((round + 1)).jsonl")
done

# 4. Voice vectors and test D (voicesteer reads its settings from VI_SETTINGS).
VI_FILES=$PWD/$f VI_CHECKPOINT=$ck/voice python3 jobs/voicesteer.py >"$VI_OUT/voicesteer.log" 2>&1   # its own checkpoint folder
note "voice: $(grep -o '"verdict": "[^"]*"' "$VI_OUT/report.json" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"

cp -r "$ck/vectors" "$VI_OUT/chat_vectors"   # voicesteer writes its own vectors/
for k in check-1 check-2 check-3; do [ -f "$ck/$k.jsonl" ] && cp "$ck/$k.jsonl" "$VI_OUT/"; done
note "done"
