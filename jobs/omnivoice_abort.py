"""Abort rules for an OmniVoice fine-tune run, from the pipeline's own logs (stdlib only).

`jobs/voicetune-fresh.sh` calls this every watchdog pass while the trainer runs:

    python3 -I jobs/omnivoice_abort.py STYLE_METRICS_JSONL COLLAPSE_GUARD_JSONL [FT_CONFIG_JSON]

It prints the abort reason, or nothing when the run is healthy:

* NaN loss — a non-finite value in `style_metrics.jsonl` (the sink floats every
  metric, and `json.dumps` writes NaN/infinity as the bare literals json.loads
  reads back). Nothing upstream stops a run on NaN, so the supervisor does.
* collapse guard fired — a probe row in `collapse_guard.jsonl` with
  `collapsed` and `strikes >= guard_patience` (from `ft_config.json`, default 2),
  taken from the **latest attempt only**: the pipeline restarts with a new seed
  after a fire and the log keeps every attempt, so an old attempt's fire row
  must not kill a restarted run that is still within its guard budget. If the
  latest attempt fires, the supervisor stops the run instead of waiting for the
  retries to burn paid time.

Malformed or partially-written trailing lines (the trainer is still writing)
are skipped. Exit code is 0 either way; the printed line is the only signal.
"""

import json
import math
import sys


def abort_reason(metrics_path, guard_path, ft_config_path=None):
    """The reason to stop this run, or "" when healthy."""
    patience = 2
    if ft_config_path:
        try:
            with open(ft_config_path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            cfg = {}
        if cfg.get("guard") is False:
            patience = None  # guard disabled: no fire rule
        else:
            patience = int(cfg.get("guard_patience", patience))
    try:
        with open(metrics_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # tail of a half-written line
                for key, value in row.items():
                    if isinstance(value, float) and not math.isfinite(value):
                        step = row.get("micro_step", "?")
                        return f"non-finite {key}={value} in style_metrics.jsonl at micro-step {step}"
    except OSError:
        pass
    if patience is not None:
        latest = 0
        rows = []
        try:
            with open(guard_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            rows = []
        for row in rows:
            latest = max(latest, int(row.get("attempt", 0)))
        for row in rows:
            if int(row.get("attempt", 0)) != latest:
                continue
            if row.get("collapsed") and int(row.get("strikes", 0)) >= patience:
                step = row.get("step", "?")
                return f"collapse guard fired at step {step} (attempt {latest})"
    return ""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: omnivoice_abort.py STYLE_METRICS_JSONL COLLAPSE_GUARD_JSONL [FT_CONFIG_JSON]")
    reason = abort_reason(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    if reason:
        print(reason)
