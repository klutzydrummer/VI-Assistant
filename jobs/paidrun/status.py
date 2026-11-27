"""Read-only status of a paid run (paid-run status) and the stop request (paid-run stop)."""

import json
import os
import re
import signal
import subprocess
import sys

from paidrun.common import (
    EXIT, clock, proc_is_run, resolve_vi_bin, state_path, state_root, stop_path,
)
from paidrun.signals import eta_seconds


def load_state(run_id):
    """The state dict for a run id, or None if there is no such run."""
    try:
        return json.loads(state_path(run_id).read_text())
    except (OSError, ValueError):
        return None


def stage_tail(run_id, n=10):
    try:
        return (state_root() / run_id / "stage-tail.log").read_text().splitlines()[-n:]
    except OSError:
        return []


def build_status(s, now):
    """Everything `status` shows, from a state dict. Open-attempt figures are estimates."""
    at = next((x for x in s["attempts"] if x.get("ended") is None), None)
    open_h = max(0.0, (now - at["started"]) / 3600) if at else 0.0
    open_spend = round((at.get("dph") or 0) * open_h, 4) if at else 0.0
    spent = round(s["spent_total"] + open_spend, 4)
    hours = round(s["hours_total"] + open_h, 4)
    pr = s.get("progress") or {}
    step, total = pr.get("step"), pr.get("total")
    eta = eta_seconds(pr.get("samples") or [], step, total) if at else None
    alive = proc_is_run(s["pid"], s["run_id"], s.get("pid_start")) if s.get("pid") else False
    return dict(
        run_id=s["run_id"], status=s["status"], exit_code=s["exit_code"], message=s["message"],
        phase=at["phase"] if at else None,
        attempt=dict(n=at["n"] if at else len(s["attempts"]), max=s["max_attempts"]),
        offer=(at or (s["attempts"][-1] if s["attempts"] else {})).get("offer"),
        price_per_hour=(at or (s["attempts"][-1] if s["attempts"] else {})).get("dph"),
        spend=dict(closed_attempts=s["spent_total"], open_attempt_estimate=open_spend,
                   total=spent, total_is_estimate=at is not None),
        hours_used=hours, max_hours=s["max_hours"], hours_left=round(s["max_hours"] - hours, 4),
        cap=s["cap"], cap_left=round(s["cap"] - spent, 4),
        train=dict(step=step, total=total, updated_at=pr.get("updated_at"),
                   eta_seconds=eta, eta_is_estimate=True),
        process_alive=alive, stop_requested=stop_path(s["run_id"]).exists(),
        stage_log_tail=stage_tail(s["run_id"]),
        # r32 WP3: the last rental's validity (state, INVALID reason class), or None
        validity=s.get("validity"))


def fmt_secs(x):
    x = int(x)
    return f"{x // 3600}h{x % 3600 // 60:02d}m"


def cmd_status(args):
    s = load_state(args.run_id)
    if s is None:
        print(f"paid-run: no run with id {args.run_id!r} (looked for {state_path(args.run_id)})",
              file=sys.stderr)
        return EXIT["args"]
    st = build_status(s, clock())
    if args.credit:
        r = subprocess.run([resolve_vi_bin(), "cloud", "status"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
        m = re.search(r"^credit \$([\d.]+)", r.stdout, re.M)
        st["credit"] = float(m.group(1)) if r.returncode == 0 and m else None
    if args.json:
        print(json.dumps(st, indent=1))
        return 0
    t, sp = st["train"], st["spend"]
    est = " (estimate)" if sp["total_is_estimate"] else ""
    print(f"run {st['run_id']}: {st['status']}"
          + (f" (exit {st['exit_code']}): {st['message']}" if st["exit_code"] is not None else ""))
    if st["status"] == "running" and not st["process_alive"]:
        print("  WARNING: no live paid_run process for this run; start it again with the same "
              "limits to resume")
    print(f"  phase    : {st['phase'] or '-'}; attempt {st['attempt']['n']}/{st['attempt']['max']}"
          f"; offer {st['offer']} at ${st['price_per_hour'] or 0:.3f}/hr")
    print(f"  spend    : ${sp['total']:.2f}{est} = ${sp['closed_attempts']:.2f} closed attempts "
          f"+ ${sp['open_attempt_estimate']:.2f} open attempt (estimate)")
    print(f"  cap      : ${st['cap']:.2f}, ${st['cap_left']:.2f} left{est}")
    print(f"  hours    : {st['hours_used']:.2f} used of {st['max_hours']}, "
          f"{st['hours_left']:.2f} left{est}")
    if t["step"] is not None:
        eta = f"ETA about {fmt_secs(t['eta_seconds'])} (estimate)" if t["eta_seconds"] is not None \
            else "ETA not known yet (needs 3 samples)"
        print(f"  training : step {t['step']}/{t['total']}; {eta}")
    else:
        print("  training : no step read yet")
    v = st["validity"]
    if v:
        print(f"  validity : {v['state']}" + (f" ({v['reason']}): {v['detail']}" if v["reason"]
                                              else "") + f"; attempt {v['attempt']}")
    if st["stop_requested"]:
        print("  a stop has been requested")
    if "credit" in st:
        print(f"  credit   : " + (f"${st['credit']:.2f}" if st["credit"] is not None
                                 else "unavailable"))
    if st["stage_log_tail"]:
        print("  last stage output:")
        for ln in st["stage_log_tail"]:
            print(f"    {ln}")
    return 0


def cmd_stop(args):
    s = load_state(args.run_id)
    if s is None:
        print(f"paid-run: no run with id {args.run_id!r}; nothing changed", file=sys.stderr)
        return EXIT["args"]
    if s["status"] != "running":
        print(f"paid-run: run {args.run_id} is not running (status {s['status']}); nothing "
              "changed", file=sys.stderr)
        return EXIT["args"]
    pid = s.get("pid")
    if pid and not proc_is_run(pid, args.run_id, s.get("pid_start")):
        print(f"paid-run: the state says running but no paid_run process for {args.run_id} is "
              "alive (it was killed). Nothing changed; start it again with the same limits to "
              "resume and release the worker.", file=sys.stderr)
        return EXIT["args"]
    if args.now:
        if not pid:
            print("paid-run: the state records no process id; cannot signal. Nothing changed",
                  file=sys.stderr)
            return EXIT["args"]
        os.kill(pid, signal.SIGTERM)
        print(f"paid-run: sent SIGTERM to paid_run (pid {pid}); it releases the worker at once")
        return 0
    stop_path(args.run_id).write_text("stop\n")
    print(f"paid-run: stop requested for {args.run_id}. It waits for the current checkpoint "
          "push (at most 10 minutes), releases the worker and exits 8. Use `status` to follow.")
    return 0
