#!/usr/bin/env python3
"""Background watcher for a paid VI run: phone pings through ntfy, plus a status file.

One pass per run of this script; a systemd user timer calls it every few minutes (see
docs/bg-watch.md). It uses only the standard library, so `python3 -I` is safe. It costs
nothing while no run is active: with no run id it exits at once and touches no network.

Events, each sent once (the key is kept in state.json):
  needs-approval        an agent is waiting for a permission answer from Brandon
  listening-set-ready   a new <platter>/step-N/clips.toml appeared (one per step)
  run-failed            the run ended failed or interrupted, or says "running" with no live process
  run-finished          the run ended done

Messages carry only the run id, the event, the step and dollars. Never corpus text, never
the ntfy topic. Quiet hours (22:00 to 08:00 local) hold EVERY push, failures included. Held
events go to pending.jsonl; the first pass at or after 08:00 sends one digest push and
clears it.

    python3 -I scripts/vi-bg-watch.py --run-id r32 --platter ~/vi/platter
    python3 -I scripts/vi-bg-watch.py --run-id r32 --dry-run     # write outbox.jsonl, send nothing
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TOPIC_FILE = "/home/brandon/Services/Docker/secrets/ntfy-topic"
QUIET_START_HOUR, QUIET_END_HOUR = 22, 8
HTTP_TIMEOUT = 10
CMD_TIMEOUT = 30
# paidrun/orchestrator.py ends a run as done, interrupted or failed; anything else is still going.
FAILED_STATUSES = ("failed", "interrupted")
SUCCESS_STATUS = "done"
EVENTS = ("needs-approval", "listening-set-ready", "run-failed", "run-finished")


def default_status_cmd():
    return f"python3 -I {REPO}/jobs/paid_run.py status --run-id {{run_id}} --json"


def in_quiet_hours(now):
    return now.hour >= QUIET_START_HOUR or now.hour < QUIET_END_HOUR


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def read_lines(path):
    out = []
    try:
        for ln in Path(path).read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def append_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def run_json(cmd):
    """Run a command, return its parsed JSON output, or None on any failure."""
    try:
        r = subprocess.run(shlex.split(cmd), capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=CMD_TIMEOUT, cwd=REPO)
        return json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def post_ntfy(topic_file, title, message, priority):
    """POST one push to ntfy.sh. True on success, False on any failure (kept for retry)."""
    try:
        topic = Path(topic_file).read_text().strip()
        if not topic:
            return False
        req = urllib.request.Request(
            "https://ntfy.sh/" + topic, data=message.encode(), method="POST",
            headers={"Title": title, "Priority": priority})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001 - a failed push must never crash the pass
        return False


def pending_approvals(agents):
    """Ids of pending permission requests from `paseo permit ls --json` (a list of requests).

    The field names of a non-empty list were not seen when this was written, so accept the
    likely ones. An item with none of them still counts, keyed by its position's JSON text.
    """
    if isinstance(agents, dict):
        agents = agents.get("permissions") or agents.get("pending") or []
    ids = []
    for it in agents if isinstance(agents, list) else []:
        if isinstance(it, dict):
            k = next((str(it[f]) for f in ("requestId", "request_id", "id", "agentId")
                      if it.get(f)), None)
            ids.append(k or json.dumps(it, sort_keys=True)[:80])
        else:
            ids.append(str(it))
    return ids


def listening_steps(platter):
    """Step numbers N that have a <platter>/step-N/clips.toml, sorted."""
    steps = []
    if platter:
        for p in Path(platter).glob("step-*/clips.toml"):
            n = p.parent.name[len("step-"):]
            if n.isdigit():
                steps.append(int(n))
    return sorted(steps)


def money(st):
    try:
        return "$%.2f" % st["spend"]["total"]
    except (KeyError, TypeError):
        return "$?"


def detect(run_id, st, approvals, steps):
    """(key, event, step, text, priority) for every event that currently holds."""
    out = []
    cost = money(st) if st else "$?"
    for a in approvals:
        out.append((f"needs-approval:{a}", "needs-approval", None,
                    f"run {run_id}: needs approval, {cost} spent", "high"))
    for n in steps:
        out.append((f"listening-set-ready:{run_id}:{n}", "listening-set-ready", n,
                    f"run {run_id}: listening set ready, step {n}, {cost} spent", "default"))
    if st:
        status = st.get("status")
        if status in FAILED_STATUSES or (status == "running" and st.get("process_alive") is False):
            why = status if status in FAILED_STATUSES else "process gone"
            code = st.get("exit_code")
            tail = f", exit {code}" if code is not None else ""
            out.append((f"run-failed:{run_id}:{why}", "run-failed", None,
                        f"run {run_id}: run failed ({why}{tail}), {cost} spent", "urgent"))
        elif status == SUCCESS_STATUS:
            out.append((f"run-finished:{run_id}", "run-finished", None,
                        f"run {run_id}: run finished, {cost} spent", "default"))
    return out


def build_card(run_id, st, approvals, steps, now):
    train = (st or {}).get("train") or {}
    return dict(
        run_id=run_id, state=(st or {}).get("status"), phase=(st or {}).get("phase"),
        spend=((st or {}).get("spend") or {}).get("total"), cap=(st or {}).get("cap"),
        cap_left=(st or {}).get("cap_left"), step=train.get("step"), total_steps=train.get("total"),
        eta_seconds=train.get("eta_seconds"),
        waiting_on_you=[f"approve request {a}" for a in approvals]
        + [f"listen to step {n}" for n in steps],
        last_pass=now.isoformat(timespec="seconds"))


def send(args, outbox, title, message, priority):
    if args.dry_run:
        append_line(outbox, dict(title=title, message=message, priority=priority))
        return True
    return post_ntfy(args.ntfy_topic_file, title, message, priority)


def digest_text(held):
    lines = [f"{h['text']}" for h in held]
    return f"{len(held)} held overnight:\n" + "\n".join(lines)


def one_pass(args):
    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    state_dir = Path(args.state_dir).expanduser()
    run_id = args.run_id
    if not run_id:  # an active run is announced by writing its id to state-dir/run-id
        try:
            run_id = (state_dir / "run-id").read_text().strip()
        except OSError:
            run_id = ""
    if not run_id:
        return 0  # idle: nothing to do, no network
    status_cmd = (args.status_cmd or default_status_cmd()).replace("{run_id}", run_id)
    st = run_json(status_cmd)
    if not isinstance(st, dict):
        return 0  # no status for this run: idle
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file, pending_file, outbox = (state_dir / n for n in
                                        ("state.json", "pending.jsonl", "outbox.jsonl"))
    state = read_json(state_file, {})
    seen = set(state.get("seen") or [])

    agents = run_json(args.agents_cmd)
    approvals = pending_approvals(agents) if agents is not None else []
    steps = listening_steps(args.platter)
    events = detect(run_id, st, approvals, steps)
    new = [e for e in events if e[0] not in seen]
    pending = read_lines(pending_file)

    if in_quiet_hours(now):
        for key, event, step, text, _ in new:  # hold everything, no exception
            append_line(pending_file, dict(key=key, event=event, step=step, text=text))
            seen.add(key)
    else:
        if pending:
            if send(args, outbox, "VI digest", digest_text(pending), "default"):
                pending_file.write_text("")
        for key, event, step, text, prio in new:
            if send(args, outbox, "VI " + event, text, prio):
                seen.add(key)  # a failed send stays unseen and is retried next pass

    state["seen"] = sorted(seen)
    write_json(state_file, state)
    write_json(state_dir / "status.json", build_card(run_id, st, approvals, steps, now))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", default="~/.cache/vi-bg")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--platter", help="directory holding step-N/clips.toml listening sets")
    ap.add_argument("--ntfy-topic-file", default=DEFAULT_TOPIC_FILE)
    ap.add_argument("--status-cmd", help="status command; {run_id} is replaced "
                    "(default: paid_run.py status --json)")
    ap.add_argument("--agents-cmd", default="paseo permit ls --json",
                    help="command printing pending permission requests as JSON")
    ap.add_argument("--now", help="ISO local time, for tests")
    ap.add_argument("--dry-run", action="store_true",
                    help="write would-send messages to state-dir/outbox.jsonl, send nothing")
    args = ap.parse_args(argv)
    try:
        return one_pass(args)
    except Exception as e:  # noqa: BLE001 - a timer pass must not crash noisily
        print(f"vi-bg-watch: pass failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
