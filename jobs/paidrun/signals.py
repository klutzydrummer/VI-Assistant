"""Prose regexes and pure parsers for worker, vi and Vast output (no I/O)."""

import json
import re


# Every failed cheap gate of jobs/worker-preflight.sh prints `worker-preflight: <what failed>`
# (disk, RAM, CUDA, VRAM, cores, missing tools). They are facts about one offer, so the next offer
# may pass; the success line `worker-preflight: ok` is not a failure. The push rate is no gate
# (round 19): its `worker-preflight WARNING:` / `worker-preflight push:` lines never match.
PREFLIGHT_GATE_RE = re.compile(r"worker-preflight: (?!ok\b)\S")
# on-worker.sh prints this when the timed 8 MB probe says the inputs would take longer than
# max_upload_minutes: also a per-offer fact (this worker's line), also before anything is sent.
UPLOAD_GATE_TEXT = "worker upload ETA"


# on-worker.sh names the worker stage's own exit code; the vi CLI wrapping it exits 1 regardless.
WORKER_RC_RE = re.compile(r"on-worker\.sh: \S+ exited with (\d+) on the worker")
# on-worker.sh's EXIT trap names its own non-zero exit (r18: ssh's 255 under set -e), and the vi
# runner's error names the stage command's exit status (`stage X failed: exit status: N`).
ON_WORKER_EXIT_RE = re.compile(r"on-worker\.sh: exiting with (\d+)")
STAGE_STATUS_RE = re.compile(r"failed: exit status: (\d+)")


def failure_rc_text(rc, out):
    """How a failed stage's exit is reported: the worker stage's exit code when on-worker.sh
    printed it (r8: 126, a bad interpreter), else on-worker.sh's own exit code (r18: 255, ssh
    lost the host), else the stage command's exit status from the vi runner's error, each with
    the vi CLI's own code beside it; else the vi CLI's code."""
    out = out or ""
    for regex, where in ((WORKER_RC_RE, "on the worker"), (ON_WORKER_EXIT_RE, "from on-worker.sh"),
                         (STAGE_STATUS_RE, "from the stage command")):
        m = regex.findall(out)
        if m and m[-1] != str(rc):
            return f"exit {m[-1]} {where}; the vi CLI exited {rc}"
    return f"exit {rc}"


# A host drop (r18): the worker's ssh ended from its side or nothing listens any more. ssh's own
# exit 255 or one of these lines is only a suspicion; host_drop() confirms it with a status check.
# r31: dbg6's host vanished with only `Connection closed by <ip> port <n>`, the return tunnel's
# `tunnel down (ssh exit 255)`, and on-worker.sh's own give-up lines, and the vi CLI exited 1.
HOST_DROP_RE = re.compile(r"closed by remote host|Connection refused|Connection reset by peer|"
                          r"Connection timed out|No route to host|kex_exchange_identification|"
                          r"Broken pipe|connection unexpectedly closed|"
                          r"Connection closed by \S+ port \d+|tunnel down \(ssh exit 255\)|"
                          r"giving up after \d+ min without a successful poll|"
                          r"could not re-prove the return path")
DROP_EXIT_RE = re.compile(r"(?:exited with|exiting with|exit status:) 255\b")


def drop_signal(rc, out):
    """The first sign that ssh lost the host, or None. Only ever applied to work that did NOT
    report its own result: a stage or gate that printed FAIL failed our own check, never a drop."""
    out = out or ""
    m = HOST_DROP_RE.search(out)
    if m:
        return m.group(0)
    if rc == 255 or DROP_EXIT_RE.search(out):
        return "ssh exit 255"
    return None


# r31 F4: this run's stage key. debug-pass.sh names the worker dir
# `omnivoice_debug.<key>.<attempt>` in its stage log paths; the vi runner's own
# `omnivoice_debug: running (<first 12 of key>)` line is the fallback.
DEBUG_WORK_RE = re.compile(r"omnivoice_debug\.([0-9a-f]{12,})\.(\d{8}T\d{6}Z)")
DEBUG_RUNNING_RE = re.compile(r"omnivoice_debug: running \(([0-9a-f]{12,})\)")


def debug_run_key(out):
    """(stage key or its 12-character prefix, attempt or None) named in this run's output, or
    (None, None)."""
    out = out or ""
    m = DEBUG_WORK_RE.findall(out)
    if m:
        return m[-1]
    m = DEBUG_RUNNING_RE.findall(out)
    return (m[-1], None) if m else (None, None)


def retryable_gate(out):
    """The gate text that makes a failed preflight stage retry on another offer, or None."""
    if UPLOAD_GATE_TEXT in out:
        return "upload-ETA"
    if PREFLIGHT_GATE_RE.search(out):
        return "worker-preflight"
    return None


def gate_kind(out):
    """How a failed worker gate should mark the machine: 'cuda' (machine and host), 'memory'
    (VRAM/RAM: raise the next VRAM floor too), or 'other' (machine only)."""
    if CUDA_GATE_RE.search(out):
        return "cuda"
    if MEMORY_GATE_RE.search(out):
        return "memory"
    return "other"


OFFER_RE = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+(\d+) GB\s+\$([\d.]+)/hr\s+reliability ([\d.]+)\s+"
    r"net down\s+(\d+) / up\s+(\d+) Mbps\s+([\d.]+) cores\s+(\d+) GB ram\s+"
    r"disk (\d+) MB/s\s*(.*)$")

# Trainer output the train stage streams (train_ft.py runs upstream's OmniTrainer, which logs
# through tqdm/accelerate; save_steps is 500 in train_ft.py). Two shapes are read, the one with
# the larger total wins when a line holds both an epoch bar and a step bar:
#   tqdm bar:   "  3%|▎  | 150/5000 [02:10<1:10:00,  1.15it/s]"
#   plain text: "Step 150/5000" / "step: 150 of 5000"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[\d+:\d+")
STEP_RE = re.compile(r"\b[Ss]teps?\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)")
# on-worker.sh prints this on home once a checkpoint is verified at home and acked to the worker.
ACK_RE = re.compile(r"on-worker\.sh: checkpoint (\d+) verified home and acked")

# WP4: worker stages report a monotonic counter as `VI-PROGRESS <stage> <counter> <integer>`
# and `VI-PROGRESS-END <stage>` once the stage is over. Only these exact shapes count; anything
# else (a partial line, a float, extra words) is ignored, never guessed at.
PROGRESS_RE = re.compile(r"^VI-PROGRESS ([a-z0-9][a-z0-9-]*) ([a-z0-9][a-z0-9-]*) (\d{1,19})\s*$")
PROGRESS_END_RE = re.compile(r"^VI-PROGRESS-END ([a-z0-9][a-z0-9-]*)\s*$")


def parse_progress(line):
    """(stage, counter, value) from one `VI-PROGRESS` line, or None for anything malformed."""
    m = PROGRESS_RE.match(line.strip("\r\n") if isinstance(line, str) else "")
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def parse_progress_end(line):
    """The stage named by one `VI-PROGRESS-END` line, or None."""
    m = PROGRESS_END_RE.match(line.strip("\r\n") if isinstance(line, str) else "")
    return m.group(1) if m else None


MIN_SAMPLES = 3


# r7: the tokeniser's pool died (pthread_create failed -> BrokenProcessPool) and the stage then
# printed nothing for hours. A fatal line followed by this long with no new output ends the
# preflight as a non-retryable failure (fork F2: a crash is a code or limits bug, not a bad
# offer). The same words as VI_FATAL_RE in jobs/worker-limits.sh. PAID_RUN_FATAL_QUIET_SECONDS
# overrides the wait for tests.
FATAL_RE = re.compile(r"BrokenProcessPool|pthread_create failed|can't start new thread|Traceback")
BOOT_FAIL_RE = re.compile(r"no ssh after|disappeared while booting|failed to start|boot (?:timeout|failure)",
                          re.I)


CUDA_GATE_RE = re.compile(r"worker-preflight: .*\b(CUDA|driver)\b", re.I)
MEMORY_GATE_RE = re.compile(r"worker-preflight: (?:vram peak \S+ of \S+ GiB|.*\b(?:VRAM|RAM)\b)", re.I)


def parse_step(line):
    """(step, total) from one trainer output line, or None."""
    found = [(int(m.group(1)), int(m.group(2))) for rx in (TQDM_RE, STEP_RE)
             for m in rx.finditer(line)]
    found = [f for f in found if f[1] > 0 and f[0] <= f[1]]
    return max(found, key=lambda f: f[1]) if found else None


def eta_seconds(samples, step, total):
    """Rough seconds left from the recent [time, step] samples; None until MIN_SAMPLES."""
    if not samples or len(samples) < MIN_SAMPLES or not total or step is None:
        return None
    (t0, s0), (t1, s1) = samples[0], samples[-1]
    if t1 <= t0 or s1 <= s0:
        return None
    return max(0.0, (total - step) * (t1 - t0) / (s1 - s0))


MACHINE_TAIL_RE = re.compile(r"\s+machine (\d+) host (\d+|\?)\s*$")


def parse_offers(text):
    """[{id, gpu, dph, up, cores, ram, loc, sig, machine, host}] in the order `vi cloud offers`
    printed them. The line ends with `machine N host M` (host `?` when Vast gave none); older
    output without it gives machine/host None."""
    out = []
    for line in text.splitlines():
        m = OFFER_RE.match(line)
        if not m:
            continue
        loc = m.group(11).strip()
        tail = MACHINE_TAIL_RE.search(loc)
        machine = host = None
        if tail:
            machine = int(tail.group(1))
            host = None if tail.group(2) == "?" else int(tail.group(2))
            loc = loc[:tail.start()].strip()
        out.append(dict(
            id=int(m.group(1)), gpu=m.group(2).strip(), dph=float(m.group(4)),
            up=int(m.group(7)), cores=m.group(8), ram=m.group(9), loc=loc,
            machine=machine, host=host,
            # A relisted machine is also recognised by what it is and where it is, for output
            # that carries no machine id.
            sig=f"{m.group(2).strip()}|{loc}|{m.group(8)}|{m.group(9)}|{m.group(7)}"))
    return out


# r21: SKIP-no-time = debug-pass.sh did not start the stage because the rental's hours ran out;
# it is shown apart from FAIL (the stage itself was never tried).
DEBUG_LINE_RE = re.compile(
    r"debug-pass: stage (\d+) ([\w-]+): (PASS|FAIL|SKIP-dependent|SKIP-no-time)(?=\s|$) ?(.*)")
DEBUG_EXCERPT_RE = re.compile(r"debug-pass:   \| (.*)")


def parse_vast_instance(text):
    """{"status", "msg"} from `vastai show instance <id> --raw` JSON (a dict, a one-item list or
    {"instances": ...}); empty when it cannot be read."""
    try:
        doc = json.loads(text)
    except ValueError:
        return {}
    stack = [doc]
    while stack:
        d = stack.pop(0)
        if isinstance(d, dict):
            if "actual_status" in d or "status_msg" in d:
                st = d.get("actual_status")
                msg = d.get("status_msg")
                return dict(status=str(st) if st is not None else "none",
                            msg=" ".join(str(msg).split())[:300] if msg else None)
            stack += list(d.values())
        elif isinstance(d, list):
            stack += d
    return {}


# r27: what Vast prints for an instance it no longer has (omnivoice-ft-dbg5, 2026-11-24): `show
# instance --raw` exits 0 with {"instances": null}, and `logs` exits 0 with the docker daemon's
# "No such container: C.<id>". Either one, from a call that succeeded, means the instance is gone.
NO_CONTAINER_RE = re.compile(r"No such container")
VAST_GONE = "gone"
# Statuses that mean the container is up or coming up: ssh refused then is not a confirmed drop.
VAST_ALIVE = ("running", "loading")


def vast_instance_gone(show_rc, show_text, logs_rc, logs_text):
    """r27: a short reason when Vast's own answers say the instance no longer exists, else None.
    Only calls that succeeded (exit 0) count. `show instance` must have answered with JSON that
    has no instance record: {"instances": null}, an empty list, or null; anything that reports an
    actual_status, or looks like an API error, is never "gone". The logs line "No such container"
    counts only next to a successful show answer without a status, so a failing Vast API is never
    read as a lost host."""
    if show_rc != 0:
        return None
    try:
        doc = json.loads(show_text or "")
    except ValueError:
        return None
    if parse_vast_instance(show_text):
        return None  # Vast still lists the instance with a status
    if isinstance(doc, dict):
        if any(k in doc for k in ("error", "msg", "success")):
            return None  # an API error answer, not a statement about the instance
        empty = "instances" in doc and doc["instances"] in (None, [], {})
    else:
        empty = doc in (None, [])
    no_container = logs_rc == 0 and bool(NO_CONTAINER_RE.search(logs_text or ""))
    reasons = (["Vast lists no such instance (instances: null)"] if empty else []) + (
        ["its logs say No such container"] if no_container else [])
    return "; ".join(reasons) or None


def parse_debug_output(text):
    """{n: (result, detail, [excerpt lines])} from the `debug-pass: stage N ...` lines of a
    command's output. The last status line of a stage wins; the excerpt lines that follow a
    status line belong to it."""
    found, cur = {}, None
    for line in (text or "").splitlines():
        m = DEBUG_LINE_RE.search(line)
        if m:
            cur = int(m.group(1))
            found[cur] = (m.group(3), m.group(4).strip(), [])
            continue
        m = DEBUG_EXCERPT_RE.search(line)
        if m and cur in found:
            found[cur][2].append(m.group(1))
            continue
        if "debug-pass: stage" in line:  # a "running" or "still running" line ends an excerpt
            cur = None
    return found


PUSH_PROBE_JSON_RE = re.compile(r"push-probe: json (\{.*\})\s*$")


def parse_push_probe(text):
    """The transfer-speed probe's result (jobs/push-probe.sh's last `push-probe: json` line), or
    None when the output has none or it does not parse."""
    found = None
    for line in (text or "").splitlines():
        m = PUSH_PROBE_JSON_RE.search(line)
        if m:
            try:
                found = json.loads(m.group(1))
            except ValueError:
                pass
    return found if isinstance(found, dict) else None


def push_probe_lines(probe):
    """Summary lines for the probe, one per leg, with MB/s (decimal) first."""
    if not probe:
        return ["  push probe    : no result (it runs only after a passed worker preflight; see "
                "the run log for `push-probe:` lines)"]
    legs = probe.get("legs") or {}

    def rate(v):
        return "?" if v is None else f"{v:g}"

    lines = [f"  push probe    : {probe.get('seconds', '?')} s of a {probe.get('cap_seconds', '?')} "
             "s cap; random bytes only; MB/s is decimal"]
    a = legs.get("a") or {}
    lines.append(f"    a tunnel-rsync : {rate(a.get('mb_s'))} MB/s, {a.get('status', '?')} "
                 f"({a.get('bytes', 0)} of {a.get('planned', '?')} bytes)")
    b = legs.get("b") or {}
    per = ", ".join(rate(x) for x in b.get("per_stream_mb_s") or [])
    lines.append(f"    b 4-streams    : {rate(b.get('mb_s'))} MB/s in all, {b.get('status', '?')}"
                 + (f" (per stream {per})" if per else ""))
    c = legs.get("c") or {}
    if c.get("status") == "skip":
        lines.append("    c public-curl  : skip (no time left)")
    else:
        up = ("not measured (no sink set)" if c.get("up_status") in (None, "none") else
              f"{rate(c.get('up_mb_s'))} MB/s, {c.get('up_status')} ({c.get('up_url')})")
        lines.append(f"    c public-curl  : download {rate(c.get('down_mb_s'))} MB/s, "
                     f"{c.get('down_status', '?')} ({c.get('down_url', '?')}); upload {up}")
    return lines


def debug_excerpt_keep(lines, n=12):
    """The excerpt kept for a debug stage: debug-pass.sh's "key: " lines (r23: the trainer's own
    error and accelerate's "exit status N", printed on FAIL before the tail) always, then the last
    n of the other lines."""
    lines = list(lines)
    key = [x for x in lines if x.startswith("key: ")]
    rest = [x for x in lines if not x.startswith("key: ")]
    return key + rest[-n:]
