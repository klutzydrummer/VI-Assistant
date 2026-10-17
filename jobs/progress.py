"""How far a running stage has got, as percentages, from its log.

    python3 jobs/progress.py corpus/tali            # once
    python3 jobs/progress.py corpus/tali --watch 60  # every 60 s, with an ETA

Every job logs `<pass>: <done>/<total>` after each chunk (vi_job's
checkpoints), and the runner writes a stage's output, the GPU worker's
included, to stages/<stage>/.tmp-<key>/log.txt while it runs. This reads
those logs, so it works for any stage, needs nothing from the job, and
doesn't change any stage's key. Standard library only.

A stage counts as running while its log changed in the last 10 minutes; an
older .tmp- folder is a run that failed or was stopped (a rerun resumes it).
A stage with several passes (embed runs one per model) shows each pass;
`--watch` estimates the time left in the current pass from its rate between
reads.
"""

import argparse
import re
import sys
import time
from pathlib import Path

LINE = re.compile(r"^([\w.-]+): (\d+)/(\d+)$")
RUNNING_S = 600


def passes(log):
    """{pass: (done, total)} in the order the passes started."""
    out = {}
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m[1]] = (int(m[2]), int(m[3]))
    return out


def logs(corpus):
    for log in sorted((corpus / "stages").glob("*/.tmp-*/log.txt")):
        yield log.parent.parent.name, log


def report(corpus, previous=None, interval=None):
    """Prints one block; returns {(stage, pass): done} for the next read's rate."""
    seen, now, any_log = {}, time.time(), False
    for stage, log in logs(corpus):
        any_log = True
        age = now - log.stat().st_mtime
        state = "running" if age < RUNNING_S else f"stopped {age / 60:.0f} min ago (failed or killed; a rerun resumes)"
        found = passes(log)
        print(f"{stage}: {state}")
        if not found:
            print("  starting (no progress logged yet)")
        for i, (name, (done, total)) in enumerate(found.items()):
            pct = 100 * done / total if total else 100.0
            line = f"  {name}: {done}/{total} ({pct:.0f}%)"
            current = i == len(found) - 1 and state == "running"
            if current and previous and interval and (stage, name) in previous:
                rate = (done - previous[(stage, name)]) / interval
                if rate > 0:
                    line += f", about {(total - done) / rate / 60:.0f} min left in this pass"
            print(line + ("  <- now" if current else ""))
            seen[(stage, name)] = done
    if not any_log:
        print("no stage is running")
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path, help="the corpus folder, e.g. corpus/tali")
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="read again every SECONDS")
    a = ap.parse_args()
    if not (a.corpus / "stages").is_dir():
        sys.exit(f"{a.corpus}: no stages/ folder")
    seen = report(a.corpus)
    while a.watch:
        time.sleep(a.watch)
        print(f"--- {time.strftime('%H:%M:%S')}")
        seen = report(a.corpus, seen, a.watch)


if __name__ == "__main__":
    main()
