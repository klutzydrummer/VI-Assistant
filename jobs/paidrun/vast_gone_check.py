#!/usr/bin/env python3
"""One read-only Vast check for on-worker.sh's early host-drop stop (VI_GONE_CHECK_CMD).

    vast_gone_check.py INSTANCE_ID

Runs `vastai show instance ID --raw` and `vastai logs ID --tail 5` once (VI_VASTAI_BIN overrides
the vastai command; the Vast key and its config files are never read here, only the CLI runs) and
asks the same judge the host-drop check uses, signals.vast_instance_gone. Exit 0: Vast says the
instance is gone. Exit 1: not gone, still listed with a status, or the answer could not be read
(our failure to ask is never evidence about the host). Exit 2: bad usage. Prints one line.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paidrun.signals import vast_instance_gone  # noqa: E402

SECONDS = 30


def ask(vastai, args):
    try:
        r = subprocess.run([vastai, *args], stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=SECONDS, env={**os.environ, "PAGER": "cat"})
        return r.returncode, r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    except (subprocess.TimeoutExpired, OSError):
        return None, ""


def main(argv):
    if len(argv) != 2 or not argv[1].isdigit():
        print("usage: vast_gone_check.py INSTANCE_ID", file=sys.stderr)
        return 2
    vastai = os.environ.get("VI_VASTAI_BIN") or "vastai"
    show_rc, show = ask(vastai, ["show", "instance", argv[1], "--raw"])
    if show_rc is None:
        print("vast-check: could not read the instance status; not treating it as gone")
        return 1
    logs_rc, logs = ask(vastai, ["logs", argv[1], "--tail", "5"])
    why = vast_instance_gone(show_rc, show, logs_rc, logs)
    print(f"vast-check: instance {argv[1]} {'is gone: ' + why if why else 'is not reported gone'}")
    return 0 if why else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
