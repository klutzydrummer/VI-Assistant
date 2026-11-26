#!/usr/bin/env python3
"""How many bytes of a checkpoint push have reached home, for the stage status.

`push_progress.py DIR [TOTAL_BYTES]` prints one line, e.g.
    received home 1234.5 MB of 7382.5 MB (16.7%)
DIR is home's receive dir (the durable vi-publish dir). Every regular file under it counts, the
rsync partial files (.rsync-partial) and the `.vichunk-` pieces of the parallel push included, so
a half-sent file shows up as it grows. Symlinks are not followed. MB is decimal (1e6), like the
push rates elsewhere. TOTAL_BYTES is optional; without it (or 0) the line has no "of".
"""
import os
import stat
import sys


def received_bytes(root):
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(base, name))
            except OSError:
                continue  # a file renamed or removed between the listing and the stat
            if stat.S_ISREG(st.st_mode):  # lstat: a symlink is not a regular file here
                total += st.st_size
    return total


def progress_line(root, total=0):
    got = received_bytes(root)
    text = f"received home {got / 1e6:.1f} MB"
    if total and total > 0:
        text += f" of {total / 1e6:.1f} MB ({min(100.0, 100.0 * got / total):.1f}%)"
    return text


def main(argv):
    if len(argv) < 2:
        print("usage: push_progress.py DIR [TOTAL_BYTES]", file=sys.stderr)
        return 2
    try:
        total = int(argv[2]) if len(argv) > 2 else 0
    except ValueError:
        total = 0
    print(progress_line(argv[1], total))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
