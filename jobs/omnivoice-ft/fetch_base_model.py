"""Download the public Base model for the OmniVoice fine-tune onto the worker.

usage: fetch_base_model.py REPO REVISION DEST_ROOT

Prints the snapshot folder on stdout (everything else goes to stderr). The folder is
DEST_ROOT/<repo with / as -->-<first 12 of revision>. The download goes into DEST.part, is
retried (the hub client reuses completed files), is checked against a pinned size table, and
only then renamed to DEST. An existing DEST is reused only if it passes the same size check, so
a truncated file never reaches training. The Base is a public hub model, not game data.

Each try runs in a forked child process. r31: HF_HUB_DOWNLOAD_TIMEOUT is a per-read timeout, so
a host whose inbound trickles at a few KB/s (omnivoice-ft-dbg6, 2026-11-25) never trips it. The
parent watches DEST.part and ends the try with "inbound too slow" when it grew by less than
VI_FETCH_MIN_MBPS MB/s over the last VI_FETCH_STALL_WINDOW seconds; the try then counts as failed.

r32 WP4: every poll also prints `VI-PROGRESS base-model-fetch bytes-on-disk <N>` on stderr (N is
the most bytes seen in DEST.part so far, so it never goes down; a stalled download repeats it),
and `VI-PROGRESS-END base-model-fetch` when the download is over. on-worker.sh forwards these
home, where only a strictly increasing value counts as progress.

Environment (tests override these): VI_FETCH_BACKOFF="10 30 90" (seconds between the 4 tries),
VI_FETCH_MIN_MBPS=1 (decimal MB/s), VI_FETCH_STALL_WINDOW=120 (seconds).
"""

import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

# Set before huggingface_hub is imported; its default is 10 s, which a slow mirror trips.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# Pinned sizes for k2-fsa/OmniVoice at revision c5fdb5cc (bytes). Only the weights are pinned:
# they are the files a cut-off download truncates.
SIZE_TABLE = {
    "model.safetensors": 2_450_344_112,
    "audio_tokenizer/model.safetensors": 805_665_628,
}
TOTAL_BYTES = 3_267_470_260
DEFAULT_REVISION_PREFIX = "c5fdb5cc"
TRIES = 4


def size_problems(folder, revision):
    """Return a list of problems with the folder (empty means it passes)."""
    folder = Path(folder)
    problems = []
    if not (folder / "config.json").is_file():
        problems.append("no config.json")
    if not revision.startswith(DEFAULT_REVISION_PREFIX):
        return problems  # another revision: the size table does not apply
    for rel, want in SIZE_TABLE.items():
        p = folder / rel
        have = p.stat().st_size if p.is_file() else None
        if have != want:
            problems.append(f"{rel}: {have} bytes, expected {want}")
    if not problems:
        everything = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        if everything < TOTAL_BYTES:
            problems.append(f"folder holds {everything} bytes, expected at least {TOTAL_BYTES}")
    return problems


def _backoff():
    raw = os.environ.get("VI_FETCH_BACKOFF", "10 30 90").split()
    return [float(x) for x in raw]


def _stall_settings():
    window = float(os.environ.get("VI_FETCH_STALL_WINDOW", "120"))
    min_rate = float(os.environ.get("VI_FETCH_MIN_MBPS", "1")) * 1_000_000
    return window, min_rate


def tree_bytes(folder):
    """Bytes of every file under folder, the hub's hidden .incomplete files included."""
    total = 0
    for f in Path(folder).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


PROGRESS_STAGE = "base-model-fetch"
PROGRESS_COUNTER = "bytes-on-disk"
_progress_high = 0


def report_progress(have):
    """Print the WP4 progress line for `have` bytes on disk (the high mark, never lower)."""
    global _progress_high
    _progress_high = max(_progress_high, int(have))
    print(f"VI-PROGRESS {PROGRESS_STAGE} {PROGRESS_COUNTER} {_progress_high}", file=sys.stderr,
          flush=True)


def report_progress_end():
    print(f"VI-PROGRESS-END {PROGRESS_STAGE}", file=sys.stderr, flush=True)


def _one_try(repo, revision, part, conn):
    """The child: one snapshot_download; sends None or the error text to the parent."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, revision=revision, local_dir=str(part))
        conn.send(None)
    except BaseException as e:  # noqa: BLE001 - any hub/network error is reported, then retried
        conn.send(f"{type(e).__name__}: {e}")


def download_once(repo, revision, part):
    """One try in a child process: None when snapshot_download returned, else the problem. The
    child is stopped when DEST.part grows slower than the floor over the whole window."""
    window, min_rate = _stall_settings()
    poll = max(0.05, min(5.0, window / 4))
    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_one_try, args=(repo, revision, part, send), daemon=True)
    child.start()
    send.close()
    samples = [(time.monotonic(), tree_bytes(part))]
    report_progress(samples[0][1])
    try:
        while True:
            child.join(poll)
            if not child.is_alive():
                break
            now, have = time.monotonic(), tree_bytes(part)
            samples.append((now, have))
            report_progress(have)
            old = [i for i, (t, _) in enumerate(samples) if now - t >= window]
            if old:
                samples = samples[old[-1]:]
                t0, b0 = samples[0]
                rate = (have - b0) / (now - t0)
                if rate < min_rate:
                    return (f"inbound too slow: {rate / 1_000_000:.3f} MB/s over the last "
                            f"{now - t0:.0f} s (floor {min_rate / 1_000_000:g} MB/s, "
                            f"{have} bytes in {part.name} so far); stopped this try")
        if recv.poll():
            return recv.recv()
        return f"the download process exited {child.exitcode} without a result"
    except (EOFError, OSError) as e:
        return f"the download process gave no result: {type(e).__name__}: {e}"
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
            if child.is_alive():
                child.kill()
                child.join(10)
        recv.close()


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    repo, revision, root = argv
    dest = Path(root) / f"{repo.replace('/', '--')}-{revision[:12]}"
    if dest.is_dir():
        problems = size_problems(dest, revision)
        if not problems:
            print(f"fetch_base_model: reusing {dest}", file=sys.stderr)
            print(dest)
            return 0
        print(f"fetch_base_model: {dest} fails the size check ({'; '.join(problems)}); "
              "downloading again", file=sys.stderr)
        shutil.rmtree(dest)
    part = dest.with_name(dest.name + ".part")
    print(f"fetch_base_model: downloading {repo}@{revision} to {part}", file=sys.stderr)
    try:
        return _download(repo, revision, dest, part)
    finally:
        report_progress_end()


def _download(repo, revision, dest, part):
    delays = _backoff()
    last = "no attempt"
    for attempt in range(TRIES):
        part.mkdir(parents=True, exist_ok=True)
        last = download_once(repo, revision, part)
        if last is None:
            problems = size_problems(part, revision)
            if not problems:
                os.replace(part, dest)
                print(dest)
                return 0
            last = "; ".join(problems)
        print(f"fetch_base_model: try {attempt + 1}/{TRIES} failed: {last}", file=sys.stderr)
        if attempt < TRIES - 1:
            time.sleep(delays[min(attempt, len(delays) - 1)])
    print(f"worker-preflight: base model download failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
