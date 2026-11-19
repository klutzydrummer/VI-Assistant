"""Download the public Base model for the OmniVoice fine-tune onto the worker.

usage: fetch_base_model.py REPO REVISION DEST_ROOT

Prints the snapshot folder on stdout (everything else goes to stderr). The folder is
DEST_ROOT/<repo with / as -->-<first 12 of revision>. The download goes into DEST.part, is
retried (the hub client reuses completed files), is checked against a pinned size table, and
only then renamed to DEST. An existing DEST is reused only if it passes the same size check, so
a truncated file never reaches training. The Base is a public hub model, not game data.

Environment (tests override these): VI_FETCH_BACKOFF="10 30 90" (seconds between the 4 tries).
"""

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
    from huggingface_hub import snapshot_download
    print(f"fetch_base_model: downloading {repo}@{revision} to {part}", file=sys.stderr)
    delays = _backoff()
    last = "no attempt"
    for attempt in range(TRIES):
        part.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo, revision=revision, local_dir=str(part))
            problems = size_problems(part, revision)
            if not problems:
                os.replace(part, dest)
                print(dest)
                return 0
            last = "; ".join(problems)
        except Exception as e:  # noqa: BLE001 - any hub/network error is retried
            last = f"{type(e).__name__}: {e}"
        print(f"fetch_base_model: try {attempt + 1}/{TRIES} failed: {last}", file=sys.stderr)
        if attempt < TRIES - 1:
            time.sleep(delays[min(attempt, len(delays) - 1)])
    print(f"worker-preflight: base model download failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
