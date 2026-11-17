"""Download the public Base model for the OmniVoice fine-tune onto the worker.

usage: fetch_base_model.py REPO REVISION DEST_ROOT

Prints the snapshot folder on stdout (everything else goes to stderr). The folder is
DEST_ROOT/<repo with / as -->-<first 12 of revision>; one that already holds config.json is
reused, so a rerun costs nothing. The Base is a public hub model, not game data.
"""

import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    repo, revision, root = argv
    dest = Path(root) / f"{repo.replace('/', '--')}-{revision[:12]}"
    if (dest / "config.json").is_file():
        print(f"fetch_base_model: reusing {dest}", file=sys.stderr)
    else:
        from huggingface_hub import snapshot_download
        print(f"fetch_base_model: downloading {repo}@{revision} to {dest}", file=sys.stderr)
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo, revision=revision, local_dir=str(dest))
        if not (dest / "config.json").is_file():
            print(f"fetch_base_model: {dest} has no config.json after the download", file=sys.stderr)
            return 1
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
