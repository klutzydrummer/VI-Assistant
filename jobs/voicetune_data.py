"""Build the voicetune_data stage's out/ folder at home, label-blind.

One home CPU job that produces everything the two GPU stages
(voicetune_preflight / voicetune_train) read through VI_INPUT_VOICETUNE_DATA:

  train_with_codes.jsonl   her clips, codec-encoded (buildset's label-blind split)
  val_with_codes.jsonl     the held-out ~10% of conversations, same shape
  val100_with_codes.jsonl  100-clip seeded sample of val, for a quick smoke
  replay_with_codes.jsonl  Base's own lines in public voices, codec-encoded
  wav/                     her 24 kHz mono clips, hard-linked (never re-encoded)
  voices/                  the public LibriTTS-R reference voices
  summary.json             split and drop counts, plus the replay source

Every `audio`/`ref_audio` path is relative to the out/ folder, so the folder
is relocatable and on-worker.sh packs it whole. The buildset split reads only
game and conversation ids (jobs/buildset.py); no mood label, rating or
hand-labelled set is consulted.

Run under the qwen-tts environment (jobs/voicetune_data.sh): the buildset part
is stdlib, prepare needs qwen-tts + torch + soundfile. Replay codes were made
by Base before the training fixes and are reused as-is; their spoken wavs
never came home, but the loader reads the precomputed `audio_codes`, so the
dangling `audio` path is never opened (validate() skips it for coded rows).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import buildset
import voicetune

HOME = Path.home()
CACHE = HOME / ".cache" / "vi" / "voicetune"
REPLAY_SRC = CACHE / "fresh" / "data" / "replay_with_codes.jsonl"
VOICES_SRC = CACHE / "abrun" / "libri" / "voices"
REUSE_WAV = CACHE / "data-v2" / "wav"


def link_or_copy(src, dst):
    """Hard-link src into dst, falling back to a copy across devices."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_raw(out, repo, reuse_wav):
    """buildset's label-blind split into out/ (raw manifests, hard-linked wavs, summary)."""
    buildset.main(["--out", str(out), "--repo", str(repo), "--reuse-wav", str(reuse_wav)])


def write_replay(out, src=REPLAY_SRC, voices=VOICES_SRC):
    """Copy the reference voices into out/voices/ and rewrite the replay rows'
    ref_audio to `voices/<file>`. `audio` stays `replay/wav/replay-*.wav`: the
    spoken clip never came home, but its codes are already in the row."""
    vdir = out / "voices"
    vdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
    for r in rows:
        name = Path(r["ref_audio"]).name
        dst = vdir / name
        if not dst.exists():
            src_v = Path(voices) / name
            if not src_v.exists():
                raise SystemExit(f"voicetune_data: replay voice {src_v} is missing")
            link_or_copy(src_v, dst)
        r["ref_audio"] = f"voices/{name}"
    voicetune.write_jsonl(out / "replay_with_codes.jsonl", rows)
    return rows


def prepare(out, stem, device, batch):
    """voicetune.py prepare on one raw manifest, writing <stem>_with_codes.jsonl.

    Runs with out/ as the working directory so the manifest's `wav/<id>.wav`
    paths resolve (they are relative to the manifest's folder, which is out/)."""
    a = argparse.Namespace(raw=str(out / f"{stem}_raw.jsonl"), out=out,
                           tokenizer=voicetune.TOKENIZER, device=device, batch=batch)
    voicetune.cmd_prepare(a)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path(os.environ.get("VI_OUT", ".")),
                   help="output folder (the stage's out/; default $VI_OUT)")
    p.add_argument("--repo", type=Path, default=buildset.REPO,
                   help="repo checkout holding corpus/tali (default: this repo)")
    p.add_argument("--reuse-wav", type=Path, default=REUSE_WAV,
                   help="folder of existing wavs to hard-link from")
    p.add_argument("--replay", type=Path, default=REPLAY_SRC,
                   help="codec-encoded replay manifest (label-free, public voices)")
    p.add_argument("--voices", type=Path, default=VOICES_SRC,
                   help="the reference voices named by the replay rows")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args(argv)

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    build_raw(out, a.repo, a.reuse_wav)

    # prepare (and everything that opens the manifest's relative paths) runs
    # from inside out/, where wav/ sits beside the manifests.
    os.chdir(out)
    for stem in ("train", "val", "val100"):
        prepare(out, stem, a.device, a.batch)
        (out / f"{stem}_raw.jsonl").unlink(missing_ok=True)
    replay = write_replay(out, a.replay, a.voices)

    summary = json.loads((out / "summary.json").read_text())
    summary["replay"] = len(replay)
    summary["replay_voices"] = sorted(p.name for p in (out / "voices").glob("*.wav"))
    summary["replay_source"] = str(a.replay)
    summary["replay_note"] = ("Base's own lines in public LibriTTS-R voices, codec-encoded "
                              "before the training fixes; codes reused, spoken wavs not kept.")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
