"""Make the blind listening set from verified raw outputs.

    python3 blind.py <raw dir> <listen root> [--refs <refs dir>] [--seed 1] [--lines lines.json]

For every group (voice, line, seed) the models' clips are resampled to 24 kHz mono, level-matched
to one RMS within the group (so loudness cannot give a model away; the target is the lower of
RMS 0.07 and what keeps every clip's peak under 0.97), and given shuffled letters A, B, C...
(shuffle seeded, so the same inputs give the same key). Writes
`<listen root>/listen-blind/v<V>/<LINE>[_s<seed>]/<letter>.wav` and
`<listen root>/listen-blind/sheet.csv` (the voicearms sheet `vi review --listen` fills in: group,
clip, sentence, answer columns left empty) and `<listen root>/listen-key/key.json` (group -> letter
-> model; kept outside the served folder). Group names are voice + line only, never the model.
Re-running picks up every model folder under <raw dir>. Listen with:
`vi review --listen <root>/listen-blind --key <root>/listen-key/key.json --refs <refs>`. Do NOT open the key until the
ratings are saved. Raw files are never touched. Run verify.py first.
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wavio  # noqa: E402

SR = 24000
TARGET_RMS = 0.07
PEAK_MAX = 0.97
HERE = Path(__file__).resolve().parent
# Sheet columns, as crates/review/src/listen.rs expects (ANSWERS, then EXTRA).
SHEET_COLUMNS = ["group", "clip", "sentence", "still_her_voice_y_n", "mood_heard_0_3",
                 "words_clear_y_n", "notes", "accent", "suit_effect", "mood_right", "pitch"]
NAME = re.compile(r"^(?P<model>[a-z0-9]+)__v(?P<voice>[A-Za-z])__(?P<line>[A-Z]\d\d)__s(?P<seed>\d+)\.wav$")


def collect(raw):
    """{(voice, line, seed): {model: path}} from every <raw>/*/ folder."""
    groups = {}
    for d in sorted(p for p in Path(raw).iterdir() if p.is_dir()):
        for w in sorted(d.glob("*.wav")):
            m = NAME.match(w.name)
            if not m:
                raise SystemExit(f"unexpected file name {w}")
            key = (m["voice"], m["line"], int(m["seed"]))
            if m["model"] in groups.setdefault(key, {}):
                raise SystemExit(f"two clips for {key} / {m['model']}")
            groups[key][m["model"]] = w
    return groups


def group_name(voice, line, seed):
    return f"v{voice}/{line}" + ("" if seed == 1 else f"_s{seed}")


def level_match(clips):
    """Scale every clip in a group to one RMS. Returns scaled copies and the gain used."""
    rmss = [wavio.rms(c) for c in clips]
    if min(rmss) <= 0:
        raise SystemExit("a silent clip cannot be level-matched; run verify.py first")
    target = min([TARGET_RMS] + [PEAK_MAX / max(abs(v) for v in c) * r for c, r in zip(clips, rmss)])
    return [[v * target / r for v in c] for c, r in zip(clips, rmss)], target


def build(raw, root, seed=1, refs=None, lines=None):
    groups = collect(raw)
    text = {ln["id"]: ln["text"] for ln in json.loads(Path(lines or HERE / "lines.json").read_text())}
    for (_, line, _) in groups:
        if line not in text:
            raise SystemExit(f"no sentence for line {line} in the lines file")
    if not groups:
        raise SystemExit("no WAVs found")
    blind, keydir = Path(root) / "listen-blind", Path(root) / "listen-key"
    for d in (blind, keydir):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d} is not empty; refusing to overwrite a set that may already be rated")
        d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key, methods, sheet = {}, set(), []
    for (voice, line, sd) in sorted(groups):
        members = groups[(voice, line, sd)]
        models = sorted(members)
        rng.shuffle(models)
        clips = []
        for m in models:
            x, sr = wavio.read_wav(members[m])
            x, meth = wavio.resample(x, sr, SR)
            methods.add(meth)
            clips.append(x)
        scaled, _ = level_match(clips)
        gdir = blind / group_name(voice, line, sd)
        gdir.mkdir(parents=True)
        key[group_name(voice, line, sd)] = {}
        for i, (m, x) in enumerate(zip(models, scaled)):
            letter = chr(ord("A") + i)
            wavio.write_wav(gdir / f"{letter}.wav", x, SR)
            key[group_name(voice, line, sd)][letter] = m
            sheet.append([group_name(voice, line, sd), letter, text[line]] + [""] * (len(SHEET_COLUMNS) - 3))
    if refs:
        (blind / "refs").mkdir()
        for r in sorted(Path(refs).glob("v*.wav")):
            x, sr = wavio.read_wav(r)
            x, _ = wavio.resample(x, sr, SR)
            wavio.write_wav(blind / "refs" / r.name, x, SR)
    # Rows in letter order within each group; the shuffle is already in the letters.
    with open(blind / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLUMNS)
        w.writerows(sheet)
    (keydir / "key.json").write_text(json.dumps(
        {"shuffle_seed": seed, "resample_methods": sorted(methods), "groups": key}, indent=2) + "\n")
    return key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lines", type=Path, default=HERE / "lines.json")
    a = ap.parse_args(argv)
    key = build(a.raw, a.root, a.seed, a.refs, a.lines)
    print(f"blind set written: {len(key)} groups under {a.root}/listen-blind (key in listen-key/key.json: do not open yet)")


if __name__ == "__main__":
    main()
