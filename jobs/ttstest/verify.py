"""Home verifier for the pushed TTS test outputs. Nonzero exit and a named culprit on any problem.

    python3 verify.py <raw dir> [--expect omnivoice:T:1,omnivoice:P:1,...] [--min-free-gb 5]

`<raw dir>` holds one folder per (model, voice, seed), as generate.py wrote them. For each folder:
manifest.sha256 present and every hash matches; no extra or missing WAV compared with the
manifest; the WAV count equals expected-count.txt; run-info.json says ok; and every WAV passes
sample rate 16-48 kHz, duration 1.5-60 s, RMS > 0.005, no NaN/Inf, under 1% of samples clipped,
not constant. With --expect, every listed folder must exist (a missing one is a culprit) and no
unlisted folder may be there.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wavio  # noqa: E402

SR_MIN, SR_MAX = 16000, 48000
DUR_MIN, DUR_MAX = 1.5, 60.0
RMS_MIN = 0.005
CLIP_FRAC_MAX = 0.01


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_wav(path):
    """List of problems with one WAV (empty means it passes)."""
    try:
        x, sr = wavio.read_wav(path)
    except Exception as e:
        return [f"unreadable: {e}"]
    probs = []
    if not x:
        return ["no samples"]
    if not (SR_MIN <= sr <= SR_MAX):
        probs.append(f"sample rate {sr} outside {SR_MIN}-{SR_MAX}")
    dur = len(x) / sr
    if not (DUR_MIN <= dur <= DUR_MAX):
        probs.append(f"duration {dur:.2f}s outside {DUR_MIN}-{DUR_MAX}s")
    if any(not math.isfinite(v) for v in x):
        probs.append("contains NaN/Inf")
        return probs
    if wavio.rms(x) <= RMS_MIN:
        probs.append(f"silent or near-silent (RMS {wavio.rms(x):.5f})")
    if sum(1 for v in x if abs(v) >= 0.999) / len(x) >= CLIP_FRAC_MAX:
        probs.append("clipped on 1% or more of samples")
    if max(x) == min(x):
        probs.append("constant signal")
    return probs


def parse_manifest(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            h, _, name = line.partition("  ")
            out[name.strip()] = h.strip()
    return out


def verify_dir(d):
    """List of (file, problem) for one (model, voice, seed) folder."""
    d = Path(d)
    probs = []
    man, cnt, info = d / "manifest.sha256", d / "expected-count.txt", d / "run-info.json"
    if info.is_file():
        try:
            ri = json.loads(info.read_text())
            if not ri.get("ok"):
                probs.append(("run-info.json", f"the run reported failure: {ri.get('error', 'no reason given')}"))
        except ValueError:
            probs.append(("run-info.json", "not valid JSON"))
    else:
        probs.append(("run-info.json", "missing"))
    if not man.is_file():
        probs.append(("manifest.sha256", "missing (the run did not finish, or the push was partial)"))
        return probs
    listed = parse_manifest(man.read_text())
    on_disk = {p.name for p in d.glob("*.wav")}
    for n in sorted(set(listed) - on_disk):
        probs.append((n, "listed in manifest but missing on disk"))
    for n in sorted(on_disk - set(listed)):
        probs.append((n, "on disk but not in manifest"))
    for n in sorted(set(listed) & on_disk):
        if sha256_file(d / n) != listed[n]:
            probs.append((n, "SHA-256 does not match manifest (corrupted or changed)"))
        else:
            probs += [(n, p) for p in check_wav(d / n)]
    if not cnt.is_file():
        probs.append(("expected-count.txt", "missing"))
    else:
        try:
            want = int(cnt.read_text().strip())
            if len(on_disk) != want:
                probs.append(("expected-count.txt", f"expected {want} WAVs, found {len(on_disk)}"))
        except ValueError:
            probs.append(("expected-count.txt", "not a number"))
    return probs


def verify_all(raw, expect=None, min_free_gb=None):
    """Return {folder name: [(file, problem), ...]} covering every folder that has a problem."""
    raw = Path(raw)
    found = {p.name for p in raw.iterdir() if p.is_dir()} if raw.is_dir() else set()
    names = set(found)
    report = {}
    if expect is not None:
        want = {f"{m}__v{v}__s{s}" for m, v, s in expect}
        for n in sorted(want - found):
            report[n] = [("(folder)", "expected but not found")]
        for n in sorted(found - want):
            report[n] = [("(folder)", "found but not expected")]
        names = found & want
    elif not found:
        report[str(raw)] = [("(folder)", "no output folders found")]
    for n in sorted(names):
        p = verify_dir(raw / n)
        if p:
            report[n] = p
    if min_free_gb is not None and raw.is_dir():
        free = shutil.disk_usage(raw).free / 1e9
        if free < min_free_gb:
            report["(disk)"] = [("home", f"only {free:.1f} GB free, need {min_free_gb}")]
    return report


def parse_expect(s):
    out = []
    for item in s.split(","):
        m, v, sd = item.split(":")
        out.append((m, v, int(sd)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("--expect")
    ap.add_argument("--min-free-gb", type=float)
    a = ap.parse_args(argv)
    report = verify_all(a.raw, parse_expect(a.expect) if a.expect else None, a.min_free_gb)
    if not report:
        print("VERIFY OK")
        return 0
    print("VERIFY FAILED. Culprits:")
    for folder, probs in report.items():
        for f, p in probs:
            print(f"  {folder}/{f}: {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
