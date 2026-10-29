"""rvctest: does a fan-made Tali RVC model, run after our Qwen3-TTS output, bring back her suit
effect, timbre and accent? (Brandon, 2026-10-29.) A home check on the CPU, no rental and no
training: the downloaded model converts a handful of voicearms3's clips, and a blind hub list
puts each one beside its plain version.

Inputs, from a finished voicearms run (its out folder): her plain neutral clone, the example-clip
trials (E1..E6), her own mood clip as the example (S) for every mood, soft and breathy ones
included, and each arm the run marked "steers". From harvest: a few of her short real lines
(1-2.5 s), since RVC is weakest on short input. Every input is one listening sentence.

    python3 jobs/rvctest.py --voicearms <voicearms out> --harvest <harvest out> --out <folder> \\
        --model <a.pth>[=<a.index>] [--model <b.pth>] [--f0 rmvpe] [--pitch 0]

Outputs in --out: wav/<n>.wav, clips.toml (blind; the hub finds it under analysis/), report.md
(what each clip is, and RVC's speed on this CPU as seconds per second of audio). Runs in its own
Python 3.10 venv with rvc-python (fairseq needs 3.10); keep it at nice 19.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

SENTENCE = 8  # the first listening sentence of voicearms' confirm half


def inputs(va, harvest, short_n):
    """(label, wav) pairs to convert."""
    rep = json.loads((va / "report.json").read_text())
    out = [("neutral clone", va / "audio" / "_neutral" / "N-icl" / f"{SENTENCE}.wav")]
    for e, x in sorted(rep.get("example_trials", {}).items()):
        out.append((f"neutral clone, example {e} ({x['clip']})", va / "audio" / "example_clips" / x["audio"] / f"{SENTENCE}.wav"))
    for mood, m in sorted(rep["moods"].items()):
        out.append((f"{mood}: her mood clip as the example", va / "audio" / mood / "S" / f"{SENTENCE}.wav"))
        for arm, v in sorted(m["arms"].items()):
            if m["verdicts"].get(arm, "").startswith("steers") and "audio" in v:
                out.append((f"{mood}: {arm} x{v['strength']:g}", va / "audio" / mood / v["audio"] / f"{SENTENCE}.wav"))
    if harvest is not None:
        lines = [json.loads(x) for x in (harvest / "lines.jsonl").read_text().splitlines() if x.strip()]
        short = sorted((r for r in lines if r.get("speaker") == "tali" and 1.0 <= r["audio"]["duration_s"] <= 2.5),
                       key=lambda r: r["id"])
        step = max(1, len(short) // max(1, short_n))
        out += [(f"her real short line {r['id']}", harvest / r["audio"]["path"]) for r in short[::step][:short_n]]
    return [(label, p) for label, p in out if p.exists()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--voicearms", type=Path, required=True)
    ap.add_argument("--harvest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="a.pth or a.pth=a.index")
    ap.add_argument("--f0", default="rmvpe")
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--short", type=int, default=4)
    ap.add_argument("--fake", action="store_true", help="copy instead of converting (tests)")
    a = ap.parse_args(argv)

    todo = inputs(a.voicearms, a.harvest, a.short)
    models = [(Path(m.split("=", 1)[0]), m.split("=", 1)[1] if "=" in m else "") for m in a.model]
    (a.out / "wav").mkdir(parents=True, exist_ok=True)
    rvc = None if a.fake else Rvc(a.f0, a.pitch)
    clips, timing = [], {}
    for mp, index in models:
        name = mp.stem
        if rvc:
            rvc.load(mp, index)
        spent = audio = 0.0
        for label, src in todo:
            dst = a.out / "wav" / f"{name}--{len(clips):03d}.wav"
            t0 = time.time()
            if rvc:
                rvc.convert(src, dst)
            else:
                shutil.copy(src, dst)
            spent += time.time() - t0
            audio += seconds(src)
            clips.append((f"{label} | RVC {name}", dst))
        timing[name] = round(spent / audio, 2) if audio else None
    for k, (label, src) in enumerate(todo):
        dst = a.out / "wav" / f"plain--{k:03d}.wav"
        shutil.copy(src, dst)
        clips.append((f"{label} | plain", dst))

    import hashlib

    clips.sort(key=lambda c: hashlib.sha256(c[0].encode()).hexdigest())
    hub = ['title = "Fan Tali RVC model after our TTS (rvctest)"',
           'question = "Is this Tali with her suit effect? Note the accent and any artifacts."',
           'answers = ["her, suit on", "her, suit partly", "her, no suit", "not her or broken"]', "blind = true",
           'asked_by = "Modeling Tali\'s suit effect"', ""]
    for label, p in clips:
        hub += ["[[clip]]", f"path = {json.dumps(str(p.relative_to(a.out)))}", f"label = {json.dumps(label)}", ""]
    (a.out / "clips.toml").write_text("\n".join(hub))
    rep = ["# Fan Tali RVC after our TTS (rvctest)", "",
           f"{len(todo)} inputs, each plain and through {len(models)} model(s), f0 {a.f0}, pitch {a.pitch:+d}.", "",
           "RVC speed on this CPU (seconds of work per second of audio; under 1 is faster than real time):", ""]
    rep += [f"- {n}: {t}" for n, t in timing.items()]
    rep += ["", "Inputs:", ""] + [f"- {label}" for label, _ in todo]
    (a.out / "report.md").write_text("\n".join(rep) + "\n")
    print(f"rvctest: {len(clips)} clips in {a.out}; speed {timing}")


def seconds(path):
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


class Rvc:
    """rvc-python on the CPU; the model's own .pth says v1 or v2."""

    def __init__(self, f0, pitch):
        from rvc_python.infer import RVCInference

        self.r = RVCInference(device="cpu:0")
        self.r.set_params(f0method=f0, f0up_key=pitch)

    def load(self, path, index):
        import torch

        version = torch.load(str(path), map_location="cpu", weights_only=False).get("version", "v1")
        self.r.load_model(str(path), version=version, index_path=index)

    def convert(self, src, dst):
        self.r.infer_file(str(src), str(dst))


if __name__ == "__main__":
    main()
