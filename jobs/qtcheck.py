"""Home check before running the voice trials on the RX 480: does qwentts.cpp (a ggml port of
Qwen3-TTS, built with Vulkan by jobs/qwentts-setup.sh) run the model the same way as the PyTorch
model voicearms uses? CPU for PyTorch, no rental, a few minutes.

    python3 jobs/qtcheck.py --qt <qwentts.cpp folder> --talker <gguf> [--device Vulkan0] --out <folder>

Without qwen-tts importable, it re-runs itself in voicearms' virtualenv (~/.cache/vi/venv-...,
built from requirements-steer.txt if the voicearms smoke hasn't already). Two runs answer the question: F32 on the CPU (is the port
right?) and Q8_0 on Vulkan0 (what the trials would use). F32 on an 8 GB card spills into RAM and
froze Brandon's PC, so a GPU device with an F32 talker is refused.

Cases: the reference clip's own x-vector; that x-vector plus a random shift the size the token
arms use (arms B, C and D change only this token); and the x-vector with arm A's steering, random
directions added at layers 21-25 at half the residual's size, once with the rescale to the old
norm and the per-frame decay and once as a plain add. For each case and test sentence, x-vector-only
voice cloning in English (the streaming text layout voicearms uses), greedy:
  - qwentts.cpp generates, with its per-stage dumps (prompt embeddings, talker layers 0/7/14/21/27,
    the final norm, the first frame's logits) and its codes;
  - PyTorch reads the same prompt and those same codes (teacher-forced, voicearms' own layout), and
    the check compares each stage (cosine) and how often PyTorch's best guess for every code is
    the code qwentts.cpp picked (agreement; a free-running comparison would split at the first
    near-tie and say nothing, see qwentts.cpp's docs/ARCHITECTURE.md, Validation);
  - qwentts.cpp writes each sentence as audio, PyTorch the first (audio/<case>-<n>-{qt,torch}.wav),
    for Brandon's ears.
With the reference's transcript (--ref-text, default <ref>.txt beside it), the voice, shifted and steered
cases also run as in-context cloning (qwen-codec's reference codes plus the transcript): qwentts.cpp's
prompt stages and first logits against PyTorch's own generate() prefill, read with hooks.
Verdict "matches" if every stage's cosine >= --min-cos and both mean agreements >= --min-agree
(a quantized model flips the odd near-tie, so single sentences may dip).
Timings say how much faster qwentts.cpp is than PyTorch on this machine.

qwentts.cpp needs jobs/qwentts/vi.patch (jobs/qwentts-setup.sh applies it): the streaming text
layout and the steering, neither of which the stock build has.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from steer import TEST
from vi_job import ensure_venv

STAGES = [("embed", "talker-input-embed"), ("L0", "talker-hidden-prefill-l0"), ("L7", "talker-hidden-prefill-l7"),
          ("L14", "talker-hidden-prefill-l14"), ("L21", "talker-hidden-prefill-l21"),
          ("L27", "talker-hidden-prefill-l27"), ("final", "talker-hidden-prefill-final")]


def read_dump(path):
    """qwentts.cpp's debug dump: int32 ndims, int32 shape[ndims], float32 data."""
    raw = Path(path).read_bytes()
    nd = int(np.frombuffer(raw[:4], "<i4")[0])
    shape = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * nd], "<i4"))
    return np.frombuffer(raw[4 + 4 * nd:], "<f4").reshape(shape)


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def shifted(x, seed=0, share=0.5):
    """x plus a fixed random direction `share` times x's size: a stand-in for a fitted token shift."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(len(x))
    return (np.asarray(x) + share * np.linalg.norm(x) * d / np.linalg.norm(d)).tolist()


def free_memory():
    """Hand freed model memory back to the system before the next big model loads."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class Clip:
    def __init__(self, path):
        self.audio, self.id, self.text = Path(path), Path(path).stem, None


def encode_ref(qt, args, ref, out):
    """qwen-codec's reference codes (.rvq) for in-context cloning, from a 24 kHz copy of the clip in out/ref cut
    to whole codec frames: qwen-codec drops a last partial frame where PyTorch's tokenizer pads it into one more,
    and the prompts would differ by that frame. The copy becomes the reference for both sides."""
    d = out / "ref"
    d.mkdir(parents=True, exist_ok=True)
    wav = d / f"{ref.id}.wav"
    if not wav.exists():
        import librosa
        import soundfile as sf

        x, _ = librosa.load(str(ref.audio), sr=24_000, mono=True)
        sf.write(str(wav), x[: len(x) // 1920 * 1920], 24_000, subtype="FLOAT")
    ref.audio = wav
    if not wav.with_suffix(".rvq").exists():
        cmd = [str(qt / "build" / "qwen-codec"), "--model", args.codec, "--talker", args.talker, "-i", str(wav)]
        cmd += ["--device", args.device] if args.device else []
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        (d / "codec.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"qtcheck: qwen-codec failed ({r.returncode}), see {d / 'codec.log'}")
    (d / "ref.txt").write_text(ref.text)
    return {"rvq": str(wav.with_suffix(".rvq")), "text": str(d / "ref.txt")}


def run_qt(qt, args, spk, sentence, dump, wav, steer=None, icl=None):
    dump.mkdir(parents=True, exist_ok=True)
    spk_file = dump / "speaker.spk"
    np.asarray(spk, "<f4").tofile(spk_file)
    cmd = [str(qt / "build" / "qwen-tts"), "--model", args.talker, "--codec", args.codec, "--ref-spk", str(spk_file),
           "--lang", "English", "--greedy", "--rep-pen", "1.0", "--max-new", str(args.max_new),
           "--dump", str(dump), "-o", str(wav)]
    # In-context cloning lays the text out streaming already; x-vector-only needs the flag to match voicearms.
    cmd += ["--ref-rvq", icl["rvq"], "--ref-text", icl["text"]] if icl else ["--stream-text"]
    if steer:
        np.asarray(steer["rows"], "<f4").tofile(dump / "steer.f32")
        cmd += ["--steer", str(dump / "steer.f32"), "--steer-decay", str(steer["decay"])]
        cmd += ["--steer-renorm"] if steer["renorm"] else []
    if args.device:
        cmd += ["--device", args.device]
    t0 = time.time()
    r = subprocess.run(cmd, check=False, input=sentence + "\n", text=True, capture_output=True)
    took = time.time() - t0
    (dump / "run.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"qtcheck: qwen-tts failed ({r.returncode}), see {dump / 'run.log'}")
    codes = read_dump(dump / "codes-full.bin").astype(np.int64)
    return codes.reshape(-1, codes.shape[-1]), took


def compare(tts, sentence, spk, codes, dump, stopped, steer=None):
    """Stage cosines and teacher-forced agreement of PyTorch against qwentts.cpp's dumps and codes.
    With steer, PyTorch steers the frame positions the way generation does (voicearms' nll path)."""
    torch = tts.torch
    with torch.no_grad():
        seq, p, targets, cd = tts._layout(sentence, codes.tolist(), torch.tensor(spk, device=tts.dev))
        if steer:
            tts._steering(steer["add"], steer["renorm"], steer["decay"])
            tts.seq_from = p
        try:
            out = tts.talker(inputs_embeds=seq,
                             attention_mask=torch.ones(seq.shape[:2], device=tts.dev, dtype=torch.long),
                             output_hidden_states=True, use_cache=False)
        finally:
            tts._off()
        hs = out.hidden_states[0]  # hs[i + 1] is layer i's output, the last one after the final norm
        ours = {"embed": hs[0][0, :p], "final": hs[-1][0, :p]}
        for name, _ in STAGES:
            if name.startswith("L"):
                i = int(name[1:])
                if i + 1 < len(hs) - 1:
                    ours[name] = hs[i + 1][0, :p]
        n = cd.shape[0]
        logits = out.logits[0, p - 1: p + n].float()
        sub_logits, _ = tts.talker.forward_sub_talker_finetune(cd, hs[-1][0, p - 1: p - 1 + n])
    stages = {}
    for name, fname in STAGES:
        f = dump / f"{fname}.bin"
        if name not in ours or not f.exists():
            continue
        theirs = read_dump(f)
        mine = ours[name].float().cpu().numpy()
        if theirs.shape != mine.shape:
            stages[name] = {"cos": None, "note": f"shape {theirs.shape} vs {mine.shape}: the prompts differ"}
            continue
        stages[name] = {"cos": round(cosine(theirs, mine), 6)}
    f = dump / "talker-logits-prefill.bin"
    if f.exists():
        stages["logits"] = {"cos": round(cosine(read_dump(f), logits[0].cpu().numpy()), 6)}
    want = targets if stopped else targets[:-1]  # the stop code counts only if qwentts.cpp stopped there
    tc = tts.m.config.talker_config
    eos = tc.codec_eos_token_id
    logits[:, tc.vocab_size - 1024:eos] = -float("inf")  # generate()'s suppress_tokens: the top 1024 but the stop
    logits[:, eos + 1:] = -float("inf")
    got0 = logits[: len(want)].argmax(-1)
    agree0 = float((got0 == want).float().mean())
    agree_sub = float((sub_logits.float().argmax(-1) == cd[:, 1:]).float().mean())
    return stages, round(agree0, 4), round(agree_sub, 4)


def stage_cosines(ours, dump):
    stages = {}
    for name, fname in STAGES + [("logits", "talker-logits-prefill")]:
        f = dump / f"{fname}.bin"
        if name not in ours or not f.exists():
            continue
        theirs, mine = read_dump(f), ours[name]
        if theirs.size != mine.size:
            stages[name] = {"cos": None, "note": f"shape {theirs.shape} vs {mine.shape}: the prompts differ"}
            continue
        stages[name] = {"cos": round(cosine(theirs, mine), 6)}
    return stages


def icl_prefill(tts):
    """Hooks that keep what the talker's first (prefill) pass of a generate() sees: input embeddings, the
    layer outputs qwentts.cpp dumps, the final norm and the last position's logits. Returns (states, remove)."""
    torch, model, got = tts.torch, tts.talker.model, {}

    def first(name, value):
        if name not in got and value.shape[1] > 1:
            got[name] = value[0].detach().float().cpu().numpy()

    hooks = [model.register_forward_pre_hook(lambda m, a, k: first("embed", k["inputs_embeds"]), with_kwargs=True),
             model.norm.register_forward_hook(lambda m, a, o: first("final", o))]
    for name, _ in STAGES:
        if name.startswith("L") and int(name[1:]) < tts.n_layers:
            hooks.append(model.layers[int(name[1:])].register_forward_hook(
                lambda m, a, o, n=name: first(n, o[0] if isinstance(o, tuple) else o)))

    def logits(m, a, o):
        if "logits" not in got and o.logits.shape[1] > 1:
            got["logits"] = o.logits[0, -1].detach().float().cpu().numpy()

    hooks.append(tts.talker.register_forward_hook(logits))
    return got, lambda: [h.remove() for h in hooks] and None


def speak_torch(tts, ref, spk, sentence, max_new, wav, steer=None, icl=False):
    """PyTorch's greedy audio; with icl, in-context cloning, returning the prefill states it saw."""
    import soundfile as sf

    torch = tts.torch
    items = tts.model.create_voice_clone_prompt(ref_audio=str(ref.audio), ref_text=ref.text if icl else None,
                                                x_vector_only_mode=not icl)
    e = items[0].ref_spk_embedding
    items[0].ref_spk_embedding = torch.tensor(spk, dtype=e.dtype, device=e.device)
    if steer:
        tts._steering(steer["add"], steer["renorm"], steer["decay"])
    got, remove = icl_prefill(tts) if icl else ({}, lambda: None)
    t0 = time.time()
    try:
        wavs, sr = tts.model.generate_voice_clone(text=sentence, language="English", voice_clone_prompt=items,
                                                  max_new_tokens=max_new, do_sample=False, subtalker_dosample=False,
                                                  repetition_penalty=1.0)
    finally:
        tts._off()
        remove()
    took = time.time() - t0
    sf.write(str(wav), wavs[0], sr)
    return (took, got) if icl else took


def steering(tts, x, sentence, renorm, decay):
    """Arm A's kind of push for the check: per layer at 21-25 (scaled for a smaller model), a fixed
    random direction at half the residual's mean norm there (read from the prompt), as [layers,
    hidden] rows (zeros elsewhere) for qwentts.cpp and {layer: row} for PyTorch."""
    torch, n = tts.torch, tts.n_layers
    layers = list(range(max(0, n - 7), n - 2))
    with torch.no_grad():
        seq, p, _, _ = tts._layout(sentence, [[0] * 16], torch.tensor(x, device=tts.dev))
        hs = tts.talker(inputs_embeds=seq, output_hidden_states=True, use_cache=False).hidden_states[0]
    rng = np.random.default_rng(7)
    hidden = len(x)
    rows = np.zeros((n, hidden), np.float32)
    for i in layers:
        d = rng.standard_normal(hidden)
        rows[i] = 0.5 * float(hs[i + 1][0, :p].float().norm(dim=-1).mean()) * d / np.linalg.norm(d)
    return {"rows": rows, "add": {i: rows[i].tolist() for i in layers}, "renorm": renorm, "decay": decay,
            "layers": layers}


def render(r):
    lines = ["# qwentts.cpp against PyTorch", "",
             (f"Verdict: **{r['verdict']}** (every stage cosine >= {r['min_cos']}, mean agreement >= {r['min_agree']}: "
              f"code 0 {r['agree_c0']}, codes 1-15 {r['agree_sub']})."),
             f"qwentts.cpp: `{r['talker']}` on {r['device'] or 'its default device'}; PyTorch: {r['model']} on CPU.", "",
             "| speaker | sentence | frames | " + " | ".join(n for n, _ in STAGES) + " | logits | agree c0 | agree 1-15 | "
             "qt s | torch s |", "|---" * (len(STAGES) + 8) + "|"]
    for c in r["cases"]:
        cells = [str(c["stages"].get(n, {}).get("cos", "-")) for n, _ in STAGES + [("logits", "")]]
        lines.append(f"| {c['speaker']} | {c['sentence']} | {c['frames']} | " + " | ".join(cells)
                     + f" | {c['agree_c0'] if c['agree_c0'] is not None else '-'} | "
                     + f"{c['agree_sub'] if c['agree_sub'] is not None else '-'} | {c['qt_seconds']:.1f} | "
                     + ("-" if c["torch_seconds"] is None else f"{c['torch_seconds']:.1f}") + " |")
    notes = [f"{c['speaker']}-{c['sentence']} {n}: {v['note']}" for c in r["cases"] for n, v in c["stages"].items()
             if v.get("note")]
    if notes:
        lines += ["", "Notes:", ""] + [f"- {x}" for x in notes]
    lines += ["", (f"Speed, first sentence of each case: qwentts.cpp {r['qt_seconds']:.1f} s against PyTorch "
                   f"{r['torch_seconds']:.1f} s ({r['speedup']}x)."), "",
              "Listen: audio/<speaker>-<n>-qt.wav (qwentts.cpp), and for sentence 0 -torch.wav (PyTorch). Greedy on both, so the",
              "two can part ways at a near-tie and still both be right; they should sound like the same voice. icl-* is",
              "in-context cloning (the reference clip's codes and transcript, the mode that keeps her voice): its prompt",
              "is compared stage by stage, so its agreement columns are empty."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--qt", required=True, help="qwentts.cpp folder (with build/qwen-tts)")
    ap.add_argument("--talker", required=True, help="talker GGUF, e.g. models/qwen-talker-1.7b-base-Q8_0.gguf")
    ap.add_argument("--codec", help="tokenizer GGUF (default: the F32 one beside the talker)")
    ap.add_argument("--device", help="qwentts.cpp device: Vulkan0, CPU (default: its own choice)")
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="the PyTorch model")
    ap.add_argument("--ref", help="reference clip for the speaker (default: qwentts.cpp's examples/freeman.wav)")
    ap.add_argument("--ref-text", help="the reference clip's transcript, for in-context cloning (default: <ref>.txt)")
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=72)
    ap.add_argument("--min-cos", type=float, default=0.995)
    ap.add_argument("--min-agree", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.device or "").upper() != "CPU" and "F32" in Path(args.talker).name:
        raise SystemExit("qtcheck: an F32 talker on the GPU overfills an 8 GB card; use Q8_0 there, F32 only with --device CPU")
    try:
        import qwen_tts  # noqa: F401
    except ImportError:
        os.environ.setdefault("VI_WORKER", "1")  # ensure_venv acts only on workers; here this machine is one
        ensure_venv("requirements-steer.txt")
    qt, out = Path(args.qt), Path(args.out)
    args.codec = args.codec or str(Path(args.talker).parent / "qwen-tokenizer-12hz-F32.gguf")
    (out / "audio").mkdir(parents=True, exist_ok=True)

    import voicearms

    # One big model in memory at a time (PyTorch F32 and the F32 GGUF together swapped a 32 GB PC): PyTorch
    # makes the speakers and steering, is freed, qwentts.cpp runs every case, then PyTorch comes back to compare.
    ref = Clip(args.ref or qt / "examples" / "freeman.wav")
    ref_txt = Path(args.ref_text) if args.ref_text else ref.audio.with_suffix(".txt")
    ref.text = ref_txt.read_text().strip() if ref_txt.exists() else None
    prep, key = out / "prep.json", [args.model, str(ref.audio)]
    p = json.loads(prep.read_text()) if prep.exists() else {}
    if p.get("key") != key:
        tts = voicearms.ArmTTS(args.model, "cpu")
        x = tts.xvector(ref)
        p = {"key": key, "x": list(map(float, x)), "steered": steering(tts, x, TEST[0], True, 0.985),
             "steered-plain": steering(tts, x, TEST[0], False, 1.0)}
        for k in ("steered", "steered-plain"):
            p[k]["rows"] = p[k]["rows"].tolist()
        prep.write_text(json.dumps(p))
        del tts
        free_memory()
    x = p["x"]
    for k in ("steered", "steered-plain"):
        p[k]["add"] = {int(i): row for i, row in p[k]["add"].items()}
    plan = [(who, spk, st, n) for who, spk, st in (("voice", x, None), ("shifted", shifted(x), None),
                                                   ("steered", x, p["steered"]), ("steered-plain", x, p["steered-plain"]))
            for n in range(args.sentences)]
    # In-context cloning (her clip as codes plus its transcript, the mode that keeps her voice): the prompt is
    # checked stage by stage; decoding is the same code as above, so codes aren't compared again.
    icl_plan = [] if not ref.text else [(f"icl-{who}", spk, st, n) for who, spk, st in (
        ("voice", x, None), ("shifted", shifted(x), None), ("steered", x, p["steered"])) for n in range(args.sentences)]
    if not ref.text:
        print(f"qtcheck: no transcript at {ref_txt}, so no in-context cases (--ref-text)")
    icl = encode_ref(qt, args, ref, out) if icl_plan else None
    runs = {}
    for who, spk, st, n in plan + icl_plan:
        dump = out / "dumps" / f"{who}-{n}"
        runs[who, n] = run_qt(qt, args, spk, TEST[n], dump, out / "audio" / f"{who}-{n}-qt.wav", st,
                              icl if who.startswith("icl-") else None)
        print(f"qtcheck: qwen-tts {who} {n}: {len(runs[who, n][0])} frames in {runs[who, n][1]:.1f} s")

    tts = voicearms.ArmTTS(args.model, "cpu")
    cases = []
    for who, spk, st, n in plan:
        dump = out / "dumps" / f"{who}-{n}"
        codes, qt_s = runs[who, n]
        stopped = len(codes) < args.max_new
        stages, a0, asub = compare(tts, TEST[n], spk, codes, dump, stopped, st)
        torch_s = None  # PyTorch on the CPU is slow: its audio (and the timing) for the first sentence only
        if n == 0:
            torch_s = speak_torch(tts, ref, spk, TEST[n], args.max_new, out / "audio" / f"{who}-{n}-torch.wav", st)
        cases.append({"speaker": who, "sentence": n, "frames": len(codes), "stopped": stopped, "stages": stages,
                      "agree_c0": a0, "agree_sub": asub, "qt_seconds": qt_s, "torch_seconds": torch_s})
        print(f"qtcheck: {who} {n}: {len(codes)} frames, agree {a0}/{asub}, "
              + ", ".join(f"{k} {v.get('cos')}" for k, v in stages.items()))
    for who, spk, st, n in icl_plan:
        dump = out / "dumps" / f"{who}-{n}"
        codes, qt_s = runs[who, n]
        wav = out / "audio" / f"{who}-{n}-torch.wav" if n == 0 else dump / "torch.wav"
        torch_s, got = speak_torch(tts, ref, spk, TEST[n], args.max_new if n == 0 else 2, wav, st, icl=True)
        stages = stage_cosines(got, dump)
        cases.append({"speaker": who, "sentence": n, "frames": len(codes), "stopped": len(codes) < args.max_new,
                      "stages": stages, "agree_c0": None, "agree_sub": None, "qt_seconds": qt_s,
                      "torch_seconds": torch_s if n == 0 else None})
        print(f"qtcheck: {who} {n}: " + ", ".join(f"{k} {v.get('cos')}" for k, v in stages.items()))
    a0 = round(float(np.mean([c["agree_c0"] for c in cases if c["agree_c0"] is not None])), 4)
    asub = round(float(np.mean([c["agree_sub"] for c in cases if c["agree_sub"] is not None])), 4)
    ok = all(v.get("cos") is not None and v["cos"] >= args.min_cos for c in cases for v in c["stages"].values()) and min(
        a0, asub) >= args.min_agree
    timed = [c for c in cases if c["torch_seconds"] is not None]
    qt_s, torch_s = sum(c["qt_seconds"] for c in timed), sum(c["torch_seconds"] for c in timed)
    report = {"verdict": "matches" if ok else "differs", "min_cos": args.min_cos, "min_agree": args.min_agree,
              "talker": Path(args.talker).name, "device": args.device, "model": args.model, "cases": cases,
              "agree_c0": a0, "agree_sub": asub,
              "qt_seconds": round(qt_s, 2), "torch_seconds": round(torch_s, 2),
              "speedup": round(torch_s / max(qt_s, 1e-9), 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "report.md").write_text(render(report))
    print(f"qtcheck: {report['verdict']}; wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
