"""Synthesis driver for the TTS listening test: ONE process per (model, voice, seed).

    python3 generate.py --model omnivoice --voice T --seed 1 --out <dir> --refs <refs dir>

Writes `<out>/<model>__v<voice>__s<seed>/` holding 13 WAVs named
`<model>__v<voice>__<LINE>__s<seed>.wav`, then (last, only if every line worked) `manifest.sha256`,
`expected-count.txt` and `run-info.json`. A failure writes run-info.json with the error and
exits nonzero WITHOUT a manifest, so the home verifier names the culprit. Each (model, voice, seed)
folder is self-contained so it can be pushed home on its own. Pushing is NOT done here.

Models: omnivoice, moss15, qwen3base (real adapters, imported only when selected) and
`stub` (sine WAVs, for dry runs; needs no GPU, no network). The three real adapters were written
from the vendors' public READMEs and have NOT been run: the per-model smoke test on the worker
(line L01 only: `--only L01`) is what proves them. Voices: T (Tali clip) and P (public control),
reference files `<refs>/v<voice>.wav` and transcripts in `<refs>/refs.json`.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wavio  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ("omnivoice", "moss15", "qwen3base", "stub")
VOICES = ("T", "P")
REPO_IDS = {
    "omnivoice": "k2-fsa/OmniVoice",
    "moss15": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "qwen3base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "stub": "stub",
}


EXTRA_REPOS = {"moss15": ("OpenMOSS-Team/MOSS-Audio-Tokenizer",)}

_SEED = [1]  # the seed of the clip being made, for adapters that run a subprocess


def seed_all(seed):
    _SEED[0] = seed
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _np_to_list(w):
    """A model's audio (tensor / ndarray / list, maybe 2-D) as a flat mono list of floats."""
    if hasattr(w, "detach"):
        w = w.detach().float().cpu().numpy()
    if hasattr(w, "ndim") and w.ndim > 1:
        w = w.squeeze()
        if w.ndim > 1:
            w = w.mean(axis=0 if w.shape[0] < w.shape[-1] else 1)
    return [float(v) for v in w]


# --- adapters: load() -> synth(text, ref_wav, ref_text) -> (samples, sample_rate) ----------------

def load_stub():
    def synth(text, ref_wav, ref_text):
        return wavio.sine(1.5 + 0.05 * len(text.split()), 24000, freq=180 + 7 * (len(text) % 20)), 24000
    return synth


def load_qwen3base():
    """Qwen3-TTS Base, full in-context cloning with the transcript (as jobs/voicetune.py does)."""
    import torch
    from qwen_tts import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained(REPO_IDS["qwen3base"], device_map="cuda:0", dtype=torch.bfloat16)

    def synth(text, ref_wav, ref_text):
        wavs, sr = m.generate_voice_clone(text=text, ref_audio=str(ref_wav), ref_text=ref_text)
        return _np_to_list(wavs[0]), sr
    return synth


def load_omnivoice():
    """OmniVoice (k2-fsa), inference only; passes ref_text so Whisper is never loaded."""
    import torch
    from omnivoice import OmniVoice

    m = OmniVoice.from_pretrained(REPO_IDS["omnivoice"], device_map="cuda:0", dtype=torch.float16)

    def synth(text, ref_wav, ref_text):
        audio = m.generate(text=text, ref_audio=str(ref_wav), ref_text=ref_text)
        return _np_to_list(audio[0]), 24000
    return synth


def load_moss15():
    """MOSS-TTS-v1.5 (8B) through plain transformers, reference audio only (no transcript)."""
    import torch
    from transformers import AutoModel, AutoProcessor

    rid = REPO_IDS["moss15"]
    proc = AutoProcessor.from_pretrained(rid, trust_remote_code=True)
    model = AutoModel.from_pretrained(rid, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    sr = int(getattr(getattr(proc, "audio_tokenizer", None), "sampling_rate", 24000))

    def synth(text, ref_wav, ref_text):
        conv = [proc.build_user_message(text=text, reference=[str(ref_wav)])]
        batch = proc(conv, mode="generation")
        with torch.no_grad():
            out = model.generate(input_ids=batch["input_ids"].to("cuda:0"),
                                 attention_mask=batch["attention_mask"].to("cuda:0"), max_new_tokens=4096)
        msgs = proc.decode(out)
        return _np_to_list(msgs[0].audio_codes_list[0]), sr
    return synth


LOADERS = {"stub": load_stub, "omnivoice": load_omnivoice, "moss15": load_moss15,
           "qwen3base": load_qwen3base}


# --- test-only fault injection (TTSTEST_STUB=1) ----------------------------------------------------
# With TTSTEST_STUB=1 every model name runs the sine stub, so jobs/ttstest/worker_run.py can be
# exercised end to end with no GPU, network or weights (see test_inject.py). Optional faults, each
# naming ONE model: TTSTEST_STUB_FAIL (MemoryError while loading), TTSTEST_STUB_CORRUPT (flip a byte
# in a WAV after the manifest is written), TTSTEST_STUB_DELETE (delete the finished folder),
# TTSTEST_STUB_DELAY (seconds per clip, so a test can kill the job halfway).

def stub_active():
    return os.environ.get("TTSTEST_STUB") == "1"


def stub_loader(model):
    def load():
        if os.environ.get("TTSTEST_STUB_FAIL") == model:
            raise MemoryError("injected by TTSTEST_STUB_FAIL")
        base = load_stub()
        delay = float(os.environ.get("TTSTEST_STUB_DELAY") or 0)

        def synth(text, ref_wav, ref_text):
            time.sleep(delay)
            return base(text, ref_wav, ref_text)
        return synth
    return load


def stub_faults_after(model, folder, only):
    """Damage a finished folder, as TTSTEST_STUB_CORRUPT / TTSTEST_STUB_DELETE ask (full runs only)."""
    if only:
        return
    if os.environ.get("TTSTEST_STUB_CORRUPT") == model:
        w = sorted(folder.glob("*.wav"))[0]
        b = bytearray(w.read_bytes())
        b[200] ^= 0xFF
        w.write_bytes(bytes(b))
    if os.environ.get("TTSTEST_STUB_DELETE") == model:
        shutil.rmtree(folder)


# --- run ------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clip_name(model, voice, line_id, seed):
    return f"{model}__v{voice}__{line_id}__s{seed}.wav"


def run_dir(out, model, voice, seed):
    return Path(out) / f"{model}__v{voice}__s{seed}"


def write_manifest(d, names, n_expected):
    (d / "manifest.sha256").write_text("".join(f"{sha256_file(d / n)}  {n}\n" for n in sorted(names)))
    (d / "expected-count.txt").write_text(f"{n_expected}\n")


def generate(model, voice, seed, out, refs, lines, only=None, loaders=None):
    """Synthesise every line; returns (exit_code, run_info). Never raises for a model failure."""
    d = run_dir(out, model, voice, seed)
    d.mkdir(parents=True, exist_ok=True)
    if any(d.iterdir()):
        raise SystemExit(f"{d} is not empty; refusing to mix runs")
    lines = [ln for ln in lines if not only or ln["id"] in only]
    info = {"model": model, "repo_id": REPO_IDS[model], "voice": voice, "seed": seed,
            "n_lines": len(lines), "python": platform.python_version(), "host": platform.node(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "timings_s": {}, "ok": False}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    code = 1
    try:
        refs = Path(refs)
        ref_wav = refs / f"v{voice}.wav"
        ref_text = json.loads((refs / "refs.json").read_text())[f"v{voice}"]["text"]
        if not ref_wav.is_file():
            raise FileNotFoundError(ref_wav)
        t0 = time.time()
        synth = (loaders or LOADERS)[model]()
        info["timings_s"]["load"] = round(time.time() - t0, 2)
        names = []
        for ln in lines:
            seed_all(seed)
            t1 = time.time()
            x, sr = synth(ln["text"], ref_wav, ref_text)
            if not x:
                raise RuntimeError(f"model returned no audio for {ln['id']}")
            name = clip_name(model, voice, ln["id"], seed)
            wavio.write_wav(d / name, x, sr)
            names.append(name)
            info["timings_s"][ln["id"]] = round(time.time() - t1, 2)
        write_manifest(d, names, len(lines))
        info["ok"] = True
        code = 0
    except BaseException as e:  # MemoryError, CUDA OOM and so on: report, never swallow silently
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
        print(f"FAILED {model} voice {voice} seed {seed}: {info['error']}", file=sys.stderr)
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            (d / "run-info.json").write_text(json.dumps(info, indent=2) + "\n")
            raise
    info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (d / "run-info.json").write_text(json.dumps(info, indent=2) + "\n")
    return code, info


def download_weights(model):
    """Fetch the model's weights into the Hugging Face cache (no token is ever used or needed)."""
    from huggingface_hub import snapshot_download

    # MOSS-TTS-v1.5's processor also loads a second repo, the audio tokenizer (7.1 GB): fetch it now, so the
    # whole download happens here and not inside the smoke test.
    for rid in (REPO_IDS[model], *EXTRA_REPOS.get(model, ())):
        path = snapshot_download(rid)
        print(f"downloaded {rid} to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--voice", choices=VOICES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--lines", type=Path, default=HERE / "lines.json")
    ap.add_argument("--only", help="comma-separated line ids (smoke test: L01)")
    ap.add_argument("--venv", metavar="REQUIREMENTS",
                    help="worker only: re-run inside the virtualenv built from jobs/<file> (vi_job.ensure_venv)")
    ap.add_argument("--download-only", action="store_true", help="fetch the weights, then stop")
    a = ap.parse_args(argv)
    if a.venv:
        sys.path.insert(0, str(HERE.parent))
        import vi_job

        vi_job.ensure_venv(a.venv)  # re-executes this command inside the venv when it is not in it yet
    if a.download_only:
        if not stub_active():
            download_weights(a.model)
        return 0
    if not (a.voice and a.out and a.refs):
        ap.error("--voice, --out and --refs are required unless --download-only")
    lines = json.loads(a.lines.read_text())
    only = set(a.only.split(",")) if a.only else None
    loaders = {m: stub_loader(m) for m in MODELS} if stub_active() else None
    code, _ = generate(a.model, a.voice, a.seed, a.out, a.refs, lines, only, loaders)
    d = run_dir(a.out, a.model, a.voice, a.seed)
    if code == 0 and stub_active():
        stub_faults_after(a.model, d, only)
    print(("OK " if code == 0 else "FAILED ") + str(d))
    return code


if __name__ == "__main__":
    sys.exit(main())
