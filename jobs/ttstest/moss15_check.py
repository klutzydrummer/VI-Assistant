"""Fail-fast import and audio check for the moss15 venv. Run it with the venv's python:

    <venv>/bin/python moss15_check.py

It imports every module the moss15 adapter and MOSS-TTS-v1.5's remote-code processor/model pull in, then
saves and loads a tiny generated WAV through torchaudio (which needs torchcodec and system FFmpeg libraries)
and resamples it. No weights, no network, no GPU. Exit 0 and a final "MOSS15 CHECK OK" line mean the
environment is complete; anything else prints the first problem and exits 1.
"""

import importlib
import math
import sys
import tempfile
from pathlib import Path

# Imported by generate.py:load_moss15, the model's processing/modeling/inference_utils files, and the
# audio tokenizer (see jobs/ttstest/README.md, "moss15 dependencies").
MODULES = ("torch", "torchaudio", "torchcodec", "torchcodec.decoders", "transformers", "transformers.models.qwen3",
           "transformers.cache_utils", "transformers.modeling_utils", "transformers.processing_utils",
           "accelerate", "safetensors", "huggingface_hub", "numpy", "scipy", "librosa", "soundfile", "tqdm", "yaml",
           "orjson", "einops", "tiktoken", "psutil", "packaging", "sentencepiece")


def main():
    for name in MODULES:
        importlib.import_module(name)
    from transformers import AutoModel, AutoProcessor, initialization  # noqa: F401  (modeling_moss_tts.py)
    import torch
    import torchaudio
    import torchcodec
    import transformers

    sr = 24000
    wav = torch.sin(2 * math.pi * 440 * torch.arange(sr // 2) / sr).unsqueeze(0).float() * 0.3
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tone.wav"
        torchaudio.save(str(p), wav, sr)
        back, back_sr = torchaudio.load(str(p))
    if back_sr != sr or back.shape != wav.shape:
        raise RuntimeError(f"torchaudio round trip changed the audio: {back_sr} Hz {tuple(back.shape)} vs {sr} Hz {tuple(wav.shape)}")
    if float((back - wav).abs().max()) > 0.01:
        raise RuntimeError("torchaudio round trip changed the samples")
    if torchaudio.functional.resample(back, back_sr, 16000).shape[-1] != sr // 2 * 16000 // sr:
        raise RuntimeError("torchaudio resample gave the wrong length")
    print(f"torch {torch.__version__}, torchaudio {torchaudio.__version__}, torchcodec {torchcodec.__version__}, "
          f"transformers {transformers.__version__}, python {sys.version.split()[0]}")
    print("MOSS15 CHECK OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 - the whole point is to name any failure
        print(f"MOSS15 CHECK FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
