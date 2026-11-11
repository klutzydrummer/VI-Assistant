"""Small WAV helpers shared by the listening-test scripts.

Samples are plain Python lists of floats in [-1, 1], channels mixed to mono. Reading and writing
use only the standard library (`wave`), so the worker driver, the home verifier and the blind
script run anywhere. If `soundfile` is importable it is used to read (float WAVs from the model
libraries); `scipy` is used to resample if present, else a linear fallback (fine for a dry run,
and noted in the blind key).
"""

import array
import math
import sys
import wave


def read_wav(path):
    """Return (mono samples as list of float, sample_rate). Raises ValueError on a bad file."""
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return [float(v) for v in x.mean(axis=1)], int(sr)
    except ImportError:
        pass
    except Exception as e:  # soundfile present but file bad
        raise ValueError(f"unreadable WAV: {e}") from e
    try:
        with wave.open(str(path), "rb") as w:
            ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise ValueError(f"unreadable WAV: {e}") from e
    if width == 2:
        a = array.array("h")
        a.frombytes(raw)
        if sys.byteorder == "big":
            a.byteswap()
        vals = [v / 32768.0 for v in a]
    elif width == 1:
        vals = [(v - 128) / 128.0 for v in raw]
    elif width == 3:
        vals = [int.from_bytes(raw[i:i + 3], "little", signed=True) / 8388608.0 for i in range(0, len(raw) - 2, 3)]
    elif width == 4:
        a = array.array("i")
        a.frombytes(raw)
        vals = [v / 2147483648.0 for v in a]
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        vals = [sum(vals[i:i + ch]) / ch for i in range(0, len(vals) - ch + 1, ch)]
    return vals, sr


def write_wav(path, samples, sr):
    """Write mono 16-bit PCM. Values are clipped to [-1, 1]; NaN becomes 0 (callers check first)."""
    a = array.array("h", (0 if v != v else int(max(-1.0, min(1.0, v)) * 32767) for v in samples))
    if sys.byteorder == "big":
        a.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(a.tobytes())


def resample(samples, sr_from, sr_to):
    """Resample to `sr_to`. Returns (samples, method) where method is 'none', 'scipy' or 'linear'."""
    if sr_from == sr_to or not samples:
        return list(samples), "none"
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_from), int(sr_to))
        return [float(v) for v in resample_poly(samples, int(sr_to) // g, int(sr_from) // g)], "scipy"
    except ImportError:
        pass
    n_out = int(round(len(samples) * sr_to / sr_from))
    step = sr_from / sr_to
    out = []
    last = len(samples) - 1
    for i in range(n_out):
        p = i * step
        j = int(p)
        f = p - j
        out.append(samples[j] * (1 - f) + samples[min(j + 1, last)] * f)
    return out, "linear"


def rms(samples):
    return math.sqrt(sum(v * v for v in samples) / len(samples)) if samples else 0.0


def sine(seconds, sr, freq=220.0, amp=0.3):
    return [amp * math.sin(2 * math.pi * freq * i / sr) for i in range(int(seconds * sr))]
