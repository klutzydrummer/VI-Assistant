"""The models the discover and label jobs run, each behind a small class with
a deterministic fake twin (settings `backend = "fake"`), so CI can run the
jobs end to end without a GPU or downloads.

The real ones import torch and friends lazily; jobs/requirements-gpu.txt
lists what the worker installs.
"""

import json
import re

from vi_job import log, unit_hash

E2V_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised"]
# No "other_voice" or "effects": on the smoke test Gemma put other_voice on
# every clip, hearing her helmet filter as a second voice.
DELIVERY = [
    "laughing", "sighing", "crying", "whispering", "shouting", "breathy", "strained",
    "effort", "pain", "slurred", "sarcastic", "music",
]  # fmt: skip

# The audio LLM accepts at most 30 s of audio. Under about 1.6 s it often
# answered by asking for the audio, so shorter clips get padded with silence.
AUDIO_LLM_MAX_S = 30.0
AUDIO_LLM_MIN_S = 2.0


def load_audio(path, rate=16_000):
    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=rate)
    return np.ascontiguousarray(data, dtype=np.float32)


def padded(clip, folder):
    """The clip's audio path, or a copy padded with trailing silence to
    AUDIO_LLM_MIN_S in `folder` when it's shorter than that."""
    if clip.duration_s >= AUDIO_LLM_MIN_S:
        return clip.audio
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(clip.audio), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    pad = np.zeros(max(0, int(AUDIO_LLM_MIN_S * sr) - len(data)), dtype=np.float32)
    path = folder / f"{clip.id}.wav"
    sf.write(str(path), np.concatenate([data, pad]), sr)
    return path


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --- acoustic emotion -------------------------------------------------------


class Emotion2Vec:
    """emotion2vec+ (or plain emotion2vec) through FunASR.
    Returns class probabilities and the utterance embedding."""

    def __init__(self, model_id):
        from funasr import AutoModel

        self.model = AutoModel(model=model_id, hub="hf", disable_update=True, device=device())

    def __call__(self, clip):
        wav = load_audio(clip.audio)
        res = self.model.generate(
            wav, fs=16_000, granularity="utterance", extract_embedding=True, disable_pbar=True
        )[0]
        probs = {}
        for label, score in zip(res.get("labels", []), res.get("scores", [])):
            name = label.split("/")[-1]
            if name in E2V_CLASSES:
                probs[name] = round(float(score), 4)
        return probs, [float(f"{x:.6g}") for x in res["feats"]]  # float32 precision, half the JSON


class Dimensions:
    """audeering's wav2vec2 arousal / dominance / valence model."""

    def __init__(self, model_id):
        import torch
        from torch import nn
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.post_init()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.classifier(hidden)

        self.torch = torch
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = EmotionModel.from_pretrained(model_id).to(device()).eval()

    def __call__(self, clip):
        wav = load_audio(clip.audio)
        x = self.extractor(wav, sampling_rate=16_000, return_tensors="pt").input_values
        with self.torch.no_grad():
            a, d, v = self.model(x.to(self.model.device))[0].tolist()
        return {"arousal": round(a, 4), "dominance": round(d, 4), "valence": round(v, 4)}


class FakeEmotion2Vec:
    def __init__(self, model_id):
        self.model_id = model_id

    def __call__(self, clip):
        raw = {c: unit_hash(self.model_id, clip.id, c) ** 3 for c in E2V_CLASSES}
        total = sum(raw.values())
        probs = {c: round(v / total, 4) for c, v in raw.items()}
        emb = [unit_hash(self.model_id, clip.id, str(i)) - 0.5 for i in range(16)]
        return probs, emb


class FakeDimensions:
    def __init__(self, model_id):
        self.model_id = model_id

    def __call__(self, clip):
        return {k: round(unit_hash(self.model_id, clip.id, k), 4) for k in ("arousal", "dominance", "valence")}


# --- the audio LLM, also the text judge --------------------------------------


def parse_json(text):
    """The first JSON object in a model reply, or None."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class Gemma:
    """Gemma 4 (12B by default) through transformers: captions clips from
    their audio, judges lines from their text, summarizes clusters.

    bf16 needs about 26 GB, so on a 24 GB card it loads in 8 bits.

    Batches hold up to `batch` conversations and, when their audio lengths
    are given, at most `batch_audio_s` seconds of audio: memory grows about
    0.07 GB per second of audio in a batch (benchmarked on an A40, bf16), so
    32 short clips fit in 48 GB but 32 thirty-second ones wouldn't."""

    def __init__(self, model_id, batch=8, quant="auto", batch_audio_s=200):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.batch = batch
        self.batch_audio_s = batch_audio_s
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
        if quant == "auto":
            vram = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            quant = "8bit" if 0 < vram < 40e9 else "none"
        if quant == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        log(f"loading {model_id} (quantization: {quant})")
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs).eval()

    def batches(self, n, durations=None):
        """Index ranges for batches of at most self.batch items and, with
        durations, at most self.batch_audio_s seconds of audio."""
        start = 0
        while start < n:
            end, total = start, 0.0
            while end < n and end - start < self.batch:
                d = durations[end] if durations else 0.0
                if end > start and total + d > self.batch_audio_s:
                    break
                total += d
                end += 1
            yield start, end
            start = end

    def generate(self, conversations, max_new_tokens, durations=None):
        """Replies, one per conversation, in batches. Pass each
        conversation's audio length (seconds) in durations to cap the audio
        per batch; sorting by length beforehand also cuts padding."""
        out = []
        for start, end in self.batches(len(conversations), durations):
            chunk = conversations[start:end]
            inputs = self.processor.apply_chat_template(
                chunk,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                padding=True,
                enable_thinking=False,
            ).to(self.model.device)
            n = inputs["input_ids"].shape[-1]
            with self.torch.no_grad():
                ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            for row in ids:
                text = self.processor.decode(row[n:], skip_special_tokens=False)
                out.append(final_answer(text))
        return out

    def json(self, conversations, max_new_tokens=256, durations=None):
        return [parse_json(t) for t in self.generate(conversations, max_new_tokens, durations)]


def final_answer(text):
    """The reply with any thought block and special tokens removed."""
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
    return re.sub(r"<[^<>\s]*\|?>|<\|[^<>\s]*>", "", text).strip()


class FakeGemma:
    """Answers in the real prompts' JSON shapes, from hashes."""

    def __init__(self, model_id, batch=8, quant="auto", batch_audio_s=200):
        self.model_id = model_id

    def json(self, conversations, max_new_tokens=256, durations=None):
        out = []
        for conv in conversations:
            prompt = json.dumps(conv, sort_keys=True, default=str)
            kind = conv[-1]["kind"]
            axes = conv[-1]["axes"]
            h = lambda *k: round(unit_hash(self.model_id, prompt, *k), 3)  # noqa: E731
            if kind == "caption":
                out.append({
                    "caption": "fake caption",
                    "emotions": [["calm", "tense", "warm", "amused"][min(3, int(h("e") * 4))]],
                    "delivery": [DELIVERY[min(len(DELIVERY) - 1, int(h("d") * len(DELIVERY)))]] if h("x") > 0.7 else [],
                    "scores": {a: h(a) for a in axes},
                })  # fmt: skip
            elif kind == "judge":
                out.append({a: {"value": h(a), "confidence": h(a, "c")} for a in axes})
            elif kind == "verify":
                out.append({"rating": min(5, 1 + int(h("r") * 5))})
            else:
                out.append({"name": f"fake {int(h('n') * 100)}", "summary": "fake summary", "axis": "none"})
        return out


# --- prompts -----------------------------------------------------------------
#
# Each conversation's last element is a marker the fake model reads; the real
# model gets the conversation without it (see strip_marker).

CHARACTER = (
    "a young woman, Tali'Zorah from Mass Effect, a quarian engineer. Her voice always "
    "comes through her suit's helmet, so a slight filter on it is normal and not an effect"
)


def axes_text(axes):
    return "\n".join(f"- {a.id}: {a.description}" for a in axes)


def caption_prompt(clip, axes, with_text=False, audio=None):
    words = ""
    if with_text:
        words = f'\nThe words are: "{clip.text}"\n'
    text = (
        f"You will hear one line spoken by {CHARACTER}.{words}\n"
        "Describe how it is delivered, judging only from the sound of the voice"
        + (" and ignoring what the words mean" if not with_text else "")
        + ".\nReply with JSON only, in this shape:\n"
        '{"caption": "one short sentence on tone, pace, loudness and voice quality, and any laughter, sighs, breaths or effort sounds",\n'
        ' "emotions": ["up to four single words for the emotions you hear"],\n'
        f' "delivery": [any of: {", ".join(DELIVERY)}],\n'
        ' "scores": {"<axis id>": 0.0 to 1.0 for each axis below}}\n'
        "Scores: 0 none, 0.2 barely there, 0.4 noticeable, 0.6 clear, 1 overwhelming. Any feeling "
        "you can hear at all is at least 0.2.\n"
        f"Axes:\n{axes_text(axes)}"
    )
    # The audio goes before the instructions: after them, Gemma 4 answered
    # 14 of 40 clips by asking for the audio file.
    return [
        {"role": "user", "content": [{"type": "audio", "audio": str(audio or clip.audio)}, {"type": "text", "text": text}]},
        {"kind": "caption", "axes": [a.id for a in axes]},
    ]


def verify_prompt(clip, axis, audio=None):
    """One narrow question for the audio LLM: how well this clip's delivery
    fits one axis, 1 to 5. Audio first, words not shown."""
    text = (
        f"You heard one line spoken by {CHARACTER}.\n"
        "Judge only how it is said: tone, pace, loudness and voice quality, not what the words mean.\n"
        f"How strongly does the delivery express this: {axis.id} ({axis.description})?\n"
        "1 not at all, 2 a hint, 3 clearly, 4 strongly, 5 overwhelmingly.\n"
        'Reply with JSON only: {"rating": <1 to 5>}'
    )
    return [
        {"role": "user", "content": [{"type": "audio", "audio": str(audio or clip.audio)}, {"type": "text", "text": text}]},
        {"kind": "verify", "axes": [axis.id]},
    ]


def judge_prompt(clip, axes):
    lines = [f"{c.get('speaker', '?')}: {c.get('text', '')}" for c in clip.context]
    if clip.nonverbal:
        lines.append(f"Tali makes a sound, not words: {clip.text}")
    else:
        lines.append(f"Tali: {clip.text}")
    scene = f"Scene tags: {', '.join(clip.scene_tags)}\n" if clip.scene_tags else ""
    if clip.stage_direction:
        scene += f"The voice director's note for Tali's last line: {clip.stage_direction}\n"
    text = (
        "You label lines Tali'Zorah speaks in Mass Effect by the feelings she expresses "
        "as she says them. Read the conversation and score Tali's last line on each axis "
        "below, from 0 (none) through 0.2 (barely there), 0.4 (noticeable) and 0.6 (clear) to 1 "
        "(overwhelming), with your confidence from 0 to 1. Score what she expresses, however "
        "lightly: a line with any trace of a feeling scores at least 0.2 on it, and 0 means "
        "none at all.\n\n"
        f"Axes:\n{axes_text(axes)}\n\n{scene}Conversation:\n" + "\n".join(lines) + "\n\n"
        'Reply with JSON only: {"<axis id>": {"value": 0.0, "confidence": 0.0}, ...} with every axis.'
    )
    return [
        {"role": "user", "content": [{"type": "text", "text": text}]},
        {"kind": "judge", "axes": [a.id for a in axes]},
    ]


def summary_prompt(exemplars, axes):
    rows = "\n".join(
        f'- "{e["text"] or "(no words)"}" | heard as: {e["caption"]} | {", ".join(e["emotions"])}'
        for e in exemplars
    )
    text = (
        "These clips of Tali'Zorah's voice fell into one cluster by how they sound. Each row is "
        "the line, then an audio model's description of the delivery.\n\n"
        f"{rows}\n\n"
        "What do they have in common: emotion, delivery, or something else (combat, recording "
        "quality, line length)? Which of these axes does it match, if any?\n"
        f"{axes_text(axes)}\n\n"
        'Reply with JSON only: {"name": "two or three words", "summary": "two sentences", '
        '"axis": "an axis id, or none"}'
    )
    return [
        {"role": "user", "content": [{"type": "text", "text": text}]},
        {"kind": "summary", "axes": [a.id for a in axes]},
    ]


def strip_marker(conversations):
    return [conv[:-1] for conv in conversations]


class Judge:
    """Wraps Gemma or FakeGemma so callers pass marked conversations."""

    def __init__(self, model):
        self.model = model

    def json(self, conversations, max_new_tokens=256, durations=None):
        if isinstance(self.model, FakeGemma):
            return self.model.json(conversations, max_new_tokens)
        return self.model.json(strip_marker(conversations), max_new_tokens, durations)


# --- discovery embeddings ---------------------------------------------------


class TextEmbedder:
    def __init__(self, model_id):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device())

    def __call__(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class FakeTextEmbedder:
    def __init__(self, model_id):
        self.model_id = model_id

    def __call__(self, texts):
        return [[unit_hash(self.model_id, t, str(i)) - 0.5 for i in range(8)] for t in texts]


def prosody(clip):
    """Pitch, loudness and pace, for clustering and for matching negatives to
    positives on length and speaking rate later."""
    import librosa
    import numpy as np

    wav = load_audio(clip.audio)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=320)[0]
    voiced = rms > max(1e-4, 0.1 * float(rms.max(initial=0)))
    f0 = librosa.yin(wav, fmin=75, fmax=600, sr=16_000, frame_length=1024, hop_length=320)
    n = min(len(f0), len(voiced))
    f0 = f0[:n][voiced[:n]]
    semis = 12 * np.log2(f0 / 100.0) if len(f0) else np.array([0.0])
    words = len(clip.text.split()) if clip.text and not clip.nonverbal else 0
    return {
        "duration_s": round(clip.duration_s, 3),
        "f0_median_hz": round(float(np.median(f0)), 1) if len(f0) else None,
        "f0_range_st": round(float(np.percentile(semis, 90) - np.percentile(semis, 10)), 2),
        "loudness_db": round(float(20 * np.log10(np.mean(rms) + 1e-9)), 2),
        "loudness_range_db": round(float(np.std(20 * np.log10(rms + 1e-9))), 2),
        "voiced_fraction": round(float(voiced.mean()), 3) if len(voiced) else 0.0,
        "words_per_s": round(words / clip.duration_s, 2) if words and clip.duration_s else None,
    }


def fake_prosody(clip):
    return {
        "duration_s": round(clip.duration_s, 3),
        "f0_median_hz": round(150 + 150 * unit_hash("f0", clip.id), 1),
        "f0_range_st": round(12 * unit_hash("st", clip.id), 2),
        "loudness_db": round(-30 + 20 * unit_hash("db", clip.id), 2),
        "loudness_range_db": round(10 * unit_hash("dbr", clip.id), 2),
        "voiced_fraction": round(unit_hash("v", clip.id), 3),
        "words_per_s": None if clip.nonverbal or not clip.text else round(1 + 3 * unit_hash("w", clip.id), 2),
    }


# --- clean: speech recognition, speaker, quality ------------------------------


class Whisper:
    """Speech recognition, only to check the game's subtitle against the audio
    (and to give subtitle-less barks words)."""

    def __init__(self, model_id, batch=16):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(model_id)
        self.batch = batch
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model.to(device()),
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device(),
        )

    def __call__(self, clips):
        inputs = [{"raw": load_audio(c.audio), "sampling_rate": 16_000} for c in clips]
        out = self.pipe(inputs, batch_size=self.batch, generate_kwargs={"language": "en", "task": "transcribe"})
        return [r["text"].strip() for r in out]


class Speaker:
    """ECAPA-TDNN speaker embeddings (SpeechBrain), for checking each clip is
    her voice."""

    def __init__(self, model_id):
        from pathlib import Path

        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        save = Path.home() / ".cache" / "vi" / "speechbrain" / model_id.replace("/", "--")
        self.model = EncoderClassifier.from_hparams(source=model_id, savedir=str(save), run_opts={"device": "cuda:0" if device() == "cuda" else "cpu"})

    def __call__(self, clip):
        wav = self.torch.from_numpy(load_audio(clip.audio)).unsqueeze(0)
        with self.torch.no_grad():
            emb = self.model.encode_batch(wav)
        return [float(x) for x in emb.squeeze().cpu()]


def quality(clip):
    """DNSMOS (Microsoft's speech quality estimate) as {p808, sig, bak, ovr},
    each about 1 to 5. Clips under its 9 s window are looped to fill it."""
    import torch
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    wav = torch.from_numpy(load_audio(clip.audio))
    # One onnxruntime thread per call; clean.py runs calls in parallel.
    p808, sig, bak, ovr = deep_noise_suppression_mean_opinion_score(
        wav, 16_000, False, device="cpu", num_threads=1
    ).tolist()
    return {"p808": round(p808, 3), "sig": round(sig, 3), "bak": round(bak, 3), "ovr": round(ovr, 3)}


def trim(clip, dest, top_db=40, pad_s=0.05):
    """Writes the clip with leading and trailing silence cut (keeping pad_s)
    as FLAC at its own rate, unchanged otherwise: loudness is left alone,
    since how loud she is says something about how she feels. Returns
    (duration_s, sample_rate, rms_dbfs)."""
    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(clip.audio), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    _, (start, end) = librosa.effects.trim(data, top_db=top_db)
    pad = int(pad_s * sr)
    data = data[max(0, start - pad) : min(len(data), end + pad)]
    sf.write(str(dest), data, sr, format="FLAC")
    rms = float(np.sqrt(np.mean(data**2))) if len(data) else 0.0
    return len(data) / sr, sr, round(20 * np.log10(rms + 1e-9), 2)


def cut(clip, dest, start_s, end_s):
    """Writes the clip between prep's trim bounds as FLAC at its own rate,
    unchanged otherwise. Returns (duration_s, sample_rate, rms_dbfs)."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(clip.audio), dtype="float32", always_2d=True)
    data = data.mean(axis=1)[int(round(start_s * sr)) : int(round(end_s * sr))]
    sf.write(str(dest), data, sr, format="FLAC")
    rms = float(np.sqrt(np.mean(data**2))) if len(data) else 0.0
    return len(data) / sr, sr, round(20 * np.log10(rms + 1e-9), 2)


class FakeWhisper:
    """The subtitle back, lowercased; a fixed phrase for barks."""

    def __init__(self, model_id, batch=16):
        pass

    def __call__(self, clips):
        return [c.text.lower() if c.text.strip() else "fake words" for c in clips]


class FakeSpeaker:
    def __init__(self, model_id):
        self.model_id = model_id

    def __call__(self, clip):
        return [1.0 + 0.1 * (unit_hash(self.model_id, clip.id, str(i)) - 0.5) for i in range(16)]


def fake_quality(clip):
    return {k: round(2.5 + 2 * unit_hash("dnsmos", clip.id, k), 3) for k in ("p808", "sig", "bak", "ovr")}
