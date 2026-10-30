"""voicetune: a full fine-tune of Qwen3-TTS-12Hz-1.7B-Base that speaks as Tali with no reference clip.
(Stage 5 of the Tali voice plan; Brandon 2026-10-30: "Let's just do a full fine tune".) One model
for every mood, never one per mood. There's no strength knob: the app's mood state drives the voice
through the projector, which later learns to write the speaker slot.

Built on the official recipe (QwenLM/Qwen3-TTS finetuning/ at 022e286b, read 2026-10-30; notes in
docs/design/qwen3tts-finetune-guide.md), with its problems fixed:
  - alignment: at generation, the hidden state at position p predicts frame p+1's first code, and
    the code predictor gets that same hidden state with frame p+1's first code (talker.forward's
    past_hidden). The official sft_12hz.py passes codec_0_labels[:, 1:] to a loss that shifts
    again (each frame learns the frame two ahead: "speech speeds up every epoch") and pairs the
    hidden state at p with frame p's codes. Here the labels go in unshifted (`train_labels`) and
    the code predictor gets the hidden state one position back (`predictor_hidden_mask`).
  - attention: sdpa, not the hard-coded flash_attention_2.
  - speaker slot: each clip's OWN x-vector (`--speaker clip`, the default), not one fixed vector.
    Her x-vector moves with her mood (Brito 2026), so training on a fixed one would teach the model
    to ignore the slot the projector needs. The saved speaker `tali` is her mean x-vector
    (rescaled to the clips' average norm, as voicearms does). `--speaker mean` puts the mean in for
    every line instead.
  - label-less style (`--speaker style`, Brandon 2026-10-30: "what if the labels are not
    accurate?"): a small style encoder hears each clip's own audio and squeezes its delivery into
    `--style-dim` numbers, added to her mean x-vector in the slot, trained with the model (global
    style tokens / reference encoder, Wang et al. 2018). No labels anywhere: the state space comes
    from her clips, to be clustered and named by ear afterwards. The game goes in as its own
    learned vector, so the style numbers needn't carry each game's suit; `speak --game` picks it.
  - style extras (Brandon 2026-10-30: reward the delivery we want, penalise the rest; each off at 0):
    `--style-shuffle S` hands the encoder the clip cut into S-second chunks in a random order, so it can
    hear delivery but not read the words back (content leakage); chunks of ~1 s keep local rhythm.
    `--game-adv W` adds a game classifier on the style code behind gradient reversal, so the code
    can't carry which game (suit) a clip is from; small W, since game and mood overlap (drunk is LE3).
    `--prosody-aux W` adds a head that predicts the clip's measured pitch range, loudness and pace
    from the code, so the code can't be ignored. `--style-dropout P` zeroes the code for a share P
    of lines, so her mean voice stays a good default. None of these exist at inference.
  - forgetting: `--kl` adds a KL-to-Base term on replay lines, i.e. lines Base itself spoke in
    OTHER voices (`replay`). Teacher-forcing on the frozen model's own samples estimates the KL to
    it (Context-Free Synthetic Data, arXiv 2505.13811), so this keeps reading skill and expressive
    range without pulling her own voice back toward Base's version of it. lr 2e-6 (2e-5 gives
    noise and endless output); a held-out loss per epoch picks the checkpoint.

    python3 jobs/voicetune.py check   --out <folder>
    python3 jobs/voicetune.py replay  --sentences s.txt --voices <dir of 24 kHz wavs> --out <folder>
    python3 jobs/voicetune.py prepare --raw train_raw.jsonl --out <folder> [--device cpu]
    python3 jobs/voicetune.py train   --data train_with_codes.jsonl --val val_with_codes.jsonl --out <folder>
                                      [--replay replay_with_codes.jsonl --kl 0.1] [--epochs 10] [--max-steps N]
    python3 jobs/voicetune.py speak   --run <folder> --sentences s.txt [--epoch K] [--icl-ref clip.wav]
                                      [--style-from her_clip.wav ...] [--game le3]

train_raw.jsonl (built at home): one line per clip, {"id", "audio" (24 kHz mono wav), "text",
"ref_audio" (the clip itself, for its own x-vector), "game", optional "language"}. `prepare` adds
codec codes (fits the home CPU: one ~0.7 GB tokenizer). `train` writes epoch-<k>/talker.safetensors
(~3.4 GB each; bring home only the chosen one), voice.json (her mean x-vector) and train_log.jsonl
(losses and peak GPU memory). `speak` makes the no-reference voice model from an epoch and writes a
blind hub list (clips.toml), with the Base model's clone from her clip beside it when --icl-ref is
given. Needs qwen-tts==0.1.1 (requirements-tune.txt) and, for training, a 48 GB GPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER = "tali"
SPEAKER_ID = 3000  # the codec-embedding row the official recipe gives a new speaker
SPEAKER_SLOT = 6  # input position that holds the speaker embedding
SR = 24000
REQUIRED = ["id", "audio", "text", "ref_audio"]


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def validate(rows, sample_rate=None):
    """Problems with a train_raw/with_codes list, as strings (empty = fine). sample_rate(path) -> Hz."""
    bad = []
    seen = set()
    for n, r in enumerate(rows):
        miss = [k for k in REQUIRED if not r.get(k)]
        if miss:
            bad.append(f"line {n + 1}: missing {', '.join(miss)}")
            continue
        if r["id"] in seen:
            bad.append(f"line {n + 1}: duplicate id {r['id']}")
        seen.add(r["id"])
        if sample_rate is not None:
            for k in ("audio", "ref_audio"):
                sr = sample_rate(r[k])
                if sr != SR:
                    bad.append(f"line {n + 1}: {k} is {sr} Hz, needs {SR}")
    return bad


def train_labels(codec_0_labels):
    """The labels to hand the talker for inputs_embeds[:, :-1]. Its transformers causal-LM loss shifts
    labels by one itself, so they must line up with the inputs, not be pre-shifted (the official
    script's codec_0_labels[:, 1:] trains each frame on the frame two ahead)."""
    return codec_0_labels[:, :-1]


def predictor_hidden_mask(codec_mask):
    """Which hidden states (of inputs_embeds[:, :-1]) feed the code predictor, one per audio frame:
    the position just before each frame, whose output predicted that frame's first code."""
    return codec_mask[:, 1:]


def mean_voice(vectors):
    """Mean of x-vectors, rescaled to their average length (a mean of vectors is shorter than they are)."""
    import numpy as np

    x = np.asarray(vectors, dtype=np.float64)
    m = x.mean(0)
    return m * np.linalg.norm(x, axis=1).mean() / max(np.linalg.norm(m), 1e-12)


def length_bands(rows, key="duration_s"):
    """Clip counts and hours by length band, for the report (short lines matter: issue #343)."""
    bands = {"<2 s": 0, "2-5 s": 0, "5-10 s": 0, "10+ s": 0}
    hours = dict.fromkeys(bands, 0.0)
    for r in rows:
        d = float(r.get(key, 0))
        b = "<2 s" if d < 2 else "2-5 s" if d < 5 else "5-10 s" if d < 10 else "10+ s"
        bands[b] += 1
        hours[b] += d / 3600
    return {b: {"clips": bands[b], "hours": round(hours[b], 3)} for b in bands}


def pad_mels(mels):
    """[T_i, 128] mels -> ([b, T, 128] zero-padded, [b] lengths) (torch)."""
    import torch

    n = torch.tensor([m.shape[0] for m in mels])
    out = torch.zeros(len(mels), int(n.max()), mels[0].shape[1])
    for i, m in enumerate(mels):
        out[i, : m.shape[0]] = m
    return out, n


def shuffle_chunks(mel, length, chunk, gen=None):
    """mel [T, d] (first `length` frames real) cut into `chunk`-frame pieces, put back in a random order (torch)."""
    import torch

    n = int(length)
    if chunk <= 0 or n <= chunk:
        return mel
    pieces = list(mel[:n].split(chunk))
    order = torch.randperm(len(pieces), generator=gen).tolist()
    out = mel.clone()
    out[:n] = torch.cat([pieces[i] for i in order])
    return out


def grad_reverse(x, scale=1.0):
    """Identity forward, gradient times -scale backward (torch)."""
    import torch

    class _Rev(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.view_as(t)

        @staticmethod
        def backward(ctx, g):
            return -scale * g

    return _Rev.apply(x)


def prosody_targets(path, text):
    """[f0 range in semitones (10-90th pct), mean loudness dB, loudness spread dB, words per second] of a clip."""
    import librosa
    import numpy as np

    wav, sr = librosa.load(path, sr=16000, mono=True)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=320)[0]
    voiced = rms > max(1e-4, 0.1 * float(rms.max(initial=0)))
    f0 = librosa.yin(wav, fmin=75, fmax=600, sr=sr, frame_length=1024, hop_length=320)
    k = min(len(f0), len(voiced))
    f0 = f0[:k][voiced[:k]]
    semis = 12 * np.log2(f0 / 100.0) if len(f0) else np.array([0.0])
    db = 20 * np.log10(rms + 1e-9)
    dur = max(len(wav) / sr, 1e-3)
    return [float(np.percentile(semis, 90) - np.percentile(semis, 10)), float(db.mean()), float(db.std()),
            len(text.split()) / dur]


def kl_to_base(base_logits, logits, mask):
    """Mean KL(base || tuned) over the masked positions, from raw logits [b, t, v] (torch)."""
    import torch.nn.functional as F

    lb = F.log_softmax(base_logits[mask].float(), -1)
    lt = F.log_softmax(logits[mask].float(), -1)
    return (lb.exp() * (lb - lt)).sum(-1).mean()


# --- GPU / model parts (not imported in CI) ---------------------------------------------------------


def load_tts(path=BASE, device="cuda:0"):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return Qwen3TTSModel.from_pretrained(path, device_map=device, dtype=dtype, attn_implementation="sdpa")


def mel(path):
    import librosa
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    wav, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz, needs {SR}")
    return mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=SR,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)


def style_encoder(dim, width=256, out=2048):
    """Reference encoder: mel [b, T, 128] + lengths -> a dim-number bottleneck -> a slot offset [b, out]."""
    import torch
    from torch import nn

    class StyleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(128, width, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU())
            self.gru = nn.GRU(width, width, batch_first=True)
            self.bottleneck = nn.Linear(width, dim)
            self.up = nn.Linear(dim, out)
            nn.init.zeros_(self.up.weight)  # starts as "no offset": her mean voice
            nn.init.zeros_(self.up.bias)

        def code(self, mels, lengths):
            h = self.conv(mels.transpose(1, 2)).transpose(1, 2)
            n = torch.clamp((lengths + 7) // 8, min=1, max=h.shape[1]).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, n, batch_first=True, enforce_sorted=False)
            _, last = self.gru(packed)
            return torch.tanh(self.bottleneck(last[-1]))

        def forward(self, mels, lengths):
            return self.up(self.code(mels, lengths))

    return StyleEncoder()


def xvectors(model, paths, cache):
    """One x-vector per ref clip, cached in a JSON file keyed by path."""
    import torch

    got = json.loads(cache.read_text()) if cache.exists() else {}
    with torch.no_grad():
        for p in dict.fromkeys(paths):
            if p not in got:
                m = mel(p).to(model.device).to(model.dtype)
                got[p] = model.speaker_encoder(m)[0].float().cpu().tolist()
    cache.write_text(json.dumps(got))
    return got


def cmd_check(a):
    """Everything the run needs, checked in one go before training (plan ahead, don't chase errors)."""
    out = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:  # noqa: BLE001 - report every failure, don't stop at the first
        out["torch_error"] = repr(e)
    for mod in ("qwen_tts", "transformers", "accelerate", "librosa", "soundfile", "huggingface_hub"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            out[f"{mod}_error"] = repr(e)
    if out.get("transformers") and out["transformers"] != "4.57.3":
        out["transformers_warning"] = "qwen-tts 0.1.1 pins 4.57.3"
    if out.get("vram_gb") is not None and out["vram_gb"] < 40:
        out["vram_warning"] = "a full fine-tune of 1.7B with --kl wants a 48 GB card"
    if not a.no_model and "qwen_tts_error" not in out:
        try:
            dev = "cuda:0" if out.get("gpu") else "cpu"
            tts = load_tts(a.base, dev)
            out["talker_params_m"] = round(sum(p.numel() for p in tts.model.talker.parameters()) / 1e6, 1)
            out["model"] = "loaded with sdpa"
            if dev.startswith("cuda"):
                import torch

                out["vram_after_load_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception as e:  # noqa: BLE001
            out["model_error"] = repr(e)
    out["ok"] = not [k for k in out if k.endswith("_error")]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


def cmd_replay(a):
    """Base speaks generic sentences in other (public) voices, x-vector only, the layout training uses."""
    import soundfile as sf

    voices = sorted(Path(a.voices).glob("*.wav"))
    if not voices:
        sys.exit("replay: no voices")
    sentences = [x.strip() for x in Path(a.sentences).read_text().splitlines() if x.strip()]
    tts = load_tts(a.base, a.device)
    (a.out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, text in enumerate(sentences):
        v = voices[n % len(voices)]
        wavs, sr = tts.generate_voice_clone(text=text, ref_audio=str(v), x_vector_only_mode=True)
        if sr != SR:
            sys.exit(f"replay: Base spoke at {sr} Hz")
        p = a.out / "wav" / f"replay-{n:04d}.wav"
        sf.write(p, wavs[0], sr)
        rows.append({"id": f"replay-{n:04d}", "audio": str(p), "text": text, "ref_audio": str(v), "replay": True})
    write_jsonl(a.out / "replay_raw.jsonl", rows)
    print(f"replay: {len(rows)} lines in {a.out}")


def cmd_prepare(a):
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    rows = read_jsonl(a.raw)
    bad = validate(rows, lambda p: sf.info(p).samplerate)
    if bad:
        sys.exit("prepare: fix these first:\n" + "\n".join(bad[:50]))
    tok = Qwen3TTSTokenizer.from_pretrained(a.tokenizer, device_map=a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / (Path(a.raw).stem.replace("_raw", "") + "_with_codes.jsonl")
    done = {r["id"] for r in read_jsonl(dst)} if dst.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with open(dst, "a") as f:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            enc = tok.encode([r["audio"] for r in chunk])
            for r, code in zip(chunk, enc.audio_codes):
                f.write(json.dumps({**r, "audio_codes": code.cpu().tolist()}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"prepare: {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"prepare: {dst}")


def loader(rows, tts, config, size, shuffle, seed, xvec, qwen_finetuning, style=False, games=None):
    """The official dataset/collate, plus each line's own x-vector. The official collate stacks every
    line's reference mel, which only works when all lines share one ref_audio, so it is handed one
    fixed clip and the x-vectors travel separately."""
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(qwen_finetuning))
    from dataset import TTSDataset

    fixed = rows[0]["ref_audio"]

    class WithVoice(TTSDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            row = self.data_list[idx]
            item["xvec"] = torch.tensor(xvec[row["_ref"]])
            if style:
                item["style_mel"] = mel(row["audio"])[0]
                item["game"] = games.get(row.get("game", ""), 0)
                item["audio_path"] = row["audio"]
            return item

        def collate_fn(self, batch):
            out = super().collate_fn(batch)
            out["xvec"] = torch.stack([b["xvec"] for b in batch])
            if style:
                out["style_mel"], out["style_len"] = pad_mels([b["style_mel"] for b in batch])
                out["game"] = torch.tensor([b["game"] for b in batch])
                out["audio_path"] = [b["audio_path"] for b in batch]
            return out

    data = [{**r, "_ref": r["ref_audio"], "ref_audio": fixed} for r in rows]
    ds = WithVoice(data, tts.processor, config)
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=size, shuffle=shuffle, generator=g, collate_fn=ds.collate_fn)


def forward(model, batch, voice):
    """The official forward pass with the alignment fixes; voice = [b, d] for the speaker slot.
    Returns (talker outputs, sub-talker loss)."""
    dev = model.device
    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    labels = batch["codec_0_labels"].to(dev)
    codec_mask = batch["codec_mask"].to(dev)
    emb = model.talker.model.text_embedding(input_ids[:, :, 0]) * batch["text_embedding_mask"].to(dev)
    cemb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"].to(dev)
    cemb[:, SPEAKER_SLOT, :] = voice.to(dev, cemb.dtype)
    emb = emb + cemb
    for i in range(1, 16):
        e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        emb = emb + e * codec_mask.unsqueeze(-1)
    out = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"].to(dev)[:, :-1],
                       labels=train_labels(labels), output_hidden_states=True)
    hidden = out.hidden_states[0][-1][predictor_hidden_mask(codec_mask)]
    _, sub_loss = model.talker.forward_sub_talker_finetune(codec_ids[codec_mask], hidden)
    return out, sub_loss


def cmd_train(a):
    import itertools

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    torch.manual_seed(a.seed)
    tts = load_tts(a.base, "cuda:0")
    model = tts.model
    config = AutoConfig.from_pretrained(a.base)
    rows = read_jsonl(a.data)
    val = read_jsonl(a.val) if a.val else []
    replay = read_jsonl(a.replay) if a.replay else []
    if a.kl and not replay:
        sys.exit("train: --kl needs --replay")
    bad = validate(rows + val + replay)
    if bad:
        sys.exit("train: " + "; ".join(bad[:20]))
    a.out.mkdir(parents=True, exist_ok=True)

    xvec = xvectors(model, [r["ref_audio"] for r in rows + val + replay], a.out / "xvectors.json")
    her = mean_voice([xvec[r["ref_audio"]] for r in rows])
    (a.out / "voice.json").write_text(json.dumps({"speaker": SPEAKER, "clips": len(rows), "xvector": her.tolist()}))
    if a.speaker == "mean":
        for r in rows + val:
            xvec[r["ref_audio"]] = her.tolist()
    style = a.speaker == "style"
    games = {g: i for i, g in enumerate(sorted({r.get("game", "") for r in rows + val}))}
    extras = any([a.style_shuffle, a.game_adv, a.prosody_aux, a.style_dropout])
    if extras and not style:
        sys.exit("train: --style-shuffle/--game-adv/--prosody-aux/--style-dropout need --speaker style")
    pros = {}
    if a.prosody_aux:
        cache = a.out / "prosody.json"
        pros = json.loads(cache.read_text()) if cache.exists() else {}
        for r in rows + val:
            if r["audio"] not in pros:
                pros[r["audio"]] = prosody_targets(r["audio"], r["text"])
        cache.write_text(json.dumps(pros))
        pv = np.array([pros[r["audio"]] for r in rows])
        pm, ps = pv.mean(0), pv.std(0) + 1e-6
        pros = {k: ((np.array(v) - pm) / ps).tolist() for k, v in pros.items()}

    base = None
    if a.kl:
        base = load_tts(a.base, "cuda:0").model  # frozen copy for the KL term
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.talker.parameters())
    for p in params:
        p.requires_grad_(True)
    her_t = torch.tensor(her, dtype=torch.float32, device=model.device)
    enc = game_emb = game_head = pros_head = None
    fresh = []  # new, randomly started parts: they get their own, much higher learning rate
    if style:
        enc = style_encoder(a.style_dim).to(model.device)
        game_emb = torch.nn.Embedding(len(games), her_t.shape[0]).to(model.device)
        torch.nn.init.zeros_(game_emb.weight)
        fresh = list(enc.parameters()) + list(game_emb.parameters())
        if a.game_adv:
            game_head = torch.nn.Sequential(torch.nn.Linear(a.style_dim, 64), torch.nn.GELU(),
                                            torch.nn.Linear(64, len(games))).to(model.device)
            fresh += list(game_head.parameters())
        if a.prosody_aux:
            pros_head = torch.nn.Linear(a.style_dim, 4).to(model.device)
            fresh += list(pros_head.parameters())
        (a.out / "style.json").write_text(json.dumps({"dim": a.style_dim, "games": games}))
    shuffle_gen = torch.Generator().manual_seed(a.seed + 7)
    chunk_frames = int(round(a.style_shuffle * SR / 256))  # mel hop 256
    # At the talker's 2e-6, a network that starts from scratch barely moves in ~500 steps (Adam moves each
    # weight about lr per step), so the style code would stay near zero and the slot would not steer.
    groups = [{"params": params, "lr": a.lr}] + ([{"params": fresh, "lr": a.style_lr}] if fresh else [])
    params = params + fresh
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    train_dl = loader(rows, tts, config, a.batch, True, a.seed, xvec, a.qwen_finetuning, style, games)
    val_dl = loader(val, tts, config, a.batch, False, a.seed, xvec, a.qwen_finetuning, style, games) if val else None
    replay_it = itertools.cycle(loader(replay, tts, config, a.batch, True, a.seed + 1, xvec, a.qwen_finetuning)) if a.kl else None

    def slot(batch, training=True):
        """The speaker slot for her lines, and the style extras' loss (0 when off or not training)."""
        if not style:
            return batch["xvec"], torch.zeros((), device=model.device)
        dev = model.device
        mels, lens = batch["style_mel"], batch["style_len"]
        if training and chunk_frames:
            mels = torch.stack([shuffle_chunks(m, n, chunk_frames, shuffle_gen) for m, n in zip(mels, lens)])
        code = enc.code(mels.to(dev), lens.to(dev))
        aux = torch.zeros((), device=dev)
        if training and game_head is not None:
            aux = aux + a.game_adv * torch.nn.functional.cross_entropy(game_head(grad_reverse(code)).float(), batch["game"].to(dev))
        if training and pros_head is not None:
            tgt = torch.tensor([pros[p] for p in batch["audio_path"]], device=dev)
            aux = aux + a.prosody_aux * torch.nn.functional.mse_loss(pros_head(code).float(), tgt)
        if training and a.style_dropout:
            keep = (torch.rand(code.shape[0], 1, device=dev) >= a.style_dropout).to(code.dtype)
            code = code * keep
        return her_t + enc.up(code) + game_emb(batch["game"].to(dev)), aux

    def her_loss(batch, training=True):
        voice, aux = slot(batch, training)
        out, sub = forward(model, batch, voice)
        return out.loss + 0.3 * sub, aux

    def replay_kl(batch):
        out, _ = forward(model, batch, batch["xvec"])
        with torch.no_grad():
            ref, _ = forward(base, batch, batch["xvec"])
        mask = train_labels(batch["codec_0_labels"]).to(model.device) != -100
        return kl_to_base(ref.logits, out.logits, mask)

    log = open(a.out / "train_log.jsonl", "a")
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        opt.zero_grad()
        for k, batch in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, aux = her_loss(batch)
                kl = replay_kl(next(replay_it)) if a.kl else torch.zeros((), device=model.device)
                total = (loss + aux + a.kl * kl) / a.accum
            total.backward()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                rec = {"epoch": epoch, "step": step, "loss": round(loss.item(), 4), "kl": round(kl.item(), 5), "aux": round(aux.item(), 5),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "s": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 10 == 0:
                    print(rec, flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
        rec = {"epoch": epoch, "end": True}
        if val_dl is not None:
            model.eval()
            if enc is not None:
                enc.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rec["val_loss"] = round(float(np.mean([her_loss(b, training=False)[0].item() for b in val_dl])), 4)
            model.train()
            if enc is not None:
                enc.train()
        ep = a.out / f"epoch-{epoch}"
        ep.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.talker.state_dict().items()},
                  str(ep / "talker.safetensors"))
        if style:
            save_file({**{f"enc.{k}": v.detach().cpu().contiguous() for k, v in enc.state_dict().items()},
                       "game": game_emb.weight.detach().cpu().contiguous()}, str(ep / "style.safetensors"))
        (ep / "tune.json").write_text(json.dumps({"base": a.base, "lr": a.lr, "kl": a.kl, "speaker": a.speaker,
                                                  "style_lr": a.style_lr if style else None, "style_shuffle": a.style_shuffle, "game_adv": a.game_adv,
                                                  "prosody_aux": a.prosody_aux, "style_dropout": a.style_dropout}))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if a.max_steps and step >= a.max_steps:
            break
    print(f"train: done in {time.time() - t0:.0f} s, {step} steps, checkpoints in {a.out}")


def voice_model(base, vec, talker_file, dst):
    """Base turned into a custom_voice model with the tuned talker and speaker `tali` = her mean
    x-vector (the official recipe's checkpoint layout)."""
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = Path(base) if Path(base).exists() else Path(snapshot_download(base))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg["tts_model_type"] = "custom_voice"
    t = cfg.setdefault("talker_config", {})
    t["spk_id"] = {SPEAKER: SPEAKER_ID}
    t["spk_is_dialect"] = {SPEAKER: False}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    state = {k: v for k, v in load_file(str(src / "model.safetensors")).items() if not k.startswith("speaker_encoder")}
    state.update({f"talker.{k}": v for k, v in load_file(str(talker_file)).items()})
    w = state["talker.model.codec_embedding.weight"]
    w[SPEAKER_ID] = torch.tensor(vec, dtype=w.dtype)
    save_file(state, str(dst / "model.safetensors"))
    return dst


def speak_slots(a, tune, voice, ep):
    """What goes in the speaker slot for each listening condition. Her mean voice always; with
    --style-from, the delivery of each given clip of hers, so the list shows whether the slot still
    steers her delivery (her x-vector of that clip, or the style encoder's reading of it)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    her = np.asarray(voice["xvector"], dtype=np.float64)
    style = tune.get("speaker") == "style"
    game_vec = np.zeros_like(her)
    enc = None
    if style:
        info = json.loads((ep.parent / "style.json").read_text())
        st = load_file(str(ep / "style.safetensors"))
        enc = style_encoder(info["dim"])
        enc.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith("enc.")})
        enc.eval()
        g = info["games"].get(a.game)
        if g is None:
            sys.exit(f"speak: game {a.game!r} not in {sorted(info['games'])}")
        game_vec = st["game"][g].float().numpy()
    out = [(f"her mean voice{f', {a.game} suit' if style else ''}", (her + game_vec).tolist())]
    if not a.style_from:
        return out
    if style:
        with torch.no_grad():
            for w in a.style_from:
                m = mel(w)
                off = enc(m, torch.tensor([m.shape[1]]))[0].numpy()
                out.append((f"delivery of {Path(w).stem}", (her + game_vec + off).tolist()))
    else:
        base = load_tts(tune["base"], a.device).model  # only for its speaker encoder
        with torch.no_grad():
            for w in a.style_from:
                m = mel(w).to(base.device).to(base.dtype)
                out.append((f"delivery of {Path(w).stem}", base.speaker_encoder(m)[0].float().cpu().tolist()))
        del base
    return out


def cmd_speak(a):
    import hashlib

    import soundfile as sf
    import torch

    run = a.run
    voice = json.loads((run / "voice.json").read_text())
    epochs = sorted(int(p.name.split("-")[1]) for p in run.glob("epoch-*") if (p / "tune.json").exists())
    if not epochs:
        sys.exit("speak: no checkpoints in the run")
    epoch = a.epoch if a.epoch is not None else epochs[-1]
    ep = run / f"epoch-{epoch}"
    tune = json.loads((ep / "tune.json").read_text())
    sentences = [x.strip() for x in Path(a.sentences).read_text().splitlines() if x.strip()]
    slots = speak_slots(a, tune, voice, ep)  # [(label, vector)]: her voice, plus one per --style-from clip
    vm = voice_model(tune["base"], voice["xvector"], ep / "talker.safetensors", run / f"voice-epoch-{epoch}")
    tts = load_tts(str(vm), a.device)
    row = tts.model.talker.model.codec_embedding.weight
    out = a.out or (run / f"speak-epoch-{epoch}")
    (out / "wav").mkdir(parents=True, exist_ok=True)
    clips, timing = [], []
    for j, (slot_label, vec) in enumerate(slots):
        with torch.no_grad():
            row[SPEAKER_ID] = torch.tensor(vec, dtype=row.dtype, device=row.device)
        for n, text in enumerate(sentences):
            t0 = time.time()
            wavs, sr = tts.generate_custom_voice(text=text, speaker=SPEAKER)
            p = out / "wav" / f"tuned-{j:02d}--{n:02d}.wav"
            sf.write(p, wavs[0], sr)
            timing.append({"slot": slot_label, "n": n, "s": round(time.time() - t0, 2), "audio_s": round(len(wavs[0]) / sr, 2)})
            clips.append((f"fine-tuned epoch {epoch}, no reference, {slot_label} | sentence {n}: {text}", p))
    if a.icl_ref:
        del tts  # one big model at a time
        base = load_tts(tune["base"], a.device)
        for n, text in enumerate(sentences):
            wavs, sr = base.generate_voice_clone(text=text, ref_audio=a.icl_ref, ref_text=a.icl_ref_text)
            p = out / "wav" / f"base-icl--{n:02d}.wav"
            sf.write(p, wavs[0], sr)
            clips.append((f"Base model with her clip as the example | sentence {n}: {text}", p))
    clips.sort(key=lambda c: hashlib.sha256(c[0].encode()).hexdigest())
    hub = ['title = "Fine-tuned voice: is this Tali, with no reference clip?"',
           'question = "Is this Tali with her suit and accent? Note short-line glitches or speed-ups."',
           'answers = ["her, suit and accent", "her, suit or accent partly", "not quite her", "not her or broken"]',
           "blind = true", 'asked_by = "Modeling Tali\'s suit effect"', ""]
    for label, p in clips:
        hub += ["[[clip]]", f"path = {json.dumps(str(p.relative_to(out)))}", f"label = {json.dumps(label)}", ""]
    (out / "clips.toml").write_text("\n".join(hub))
    write_jsonl(out / "timing.jsonl", timing)
    print(f"speak: {len(clips)} clips in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--base", default=BASE)
    c.add_argument("--no-model", action="store_true", help="imports and GPU only")
    r = sub.add_parser("replay")
    r.add_argument("--sentences", type=Path, required=True)
    r.add_argument("--voices", type=Path, required=True, help="24 kHz wavs of other (public) speakers")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--base", default=BASE)
    r.add_argument("--device", default="cuda:0")
    p = sub.add_parser("prepare")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--val", type=Path)
    t.add_argument("--replay", type=Path, help="replay_with_codes.jsonl, for --kl")
    t.add_argument("--kl", type=float, default=0.0, help="weight of the KL-to-Base term on replay lines")
    t.add_argument("--speaker", choices=["clip", "mean", "style"], default="clip")
    t.add_argument("--style-dim", type=int, default=16, help="--speaker style: numbers per clip's delivery")
    t.add_argument("--style-lr", type=float, default=1e-4,
                   help="--speaker style: learning rate of the style encoder, game vectors and helper heads")
    t.add_argument("--style-shuffle", type=float, default=0.0, help="--speaker style: shuffle the encoder's input in chunks of this many seconds")
    t.add_argument("--game-adv", type=float, default=0.0, help="--speaker style: weight of the gradient-reversed game classifier on the code")
    t.add_argument("--prosody-aux", type=float, default=0.0, help="--speaker style: weight of the pitch/loudness/pace head on the code")
    t.add_argument("--style-dropout", type=float, default=0.0, help="--speaker style: share of lines trained with the code zeroed")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--base", default=BASE)
    t.add_argument("--qwen-finetuning", type=Path, default=Path("Qwen3-TTS/finetuning"),
                   help="the official finetuning/ folder (for dataset.py), pinned commit 022e286b")
    t.add_argument("--lr", type=float, default=2e-6)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--max-steps", type=int, default=0, help="stop early (smoke)")
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("speak")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--sentences", type=Path, required=True)
    s.add_argument("--epoch", type=int)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", type=Path)
    s.add_argument("--icl-ref", help="her clip, for a Base-model ICL comparison")
    s.add_argument("--style-from", action="append", default=[], help="her clip whose delivery goes in the slot")
    s.add_argument("--game", default="le3", help="--speaker style runs: which game's suit")
    s.add_argument("--icl-ref-text", default="")
    a = ap.parse_args(argv)
    cmds = {"check": cmd_check, "replay": cmd_replay, "prepare": cmd_prepare, "train": cmd_train, "speak": cmd_speak}
    return cmds[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
