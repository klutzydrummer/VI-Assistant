"""pairingcheck: does voicetune.py hand the untouched Base model the inputs and targets it was trained on?
(2026-11-02. Home CPU, one model, bf16, no training. The diagnostic that found the ~12.6 starting loss,
and now confirms the fix.)

Teacher-forces Base on a few replay lines (LibriTTS-R voices spoken by Base itself) and reports
cross-entropy per codebook under each convention in play:
  - text input: voicetune/official training (raw text_embedding) vs generation (text_projection of it)
  - talker target: voicetune (logits at p predict the frame at p+1) vs official sft_12hz (frame two ahead)
  - code-predictor hidden: voicetune (position before the frame) vs official (the frame's own position)
  - code-predictor loss: the model's own loss_function (which shifts again) vs aligned (logit k vs code k)
A convention Base was trained with gives a low loss; a wrong one sits near or above uniform guessing.
`--mode voicetune` skips the hand-built comparison and reports only voicetune.forward's own two
losses (the shipped code): the talker loss and the aligned sub-talker loss.

    python3 jobs/pairingcheck.py --data replay_with_codes.jsonl --root <dir the paths are relative to>
                                 --base <Base snapshot> --qwen-finetuning <official finetuning/ folder> [--n 5]
"""

import argparse
import json
import math
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import voicetune  # noqa: E402


def ce(logits, targets):
    """Mean cross-entropy (nats) of [n, v] logits against [n] targets, in float32."""
    import torch.nn.functional as F

    return F.cross_entropy(logits.float(), targets).item()


def main(argv=None):
    import torch
    from qwen_tts import Qwen3TTSModel
    from transformers import AutoConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--qwen-finetuning", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--mode", choices=["compare", "voicetune"], default="compare",
                    help="compare: every convention side by side; voicetune: just voicetune.forward's two losses")
    a = ap.parse_args(argv)

    torch.set_grad_enabled(False)
    rows = voicetune.read_jsonl(a.data)[: a.n]
    for r in rows:
        r["audio"], r["ref_audio"] = str(a.root / r["audio"]), str(a.root / r["ref_audio"])
    tts = Qwen3TTSModel.from_pretrained(a.base, device_map="cpu", dtype=torch.bfloat16, attn_implementation="sdpa")
    model = tts.model.eval()
    talker = model.talker
    config = AutoConfig.from_pretrained(a.base)
    xvec = {}
    for r in rows:
        if r["ref_audio"] not in xvec:
            xvec[r["ref_audio"]] = model.speaker_encoder(voicetune.mel(r["ref_audio"]).to(model.dtype))[0].float().tolist()
    dl = voicetune.loader(rows, tts, config, 1, False, 0, xvec, a.qwen_finetuning)

    acc = {}

    def add(key, value, weight):
        s, w = acc.get(key, (0.0, 0))
        acc[key] = (s + value * weight, w + weight)

    for batch in dl:
        # 1. voicetune.forward exactly as training calls it: the two losses the fix targets.
        out, sub = voicetune.forward(model, batch, batch["xvec"])
        n_frames = int(batch["codec_mask"].sum())
        add("voicetune talker loss (out.loss)", out.loss.item(), n_frames)
        add("voicetune sub-talker loss (aligned)", sub.item(), n_frames)
        add("voicetune total (talker + 0.3 sub)", out.loss.item() + 0.3 * sub.item(), n_frames)

        if a.mode != "compare":
            continue

        # 2. the same pass, built by hand, with and without text_projection
        ids, codec_ids = batch["input_ids"], batch["codec_ids"]
        mask = batch["codec_mask"]
        pos = torch.nonzero(mask[0]).squeeze(1)  # input positions of frames 0..F-1
        codes = codec_ids[0, pos]  # [F, 16]
        for text_mode in ("raw", "projected"):
            temb = talker.model.text_embedding(ids[:, :, 0])
            if text_mode == "projected":
                temb = talker.text_projection(temb)
            emb = temb * batch["text_embedding_mask"]
            cemb = talker.model.codec_embedding(ids[:, :, 1]) * batch["codec_embedding_mask"]
            cemb[:, voicetune.SPEAKER_SLOT, :] = batch["xvec"].to(cemb.dtype)
            emb = emb + cemb
            for i in range(1, 16):
                emb = emb + talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i]) * mask.unsqueeze(-1)
            o = talker(inputs_embeds=emb[:, :-1], attention_mask=batch["attention_mask"][:, :-1], output_hidden_states=True)
            hid = o.hidden_states[0][-1][0]  # [T-1, D]
            logits = o.logits[0]
            if text_mode == "raw":
                add("check: codec_head(hidden_states[-1]) == logits (max abs diff)",
                    (talker.codec_head(hid) - logits).abs().max().item(), 1)
            eos = config.talker_config.codec_eos_token_id
            tgt0 = torch.cat([codes[:, 0], torch.tensor([eos])])  # frames 0..F-1 then eos
            src = torch.cat([pos - 1, pos[-1:]])  # the position before each target
            add(f"talker cb0 CE, next frame (generation) [{text_mode} text]", ce(logits[src], tgt0), len(tgt0))
            two = torch.cat([codes[1:, 0], torch.tensor([eos])])  # what official labels[:, 1:] + HF shift trains
            add(f"talker cb0 CE, frame two ahead (official sft) [{text_mode} text]", ce(logits[pos - 1], two), len(two))
            for hname, hpos in (("before frame (voicetune)", pos - 1), ("at frame (official)", pos)):
                h = hid[hpos.clamp(max=hid.shape[0] - 1)]
                sub_logits, sub_loss = talker.forward_sub_talker_finetune(codes, h)  # logits [F, 15, V]
                add(f"sub-talker as built (loss_function) [{hname}, {text_mode} text]", sub_loss.item(), len(codes))
                shifted = ce(sub_logits[:, :-1].reshape(-1, sub_logits.shape[-1]), codes[:, 2:].reshape(-1))
                add(f"  = manual shifted CE (logit k vs code k+1) [{hname}, {text_mode} text]", shifted, len(codes))
                for k in range(15):
                    add(f"sub-talker cb{k + 1:02d} CE aligned [{hname}, {text_mode} text]",
                        ce(sub_logits[:, k], codes[:, k + 1]), len(codes))
                aligned = ce(sub_logits.reshape(-1, sub_logits.shape[-1]), codes[:, 1:].reshape(-1))
                add(f"sub-talker mean CE aligned [{hname}, {text_mode} text]", aligned, len(codes))
        print(f"line done: {n_frames} frames, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB",
              flush=True)

    print(f"\nuniform guess: talker cb0 ln({config.talker_config.vocab_size}) = {math.log(config.talker_config.vocab_size):.2f},"
          f" sub-talker ln(2048) = {math.log(2048):.2f}  (nats)")
    for k, (s, w) in acc.items():
        print(f"{s / w:9.4f}  {k}")
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB")


if __name__ == "__main__":
    main()
