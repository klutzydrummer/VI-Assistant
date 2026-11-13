# Full fine-tune support: OmniVoice vs MOSS-TTS-v1.5

Research date: 2026-11-12. Sources: official GitHub repos and Hugging Face model cards only. Anything else is marked community or inference.

## 1. k2-fsa/OmniVoice

| Claim | Source |
|---|---|
| Training pipeline lives in `examples/`, covering data prep, training, evaluation and fine-tuning | https://github.com/k2-fsa/OmniVoice (README) |
| Training doc exists; fine-tuning from a checkpoint is set with `"init_from_checkpoint"` | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |
| Full-model training is the default; LoRA only if `"use_lora": true` is set | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |
| LoRA recipe doc is `docs/lora_finetuning.md` (linked, not read) | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |
| Default LLM backbone is Qwen3-0.6B; default 300,000 steps, lr 1e-4 | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |
| Code licence: Apache 2.0 | https://github.com/k2-fsa/OmniVoice/blob/master/LICENSE |
| Pre-trained weights licence: CC-BY-NC (non-commercial), because of training data (Emilia) | https://huggingface.co/k2-fsa/OmniVoice (License section) |
| Weights carry a usage prohibition on unauthorised cloning, impersonation and fraud | https://huggingface.co/k2-fsa/OmniVoice |
| Training code is released (`omnivoice.cli.train`, configs under `examples/config/`) | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |
| Emotion/style labels: none named as required. Voice design uses speaker attributes (gender, age, pitch, accent, whisper) | https://github.com/k2-fsa/OmniVoice (README); https://huggingface.co/k2-fsa/OmniVoice |
| Flex attention needs PyTorch 2.5+ and Ampere or newer. Otherwise use `"sdpa"` | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |
| Example launch uses 8 GPUs. No VRAM figure given | https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md |

**Official fine-tune recipe:** yes, `docs/training.md` plus `examples/`. Full fine-tune by default. LoRA optional.

**Data format:** not confirmed. The `examples/config/data_config_emilia.json` file was not read. Expect JSONL of audio plus transcript; the exact schema is unverified.

**Blockers and gaps:**
- No stated VRAM for fine-tuning.
- The data schema is unverified.
- Weights are CC-BY-NC. Private, non-commercial use is the only reading that fits. Fine-tuned weights inherit this.
- Official material says the model is primarily trained on voice cloning. Voice design is trained only on Chinese and English.
- Tokenizer and codec trainability are not stated on the pages read.

## 2. OpenMOSS-Team/MOSS-TTS-v1.5

| Claim | Source |
|---|---|
| Official fine-tune recipe for MOSS-TTS-v1.5 (MossTTSDelay, 8B): `moss_tts_delay/finetuning/` with `sft.py`, `prepare_data.py`, `run_train.sh` | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md |
| Recipe is supervised fine-tuning (SFT). Full-parameter. No LoRA option documented | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md |
| Data format: JSONL, one record per line, with `audio` (required), `text` (required), `language` (recommended), optional `ref_audio` | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md |
| Emotion/style labels not required. Optional fields: `instruction`, `ambient_sound`, `quality`, `sound_event` | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md |
| For the 8B model, FSDP or DeepSpeed ZeRO-3 is recommended over single-card training. ZeRO-3 needs `deepspeed` | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md |
| Installed via `pip install -e ".[torch-runtime,finetune]"` | https://github.com/OpenMOSS/MOSS-TTS (README, summary only) |
| Code licence: Apache 2.0 | https://github.com/OpenMOSS/MOSS-TTS/blob/main/LICENSE |
| Weights licence: Apache 2.0 (HF sidebar `apache-2.0`) | https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5 |
| Training code is released (`sft.py`, `dataset.py`, accelerate configs) | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md |
| Second recipe for MOSS-TTS-Local-Transformer-v1.5 (4B): `moss_tts_local_v1.5/finetuning/`, DDP or ZeRO-3. Not the v1.5 base model | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_local_v1.5/finetuning/README.md |
| Recipe B fixes codec dtype to fp32. Does not document multi-speaker training | https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_local_v1.5/finetuning/README.md |
| No VRAM figure stated in either recipe | Both recipe READMEs above |

**Official fine-tune recipe:** yes, `moss_tts_delay/finetuning/`. Full SFT. No LoRA route documented.

**Data format:** JSONL with `audio`, `text`, `language`. Emotion or style labels are not required.

**Blockers and gaps:**
- No stated VRAM. The 8B model is directed to FSDP or ZeRO-3 sharding, so single-card training is unlikely to fit at 8 GB. Treat that as inference.
- The README was truncated in the fetch. Its later sections came back as a summary, not verbatim. The fine-tune links were read directly.
- The `moss_tts_realtime` and `moss_tts_local` recipes were not read.
- Recipe A warns: mixing MOSS-TTSD base files with the v1.5 base makes training loss look normal while inference collapses.

## Uncertainty

- The MOSS-TTS README was truncated; the fine-tune pointers came from a summary and were then verified by direct fetch.
- OmniVoice data schema and VRAM were not verified. The fine-tune recipe is inferred from `training.md`, which links further pages not read.
- Model card licences were read from the HF pages. The OmniVoice licence split (code Apache, weights CC-BY-NC) is taken from the model card text.

## Verdict

1. **MOSS-TTS-v1.5:** official full SFT recipe, Apache 2.0 code and weights, JSONL with no emotion labels required. Best fit on licence and labels. VRAM for the 8B model is unstated and likely needs multi-GPU sharding.
2. **OmniVoice:** official full fine-tune via `docs/training.md`, with LoRA optional. Pre-trained weights are CC-BY-NC, so fine-tuned weights must stay non-commercial. Data schema and VRAM unverified.
3. **Flags:** neither recipe requires emotion labels. Both need VRAM confirmation before any rental, and OmniVoice's licence needs a decision on whether non-commercial private use is acceptable.
