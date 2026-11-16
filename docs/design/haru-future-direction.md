# Future direction: Haru and a thinner chat layer

Status: **future direction, under consideration.** Nothing here is current work, approved, scheduled or funded. It records Brandon's ideas (2026-11-16) so they are not lost. Current work is in `docs/project-state.md`.

## What Brandon has said

- Once Tali's voice and the chat model both work, we could generate a large synthetic dataset and fine-tune [Haru](https://github.com/higurashi-ai/Haru), a full-duplex speech model. A KL-divergence penalty would guard against catastrophic forgetting.
- He also wants to consider replacing the interim chat model with a smaller one. The possible end goal is a small, specialised layer between the human and a more powerful AI model: personality and flavour as a thin layer over a more capable system.

Both are ideas to weigh, not decisions. No model has been chosen, and no run is planned.

## What we know about Haru (checked 2026-11-16)

From the [GitHub README](https://github.com/higurashi-ai/Haru), the [model card](https://huggingface.co/higurashi-ai/Haru-4B) and the [technical blog](https://blog.higurashi.ai/melte/haru4b/):

- **License:** Apache-2.0 for code and weights. It is not a non-commercial license. Third-party notices are in the repo's `THIRD_PARTY_NOTICES.md`.
- **Full-duplex:** it listens and speaks in the same frame (133 ms).
- **Size:** the model card names a Qwen3.5-4B text backbone, but the Hub metadata shows 5B parameters. The sources do not reconcile the two, so the exact total is **unverified**.
- **Input and output:** user audio at 24 kHz in; model audio plus text out.
- **Hardware for inference:** an NVIDIA GPU with about 12 GB of memory (README). The home RX 480 has 8 GB and is not NVIDIA, so Haru would need a rented or other GPU. This is an **open question**; no rental or spend is authorised.
- **Training code and fine-tune recipe: not released.** The blog says training code and a technical report are planned. Fine-tuning would need that recipe or one we build ourselves.
- **Base model, not instruction-tuned.** The model card says it does not follow instructions like an assistant. It is not a drop-in assistant.
- **Unverified:** how well Haru fine-tunes, how much data it needs, and whether the KL penalty works the way we would want. The blog reports no quantitative benchmarks.

## Rules that carry over

- **Raw, unsupervised learning.** The legacy hand-labelled mood data must not be used to filter, weight, prompt, score or select data or models.
- **Copyright.** Tali's game audio and text are copyrighted. Synthetic data derived from them stays on Brandon's PC and the rented worker only. It must not be published, put on a third-party dataset host (Hugging Face included) or committed to this public repo.
- **Listening decides.** Any run that produces voice output makes a small, clearly labelled listening set, and human listening decides voice quality, not an automatic score.
- **Spending.** Every paid run needs Brandon's explicit approval for that run and its budget.

## Open questions

- Which GPU would run Haru, and what would it cost?
- Is a smaller chat model good enough for the thin personality layer, and which one?
- How would the thin layer hand work to a stronger model, and which stronger model (this project allows only self-hosted models, apart from Claude for text-only labelling)?
