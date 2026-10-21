# VI Assistant

A conversational assistant that embodies a character: their persona, lore and voice. Point it at a character pack and it talks like them, with a cloned voice and emotion, and can send out agents to handle tasks. The pilot character is Tali'Zorah from Mass Effect.

Everything runs on hardware you control: a rented GPU today, your own later. No hosted model APIs.

## Status
Stage 1 in progress: Tali replies in character, with voice, from typed text.

## Character packs
A character is a folder under `characters/` (see `characters/tali/`):
- `character.toml`: names, file paths, and voice reference clips per emotion
- `persona.md`: the system prompt that defines who they are and how they speak
- `examples.md`: example exchanges in their voice
- `states.toml` and `states/`: the state machine (continuous mood and intoxication, derived states, and the mixing table to control vectors) and per-state prompt overlays
- `drones.toml`: the names the character gives the agents it sends out
- `voice/`: reference clips, extracted locally and never committed

Swapping characters means pointing at a different folder. No fine-tuning is needed.

Check a pack:

    cargo run -p vi -- pack check characters/tali

Talk to a character through the full turn pipeline (placeholder dialogue and printed voice):

    cargo run -p vi -- chat characters/tali

Or with a real dialogue model: any chat GGUF, run in-process by llama.cpp and steered by the control vectors in a folder (one `cvector-generator` file per vector name in the pack's `llm` mixing table, plus optional limits in `vectors.toml`; see `crates/llm/src/vectors.rs`):

    cargo run --release -p vi -- chat characters/tali --model qwen3-8b.gguf --vectors vectors/qwen3-8b

Building needs cmake and a C++ compiler, since llama.cpp is compiled from source. It runs on the CPU by default; add `--features cuda` (rented GPU) or `--features vulkan` (RX 480) for a GPU backend.

### Vulkan (RX 480 at home)
A Vulkan build also needs `glslc` and the Vulkan and SPIR-V headers. On NixOS, from the repo:

    nice -n 19 nix-shell -p cargo rustc gcc pkg-config openssl cmake clang llvmPackages.libclang vulkan-headers vulkan-loader shaderc.bin spirv-headers --run 'export LIBCLANG_PATH=$(dirname $(find $(nix-build --no-out-link "<nixpkgs>" -A llvmPackages.libclang.lib) -name "libclang.so*" | head -1)); nice -n 19 cargo build --release -p vi --features vulkan'

- Every build at home must pass `--features vulkan`, `cargo run` included. A build without it quietly
  replaces `target/release/vi` with a CPU-only one (and recompiles llama.cpp, about 8 minutes).
- A GPU build offloads every layer by default; `--gpu-layers 0` runs on the CPU, `--gpu-layers N`
  offloads N layers. 8 GB holds models up to about 8B at Q4; Qwen3.8-27B doesn't fit.
- `vi vectors selftest` captures on the GPU and checks it against the CPU (the RX 480 passed on
  2026-10-21, worst relative error 2.1e-4). A capture store records where it ran (`backend` in
  `capture.json`) and won't resume on another backend.
- Only llama.cpp uses the GPU. The Python models (Qwen3-TTS, emotion2vec, Whisper) have no Vulkan
  path on the RX 480 and stay on the CPU or the rental.
- `.github/workflows/vulkan.yml` builds it and runs the selftest and builder tests on Mesa's software
  Vulkan driver, without a GPU.

## Cloud worker
Heavy models and dataset jobs run on a rented Vast.ai GPU that releases itself when idle:

    cargo run -p vi -- cloud offers
    cargo run -p vi -- cloud up
    cargo run -p vi -- cloud down

See `docs/cloud-worker.md`.

## Layout
- `crates/charpack`: loads and validates character packs, and runs their state machines
- `crates/engine`: the turn pipeline: director, prompt assembly, sentence streaming to the voice
- `crates/cloud`: rents, watches and releases the Vast.ai GPU worker
- `crates/llm`: the dialogue model: llama.cpp embedded, with control vectors blended per turn
- `crates/vi`: the command-line tool
- `jobs/`: scripts that run on the worker, including `on-worker.sh`, which runs a dataset stage there
- `docs/`: design notes, including how to extract game audio
