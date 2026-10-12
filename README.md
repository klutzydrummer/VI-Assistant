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

## Layout
- `crates/charpack`: loads and validates character packs, and runs their state machines
- `crates/engine`: the turn pipeline: director, prompt assembly, sentence streaming to the voice
- `crates/llm`: the dialogue model: llama.cpp embedded, with control vectors blended per turn
- `crates/vi`: the command-line tool
- `docs/`: design notes, including how to extract game audio
