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

Talk to a character through the full turn pipeline (placeholder dialogue and printed voice until the models are connected):

    cargo run -p vi -- chat characters/tali

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
- `crates/vi`: the command-line tool
- `jobs/`: scripts that run on the worker, including `on-worker.sh`, which runs a dataset stage there
- `docs/`: design notes, including how to extract game audio
