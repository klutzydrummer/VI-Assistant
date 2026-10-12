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
- `voice/`: reference clips, extracted locally and never committed

Swapping characters means pointing at a different folder. No fine-tuning is needed.

Check a pack:

    cargo run -p vi -- pack check characters/tali

## Layout
- `crates/charpack`: loads and validates character packs
- `crates/vi`: the command-line tool
- `docs/`: design notes, including how to extract game audio
