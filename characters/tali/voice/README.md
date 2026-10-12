# Tali voice clips

This folder holds the reference clips listed in `../character.toml`. They come from your own copy of Mass Effect Legendary Edition and are ignored by git, so they never leave your machine.

For each emotion, pick one clip that is:
- 8 to 20 seconds of Tali alone, with no music, effects or other speakers
- clearly in that emotion
- saved as mono WAV (the game's dialogue is 24 kHz mono, keep it as is)

Then paste the exact spoken words into that clip's `transcript` in `character.toml` and run:

    cargo run -p vi -- pack check characters/tali

See `docs/extracting-game-audio.md` for how to get the clips out of the game.
