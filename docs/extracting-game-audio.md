# Extracting Tali's voice lines from Legendary Edition

Status: plan, not yet tested against real game files.

## Where the audio lives
- LE2 and LE3 store dialogue as Wwise audio inside `.afc` files, referenced by `WwiseStream` exports in the game's `.pcc` packages. LE1 stores it differently (embedded in packages).
- Dialogue is 24 kHz mono Wwise Vorbis.
- Packages are Oodle-compressed in the Legendary Edition. The pure-Rust `oozextract` crate can decompress Oodle Kraken, so a native Linux extractor is possible.
- Subtitle text lives in `.tlk` string tables, keyed by string ID, which lets each clip be paired with its exact transcript.

## Route A: Legendary Explorer under Proton (fastest to try)
Legendary Explorer (ME3Tweaks) is the community's modding toolkit. Its Dialogue Editor can dump all audio for one speaker. It is a Windows .NET app; people report ME3Tweaks tools running under Proton Experimental with .NET installed in the prefix, but Legendary Explorer on Linux is unconfirmed.

## Route B: native Rust extractor (`vi extract`)
1. Find `.pcc`, `.afc` and `.tlk` files under the game folder.
2. Decompress packages (Oodle via `oozextract`) and list `WwiseStream` exports whose names mark Tali as the speaker.
3. Copy each stream's bytes out of its `.afc` and decode with `vgmstream-cli` (in nixpkgs as `vgmstream`).
4. Look up the subtitle text in the `.tlk` and write `extracted/tali/<id>.wav` plus a `lines.jsonl` index with transcript and source.
5. Pick the best clip per emotion into `characters/tali/voice/`.

This needs iteration against the real files, so it is best built on the machine where the game is installed.

Either way, everything extracted stays in `extracted/` and `characters/*/voice/`, which git ignores.
