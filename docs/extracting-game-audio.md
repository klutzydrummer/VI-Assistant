# Extracting Tali's voice lines from Legendary Edition

Status: LE2/LE3 extraction works on the real install (`vi extract`); LE1 audio not yet.

## Where the audio lives
- LE2 and LE3 store dialogue as Wwise audio inside `.afc` files, referenced by `WwiseStream` exports in the game's `.pcc` packages. LE1 has no Wwise: its dialogue is ISACT banks (usually Ogg Vorbis samples) inside `BioSoundNodeWaveStreamingData` exports in packages.
- Dialogue is 44.1 kHz mono Wwise Vorbis; the corpus stores it resampled to 24 kHz.
- Packages are Oodle-compressed in the Legendary Edition. The pure-Rust `oozextract` crate decompresses Oodle Leviathan, the codec LE uses, so a native Linux extractor is possible.
- Subtitle text lives in string tables keyed by string ID (standalone `.tlk` files in LE2/LE3, `BioTlkFile` exports inside packages in LE1), which lets each clip be paired with its exact transcript.

## Route A: Legendary Explorer under Proton (fastest to try)
Legendary Explorer (ME3Tweaks) is the community's modding toolkit. Its Dialogue Editor can dump all audio for one speaker. It is a Windows .NET app; people report ME3Tweaks tools running under Proton Experimental with .NET installed in the prefix, but Legendary Explorer on Linux is unconfirmed.

## Route B: native Rust extractor (`vi extract`)
The chosen route; the full plan is [design/harvester-plan.md](design/harvester-plan.md). The package and subtitle readers are in `crates/harvest`, and `vi harvest scan|package|tlk` exercises them on an install. For LE2/LE3, with `vgmstream-cli` on `PATH`:

```
vi extract --character characters/tali "<LE root>" --out corpus/tali/harvest
vi corpus check corpus/tali/harvest
```

1. Find `.pcc`, `.afc` and `.tlk` files under the game folder.
2. Decompress packages (Oodle via `oozextract`) and list `WwiseStream` exports whose names mark Tali as the speaker.
3. Copy each stream's bytes out of its `.afc` and decode with `vgmstream-cli` (in nixpkgs as `vgmstream`).
4. Look up the subtitle text in the `.tlk` and write `extracted/tali/<id>.wav` plus a `lines.jsonl` index with transcript and source.
5. Pick the best clip per emotion into `characters/tali/voice/`.

This needs iteration against the real files, so it is best built on the machine where the game is installed.

Either way, everything extracted stays in `extracted/` and `characters/*/voice/`, which git ignores.
