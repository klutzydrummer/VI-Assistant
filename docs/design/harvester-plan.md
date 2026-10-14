# Harvester plan (dataset stage 1)

Status: 2026-10-12. Readers for milestones 1 to 3 (inventory, packages with Oodle decompression, both TLK layouts) are in `crates/harvest`; try them with `vi harvest scan|package|tlk`. `vi extract` covers milestones 4 to 6 for all three games. In LE2/LE3, clips and subtitles are found through stream names; in LE1, through conversation nodes and ISACT banks. The conversation reader (`harvest::conversation`) gives each line's node, listener, `prev`/`next`, context and LE3 stage direction, and each conversation's graph is written to `conversations/`. Scene tags come from `harvest::scenes` (see below); `vi corpus tag` recomputes them in an existing harvest folder without touching audio.

Checked on Brandon's install (Steam, all three games plus DLC):
- All 19,430 package headers parse. The compression type is `0x400` (Leviathan); a few hundred packages are uncompressed. Every package but one decompresses with `oozextract`. The exception is LE3 `SFXImages_Codex_4.pcc`, where the last block of chunk 13 (306 bytes packed, 462 unpacked) fails inside `oozextract` while the four full blocks before it decode. Likely an `oozextract` bug on tiny blocks; it's a codex image package, so the harvester doesn't need it.
- Export table entries have no component map, and export data starts with a 4-byte net index: property tags walk cleanly to the binary part on every `BioConversation` sampled in all three games. LE1 bool tags have a value size of 0 with the 1-byte value in the tag.
- All 719 LE2/LE3 `.tlk` files and the `BioTlkFile` exports in all 490 LE1 `_LOC_INT` packages decode to the right text. LE2/LE3 female tables hold a single entry each; gendered lines mostly have their own string ids.
- LE1 `_LOC_INT` packages carry every language, named `<table>.tlk`, `<table>.tlk_M`, `<table>.tlk_DE`, `<table>.tlk_M_DE` and so on. `_M` is male Shepard, no suffix is female (the Czech tables use "velitelko" in the plain one and "veliteli" in `_M`). The English pair is identical in all 490 `_LOC_INT` packages and in `Startup_INT.pcc`.
- Dialogue `WwiseStream` names: LE2 `en_us_<speaker tag>_<conversation>_<8-digit string id>_<m|f>_wav` (the conversation is also the `BankName` property, which is how tag and conversation are split); LE3 `en-us,<speaker tag>,<conversation>,<8-digit string id>_<m|f>_wav`. Tali is `hench_tali` in both games.
- `BioConversation` nodes are tagged-property structs all the way down (`m_EntryList`, `m_ReplyList`, `ReplyListNew`, `m_SpeakerList`, `m_aStageDirections`); only struct-versus-plain arrays need hardcoding. All 6,024 conversations in the `_LOC_INT` packages of the three games parse. LE2/LE3 conversations sit in the same `_LOC_INT` package as their streams. LE3 has 6,830 stage directions, and they read like voice-direction notes ("Quiet sad delivery. Not flippant here.").
- LE1 audio is simpler than the FaceFX route below. Each conversation has an ISACT bank on disk, `BioGame/Content/Packages/ISACT/<conversation>.isb` (language variants are `_de`, `_fr` and so on), whose streams are named `EN_<conversation>_<8-digit string id>_<M|F>.wav`: 48 kHz Ogg Vorbis, which vgmstream decodes by subsong. The `BioSoundNodeWaveStreamingData` export inside the package holds only the bank's headers. Female-Shepard variants (`_F`) exist only where the recording differs.
- LE1 speaker tags for Tali: `hench_quarian` (squad lines) and `sta20_quarian` (recruitment). The conversations she owns, whose speaker-index -1 lines are hers, are `cha00_quarian_tali` (Normandy), `sta60_quarian` (Fist's ambush) and `jug20_amb_tali` (Virmire ambient lines). The streams don't name speakers, so these come from the census and live in `characters/tali/sources.toml`.
- `WwiseStream` binary ends with the 16-byte bulk header (flags 1, element count, size on disk, offset into the `.afc`); LE2 has 8 more bytes in front. The bytes at the offset are a complete RIFF `.wem`, Wwise Vorbis at **44.1 kHz** mono, not 24 kHz. vgmstream decodes them.

Goal: on Brandon's NixOS machine, read the Mass Effect Legendary Edition install and write every Tali line as a clip plus a record: audio, subtitle (via TLK string id), speaker, surrounding lines, and scene tags. This is stage 1 of [control-vector-datasets.md](control-vector-datasets.md). Output follows [corpus-format.md](corpus-format.md) v1.

## How the game stores what we need

Sources: the LegendaryExplorerCore (LEC) source in [ME3Tweaks/LegendaryExplorer](https://github.com/ME3Tweaks/LegendaryExplorer), read directly (file paths below are under `LegendaryExplorer/LegendaryExplorerCore/`), plus vgmstream's format list. These are findings from source code, not from running anything.

### Install layout
`<LE root>/Game/ME1|ME2|ME3/BioGame/CookedPCConsole/*.pcc`, DLC in `BioGame/DLC/<name>/CookedPCConsole/` (`GameFilesystem/LE3Directory.cs`). LE ships DLC as plain folders, so no SFAR unpacking (that was an original-trilogy problem). Localized content lives in `*_LOC_INT.pcc` packages and INT TLKs.

### Packages (.pcc)
- Unreal Engine 3 packages: header, name table, import table, export table, then export data. Versions: LE1 684/171, LE2 684/168, LE3 685/205 (`Packages/MEPackage.cs`).
- LE compresses packages with **Oodle Leviathan** (`MEPackage.cs:983`), not Kraken as earlier notes said. Data is split into chunks, each split into blocks, each block compressed separately (`Packages/CompressionHelper.cs`).
- LEC decompresses by loading the game's own Windows DLL (`oo2core_8_win64.dll`, `Compression/OodleHelper.cs`). That DLL can't load natively on Linux. **`oozextract`** (crates.io, MIT, pure Rust, v0.5.5, updated 2026-08) decodes Leviathan, so a native extractor needs no Windows code at all.
- Export properties are UE3 tagged properties. They are self-describing except that array elements don't record their type, so the reader needs a small hardcoded schema for the handful of classes we touch (below). That is the main implementation risk, and it's bounded.

### Subtitles (TLK)
- LE2 and LE3: standalone Huffman-compressed `.tlk` files (`BIOGame_INT.tlk`, `BioGame_Update_INT.tlk`, one per DLC like `DLC_EXP_Pack003_INT.tlk`). Format in `TLK/ME2ME3/`.
- LE1: TLKs are `BioTlkFile` exports **inside packages** (global ones in `Startup_INT.pcc`, plus per-level `_LOC_INT` packages), same Huffman idea, different layout (`TLK/ME1/`, `TLK/LE1Talkfiles.cs`). Lookup checks the line's own package first, then the global tables.
- TLKs carry male and female variants of a string; record which one was used.

### Conversations
`BioConversation` exports hold the dialogue graph (`Dialogue/ConversationExtended.cs`):
- `m_EntryList`: lines spoken by NPCs. Each node has `srText` (TLK string id), `nSpeakerIndex`, `ReplyListNew` (links to replies), conditionals and plot transitions.
- `m_ReplyList`: Shepard's lines, each with `EntryList` (links onward) and `ReplyType`.
- `m_StartingList`: entry points.
- Speakers: `m_SpeakerList` structs with `sSpeakerTag` in LE1/LE2, a plain `m_aSpeakerList` of names in LE3. Index -1 is the conversation owner, -2 is Shepard.
- **LE3 bonus:** `m_aStageDirections` has the voice director's notes per line (`srStrRef` plus `sText`). If these turn out to read like "sad, quiet" they're a strong free label prior for the label stage. **Verify** what they contain.
- The Dialogue Dumper tool in Legendary Explorer dumps exactly speaker, string id, line, conversation, game, file, object number for every line. Its logic is the reference for which packages to scan (it skips `*LAY`, `*SND`, `*_T`, `BIOG*`, `BIOC*` in game 1, and non-localized files in games 2 and 3).

### Audio
- **LE2 and LE3 (Wwise 2019.1.6):** each line has `WwiseStream` exports named with the string id and a gender suffix (`..._<strref>_m` / `_f`), which is how LEC matches line to audio (`parseLinesAudioStreams`). The stream's `Filename` property names an `.afc` file and its binary gives `DataOffset` and `DataSize` (`Unreal/BinaryConverters/WwiseStream.cs`). The bytes at that offset are a complete RIFF `.wem` file, Wwise Vorbis (codec 0xFFFF). The stream names also carry the speaker tag (expected `en_us_hench_tali_...`), which resolves "owner" lines without walking level scripts. **Verify** the naming on real files.
- **LE1 (no Wwise; ISACT, like the original ME1):** audio sits in `BioSoundNodeWaveStreamingData` exports inside packages, each holding an ISACT ICB+ISB bank pair; dialogue samples are usually Ogg Vorbis inside the ISB (`Unreal/BinaryConverters/BioSoundNodeWaveStreamingData.cs`, `Sound/ISACT/`). Linking a line to its sample goes through FaceFX: the conversation's per-speaker FaceFX animset has a line named `FXA_<strref>_M/F` whose index points into the animset's `ReferencedSoundCues`, which lead to the `SoundNodeWave` and its streaming data. **Verify**: this path is inferred from LEC's data structures, and LE1 is the least certain part of the plan.
- **Decoding:** vgmstream handles both: Wwise RIFF (Vorbis_custom) and Creative ISACT `.isb` with Ogg Vorbis. It's in nixpkgs as `vgmstream`. The harvester writes each payload to a temp `.wem`/`.isb` and shells out to `vgmstream-cli`. Re-implementing Wwise's custom Vorbis in Rust isn't worth it.

### Licensing note
LEC is GPL-3.0. Reading its source to learn the formats is fine; copying its code into our crate would make that crate GPL-3. The plan is to write our own reader from these notes and cite LEC as documentation. If porting code verbatim ever becomes the faster route, put it in a separate GPL-3 crate.

## Route

**Recommended: native Rust harvester in the workspace** (`crates/harvest`, run as `vi extract`, the harvest stage's command in the pipeline file).
- Fits the Rust preference, runs natively on NixOS, and needs no Windows DLL (oozextract replaces it).
- Only native dependency is `vgmstream-cli`, added to the repo's Nix dev shell.
- Reads the game read-only; never writes to the install.

**Cross-check (optional): Legendary Explorer's Dialogue Dumper under Wine/Proton.** A WPF app on .NET, so it needs a Wine prefix with the .NET desktop runtime; whether it runs on Linux is unconfirmed. If it does, its spreadsheet of every line is a ready-made oracle for our counts and string ids. If it doesn't, we validate against TLK text and listening checks instead. Not on the critical path.

**Rejected as the main route:** calling LEC from a C# console app on Linux. LEC builds only in Windows configurations (`WinRelease` etc.) with `WINDOWS` defined, and its Oodle path loads the Windows DLL, so it would need patching or Wine anyway.

## Build milestones

Each milestone ends with something Brandon can run and look at. Built on his machine in a Remote Control session, because every step needs the real files.

1. **Inventory.** Walk the install, detect LE1/2/3, list packages, TLKs, AFCs with sizes; read each package header and report version and compression. Done when all headers parse.
2. **Package reader.** Decompress (oozextract), read name/import/export tables, print class counts per package. Done when every package in the scan set opens and export counts look sane.
3. **TLK reader.** Both TLK layouts; dump string id to text. Done when a handful of known Tali lines (from the wiki or memory) come back by id, for each game.
4. **Conversation reader.** Hardcoded schemas for `BioConversation` and its node structs; emit one graph per conversation (nodes, speakers, edges, string ids, stage directions in LE3). Done when Tali's known conversations (Normandy talks, her loyalty missions) show the right lines in order.
5. **LE2/LE3 audio.** Match `WwiseStream` by string id and gender, slice the AFC, decode with vgmstream. Done when 20 random Tali clips play and match their subtitles by ear.
6. **LE1 audio.** The FaceFX path above. Same 20-clip check. If this stalls, ship LE2/LE3 first; LE1 is roughly the smallest share of Tali's lines.
7. **Tali filter, context, tags, report.** Select her lines, attach context and scene tags, write the output, and print a coverage report: lines with text but no audio, audio with no text, duplicates merged.

Then the 20-clip listening check per game is the acceptance test. Whisper verification of every clip happens in the clean stage, not here.

## Selecting Tali

Driven by a per-character source file in the pack, so other characters reuse the harvester: e.g. `characters/tali/sources.toml` listing, per game, the speaker tags and stream-name tags that mean "this character". Expected values like `hench_tali` for LE2/LE3; LE1's tag is unknown. Milestone 4 prints every speaker tag with line counts and sample lines, and we pick Tali's from that list rather than guess.

Out of scope for the first pass: combat barks and squad ambient lines that live in Wwise banks without a conversation node. They usually have no TLK subtitle. They can be a later source flagged `bark` (the corpus format already has that flag), transcribed by Whisper in the clean stage; the dataset plan already keeps barks out of most contrast sets.

## Context and scene tags

- **Context:** a conversation is a graph, so "the previous lines" depends on the path. The harvester records each line's immediate predecessors and successors (ids), plus one canonical preceding path: the shortest path from a starting node, up to 4 lines, with speaker and text. The full graph per conversation is also written, so later stages can choose differently without re-harvesting.
- **Scene tags:** derived from the level code (`scene.prefix`), the DLC folder (`scene.dlc`) and the conversation name, by the reviewed table in `crates/harvest/src/scenes.toml`. Its sources are cited at the top of the file: the Mass Effect Wiki's level lists for places, the DLC folders' contents for DLC, and Tali's lines for conversation keywords. Each line gets:
  - a place (`normandy`, `migrant_fleet`, `rannoch`, `despoina`, ...) and the kind of mission (`hub`, `main_mission`, `side_mission`, `recruitment`, `loyalty`, `n7_assignment`, `endgame`, a DLC like `citadel_dlc`);
  - situation tags from the conversation name (`squad_banter`, `ambient`, `romance`, `mission_reaction`, `debrief`, `citadel_party`, `combat`, `squad_comment`, `radio`, `dream`, `shuttle`);
  - the character's own tags from `sources.toml` `[<game>.scene_tags]`; for Tali, `personal_arc` (her storyline) and `personal_talk` (one-on-one time with Shepard).
  Barks that play anywhere (`ss_*`, LE1's `combat_banter_*`) get only their situation tags, because the package storing them isn't where they're heard. Level codes the table doesn't know are listed in `harvest.json` (`unknown_levels`). Plot conditionals are recorded raw (ids only); interpreting them is out of scope.

## Output (per corpus-format.md v1)

The record format is defined in [corpus-format.md](corpus-format.md), section "harvest"; the harvester writes exactly that. In short:
- Runs as the `harvest` stage: `vi extract --character tali <LE root>`, writing into `$VI_OUT`. The runner hashes the game folder (listed in `files`) and keys the run, so the harvester doesn't compute or report input hashes itself. Bump the stage `version` whenever extractor behavior changes.
- `lines.jsonl`: one record per distinct line. Id is `<game>.s<strref>_<gender>` (`le2.s348112_f`, or `le3.s612004` with no variant); the same string id and gender with genuinely different audio elsewhere gets `.2`, `.3`. Duplicates across packages merge into one record with every place in `source.occurrences`.
- `audio/<id>.wav`: 24 kHz mono 16-bit PCM, plus `audio.source` (AFC or package, offset, size, export, codec, original rate, raw-bytes hash).
- `context` (canonical preceding path, up to 4 lines), `prev`/`next` ids, `scene_tags` (lowercase with underscores, like `citadel_party`), `scene`, `stage_direction` (LE3), `flags` (`no_subtitle`, `nonverbal`, `bark`).
- `conversations/<conversation>.json`: the full graph of each conversation she appears in.
- `harvest.json`: harvester version, game build, per-game counts, and the coverage report from milestone 7.

Before declaring a run done, `vi corpus check` must pass on the output. `vi corpus synth` gives a valid example to compare against while building.

## Caching

Handled by the stage runner (corpus-format.md, "How caching works"). Within one run, the harvester may keep a private decode cache keyed by raw-source-bytes hash outside `$VI_OUT` so re-harvesting after a pack change skips vgmstream; it must not affect output bytes.

## Packaging on NixOS

- Add `vgmstream` and the Rust toolchain to the repo's Nix dev shell (flake `devShells.default`). Nothing else native is required.
- Optional cross-check only: `wine` plus a prefix with the .NET desktop runtime for Legendary Explorer.
- No Oodle DLL needed; oozextract is pure Rust.

## Data handling

Everything stays local and gitignored (`extracted/`, `corpus/`, `characters/*/voice/`). The harvester never uploads anything; clips reach the GPU worker later only by rsync, per the project decision. Test fixtures in the repo must be synthetic, never game bytes.

## Open questions to settle on real files

1. ~~WwiseStream naming and speaker tags for Tali in LE2/LE3~~: settled, see the status section.
2. ~~The LE1 FaceFX to SoundCue to ISACT sample path, and Tali's LE1 speaker tag~~: settled, see the status section (no FaceFX needed).
3. ~~What LE3 stage directions actually say~~: blocking and delivery notes per line, often with emotion words; a useful label prior.
4. ~~Source sample rate of dialogue~~: 44.1 kHz on the clips checked; the harvester resamples to 24 kHz.
5. Whether Legendary Explorer runs under Wine on NixOS (only matters for the cross-check).
6. How many Tali lines exist per game (the dataset plan guesses a few thousand total).

## Correction for the repo doc

Fixed: `docs/extracting-game-audio.md` now says Leviathan, and that LE1 audio is ISACT, not Wwise.
