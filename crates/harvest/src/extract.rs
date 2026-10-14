//! `vi extract`: one character's LE2/LE3 dialogue as a harvest corpus
//! (`docs/design/corpus-format.md`, section "harvest").
//!
//! Lines are found by their `WwiseStream` exports (see [`crate::wwise`]):
//! the stream's name gives the speaker tag, conversation, string id and
//! gender, the TLK gives the subtitle, and the `.afc` gives the audio,
//! decoded with `vgmstream-cli` and resampled to 24 kHz mono 16-bit.
//!
//! The line's `BioConversation` (see [`crate::conversation`]) gives its
//! node, listener, `prev`/`next`, the canonical preceding `context`, and in
//! LE3 the stage direction; each conversation's full graph goes to
//! `conversations/<name>.json`.
//!
//! LE1 lines come from conversation nodes and each conversation's ISACT
//! bank (see `find_le1`). Scene tags come from [`crate::scenes`]: the
//! level code, DLC folder and conversation name, plus the character's own
//! rules from `sources.toml`.
//!
//! Reads the install only; everything is written under `out`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use serde_json::{json, Map, Value};

use crate::conversation::{self, Conversation, Kind, Node, OWNER, SHEPARD};
use crate::install::GameDir;
use crate::scenes;
use crate::tlk::{Gender, TalkFile};
use crate::{wwise, Game, Package};

/// Bump when output changes for the same input.
pub const VERSION: u32 = 1;
pub const SAMPLE_RATE: u32 = 24_000;
/// Clips shorter than this are placeholders (LE3 has 306-byte, 13 ms
/// streams for lines whose audio is in a neighbouring clip) and are left
/// out.
pub const MIN_SECONDS: f64 = 0.05;

/// Who to extract.
#[derive(Debug, Clone)]
pub struct Character {
    /// Pack id, used as `speaker` in records (`tali`).
    pub id: String,
    /// Speaker tags per game, as they appear in stream names (LE2/LE3)
    /// and conversation speaker lists.
    pub tags: BTreeMap<Game, Vec<String>>,
    /// Conversations the character owns, whose owner lines (speaker index
    /// -1) are theirs. LE1 needs these: its audio doesn't name speakers.
    pub owns: BTreeMap<Game, Vec<String>>,
    /// The character's own scene tags (`[<game>.scene_tags]`).
    pub scene_rules: BTreeMap<Game, Vec<scenes::Rule>>,
    /// LE1 squad sound-set banks (`bark_banks`): ISACT banks of combat barks
    /// that belong to no conversation, like `tali` (`tali.isb`).
    pub bark_banks: BTreeMap<Game, Vec<String>>,
    /// Hand-checked flags on single lines (`[<game>.line_flags]`), for facts
    /// the game data doesn't record, like a sung line: flag -> line ids.
    pub line_flags: BTreeMap<String, Vec<String>>,
}

impl Character {
    /// Read `sources.toml` from a character pack:
    ///
    /// ```toml
    /// [le1]
    /// speaker_tags = ["hench_quarian"]
    /// owner_of = ["cha00_quarian_tali"]
    /// ```
    pub fn from_sources(id: &str, text: &str) -> Result<Character, String> {
        let table: toml::Table = text.parse().map_err(|e| format!("sources.toml: {e}"))?;
        let mut tags = BTreeMap::new();
        let mut owns = BTreeMap::new();
        let mut scene_rules = BTreeMap::new();
        let mut bark_banks = BTreeMap::new();
        let mut line_flags: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for game in [Game::Le1, Game::Le2, Game::Le3] {
            let Some(section) = table.get(game.id()) else {
                continue;
            };
            let list = section
                .get("speaker_tags")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("sources.toml: [{}] needs speaker_tags", game.id()))?;
            let list: Option<Vec<String>> = list
                .iter()
                .map(|v| v.as_str().map(str::to_ascii_lowercase))
                .collect();
            tags.insert(
                game,
                list.ok_or_else(|| {
                    format!("sources.toml: [{}] speaker_tags must be strings", game.id())
                })?,
            );
            if let Some(list) = section.get("owner_of") {
                let list: Option<Vec<String>> = list
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|v| v.as_str().map(str::to_ascii_lowercase))
                    .collect();
                owns.insert(
                    game,
                    list.ok_or_else(|| {
                        format!("sources.toml: [{}] owner_of must be strings", game.id())
                    })?,
                );
            }
            if let Some(rules) = section.get("scene_tags") {
                let what = format!("sources.toml: [{}.scene_tags]", game.id());
                let rules = rules
                    .as_table()
                    .ok_or_else(|| format!("{what} must be a table"))?;
                scene_rules.insert(game, scenes::parse_rules(rules, &what)?);
            }
            if let Some(list) = section.get("bark_banks") {
                let list: Option<Vec<String>> = list
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|v| v.as_str().map(str::to_ascii_lowercase))
                    .collect();
                bark_banks.insert(
                    game,
                    list.ok_or_else(|| {
                        format!("sources.toml: [{}] bark_banks must be strings", game.id())
                    })?,
                );
            }
            if let Some(flags) = section.get("line_flags") {
                let what = format!("sources.toml: [{}.line_flags]", game.id());
                let flags = flags
                    .as_table()
                    .ok_or_else(|| format!("{what} must be a table"))?;
                for (flag, ids) in flags {
                    if !scenes::valid_tag(flag) {
                        return Err(format!(
                            "{what}: flag {flag:?} must be lowercase a-z, 0-9 and _"
                        ));
                    }
                    let ids: Option<Vec<String>> = ids
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|v| v.as_str().map(str::to_string))
                        .collect();
                    let ids =
                        ids.ok_or_else(|| format!("{what}.{flag} must be a list of line ids"))?;
                    if let Some(bad) = ids
                        .iter()
                        .find(|i| !i.starts_with(&format!("{}.", game.id())))
                    {
                        return Err(format!(
                            "{what}.{flag}: {bad} isn't a {} line id",
                            game.id()
                        ));
                    }
                    line_flags.entry(flag.clone()).or_default().extend(ids);
                }
            }
        }
        Ok(Character {
            id: id.to_string(),
            tags,
            owns,
            scene_rules,
            bark_banks,
            line_flags,
        })
    }
}

/// Every TLK string a game's English subtitles define: the base game's
/// tables first, then DLC, first definition wins.
struct Subtitles {
    male: BTreeMap<i32, (String, String)>,
    female: BTreeMap<i32, (String, String)>,
}

impl Subtitles {
    fn load(game: &GameDir) -> Result<Subtitles, String> {
        let mut files: Vec<&PathBuf> = game
            .files
            .tlks
            .iter()
            .map(|f| &f.path)
            .filter(|p| {
                let name = file_name(p).to_ascii_lowercase();
                name.ends_with("_int.tlk") && !name.contains("_test_")
            })
            .collect();
        // Base game before DLC.
        files.sort_by_key(|p| (is_dlc(p), (*p).clone()));
        let mut subs = Subtitles {
            male: BTreeMap::new(),
            female: BTreeMap::new(),
        };
        for path in files {
            let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let tlk = TalkFile::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            let name = file_name(path).to_string();
            for (id, text) in tlk.male {
                subs.male.entry(id).or_insert_with(|| (text, name.clone()));
            }
            for (id, text) in tlk.female {
                subs.female
                    .entry(id)
                    .or_insert_with(|| (text, name.clone()));
            }
        }
        Ok(subs)
    }

    /// The text heard by a Shepard of `gender`, and the file it came from.
    fn get(&self, id: i32, gender: Gender) -> Option<&(String, String)> {
        if gender == Gender::Female {
            if let Some(t) = self.female.get(&id) {
                return Some(t);
            }
        }
        self.male.get(&id)
    }
}

fn file_name(p: &Path) -> &str {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("")
}

fn is_dlc(p: &Path) -> bool {
    p.components()
        .any(|c| c.as_os_str().eq_ignore_ascii_case("DLC"))
}

/// The DLC folder a file sits in, if any.
fn dlc_name(p: &Path) -> Option<String> {
    let parts: Vec<_> = p.components().map(|c| c.as_os_str()).collect();
    let i = parts.iter().position(|c| c.eq_ignore_ascii_case("DLC"))?;
    parts
        .get(i + 1)
        .and_then(|s| s.to_str())
        .map(str::to_string)
}

/// One place a line's audio was found.
#[derive(Debug, Clone)]
struct Found {
    package: PathBuf,
    export: String,
    conversation: String,
    speaker_tag: String,
    /// The `.afc` (LE2/LE3) or `.isb` bank (LE1).
    container: PathBuf,
    /// Byte offset in an `.afc`; subsong index in an `.isb`.
    offset: usize,
    /// Bytes in an `.afc`; 0 for an `.isb`.
    size: usize,
}

/// Where a recording's audio comes from.
enum Raw {
    /// A complete `.wem`, sliced from an `.afc`.
    Wem(Vec<u8>),
    /// A subsong of an ISACT bank (1-based, as vgmstream counts).
    Subsong(PathBuf, usize),
}

/// A distinct recording: one string id, gender and set of audio bytes.
struct Recording {
    string_id: i32,
    gender: Gender,
    /// False for recordings with no gender variant (LE1 barks): their id has
    /// no `_m`/`_f` and `source.gender` is `none`.
    variant: bool,
    /// LE1 bark category code from the stream name (`png`, `dth`, ...),
    /// recorded as-is.
    bark_category: Option<String>,
    raw: Raw,
    /// Hash of the source bytes, when we have them (`.wem` only).
    raw_sha: Option<String>,
    found: Vec<Found>,
}

#[derive(Debug, Default)]
pub struct GameReport {
    pub packages: usize,
    pub streams: usize,
    pub lines: usize,
    pub merged_duplicates: usize,
    pub no_subtitle: usize,
    pub missing_afc: usize,
    /// Conversations the character speaks in.
    pub conversations: usize,
    /// Lines whose string id isn't a node of their conversation.
    pub without_node: usize,
    /// Ids of streams skipped as placeholders (see [`MIN_SECONDS`]).
    pub placeholders: Vec<String>,
    /// Level codes `scenes.toml` doesn't know, with how many lines each.
    pub unknown_levels: BTreeMap<String, usize>,
    pub errors: Vec<String>,
}

/// Extract `character` from the install at `root` into `out`. `progress`
/// gets a line now and then.
pub fn run(
    root: &Path,
    character: &Character,
    out: &Path,
    progress: &dyn Fn(&str),
) -> Result<BTreeMap<Game, GameReport>, String> {
    let games = crate::install::scan(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let audio_dir = out.join("audio");
    fs::create_dir_all(&audio_dir).map_err(|e| format!("{}: {e}", audio_dir.display()))?;
    let tmp = out.join(".decode");
    fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;

    let mut lines = Vec::new();
    let mut reports = BTreeMap::new();
    for game in &games {
        let Some(tags) = character.tags.get(&game.game) else {
            continue;
        };
        let mut report = GameReport::default();
        let (recordings, graphs, subs) = if game.game == Game::Le1 {
            progress("le1: scanning packages");
            find_le1(game, character, &mut report, progress)?
        } else {
            let subs = Subtitles::load(game)?;
            progress(&format!(
                "{}: {} subtitles, scanning packages",
                game.game,
                subs.male.len()
            ));
            let (r, g) = find(game, tags, &mut report, progress);
            (r, g, subs)
        };
        progress(&format!(
            "{}: {} streams, {} recordings, decoding",
            game.game,
            report.streams,
            recordings.len()
        ));
        let mut game_lines = decode_all(
            game,
            root,
            character,
            &subs,
            recordings,
            out,
            &tmp,
            &mut report,
        )?;
        report.conversations = graphs.conversations.len();
        report.without_node = attach_context(
            game.game,
            character,
            &subs,
            &graphs,
            &mut game_lines,
            out,
            root,
        )?;
        report.unknown_levels = tag_lines(character, &mut game_lines);
        apply_line_flags(character, &mut game_lines);
        report.lines = game_lines.len();
        report.no_subtitle = game_lines
            .iter()
            .filter(|l| l.flags.iter().any(|f| f == "no_subtitle"))
            .count();
        lines.append(&mut game_lines);
        reports.insert(game.game, report);
    }
    let _ = fs::remove_dir_all(&tmp);

    lines.sort_by(|a, b| a.id.cmp(&b.id));
    dataset::corpus::write_lines(out, &lines).map_err(|e| e.to_string())?;

    let per_game: Map<String, Value> = reports
        .iter()
        .map(|(g, r)| {
            (
                g.id().to_string(),
                json!({
                    "packages_scanned": r.packages,
                    "streams": r.streams,
                    "lines": r.lines,
                    "merged_duplicates": r.merged_duplicates,
                    "no_subtitle": r.no_subtitle,
                    "missing_afc": r.missing_afc,
                    "conversations": r.conversations,
                    "lines_without_node": r.without_node,
                    "placeholders_skipped": r.placeholders,
                    "unknown_levels": r.unknown_levels,
                    "errors": r.errors,
                }),
            )
        })
        .collect();
    let summary = json!({
        "harvester": "vi extract",
        "version": VERSION,
        "character": character.id,
        "games": per_game,
        "not_covered": ["barks without a stream name or conversation node"],
    });
    let path = out.join("harvest.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(reports)
}

/// The conversations the character speaks in, and who speaks each line.
#[derive(Default)]
struct Graphs {
    /// By conversation name as streams spell it (`nortla_debrief_d`, the
    /// export name without `_dlg`), lowercase; the first package wins.
    conversations: BTreeMap<String, (Conversation, PathBuf)>,
    /// Speaker tag of every line that has a stream, by conversation and
    /// string id. This is how owner lines (speaker index -1) get a name.
    tags: BTreeMap<(String, i32), String>,
}

/// Every stream spoken under one of `tags`, grouped into distinct
/// recordings, plus the conversations they're in.
fn find(
    game: &GameDir,
    tags: &[String],
    report: &mut GameReport,
    progress: &dyn Fn(&str),
) -> (Vec<Recording>, Graphs) {
    let mut graphs = Graphs::default();
    // `.afc` files by lowercase name; base game first so DLC can't shadow it.
    let mut afcs: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut sorted: Vec<_> = game.files.afcs.iter().map(|f| f.path.clone()).collect();
    sorted.sort_by_key(|p| (is_dlc(p), p.clone()));
    for p in sorted {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        afcs.entry(stem).or_default().push(p);
    }

    let packages: Vec<_> = game
        .files
        .packages
        .iter()
        .filter(|f| {
            file_name(&f.path)
                .to_ascii_lowercase()
                .ends_with("_loc_int.pcc")
        })
        .collect();
    let mut by_key: BTreeMap<(i32, Gender), Vec<Recording>> = BTreeMap::new();
    for (n, info) in packages.iter().enumerate() {
        if n % 100 == 0 {
            progress(&format!("{}: package {n}/{}", game.game, packages.len()));
        }
        report.packages += 1;
        let pkg = match Package::read(&info.path) {
            Ok(p) => p,
            Err(e) => {
                report.errors.push(e);
                continue;
            }
        };
        let mut here = 0;
        let mut speakers = Vec::new();
        for i in pkg.exports_of_class("WwiseStream").collect::<Vec<_>>() {
            let stream = match wwise::read_stream(&pkg, i) {
                Ok(Some(s)) => s,
                Ok(None) => continue,
                Err(e) => {
                    report.errors.push(format!("{}: {e}", info.path.display()));
                    continue;
                }
            };
            let Some(name) = wwise::parse_name(&stream.name, stream.bank.as_deref()) else {
                continue;
            };
            speakers.push((
                (name.conversation.to_ascii_lowercase(), name.string_id),
                name.speaker_tag.to_ascii_lowercase(),
            ));
            if !tags
                .iter()
                .any(|t| t.eq_ignore_ascii_case(&name.speaker_tag))
            {
                continue;
            }
            report.streams += 1;
            here += 1;
            // Prefer the AFC beside the package.
            let candidates = afcs
                .get(&stream.filename.to_ascii_lowercase())
                .cloned()
                .unwrap_or_default();
            let afc = candidates
                .iter()
                .find(|a| a.parent() == info.path.parent())
                .or(candidates.first());
            let Some(afc) = afc else {
                report.missing_afc += 1;
                continue;
            };
            let raw = match read_slice(afc, stream.offset, stream.size) {
                Ok(raw) if wwise::wem_sample_rate(&raw).is_some() => raw,
                Ok(_) => {
                    report.errors.push(format!(
                        "{} at {}: not a RIFF .wem ({})",
                        afc.display(),
                        stream.offset,
                        stream.name
                    ));
                    continue;
                }
                Err(e) => {
                    report.errors.push(e);
                    continue;
                }
            };
            let found = Found {
                package: info.path.clone(),
                export: pkg.full_path(i as i32 + 1),
                conversation: name.conversation,
                speaker_tag: name.speaker_tag,
                container: afc.clone(),
                offset: stream.offset,
                size: stream.size,
            };
            let raw_sha = dataset::hash::sha256_bytes(&raw);
            let list = by_key.entry((name.string_id, name.gender)).or_default();
            if let Some(r) = list
                .iter_mut()
                .find(|r| r.raw_sha.as_ref() == Some(&raw_sha))
            {
                r.found.push(found);
                report.merged_duplicates += 1;
            } else {
                list.push(Recording {
                    string_id: name.string_id,
                    gender: name.gender,
                    variant: true,
                    bark_category: None,
                    raw: Raw::Wem(raw),
                    raw_sha: Some(raw_sha),
                    found: vec![found],
                });
            }
        }
        if here == 0 {
            continue;
        }
        for (key, tag) in speakers {
            graphs.tags.entry(key).or_insert(tag);
        }
        for (path, conversation) in conversation::all(&pkg) {
            match conversation {
                Ok(c) => {
                    let key = c.name.to_ascii_lowercase();
                    let key = key.strip_suffix("_dlg").unwrap_or(&key).to_string();
                    graphs
                        .conversations
                        .entry(key)
                        .or_insert_with(|| (c, info.path.clone()));
                }
                Err(e) => report
                    .errors
                    .push(format!("{} {path}: {e}", info.path.display())),
            }
        }
    }
    (by_key.into_values().flatten().collect(), graphs)
}

/// LE1: the character's lines from conversation nodes (their speaker
/// tags, or owner lines of the conversations they own), with audio from
/// each conversation's ISACT bank (`Content/Packages/ISACT/<name>.isb`,
/// streams named `EN_<name>_<8-digit string id>_<M|F>.wav`). Subtitles
/// come from the `BioTlkFile` exports: `Startup_INT.pcc` and each
/// `_LOC_INT` package, English tables only (`.tlk` female, `.tlk_M` male).
fn find_le1(
    game: &GameDir,
    character: &Character,
    report: &mut GameReport,
    progress: &dyn Fn(&str),
) -> Result<(Vec<Recording>, Graphs, Subtitles), String> {
    let tags = character.tags.get(&game.game).cloned().unwrap_or_default();
    let owns = character.owns.get(&game.game).cloned().unwrap_or_default();
    let mut subs = Subtitles {
        male: BTreeMap::new(),
        female: BTreeMap::new(),
    };
    let mut add_tables = |pkg: &Package, file: &str| {
        let Ok(tables) = crate::tlk::le1_tables(pkg) else {
            return;
        };
        for (name, strings) in tables {
            let map = if name.ends_with("tlk_M") {
                &mut subs.male
            } else if name.ends_with("tlk") {
                &mut subs.female
            } else {
                continue; // another language
            };
            for (id, text) in strings {
                map.entry(id).or_insert_with(|| (text, file.to_string()));
            }
        }
    };
    let mut packages: Vec<_> = game
        .files
        .packages
        .iter()
        .filter(|f| {
            let n = file_name(&f.path).to_ascii_lowercase();
            n.ends_with("_loc_int.pcc") || n == "startup_int.pcc"
        })
        .collect();
    // Startup (the global tables) first, then base game before DLC.
    packages.sort_by_key(|f| {
        (
            !file_name(&f.path).eq_ignore_ascii_case("startup_int.pcc"),
            is_dlc(&f.path),
            f.path.clone(),
        )
    });

    let mut graphs = Graphs::default();
    // Conversation -> (string id -> speaker tag) for the character's lines.
    let mut wanted: BTreeMap<String, BTreeMap<i32, String>> = BTreeMap::new();
    for (n, info) in packages.iter().enumerate() {
        if n % 100 == 0 {
            progress(&format!("{}: package {n}/{}", game.game, packages.len()));
        }
        report.packages += 1;
        let pkg = match Package::read(&info.path) {
            Ok(p) => p,
            Err(e) => {
                report.errors.push(e);
                continue;
            }
        };
        add_tables(&pkg, file_name(&info.path));
        for (path, conversation) in conversation::all(&pkg) {
            let c = match conversation {
                Ok(c) => c,
                Err(e) => {
                    report
                        .errors
                        .push(format!("{} {path}: {e}", info.path.display()));
                    continue;
                }
            };
            let key = c.name.to_ascii_lowercase();
            let key = key.strip_suffix("_dlg").unwrap_or(&key).to_string();
            if graphs.conversations.contains_key(&key) {
                continue;
            }
            let owned = owns.contains(&key);
            let mut lines = BTreeMap::new();
            for e in &c.entries {
                if e.string_id < 0 {
                    continue;
                }
                let tag = match e.speaker_index {
                    OWNER if owned => tags.first().cloned(),
                    i => c
                        .speaker_tag(i)
                        .map(str::to_ascii_lowercase)
                        .filter(|t| tags.contains(t)),
                };
                if let Some(tag) = tag {
                    if e.speaker_index == OWNER {
                        graphs.tags.insert((key.clone(), e.string_id), tag.clone());
                    }
                    lines.insert(e.string_id, tag);
                }
            }
            if !lines.is_empty() {
                wanted.insert(key.clone(), lines);
            }
            graphs.conversations.insert(key, (c, info.path.clone()));
        }
    }
    // Keep only the conversations the character is in.
    graphs.conversations.retain(|k, _| wanted.contains_key(k));

    let banks: BTreeMap<String, PathBuf> = game
        .files
        .isbs
        .iter()
        .map(|f| {
            let stem = f
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            (stem, f.path.clone())
        })
        .collect();
    let mut recordings = Vec::new();
    progress(&format!(
        "{}: {} conversations, listing banks",
        game.game,
        wanted.len()
    ));
    for (key, lines) in &wanted {
        let Some(bank) = banks.get(key) else {
            report.missing_afc += lines.len();
            continue;
        };
        let streams = match list_subsongs(bank) {
            Ok(s) => s,
            Err(e) => {
                report.errors.push(format!("{}: {e}", bank.display()));
                continue;
            }
        };
        let package = graphs.conversations[key].1.clone();
        for (index, name) in streams {
            let Some((string_id, gender)) = parse_isact_name(&name) else {
                continue;
            };
            let Some(tag) = lines.get(&string_id) else {
                continue;
            };
            report.streams += 1;
            recordings.push(Recording {
                string_id,
                gender,
                variant: true,
                bark_category: None,
                raw: Raw::Subsong(bank.clone(), index),
                raw_sha: None,
                found: vec![Found {
                    package: package.clone(),
                    export: name,
                    conversation: key.clone(),
                    speaker_tag: tag.clone(),
                    container: bank.clone(),
                    offset: index,
                    size: 0,
                }],
            });
        }
    }
    // Squad sound sets: combat barks outside any conversation, one bank per
    // character (`tali.isb`, streams `EN_tali_<string id>_<category><n>.wav`).
    for name in character.bark_banks.get(&game.game).into_iter().flatten() {
        let Some(bank) = banks.get(name) else {
            report.missing_afc += 1;
            report
                .errors
                .push(format!("bark bank {name}.isb not found"));
            continue;
        };
        let streams = match list_subsongs(bank) {
            Ok(s) => s,
            Err(e) => {
                report.errors.push(format!("{}: {e}", bank.display()));
                continue;
            }
        };
        let package = game
            .files
            .packages
            .iter()
            .find(|f| file_name(&f.path).eq_ignore_ascii_case(&format!("{name}_ss.pcc")))
            .map_or_else(|| bank.clone(), |f| f.path.clone());
        for (index, stream) in streams {
            let Some((string_id, category)) = parse_isact_bark(name, &stream) else {
                continue;
            };
            report.streams += 1;
            recordings.push(Recording {
                string_id,
                gender: Gender::Male,
                variant: false,
                bark_category: Some(category),
                raw: Raw::Subsong(bank.clone(), index),
                raw_sha: None,
                found: vec![Found {
                    package: package.clone(),
                    export: stream,
                    conversation: format!("{name}_ss"),
                    speaker_tag: tags.first().cloned().unwrap_or_default(),
                    container: bank.clone(),
                    offset: index,
                    size: 0,
                }],
            });
        }
    }
    Ok((recordings, graphs, subs))
}

/// `EN_tali_00168422_png00.wav` in bank `tali` -> (168422, `png`): an LE1
/// squad bark, whose name has a category code where dialogue has a gender.
fn parse_isact_bark(bank: &str, name: &str) -> Option<(i32, String)> {
    let rest = name.strip_prefix("EN_")?.strip_suffix(".wav")?;
    let rest = rest
        .get(..bank.len())
        .filter(|b| b.eq_ignore_ascii_case(bank))
        .map(|_| &rest[bank.len()..])?
        .strip_prefix('_')?;
    let (id, code) = rest.split_once('_')?;
    if id.len() != 8 || !id.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let category: String = code
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if category.is_empty() || !code[category.len()..].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((id.parse().ok()?, category.to_ascii_lowercase()))
}

/// `EN_cha00_quarian_tali_00127848_M.wav` -> (127848, male). English only.
fn parse_isact_name(name: &str) -> Option<(i32, Gender)> {
    let rest = name.strip_prefix("EN_")?.strip_suffix(".wav")?;
    let (rest, g) = rest.rsplit_once('_')?;
    let gender = match g {
        "M" | "m" => Gender::Male,
        "F" | "f" => Gender::Female,
        _ => return None,
    };
    let (_, id) = rest.rsplit_once('_')?;
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((id.parse().ok()?, gender))
}

/// The subsongs of a bank as `(1-based index, stream name)`, from
/// `vgmstream-cli -I -m -S 0` (one JSON object per line).
fn list_subsongs(bank: &Path) -> Result<Vec<(usize, String)>, String> {
    let output = Command::new("vgmstream-cli")
        .args(["-I", "-m", "-S", "0"])
        .arg(bank)
        .output()
        .map_err(|e| format!("running vgmstream-cli: {e} (is vgmstream installed?)"))?;
    if !output.status.success() {
        return Err(format!(
            "vgmstream-cli failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut out = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let info = &v["streamInfo"];
        if let (Some(i), Some(n)) = (info["index"].as_u64(), info["name"].as_str()) {
            out.push((i as usize, n.to_string()));
        }
    }
    Ok(out)
}

fn read_slice(path: &Path, offset: usize, size: usize) -> Result<Vec<u8>, String> {
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = fs::File::open(path).map_err(err)?;
    f.seek(SeekFrom::Start(offset as u64)).map_err(err)?;
    let mut buf = vec![0; size];
    f.read_exact(&mut buf).map_err(err)?;
    Ok(buf)
}

#[allow(clippy::too_many_arguments)]
fn decode_all(
    game: &GameDir,
    root: &Path,
    character: &Character,
    subs: &Subtitles,
    recordings: Vec<Recording>,
    out: &Path,
    tmp: &Path,
    report: &mut GameReport,
) -> Result<Vec<dataset::corpus::Line>, String> {
    // Ids: `<game>.s<id>_<gender>`, then `.2`, `.3` for other recordings
    // of the same string and gender, in order of first package.
    let mut jobs = Vec::new();
    let mut seen: BTreeMap<(i32, Gender), usize> = BTreeMap::new();
    for mut r in recordings {
        r.found.sort_by(|a, b| a.package.cmp(&b.package));
        let n = seen.entry((r.string_id, r.gender)).or_default();
        *n += 1;
        let mut id = if r.variant {
            format!("{}.s{}_{}", game.game.id(), r.string_id, r.gender.suffix())
        } else {
            format!("{}.s{}", game.game.id(), r.string_id)
        };
        if *n > 1 {
            id.push_str(&format!(".{n}"));
        }
        jobs.push((id, r));
    }

    let rel = |p: &Path| {
        p.strip_prefix(root)
            .unwrap_or(p)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let results = Mutex::new(Vec::new());
    let errors = Mutex::new(Vec::new());
    let placeholders = Mutex::new(Vec::new());
    let next = Mutex::new(jobs.into_iter());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for w in 0..workers {
            let (results, errors, next, placeholders) = (&results, &errors, &next, &placeholders);
            let rel = &rel;
            s.spawn(move || loop {
                let Some((id, r)) = next.lock().unwrap().next() else {
                    break;
                };
                let wem = tmp.join(format!("{w}.wem"));
                let wav = tmp.join(format!("{w}.wav"));
                let decoded = match &r.raw {
                    Raw::Wem(bytes) => fs::write(&wem, bytes)
                        .map_err(|e| e.to_string())
                        .and_then(|_| decode(&wem, None, &wav)),
                    Raw::Subsong(bank, n) => decode(bank, Some(*n), &wav),
                };
                let (rate, samples) = match decoded {
                    Ok(d) => d,
                    Err(e) => {
                        errors.lock().unwrap().push(format!("{id}: {e}"));
                        continue;
                    }
                };
                if (samples.len() as f64) < MIN_SECONDS * rate as f64 {
                    placeholders.lock().unwrap().push(id);
                    continue;
                }
                let pcm = match resample(&samples, rate, SAMPLE_RATE) {
                    Ok(p) => p,
                    Err(e) => {
                        errors.lock().unwrap().push(format!("{id}: {e}"));
                        continue;
                    }
                };
                let path = format!("audio/{id}.wav");
                let file = out.join(&path);
                let ints: Vec<i16> = pcm
                    .iter()
                    .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                    .collect();
                if let Err(e) = dataset::wav::write_mono16(&file, SAMPLE_RATE, &ints) {
                    errors.lock().unwrap().push(format!("{id}: {e}"));
                    continue;
                }
                let sha = match dataset::hash::sha256_file(&file) {
                    Ok(h) => h,
                    Err(e) => {
                        errors.lock().unwrap().push(format!("{id}: {e}"));
                        continue;
                    }
                };
                let first = &r.found[0];
                let looked_up = if r.variant {
                    subs.get(r.string_id, r.gender)
                } else {
                    subs.male.get(&r.string_id)
                };
                let (text, tlk) = match looked_up {
                    Some((t, f)) => (t.clone(), Some(f.clone())),
                    None => (String::new(), None),
                };
                let flags = flags_for(&text, &first.conversation);
                let mut audio_extra = Map::new();
                audio_extra.insert(
                    "source".into(),
                    match &r.raw {
                        Raw::Wem(_) => json!({
                            "container": rel(&first.container),
                            "offset": first.offset,
                            "size": first.size,
                            "export": first.export,
                            "codec": "wwise_vorbis",
                            "sample_rate": rate,
                            "sha256": r.raw_sha,
                        }),
                        Raw::Subsong(_, n) => json!({
                            "container": rel(&first.container),
                            "subsong": n,
                            "stream": first.export,
                            "codec": "isact_vorbis",
                            "sample_rate": rate,
                        }),
                    },
                );
                let mut source = Map::new();
                source.insert("package".into(), json!(rel(&first.package)));
                source.insert("conversation".into(), json!(first.conversation));
                source.insert("string_id".into(), json!(r.string_id));
                source.insert(
                    "gender".into(),
                    json!(if r.variant { r.gender.suffix() } else { "none" }),
                );
                if let Some(category) = &r.bark_category {
                    source.insert("bark_category".into(), json!(category));
                }
                if let Some(tlk) = tlk {
                    source.insert("tlk".into(), json!(tlk));
                }
                source.insert("speaker_tag".into(), json!(first.speaker_tag));
                source.insert(
                    "occurrences".into(),
                    Value::Array(
                        r.found
                            .iter()
                            .map(|f| {
                                json!({
                                    "package": rel(&f.package),
                                    "conversation": f.conversation,
                                    "export": f.export,
                                })
                            })
                            .collect(),
                    ),
                );
                let mut extra = Map::new();
                extra.insert(
                    "scene".into(),
                    json!({
                        "dlc": dlc_name(&first.package),
                        "prefix": package_prefix(&first.package),
                    }),
                );
                results.lock().unwrap().push(dataset::corpus::Line {
                    id,
                    speaker: character.id.clone(),
                    text,
                    flags,
                    audio: dataset::corpus::Audio {
                        path,
                        sha256: sha,
                        sample_rate: SAMPLE_RATE,
                        channels: 1,
                        duration_s: ints.len() as f64 / SAMPLE_RATE as f64,
                        extra: audio_extra,
                    },
                    source: dataset::corpus::Source {
                        game: game.game.id().to_string(),
                        extra: source,
                    },
                    context: Vec::new(),
                    scene_tags: Vec::new(),
                    stage_direction: None,
                    extra,
                });
            });
        }
    });
    report.errors.extend(errors.into_inner().unwrap());
    report.placeholders = placeholders.into_inner().unwrap();
    report.placeholders.sort();
    Ok(results.into_inner().unwrap())
}

/// Most lines a context path keeps.
const CONTEXT: usize = dataset::corpus::MAX_CONTEXT;

/// How a speaker index reads in records: the character's pack id for
/// their own tags, `shepard`, or the speaker tag.
fn speaker_label(
    character: &Character,
    game: Game,
    conversation: &Conversation,
    conversation_key: &str,
    graphs: &Graphs,
    node: &Node,
) -> String {
    let tag = match node.speaker_index {
        SHEPARD => return "shepard".into(),
        OWNER => graphs
            .tags
            .get(&(conversation_key.to_string(), node.string_id))
            .cloned(),
        i => conversation.speaker_tag(i).map(str::to_ascii_lowercase),
    };
    let Some(tag) = tag else {
        return "owner".into();
    };
    let ours = character
        .tags
        .get(&game)
        .is_some_and(|t| t.iter().any(|x| x.eq_ignore_ascii_case(&tag)));
    if ours {
        character.id.clone()
    } else {
        tag
    }
}

/// A listener index as a label (no stream to resolve the owner by).
fn listener_label(conversation: &Conversation, index: i32) -> String {
    match index {
        SHEPARD => "shepard".into(),
        OWNER => "owner".into(),
        i => conversation
            .speaker_tag(i)
            .map_or_else(|| format!("speaker_{i}"), str::to_ascii_lowercase),
    }
}

/// The text of a node, if it has a line.
fn node_text<'a>(subs: &'a Subtitles, node: &Node, gender: Gender) -> Option<&'a str> {
    if node.string_id < 0 {
        return None;
    }
    subs.get(node.string_id, gender)
        .map(|(t, _)| t.as_str())
        .filter(|t| !t.trim().is_empty())
}

/// The nearest nodes with a line in one direction, passing through nodes
/// without one (autocontinues, cut lines).
fn nearest_lines(
    c: &Conversation,
    subs: &Subtitles,
    gender: Gender,
    start: (Kind, usize),
    backwards: bool,
) -> Vec<(Kind, usize)> {
    let step = |k: Kind, i: usize| -> Vec<(Kind, usize)> {
        if backwards {
            c.predecessors(k, i)
        } else {
            c.node(k, i).map(|n| c.successors(n)).unwrap_or_default()
        }
    };
    let mut seen = std::collections::BTreeSet::from([start]);
    let mut queue: std::collections::VecDeque<_> = step(start.0, start.1).into();
    let mut out = Vec::new();
    while let Some(at) = queue.pop_front() {
        if !seen.insert(at) {
            continue;
        }
        let Some(node) = c.node(at.0, at.1) else {
            continue;
        };
        if node_text(subs, node, gender).is_some() {
            out.push(at);
        } else {
            queue.extend(step(at.0, at.1));
        }
    }
    out
}

/// The shortest path from a starting entry to `target`, oldest first,
/// target excluded.
fn path_to(c: &Conversation, target: (Kind, usize)) -> Vec<(Kind, usize)> {
    let mut parent: BTreeMap<(Kind, usize), Option<(Kind, usize)>> = BTreeMap::new();
    let mut queue = std::collections::VecDeque::new();
    for &s in &c.starting {
        if c.node(Kind::Entry, s).is_some() && !parent.contains_key(&(Kind::Entry, s)) {
            parent.insert((Kind::Entry, s), None);
            queue.push_back((Kind::Entry, s));
        }
    }
    while let Some(at) = queue.pop_front() {
        if at == target {
            let mut path = Vec::new();
            let mut cur = parent[&at];
            while let Some(p) = cur {
                path.push(p);
                cur = parent[&p];
            }
            path.reverse();
            return path;
        }
        let Some(node) = c.node(at.0, at.1) else {
            continue;
        };
        for next in c.successors(node) {
            if let std::collections::btree_map::Entry::Vacant(e) = parent.entry(next) {
                e.insert(Some(at));
                queue.push_back(next);
            }
        }
    }
    Vec::new()
}

/// Up to [`CONTEXT`] lines before `start`, oldest first, following the
/// nearest predecessor with a line each step (the first, when there are
/// several) and never visiting a node twice.
fn path_back(
    c: &Conversation,
    subs: &Subtitles,
    gender: Gender,
    start: (Kind, usize),
) -> Vec<(Kind, usize)> {
    let mut seen = std::collections::BTreeSet::from([start]);
    let mut out = Vec::new();
    let mut at = start;
    while out.len() < CONTEXT {
        let Some(&prev) = nearest_lines(c, subs, gender, at, true)
            .iter()
            .find(|p| !seen.contains(*p))
        else {
            break;
        };
        seen.insert(prev);
        out.push(prev);
        at = prev;
    }
    out.reverse();
    out
}

/// What kind of conversation this is, from its graph alone:
/// - `one_liner`: a single node with a line;
/// - `dialogue_choices`: Shepard picks a reply somewhere (an entry leads to
///   two or more replies that aren't autocontinues);
/// - `exchange`: several lines, no choice for the player.
fn conversation_kind(c: &Conversation, subs: &Subtitles, gender: Gender) -> &'static str {
    let lines = c
        .entries
        .iter()
        .chain(&c.replies)
        .filter(|n| node_text(subs, n, gender).is_some())
        .count();
    if lines <= 1 {
        return "one_liner";
    }
    let choice = c.entries.iter().any(|e| {
        e.links
            .iter()
            .filter_map(|&i| c.replies.get(i))
            .filter(|r| r.reply_type.as_deref() != Some("REPLY_AUTOCONTINUE"))
            .count()
            >= 2
    });
    if choice {
        "dialogue_choices"
    } else {
        "exchange"
    }
}

fn line_id(game: Game, string_id: i32, gender: Gender) -> String {
    format!("{}.s{}_{}", game.id(), string_id, gender.suffix())
}

/// Fill in each line's node, listener, `prev`/`next`, `context` and stage
/// direction, and write `conversations/<name>.json`. Returns how many
/// lines had no node.
fn attach_context(
    game: Game,
    character: &Character,
    subs: &Subtitles,
    graphs: &Graphs,
    lines: &mut [dataset::corpus::Line],
    out: &Path,
    root: &Path,
) -> Result<usize, String> {
    let mut missing = 0;
    let mut used = std::collections::BTreeSet::new();
    for line in lines.iter_mut() {
        let key = line
            .source
            .extra
            .get("conversation")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let string_id = line
            .source
            .extra
            .get("string_id")
            .and_then(Value::as_i64)
            .unwrap_or(-1) as i32;
        let gender = match line.source.extra.get("gender").and_then(Value::as_str) {
            Some("f") => Gender::Female,
            _ => Gender::Male,
        };
        let Some((c, _)) = graphs.conversations.get(&key) else {
            missing += 1;
            continue;
        };
        // Prefer an entry (NPC line) over a reply with the same string.
        let node = c
            .entries
            .iter()
            .chain(&c.replies)
            .find(|n| n.string_id == string_id);
        let Some(node) = node else {
            missing += 1;
            continue;
        };
        used.insert(key.clone());
        let at = (node.kind, node.index);
        let s = &mut line.source.extra;
        s.insert(
            "node".into(),
            json!({"kind": node.kind.as_str(), "index": node.index}),
        );
        s.insert("speaker_index".into(), json!(node.speaker_index));
        s.insert(
            "listener".into(),
            json!(listener_label(c, node.listener_index)),
        );
        s.insert(
            "conversation_kind".into(),
            json!(conversation_kind(c, subs, gender)),
        );
        let ids = |nodes: Vec<(Kind, usize)>| -> Value {
            Value::Array(
                nodes
                    .into_iter()
                    .filter_map(|(k, i)| c.node(k, i))
                    .map(|n| json!(line_id(game, n.string_id, gender)))
                    .collect(),
            )
        };
        let prev = nearest_lines(c, subs, gender, at, true);
        let next = nearest_lines(c, subs, gender, at, false);
        // The line's own kind: alone (nothing before or after it), or part
        // of its conversation's exchange.
        let kind = if prev.is_empty() && next.is_empty() {
            "one_liner"
        } else {
            conversation_kind(c, subs, gender)
        };
        line.source.extra.insert("line_kind".into(), json!(kind));
        line.extra.insert("prev".into(), ids(prev));
        line.extra.insert("next".into(), ids(next.clone()));
        // Every line directly after hers, all branches, one step: how the
        // scene answers her.
        line.extra.insert(
            "following".into(),
            Value::Array(
                next.into_iter()
                    .filter_map(|(k, i)| c.node(k, i))
                    .map(|n| {
                        json!({
                            "id": line_id(game, n.string_id, gender),
                            "speaker": speaker_label(character, game, c, &key, graphs, n),
                            "text": node_text(subs, n, gender).unwrap_or(""),
                            "string_id": n.string_id,
                        })
                    })
                    .collect(),
            ),
        );
        let mut path: Vec<&Node> = path_to(c, at)
            .into_iter()
            .filter_map(|(k, i)| c.node(k, i))
            .filter(|n| node_text(subs, n, gender).is_some())
            .collect();
        if path.is_empty() {
            // Not reachable from a starting node (scripted conversations
            // that begin partway in): follow the line's own predecessors.
            path = path_back(c, subs, gender, at)
                .into_iter()
                .filter_map(|(k, i)| c.node(k, i))
                .collect();
        }
        line.context = path[path.len().saturating_sub(CONTEXT)..]
            .iter()
            .map(|n| dataset::corpus::ContextLine {
                id: Some(line_id(game, n.string_id, gender)),
                speaker: speaker_label(character, game, c, &key, graphs, n),
                text: node_text(subs, n, gender).unwrap_or("").to_string(),
                string_id: u64::try_from(n.string_id).ok(),
                extra: Map::new(),
            })
            .collect();
        line.stage_direction = c.stage_direction(string_id).map(str::to_string);
    }

    let dir = out.join("conversations");
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for key in used {
        let (c, package) = &graphs.conversations[&key];
        let node = |n: &Node| {
            let text = node_text(subs, n, Gender::Male);
            json!({
                "kind": n.kind.as_str(),
                "index": n.index,
                "string_id": n.string_id,
                "speaker": speaker_label(character, game, c, &key, graphs, n),
                "speaker_index": n.speaker_index,
                "listener": listener_label(c, n.listener_index),
                "text": text,
                "links": n.links,
                "paraphrases": n.paraphrases,
                "reply_type": n.reply_type,
                "non_text_line": n.non_text_line,
                "conditional": [n.conditional_func, n.conditional_param],
                "transition": [n.state_transition, n.state_transition_param],
                "stage_direction": c.stage_direction(n.string_id),
            })
        };
        let graph = json!({
            "game": game.id(),
            "conversation": key,
            "export": c.name,
            "package": package.strip_prefix(root).unwrap_or(package).to_string_lossy(),
            "speakers": c.speakers,
            "starting": c.starting,
            "entries": c.entries.iter().map(node).collect::<Vec<_>>(),
            "replies": c.replies.iter().map(node).collect::<Vec<_>>(),
        });
        let path = dir.join(format!("{key}.json"));
        fs::write(&path, serde_json::to_string_pretty(&graph).unwrap() + "\n")
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(missing)
}

/// Set every line's `scene_tags` from its game, `scene.prefix`,
/// `scene.dlc` and conversation. Returns the level codes the table
/// doesn't know, with their line counts.
pub fn tag_lines(
    character: &Character,
    lines: &mut [dataset::corpus::Line],
) -> BTreeMap<String, usize> {
    let table = scenes::Table::builtin();
    let mut unknown = BTreeMap::new();
    for line in lines.iter_mut() {
        let Some(game) = Game::from_id(&line.source.game) else {
            continue;
        };
        let scene = line.extra.get("scene");
        let prefix = scene
            .and_then(|s| s.get("prefix"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let dlc = scene.and_then(|s| s.get("dlc")).and_then(Value::as_str);
        let conversation = line
            .source
            .extra
            .get("conversation")
            .and_then(Value::as_str)
            .unwrap_or("");
        let rules = character
            .scene_rules
            .get(&game)
            .map_or(&[][..], Vec::as_slice);
        let (tags, known) = table.tags(
            scenes::Scene {
                game,
                prefix,
                dlc,
                conversation,
            },
            rules,
        );
        if !known {
            *unknown
                .entry(format!("{}:{prefix}", game.id()))
                .or_default() += 1;
        }
        let mut tags = tags;
        tags.extend(structure_tags(&line.source.extra));
        tags.sort();
        tags.dedup();
        line.scene_tags = tags;
    }
    unknown
}

/// Add the hand-checked `line_flags` from sources.toml to their lines.
pub fn apply_line_flags(character: &Character, lines: &mut [dataset::corpus::Line]) {
    for line in lines.iter_mut() {
        for (flag, ids) in &character.line_flags {
            if ids.contains(&line.id) && !line.flags.contains(flag) {
                line.flags.push(flag.clone());
            }
        }
    }
}

/// Tags from the conversation graph's facts on a record: the line's kind
/// (`source.line_kind`: `one_liner` if nothing comes before or after it,
/// else its conversation's `exchange` or `dialogue_choices`) and who she's
/// speaking to
/// (`to_shepard`; `to_squad` for squadmates, whose tags start `hench_`;
/// `to_npc` for anyone else named). An unnamed listener (the conversation's
/// owner) gets no listener tag.
fn structure_tags(source: &Map<String, Value>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(kind) = source.get("line_kind").and_then(Value::as_str) {
        out.push(kind.to_string());
    }
    match source.get("listener").and_then(Value::as_str) {
        Some("shepard") => out.push("to_shepard".into()),
        Some("owner") | None => {}
        Some(l) if l.starts_with("speaker_") => {}
        Some(l) if l.starts_with("hench_") => out.push("to_squad".into()),
        Some(_) => out.push("to_npc".into()),
    }
    out
}

/// Re-tag an existing harvest folder in place (`vi corpus tag`): only
/// `scene_tags` in `lines.jsonl` change; audio and everything else stay.
/// Returns how many lines changed and the unknown level codes.
pub fn retag(
    dir: &Path,
    character: &Character,
) -> Result<(usize, BTreeMap<String, usize>), String> {
    let mut lines = dataset::corpus::read_lines(dir).map_err(|e| e.to_string())?;
    let before: Vec<Vec<String>> = lines.iter().map(|l| l.scene_tags.clone()).collect();
    let unknown = tag_lines(character, &mut lines);
    apply_line_flags(character, &mut lines);
    let changed = lines
        .iter()
        .zip(&before)
        .filter(|(l, b)| &l.scene_tags != *b)
        .count();
    dataset::corpus::write_lines(dir, &lines).map_err(|e| e.to_string())?;
    // Keep harvest.json's notes in step with what the lines now carry.
    let summary = dir.join("harvest.json");
    if let Ok(text) = fs::read_to_string(&summary) {
        if let Ok(mut v) = serde_json::from_str::<Value>(&text) {
            if let Some(list) = v.get_mut("not_covered").and_then(Value::as_array_mut) {
                list.retain(|x| x.as_str() != Some("scene tags"));
            }
            v["unknown_levels"] = json!(unknown);
            fs::write(&summary, serde_json::to_string_pretty(&v).unwrap() + "\n")
                .map_err(|e| format!("{}: {e}", summary.display()))?;
        }
    }
    Ok((changed, unknown))
}

/// Corpus flags for a line: `no_subtitle` for empty text, `nonverbal` for
/// subtitles that only describe a sound (`{Sighs}`), `bark` for the squad
/// sound sets (`ss_global_hench_tali`, `ss_tali_minimal`, LE1's `tali_ss`),
/// which have no conversation node and no subtitles.
fn flags_for(text: &str, conversation: &str) -> Vec<String> {
    let text = text.trim();
    let mut flags = Vec::new();
    if text.is_empty() {
        flags.push("no_subtitle".to_string());
    } else if text.starts_with('{') && text.ends_with('}') {
        flags.push("nonverbal".to_string());
    }
    let conversation = conversation.to_ascii_lowercase();
    if conversation.starts_with("ss_") || conversation.ends_with("_ss") {
        flags.push("bark".to_string());
    }
    flags
}

/// `BioD_Nor_111DebriefTali_LOC_INT.pcc` -> `BioD_Nor`.
fn package_prefix(p: &Path) -> String {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    stem.splitn(3, '_').take(2).collect::<Vec<_>>().join("_")
}

/// Decode a `.wem`, or one subsong of a bank, with `vgmstream-cli`; mono
/// samples in [-1, 1] and the rate.
fn decode(input: &Path, subsong: Option<usize>, wav: &Path) -> Result<(u32, Vec<f32>), String> {
    let mut cmd = Command::new("vgmstream-cli");
    if let Some(n) = subsong {
        cmd.arg("-s").arg(n.to_string());
    }
    let output = cmd
        .arg("-o")
        .arg(wav)
        .arg(input)
        .output()
        .map_err(|e| format!("running vgmstream-cli: {e} (is vgmstream installed?)"))?;
    if !output.status.success() {
        return Err(format!(
            "vgmstream-cli failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let bytes = fs::read(wav).map_err(|e| e.to_string())?;
    read_pcm(&bytes)
}

/// Read a PCM WAV (16-bit integer or 32-bit float), mixing to mono.
pub fn read_pcm(bytes: &[u8]) -> Result<(u32, Vec<f32>), String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("decoder output isn't a WAV file".into());
    }
    let mut pos = 12;
    let mut fmt = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                let format = u16::from_le_bytes([body[0], body[1]]);
                let channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes([body[14], body[15]]);
                fmt = Some((format, channels, rate, bits));
            }
            b"data" => {
                let (format, channels, rate, bits) = fmt.ok_or("data before fmt")?;
                if channels == 0 {
                    return Err("zero channels".into());
                }
                let samples: Vec<f32> = match (format, bits) {
                    (1, 16) => body
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                        .collect(),
                    (3, 32) => body
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                    _ => return Err(format!("unsupported WAV format {format}/{bits} bits")),
                };
                let mono = samples
                    .chunks_exact(channels)
                    .map(|f| f.iter().sum::<f32>() / channels as f32)
                    .collect();
                return Ok((rate, mono));
            }
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    Err("no data chunk".into())
}

/// Resample mono audio (rubato's FFT resampler, which suits a fixed
/// ratio like 44.1 kHz to 24 kHz).
pub fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>, String> {
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};
    if from == to || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let mut r = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Input)
        .map_err(|e| format!("resampler: {e}"))?;
    let input =
        InterleavedSlice::new(samples, 1, samples.len()).map_err(|e| format!("resampler: {e}"))?;
    let out = r
        .process_all(&input, samples.len(), None)
        .map_err(|e| format!("resampler: {e}"))?;
    Ok(out.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_sources() {
        let c = Character::from_sources(
            "tali",
            "[le2]\nspeaker_tags = [\"hench_tali\"]\n[le3]\nspeaker_tags = [\"Hench_Tali\", \"x\"]\n",
        )
        .unwrap();
        assert_eq!(c.tags[&Game::Le2], ["hench_tali"]);
        assert_eq!(c.tags[&Game::Le3], ["hench_tali", "x"]);
        assert!(!c.tags.contains_key(&Game::Le1));
        let c = Character::from_sources(
            "tali",
            "[le1]\nspeaker_tags = [\"hench_quarian\"]\nowner_of = [\"Cha00_Quarian_Tali\"]\n",
        )
        .unwrap();
        assert_eq!(c.owns[&Game::Le1], ["cha00_quarian_tali"]);
        assert!(Character::from_sources("t", "[le2]\n").is_err());
    }

    fn node(kind: Kind, index: usize, string_id: i32, speaker: i32, links: &[usize]) -> Node {
        Node {
            kind,
            index,
            string_id,
            speaker_index: speaker,
            listener_index: -1,
            links: links.to_vec(),
            paraphrases: vec![],
            reply_type: None,
            non_text_line: false,
            conditional_func: -1,
            conditional_param: -1,
            state_transition: -1,
            state_transition_param: -1,
            export_id: -1,
        }
    }

    #[test]
    fn context_skips_lines_without_text() {
        // entry 0 (owner, 10) -> reply 0 (no text) -> entry 1 (tali, 11)
        //   -> reply 1 (shepard, 12) -> entry 2 (tali, 13)
        let c = Conversation {
            name: "t_d_dlg".into(),
            speakers: vec!["hench_tali".into()],
            starting: vec![0],
            entries: vec![
                node(Kind::Entry, 0, 10, OWNER, &[0]),
                node(Kind::Entry, 1, 11, 0, &[1]),
                node(Kind::Entry, 2, 13, 0, &[]),
            ],
            replies: vec![
                node(Kind::Reply, 0, -1, SHEPARD, &[1]),
                node(Kind::Reply, 1, 12, SHEPARD, &[2]),
            ],
            stage_directions: vec![],
        };
        let mut subs = Subtitles {
            male: BTreeMap::new(),
            female: BTreeMap::new(),
        };
        for (id, t) in [
            (10, "Owner."),
            (11, "Tali."),
            (12, "Shepard."),
            (13, "Tali again."),
        ] {
            subs.male.insert(id, (t.to_string(), "x.tlk".into()));
        }
        let g = Gender::Male;
        assert_eq!(
            nearest_lines(&c, &subs, g, (Kind::Entry, 1), true),
            [(Kind::Entry, 0)]
        );
        assert_eq!(
            nearest_lines(&c, &subs, g, (Kind::Entry, 1), false),
            [(Kind::Reply, 1)]
        );
        assert_eq!(
            path_to(&c, (Kind::Entry, 2)),
            [
                (Kind::Entry, 0),
                (Kind::Reply, 0),
                (Kind::Entry, 1),
                (Kind::Reply, 1)
            ]
        );
        // Not reachable from a start: context follows predecessors.
        let mut cut = c.clone();
        cut.starting = vec![];
        assert!(path_to(&cut, (Kind::Entry, 2)).is_empty());
        assert_eq!(
            path_back(&cut, &subs, g, (Kind::Entry, 2)),
            [(Kind::Entry, 0), (Kind::Entry, 1), (Kind::Reply, 1)]
        );
        assert_eq!(conversation_kind(&c, &subs, g), "exchange");
        let mut choices = c.clone();
        choices.entries[1].links = vec![0, 1];
        assert_eq!(conversation_kind(&choices, &subs, g), "dialogue_choices");
        let mut auto = choices.clone();
        auto.replies[1].reply_type = Some("REPLY_AUTOCONTINUE".into());
        assert_eq!(conversation_kind(&auto, &subs, g), "exchange");
        let one = Conversation {
            entries: vec![node(Kind::Entry, 0, 11, 0, &[])],
            replies: vec![],
            ..c.clone()
        };
        assert_eq!(conversation_kind(&one, &subs, g), "one_liner");
        let mut graphs = Graphs::default();
        graphs
            .tags
            .insert(("t_d".into(), 10), "cit_diplomat".into());
        let tali = Character {
            id: "tali".into(),
            tags: BTreeMap::from([(Game::Le3, vec!["hench_tali".into()])]),
            owns: BTreeMap::new(),
            scene_rules: BTreeMap::new(),
            bark_banks: BTreeMap::new(),
            line_flags: BTreeMap::new(),
        };
        let label = |n: &Node| speaker_label(&tali, Game::Le3, &c, "t_d", &graphs, n);
        assert_eq!(label(&c.entries[0]), "cit_diplomat");
        assert_eq!(label(&c.entries[1]), "tali");
        assert_eq!(label(&c.replies[1]), "shepard");
    }

    #[test]
    fn structure() {
        let tags = |listener: &str| {
            let mut m = Map::new();
            m.insert("listener".into(), json!(listener));
            m.insert("line_kind".into(), json!("exchange"));
            structure_tags(&m)
        };
        assert_eq!(tags("shepard"), ["exchange", "to_shepard"]);
        assert_eq!(tags("hench_garrus"), ["exchange", "to_squad"]);
        assert_eq!(tags("citprs_taliplot_csec"), ["exchange", "to_npc"]);
        assert_eq!(tags("owner"), ["exchange"]);
        assert_eq!(tags("speaker_3"), ["exchange"]);
        assert!(structure_tags(&Map::new()).is_empty());
    }

    #[test]
    fn isact_barks() {
        assert_eq!(
            parse_isact_bark("tali", "EN_tali_00168422_png00.wav"),
            Some((168422, "png".to_string()))
        );
        assert_eq!(
            parse_isact_bark("tali", "EN_tali_00172830_lwh00.wav"),
            Some((172830, "lwh".to_string()))
        );
        // Another bank's streams, dialogue names, and malformed ones.
        assert_eq!(
            parse_isact_bark("tali", "EN_cha00_quarian_tali_00127848_M.wav"),
            None
        );
        assert_eq!(parse_isact_bark("tali", "EN_tali_0016842_png00.wav"), None);
        assert_eq!(parse_isact_bark("tali", "EN_tali_00168422_00.wav"), None);
        assert_eq!(flags_for("", "tali_ss"), ["no_subtitle", "bark"]);
    }

    #[test]
    fn line_flags_from_sources() {
        let c = Character::from_sources(
            "tali",
            "[le3]\nspeaker_tags = [\"hench_tali\"]\n[le3.line_flags]\nsinging = [\"le3.s769911_m\"]\n",
        )
        .unwrap();
        assert_eq!(c.line_flags["singing"], ["le3.s769911_m"]);
        let bad = "[le3]\nspeaker_tags = []\n[le3.line_flags]\nsinging = [\"le2.s1_m\"]\n";
        assert!(Character::from_sources("tali", bad).is_err());
    }

    #[test]
    fn isact_names() {
        assert_eq!(
            parse_isact_name("EN_cha00_quarian_tali_00127848_M.wav"),
            Some((127848, Gender::Male))
        );
        assert_eq!(
            parse_isact_name("EN_x_00000007_F.wav"),
            Some((7, Gender::Female))
        );
        assert_eq!(parse_isact_name("FR_x_00000007_F.wav"), None);
        assert_eq!(parse_isact_name("EN_x_y_M.wav"), None);
    }

    #[test]
    fn flags() {
        assert_eq!(
            flags_for("Keelah.", "nortla_debrief_d"),
            Vec::<String>::new()
        );
        assert_eq!(flags_for(" {Sighs} ", "x"), ["nonverbal"]);
        assert_eq!(
            flags_for("", "ss_global_hench_tali"),
            ["no_subtitle", "bark"]
        );
    }

    #[test]
    fn prefixes_and_dlc() {
        let p = Path::new(
            "Game/ME3/BioGame/DLC/DLC_EXP_Pack003/CookedPCConsole/BioD_Cit_200_LOC_INT.pcc",
        );
        assert_eq!(package_prefix(p), "BioD_Cit");
        assert_eq!(dlc_name(p).as_deref(), Some("DLC_EXP_Pack003"));
        assert_eq!(
            dlc_name(Path::new("Game/ME2/BioGame/CookedPCConsole/a.pcc")),
            None
        );
    }

    #[test]
    fn resamples_a_tone() {
        let from = 44_100;
        let tone: Vec<f32> = (0..from)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / from as f32).sin() * 0.5)
            .collect();
        let out = resample(&tone, from, SAMPLE_RATE).unwrap();
        assert!(
            (out.len() as i64 - SAMPLE_RATE as i64).abs() < 50,
            "{}",
            out.len()
        );
        let peak = out[1000..out.len() - 1000]
            .iter()
            .fold(0f32, |m, s| m.max(s.abs()));
        assert!((peak - 0.5).abs() < 0.02, "{peak}");
    }

    #[test]
    fn reads_both_pcm_kinds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        dataset::wav::write_mono16(&path, 44_100, &[0, 16384, -16384]).unwrap();
        let (rate, s) = read_pcm(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(rate, 44_100);
        assert_eq!(s, [0.0, 0.5, -0.5]);
    }
}
