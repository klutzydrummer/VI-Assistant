//! `vi extract`: one character's LE2/LE3 dialogue as a harvest corpus
//! (`docs/design/corpus-format.md`, section "harvest").
//!
//! Lines are found by their `WwiseStream` exports (see [`crate::wwise`]):
//! the stream's name gives the speaker tag, conversation, string id and
//! gender, the TLK gives the subtitle, and the `.afc` gives the audio,
//! decoded with `vgmstream-cli` and resampled to 24 kHz mono 16-bit.
//!
//! Not yet here: LE1 (its audio is ISACT, reached through FaceFX), the
//! conversation graph (context, `prev`/`next`, node indices), scene tags
//! and LE3 stage directions. Records leave those fields out.
//!
//! Reads the install only; everything is written under `out`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use serde_json::{json, Map, Value};

use crate::install::GameDir;
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
    /// Speaker tags per game, as they appear in stream names.
    pub tags: BTreeMap<Game, Vec<String>>,
}

impl Character {
    /// Read `sources.toml` from a character pack:
    ///
    /// ```toml
    /// [le2]
    /// speaker_tags = ["hench_tali"]
    /// ```
    pub fn from_sources(id: &str, text: &str) -> Result<Character, String> {
        let table: toml::Table = text.parse().map_err(|e| format!("sources.toml: {e}"))?;
        let mut tags = BTreeMap::new();
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
        }
        Ok(Character {
            id: id.to_string(),
            tags,
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
    afc: PathBuf,
    offset: usize,
    size: usize,
}

/// A distinct recording: one string id, gender and set of audio bytes.
struct Recording {
    string_id: i32,
    gender: Gender,
    raw: Vec<u8>,
    raw_sha: String,
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
    /// Ids of streams skipped as placeholders (see [`MIN_SECONDS`]).
    pub placeholders: Vec<String>,
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
        if game.game == Game::Le1 {
            progress("le1: skipped (LE1 audio isn't supported yet)");
            continue;
        }
        let mut report = GameReport::default();
        let subs = Subtitles::load(game)?;
        progress(&format!(
            "{}: {} subtitles, scanning packages",
            game.game,
            subs.male.len()
        ));
        let recordings = find(game, tags, &mut report, progress);
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
                    "placeholders_skipped": r.placeholders,
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
        "not_covered": ["le1 audio", "conversation context", "scene tags", "stage directions", "barks without a stream name"],
    });
    let path = out.join("harvest.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&summary).unwrap() + "\n",
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(reports)
}

/// Every stream spoken under one of `tags`, grouped into distinct
/// recordings.
fn find(
    game: &GameDir,
    tags: &[String],
    report: &mut GameReport,
    progress: &dyn Fn(&str),
) -> Vec<Recording> {
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
            if !tags
                .iter()
                .any(|t| t.eq_ignore_ascii_case(&name.speaker_tag))
            {
                continue;
            }
            report.streams += 1;
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
                afc: afc.clone(),
                offset: stream.offset,
                size: stream.size,
            };
            let raw_sha = dataset::hash::sha256_bytes(&raw);
            let list = by_key.entry((name.string_id, name.gender)).or_default();
            if let Some(r) = list.iter_mut().find(|r| r.raw_sha == raw_sha) {
                r.found.push(found);
                report.merged_duplicates += 1;
            } else {
                list.push(Recording {
                    string_id: name.string_id,
                    gender: name.gender,
                    raw,
                    raw_sha,
                    found: vec![found],
                });
            }
        }
    }
    by_key.into_values().flatten().collect()
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
        let mut id = format!("{}.s{}_{}", game.game.id(), r.string_id, r.gender.suffix());
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
                let decoded = fs::write(&wem, &r.raw)
                    .map_err(|e| e.to_string())
                    .and_then(|_| decode(&wem, &wav));
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
                let (text, tlk) = match subs.get(r.string_id, r.gender) {
                    Some((t, f)) => (t.clone(), Some(f.clone())),
                    None => (String::new(), None),
                };
                let flags = flags_for(&text, &first.conversation);
                let mut audio_extra = Map::new();
                audio_extra.insert(
                    "source".into(),
                    json!({
                        "container": rel(&first.afc),
                        "offset": first.offset,
                        "size": first.size,
                        "export": first.export,
                        "codec": "wwise_vorbis",
                        "sample_rate": rate,
                        "sha256": r.raw_sha,
                    }),
                );
                let mut source = Map::new();
                source.insert("package".into(), json!(rel(&first.package)));
                source.insert("conversation".into(), json!(first.conversation));
                source.insert("string_id".into(), json!(r.string_id));
                source.insert("gender".into(), json!(r.gender.suffix()));
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

/// Corpus flags for a line: `no_subtitle` for empty text, `nonverbal` for
/// subtitles that only describe a sound (`{Sighs}`), `bark` for the squad
/// sound sets (`ss_global_hench_tali`, `ss_tali_minimal`), which have no
/// conversation node and no subtitles.
fn flags_for(text: &str, conversation: &str) -> Vec<String> {
    let text = text.trim();
    let mut flags = Vec::new();
    if text.is_empty() {
        flags.push("no_subtitle".to_string());
    } else if text.starts_with('{') && text.ends_with('}') {
        flags.push("nonverbal".to_string());
    }
    if conversation.to_ascii_lowercase().starts_with("ss_") {
        flags.push("bark".to_string());
    }
    flags
}

/// `BioD_Nor_111DebriefTali_LOC_INT.pcc` -> `BioD_Nor`.
fn package_prefix(p: &Path) -> String {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    stem.splitn(3, '_').take(2).collect::<Vec<_>>().join("_")
}

/// Decode a `.wem` with `vgmstream-cli`; mono samples in [-1, 1] and the
/// rate.
fn decode(wem: &Path, wav: &Path) -> Result<(u32, Vec<f32>), String> {
    let output = Command::new("vgmstream-cli")
        .arg("-o")
        .arg(wav)
        .arg(wem)
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
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                        .collect(),
                    (3, 32) => body
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
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
        assert!(Character::from_sources("t", "[le2]\n").is_err());
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
