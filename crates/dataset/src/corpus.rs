//! Harvest records (`lines.jsonl`) and their validation.
//!
//! Field meanings are in `docs/design/corpus-format.md`. Unknown fields are
//! kept in `extra` so tools that rewrite records don't drop them.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{hash, wav, Error, IoContext};

pub const LINES: &str = "lines.jsonl";
pub const GAMES: [&str; 3] = ["le1", "le2", "le3"];
pub const MAX_CONTEXT: usize = 4;
const MAX_ID_LEN: usize = 120;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub id: String,
    pub speaker: String,
    /// Empty only for lines flagged `no_subtitle`.
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
    pub audio: Audio,
    pub source: Source,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<ContextLine>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scene_tags: Vec<String>,
    /// LE3 voice director's note, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_direction: Option<String>,
    /// `prev`, `next`, `scene` and anything newer.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Audio {
    /// Relative to the stage's `out/` folder.
    pub path: String,
    pub sha256: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_s: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub game: String,
    /// Package, conversation, string id, export and so on: whatever the
    /// extractor knows.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextLine {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub speaker: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub string_id: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Whether `id` is a valid line id: `<game>.<rest>` in `[a-z0-9._-]`.
pub fn valid_id(id: &str) -> bool {
    let Some((game, rest)) = id.split_once('.') else {
        return false;
    };
    GAMES.contains(&game)
        && !rest.is_empty()
        && id.len() <= MAX_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
}

/// Reads `lines.jsonl` from a harvest output folder.
pub fn read_lines(dir: &Path) -> Result<Vec<Line>, Error> {
    let path = dir.join(LINES);
    let text = fs::read_to_string(&path).at(&path)?;
    text.lines()
        .enumerate()
        .map(|(n, line)| {
            serde_json::from_str(line).map_err(|e| Error::Invalid {
                path: path.clone(),
                message: format!("line {}: {e}", n + 1),
            })
        })
        .collect()
}

pub fn write_lines(dir: &Path, lines: &[Line]) -> Result<(), Error> {
    let path = dir.join(LINES);
    let mut text = String::new();
    for line in lines {
        text.push_str(&serde_json::to_string(line).expect("records serialize"));
        text.push('\n');
    }
    fs::write(&path, text).at(&path)
}

#[derive(Debug, Default)]
pub struct Report {
    pub lines: usize,
    pub seconds: f64,
    pub errors: Vec<String>,
}

/// Checks a harvest output folder against the format: every record parses,
/// ids are valid and unique, and each clip exists, matches its hash and
/// has the header its record claims.
pub fn check(dir: &Path) -> Result<Report, Error> {
    let path = dir.join(LINES);
    let text = fs::read_to_string(&path).at(&path)?;
    let mut report = Report::default();
    let mut ids = BTreeSet::new();
    for (n, raw) in text.lines().enumerate() {
        let at = |msg: String| format!("{LINES}:{}: {msg}", n + 1);
        if raw.trim().is_empty() {
            report.errors.push(at("blank line".into()));
            continue;
        }
        let line: Line = match serde_json::from_str(raw) {
            Ok(line) => line,
            Err(e) => {
                report.errors.push(at(e.to_string()));
                continue;
            }
        };
        report.lines += 1;
        let mut problems = Vec::new();
        if !valid_id(&line.id) {
            problems.push(format!("invalid id {:?}", line.id));
        } else if !line.id.starts_with(&format!("{}.", line.source.game)) {
            problems.push(format!(
                "id {} doesn't start with its game {}",
                line.id, line.source.game
            ));
        }
        if !ids.insert(line.id.clone()) {
            problems.push(format!("duplicate id {}", line.id));
        }
        if !GAMES.contains(&line.source.game.as_str()) {
            problems.push(format!("unknown game {:?}", line.source.game));
        }
        if line.speaker.is_empty() || line.speaker != line.speaker.to_lowercase() {
            problems.push(format!("speaker {:?} must be lowercase", line.speaker));
        }
        if line.text.trim().is_empty() && !line.flags.iter().any(|f| f == "no_subtitle") {
            problems.push("empty text without the no_subtitle flag".into());
        }
        if line.context.len() > MAX_CONTEXT {
            problems.push(format!("more than {MAX_CONTEXT} context lines"));
        }
        if line.scene_tags.iter().any(|t| t != &t.to_lowercase()) {
            problems.push("scene tags must be lowercase".into());
        }
        problems.extend(check_audio(dir, &line.audio));
        report.seconds += line.audio.duration_s;
        report.errors.extend(
            problems
                .into_iter()
                .map(|p| at(format!("{}: {p}", line.id))),
        );
    }
    Ok(report)
}

fn check_audio(dir: &Path, audio: &Audio) -> Vec<String> {
    let rel = Path::new(&audio.path);
    if rel.is_absolute() || rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return vec![format!(
            "audio path {:?} must stay inside the folder",
            audio.path
        )];
    }
    let file = dir.join(rel);
    let mut problems = Vec::new();
    match hash::sha256_file(&file) {
        Ok(h) if h == audio.sha256 => {}
        Ok(h) => problems.push(format!(
            "{} has hash {h}, record says {}",
            audio.path, audio.sha256
        )),
        Err(_) => return vec![format!("missing audio {}", audio.path)],
    }
    match wav::read_info(&file) {
        Ok(info) => {
            if info.channels != 1 || audio.channels != 1 {
                problems.push(format!("{} is not mono", audio.path));
            }
            if info.sample_rate != audio.sample_rate {
                problems.push(format!(
                    "{} is {} Hz, record says {}",
                    audio.path, info.sample_rate, audio.sample_rate
                ));
            }
            if (info.duration_s - audio.duration_s).abs() > 0.01 {
                problems.push(format!(
                    "{} lasts {:.3}s, record says {:.3}s",
                    audio.path, info.duration_s, audio.duration_s
                ));
            }
        }
        Err(e) => problems.push(e.to_string()),
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert!(valid_id("le2.s348112_f"));
        assert!(valid_id("le3.s612004.2"));
        assert!(!valid_id("le4.x"));
        assert!(!valid_id("le2."));
        assert!(!valid_id("le2.Upper"));
        assert!(!valid_id("le2.a/b"));
        assert!(!valid_id(&format!("le1.{}", "a".repeat(200))));
    }

    #[test]
    fn durations_survive_a_round_trip_exactly() {
        // Without serde_json's float_roundtrip feature this reads back as
        // ...337, so rewriting lines.jsonl would change every such clip.
        let raw = r#"{"id":"le1.x","speaker":"tali","text":"hi","audio":{"path":"a.wav","sha256":"00","sample_rate":24000,"channels":1,"duration_s":1.8145833333333334},"source":{"game":"le1"}}"#;
        let line: Line = serde_json::from_str(raw).unwrap();
        assert_eq!(serde_json::to_string(&line).unwrap(), raw);
    }

    #[test]
    fn unknown_fields_survive_a_round_trip() {
        let raw = r#"{"id":"le1.x","speaker":"tali","text":"hi","audio":{"path":"audio/le1.x.wav","sha256":"00","sample_rate":24000,"channels":1,"duration_s":1.0,"codec":"pcm"},"source":{"game":"le1","string_id":5},"future":true}"#;
        let line: Line = serde_json::from_str(raw).unwrap();
        let back: Value = serde_json::to_value(&line).unwrap();
        assert_eq!(back, serde_json::from_str::<Value>(raw).unwrap());
    }
}
