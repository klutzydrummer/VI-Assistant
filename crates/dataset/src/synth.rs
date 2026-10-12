//! Synthetic harvest output: tones and invented lines in the real format.
//!
//! Game audio can't go in the repo or tests, so this stands in for the
//! extractor wherever something needs a corpus to read.

use std::f64::consts::TAU;
use std::fs;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::corpus::{self, Audio, ContextLine, Line, Source};
use crate::{hash, wav, Error, IoContext};

const RATE: u32 = 24_000;
const SPEAKERS: [&str; 3] = ["shepard", "garrus", "admiral_raan"];
const TAGS: [&[&str]; 4] = [&[], &["combat"], &["citadel_party"], &["flotilla", "trial"]];
const PHRASES: [&str; 6] = [
    "The drive core is humming nicely today.",
    "I could recalibrate that in my sleep.",
    "Keelah, that was close.",
    "Have you eaten anything that wasn't a ration bar?",
    "This is synthetic line number {n}, not a real recording.",
    "Shields up, I'm going in!",
];

/// Writes `lines.jsonl` and `audio/` for `count` invented lines into `dir`.
/// Deterministic: the same count always gives byte-identical files.
pub fn write(dir: &Path, character: &str, count: usize) -> Result<Vec<Line>, Error> {
    let audio_dir = dir.join("audio");
    fs::create_dir_all(&audio_dir).at(&audio_dir)?;
    let mut lines = Vec::with_capacity(count);
    for n in 0..count {
        let game = corpus::GAMES[n % 3];
        let id = format!("{game}.s{:06}_f", 900_000 + n);
        // 0.6 s to 3.6 s, at pitches a voice might sit around.
        let frames = (RATE as usize * (6 + (n * 7) % 31)) / 10;
        let freq = 180.0 + (n * 37 % 160) as f64;
        let samples: Vec<i16> = (0..frames)
            .map(|i| {
                let t = i as f64 / RATE as f64;
                let fade =
                    (t * 20.0).min(1.0) * ((frames - i) as f64 / RATE as f64 * 20.0).min(1.0);
                ((TAU * freq * t).sin() * fade * 8000.0) as i16
            })
            .collect();
        let rel = format!("audio/{id}.wav");
        let path = dir.join(&rel);
        wav::write_mono16(&path, RATE, &samples)?;

        let text = PHRASES[n % PHRASES.len()].replace("{n}", &n.to_string());
        let context = (0..n % 3)
            .map(|k| ContextLine {
                id: None,
                speaker: SPEAKERS[(n + k) % SPEAKERS.len()].into(),
                text: format!("Synthetic context line {k} before line {n}."),
                string_id: Some(1_000 + (n * 10 + k) as u64),
                extra: Map::new(),
            })
            .collect();
        let mut source_extra = Map::new();
        source_extra.insert(
            "conversation".into(),
            json!(format!("synthetic_conv_{}", n / 5)),
        );
        source_extra.insert("string_id".into(), json!(1_000 + n as u64 * 10 + 9));
        source_extra.insert("synthetic".into(), Value::Bool(true));

        lines.push(Line {
            id,
            speaker: character.into(),
            text,
            flags: Vec::new(),
            audio: Audio {
                sha256: hash::sha256_file(&path)?,
                path: rel,
                sample_rate: RATE,
                channels: 1,
                duration_s: frames as f64 / RATE as f64,
                extra: Map::new(),
            },
            source: Source {
                game: game.into(),
                extra: source_extra,
            },
            context,
            scene_tags: TAGS[n % TAGS.len()].iter().map(|t| t.to_string()).collect(),
            stage_direction: (game == "le3").then(|| "Synthetic direction: quietly.".to_string()),
            extra: Map::new(),
        });
    }
    corpus::write_lines(dir, &lines)?;
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_corpus_passes_check_and_is_deterministic() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        write(a.path(), "tali", 12).unwrap();
        write(b.path(), "tali", 12).unwrap();

        let report = corpus::check(a.path()).unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.lines, 12);

        let mut cache = hash::HashCache::default();
        assert_eq!(cache.path(a.path()).unwrap(), cache.path(b.path()).unwrap());
    }

    #[test]
    fn check_catches_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let mut lines = write(dir.path(), "tali", 3).unwrap();
        fs::write(dir.path().join(&lines[0].audio.path), b"RIFF").unwrap();
        lines[1].id = lines[2].id.clone();
        lines[2].audio.duration_s += 1.0;
        corpus::write_lines(dir.path(), &lines).unwrap();
        let errors = corpus::check(dir.path()).unwrap().errors;
        assert!(errors.iter().any(|e| e.contains("has hash")), "{errors:?}");
        assert!(
            errors.iter().any(|e| e.contains("duplicate id")),
            "{errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.contains("record says")),
            "{errors:?}"
        );
    }
}
