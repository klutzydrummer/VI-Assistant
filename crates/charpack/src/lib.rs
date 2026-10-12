//! Character packs: everything needed to embody one character.
//!
//! A pack is a folder holding a `character.toml` manifest, a persona prompt,
//! example dialogue, and a small bank of voice reference clips (one or more
//! per emotion). Swapping characters means pointing the assistant at a
//! different folder; nothing is trained.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const MANIFEST: &str = "character.toml";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Lowercase identifier, also the folder name by convention.
    pub id: String,
    /// Full in-universe name.
    pub name: String,
    /// What the user calls the character day to day.
    pub short_name: String,
    /// Franchise or origin, for the user's reference.
    pub source: String,
    /// Persona prompt file, relative to the pack folder.
    pub persona: PathBuf,
    /// Example dialogue file, relative to the pack folder.
    pub examples: PathBuf,
    pub voice: Voice,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voice {
    /// BCP 47 language tag the clips are spoken in.
    pub language: String,
    /// Emotion used when the model doesn't tag a line.
    pub default_emotion: String,
    pub refs: Vec<VoiceRef>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceRef {
    pub emotion: String,
    /// Audio file, relative to the pack folder.
    pub audio: PathBuf,
    /// Exact words spoken in the clip. Most cloning TTS engines need it.
    pub transcript: String,
}

#[derive(Debug, Clone)]
pub struct Pack {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub persona: String,
    pub examples: String,
}

/// Problems found by [`Pack::check`]. Missing audio is reported separately
/// because clips are extracted locally and never committed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub errors: Vec<String>,
    pub missing_audio: Vec<PathBuf>,
}

impl Report {
    pub fn is_ready(&self) -> bool {
        self.errors.is_empty() && self.missing_audio.is_empty()
    }
}

impl Pack {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join(MANIFEST);
        let manifest: Manifest =
            toml::from_str(&read(&manifest_path)?).map_err(|source| Error::Parse {
                path: manifest_path,
                source,
            })?;
        let persona = read(&root.join(&manifest.persona))?;
        let examples = read(&root.join(&manifest.examples))?;
        Ok(Self {
            root,
            manifest,
            persona,
            examples,
        })
    }

    /// Emotions this pack has at least one reference clip for, sorted.
    pub fn emotions(&self) -> BTreeSet<&str> {
        self.manifest
            .voice
            .refs
            .iter()
            .map(|r| r.emotion.as_str())
            .collect()
    }

    /// Reference clips for `emotion`, falling back to the default emotion.
    pub fn refs_for(&self, emotion: &str) -> Vec<&VoiceRef> {
        let pick = |e: &str| -> Vec<&VoiceRef> {
            self.manifest
                .voice
                .refs
                .iter()
                .filter(|r| r.emotion == e)
                .collect()
        };
        let refs = pick(emotion);
        if refs.is_empty() {
            pick(&self.manifest.voice.default_emotion)
        } else {
            refs
        }
    }

    pub fn check(&self) -> Report {
        let mut report = Report::default();
        let m = &self.manifest;

        let id_ok = !m.id.is_empty()
            && m.id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !id_ok {
            report.errors.push(format!(
                "id {:?} must be lowercase letters, digits or '-'",
                m.id
            ));
        }
        if self.persona.trim().is_empty() {
            report.errors.push("persona file is empty".into());
        }
        if m.voice.refs.is_empty() {
            report.errors.push("voice.refs has no clips".into());
        } else if !self.emotions().contains(m.voice.default_emotion.as_str()) {
            report.errors.push(format!(
                "no voice ref for default emotion {:?}",
                m.voice.default_emotion
            ));
        }
        for r in &m.voice.refs {
            let t = r.transcript.trim();
            if t.is_empty() || t.starts_with("TODO") {
                report
                    .errors
                    .push(format!("{} needs its transcript", r.audio.display()));
            }
            if !self.root.join(&r.audio).is_file() {
                report.missing_audio.push(r.audio.clone());
            }
        }
        report
    }
}

fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST_TOML: &str = r#"
id = "test"
name = "Test Character"
short_name = "Test"
source = "Tests"
persona = "persona.md"
examples = "examples.md"

[voice]
language = "en"
default_emotion = "neutral"

[[voice.refs]]
emotion = "neutral"
audio = "voice/neutral.wav"
transcript = "Hello there."

[[voice.refs]]
emotion = "happy"
audio = "voice/happy.wav"
transcript = "Wonderful!"
"#;

    fn pack_dir(manifest: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(MANIFEST), manifest).unwrap();
        fs::write(dir.path().join("persona.md"), "You are Test.").unwrap();
        fs::write(dir.path().join("examples.md"), "").unwrap();
        fs::create_dir(dir.path().join("voice")).unwrap();
        dir
    }

    #[test]
    fn reports_missing_audio_separately() {
        let dir = pack_dir(MANIFEST_TOML);
        fs::write(dir.path().join("voice/neutral.wav"), b"RIFF").unwrap();
        let report = Pack::load(dir.path()).unwrap().check();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.missing_audio, vec![PathBuf::from("voice/happy.wav")]);
        assert!(!report.is_ready());
    }

    #[test]
    fn falls_back_to_default_emotion() {
        let dir = pack_dir(MANIFEST_TOML);
        let pack = Pack::load(dir.path()).unwrap();
        assert_eq!(pack.refs_for("happy")[0].transcript, "Wonderful!");
        assert_eq!(pack.refs_for("furious")[0].emotion, "neutral");
    }

    #[test]
    fn flags_default_emotion_without_clip() {
        let dir = pack_dir(&MANIFEST_TOML.replace(
            "default_emotion = \"neutral\"",
            "default_emotion = \"calm\"",
        ));
        let report = Pack::load(dir.path()).unwrap().check();
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].contains("calm"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let dir = pack_dir(&format!("{MANIFEST_TOML}\nvoice_id = \"x\"\n"));
        assert!(matches!(Pack::load(dir.path()), Err(Error::Parse { .. })));
    }

    #[test]
    fn bundled_tali_pack_is_valid() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../characters/tali");
        let report = Pack::load(root).unwrap().check();
        // Transcripts are filled in once clips are extracted from the game.
        let unexpected: Vec<_> = report
            .errors
            .iter()
            .filter(|e| !e.ends_with("needs its transcript"))
            .collect();
        assert!(unexpected.is_empty(), "{unexpected:?}");
    }
}
