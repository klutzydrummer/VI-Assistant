//! Character packs: everything needed to embody one character.
//!
//! A pack is a folder holding a `character.toml` manifest, a persona prompt,
//! example dialogue, and a small bank of voice reference clips (one or more
//! per emotion). Swapping characters means pointing the assistant at a
//! different folder; nothing is trained.
//!
//! Packs may also define a state machine (see [`machine`]) and a pool of
//! names the character gives the agents it dispatches.

pub mod machine;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use machine::{Machine, MachineSpec};

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
    /// Names the character gives the agents it sends out, relative to the
    /// pack folder.
    pub agents: Option<PathBuf>,
    /// State machine definition, relative to the pack folder.
    pub states: Option<PathBuf>,
    pub voice: Voice,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentsFile {
    agent: Vec<AgentName>,
}

/// One name from the character's pool for dispatched agents.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentName {
    /// What the character calls it day to day.
    pub name: String,
    pub full_name: String,
    /// The kind of job the character prefers to give it.
    pub role: String,
    pub temperament: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voice {
    /// BCP 47 language tag the clips are spoken in.
    pub language: String,
    /// Emotion used when the director gives none.
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
    pub machine_spec: Option<MachineSpec>,
    /// State id to that state's extra persona instructions.
    pub state_prompts: BTreeMap<String, String>,
    pub agents: Vec<AgentName>,
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
        let machine_spec: Option<MachineSpec> = match &manifest.states {
            Some(rel) => {
                let path = root.join(rel);
                Some(
                    toml::from_str(&read(&path)?)
                        .map_err(|source| Error::Parse { path, source })?,
                )
            }
            None => None,
        };
        let state_prompts = machine_spec
            .iter()
            .flat_map(|spec| &spec.states)
            .filter_map(|st| st.prompt.as_ref().map(|p| (st, p)))
            .map(|(st, p)| Ok((st.id.clone(), read(&root.join(p))?)))
            .collect::<Result<_, Error>>()?;
        let agents = match &manifest.agents {
            Some(rel) => {
                let path = root.join(rel);
                let file: AgentsFile = toml::from_str(&read(&path)?)
                    .map_err(|source| Error::Parse { path, source })?;
                file.agent
            }
            None => Vec::new(),
        };
        Ok(Self {
            root,
            manifest,
            persona,
            examples,
            machine_spec,
            state_prompts,
            agents,
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

    /// The pack's state machine, if it defines one, or why it is invalid.
    pub fn machine(&self) -> Option<Result<Machine, Vec<String>>> {
        self.machine_spec.clone().map(Machine::new)
    }

    /// Reference clips for a line with `emotion` while in `state`, for
    /// engines that take clips rather than control vectors. A state with
    /// its own voice overrides the line's emotion.
    pub fn refs_in_state(&self, state: &str, emotion: &str) -> Vec<&VoiceRef> {
        let voice = self
            .machine_spec
            .iter()
            .flat_map(|spec| &spec.states)
            .find(|st| st.id == state)
            .and_then(|st| st.voice.as_deref());
        self.refs_for(voice.unwrap_or(emotion))
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

        if let Some(Err(errors)) = self.machine() {
            report.errors.extend(errors);
        }
        let emotions = self.emotions();
        for st in self.machine_spec.iter().flat_map(|spec| &spec.states) {
            if let Some(e) = &st.voice {
                if !emotions.contains(e.as_str()) {
                    report.errors.push(format!(
                        "state {:?} uses voice {e:?}, which has no voice ref",
                        st.id
                    ));
                }
            }
        }

        let mut agent_names = BTreeSet::new();
        for a in &self.agents {
            if a.name.trim().is_empty() {
                report.errors.push("an agent has an empty name".into());
            } else if !agent_names.insert(a.name.to_lowercase()) {
                report
                    .errors
                    .push(format!("agent name {:?} is used twice", a.name));
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
states = "states.toml"

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

    const STATES_TOML: &str = r#"
initial = "calm"

[[event]]
id = "good_news"
description = "Hears good news"

[[state]]
id = "calm"
description = "Even keel"

[[state]]
id = "giddy"
description = "Overjoyed"
prompt = "states/giddy.md"
voice = "happy"

[[transition]]
from = ["calm"]
on = "good_news"
to = "giddy"
"#;

    fn pack_dir(manifest: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(MANIFEST), manifest).unwrap();
        fs::write(dir.path().join("persona.md"), "You are Test.").unwrap();
        fs::write(dir.path().join("examples.md"), "").unwrap();
        fs::create_dir(dir.path().join("voice")).unwrap();
        fs::write(dir.path().join("states.toml"), STATES_TOML).unwrap();
        fs::create_dir(dir.path().join("states")).unwrap();
        fs::write(dir.path().join("states/giddy.md"), "Everything is great.").unwrap();
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
    fn state_voice_overrides_line_emotion() {
        let dir = pack_dir(MANIFEST_TOML);
        let pack = Pack::load(dir.path()).unwrap();
        assert_eq!(pack.state_prompts["giddy"], "Everything is great.");
        assert_eq!(pack.refs_in_state("giddy", "neutral")[0].emotion, "happy");
        assert_eq!(pack.refs_in_state("calm", "neutral")[0].emotion, "neutral");
        let machine = pack.machine().unwrap().unwrap();
        let mut snap = machine.start();
        machine.apply(&mut snap, "good_news").unwrap();
        assert_eq!(snap.state, "giddy");
    }

    #[test]
    fn flags_state_voice_without_clip() {
        let dir = pack_dir(MANIFEST_TOML);
        fs::write(
            dir.path().join("states.toml"),
            STATES_TOML.replace("voice = \"happy\"", "voice = \"drunk\""),
        )
        .unwrap();
        let report = Pack::load(dir.path()).unwrap().check();
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].contains("drunk"));
    }

    #[test]
    fn flags_duplicate_agent_names() {
        let dir = pack_dir(&MANIFEST_TOML.replace(
            "examples = \"examples.md\"",
            "examples = \"examples.md\"\nagents = \"agents.toml\"",
        ));
        let agent = |name: &str| {
            format!("[[agent]]\nname = \"{name}\"\nfull_name = \"{name} vas Test\"\nrole = \"r\"\ntemperament = \"t\"\n")
        };
        fs::write(
            dir.path().join("agents.toml"),
            agent("Zikka") + &agent("zikka"),
        )
        .unwrap();
        let pack = Pack::load(dir.path()).unwrap();
        assert_eq!(pack.agents.len(), 2);
        let report = pack.check();
        assert_eq!(report.errors, vec!["agent name \"zikka\" is used twice"]);
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
