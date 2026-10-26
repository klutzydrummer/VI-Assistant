//! Clip lists: the quickest way for a job or a Claude thread to put clips
//! in front of Brandon. It writes one small `clips.toml` (a question and
//! the clips, by corpus line id or audio path); the hub finds it, plays
//! each clip with her subtitle, and saves his quick answers beside it in
//! `answers.jsonl`. See "Clip lists" in `docs/design/review-page.md`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::corpus::current_out;
use crate::server::{now, Reply};

pub const CLIPS: &str = "clips.toml";
pub const ANSWERS: &str = "answers.jsonl";
pub const DEFAULT_ANSWERS: [&str; 3] = ["yes", "no", "unsure"];

const PAGE: &str = include_str!("clips.html");

/// `clips.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct File {
    pub title: String,
    /// What to listen for, shown above every clip.
    #[serde(default)]
    pub question: String,
    /// The answer buttons; yes / no / unsure unless given.
    #[serde(default)]
    pub answers: Vec<String>,
    /// Hide each clip's label until every clip is answered.
    #[serde(default)]
    pub blind: bool,
    /// The thread the summary goes back to.
    #[serde(default)]
    pub asked_by: String,
    #[serde(default)]
    pub clip: Vec<Entry>,
}

/// One clip: a corpus line `id` (her game clip, cleaned if clean has run),
/// or an audio `path` relative to the `clips.toml` folder (`corpus:` for
/// relative to the corpus folder).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub label: String,
    /// Shown under the clip (the subtitle, for an id, unless given).
    #[serde(default)]
    pub text: Option<String>,
}

impl Entry {
    /// What answers are saved under.
    pub fn key(&self) -> &str {
        self.id.as_deref().or(self.path.as_deref()).unwrap_or("")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub clip: String,
    pub answer: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub at: u64,
}

/// Her lines' audio and subtitles, from harvest and clean, read once and
/// shared by every clip list.
#[derive(Default)]
pub struct Index {
    audio: HashMap<String, PathBuf>,
    text: HashMap<String, String>,
}

fn relative(rel: &str) -> Option<&Path> {
    let p = Path::new(rel);
    (!rel.is_empty()
        && p.components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)))
    .then_some(p)
}

fn jsonl(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

impl Index {
    pub fn load(root: &Path) -> Self {
        let mut index = Index::default();
        if let Ok(Some(harvest)) = current_out(root, "harvest") {
            for v in jsonl(&harvest.join("lines.jsonl")) {
                let Some(id) = v["id"].as_str() else { continue };
                if let Some(t) = v["text"].as_str() {
                    index.text.insert(id.into(), t.into());
                }
                if let Some(p) = v["audio"]["path"].as_str().and_then(relative) {
                    index.audio.insert(id.into(), harvest.join(p));
                }
            }
        }
        if let Ok(Some(clean)) = current_out(root, "clean") {
            for v in jsonl(&clean.join(crate::corpus::CLIPS)) {
                if let (Some(id), Some(p)) = (
                    v["id"].as_str(),
                    v["audio"]["path"].as_str().and_then(relative),
                ) {
                    index.audio.insert(id.into(), clean.join(p));
                }
            }
        }
        index
    }
}

/// One clip list, with its clips resolved to audio.
pub struct Clips {
    pub dir: PathBuf,
    pub file: File,
    audio: Vec<Option<PathBuf>>,
    text: Vec<String>,
    answers: BTreeMap<String, Answer>,
    seen: Option<SystemTime>,
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl Clips {
    pub fn load(root: &Path, dir: &Path, index: &Index) -> Result<Self, String> {
        let path = dir.join(CLIPS);
        let file: File = fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| toml::from_str(&t).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut audio = Vec::new();
        let mut text = Vec::new();
        let mut keys = std::collections::HashSet::new();
        for (n, c) in file.clip.iter().enumerate() {
            let where_ = || format!("{}: clip {}", path.display(), n + 1);
            let a = match (&c.id, &c.path) {
                (Some(id), None) => index.audio.get(id).cloned(),
                (None, Some(p)) => {
                    let (base, rel) = match p.strip_prefix("corpus:") {
                        Some(rel) => (root, rel),
                        None => (dir, p.as_str()),
                    };
                    let rel = relative(rel)
                        .ok_or_else(|| format!("{}: path must stay inside its folder", where_()))?;
                    Some(base.join(rel))
                }
                _ => return Err(format!("{}: give an id or a path, not both", where_())),
            };
            if !keys.insert(c.key().to_string()) {
                return Err(format!("{}: {} is listed twice", where_(), c.key()));
            }
            text.push(
                c.text
                    .clone()
                    .or_else(|| c.id.as_ref().and_then(|id| index.text.get(id).cloned()))
                    .unwrap_or_default(),
            );
            audio.push(a.filter(|p| p.is_file()));
        }
        let mut answers = BTreeMap::new();
        for v in jsonl(&dir.join(ANSWERS)) {
            if let Ok(a) = serde_json::from_value::<Answer>(v) {
                answers.insert(a.clip.clone(), a);
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            file,
            audio,
            text,
            answers,
            seen: mtime(&path),
        })
    }

    /// Whether `clips.toml` changed since it was read.
    pub fn stale(&self) -> bool {
        mtime(&self.dir.join(CLIPS)) != self.seen
    }

    fn choices(&self) -> Vec<String> {
        if self.file.answers.is_empty() {
            DEFAULT_ANSWERS.map(String::from).to_vec()
        } else {
            self.file.answers.clone()
        }
    }

    pub fn total(&self) -> usize {
        self.file.clip.len()
    }

    pub fn done(&self) -> usize {
        self.file
            .clip
            .iter()
            .filter(|c| self.answers.contains_key(c.key()))
            .count()
    }

    pub fn session(&self) -> Value {
        let show = !self.file.blind || self.done() == self.total();
        json!({
            "title": self.file.title,
            "question": self.file.question,
            "choices": self.choices(),
            "blind": self.file.blind,
            "done": self.done(),
            "total": self.total(),
            "clips": self.file.clip.iter().enumerate().map(|(n, c)| json!({
                "n": n,
                "label": show.then_some(&c.label).filter(|l| !l.is_empty()),
                "text": self.text[n],
                "audio": self.audio[n].is_some(),
                "answer": self.answers.get(c.key()),
            })).collect::<Vec<_>>(),
        })
    }

    pub fn answer(&mut self, n: usize, answer: String, note: String) -> Result<(), String> {
        let c = self
            .file
            .clip
            .get(n)
            .ok_or_else(|| format!("no clip {n}"))?;
        if !self.choices().contains(&answer) {
            return Err(format!("answer must be one of {:?}", self.choices()));
        }
        if note.len() > 2000 {
            return Err("note too long".into());
        }
        let a = Answer {
            clip: c.key().to_string(),
            answer,
            note,
            at: now(),
        };
        let path = self.dir.join(ANSWERS);
        let line = serde_json::to_string(&a).map_err(|e| e.to_string())? + "\n";
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(line.as_bytes()))
            .map_err(|e| format!("could not save {}: {e}", path.display()))?;
        self.answers.insert(a.clip.clone(), a);
        Ok(())
    }

    /// His answers, one line per clip, for the thread.
    pub fn report(&self) -> String {
        self.file
            .clip
            .iter()
            .filter_map(|c| {
                let a = self.answers.get(c.key())?;
                let name = if c.label.is_empty() {
                    c.key().to_string()
                } else {
                    format!("{} ({})", c.label, c.key())
                };
                let note = if a.note.is_empty() {
                    String::new()
                } else {
                    format!(": {}", a.note.replace('\n', " "))
                };
                Some(format!("- {name}: {}{note}\n", a.answer))
            })
            .collect()
    }
}

#[derive(Deserialize)]
struct AnswerBody {
    n: usize,
    answer: String,
    #[serde(default)]
    note: String,
}

/// A clip list's routes, relative to its own address.
pub fn handle(c: &mut Clips, post: bool, url: &str, body: &str) -> Reply {
    let path = url.split('?').next().unwrap_or(url);
    match (post, path) {
        (false, "/") => Reply::Page(PAGE),
        (false, "/api/session") => Reply::Json(c.session()),
        (false, "/audio") => {
            let n = url
                .split_once("?n=")
                .and_then(|(_, n)| n.split('&').next()?.parse::<usize>().ok());
            match n.and_then(|n| c.audio.get(n).cloned().flatten()) {
                Some(p) => Reply::Audio(p),
                None => Reply::NotFound,
            }
        }
        (true, "/api/answer") => match serde_json::from_str::<AnswerBody>(body) {
            Ok(b) => match c.answer(b.n, b.answer, b.note) {
                Ok(()) => Reply::Json(c.session()),
                Err(e) => Reply::Bad(e),
            },
            Err(e) => Reply::Bad(e.to_string()),
        },
        _ => Reply::NotFound,
    }
}
