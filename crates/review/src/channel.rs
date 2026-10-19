//! Channel mode: Brandon marks, by ear, whether a clip carries an extra effect
//! on top of the suit filter (a radio or comms effect, or something else),
//! so a detector for it can be fitted and tested (see
//! `docs/design/after-labels-plan.md`, the channel check before step 3).
//!
//! - The sample is `analysis/after_labels/channel_key.jsonl` (written by
//!   `analysis/after_labels/sample_channel.py`): which clips, in what order.
//!   Only the ids and the order are read. Why a clip was picked (a radio tag,
//!   a measured outlier, a plain control) never reaches the page, and
//!   neither does its scene, so nothing anchors him.
//! - Each clip shows only her line's words and a play button. A mark can't be
//!   saved before the clip has been played.
//! - Per clip: plain, radio, other effect or can't tell, and a note.
//! - Every save is appended to `channel.jsonl` in the corpus folder; the last
//!   mark of a clip wins, and a restart resumes at the first clip not yet
//!   marked.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::corpus::{current_out, Error};
use crate::server::Reply;

const PAGE: &str = include_str!("channel.html");

pub const MARKS: &str = "channel.jsonl";
pub const SAMPLE: &str = "analysis/after_labels/channel_key.jsonl";
pub const CHOICES: [&str; 4] = ["plain", "radio", "other_effect", "cant_tell"];

/// One line of `channel.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    pub kind: String,
    pub id: String,
    pub channel: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Clip {
    pub id: String,
    pub text: String,
}

pub struct Channel {
    pub root: PathBuf,
    pub clips: Vec<Clip>,
    pub marks: HashMap<String, Mark>,
    audio: HashMap<String, PathBuf>,
}

fn read_values(path: &Path) -> Result<Vec<Value>, Error> {
    let text = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|source| Error::Parse {
                path: path.to_path_buf(),
                line: i + 1,
                source,
            })
        })
        .collect()
}

fn missing(path: &Path, what: &str) -> Error {
    Error::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, what.to_string()),
    }
}

impl Channel {
    /// `root` is the corpus folder.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let sample = root.join(SAMPLE);
        if !sample.exists() {
            return Err(missing(
                &sample,
                "no channel sample: run analysis/after_labels/sample_channel.py",
            ));
        }
        let harvest = current_out(&root, "harvest")?.ok_or(Error::NotRun("harvest"))?;
        // Her clips and words as the game has them. Harvest paths are
        // relative to its output and may not leave it.
        let mut found: HashMap<String, (String, PathBuf)> = HashMap::new();
        for v in read_values(&harvest.join("lines.jsonl"))? {
            if let (Some(id), Some(rel)) = (v["id"].as_str(), v["audio"]["path"].as_str()) {
                let rel = Path::new(rel);
                if rel
                    .components()
                    .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
                {
                    let text = v["text"].as_str().unwrap_or("").to_string();
                    found.insert(id.to_string(), (text, harvest.join(rel)));
                }
            }
        }
        let mut picks: Vec<(u64, String)> = read_values(&sample)?
            .into_iter()
            .filter_map(|r| Some((r["n"].as_u64()?, r["id"].as_str()?.to_string())))
            .collect();
        picks.sort();
        let mut clips = Vec::new();
        let mut audio = HashMap::new();
        for (_, id) in picks {
            // A clip without audio can't be marked by ear, so it's left out.
            let Some((text, path)) = found.remove(&id) else {
                continue;
            };
            clips.push(Clip {
                id: id.clone(),
                text,
            });
            audio.insert(id, path);
        }
        let mut marks = HashMap::new();
        let path = root.join(MARKS);
        if path.exists() {
            for v in read_values(&path)? {
                if let Ok(m) = serde_json::from_value::<Mark>(v) {
                    marks.insert(m.id.clone(), m);
                }
            }
        }
        Ok(Channel {
            root,
            clips,
            marks,
            audio,
        })
    }

    pub fn audio_path(&self, n: usize) -> Option<&PathBuf> {
        let clip = self.clips.get(n.checked_sub(1)?)?;
        self.audio.get(&clip.id)
    }

    pub fn marked(&self) -> usize {
        self.clips
            .iter()
            .filter(|c| self.marks.contains_key(&c.id))
            .count()
    }

    pub fn next_open(&self) -> Option<usize> {
        self.clips
            .iter()
            .position(|c| !self.marks.contains_key(&c.id))
            .map(|i| i + 1)
    }

    pub fn session(&self) -> Value {
        json!({
            "choices": CHOICES,
            "count": self.clips.len(),
            "marked": self.marked(),
            "next": self.next_open(),
            "done": self.clips.iter().map(|c| self.marks.contains_key(&c.id)).collect::<Vec<_>>(),
        })
    }

    pub fn item(&self, n: usize) -> Option<Value> {
        let clip = self.clips.get(n.checked_sub(1)?)?;
        Some(
            json!({"n": n, "count": self.clips.len(), "clip": clip, "mark": self.marks.get(&clip.id)}),
        )
    }

    pub fn mark(&mut self, body: MarkBody) -> Result<(), String> {
        let clip = body
            .n
            .checked_sub(1)
            .and_then(|i| self.clips.get(i))
            .ok_or_else(|| format!("no clip {}", body.n))?;
        if !CHOICES.contains(&body.channel.as_str()) {
            return Err(format!("no choice {}", body.channel));
        }
        if !body.heard {
            return Err("play the clip first".into());
        }
        let mark = Mark {
            kind: "mark".into(),
            id: clip.id.clone(),
            channel: body.channel,
            notes: body.notes.trim().to_string(),
            seconds: body.seconds.max(0.0),
            at: now(),
        };
        let path = self.root.join(MARKS);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut text = serde_json::to_string(&mark).expect("marks serialize");
        text.push('\n');
        file.write_all(text.as_bytes())
            .and_then(|()| file.sync_data())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.marks.insert(mark.id.clone(), mark);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct MarkBody {
    pub n: usize,
    pub channel: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub heard: bool,
    #[serde(default)]
    pub seconds: f64,
}

pub fn handle(c: &mut Channel, post: bool, url: &str, body: &str) -> Reply {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let n: Option<usize> = query
        .split('&')
        .find_map(|kv| kv.strip_prefix("n="))
        .and_then(|v| v.parse().ok());
    match (post, path) {
        (false, "/") => Reply::Page(PAGE),
        (false, "/api/session") => Reply::Json(c.session()),
        (false, "/audio") => match n.and_then(|n| c.audio_path(n)) {
            Some(p) => Reply::Audio(p.clone()),
            None => Reply::NotFound,
        },
        (false, "/api/item") => match n.and_then(|n| c.item(n)) {
            Some(v) => Reply::Json(v),
            None => Reply::NotFound,
        },
        (true, "/api/mark") => {
            let parsed: Result<MarkBody, _> = serde_json::from_str(body);
            match parsed.map_err(|e| e.to_string()).and_then(|b| c.mark(b)) {
                Ok(()) => Reply::Json(c.session()),
                Err(msg) => Reply::Bad(msg),
            }
        }
        _ => Reply::NotFound,
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corpus with harvest (three clips, one escaping its folder) and a
    /// sample of four (one unknown, one without a usable clip).
    fn corpus() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let harvest = root.join("stages/harvest/h1/out");
        fs::create_dir_all(harvest.join("audio")).unwrap();
        fs::write(root.join("stages/harvest/current"), "h1\n").unwrap();
        fs::write(harvest.join("audio/a.wav"), b"RIFFa").unwrap();
        fs::write(harvest.join("audio/b.wav"), b"RIFFb").unwrap();
        let lines = [
            json!({"id": "a", "text": "On it!", "audio": {"path": "audio/a.wav"}}),
            json!({"id": "b", "text": "Shepard!", "audio": {"path": "audio/b.wav"}}),
            json!({"id": "c", "text": "Hah!", "audio": {"path": "../../escape.wav"}}),
        ];
        fs::write(
            harvest.join("lines.jsonl"),
            lines
                .iter()
                .map(|v| v.to_string() + "\n")
                .collect::<String>(),
        )
        .unwrap();
        fs::create_dir_all(root.join("analysis/after_labels")).unwrap();
        let sample = [
            json!({"n": 2, "id": "a", "why": "radio_tag"}),
            json!({"n": 1, "id": "b", "why": "plain"}),
            json!({"n": 3, "id": "c"}),
            json!({"n": 4, "id": "gone"}),
        ];
        fs::write(
            root.join(SAMPLE),
            sample
                .iter()
                .map(|v| v.to_string() + "\n")
                .collect::<String>(),
        )
        .unwrap();
        dir
    }

    fn body(n: usize, channel: &str, heard: bool) -> MarkBody {
        MarkBody {
            n,
            channel: channel.into(),
            notes: String::new(),
            heard,
            seconds: 4.0,
        }
    }

    #[test]
    fn loads_playable_clips_in_order() {
        let dir = corpus();
        let c = Channel::load(dir.path()).unwrap();
        assert_eq!(
            c.clips.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(),
            ["b", "a"]
        );
        assert_eq!(c.clips[1].text, "On it!");
        assert_eq!(c.next_open(), Some(1));
    }

    #[test]
    fn the_page_never_sees_why_a_clip_was_picked() {
        let dir = corpus();
        let c = Channel::load(dir.path()).unwrap();
        let shown = format!(
            "{}{}{}",
            c.session(),
            c.item(1).unwrap(),
            c.item(2).unwrap()
        );
        assert!(!shown.contains("why"), "{shown}");
        assert!(!shown.contains("radio_tag"), "{shown}");
    }

    #[test]
    fn marks_need_a_play_persist_and_resume() {
        let dir = corpus();
        let mut c = Channel::load(dir.path()).unwrap();
        assert!(c.mark(body(1, "radio", false)).is_err());
        assert!(c.mark(body(1, "loud", true)).is_err());
        assert!(c.mark(body(3, "plain", true)).is_err());
        assert!(!dir.path().join(MARKS).exists());
        c.mark(body(1, "radio", true)).unwrap();
        c.mark(body(1, "other_effect", true)).unwrap();
        let c = Channel::load(dir.path()).unwrap();
        assert_eq!(c.marks["b"].channel, "other_effect");
        assert_eq!(c.marked(), 1);
        assert_eq!(c.next_open(), Some(2));
    }

    #[test]
    fn routes() {
        let dir = corpus();
        let mut c = Channel::load(dir.path()).unwrap();
        assert!(matches!(handle(&mut c, false, "/", ""), Reply::Page(_)));
        match handle(&mut c, false, "/audio?n=2", "") {
            Reply::Audio(p) => assert!(p.ends_with("audio/a.wav")),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            handle(&mut c, false, "/audio?n=3", ""),
            Reply::NotFound
        ));
        assert!(matches!(
            handle(&mut c, false, "/api/item?n=9", ""),
            Reply::NotFound
        ));
        let ok = handle(
            &mut c,
            true,
            "/api/mark",
            r#"{"n": 2, "channel": "plain", "heard": true}"#,
        );
        assert!(matches!(ok, Reply::Json(_)));
        assert!(matches!(
            handle(&mut c, true, "/api/mark", "{nope"),
            Reply::Bad(_)
        ));
    }

    #[test]
    fn without_a_sample_is_a_clear_error() {
        let dir = corpus();
        fs::remove_file(dir.path().join(SAMPLE)).unwrap();
        let err = Channel::load(dir.path()).err().unwrap().to_string();
        assert!(err.contains("sample_channel.py"), "{err}");
    }
}
