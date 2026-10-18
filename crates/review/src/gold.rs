//! Gold mode: Brandon labels a fixed sample of her lines from their words
//! and situation alone, for the bake-off that judges the text models (see
//! `docs/design/corpus-pipeline.md`, "Seeded moods", step 3).
//!
//! - The sample is `analysis/gold/gold_key.jsonl` (written by
//!   `analysis/gold/sample_gold.py`): which lines, in what order. Only the
//!   ids and the order are read; the model scores in it never reach the page,
//!   so nothing anchors him.
//! - Each line shows its scene, the lines before, hers, and the lines after,
//!   from the context stage, the same text the models read. No audio: the
//!   models judged from the words, so he does too.
//! - Per line: every mood that's really there (any number, or none), one level
//!   per knob, an unsure flag and a note. The moods and knobs are
//!   `moods.toml`'s.
//! - Every save is appended to `gold.jsonl` in the corpus folder; the last
//!   label of a line wins, so he can go back and change one, and a restart
//!   resumes at the first line not yet labelled.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::corpus::{current_out, Error};
use crate::server::Reply;

const PAGE: &str = include_str!("gold.html");

pub const GOLD: &str = "gold.jsonl";
pub const SAMPLE: &str = "analysis/gold/gold_key.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mood {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Knob {
    pub id: String,
    pub description: String,
    pub levels: Vec<Mood>,
}

#[derive(Debug, Deserialize)]
struct MoodsFile {
    #[serde(default)]
    mood: Vec<Mood>,
    #[serde(default)]
    knob: Vec<Knob>,
}

/// One line of `gold.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label {
    pub kind: String,
    pub id: String,
    pub moods: Vec<String>,
    pub knobs: BTreeMap<String, String>,
    #[serde(default)]
    pub unsure: bool,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Line {
    pub id: String,
    pub scene: String,
    pub before: Vec<Value>,
    pub text: String,
    pub after: Vec<Value>,
    pub stage_direction: Option<String>,
    pub listener: Option<String>,
}

pub struct Gold {
    pub root: PathBuf,
    pub moods: Vec<Mood>,
    pub knobs: Vec<Knob>,
    pub lines: Vec<Line>,
    pub labels: HashMap<String, Label>,
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

impl Gold {
    /// `root` is the corpus folder; `moods` the character's `moods.toml`.
    pub fn load(root: impl AsRef<Path>, moods: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let moods_path = moods.as_ref();
        let text = fs::read_to_string(moods_path).map_err(|source| Error::Io {
            path: moods_path.to_path_buf(),
            source,
        })?;
        let file: MoodsFile = toml::from_str(&text)
            .map_err(|e| missing(moods_path, &format!("not a moods file: {e}")))?;

        let out = current_out(&root, "context")?.ok_or(Error::NotRun("context"))?;
        let mut context: HashMap<String, Value> = HashMap::new();
        for r in read_values(&out.join("context.jsonl"))? {
            if let Some(id) = r["id"].as_str() {
                context.insert(id.to_string(), r);
            }
        }

        let sample = root.join(SAMPLE);
        if !sample.exists() {
            return Err(missing(
                &sample,
                "no gold sample: run analysis/gold/sample_gold.py",
            ));
        }
        let mut picks: Vec<(u64, String)> = read_values(&sample)?
            .into_iter()
            .filter_map(|r| Some((r["n"].as_u64()?, r["id"].as_str()?.to_string())))
            .collect();
        picks.sort();
        let mut lines = Vec::new();
        for (_, id) in picks {
            let Some(c) = context.get(&id) else { continue };
            let prompt = c["prompt"].as_str().unwrap_or("");
            let scene = prompt
                .lines()
                .next()
                .and_then(|l| l.strip_prefix("Scene: "))
                .unwrap_or("")
                .to_string();
            lines.push(Line {
                id: id.clone(),
                scene,
                before: c["before"].as_array().cloned().unwrap_or_default(),
                text: c["text"].as_str().unwrap_or("").to_string(),
                after: c["after"].as_array().cloned().unwrap_or_default(),
                stage_direction: c["stage_direction"].as_str().map(str::to_string),
                listener: c["listener"].as_str().map(str::to_string),
            });
        }

        let mut labels = HashMap::new();
        let path = root.join(GOLD);
        if path.exists() {
            for v in read_values(&path)? {
                if let Ok(l) = serde_json::from_value::<Label>(v) {
                    labels.insert(l.id.clone(), l);
                }
            }
        }
        Ok(Gold {
            root,
            moods: file.mood,
            knobs: file.knob,
            lines,
            labels,
        })
    }

    pub fn labelled(&self) -> usize {
        self.lines
            .iter()
            .filter(|l| self.labels.contains_key(&l.id))
            .count()
    }

    /// 1-based position of the first line not yet labelled.
    pub fn next_open(&self) -> Option<usize> {
        self.lines
            .iter()
            .position(|l| !self.labels.contains_key(&l.id))
            .map(|i| i + 1)
    }

    pub fn session(&self) -> Value {
        json!({
            "moods": self.moods,
            "knobs": self.knobs,
            "count": self.lines.len(),
            "labelled": self.labelled(),
            "next": self.next_open(),
            "done": self.lines.iter().map(|l| self.labels.contains_key(&l.id)).collect::<Vec<_>>(),
        })
    }

    pub fn item(&self, n: usize) -> Option<Value> {
        let line = self.lines.get(n.checked_sub(1)?)?;
        Some(
            json!({"n": n, "count": self.lines.len(), "line": line, "label": self.labels.get(&line.id)}),
        )
    }

    pub fn label(&mut self, body: LabelBody) -> Result<(), String> {
        let line = body
            .n
            .checked_sub(1)
            .and_then(|i| self.lines.get(i))
            .ok_or_else(|| format!("no line {}", body.n))?;
        let known: HashSet<&str> = self.moods.iter().map(|m| m.id.as_str()).collect();
        let mut moods = Vec::new();
        for m in body.moods {
            if !known.contains(m.as_str()) {
                return Err(format!("no mood {m}"));
            }
            if !moods.contains(&m) {
                moods.push(m);
            }
        }
        let mut knobs = BTreeMap::new();
        for (k, v) in body.knobs {
            let knob = self
                .knobs
                .iter()
                .find(|x| x.id == k)
                .ok_or_else(|| format!("no knob {k}"))?;
            if !knob.levels.iter().any(|lv| lv.id == v) {
                return Err(format!("{k} has no level {v}"));
            }
            knobs.insert(k, v);
        }
        let label = Label {
            kind: "label".into(),
            id: line.id.clone(),
            moods,
            knobs,
            unsure: body.unsure,
            notes: body.notes.trim().to_string(),
            seconds: body.seconds.max(0.0),
            at: now(),
        };
        let path = self.root.join(GOLD);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut text = serde_json::to_string(&label).expect("labels serialize");
        text.push('\n');
        file.write_all(text.as_bytes())
            .and_then(|()| file.sync_data())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.labels.insert(label.id.clone(), label);
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct LabelBody {
    pub n: usize,
    #[serde(default)]
    pub moods: Vec<String>,
    #[serde(default)]
    pub knobs: BTreeMap<String, String>,
    #[serde(default)]
    pub unsure: bool,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub seconds: f64,
}

pub fn handle(g: &mut Gold, post: bool, url: &str, body: &str) -> Reply {
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    match (post, path) {
        (false, "/") => Reply::Page(PAGE),
        (false, "/api/session") => Reply::Json(g.session()),
        (false, "/api/item") => {
            let n = query
                .split('&')
                .find_map(|kv| kv.strip_prefix("n="))
                .and_then(|v| v.parse().ok());
            match n.and_then(|n| g.item(n)) {
                Some(v) => Reply::Json(v),
                None => Reply::NotFound,
            }
        }
        (true, "/api/label") => {
            let parsed: Result<LabelBody, _> = serde_json::from_str(body);
            match parsed.map_err(|e| e.to_string()).and_then(|b| g.label(b)) {
                Ok(()) => Reply::Json(g.session()),
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

    const MOODS: &str = r#"
[[mood]]
id = "warm"
description = "warm and friendly"

[[mood]]
id = "teasing"
description = "teasing"

[[knob]]
id = "drunk"
description = "how drunk she is"
levels = [{ id = "sober", description = "sober" }, { id = "drunk", description = "drunk" }]
"#;

    /// A corpus with a context stage, a sample of three lines (one missing
    /// from the context, so skipped) and the moods file.
    fn corpus() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let out = root.join("stages/context/abc/out");
        fs::create_dir_all(&out).unwrap();
        fs::write(root.join("stages/context/current"), "abc\n").unwrap();
        let ctx = [
            json!({"id": "a", "text": "Thank you.", "prompt": "Scene: normandy\n>>> Tali: Thank you.",
                   "before": [{"speaker": "Shepard", "text": "Here."}], "after": []}),
            json!({"id": "b", "text": "Hah!", "prompt": "Scene: citadel party\n>>> Tali: Hah!", "before": [], "after": [],
                   "stage_direction": "drunk"}),
        ];
        fs::write(
            out.join("context.jsonl"),
            ctx.iter().map(|v| v.to_string() + "\n").collect::<String>(),
        )
        .unwrap();
        fs::create_dir_all(root.join("analysis/gold")).unwrap();
        let sample = [
            json!({"n": 2, "id": "b", "scores": {"kev": {"warm": 0.9}}}),
            json!({"n": 1, "id": "a"}),
            json!({"n": 3, "id": "gone"}),
        ];
        fs::write(
            root.join(SAMPLE),
            sample
                .iter()
                .map(|v| v.to_string() + "\n")
                .collect::<String>(),
        )
        .unwrap();
        let moods = root.join("moods.toml");
        fs::write(&moods, MOODS).unwrap();
        (dir, moods)
    }

    fn body(n: usize, moods: &[&str], knobs: &[(&str, &str)]) -> LabelBody {
        LabelBody {
            n,
            moods: moods.iter().map(|s| s.to_string()).collect(),
            knobs: knobs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            unsure: false,
            notes: String::new(),
            seconds: 3.0,
        }
    }

    #[test]
    fn loads_the_sample_in_order_with_its_context() {
        let (dir, moods) = corpus();
        let g = Gold::load(dir.path(), &moods).unwrap();
        assert_eq!(
            g.lines.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(g.lines[0].scene, "normandy");
        assert_eq!(g.lines[1].stage_direction.as_deref(), Some("drunk"));
        assert_eq!(g.next_open(), Some(1));
        assert_eq!(g.knobs[0].levels.len(), 2);
    }

    #[test]
    fn the_page_never_sees_model_scores() {
        let (dir, moods) = corpus();
        let g = Gold::load(dir.path(), &moods).unwrap();
        let shown = format!(
            "{}{}{}",
            g.session(),
            g.item(1).unwrap(),
            g.item(2).unwrap()
        );
        assert!(!shown.contains("scores"), "{shown}");
        assert!(!shown.contains("0.9"), "{shown}");
    }

    #[test]
    fn labels_persist_the_last_one_wins_and_it_resumes() {
        let (dir, moods) = corpus();
        let mut g = Gold::load(dir.path(), &moods).unwrap();
        g.label(body(1, &["warm"], &[("drunk", "sober")])).unwrap();
        g.label(body(1, &["warm", "teasing", "warm"], &[])).unwrap();
        assert_eq!(g.next_open(), Some(2));
        let g = Gold::load(dir.path(), &moods).unwrap();
        let l = &g.labels["a"];
        assert_eq!(l.moods, ["warm", "teasing"]);
        assert!(l.knobs.is_empty());
        assert_eq!(g.labelled(), 1);
        assert_eq!(g.next_open(), Some(2));
        assert_eq!(
            fs::read_to_string(dir.path().join(GOLD))
                .unwrap()
                .lines()
                .count(),
            2
        );
    }

    #[test]
    fn unknown_moods_levels_and_lines_are_refused() {
        let (dir, moods) = corpus();
        let mut g = Gold::load(dir.path(), &moods).unwrap();
        assert!(g.label(body(1, &["grumpy"], &[])).is_err());
        assert!(g.label(body(1, &[], &[("drunk", "wasted")])).is_err());
        assert!(g.label(body(1, &[], &[("energy", "calm")])).is_err());
        assert!(g.label(body(3, &[], &[])).is_err());
        assert!(g.label(body(0, &[], &[])).is_err());
        assert!(!dir.path().join(GOLD).exists());
    }

    #[test]
    fn routes() {
        let (dir, moods) = corpus();
        let mut g = Gold::load(dir.path(), &moods).unwrap();
        assert!(matches!(handle(&mut g, false, "/", ""), Reply::Page(_)));
        assert!(matches!(
            handle(&mut g, false, "/api/item?n=2", ""),
            Reply::Json(_)
        ));
        assert!(matches!(
            handle(&mut g, false, "/api/item?n=9", ""),
            Reply::NotFound
        ));
        let ok = handle(
            &mut g,
            true,
            "/api/label",
            r#"{"n": 2, "moods": ["teasing"], "knobs": {"drunk": "drunk"}}"#,
        );
        assert!(matches!(ok, Reply::Json(_)));
        assert!(matches!(
            handle(&mut g, true, "/api/label", "{nope"),
            Reply::Bad(_)
        ));
    }

    #[test]
    fn without_a_sample_is_a_clear_error() {
        let (dir, moods) = corpus();
        fs::remove_file(dir.path().join(SAMPLE)).unwrap();
        let err = Gold::load(dir.path(), &moods).err().unwrap().to_string();
        assert!(err.contains("sample_gold.py"), "{err}");
    }
}
