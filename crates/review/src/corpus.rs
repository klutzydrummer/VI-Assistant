//! Reading the label stage's output and recording review verdicts.
//!
//! PROVISIONAL: the corpus format is being designed separately
//! (`corpus-format.md`). Until it lands this module assumes the shape below,
//! and it is the only place that knows it, so adapting means editing this
//! file alone.
//!
//! ```text
//! <corpus>/lines.jsonl    one Line per clip (harvest + clean)
//! <corpus>/labels.jsonl   one Label per clip (label stage)
//! <corpus>/reviews.jsonl  appended by the review page; latest per id wins
//! <corpus>/audio/...      clips, referenced by Line::audio
//! ```
//!
//! Reviews never rewrite the label stage's output. They are a separate,
//! append-only file, so the stage runner can treat them as one more input
//! (to contrast sets and to recalibrating the fusion weights).

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub const LINES: &str = "lines.jsonl";
pub const LABELS: &str = "labels.jsonl";
pub const REVIEWS: &str = "reviews.jsonl";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}:{line}: {source}")]
    Parse {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },
}

/// One spoken line, with the clip and the game's own transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub id: String,
    /// Clip path relative to the corpus folder.
    pub audio: String,
    pub text: String,
    #[serde(default)]
    pub speaker: String,
    /// Game, conversation and anything else worth showing, e.g. "LE2".
    #[serde(default)]
    pub game: Option<String>,
    #[serde(default)]
    pub conversation: Option<String>,
    /// The lines just before this one, oldest first.
    #[serde(default)]
    pub context: Vec<ContextLine>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLine {
    pub speaker: String,
    pub text: String,
}

/// The label stage's fused scores for one clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub id: String,
    /// Pack variable id (joy, intoxication, ...) to its score.
    pub scores: BTreeMap<String, AxisScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisScore {
    /// Fused score, 0 to 1.
    pub value: f32,
    /// How much the fusion trusts it, 0 to 1.
    pub confidence: f32,
    /// Each signal's own score (audio, text, prior, audio_llm), when known.
    #[serde(default)]
    pub signals: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Scores are right as given, possibly after nudging.
    Keep,
    /// The clip itself is bad: wrong clip, music over it, not the character.
    Flag,
    /// Come back to it later.
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Review {
    pub id: String,
    pub verdict: Verdict,
    /// The reviewer's scores for every axis (Keep only).
    #[serde(default)]
    pub scores: BTreeMap<String, f32>,
    /// The label stage's scores as shown, so corrections can be measured
    /// even after the labels are regenerated.
    #[serde(default)]
    pub model: BTreeMap<String, f32>,
    /// Why a clip was flagged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Time spent on the clip, for pacing the hour.
    #[serde(default)]
    pub seconds: f32,
    /// Unix time the verdict was recorded.
    #[serde(default)]
    pub at: u64,
}

pub struct Corpus {
    pub root: PathBuf,
    pub lines: Vec<Line>,
    pub labels: HashMap<String, Label>,
    /// Every review in file order; use [`Corpus::latest`] for current ones.
    pub reviews: Vec<Review>,
}

impl Corpus {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let lines = read_jsonl(&root.join(LINES))?;
        let labels = read_jsonl::<Label>(&root.join(LABELS))?
            .into_iter()
            .map(|l| (l.id.clone(), l))
            .collect();
        let reviews_path = root.join(REVIEWS);
        let reviews = if reviews_path.exists() {
            read_jsonl(&reviews_path)?
        } else {
            Vec::new()
        };
        Ok(Self {
            root,
            lines,
            labels,
            reviews,
        })
    }

    pub fn line(&self, id: &str) -> Option<&Line> {
        self.lines.iter().find(|l| l.id == id)
    }

    /// Axis ids in the order the labels use them.
    pub fn axes(&self) -> Vec<String> {
        let mut axes: Vec<String> = Vec::new();
        for line in &self.lines {
            if let Some(label) = self.labels.get(&line.id) {
                for axis in label.scores.keys() {
                    if !axes.contains(axis) {
                        axes.push(axis.clone());
                    }
                }
            }
        }
        axes
    }

    /// The most recent review of each clip.
    pub fn latest(&self) -> HashMap<&str, &Review> {
        self.reviews.iter().map(|r| (r.id.as_str(), r)).collect()
    }

    /// Append a verdict to `reviews.jsonl` and keep it in memory.
    pub fn record(&mut self, review: Review) -> Result<(), Error> {
        let path = self.root.join(REVIEWS);
        let io = |source| Error::Io {
            path: path.clone(),
            source,
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(io)?;
        let mut json = serde_json::to_string(&review).expect("reviews serialize");
        json.push('\n');
        file.write_all(json.as_bytes()).map_err(io)?;
        file.sync_data().map_err(io)?;
        self.reviews.push(review);
        Ok(())
    }

    /// Where a line's clip lives, refusing paths that escape the corpus.
    pub fn audio_path(&self, line: &Line) -> Option<PathBuf> {
        let rel = Path::new(&line.audio);
        let inside = rel
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        inside.then(|| self.root.join(rel))
    }
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, Error> {
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).map_err(|source| Error::Parse {
            path: path.to_path_buf(),
            line: i + 1,
            source,
        })?);
    }
    Ok(out)
}

/// Write records as JSON lines, replacing the file.
pub fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> std::io::Result<()> {
    let mut out = String::new();
    for r in records {
        out.push_str(&serde_json::to_string(r).expect("records serialize"));
        out.push('\n');
    }
    fs::write(path, out)
}
