//! Reading the pipeline's output and recording review verdicts.
//!
//! Follows corpus format v1 (`docs/design/corpus-format.md`, "What the
//! review page reads"):
//!
//! ```text
//! <corpus>/stages/<stage>/current          key of the latest successful run
//! <corpus>/stages/<stage>/<key>/out/       that run's files
//!   harvest: lines.jsonl, audio/<id>.wav
//!   clean:   clips.jsonl, audio/<id>.flac   (optional)
//!   label:   labels.jsonl, axes.json
//! <corpus>/reviews.jsonl                   appended here; latest per id wins
//! ```
//!
//! Reviews never rewrite a stage's output. They are human input that later
//! stages (contrast sets, fusion recalibration) list as a file.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub const LINES: &str = "lines.jsonl";
pub const CLIPS: &str = "clips.jsonl";
pub const LABELS: &str = "labels.jsonl";
pub const AXES: &str = "axes.json";
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
    #[error("the {0} stage has not run yet (no stages/{0}/current)")]
    NotRun(&'static str),
}

/// One spoken line from harvest, with the game's own subtitle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub id: String,
    #[serde(default)]
    pub speaker: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub flags: Vec<String>,
    pub audio: Audio,
    #[serde(default)]
    pub source: Source,
    /// The lines just before this one, oldest first.
    #[serde(default)]
    pub context: Vec<ContextLine>,
    #[serde(default)]
    pub scene_tags: Vec<String>,
    /// LE3's voice director note, verbatim.
    #[serde(default)]
    pub stage_direction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audio {
    /// Relative to the `out/` folder of the stage that wrote the record.
    pub path: String,
    #[serde(default)]
    pub duration_s: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Source {
    #[serde(default)]
    pub game: Option<String>,
    #[serde(default)]
    pub conversation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextLine {
    pub speaker: String,
    pub text: String,
}

/// Clean's verdict on one clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub id: String,
    pub keep: bool,
    pub audio: Audio,
    /// What was actually spoken, when it differs from the subtitle.
    #[serde(default)]
    pub transcript: Option<String>,
    #[serde(default)]
    pub flags: Vec<String>,
}

/// The label stage's fused scores for one clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub id: String,
    /// Pack variable id (joy, intoxication, ...) to its score.
    pub scores: BTreeMap<String, AxisScore>,
    /// Non-emotion labels from discovery, like `combat_bark`.
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisScore {
    /// Fused score, 0 to 1.
    pub value: f32,
    /// How much the fusion trusts it, 0 to 1.
    pub confidence: f32,
    /// Each signal's own score (audio, text, prior, audio_llm); null when a
    /// signal didn't score this axis.
    #[serde(default)]
    pub signals: BTreeMap<String, Option<f32>>,
}

impl AxisScore {
    /// The signals that did score this axis.
    pub fn scored(&self) -> impl Iterator<Item = f32> + '_ {
        self.signals.values().filter_map(|v| *v)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Axis {
    pub id: String,
    #[serde(default)]
    pub description: String,
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
    /// Why a clip was flagged: `wrong_clip`, `music`, `other_voice`, ...
    #[serde(default)]
    pub flag: Option<String>,
    /// Time spent on the clip, for pacing the hour.
    #[serde(default)]
    pub seconds: f32,
    /// Unix time the verdict was recorded.
    #[serde(default)]
    pub at: u64,
}

pub struct Corpus {
    pub root: PathBuf,
    /// Lines with a playable clip; ones clean dropped are left out.
    pub lines: Vec<Line>,
    /// How many lines clean dropped.
    pub dropped: usize,
    pub labels: HashMap<String, Label>,
    /// Axes in the pack's order.
    pub axes: Vec<Axis>,
    /// Every review in file order; use [`Corpus::latest`] for current ones.
    pub reviews: Vec<Review>,
    harvest_out: PathBuf,
    clean: Option<(PathBuf, HashMap<String, Clip>)>,
}

/// A stage's current output folder, or None if it hasn't run.
pub fn current_out(corpus: &Path, stage: &str) -> Result<Option<PathBuf>, Error> {
    let stage_dir = corpus.join("stages").join(stage);
    let current = stage_dir.join("current");
    match fs::read_to_string(&current) {
        Ok(key) => Ok(Some(stage_dir.join(key.trim()).join("out"))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::Io {
            path: current,
            source,
        }),
    }
}

impl Corpus {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let harvest_out = current_out(&root, "harvest")?.ok_or(Error::NotRun("harvest"))?;
        let label_out = current_out(&root, "label")?.ok_or(Error::NotRun("label"))?;

        let clean = match current_out(&root, "clean")? {
            Some(out) => {
                let clips: HashMap<String, Clip> = read_jsonl::<Clip>(&out.join(CLIPS))?
                    .into_iter()
                    .map(|c| (c.id.clone(), c))
                    .collect();
                Some((out, clips))
            }
            None => None,
        };
        let mut lines: Vec<Line> = read_jsonl(&harvest_out.join(LINES))?;
        let before = lines.len();
        if let Some((_, clips)) = &clean {
            lines.retain(|l| clips.get(&l.id).is_some_and(|c| c.keep));
        }
        let dropped = before - lines.len();

        let labels = read_jsonl::<Label>(&label_out.join(LABELS))?
            .into_iter()
            .map(|l| (l.id.clone(), l))
            .collect();
        let axes_path = label_out.join(AXES);
        let axes_json = fs::read_to_string(&axes_path).map_err(|source| Error::Io {
            path: axes_path.clone(),
            source,
        })?;
        let axes = serde_json::from_str(&axes_json).map_err(|source| Error::Parse {
            path: axes_path,
            line: 1,
            source,
        })?;

        let reviews_path = root.join(REVIEWS);
        let reviews = if reviews_path.exists() {
            read_jsonl(&reviews_path)?
        } else {
            Vec::new()
        };
        Ok(Self {
            root,
            lines,
            dropped,
            labels,
            axes,
            reviews,
            harvest_out,
            clean,
        })
    }

    pub fn line(&self, id: &str) -> Option<&Line> {
        self.lines.iter().find(|l| l.id == id)
    }

    /// Clean's record for a line, if clean has run.
    pub fn clip(&self, id: &str) -> Option<&Clip> {
        self.clean.as_ref().and_then(|(_, clips)| clips.get(id))
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

    /// Where a line's clip lives: clean's FLAC if clean has run, otherwise
    /// harvest's WAV. Refuses paths that escape the stage folder.
    pub fn audio_path(&self, line: &Line) -> Option<PathBuf> {
        let (out, rel) = match &self.clean {
            Some((out, clips)) => (out, &clips.get(&line.id)?.audio.path),
            None => (&self.harvest_out, &line.audio.path),
        };
        let rel = Path::new(rel);
        let inside = rel
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        inside.then(|| out.join(rel))
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
