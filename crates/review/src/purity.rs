//! Purity mode: Brandon listens to clips from one cluster at a time and says
//! whether each fits, then names, merges, splits or rejects the cluster.
//! See `docs/design/corpus-pipeline.md`, step 6.
//!
//! - He picks which clusters to listen to; with several picked, clips
//!   rotate between them (the one with fewest answers next), each judged
//!   against its own cluster. With none picked, clusters go one at a time.
//! - Any clip can be skipped. It goes back to the end of its queue and comes
//!   up again later; skips are recorded, and counted per cluster, because
//!   skipping only the hard clips would inflate purity.
//! - Each cluster opens with up to five reference clips, its most typical
//!   members from different conversations, so there is something to judge
//!   the first clip against. They (and their other takes) are never asked.
//! - Members come stratified across scenes and conversations, so a cluster
//!   isn't judged on one scene.
//! - Every fifth clip is a foil: a clip from another cluster at a similar
//!   length. His "fits" on foils measures his false-yes rate, which corrects
//!   the purity estimate.
//! - Every eighth clip is a hidden repeat of one he already answered, up to
//!   50, for his consistency (Cohen's kappa).
//! - A cluster stops when Wald's SPRT decides between purity 0.5 and 0.8
//!   (alpha = beta = 0.05), or when its members run out.
//! - Nothing about the clip (line, context, tags, prosody, EIV heads) shows
//!   until he has answered, so it can't anchor him.
//!
//! The page never learns which clips are foils or repeats: the server works
//! out the next clip from `purity.jsonl` alone, so a restart resumes exactly.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::corpus::{current_out, Error};
use crate::server::{decode, Reply};

const PAGE: &str = include_str!("purity.html");

pub const PURITY: &str = "purity.jsonl";
const P0: f64 = 0.5;
const P1: f64 = 0.8;
const ALPHA: f64 = 0.05;
const BETA: f64 = 0.05;
/// One foil per four members.
pub const FOIL_EVERY: usize = 5;
/// About one hidden repeat in eight clips...
pub const REPEAT_EVERY: usize = 8;
/// ...until there are this many.
pub const REPEAT_TARGET: usize = 50;
/// A cluster the SPRT hasn't decided after this many members is decided by
/// hand anyway.
pub const MAX_MEMBERS: usize = 40;
/// Reference clips per cluster, at most a quarter of its members.
pub const REFERENCES: usize = 5;
/// A foil must belong to the cluster with at most this responsibility.
const FOIL_MAX_RESPONSIBILITY: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Name,
    Merge,
    Split,
    Reject,
}

/// One line of `purity.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    Answer {
        id: String,
        cluster: String,
        fits: bool,
        foil: bool,
        repeat: bool,
        #[serde(default)]
        seconds: f64,
        #[serde(default)]
        at: u64,
    },
    Decision {
        cluster: String,
        action: Action,
        #[serde(default)]
        name: Option<String>,
        /// The cluster it merges into.
        #[serde(default)]
        into: Option<String>,
        #[serde(default)]
        note: Option<String>,
        #[serde(default)]
        at: u64,
    },
    /// A clip he skipped; it goes to the back of its queue.
    Skip {
        id: String,
        cluster: String,
        #[serde(default)]
        at: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Member,
    Foil,
    Repeat,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Item {
    pub id: String,
    #[serde(skip)]
    pub kind: Kind,
}

/// Where a cluster's listening stands.
#[derive(Debug, Clone, Serialize)]
pub struct ClassState {
    pub cluster: String,
    pub answers: usize,
    pub members_yes: usize,
    pub members_no: usize,
    pub foils: usize,
    pub skips: usize,
    /// The SPRT's log likelihood ratio for purity 0.8 over 0.5.
    pub llr: f64,
    /// `pure`, `impure`, `undecided` (members ran out), or null (listening).
    pub sprt: Option<&'static str>,
    pub observed: Option<f64>,
    /// Corrected for the false-yes rate on foils (all clusters so far).
    pub purity: Option<f64>,
    pub decision: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct Membership {
    id: String,
    cluster: String,
    #[serde(default)]
    responsibilities: BTreeMap<String, f64>,
    #[serde(default)]
    formed: bool,
}

pub struct Purity {
    root: PathBuf,
    /// Cluster ids in review order, with cluster.json's entry for each.
    pub classes: Vec<(String, Value)>,
    /// Per cluster: its reference clips, members in listening order, and
    /// foils in order.
    pub references: HashMap<String, Vec<String>>,
    order: HashMap<String, Vec<String>>,
    foils: HashMap<String, Vec<String>>,
    lines: HashMap<String, Value>,
    audio: HashMap<String, PathBuf>,
    eiv: HashMap<String, BTreeMap<String, f64>>,
    eiv_stats: BTreeMap<String, (f64, f64)>,
    prosody: HashMap<String, Value>,
    pub records: Vec<Record>,
    /// The clusters he picked to listen to; empty means one at a time, in
    /// order. Not saved: it's how he listens, not what he heard.
    pub selected: Vec<String>,
}

fn sha(id: &str) -> String {
    Sha256::digest(id.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
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

fn inside(out: &Path, rel: &str) -> Option<PathBuf> {
    let rel = Path::new(rel);
    rel.components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        .then(|| out.join(rel))
}

/// Round-robin over groups, each group's items in their own order.
/// The line both takes of a clip share: `le2.s348112_f` -> `le2.s348112`.
fn take_base(id: &str) -> &str {
    id.rsplit_once('_').map_or(id, |(base, _)| base)
}

fn interleave(groups: Vec<Vec<String>>) -> Vec<String> {
    let mut out = Vec::new();
    let longest = groups.iter().map(Vec::len).max().unwrap_or(0);
    for i in 0..longest {
        for g in &groups {
            if let Some(id) = g.get(i) {
                out.push(id.clone());
            }
        }
    }
    out
}

/// Tags that say where she is (not the structural ones), for stratifying.
fn scene_of(line: &Value) -> String {
    const STRUCTURAL: [&str; 6] = [
        "one_liner",
        "exchange",
        "dialogue_choices",
        "to_shepard",
        "to_squad",
        "to_npc",
    ];
    line["scene_tags"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|t| !STRUCTURAL.contains(t))
        .unwrap_or("none")
        .to_string()
}

impl Purity {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref().to_path_buf();
        let harvest = current_out(&root, "harvest")?.ok_or(Error::NotRun("harvest"))?;
        let cluster = current_out(&root, "cluster")?.ok_or(Error::NotRun("cluster"))?;
        let lines: HashMap<String, Value> = read_values(&harvest.join("lines.jsonl"))?
            .into_iter()
            .filter_map(|v| Some((v["id"].as_str()?.to_string(), v)))
            .collect();

        let mut audio = HashMap::new();
        let clean = current_out(&root, "clean")?;
        for (id, line) in &lines {
            let path = match &clean {
                Some(_) => None,
                None => line["audio"]["path"]
                    .as_str()
                    .and_then(|rel| inside(&harvest, rel)),
            };
            if let Some(p) = path {
                audio.insert(id.clone(), p);
            }
        }
        if let Some(out) = &clean {
            for c in read_values(&out.join("clips.jsonl"))? {
                if let (Some(id), Some(rel)) = (c["id"].as_str(), c["audio"]["path"].as_str()) {
                    if let Some(p) = inside(out, rel) {
                        audio.insert(id.to_string(), p);
                    }
                }
            }
        }

        let report: Value = {
            let path = cluster.join("cluster.json");
            let text = fs::read_to_string(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            serde_json::from_str(&text).map_err(|source| Error::Parse {
                path,
                line: 1,
                source,
            })?
        };
        let classes: Vec<(String, Value)> = report["classes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| Some((c["class"].as_str()?.to_string(), c.clone())))
            .collect();
        let memberships: Vec<Membership> = read_values(&cluster.join("clusters.jsonl"))?
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();

        let duration = |id: &str| -> f64 {
            lines
                .get(id)
                .and_then(|l| l["audio"]["duration_s"].as_f64())
                .unwrap_or(0.0)
        };
        let playable: Vec<&Membership> = memberships
            .iter()
            .filter(|m| m.formed && audio.contains_key(&m.id) && lines.contains_key(&m.id))
            .collect();
        let conversation = |id: &str| -> String {
            lines[id]["source"]["conversation"]
                .as_str()
                .unwrap_or(id)
                .to_string()
        };
        let mut references = HashMap::new();
        let mut order = HashMap::new();
        let mut foils = HashMap::new();
        for (class, _) in &classes {
            // References: the members most surely in the cluster, one per
            // conversation while that lasts.
            let mut typical: Vec<&Membership> = playable
                .iter()
                .copied()
                .filter(|m| &m.cluster == class)
                .collect();
            let own = |m: &Membership| m.responsibilities.get(class).copied().unwrap_or(0.0);
            typical.sort_by(|a, b| {
                own(b)
                    .total_cmp(&own(a))
                    .then_with(|| sha(&a.id).cmp(&sha(&b.id)))
            });
            let want = REFERENCES.min(typical.len() / 4);
            let mut refs: Vec<String> = Vec::new();
            let mut convs = HashSet::new();
            for m in &typical {
                if refs.len() < want && convs.insert(conversation(&m.id)) {
                    refs.push(m.id.clone());
                }
            }
            for m in &typical {
                if refs.len() < want && !refs.contains(&m.id) {
                    refs.push(m.id.clone());
                }
            }
            let heard: HashSet<&str> = refs.iter().map(|id| take_base(id)).collect();

            // Members: by scene, then by conversation, each in hash order.
            let mut by_scene: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
            for m in playable
                .iter()
                .filter(|m| &m.cluster == class && !heard.contains(take_base(&m.id)))
            {
                let line = &lines[&m.id];
                let conv = conversation(&m.id);
                by_scene
                    .entry(scene_of(line))
                    .or_default()
                    .entry(conv)
                    .or_default()
                    .push(m.id.clone());
            }
            let scenes = by_scene
                .into_values()
                .map(|convs| {
                    let mut convs: Vec<Vec<String>> = convs.into_values().collect();
                    for c in &mut convs {
                        c.sort_by_key(|id| sha(id));
                    }
                    convs.sort_by_key(|c| sha(&c[0]));
                    interleave(convs)
                })
                .collect();
            let members = interleave(scenes);

            // Foils: for every fourth member, the unused clip from another
            // cluster nearest its length.
            let mut pool: Vec<&Membership> = playable
                .iter()
                .copied()
                .filter(|m| {
                    &m.cluster != class
                        && m.responsibilities.get(class).copied().unwrap_or(0.0)
                            <= FOIL_MAX_RESPONSIBILITY
                })
                .collect();
            pool.sort_by_key(|m| sha(&m.id));
            let mut chosen = Vec::new();
            for target in members.iter().step_by(FOIL_EVERY - 1) {
                let want = duration(target);
                let best = pool.iter().enumerate().min_by(|a, b| {
                    (duration(&a.1.id) - want)
                        .abs()
                        .total_cmp(&(duration(&b.1.id) - want).abs())
                });
                if let Some((i, _)) = best {
                    chosen.push(pool.remove(i).id.clone());
                }
            }
            order.insert(class.clone(), members);
            foils.insert(class.clone(), chosen);
            references.insert(class.clone(), refs);
        }

        let embed = current_out(&root, "embed")?;
        let mut eiv = HashMap::new();
        let mut prosody = HashMap::new();
        if let Some(out) = &embed {
            if out.join("scores.jsonl").exists() {
                for r in read_values(&out.join("scores.jsonl"))? {
                    let heads: BTreeMap<String, f64> =
                        serde_json::from_value(r["eiv"].clone()).unwrap_or_default();
                    if let Some(id) = r["id"].as_str() {
                        eiv.insert(id.to_string(), heads);
                    }
                }
            }
            if out.join("prosody.jsonl").exists() {
                for r in read_values(&out.join("prosody.jsonl"))? {
                    if let Some(id) = r["id"].as_str() {
                        prosody.insert(id.to_string(), r.clone());
                    }
                }
            }
        }
        let mut sums: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();
        for heads in eiv.values() {
            for (h, v) in heads {
                let e = sums.entry(h.clone()).or_default();
                e.0 += v;
                e.1 += v * v;
                e.2 += 1.0;
            }
        }
        let eiv_stats = sums
            .into_iter()
            .map(|(h, (s, ss, n))| {
                let mean = s / n;
                (h, (mean, (ss / n - mean * mean).max(1e-12).sqrt()))
            })
            .collect();

        let path = root.join(PURITY);
        let records = if path.exists() {
            read_values(&path)?
                .into_iter()
                .filter_map(|v| serde_json::from_value(v).ok())
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            root,
            classes,
            references,
            order,
            foils,
            lines,
            audio,
            eiv,
            eiv_stats,
            prosody,
            records,
            selected: Vec::new(),
        })
    }

    pub fn audio_path(&self, id: &str) -> Option<&PathBuf> {
        self.audio.get(id)
    }

    fn answers(&self, class: &str) -> Vec<&Record> {
        self.records
            .iter()
            .filter(|r| matches!(r, Record::Answer { cluster, .. } if cluster == class))
            .collect()
    }

    fn decision(&self, class: &str) -> Option<&Record> {
        self.records
            .iter()
            .rev()
            .find(|r| matches!(r, Record::Decision { cluster, .. } if cluster == class))
    }

    /// Where each clip skipped in a cluster was last skipped (its record
    /// index), so skipped clips go to the back of the queue, oldest skip
    /// first.
    fn skipped(&self, class: &str) -> HashMap<&str, usize> {
        self.records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r {
                Record::Skip { id, cluster, .. } if cluster == class => Some((id.as_str(), i)),
                _ => None,
            })
            .collect()
    }

    /// The cluster being reviewed. With clusters picked: one that is ready
    /// to decide, else the one with fewest answers (ties in review order).
    /// With none picked, or all of them decided: the first without a
    /// decision.
    pub fn current(&self) -> Option<&str> {
        let undecided = || {
            self.classes
                .iter()
                .map(|(c, _)| c.as_str())
                .filter(|c| self.decision(c).is_none())
        };
        let pool: Vec<&str> = undecided()
            .filter(|c| self.selected.iter().any(|s| s == c))
            .collect();
        if pool.is_empty() {
            return undecided().next();
        }
        pool.iter()
            .find(|c| self.next(c).is_none())
            .or_else(|| pool.iter().min_by_key(|c| self.answers(c).len()))
            .copied()
    }

    /// Pick the clusters to listen to (empty: one at a time, in order).
    pub fn select(&mut self, clusters: Vec<String>) -> Result<(), String> {
        if let Some(bad) = clusters
            .iter()
            .find(|c| !self.classes.iter().any(|(id, _)| id == *c))
        {
            return Err(format!("no cluster {bad}"));
        }
        self.selected = clusters;
        Ok(())
    }

    pub fn decided(&self) -> usize {
        self.classes
            .iter()
            .filter(|(c, _)| self.decision(c).is_some())
            .count()
    }

    /// His false-yes rate on foils so far, over every cluster.
    pub fn false_yes(&self) -> Option<f64> {
        let (yes, n) = self.records.iter().fold((0, 0), |(y, n), r| match r {
            Record::Answer {
                foil: true, fits, ..
            } => (y + usize::from(*fits), n + 1),
            _ => (y, n),
        });
        (n > 0).then(|| yes as f64 / n as f64)
    }

    fn repeats(&self) -> usize {
        self.records
            .iter()
            .filter(|r| matches!(r, Record::Answer { repeat: true, .. }))
            .count()
    }

    pub fn state(&self, class: &str) -> ClassState {
        let (a, b) = (((1.0 - BETA) / ALPHA).ln(), (BETA / (1.0 - ALPHA)).ln());
        let mut s = ClassState {
            cluster: class.to_string(),
            answers: 0,
            members_yes: 0,
            members_no: 0,
            foils: 0,
            skips: self
                .records
                .iter()
                .filter(|r| matches!(r, Record::Skip { cluster, .. } if cluster == class))
                .count(),
            llr: 0.0,
            sprt: None,
            observed: None,
            purity: None,
            decision: self.decision(class).map(|d| json!(d)),
        };
        for r in self.answers(class) {
            let Record::Answer {
                fits, foil, repeat, ..
            } = r
            else {
                continue;
            };
            s.answers += 1;
            if *foil {
                s.foils += 1;
            } else if !*repeat {
                if *fits {
                    s.members_yes += 1;
                    s.llr += (P1 / P0).ln();
                } else {
                    s.members_no += 1;
                    s.llr += ((1.0 - P1) / (1.0 - P0)).ln();
                }
                if s.sprt.is_none() {
                    if s.llr >= a {
                        s.sprt = Some("pure");
                    } else if s.llr <= b {
                        s.sprt = Some("impure");
                    }
                }
            }
        }
        let members = s.members_yes + s.members_no;
        let available = self.order.get(class).map_or(0, Vec::len);
        if s.sprt.is_none() && members > 0 && (members >= available || members >= MAX_MEMBERS) {
            s.sprt = Some("undecided");
        }
        if members > 0 {
            let y = s.members_yes as f64 / members as f64;
            s.observed = Some(y);
            s.purity = Some(match self.false_yes() {
                Some(f) if f < 1.0 => ((y - f) / (1.0 - f)).clamp(0.0, 1.0),
                _ => y,
            });
        }
        s
    }

    /// The next clip to play for a cluster, or None when it's ready to decide.
    pub fn next(&self, class: &str) -> Option<Item> {
        let state = self.state(class);
        if state.sprt.is_some() || state.decision.is_some() {
            return None;
        }
        let answers = self.answers(class);
        let answered: HashSet<&str> = answers
            .iter()
            .filter_map(|r| match r {
                Record::Answer {
                    id,
                    foil: false,
                    repeat: false,
                    ..
                } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let used_foils: HashSet<&str> = answers
            .iter()
            .filter_map(|r| match r {
                Record::Answer { id, foil: true, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let repeated: HashSet<&str> = self
            .records
            .iter()
            .filter_map(|r| match r {
                Record::Answer {
                    id, repeat: true, ..
                } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let skipped = self.skipped(class);
        // The first candidate never skipped, else the one skipped longest ago.
        let pick = |ids: &mut dyn Iterator<Item = &String>| -> Option<String> {
            ids.enumerate()
                .min_by_key(|(i, id)| (skipped.get(id.as_str()).copied(), *i))
                .map(|(_, id)| id.clone())
        };
        let slot = answers.len() + 1;
        let member = || {
            pick(
                &mut self.order[class]
                    .iter()
                    .filter(|id| !answered.contains(id.as_str())),
            )
            .map(|id| Item {
                id,
                kind: Kind::Member,
            })
        };
        if slot.is_multiple_of(REPEAT_EVERY) && self.repeats() < REPEAT_TARGET {
            let again = pick(&mut answers.iter().filter_map(|r| match r {
                Record::Answer {
                    id,
                    foil: false,
                    repeat: false,
                    ..
                } if !repeated.contains(id.as_str()) => Some(id),
                _ => None,
            }));
            if let Some(id) = again {
                return Some(Item {
                    id,
                    kind: Kind::Repeat,
                });
            }
        }
        if slot.is_multiple_of(FOIL_EVERY) {
            let foil = pick(
                &mut self.foils[class]
                    .iter()
                    .filter(|id| !used_foils.contains(id.as_str())),
            );
            if let Some(id) = foil {
                return Some(Item {
                    id,
                    kind: Kind::Foil,
                });
            }
        }
        member()
    }

    /// The clip waiting to be played, if any.
    pub fn waiting(&self) -> Option<Item> {
        self.next(self.current()?)
    }

    /// Record an answer to the clip waiting, and return what can now be shown
    /// about it. `n` is the record count the page saw, so a double click or a
    /// stale tab can't answer the wrong clip.
    pub fn answer(&mut self, n: usize, fits: bool, seconds: f64) -> Result<Value, String> {
        if n != self.records.len() {
            return Err("the page is out of date; reload it".into());
        }
        let class = self
            .current()
            .ok_or("every cluster is decided")?
            .to_string();
        let item = self
            .next(&class)
            .ok_or("this cluster is ready to decide; no clip is waiting")?;
        self.append(Record::Answer {
            id: item.id.clone(),
            cluster: class,
            fits,
            foil: item.kind == Kind::Foil,
            repeat: item.kind == Kind::Repeat,
            seconds: seconds.clamp(0.0, 300.0),
            at: now(),
        })?;
        Ok(self.reveal(&item.id))
    }

    /// Skip the clip waiting: it goes to the back of its queue. Recorded,
    /// so a restart resumes exactly and the skips can be counted. `n` as in
    /// `answer`.
    pub fn skip(&mut self, n: usize) -> Result<(), String> {
        if n != self.records.len() {
            return Err("the page is out of date; reload it".into());
        }
        let class = self
            .current()
            .ok_or("every cluster is decided")?
            .to_string();
        let item = self
            .next(&class)
            .ok_or("this cluster is ready to decide; no clip is waiting")?;
        self.append(Record::Skip {
            id: item.id,
            cluster: class,
            at: now(),
        })
    }

    pub fn decide(
        &mut self,
        class: &str,
        action: Action,
        name: Option<String>,
        into: Option<String>,
        note: Option<String>,
    ) -> Result<(), String> {
        if self.current() != Some(class) {
            return Err(format!("{class} is not the cluster under review"));
        }
        let name = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
        match action {
            Action::Name if name.is_none() => return Err("a name is needed".into()),
            Action::Merge
                if !into
                    .as_deref()
                    .is_some_and(|i| i != class && self.classes.iter().any(|(c, _)| c == i)) =>
            {
                return Err("merge needs another cluster to merge into".into())
            }
            _ => {}
        }
        self.append(Record::Decision {
            cluster: class.to_string(),
            action,
            name,
            into,
            note,
            at: now(),
        })
    }

    fn append(&mut self, record: Record) -> Result<(), String> {
        let path = self.root.join(PURITY);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut line = serde_json::to_string(&record).expect("records serialize");
        line.push('\n');
        file.write_all(line.as_bytes())
            .and_then(|()| file.sync_data())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.records.push(record);
        Ok(())
    }

    /// Everything about a clip, shown only after he has answered. Not
    /// whether it was a foil or a repeat: feedback on foils would train his
    /// answers and bias the false-yes rate they measure.
    pub fn reveal(&self, id: &str) -> Value {
        let line = self.lines.get(id).cloned().unwrap_or(Value::Null);
        let mut heads: Vec<(String, f64)> = self
            .eiv
            .get(id)
            .into_iter()
            .flatten()
            .filter_map(|(h, v)| {
                let (mean, sd) = self.eiv_stats.get(h)?;
                Some((h.clone(), (v - mean) / sd))
            })
            .collect();
        heads.sort_by(|a, b| b.1.total_cmp(&a.1));
        heads.truncate(3);
        json!({
            "id": id,
            "text": line["text"],
            "game": line["source"]["game"],
            "conversation": line["source"]["conversation"],
            "context": line["context"],
            "following": line["following"],
            "scene_tags": line["scene_tags"],
            "stage_direction": line["stage_direction"],
            "prosody": self.prosody.get(id),
            "eiv_top": heads.into_iter().map(|(h, z)| json!({"head": h, "z": (z * 100.0).round() / 100.0})).collect::<Vec<_>>(),
        })
    }

    /// His consistency on hidden repeats: Cohen's kappa of first answer
    /// against repeat, and how many pairs it rests on.
    pub fn kappa(&self) -> (Option<f64>, usize) {
        let mut first: HashMap<&str, bool> = HashMap::new();
        let mut pairs = Vec::new();
        for r in &self.records {
            if let Record::Answer {
                id,
                fits,
                foil: false,
                repeat,
                ..
            } = r
            {
                if *repeat {
                    if let Some(&a) = first.get(id.as_str()) {
                        pairs.push((a, *fits));
                    }
                } else {
                    first.entry(id.as_str()).or_insert(*fits);
                }
            }
        }
        let n = pairs.len() as f64;
        if pairs.is_empty() {
            return (None, 0);
        }
        let agree = pairs.iter().filter(|(a, b)| a == b).count() as f64 / n;
        let pa = pairs.iter().filter(|(a, _)| *a).count() as f64 / n;
        let pb = pairs.iter().filter(|(_, b)| *b).count() as f64 / n;
        let chance = pa * pb + (1.0 - pa) * (1.0 - pb);
        let kappa = if chance < 1.0 {
            Some((agree - chance) / (1.0 - chance))
        } else {
            None
        };
        (kappa, pairs.len())
    }

    /// The page's view: the cluster under review, whether a clip is waiting
    /// (not which: its id names the game), and every cluster's progress.
    /// A cluster's audit numbers show only once its listening is done, so
    /// they can't anchor the answers either.
    pub fn session(&self) -> Value {
        let current = self.current();
        let (kappa, pairs) = self.kappa();
        let info = |c: &str| {
            self.classes
                .iter()
                .find(|(id, _)| id == c)
                .map(|(_, v)| v.clone())
        };
        json!({
            "n": self.records.len(),
            "selected": self.selected,
            "current": current.map(|c| {
                let state = self.state(c);
                let waiting = self.next(c).is_some();
                json!({
                    "cluster": c,
                    "references": self.references.get(c).map_or(0, Vec::len),
                    "waiting": waiting,
                    "info": (!waiting).then(|| info(c)),
                    "state": state,
                })
            }),
            "clusters": self.classes.iter().map(|(c, info)| json!({
                "cluster": c,
                "size": info["size"],
                "state": self.state(c),
            })).collect::<Vec<_>>(),
            "false_yes": self.false_yes(),
            "kappa": kappa,
            "repeat_pairs": pairs,
            "repeat_target": REPEAT_TARGET,
            "seconds": self.records.iter().map(|r| match r { Record::Answer { seconds, .. } => *seconds, _ => 0.0 }).sum::<f64>(),
        })
    }
}

#[derive(Deserialize)]
struct AnswerBody {
    n: usize,
    fits: bool,
    #[serde(default)]
    seconds: f64,
}

#[derive(Deserialize)]
struct NBody {
    n: usize,
}

#[derive(Deserialize)]
struct SelectBody {
    clusters: Vec<String>,
}

#[derive(Deserialize)]
struct DecideBody {
    cluster: String,
    action: Action,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    into: Option<String>,
    #[serde(default)]
    note: Option<String>,
}

/// The purity page's routes.
pub fn handle(p: &mut Purity, post: bool, url: &str, body: &str) -> Reply {
    let path = url.split('?').next().unwrap_or(url);
    match (post, path) {
        (false, "/") => Reply::Page(PAGE),
        (false, "/api/session") => Reply::Json(p.session()),
        (false, "/audio/next") => match p.waiting().and_then(|i| p.audio_path(&i.id).cloned()) {
            Some(path) => Reply::Audio(path),
            None => Reply::NotFound,
        },
        // Replays of clips already answered (the reveal names them).
        (false, path) if path.starts_with("/audio/ref/") => {
            let clip = p.current().and_then(|c| {
                let i: usize = path["/audio/ref/".len()..].parse().ok()?;
                p.references.get(c)?.get(i)
            });
            match clip.and_then(|id| p.audio_path(id)) {
                Some(path) => Reply::Audio(path.clone()),
                None => Reply::NotFound,
            }
        }
        (false, path) if path.starts_with("/audio/") => {
            let id = decode(&path["/audio/".len()..]);
            let heard = p
                .records
                .iter()
                .any(|r| matches!(r, Record::Answer { id: a, .. } if *a == id));
            match p.audio_path(&id) {
                Some(path) if heard => Reply::Audio(path.clone()),
                _ => Reply::NotFound,
            }
        }
        (true, "/api/answer") => {
            let parsed: Result<AnswerBody, _> = serde_json::from_str(body);
            match parsed
                .map_err(|e| e.to_string())
                .and_then(|b| p.answer(b.n, b.fits, b.seconds))
            {
                Ok(reveal) => Reply::Json(json!({"reveal": reveal, "session": p.session()})),
                Err(msg) => Reply::Bad(msg),
            }
        }
        (true, "/api/skip") => {
            let parsed: Result<NBody, _> = serde_json::from_str(body);
            match parsed.map_err(|e| e.to_string()).and_then(|b| p.skip(b.n)) {
                Ok(()) => Reply::Json(p.session()),
                Err(msg) => Reply::Bad(msg),
            }
        }
        (true, "/api/select") => {
            let parsed: Result<SelectBody, _> = serde_json::from_str(body);
            match parsed
                .map_err(|e| e.to_string())
                .and_then(|b| p.select(b.clusters))
            {
                Ok(()) => Reply::Json(p.session()),
                Err(msg) => Reply::Bad(msg),
            }
        }
        (true, "/api/decide") => {
            let parsed: Result<DecideBody, _> = serde_json::from_str(body);
            match parsed
                .map_err(|e| e.to_string())
                .and_then(|b| p.decide(&b.cluster, b.action, b.name, b.into, b.note))
            {
                Ok(()) => Reply::Json(p.session()),
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
    use crate::fixture;

    fn demo(count: usize, classes: usize) -> (tempfile::TempDir, Purity) {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), count).unwrap();
        fixture::write_clusters(dir.path(), classes).unwrap();
        let p = Purity::load(dir.path()).unwrap();
        (dir, p)
    }

    /// Answer the waiting clip: members by `member`, foils "no", repeats as
    /// first answered.
    fn answer_with(p: &mut Purity, member: impl Fn(usize) -> bool) -> Kind {
        let item = p.waiting().unwrap();
        let fits = match item.kind {
            Kind::Foil => false,
            Kind::Repeat => p.records.iter().any(
                |r| matches!(r, Record::Answer { id, fits: true, repeat: false, .. } if *id == item.id),
            ),
            Kind::Member => member(p.records.len()),
        };
        p.answer(p.records.len(), fits, 2.0).unwrap();
        item.kind
    }

    #[test]
    fn members_are_formed_and_foils_come_from_other_clusters() {
        let (_dir, p) = demo(120, 3);
        assert_eq!(p.classes.len(), 3);
        let formed: usize = p
            .order
            .values()
            .chain(p.references.values())
            .map(Vec::len)
            .sum();
        assert_eq!(
            formed, 108,
            "every tenth clip is unformed; references aren't asked"
        );
        for (class, foils) in &p.foils {
            assert!(!foils.is_empty());
            assert!(foils.iter().all(|f| !p.order[class].contains(f)));
        }
        // Stratified: the first few members span several conversations.
        let convs: HashSet<&str> = p.order["c0"][..6]
            .iter()
            .map(|id| p.lines[id]["source"]["conversation"].as_str().unwrap())
            .collect();
        assert!(convs.len() >= 4, "{convs:?}");
    }

    #[test]
    fn sprt_stops_a_pure_cluster_with_foils_and_repeats_mixed_in() {
        let (_dir, mut p) = demo(120, 3);
        let mut kinds = Vec::new();
        while p.current() == Some("c0") && p.waiting().is_some() {
            kinds.push(answer_with(&mut p, |_| true));
        }
        let s = p.state("c0");
        assert_eq!(s.sprt, Some("pure"));
        // ln(19) / ln(1.6) = 6.3, so seven yeses decide it.
        assert_eq!(s.members_yes, 7);
        assert_eq!(kinds[FOIL_EVERY - 1], Kind::Foil);
        assert_eq!(kinds[REPEAT_EVERY - 1], Kind::Repeat);
        assert_eq!(s.purity, Some(1.0));
        assert_eq!(p.false_yes(), Some(0.0));
    }

    #[test]
    fn sprt_stops_an_impure_cluster_and_purity_corrects_for_false_yes() {
        let (_dir, mut p) = demo(120, 3);
        // Members: no, no, yes, no, ... ; ln(0.4) = -0.92, so it needs four noes.
        while p.waiting().is_some() && p.current() == Some("c0") {
            answer_with(&mut p, |n| n % 3 == 2);
        }
        let s = p.state("c0");
        assert_eq!(s.sprt, Some("impure"));
        // A false yes on a foil lowers the corrected purity below the heard one.
        p.records.push(Record::Answer {
            id: "x".into(),
            cluster: "c1".into(),
            fits: true,
            foil: true,
            repeat: false,
            seconds: 0.0,
            at: 0,
        });
        let s = p.state("c0");
        assert!(s.purity.unwrap() < s.observed.unwrap());
    }

    #[test]
    fn answers_persist_and_resume_where_they_left_off() {
        let (dir, mut p) = demo(60, 2);
        for _ in 0..9 {
            answer_with(&mut p, |n| n % 2 == 0);
        }
        let again = Purity::load(dir.path()).unwrap();
        assert_eq!(again.records, p.records);
        assert_eq!(again.waiting(), p.waiting());
        assert!(p.answer(3, true, 1.0).is_err(), "a stale page can't answer");
    }

    #[test]
    fn decisions_move_on_to_the_next_cluster() {
        let (dir, mut p) = demo(60, 3);
        assert!(p.decide("c1", Action::Reject, None, None, None).is_err());
        assert!(p
            .decide("c0", Action::Name, Some("  ".into()), None, None)
            .is_err());
        assert!(p
            .decide("c0", Action::Merge, None, Some("c0".into()), None)
            .is_err());
        p.decide("c0", Action::Name, Some(" teasing ".into()), None, None)
            .unwrap();
        assert_eq!(p.current(), Some("c1"));
        p.decide("c1", Action::Merge, None, Some("c0".into()), None)
            .unwrap();
        p.decide("c2", Action::Split, None, None, Some("two moods".into()))
            .unwrap();
        assert_eq!(p.current(), None);
        let again = Purity::load(dir.path()).unwrap();
        assert!(
            matches!(&again.records[0], Record::Decision { name: Some(n), .. } if n == "teasing")
        );
        assert!(again.waiting().is_none());
    }

    #[test]
    fn a_skipped_clip_goes_to_the_back_and_comes_back() {
        let (dir, mut p) = demo(120, 3);
        let first = p.waiting().unwrap();
        assert!(p.skip(5).is_err(), "a stale page can't skip");
        p.skip(0).unwrap();
        let second = p.waiting().unwrap();
        assert_ne!(second.id, first.id);
        assert_eq!(p.state("c0").answers, 0, "a skip isn't an answer");
        assert_eq!(p.state("c0").skips, 1);
        // It comes back once every other member has been asked.
        let mut seen = Vec::new();
        while p.current() == Some("c0") && p.waiting().is_some() {
            let item = p.waiting().unwrap();
            if item.kind == Kind::Member {
                seen.push(item.id.clone());
            }
            p.answer(p.records.len(), seen.len() % 2 == 0, 1.0).unwrap();
        }
        let members = &p.order["c0"];
        if seen.len() == members.len() {
            assert_eq!(seen.last(), Some(&first.id));
        } else {
            assert!(!seen.contains(&first.id));
        }
        // Skips persist, so a restart resumes exactly.
        let again = Purity::load(dir.path()).unwrap();
        assert_eq!(again.records, p.records);
        assert_eq!(again.waiting(), p.waiting());
    }

    #[test]
    fn a_skipped_foil_is_replaced_by_another_foil() {
        let (_dir, mut p) = demo(120, 3);
        for _ in 0..FOIL_EVERY - 1 {
            answer_with(&mut p, |_| true);
        }
        let foil = p.waiting().unwrap();
        assert_eq!(foil.kind, Kind::Foil);
        p.skip(p.records.len()).unwrap();
        let next = p.waiting().unwrap();
        assert_eq!(next.kind, Kind::Foil, "the slot stays a foil");
        assert_ne!(next.id, foil.id);
    }

    #[test]
    fn picked_clusters_rotate_and_the_rest_wait() {
        let (_dir, mut p) = demo(240, 4);
        assert!(p.select(vec!["c9".into()]).is_err());
        p.select(vec!["c1".into(), "c3".into()]).unwrap();
        let mut order = Vec::new();
        for _ in 0..6 {
            order.push(p.current().unwrap().to_string());
            answer_with(&mut p, |n| n % 2 == 0);
        }
        assert_eq!(order, ["c1", "c3", "c1", "c3", "c1", "c3"]);
        assert_eq!(p.state("c0").answers, 0);
        // A picked cluster that is decided drops out; then the next undecided.
        p.select(vec!["c2".into()]).unwrap();
        p.decide("c2", Action::Reject, None, None, None).unwrap();
        assert_eq!(p.current(), Some("c0"));
        let Reply::Json(s) = handle(&mut p, true, "/api/select", r#"{"clusters":["c3"]}"#) else {
            panic!("select failed")
        };
        assert_eq!(s["current"]["cluster"], "c3");
        assert_eq!(s["selected"], json!(["c3"]));
    }

    #[test]
    fn kappa_from_hidden_repeats() {
        let (_dir, mut p) = demo(240, 2);
        let mut flip = 0;
        // Members alternate yes and no, so the SPRT runs long and repeats
        // come round; every repeat matches except the first.
        while p.records.len() < 30 && p.waiting().is_some() {
            let item = p.waiting().unwrap();
            let fits = match item.kind {
                Kind::Repeat => {
                    flip += 1;
                    let first = p.records.iter().any(|r| {
                        matches!(r, Record::Answer { id, fits: true, repeat: false, .. } if *id == item.id)
                    });
                    if flip == 1 {
                        !first
                    } else {
                        first
                    }
                }
                Kind::Foil => false,
                Kind::Member => p.records.len() % 2 == 0,
            };
            p.answer(p.records.len(), fits, 1.0).unwrap();
        }
        let (kappa, pairs) = p.kappa();
        assert!(pairs >= 3, "{pairs}");
        assert!(kappa.unwrap() < 1.0);
    }

    #[test]
    fn references_are_typical_members_never_asked() {
        let (_dir, p) = demo(200, 4);
        for (class, _) in &p.classes {
            let refs = &p.references[class];
            assert!(!refs.is_empty() && refs.len() <= REFERENCES);
            let heard: HashSet<&str> = refs.iter().map(|id| take_base(id)).collect();
            assert!(p.order[class]
                .iter()
                .all(|id| !heard.contains(take_base(id))));
            assert!(p.foils[class].iter().all(|id| !refs.contains(id)));
        }
        let mut p = p;
        let current = p.current().unwrap().to_string();
        let Reply::Json(s) = handle(&mut p, false, "/api/session", "") else {
            panic!("session")
        };
        assert_eq!(s["current"]["references"], p.references[&current].len());
        assert!(matches!(
            handle(&mut p, false, "/audio/ref/0", ""),
            Reply::Audio(_)
        ));
        assert!(matches!(
            handle(&mut p, false, "/audio/ref/99", ""),
            Reply::NotFound
        ));
    }

    #[test]
    fn the_page_never_sees_an_id_before_the_answer() {
        let (_dir, mut p) = demo(60, 2);
        let waiting = p.waiting().unwrap().id;
        let Reply::Json(s) = handle(&mut p, false, "/api/session", "") else {
            panic!("session")
        };
        assert!(!s.to_string().contains(&waiting));
        assert!(
            s["current"]["info"].is_null(),
            "audits wait for the verdict"
        );
        assert!(matches!(handle(&mut p, false, "/", ""), Reply::Page(_)));
        assert!(matches!(
            handle(&mut p, false, "/audio/next", ""),
            Reply::Audio(_)
        ));
        let by_id = format!("/audio/{waiting}");
        assert!(matches!(handle(&mut p, false, &by_id, ""), Reply::NotFound));
        let Reply::Json(r) = handle(
            &mut p,
            true,
            "/api/answer",
            r#"{"n":0,"fits":true,"seconds":3}"#,
        ) else {
            panic!("answer")
        };
        assert_eq!(r["reveal"]["id"], waiting.as_str());
        assert!(r["reveal"]["text"].is_string());
        assert_eq!(r["reveal"]["eiv_top"].as_array().unwrap().len(), 3);
        assert!(r["reveal"].get("was").is_none());
        assert!(matches!(handle(&mut p, false, &by_id, ""), Reply::Audio(_)));
        assert!(matches!(
            handle(&mut p, true, "/api/answer", r#"{"n":0,"fits":true}"#),
            Reply::Bad(_)
        ));
        let Reply::Json(s) = handle(
            &mut p,
            true,
            "/api/decide",
            r#"{"cluster":"c0","action":"name","name":"calm"}"#,
        ) else {
            panic!("decide")
        };
        assert_eq!(s["current"]["cluster"], "c1");
    }

    #[test]
    fn without_cluster_is_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), 3).unwrap();
        let err = Purity::load(dir.path()).err().unwrap();
        assert!(err.to_string().contains("cluster stage has not run"));
    }
}
