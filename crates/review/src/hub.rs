//! The hub: one address for every page where Brandon does something for
//! the project. Its home page lists what needs him and what he has
//! finished; each task's page is the existing review page, served under the
//! task's own path with a bar on top that links back and turns green when
//! the task is done.
//!
//! Tasks are found, not registered: purity, gold and channel when their
//! inputs exist, every blind listening set (`listen/sheet.csv` with its
//! `key.json`) and qtcheck folder under a stage's current output or
//! `analysis/`, any `clips.toml` (a quick clip list a job or a Claude
//! thread writes to play him some clips), and any `task.toml` a job leaves
//! beside its output for a review the hub doesn't know yet. The home page rescans on every visit.
//!
//! When a task's state changes (finished, or reopened by new data) the
//! hub appends it to `hub.jsonl`, so Done shows when it was finished.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::channel::{self, Channel};
use crate::clips::{self, Clips, Index};
use crate::corpus::current_out;
use crate::gold::{self, Gold};
use crate::listen::{self, Listen, QtCheck};
use crate::purity::{self, Purity};
use crate::server::{decode, now, Reply};

pub const LOG: &str = "hub.jsonl";
/// A job's note that something needs Brandon, beside its output.
pub const TASK: &str = "task.toml";
/// Written beside a `task.toml` when he marks it done.
pub const TASK_DONE: &str = "task.done";

const PAGE: &str = include_str!("hub.html");
const BAR: &str = include_str!("bar.html");
/// How deep under a stage's output or `analysis/` to look for tasks.
const DEPTH: usize = 4;

enum Slot<T> {
    Ready(T),
    Missing(String),
}

impl<T> Slot<T> {
    fn ready(&mut self) -> Option<&mut T> {
        match self {
            Slot::Ready(t) => Some(t),
            Slot::Missing(_) => None,
        }
    }
}

/// A job's `task.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Note {
    pub title: String,
    #[serde(default)]
    pub about: String,
    /// Which thread asked for it, to say where the summary goes.
    #[serde(default)]
    pub asked_by: String,
    /// A page to open for it, if it has one.
    #[serde(default)]
    pub link: Option<String>,
    #[serde(skip)]
    dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Event {
    task: String,
    done: bool,
    at: u64,
}

/// One task on the home page.
#[derive(Debug, Clone, Serialize)]
pub struct Task {
    pub id: String,
    pub kind: &'static str,
    pub title: String,
    pub about: String,
    /// Where its files are, from the corpus folder.
    pub folder: String,
    /// Its page, when it has one.
    pub url: Option<String>,
    pub done: usize,
    pub total: usize,
    pub unit: &'static str,
    pub complete: bool,
    /// When it was last finished, while it stays finished.
    pub finished_at: Option<u64>,
    /// The summary to paste into the thread, once finished.
    pub report: Option<String>,
}

pub struct Hub {
    root: PathBuf,
    moods: PathBuf,
    purity: Slot<Purity>,
    gold: Slot<Gold>,
    channel: Slot<Channel>,
    /// By folder, from the corpus root.
    listens: BTreeMap<String, Listen>,
    checks: BTreeMap<String, QtCheck>,
    notes: BTreeMap<String, Note>,
    lists: BTreeMap<String, Clips>,
    /// Her lines' audio, read when the first clip list needs it.
    index: Option<Index>,
    /// Problems found while scanning, shown under "Not ready".
    problems: BTreeMap<String, String>,
    log: Vec<Event>,
}

fn slug(rel: &str) -> String {
    rel.replace('/', "~")
}

impl Hub {
    /// `root` is the corpus folder; `moods` the character's `moods.toml`
    /// for the gold page.
    pub fn load(root: impl AsRef<Path>, moods: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(format!("{} is not a folder", root.display()));
        }
        let mut log = Vec::new();
        if let Ok(text) = fs::read_to_string(root.join(LOG)) {
            for (n, line) in text
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
            {
                log.push(
                    serde_json::from_str(line)
                        .map_err(|e| format!("{}:{}: {e}", root.join(LOG).display(), n + 1))?,
                );
            }
        }
        let mut hub = Self {
            moods: moods.as_ref().to_path_buf(),
            purity: Slot::Missing(String::new()),
            gold: Slot::Missing(String::new()),
            channel: Slot::Missing(String::new()),
            listens: BTreeMap::new(),
            checks: BTreeMap::new(),
            notes: BTreeMap::new(),
            lists: BTreeMap::new(),
            index: None,
            problems: BTreeMap::new(),
            log,
            root,
        };
        hub.scan();
        Ok(hub)
    }

    fn rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned()
    }

    /// Load what wasn't there before and pick up new or replaced folders.
    /// The three built-in pages load once: their answers are on disk, but
    /// purity keeps which clusters he picked in memory.
    pub fn scan(&mut self) {
        if let Slot::Missing(_) = self.purity {
            self.purity =
                Purity::load(&self.root).map_or_else(|e| Slot::Missing(e.to_string()), Slot::Ready);
        }
        if let Slot::Missing(_) = self.gold {
            self.gold = Gold::load(&self.root, &self.moods)
                .map_or_else(|e| Slot::Missing(e.to_string()), Slot::Ready);
        }
        if let Slot::Missing(_) = self.channel {
            self.channel = Channel::load(&self.root)
                .map_or_else(|e| Slot::Missing(e.to_string()), Slot::Ready);
        }

        let mut tops = Vec::new();
        if let Ok(stages) = fs::read_dir(self.root.join("stages")) {
            let mut names: Vec<String> = stages
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .collect();
            names.sort();
            for name in names {
                if let Ok(Some(out)) = current_out(&self.root, &name) {
                    tops.push(out);
                }
            }
        }
        tops.push(self.root.join("analysis"));
        let mut found = Found::default();
        for top in tops {
            walk(&top, DEPTH, &mut found);
        }
        self.problems.clear();

        let listens: BTreeSet<String> = found.listens.iter().map(|d| self.rel(d)).collect();
        self.listens
            .retain(|k, l| listens.contains(k) && !l.stale());
        for dir in &found.listens {
            let rel = self.rel(dir);
            if let Entry::Vacant(slot) = self.listens.entry(rel.clone()) {
                match Listen::load(dir) {
                    Ok(l) => {
                        slot.insert(l);
                    }
                    Err(e) => {
                        self.problems.insert(format!("listen:{rel}"), e);
                    }
                }
            }
        }
        let checks: BTreeSet<String> = found.checks.iter().map(|d| self.rel(d)).collect();
        self.checks.retain(|k, _| checks.contains(k));
        for dir in &found.checks {
            let rel = self.rel(dir);
            if let Entry::Vacant(slot) = self.checks.entry(rel.clone()) {
                match QtCheck::load(dir) {
                    Ok(q) => {
                        slot.insert(q);
                    }
                    Err(e) => {
                        self.problems.insert(format!("qtcheck:{rel}"), e);
                    }
                }
            }
        }
        let lists: BTreeSet<String> = found.lists.iter().map(|d| self.rel(d)).collect();
        self.lists.retain(|k, c| lists.contains(k) && !c.stale());
        for dir in &found.lists {
            let rel = self.rel(dir);
            if let Entry::Vacant(slot) = self.lists.entry(rel.clone()) {
                let index = self.index.get_or_insert_with(|| Index::load(&self.root));
                match Clips::load(&self.root, dir, index) {
                    Ok(c) => {
                        slot.insert(c);
                    }
                    Err(e) => {
                        self.problems.insert(format!("clips:{rel}"), e);
                    }
                }
            }
        }
        self.notes.clear();
        for dir in &found.notes {
            let rel = self.rel(dir);
            let path = dir.join(TASK);
            match fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|t| toml::from_str::<Note>(&t).map_err(|e| e.to_string()))
            {
                Ok(mut n) => {
                    n.dir = dir.clone();
                    self.notes.insert(rel, n);
                }
                Err(e) => {
                    self.problems
                        .insert(format!("task:{rel}"), format!("{}: {e}", path.display()));
                }
            }
        }
    }

    /// Every task, with its state brought up to date in the log.
    pub fn tasks(&mut self) -> Vec<Task> {
        let root = self.root.display().to_string();
        let mut tasks = Vec::new();
        let mut push = |id: String,
                        kind: &'static str,
                        title: String,
                        about: String,
                        folder: String,
                        url: Option<String>,
                        (done, total): (usize, usize),
                        unit: &'static str,
                        report: &dyn Fn() -> String| {
            if total == 0 {
                return;
            }
            let complete = done >= total;
            tasks.push(Task {
                id,
                kind,
                title,
                about,
                folder,
                url,
                done,
                total,
                unit,
                complete,
                finished_at: None,
                report: complete.then(report),
            });
        };

        if let Some(p) = self.purity.ready() {
            let s = p.session();
            let clusters = s["clusters"].as_array().cloned().unwrap_or_default();
            let decided = p.decided();
            push(
                "purity".into(),
                "purity",
                "Cluster purity".into(),
                "Listen to clips from each voice cluster, say whether each fits, then name, merge, split or reject the cluster.".into(),
                purity::PURITY.into(),
                Some("/purity/".into()),
                (decided, clusters.len()),
                "clusters decided",
                &|| {
                    let mut r = format!(
                        "Cluster purity is done: all {} clusters decided (answers in {root}/{}).\n",
                        clusters.len(),
                        purity::PURITY
                    );
                    for c in &clusters {
                        let d = &c["state"]["decision"];
                        let field = |k: &str| d[k].as_str().filter(|v| !v.is_empty()).map(str::to_string);
                        let mut line = format!("- {}: {}", c["cluster"].as_str().unwrap_or("?"), field("action").unwrap_or_default());
                        if let Some(n) = field("name") {
                            line += &format!(" \"{n}\"");
                        }
                        if let Some(i) = field("into") {
                            line += &format!(" into {i}");
                        }
                        if let Some(n) = field("note") {
                            line += &format!(" ({n})");
                        }
                        r += &line;
                        r.push('\n');
                    }
                    r
                },
            );
        }
        if let Some(g) = self.gold.ready() {
            let (done, total) = (g.labelled(), g.lines.len());
            push(
                "gold".into(),
                "gold",
                "Gold labels".into(),
                "Label a sample of her lines: the moods each really carries and the knobs.".into(),
                gold::GOLD.into(),
                Some("/gold/".into()),
                (done, total),
                "lines labelled",
                &|| {
                    format!(
                        "Gold labels are done: {total} lines labelled (in {root}/{}).\n",
                        gold::GOLD
                    )
                },
            );
        }
        if let Some(c) = self.channel.ready() {
            let (done, total) = (c.marked(), c.clips.len());
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for m in c.marks.values() {
                *counts.entry(m.channel.clone()).or_default() += 1;
            }
            push(
                "channel".into(),
                "channel",
                "Radio and effect marks".into(),
                "Mark, by ear, which clips carry a radio or other effect on top of the suit filter.".into(),
                channel::MARKS.into(),
                Some("/channel/".into()),
                (done, total),
                "clips marked",
                &|| {
                    let tally: Vec<String> = counts.iter().map(|(k, n)| format!("{k} {n}")).collect();
                    format!("Channel marks are done: {total} clips marked ({}), in {root}/{}.\n", tally.join(", "), channel::MARKS)
                },
            );
        }
        for (rel, l) in &self.listens {
            let from = rel
                .strip_prefix("stages/")
                .and_then(|r| r.split('/').next())
                .unwrap_or("a listening run");
            push(
                format!("listen:{rel}"),
                "listen",
                format!("Blind listening set ({from})"),
                "For each clip: is it still her voice, how much of the mood do you hear, are the words clear. Her real example clip is beside each set.".into(),
                rel.clone(),
                Some(format!("/listen/{}/", slug(rel))),
                (l.done(), l.total()),
                "clips answered",
                &|| {
                    let table = l.results().unwrap_or_else(|e| format!("(could not unblind: {e})\n"));
                    format!("Blind listening set {root}/{rel} is done. What I heard:\n\n{table}")
                },
            );
        }
        for (rel, q) in &self.checks {
            push(
                format!("qtcheck:{rel}"),
                "qtcheck",
                "Vulkan voice check (qtcheck)".into(),
                "Does each qwentts.cpp clip sound right? PyTorch's version plays beside the first sentence.".into(),
                rel.clone(),
                Some(format!("/qtcheck/{}/", slug(rel))),
                (q.done(), q.total()),
                "clips heard",
                &|| format!("qtcheck listening for {root}/{rel} is done:\n{}", q.report()),
            );
        }
        for (rel, c) in &self.lists {
            let asked = if c.file.asked_by.is_empty() {
                String::new()
            } else {
                format!(" (for \"{}\")", c.file.asked_by)
            };
            push(
                format!("clips:{rel}"),
                "clips",
                c.file.title.clone(),
                c.file.question.clone(),
                rel.clone(),
                Some(format!("/clips/{}/", slug(rel))),
                (c.done(), c.total()),
                "clips answered",
                &|| {
                    format!(
                        "Clips \"{}\"{asked} are done ({root}/{rel}/{}):\n{}",
                        c.file.title,
                        clips::ANSWERS,
                        c.report()
                    )
                },
            );
        }
        for (rel, n) in &self.notes {
            let done = n.dir.join(TASK_DONE).is_file();
            let asked = (!n.asked_by.is_empty()).then(|| format!(" (for \"{}\")", n.asked_by));
            push(
                format!("task:{rel}"),
                "note",
                n.title.clone(),
                n.about.clone(),
                rel.clone(),
                n.link.clone(),
                (done as usize, 1),
                "done",
                &|| {
                    format!(
                        "{} is done{}. Files: {root}/{rel}\n",
                        n.title,
                        asked.clone().unwrap_or_default()
                    )
                },
            );
        }

        let mut last: BTreeMap<String, Event> = BTreeMap::new();
        for e in &self.log {
            last.insert(e.task.clone(), e.clone());
        }
        let mut new = Vec::new();
        for t in &mut tasks {
            let was = last.get(&t.id);
            if was.is_some_and(|e| e.done) != t.complete {
                let e = Event {
                    task: t.id.clone(),
                    done: t.complete,
                    at: now(),
                };
                last.insert(t.id.clone(), e.clone());
                new.push(e);
            }
            if t.complete {
                t.finished_at = last.get(&t.id).map(|e| e.at);
            }
        }
        if !new.is_empty() {
            let text: String = new
                .iter()
                .filter_map(|e| serde_json::to_string(e).ok())
                .map(|l| l + "\n")
                .collect();
            let path = self.root.join(LOG);
            let written = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut f| f.write_all(text.as_bytes()));
            if let Err(e) = written {
                eprintln!("could not write {}: {e}", path.display());
            }
            self.log.extend(new);
        }
        tasks
    }

    /// The home page's view.
    pub fn view(&mut self) -> Value {
        self.scan();
        let mut tasks = self.tasks();
        // Started ones first, then the rest in order found.
        tasks.sort_by_key(|t| (t.complete, t.done == 0));
        let (done, pending): (Vec<Task>, Vec<Task>) = tasks.into_iter().partition(|t| t.complete);
        let mut done = done;
        done.sort_by_key(|t| std::cmp::Reverse(t.finished_at));
        let mut problems: Vec<Value> = Vec::new();
        for (what, slot) in [
            ("Cluster purity", missing(&self.purity)),
            ("Gold labels", missing(&self.gold)),
            ("Radio and effect marks", missing(&self.channel)),
        ] {
            if let Some(why) = slot {
                problems.push(json!({"task": what, "why": why}));
            }
        }
        for (what, why) in &self.problems {
            problems.push(json!({"task": what, "why": why}));
        }
        json!({
            "corpus": self.root.display().to_string(),
            "pending": pending,
            "done": done,
            "not_ready": problems,
            "now": now(),
        })
    }

    fn task(&mut self, id: &str) -> Option<Task> {
        self.tasks().into_iter().find(|t| t.id == id)
    }

    fn mark(&mut self, id: &str, done: bool) -> Result<(), String> {
        let rel = id
            .strip_prefix("task:")
            .ok_or("only a job's own task can be marked by hand")?;
        let n = self.notes.get(rel).ok_or_else(|| format!("no task {id}"))?;
        let path = n.dir.join(TASK_DONE);
        let r = if done {
            fs::write(&path, format!("{}\n", now()))
        } else {
            fs::remove_file(&path).or_else(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            })
        };
        r.map_err(|e| format!("{}: {e}", path.display()))
    }
}

fn missing<T>(slot: &Slot<T>) -> Option<String> {
    match slot {
        Slot::Ready(_) => None,
        Slot::Missing(why) => Some(why.clone()),
    }
}

#[derive(Default)]
struct Found {
    listens: Vec<PathBuf>,
    checks: Vec<PathBuf>,
    notes: Vec<PathBuf>,
    lists: Vec<PathBuf>,
}

fn walk(dir: &Path, depth: usize, found: &mut Found) {
    if Listen::is_set(dir) {
        found.listens.push(dir.to_path_buf());
        return;
    }
    if QtCheck::is_check(dir) {
        found.checks.push(dir.to_path_buf());
    }
    if dir.join(TASK).is_file() {
        found.notes.push(dir.to_path_buf());
    }
    if dir.join(clips::CLIPS).is_file() {
        found.lists.push(dir.to_path_buf());
    }
    if depth == 0 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut subs: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            !name.starts_with('.') && name != "audio"
        })
        .map(|e| e.path())
        .collect();
    subs.sort();
    for sub in subs {
        walk(&sub, depth - 1, found);
    }
}

/// The task's page with the hub's bar on top.
fn with_bar(page: &str, id: &str) -> String {
    let bar = BAR.replace("__TASK__", &serde_json::to_string(id).unwrap_or_default());
    match page.find("<body>") {
        Some(i) => format!("{}{bar}{}", &page[..i + 6], &page[i + 6..]),
        None => format!("{bar}{page}"),
    }
}

fn barred(reply: Reply, id: &str) -> Reply {
    match reply {
        Reply::Page(p) => Reply::Html(with_bar(p, id)),
        other => other,
    }
}

#[derive(Deserialize)]
struct MarkBody {
    id: String,
    done: bool,
}

/// Split `/<seg>/<rest>` after a prefix into the decoded segment and the
/// rest (always starting with `/`), keeping the query.
fn split_seg(rest: &str) -> Option<(String, String)> {
    let (seg, tail) = rest.split_once('/')?;
    Some((decode(seg), format!("/{tail}")))
}

/// The hub's routes, and every task page's under its prefix.
pub fn handle(h: &mut Hub, post: bool, url: &str, body: &str) -> Reply {
    let path = url.split('?').next().unwrap_or(url);
    for (prefix, id) in [
        ("/purity", "purity"),
        ("/gold", "gold"),
        ("/channel", "channel"),
    ] {
        if path == prefix {
            return Reply::Redirect(format!("{prefix}/"));
        }
        if let Some(rest) = url.strip_prefix(prefix).filter(|r| r.starts_with('/')) {
            let reply = match id {
                "purity" => h
                    .purity
                    .ready()
                    .map(|p| purity::handle(p, post, rest, body)),
                "gold" => h.gold.ready().map(|g| gold::handle(g, post, rest, body)),
                _ => h
                    .channel
                    .ready()
                    .map(|c| channel::handle(c, post, rest, body)),
            };
            return reply.map_or(Reply::NotFound, |r| barred(r, id));
        }
    }
    if let Some(rest) = url.strip_prefix("/listen/") {
        let Some((rel, rest)) = split_seg(rest).map(|(s, r)| (s.replace('~', "/"), r)) else {
            return Reply::Redirect(format!("{url}/"));
        };
        let Some(l) = h.listens.get_mut(&rel) else {
            return Reply::NotFound;
        };
        return barred(
            listen::handle_listen(l, post, &rest, body),
            &format!("listen:{rel}"),
        );
    }
    if let Some(rest) = url.strip_prefix("/clips/") {
        let Some((rel, rest)) = split_seg(rest).map(|(s, r)| (s.replace('~', "/"), r)) else {
            return Reply::Redirect(format!("{url}/"));
        };
        let Some(c) = h.lists.get_mut(&rel) else {
            return Reply::NotFound;
        };
        return barred(clips::handle(c, post, &rest, body), &format!("clips:{rel}"));
    }
    if let Some(rest) = url.strip_prefix("/qtcheck/") {
        let Some((rel, rest)) = split_seg(rest).map(|(s, r)| (s.replace('~', "/"), r)) else {
            return Reply::Redirect(format!("{url}/"));
        };
        let Some(q) = h.checks.get_mut(&rel) else {
            return Reply::NotFound;
        };
        return barred(
            listen::handle_qt(q, post, &rest, body),
            &format!("qtcheck:{rel}"),
        );
    }
    match (post, path) {
        (false, "/") => Reply::Page(PAGE),
        (false, "/api/tasks") => Reply::Json(h.view()),
        (false, "/api/task") => {
            let id = url
                .split_once("?id=")
                .map(|(_, v)| decode(v.split('&').next().unwrap_or(v)))
                .unwrap_or_default();
            match h.task(&id) {
                Some(t) => Reply::Json(json!(t)),
                None => Reply::NotFound,
            }
        }
        (true, "/api/mark") => match serde_json::from_str::<MarkBody>(body) {
            Ok(m) => match h.mark(&m.id, m.done) {
                Ok(()) => Reply::Json(h.view()),
                Err(e) => Reply::Bad(e),
            },
            Err(e) => Reply::Bad(e.to_string()),
        },
        _ => Reply::NotFound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;
    use crate::listen::{parse_csv, write_csv};

    fn demo() -> (tempfile::TempDir, Hub) {
        let dir = tempfile::tempdir().unwrap();
        fixture::write(dir.path(), 60).unwrap();
        fixture::write_clusters(dir.path(), 3).unwrap();
        fixture::write_listening(dir.path()).unwrap();
        let hub = Hub::load(dir.path(), dir.path().join("no-moods.toml")).unwrap();
        (dir, hub)
    }

    fn json(r: Reply) -> Value {
        match r {
            Reply::Json(v) => v,
            other => panic!("expected json, got {other:?}"),
        }
    }

    fn ids(v: &Value, k: &str) -> Vec<String> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|t| t["id"].as_str().unwrap().to_string())
            .collect()
    }

    const SET: &str = "stages/voicearms/synthetic/out/listen";

    #[test]
    fn finds_every_task_and_what_isnt_ready() {
        let (_dir, mut hub) = demo();
        let v = hub.view();
        let pending = ids(&v, "pending");
        assert!(pending.contains(&"purity".to_string()));
        assert!(pending.contains(&format!("listen:{SET}")));
        assert!(pending.contains(&"qtcheck:analysis/qtcheck".to_string()));
        assert!(ids(&v, "done").is_empty());
        let not_ready: Vec<&str> = v["not_ready"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["task"].as_str().unwrap())
            .collect();
        assert!(not_ready.contains(&"Gold labels"), "{not_ready:?}");
        let set = &v["pending"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["kind"] == "listen")
            .unwrap();
        assert_eq!(set["total"], 5);
        assert_eq!(set["url"], format!("/listen/{}/", slug(SET)));
    }

    fn answer(hub: &mut Hub, group: &str, clip: &str, mood: &str) -> Reply {
        let url = format!("/listen/{}/api/answer", slug(SET));
        let body = json!({"group": group, "clip": clip, "voice": "y", "mood": mood, "clear": "n", "notes": "a, \"quoted\"\nnote"});
        handle(hub, true, &url, &body.to_string())
    }

    #[test]
    fn listening_set_fills_the_sheet_and_unblinds_only_when_done() {
        let (dir, mut hub) = demo();
        let base = format!("/listen/{}", slug(SET));
        let s = json(handle(&mut hub, false, &format!("{base}/api/session"), ""));
        let text = s.to_string();
        assert!(s["results"].is_null());
        assert!(
            !text.contains("A21") && !text.contains("swap"),
            "the key leaks: {text}"
        );

        assert!(matches!(
            answer(&mut hub, "warmth", "01", "5"),
            Reply::Bad(_)
        ));
        assert!(matches!(
            answer(&mut hub, "warmth", "09", "1"),
            Reply::Bad(_)
        ));
        for clip in ["01", "02", "03"] {
            json(answer(&mut hub, "warmth", clip, "2"));
        }
        json(answer(&mut hub, "warmth+amused", "01", "warmth:1 amused:3"));
        let t = json(handle(
            &mut hub,
            false,
            &format!("/api/task?id=listen:{SET}"),
            "",
        ));
        assert_eq!(
            (t["done"].as_u64(), t["complete"].as_bool()),
            (Some(4), Some(false))
        );
        assert!(!dir.path().join(SET).join("results.md").exists());

        let s = json(answer(&mut hub, "warmth+amused", "02", "warmth:0 amused:0"));
        let results = s["results"].as_str().unwrap();
        assert!(
            results.contains("| warmth | A21@1 | y | 2 | n |"),
            "{results}"
        );
        assert!(results.contains("warmth:1 amused:3"));
        assert!(dir.path().join(SET).join("results.md").exists());

        // The sheet keeps voicearms' layout, so --unblind still reads it.
        let sheet = fs::read_to_string(dir.path().join(SET).join("sheet.csv")).unwrap();
        let rows = parse_csv(&sheet);
        assert_eq!(rows[0][3], "still_her_voice_y_n");
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[1][6], "a, \"quoted\"\nnote");

        let v = hub.view();
        let done = &v["done"][0];
        assert_eq!(done["id"], format!("listen:{SET}"));
        assert!(done["finished_at"].as_u64().unwrap() > 0);
        assert!(done["report"]
            .as_str()
            .unwrap()
            .contains("What Brandon heard"));
        let log = fs::read_to_string(dir.path().join(LOG)).unwrap();
        assert_eq!(log.lines().count(), 1, "one event per change: {log}");
        hub.view();
        assert_eq!(
            fs::read_to_string(dir.path().join(LOG))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }

    #[test]
    fn a_new_run_replaces_the_set_and_reopens_it() {
        let (dir, mut hub) = demo();
        for (g, c, m) in [
            ("warmth", "01", "1"),
            ("warmth", "02", "1"),
            ("warmth", "03", "1"),
            ("warmth+amused", "01", "warmth:1 amused:1"),
            ("warmth+amused", "02", "warmth:1 amused:1"),
        ] {
            json(answer(&mut hub, g, c, m));
        }
        assert_eq!(ids(&hub.view(), "done"), [format!("listen:{SET}")]);
        // A rerun writes a fresh, empty sheet.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let path = dir.path().join(SET).join("sheet.csv");
        let mut rows = parse_csv(&fs::read_to_string(&path).unwrap());
        for r in rows.iter_mut().skip(1) {
            for f in &mut r[3..] {
                f.clear();
            }
        }
        fs::write(&path, write_csv(&rows) + "\n").unwrap();
        let file = fs::File::options().append(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5))
            .unwrap();
        let v = hub.view();
        assert!(ids(&v, "pending").contains(&format!("listen:{SET}")));
        let log = fs::read_to_string(dir.path().join(LOG)).unwrap();
        assert_eq!(log.lines().count(), 2);
    }

    #[test]
    fn task_pages_get_the_bar_and_only_their_own_files() {
        let (_dir, mut hub) = demo();
        assert!(
            matches!(handle(&mut hub, false, "/purity", ""), Reply::Redirect(to) if to == "/purity/")
        );
        match handle(&mut hub, false, "/purity/", "") {
            Reply::Html(p) => {
                assert!(p.contains("id=\"hubbar\""));
                assert!(p.contains("\"purity\""));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            handle(&mut hub, false, "/purity/api/session", ""),
            Reply::Json(_)
        ));
        assert!(matches!(
            handle(&mut hub, false, "/gold/", ""),
            Reply::NotFound
        ));
        let base = format!("/listen/{}", slug(SET));
        assert!(matches!(
            handle(&mut hub, false, &format!("{base}/"), ""),
            Reply::Html(_)
        ));
        assert!(
            matches!(handle(&mut hub, false, &format!("{base}/audio?g=warmth&c=01"), ""), Reply::Audio(p) if p.ends_with("warmth/01.wav"))
        );
        assert!(
            matches!(handle(&mut hub, false, &format!("{base}/audio?g=warmth&c=her_example"), ""), Reply::Audio(p) if p.ends_with("her_example.wav"))
        );
        assert!(matches!(
            handle(
                &mut hub,
                false,
                &format!("{base}/audio?g=warmth%2Bamused&c=02"),
                ""
            ),
            Reply::Audio(_)
        ));
        for bad in [
            "g=warmth&c=../../key",
            "g=..&c=key",
            "g=warmth&c=09",
            "g=warmth+amused&c=her_example",
        ] {
            assert!(
                matches!(
                    handle(&mut hub, false, &format!("{base}/audio?{bad}"), ""),
                    Reply::NotFound
                ),
                "{bad}"
            );
        }
        assert!(matches!(
            handle(&mut hub, false, &format!("{base}/key.json"), ""),
            Reply::NotFound
        ));
        assert!(matches!(
            handle(&mut hub, false, "/listen/nope/", ""),
            Reply::NotFound
        ));
        assert!(matches!(handle(&mut hub, false, "/", ""), Reply::Page(_)));
    }

    #[test]
    fn qtcheck_is_heard_file_by_file() {
        let (dir, mut hub) = demo();
        let base = "/qtcheck/analysis~qtcheck";
        let s = json(handle(&mut hub, false, &format!("{base}/api/session"), ""));
        assert_eq!(s["total"], 2);
        assert_eq!(s["files"][0]["torch"], "freeman-0-torch.wav");
        assert!(s["files"][1]["torch"].is_null());
        assert!(matches!(
            handle(
                &mut hub,
                false,
                &format!("{base}/audio?f=freeman-0-torch.wav"),
                ""
            ),
            Reply::Audio(_)
        ));
        assert!(matches!(
            handle(
                &mut hub,
                false,
                &format!("{base}/audio?f=../report.json"),
                ""
            ),
            Reply::NotFound
        ));
        let hear = |hub: &mut Hub, f: &str, ok: bool| {
            handle(
                hub,
                true,
                &format!("{base}/api/heard"),
                &json!({"file": f, "ok": ok, "note": "buzz"}).to_string(),
            )
        };
        assert!(matches!(hear(&mut hub, "nope-qt.wav", true), Reply::Bad(_)));
        json(hear(&mut hub, "freeman-0-qt.wav", true));
        json(hear(&mut hub, "freeman-1-qt.wav", false));
        let v = hub.view();
        let t = v["done"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["kind"] == "qtcheck")
            .unwrap();
        assert!(t["report"]
            .as_str()
            .unwrap()
            .contains("freeman-1-qt.wav: NOT ok (buzz)"));
        // Answers survive a restart.
        let again = Hub::load(dir.path(), dir.path().join("no-moods.toml"))
            .unwrap()
            .checks
            .remove("analysis/qtcheck")
            .unwrap();
        assert_eq!(again.done(), 2);
    }

    #[test]
    fn a_jobs_task_toml_is_marked_done_by_hand() {
        let (dir, mut hub) = demo();
        let at = dir.path().join("analysis/after_labels");
        fs::create_dir_all(&at).unwrap();
        fs::write(at.join(TASK), "title = \"Read the steering report\"\nabout = \"Say which arm to keep.\"\nasked_by = \"TTS steering\"\n").unwrap();
        let id = "task:analysis/after_labels";
        assert!(ids(&hub.view(), "pending").contains(&id.to_string()));
        assert!(matches!(
            handle(
                &mut hub,
                true,
                "/api/mark",
                r#"{"id":"purity","done":true}"#
            ),
            Reply::Bad(_)
        ));
        let v = json(handle(
            &mut hub,
            true,
            "/api/mark",
            &json!({"id": id, "done": true}).to_string(),
        ));
        let t = v["done"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["id"] == id)
            .unwrap();
        assert!(t["report"]
            .as_str()
            .unwrap()
            .contains("for \"TTS steering\""));
        json(handle(
            &mut hub,
            true,
            "/api/mark",
            &json!({"id": id, "done": false}).to_string(),
        ));
        assert!(ids(&hub.view(), "pending").contains(&id.to_string()));

        fs::write(at.join(TASK), "nope = ").unwrap();
        let v = hub.view();
        assert!(v["not_ready"].to_string().contains("task.toml"));
    }

    #[test]
    fn purity_finishes_when_every_cluster_is_decided() {
        let (_dir, mut hub) = demo();
        for c in ["c0", "c1", "c2"] {
            let body = json!({"cluster": c, "action": "reject", "note": "demo"}).to_string();
            json(handle(&mut hub, true, "/purity/api/decide", &body));
        }
        let t = json(handle(&mut hub, false, "/api/task?id=purity", ""));
        assert_eq!(t["complete"], true);
        assert!(
            t["report"]
                .as_str()
                .unwrap()
                .contains("- c1: reject (demo)"),
            "{}",
            t["report"]
        );
    }

    #[test]
    fn a_clip_list_plays_ids_and_paths_and_hides_labels_when_blind() {
        let (dir, mut hub) = demo();
        let corpus = crate::Corpus::load(dir.path()).unwrap();
        let line = &corpus.lines[0];
        let at = dir.path().join("analysis/steering/round1");
        fs::create_dir_all(at.join("wav")).unwrap();
        fs::write(at.join("wav/a.wav"), b"RIFF").unwrap();
        let toml = format!(
            "title = \"Fitted tokens\"\nquestion = \"Warm?\"\nanswers = [\"warm\", \"not\"]\nblind = true\nasked_by = \"TTS steering\"\n\
             [[clip]]\nid = \"{}\"\nlabel = \"her real clip\"\n\
             [[clip]]\npath = \"wav/a.wav\"\nlabel = \"C-mean\"\n\
             [[clip]]\npath = \"corpus:analysis/steering/round1/wav/a.wav\"\ntext = \"same file, from the corpus root\"\n\
             [[clip]]\npath = \"wav/missing.wav\"\n",
            line.id
        );
        fs::write(at.join(clips::CLIPS), toml).unwrap();
        let id = "clips:analysis/steering/round1".to_string();
        assert!(ids(&hub.view(), "pending").contains(&id));

        let base = "/clips/analysis~steering~round1";
        let s = json(handle(&mut hub, false, &format!("{base}/api/session"), ""));
        assert_eq!(s["total"], 4);
        assert!(s["clips"][0]["label"].is_null(), "blind hides labels");
        assert_eq!(s["clips"][0]["text"], line.text.as_str());
        assert_eq!(s["clips"][3]["audio"], false);
        assert!(
            matches!(handle(&mut hub, false, &format!("{base}/audio?n=0"), ""), Reply::Audio(p) if p.exists())
        );
        assert!(
            matches!(handle(&mut hub, false, &format!("{base}/audio?n=2"), ""), Reply::Audio(p) if p.ends_with("wav/a.wav"))
        );
        assert!(matches!(
            handle(&mut hub, false, &format!("{base}/audio?n=3"), ""),
            Reply::NotFound
        ));
        assert!(
            matches!(handle(&mut hub, false, &format!("{base}/"), ""), Reply::Html(p) if p.contains("hubbar"))
        );

        let answer = |hub: &mut Hub, n: usize, a: &str| {
            handle(
                hub,
                true,
                &format!("{base}/api/answer"),
                &json!({"n": n, "answer": a, "note": "bit flat"}).to_string(),
            )
        };
        assert!(
            matches!(answer(&mut hub, 0, "yes"), Reply::Bad(_)),
            "only its own answers"
        );
        assert!(matches!(answer(&mut hub, 9, "warm"), Reply::Bad(_)));
        for n in 0..3 {
            json(answer(&mut hub, n, "warm"));
        }
        let s = json(answer(&mut hub, 3, "not"));
        assert_eq!(
            s["clips"][1]["label"], "C-mean",
            "labels show once all are answered"
        );
        let t = json(handle(&mut hub, false, &format!("/api/task?id={id}"), ""));
        let report = t["report"].as_str().unwrap();
        assert!(report.contains("(for \"TTS steering\")"), "{report}");
        assert!(
            report.contains("- C-mean (wav/a.wav): warm: bit flat"),
            "{report}"
        );

        // Answers live beside the list and survive a restart.
        let again = Hub::load(dir.path(), dir.path().join("no-moods.toml")).unwrap();
        assert_eq!(again.lists["analysis/steering/round1"].done(), 4);

        for bad in [
            "path = \"../x.wav\"",
            "path = \"/etc/passwd\"",
            "id = \"a\"\npath = \"b\"",
            "nope = 1",
        ] {
            fs::write(
                at.join(clips::CLIPS),
                format!("title = \"t\"\n[[clip]]\n{bad}\n"),
            )
            .unwrap();
            let file = fs::File::options()
                .append(true)
                .open(at.join(clips::CLIPS))
                .unwrap();
            file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(9))
                .unwrap();
            let v = hub.view();
            assert!(v["not_ready"].to_string().contains("clips.toml"), "{bad}");
            assert!(!ids(&v, "pending").contains(&id), "{bad}");
        }
    }

    #[test]
    fn csv_round_trips() {
        let rows = vec![
            vec!["a".to_string(), "b, c".into(), "say \"hi\"".into()],
            vec!["".into(), "two\nlines".into(), "x".into()],
        ];
        assert_eq!(parse_csv(&write_csv(&rows)), rows);
        assert_eq!(
            parse_csv("a,b\n1,\"2,3\"\n"),
            vec![vec!["a", "b"], vec!["1", "2,3"]]
        );
    }
}
