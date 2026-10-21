//! Listening tasks: the blind sets the TTS steering jobs write
//! (`listen/<group>/NN.wav`, `sheet.csv`, the hidden `key.json`), filled in
//! from the page straight into `sheet.csv`, and the qtcheck audio, heard
//! file by file with ok / not ok and a note.
//!
//! The page never sees `key.json` until every clip in the set is answered;
//! then the set's results go to `listen/results.md`, the same table
//! `voicearms.py --unblind` writes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::Reply;

pub const SHEET: &str = "sheet.csv";
pub const KEY: &str = "key.json";
pub const RESULTS: &str = "results.md";
/// The answer columns of a sheet, after group, clip and sentence.
pub const ANSWERS: [&str; 4] = [
    "still_her_voice_y_n",
    "mood_heard_0_3",
    "words_clear_y_n",
    "notes",
];
/// Where qtcheck answers go, beside its report.
pub const HEARD: &str = "heard.jsonl";

const LISTEN_PAGE: &str = include_str!("listen.html");
const QT_PAGE: &str = include_str!("qtcheck.html");

/// Fields of one CSV line. Quoted fields may hold commas, quotes ("") and
/// newlines, as Python's csv module writes them.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            (true, '"') => quoted = false,
            (true, c) => field.push(c),
            (false, '"') if field.is_empty() => quoted = true,
            (false, ',') => row.push(std::mem::take(&mut field)),
            (false, '\r') => {}
            (false, '\n') => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            (false, c) => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

pub fn write_csv(rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    for row in rows {
        let line: Vec<String> = row
            .iter()
            .map(|f| {
                if f.contains([',', '"', '\n', '\r']) {
                    format!("\"{}\"", f.replace('"', "\"\""))
                } else {
                    f.clone()
                }
            })
            .collect();
        out.push_str(&line.join(","));
        out.push_str("\r\n");
    }
    out
}

fn mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// A blind listening set: the `listen/` folder of a voicearms run.
pub struct Listen {
    pub dir: PathBuf,
    header: Vec<String>,
    rows: Vec<Vec<String>>,
    /// The sheet's mtime when last read or written, to notice a new run.
    seen: Option<SystemTime>,
}

impl Listen {
    pub fn is_set(dir: &Path) -> bool {
        dir.join(SHEET).is_file() && dir.join(KEY).is_file()
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join(SHEET);
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        let mut rows = parse_csv(&text).into_iter();
        let header = rows.next().unwrap_or_default();
        if header.len() < 3 + ANSWERS.len() || header[..3] != ["group", "clip", "sentence"] {
            return Err(format!("{}: not a listening sheet", path.display()));
        }
        let width = header.len();
        let rows = rows
            .filter(|r| r.iter().any(|f| !f.is_empty()))
            .map(|mut r| {
                r.resize(width, String::new());
                r
            })
            .collect();
        Ok(Self {
            dir,
            header,
            rows,
            seen: mtime(&path),
        })
    }

    /// Whether the sheet changed on disk since we last read or wrote it.
    pub fn stale(&self) -> bool {
        mtime(&self.dir.join(SHEET)) != self.seen
    }

    fn col(&self, name: &str) -> usize {
        self.header.iter().position(|h| h == name).unwrap_or(0)
    }

    fn answered(&self, row: &[String]) -> bool {
        ANSWERS[..3]
            .iter()
            .all(|a| !row[self.col(a)].trim().is_empty())
    }

    pub fn total(&self) -> usize {
        self.rows.len()
    }

    pub fn done(&self) -> usize {
        self.rows.iter().filter(|r| self.answered(r)).count()
    }

    fn row(&self, group: &str, clip: &str) -> Option<usize> {
        self.rows.iter().position(|r| r[0] == group && r[1] == clip)
    }

    fn example(&self, group: &str) -> Option<PathBuf> {
        let entries = fs::read_dir(self.dir.join(group)).ok()?;
        entries
            .flatten()
            .map(|e| e.path())
            .find(|p| p.file_stem().is_some_and(|s| s == "her_example"))
    }

    fn audio(&self, group: &str, clip: &str) -> Option<PathBuf> {
        if !self.rows.iter().any(|r| r[0] == group) {
            return None;
        }
        if clip == "her_example" {
            return self.example(group);
        }
        self.row(group, clip)?;
        Some(self.dir.join(group).join(format!("{clip}.wav")))
    }

    pub fn session(&self) -> Value {
        let mut groups: Vec<(String, Vec<Value>)> = Vec::new();
        for r in &self.rows {
            let clip = json!({
                "clip": r[1],
                "sentence": r[2],
                "voice": r[self.col(ANSWERS[0])],
                "mood": r[self.col(ANSWERS[1])],
                "clear": r[self.col(ANSWERS[2])],
                "notes": r[self.col(ANSWERS[3])],
            });
            match groups.last_mut() {
                Some((g, clips)) if *g == r[0] => clips.push(clip),
                _ => groups.push((r[0].clone(), vec![clip])),
            }
        }
        let done = self.done();
        json!({
            "groups": groups.into_iter().map(|(g, clips)| json!({
                "group": g,
                "moods": g.split('+').collect::<Vec<_>>(),
                "example": self.example(&g).is_some(),
                "clips": clips,
            })).collect::<Vec<_>>(),
            "done": done,
            "total": self.total(),
            "results": (done == self.total()).then(|| self.results().ok()).flatten(),
        })
    }

    pub fn answer(&mut self, a: Answer) -> Result<(), String> {
        let check = |v: &str, ok: &[&str], what: &str| {
            if v.is_empty() || ok.contains(&v) {
                Ok(())
            } else {
                Err(format!("{what} must be one of {ok:?}"))
            }
        };
        check(&a.voice, &["y", "n"], "voice")?;
        check(&a.clear, &["y", "n"], "clear")?;
        let blend = a.group.contains('+');
        if !blend {
            check(&a.mood, &["0", "1", "2", "3"], "mood")?;
        }
        if a.mood.len() > 120 || a.notes.len() > 2000 || a.mood.contains('\n') {
            return Err("answer too long".into());
        }
        let i = self
            .row(&a.group, &a.clip)
            .ok_or_else(|| format!("no clip {}/{}", a.group, a.clip))?;
        let cols = ANSWERS.map(|c| self.col(c));
        let row = &mut self.rows[i];
        for (c, v) in cols.into_iter().zip([a.voice, a.mood, a.clear, a.notes]) {
            row[c] = v;
        }
        self.save()?;
        if self.done() == self.total() {
            let table = self.results()?;
            let path = self.dir.join(RESULTS);
            fs::write(&path, table).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(())
    }

    fn save(&mut self) -> Result<(), String> {
        let path = self.dir.join(SHEET);
        let tmp = self.dir.join(format!("{SHEET}.tmp"));
        let mut all = vec![self.header.clone()];
        all.extend(self.rows.iter().cloned());
        fs::write(&tmp, write_csv(&all))
            .and_then(|()| fs::rename(&tmp, &path))
            .map_err(|e| format!("could not save {}: {e}", path.display()))?;
        self.seen = mtime(&path);
        Ok(())
    }

    /// The unblinded table, as `voicearms.py --unblind` writes it.
    pub fn results(&self) -> Result<String, String> {
        let path = self.dir.join(KEY);
        let key: BTreeMap<String, Value> = fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut by: BTreeMap<&str, Vec<(String, &Vec<String>)>> = BTreeMap::new();
        for r in &self.rows {
            if let Some(k) = key.get(&format!("{}/{}", r[0], r[1])) {
                let cond = k["condition"].as_str().unwrap_or("?").to_string();
                by.entry(&r[0]).or_default().push((cond, r));
            }
        }
        let mut out = String::from(
            "# What Brandon heard\n\n| group | condition | her voice | mood heard | words clear | notes |\n|---|---|---|---|---|---|\n",
        );
        for (group, mut items) in by {
            items.sort_by(|a, b| a.0.cmp(&b.0));
            for (cond, r) in items {
                let [v, m, c, n] = ANSWERS.map(|a| r[self.col(a)].as_str());
                out.push_str(&format!("| {group} | {cond} | {v} | {m} | {c} | {n} |\n"));
            }
        }
        Ok(out)
    }
}

#[derive(Deserialize)]
pub struct Answer {
    pub group: String,
    pub clip: String,
    #[serde(default)]
    pub voice: String,
    #[serde(default)]
    pub mood: String,
    #[serde(default)]
    pub clear: String,
    #[serde(default)]
    pub notes: String,
}

fn query(url: &str, name: &str) -> Option<String> {
    url.split_once('?')?
        .1
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| crate::server::decode(v))
}

/// The listening page's routes, relative to the set's own address.
pub fn handle_listen(l: &mut Listen, post: bool, url: &str, body: &str) -> Reply {
    let path = url.split('?').next().unwrap_or(url);
    match (post, path) {
        (false, "/") => Reply::Page(LISTEN_PAGE),
        (false, "/api/session") => Reply::Json(l.session()),
        (false, "/audio") => match (query(url, "g"), query(url, "c")) {
            (Some(g), Some(c)) => l.audio(&g, &c).map_or(Reply::NotFound, Reply::Audio),
            _ => Reply::NotFound,
        },
        (true, "/api/answer") => match serde_json::from_str::<Answer>(body) {
            Ok(a) => match l.answer(a) {
                Ok(()) => Reply::Json(l.session()),
                Err(e) => Reply::Bad(e),
            },
            Err(e) => Reply::Bad(e.to_string()),
        },
        _ => Reply::NotFound,
    }
}

/// A qtcheck output folder: `report.json` and `audio/<case>-<n>-qt.wav`,
/// with `-torch.wav` beside some of them.
pub struct QtCheck {
    pub dir: PathBuf,
    /// The qwentts.cpp files, by name, sorted.
    files: Vec<String>,
    heard: BTreeMap<String, Heard>,
}

#[derive(Clone, Deserialize, serde::Serialize)]
pub struct Heard {
    pub file: String,
    pub ok: bool,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub at: u64,
}

impl QtCheck {
    pub fn is_check(dir: &Path) -> bool {
        dir.join("report.json").is_file() && !qt_files(&dir.join("audio")).is_empty()
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref().to_path_buf();
        let files = qt_files(&dir.join("audio"));
        let mut heard = BTreeMap::new();
        if let Ok(text) = fs::read_to_string(dir.join(HEARD)) {
            for (n, line) in text
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
            {
                let h: Heard = serde_json::from_str(line)
                    .map_err(|e| format!("{}:{}: {e}", dir.join(HEARD).display(), n + 1))?;
                heard.insert(h.file.clone(), h);
            }
        }
        Ok(Self { dir, files, heard })
    }

    pub fn total(&self) -> usize {
        self.files.len()
    }

    pub fn done(&self) -> usize {
        self.files
            .iter()
            .filter(|f| self.heard.contains_key(*f))
            .count()
    }

    fn torch(&self, qt: &str) -> Option<String> {
        let t = qt.strip_suffix("-qt.wav")?.to_string() + "-torch.wav";
        self.dir.join("audio").join(&t).is_file().then_some(t)
    }

    pub fn session(&self) -> Value {
        let verdict = fs::read_to_string(self.dir.join("report.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|r| r.get("verdict").cloned());
        json!({
            "verdict": verdict,
            "done": self.done(),
            "total": self.total(),
            "files": self.files.iter().map(|f| json!({
                "file": f,
                "torch": self.torch(f),
                "heard": self.heard.get(f),
            })).collect::<Vec<_>>(),
        })
    }

    pub fn hear(&mut self, h: Heard) -> Result<(), String> {
        if !self.files.contains(&h.file) {
            return Err(format!("no file {}", h.file));
        }
        if h.note.len() > 2000 {
            return Err("note too long".into());
        }
        let path = self.dir.join(HEARD);
        let line = serde_json::to_string(&h).map_err(|e| e.to_string())? + "\n";
        use std::io::Write;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(line.as_bytes()))
            .map_err(|e| format!("could not save {}: {e}", path.display()))?;
        self.heard.insert(h.file.clone(), h);
        Ok(())
    }

    /// What he heard, one line per file, for the thread.
    pub fn report(&self) -> String {
        self.files
            .iter()
            .filter_map(|f| self.heard.get(f))
            .map(|h| {
                let mark = if h.ok { "ok" } else { "NOT ok" };
                if h.note.is_empty() {
                    format!("- {}: {mark}\n", h.file)
                } else {
                    format!("- {}: {mark} ({})\n", h.file, h.note.replace('\n', " "))
                }
            })
            .collect()
    }
}

fn qt_files(audio: &Path) -> Vec<String> {
    let mut files: Vec<String> = fs::read_dir(audio)
        .map(|d| {
            d.flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| n.ends_with("-qt.wav"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// The qtcheck page's routes, relative to the check's own address.
pub fn handle_qt(q: &mut QtCheck, post: bool, url: &str, body: &str) -> Reply {
    let path = url.split('?').next().unwrap_or(url);
    match (post, path) {
        (false, "/") => Reply::Page(QT_PAGE),
        (false, "/api/session") => Reply::Json(q.session()),
        (false, "/audio") => match query(url, "f") {
            Some(f)
                if q.files.contains(&f)
                    || q.files.iter().any(|x| q.torch(x) == Some(f.clone())) =>
            {
                Reply::Audio(q.dir.join("audio").join(f))
            }
            _ => Reply::NotFound,
        },
        (true, "/api/heard") => match serde_json::from_str::<Heard>(body) {
            Ok(mut h) => {
                h.at = crate::server::now();
                match q.hear(h) {
                    Ok(()) => Reply::Json(q.session()),
                    Err(e) => Reply::Bad(e),
                }
            }
            Err(e) => Reply::Bad(e.to_string()),
        },
        _ => Reply::NotFound,
    }
}
