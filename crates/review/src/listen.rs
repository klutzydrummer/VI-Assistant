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
/// Optional answer columns (Brandon, 2026-10-27), added to the end of an
/// older sheet on load so `--unblind` and earlier answers keep working. They
/// don't count toward a clip being answered. Each is (column, allowed values).
pub const EXTRA: [(&str, &[&str]); 4] = [
    ("accent", &["missing", "partial", "full"]),
    ("suit_effect", &["missing", "partial", "full"]),
    ("mood_right", &["y", "mimic", "n"]),
    ("pitch", &["lower", "correct", "higher"]),
];
/// Where qtcheck answers go, beside its report.
pub const HEARD: &str = "heard.jsonl";

/// The TTS test script, compiled in. Used only to show the listener what each
/// line was meant to sound like (display); never to select, weight or score.
const TTS_LINES: &str = include_str!("../../../jobs/ttstest/lines.json");

/// The listener's view of a group's line: `None` if the group name carries no
/// known line id (an older or different set), else the intended tone, or
/// `Some(None)` inside when the line has no tone target (numbers, a long
/// paragraph, invented words). Display only.
pub fn tone_for(group: &str) -> Option<Option<String>> {
    let last = group.rsplit('/').next()?;
    let id = last.split('_').next()?;
    let lines: Vec<Value> = serde_json::from_str(TTS_LINES).ok()?;
    let line = lines.iter().find(|l| l["id"] == id)?;
    if line["tone_target"] == false {
        return Some(None);
    }
    Some(line["register"].as_str().map(str::to_string))
}

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
    /// Where the hidden key is read from, only once every clip is answered.
    /// Defaults to `<dir>/key.json`; `--key` may point anywhere.
    key: PathBuf,
    /// Labelled reference clips (top-level `*.wav` of this folder), not rated.
    refs: Option<PathBuf>,
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
        let mut header = rows.next().unwrap_or_default();
        if header.len() < 3 + ANSWERS.len() || header[..3] != ["group", "clip", "sentence"] {
            return Err(format!("{}: not a listening sheet", path.display()));
        }
        for (c, _) in EXTRA {
            if !header.iter().any(|h| h == c) {
                header.push(c.to_string());
            }
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
            key: dir.join(KEY),
            refs: None,
            dir,
            header,
            rows,
            seen: mtime(&path),
        })
    }

    /// Read the key from `key` instead of `<dir>/key.json`.
    pub fn with_key(mut self, key: impl Into<PathBuf>) -> Self {
        self.key = key.into();
        self
    }

    /// Offer the top-level `*.wav` files of `dir` as reference clips.
    pub fn with_refs(mut self, dir: impl Into<PathBuf>) -> Self {
        self.refs = Some(dir.into());
        self
    }

    /// The reference clips as (file name, label), sorted. Only plain
    /// top-level `*.wav` files with a safe name; the label is the `label`
    /// of the stem's entry in `refs.json` if there is one, else the stem.
    pub fn ref_clips(&self) -> Vec<(String, String)> {
        let Some(dir) = &self.refs else {
            return Vec::new();
        };
        let labels: Value = fs::read_to_string(dir.join("refs.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null);
        let mut out: Vec<(String, String)> = fs::read_dir(dir)
            .map(|d| {
                d.flatten()
                    .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| safe_ref(n))
                    .map(|n| {
                        let stem = n.trim_end_matches(".wav").to_string();
                        let label = labels[&stem]["label"].as_str().unwrap_or(&stem).to_string();
                        (n, label)
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }

    fn ref_audio(&self, name: &str) -> Option<PathBuf> {
        let dir = self.refs.as_ref()?;
        self.ref_clips()
            .iter()
            .any(|(n, _)| n == name)
            .then(|| dir.join(name))
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
                "accent": r[self.col(EXTRA[0].0)],
                "suit": r[self.col(EXTRA[1].0)],
                "mood_right": r[self.col(EXTRA[2].0)],
                "pitch": r[self.col(EXTRA[3].0)],
            });
            match groups.last_mut() {
                Some((g, clips)) if *g == r[0] => clips.push(clip),
                _ => groups.push((r[0].clone(), vec![clip])),
            }
        }
        let done = self.done();
        json!({
            "groups": groups.into_iter().map(|(g, clips)| json!({
                // Display only: what the test script meant the line to sound like.
                "tone": tone_for(&g).map(|t| json!({"target": t})),
                "group": g,
                "moods": g.split('+').collect::<Vec<_>>(),
                "example": self.example(&g).is_some(),
                "clips": clips,
            })).collect::<Vec<_>>(),
            "refs": self.ref_clips().into_iter().map(|(name, label)| json!({
                "name": name,
                "label": label,
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
        let extra = [a.accent, a.suit, a.mood_right, a.pitch];
        for ((c, ok), v) in EXTRA.iter().zip(&extra) {
            check(v, ok, c)?;
        }
        let blend = a.group.contains('+');
        if !blend {
            // "na" only for a line with no tone target.
            if a.mood == "na" && tone_for(&a.group) == Some(None) {
            } else {
                check(&a.mood, &["0", "1", "2", "3"], "mood")?;
            }
        }
        if a.mood.len() > 120 || a.notes.len() > 2000 || a.mood.contains('\n') {
            return Err("answer too long".into());
        }
        let i = self
            .row(&a.group, &a.clip)
            .ok_or_else(|| format!("no clip {}/{}", a.group, a.clip))?;
        let cols = ANSWERS.map(|c| self.col(c));
        let extra_cols = EXTRA.map(|(c, _)| self.col(c));
        let row = &mut self.rows[i];
        for (c, v) in cols.into_iter().zip([a.voice, a.mood, a.clear, a.notes]) {
            row[c] = v;
        }
        for (c, v) in extra_cols.into_iter().zip(extra) {
            row[c] = v;
        }
        self.save()?;
        // A set with no key (e.g. a plain sheet) just has no results table.
        if self.done() == self.total() && self.key.is_file() {
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
        let path = &self.key;
        let mut key: BTreeMap<String, Value> = fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // The TTS test key: {"groups": {group: {letter: model}}}.
        if let Some(Value::Object(groups)) = key.remove("groups") {
            for (g, letters) in groups {
                for (l, model) in letters.as_object().into_iter().flatten() {
                    key.insert(format!("{g}/{l}"), json!({"condition": model}));
                }
            }
        }
        let mut by: BTreeMap<&str, Vec<(String, &Vec<String>)>> = BTreeMap::new();
        for r in &self.rows {
            if let Some(k) = key.get(&format!("{}/{}", r[0], r[1])) {
                let cond = k["condition"].as_str().unwrap_or("?").to_string();
                by.entry(&r[0]).or_default().push((cond, r));
            }
        }
        let mut out = String::from(
            "# What Brandon heard\n\n| group | condition | her voice | mood heard | words clear | accent | suit effect | mood right | pitch | notes |\n|---|---|---|---|---|---|---|---|---|---|\n",
        );
        for (group, mut items) in by {
            items.sort_by(|a, b| a.0.cmp(&b.0));
            for (cond, r) in items {
                let [v, m, c, n] = ANSWERS.map(|a| r[self.col(a)].as_str());
                let [ac, su, mr, pi] = EXTRA.map(|(a, _)| r[self.col(a)].as_str());
                out.push_str(&format!(
                    "| {group} | {cond} | {v} | {m} | {c} | {ac} | {su} | {mr} | {pi} | {n} |\n"
                ));
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
    #[serde(default)]
    pub accent: String,
    #[serde(default)]
    pub suit: String,
    #[serde(default)]
    pub mood_right: String,
    #[serde(default)]
    pub pitch: String,
}

/// A reference file name: `<letters, digits, - _>.wav`, nothing else.
fn safe_ref(name: &str) -> bool {
    name.strip_suffix(".wav").is_some_and(|stem| {
        !stem.is_empty()
            && stem.len() <= 64
            && stem
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    })
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
        (false, p) if p.starts_with("/ref/") => l
            .ref_audio(&crate::server::decode(&p["/ref/".len()..]))
            .map_or(Reply::NotFound, Reply::Audio),
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

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "TOPSECRET-MODEL";

    /// A set outside any `analysis/` tree, its key and refs elsewhere.
    fn setup() -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
        let t = tempfile::tempdir().unwrap();
        let (set, keydir, refs) = (
            t.path().join("set"),
            t.path().join("listen-key"),
            t.path().join("refs"),
        );
        for d in [&set, &keydir, &refs] {
            fs::create_dir_all(d).unwrap();
        }
        fs::create_dir_all(set.join("vT/L01")).unwrap();
        fs::write(set.join("vT/L01/A.wav"), b"RIFFa").unwrap();
        fs::write(set.join("vT/L01/B.wav"), b"RIFFb").unwrap();
        let mut rows = vec![["group", "clip", "sentence"]
            .into_iter()
            .chain(ANSWERS)
            .map(str::to_string)
            .collect::<Vec<_>>()];
        for c in ["A", "B"] {
            rows.push(vec![
                "vT/L01".into(),
                c.into(),
                "Hello.".into(),
                "".into(),
                "".into(),
                "".into(),
                "".into(),
            ]);
        }
        fs::write(set.join(SHEET), write_csv(&rows)).unwrap();
        fs::write(
            keydir.join(KEY),
            format!(r#"{{"groups": {{"vT/L01": {{"A": "{SECRET}", "B": "other"}}}}}}"#),
        )
        .unwrap();
        fs::write(refs.join("vT.wav"), b"RIFFref").unwrap();
        fs::write(refs.join("vP.wav"), b"RIFFref2").unwrap();
        fs::write(refs.join("refs.json"), r#"{"vT": {"label": "Tali ref"}}"#).unwrap();
        fs::create_dir_all(refs.join("sub")).unwrap();
        fs::write(refs.join("sub/x.wav"), b"RIFF").unwrap();
        fs::write(t.path().join("secret.wav"), b"RIFF").unwrap();
        (t, set, keydir.join(KEY), refs)
    }

    fn get(l: &mut Listen, url: &str) -> Reply {
        handle_listen(l, false, url, "")
    }

    fn text(r: Reply) -> String {
        match r {
            Reply::Json(v) => v.to_string(),
            Reply::Page(p) => p.to_string(),
            Reply::Audio(p) => p.display().to_string(),
            _ => String::new(),
        }
    }

    #[test]
    fn tone_comes_from_the_test_script_for_display() {
        assert_eq!(tone_for("vT/L05"), Some(Some("angry".into())));
        assert_eq!(tone_for("vP/L05_s2"), Some(Some("angry".into())));
        assert_eq!(tone_for("vT/L01"), Some(Some("neutral".into())));
        for id in ["L09", "P01", "F01", "F02", "F03"] {
            assert_eq!(tone_for(&format!("vT/{id}")), Some(None), "{id}");
        }
        assert_eq!(tone_for("happy+sad"), None);
        assert_eq!(tone_for("vT/Z99"), None);
    }

    #[test]
    fn session_carries_tone_and_na_mood_only_for_no_target_lines() {
        let (_t, set, _key, _refs) = setup();
        let mut l = Listen::load(&set).unwrap();
        // Old sheet (no tone column) still loads; tone comes from the line id.
        assert_eq!(l.session()["groups"][0]["tone"]["target"], json!("neutral"));
        l.rows[0][0] = "vT/L09".into();
        let s = l.session();
        assert!(s["groups"][0]["tone"]["target"].is_null());
        assert!(l.session()["groups"][0]["tone"].is_object());
        let ok = Answer { group: "vT/L09".into(), clip: "A".into(), voice: "y".into(), mood: "na".into(), clear: "y".into(), notes: String::new(), accent: String::new(), suit: String::new(), mood_right: String::new(), pitch: String::new() };
        assert!(l.answer(ok).is_ok());
        let bad = Answer { group: "vT/L09".into(), clip: "B".into(), voice: String::new(), mood: "na".into(), clear: String::new(), notes: String::new(), accent: String::new(), suit: String::new(), mood_right: String::new(), pitch: String::new() };
        let mut l2 = Listen::load(&set).unwrap();
        let bad = Answer { group: "vT/L01".into(), clip: "B".into(), ..bad };
        assert!(l2.answer(bad).is_err());
    }

    #[test]
    fn page_has_instructions_and_ref_caption() {
        assert!(LISTEN_PAGE.contains("Voice reference - same for every line"));
        assert!(LISTEN_PAGE.contains("No tone target: rate voice and clarity only"));
        assert!(LISTEN_PAGE.contains("no mood instruction"));
    }

    #[test]
    fn key_stays_hidden_until_done_then_unblinds() {
        let (_t, set, key, refs) = setup();
        let mut l = Listen::load(&set).unwrap().with_key(&key).with_refs(&refs);
        let keystr = key.display().to_string();
        let leaks = |l: &mut Listen| {
            let mut all = String::new();
            for u in ["/", "/api/session", "/audio?g=vT/L01&c=A", "/ref/vT.wav"] {
                all += &text(get(l, u));
            }
            all.contains(SECRET) || all.contains(&keystr) || all.contains("listen-key")
        };
        assert!(!leaks(&mut l));
        for u in [
            "/key.json",
            "/listen-key/key.json",
            "/../listen-key/key.json",
            "/ref/key.json",
            "/audio?g=..&c=key",
        ] {
            assert!(matches!(get(&mut l, u), Reply::NotFound), "{u}");
        }
        let ans = |c: &str| {
            format!(r#"{{"group":"vT/L01","clip":"{c}","voice":"y","mood":"1","clear":"y"}}"#)
        };
        let r = handle_listen(&mut l, true, "/api/answer", &ans("A"));
        assert!(!text(r).contains(SECRET));
        assert!(!leaks(&mut l));
        let r = handle_listen(&mut l, true, "/api/answer", &ans("B"));
        let done = text(r);
        assert!(done.contains(SECRET), "unblinds once complete");
        assert!(!done.contains(&keystr));
        assert!(fs::read_to_string(set.join(RESULTS))
            .unwrap()
            .contains(SECRET));
        // ratings persisted in the sheet
        assert!(fs::read_to_string(set.join(SHEET))
            .unwrap()
            .contains("y,1,y"));
    }

    #[test]
    fn without_a_key_the_set_still_works() {
        let (_t, set, _key, _refs) = setup();
        let mut l = Listen::load(&set).unwrap();
        for c in ["A", "B"] {
            let b =
                format!(r#"{{"group":"vT/L01","clip":"{c}","voice":"y","mood":"1","clear":"y"}}"#);
            assert!(matches!(
                handle_listen(&mut l, true, "/api/answer", &b),
                Reply::Json(_)
            ));
        }
        assert!(!set.join(RESULTS).exists());
    }

    #[test]
    fn refs_are_listed_with_labels_and_served_read_only() {
        let (_t, set, key, refs) = setup();
        let mut l = Listen::load(&set).unwrap().with_key(&key).with_refs(&refs);
        let s: Value = serde_json::from_str(&text(get(&mut l, "/api/session"))).unwrap();
        assert_eq!(
            s["refs"],
            json!([{"name": "vP.wav", "label": "vP"}, {"name": "vT.wav", "label": "Tali ref"}])
        );
        assert!(matches!(get(&mut l, "/ref/vT.wav"), Reply::Audio(p) if p == refs.join("vT.wav")));
        for bad in [
            "/ref/../secret.wav",
            "/ref/%2e%2e/secret.wav",
            "/ref/..%2fsecret.wav",
            "/ref/sub/x.wav",
            "/ref/sub%2fx.wav",
            "/ref/refs.json",
            "/ref/",
            "/ref/nope.wav",
            "/ref/vT.wav/..",
            "/ref/%2fetc%2fpasswd",
        ] {
            assert!(matches!(get(&mut l, bad), Reply::NotFound), "{bad}");
        }
        assert!(matches!(
            handle_listen(&mut l, true, "/ref/vT.wav", ""),
            Reply::NotFound
        ));
        // no --refs: nothing served, nothing listed
        let mut none = Listen::load(&set).unwrap();
        assert!(matches!(get(&mut none, "/ref/vT.wav"), Reply::NotFound));
        assert_eq!(none.session()["refs"], json!([]));
    }
}
