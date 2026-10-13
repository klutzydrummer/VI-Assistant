//! Runs pipeline stages, skipping any whose inputs haven't changed.
//!
//! A run's key hashes everything that went into it: the stage's command,
//! version and settings, the files it lists, and the output hash of each
//! upstream stage. Using upstream *output* hashes rather than keys means a
//! rerun that produces identical output doesn't ripple downstream.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::hash::{self, HashCache};
use crate::pipeline::{Pipeline, CORPUS_PREFIX};
use crate::{Error, IoContext, FORMAT};

pub const MANIFEST: &str = "manifest.json";
pub const LOG: &str = "log.txt";
pub const OUT: &str = "out";
pub const CURRENT: &str = "current";
const TMP_PREFIX: &str = ".tmp-";
/// Where a run keeps work it can resume from if it fails part way (a lost
/// GPU worker, say); `VI_CHECKPOINT`. Survives failed runs of the same key,
/// deleted once the run succeeds.
const CHECKPOINT_PREFIX: &str = ".checkpoint-";

/// The current output folder of `stage` in a corpus, for readers outside
/// the runner such as the review page.
pub fn current_out(corpus: &Path, stage: &str) -> Option<PathBuf> {
    let dir = corpus.join("stages").join(stage);
    let key = fs::read_to_string(dir.join(CURRENT)).ok()?;
    let out = dir.join(key.trim()).join(OUT);
    out.is_dir().then_some(out)
}

/// What `manifest.json` records about one successful run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub stage: String,
    pub key: String,
    pub output_hash: String,
    pub version: u32,
    pub command: Vec<String>,
    pub settings: Value,
    /// Upstream stage -> its output hash.
    pub inputs: BTreeMap<String, String>,
    /// `files` entry -> content hash.
    pub files: BTreeMap<String, String>,
    /// Path under `out/` -> file hash.
    pub outputs: BTreeMap<String, String>,
    pub started: String,
    pub finished: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Already run with exactly these inputs.
    Cached { key: String },
    /// Inputs changed (or it never ran), so it will run.
    Stale { key: String },
    /// An upstream stage has to run first before the key is known.
    Waiting { on: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Cached,
    Ran,
}

#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage: String,
    pub key: String,
    pub output_hash: String,
    pub outcome: Outcome,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    pub runs_removed: usize,
    pub objects_removed: usize,
    pub bytes_freed: u64,
}

pub struct Runner {
    pipeline: Pipeline,
    cache: HashCache,
    vi_bin: PathBuf,
    /// Echo job output to stderr as it runs, besides the log file.
    pub echo: bool,
}

impl Runner {
    /// `vi_bin` is what a stage command of `vi` runs.
    pub fn new(mut pipeline: Pipeline, vi_bin: PathBuf) -> Result<Self, Error> {
        fs::create_dir_all(&pipeline.corpus).at(&pipeline.corpus)?;
        pipeline.corpus = fs::canonicalize(&pipeline.corpus).at(&pipeline.corpus)?;
        pipeline.dir = fs::canonicalize(&pipeline.dir).at(&pipeline.dir)?;
        let cache = HashCache::load(pipeline.corpus.join(".hash-cache.json"));
        Ok(Runner {
            pipeline,
            cache,
            vi_bin,
            echo: false,
        })
    }

    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub fn stage_dir(&self, stage: &str) -> PathBuf {
        self.pipeline.corpus.join("stages").join(stage)
    }

    pub fn run_dir(&self, stage: &str, key: &str) -> PathBuf {
        self.stage_dir(stage).join(key)
    }

    /// Key of the stage's latest successful run, if any.
    pub fn current(&self, stage: &str) -> Option<String> {
        let text = fs::read_to_string(self.stage_dir(stage).join(CURRENT)).ok()?;
        Some(text.trim().to_string()).filter(|k| !k.is_empty())
    }

    pub fn manifest(&self, stage: &str, key: &str) -> Option<Manifest> {
        let bytes = fs::read(self.run_dir(stage, key).join(MANIFEST)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// The key a stage would run under, given its upstream output hashes.
    fn plan(&mut self, name: &str, inputs: BTreeMap<String, String>) -> Result<Planned, Error> {
        let stage = self.pipeline.stages[name].clone();
        let mut files = BTreeMap::new();
        for entry in &stage.files {
            let path = self.pipeline.resolve_file(entry);
            if !path.exists() && entry.starts_with(CORPUS_PREFIX) {
                // Corpus files such as reviews.jsonl appear later; their
                // arrival then changes the key.
                files.insert(entry.clone(), "missing".into());
                continue;
            }
            if !path.exists() {
                return Err(Error::Stage {
                    stage: name.into(),
                    message: format!("listed file {} does not exist", path.display()),
                });
            }
            files.insert(entry.clone(), self.cache.path(&path)?);
        }
        let settings = serde_json::to_value(&stage.settings).expect("TOML converts to JSON");
        // serde_json maps are sorted, so this is canonical.
        let doc = json!({
            "format": FORMAT,
            "stage": name,
            "version": stage.version,
            "command": stage.command,
            "settings": settings,
            "inputs": inputs,
            "files": files,
        });
        let key = hash::sha256_bytes(&serde_json::to_vec(&doc).expect("JSON serializes"));
        Ok(Planned {
            key,
            settings,
            inputs,
            files,
        })
    }

    /// Upstream output hashes for `name`, or the first upstream stage whose
    /// output isn't known yet.
    fn upstream(
        &self,
        name: &str,
        outputs: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, String>, String> {
        self.pipeline.stages[name]
            .inputs
            .iter()
            .map(|i| match outputs.get(i) {
                Some(h) => Ok((i.clone(), h.clone())),
                None => Err(i.clone()),
            })
            .collect()
    }

    /// Every stage's status, without running anything.
    pub fn status(&mut self) -> Result<Vec<(String, Status)>, Error> {
        let order = self
            .pipeline
            .order_all()
            .map_err(|message| Error::Invalid {
                path: self.pipeline.dir.clone(),
                message,
            })?;
        let mut outputs = BTreeMap::new();
        let mut out = Vec::new();
        for name in order {
            let status = match self.upstream(&name, &outputs) {
                Err(on) => Status::Waiting { on },
                Ok(inputs) => {
                    let key = self.plan(&name, inputs)?.key;
                    match self.manifest(&name, &key) {
                        Some(m) => {
                            outputs.insert(name.clone(), m.output_hash);
                            Status::Cached { key }
                        }
                        None => Status::Stale { key },
                    }
                }
            };
            out.push((name, status));
        }
        self.cache.save()?;
        Ok(out)
    }

    /// Runs `target` (or every stage) and whatever it depends on. `force`
    /// reruns the target (or every stage) even when cached.
    pub fn run(
        &mut self,
        target: Option<&str>,
        force: bool,
        mut progress: impl FnMut(&str),
    ) -> Result<Vec<StageResult>, Error> {
        let _lock = Lock::take(&self.pipeline.corpus)?;
        let order = self.order(target)?;
        let mut outputs = BTreeMap::new();
        let mut results = Vec::new();
        for name in order {
            let inputs = self
                .upstream(&name, &outputs)
                .expect("run order puts inputs first");
            let planned = self.plan(&name, inputs)?;
            self.cache.save()?;
            let forced = force && target.is_none_or(|t| t == name);
            let (output_hash, outcome) = match self.manifest(&name, &planned.key) {
                Some(m) if !forced => {
                    progress(&format!("{name}: cached ({})", short(&planned.key)));
                    (m.output_hash, Outcome::Cached)
                }
                _ => {
                    progress(&format!("{name}: running ({})", short(&planned.key)));
                    let m = self.execute(&name, &planned)?;
                    progress(&format!(
                        "{name}: done, {} files ({})",
                        m.outputs.len(),
                        short(&m.output_hash)
                    ));
                    (m.output_hash, Outcome::Ran)
                }
            };
            self.set_current(&name, &planned.key)?;
            outputs.insert(name.clone(), output_hash.clone());
            results.push(StageResult {
                stage: name,
                key: planned.key,
                output_hash,
                outcome,
            });
        }
        Ok(results)
    }

    fn order(&self, target: Option<&str>) -> Result<Vec<String>, Error> {
        self.pipeline
            .order(target)
            .map_err(|message| Error::Invalid {
                path: self.pipeline.dir.clone(),
                message,
            })
    }

    fn execute(&mut self, name: &str, planned: &Planned) -> Result<Manifest, Error> {
        let stage = self.pipeline.stages[name].clone();
        let stage_dir = self.stage_dir(name);
        let tmp = stage_dir.join(format!("{TMP_PREFIX}{}", planned.key));
        if tmp.exists() {
            fs::remove_dir_all(&tmp).at(&tmp)?;
        }
        let out = tmp.join(OUT);
        fs::create_dir_all(&out).at(&out)?;
        let checkpoint = stage_dir.join(format!("{CHECKPOINT_PREFIX}{}", planned.key));
        fs::create_dir_all(&checkpoint).at(&checkpoint)?;
        let settings_path = tmp.join("settings.json");
        fs::write(
            &settings_path,
            serde_json::to_vec_pretty(&planned.settings).unwrap(),
        )
        .at(&settings_path)?;

        let program = match stage.command[0].as_str() {
            "vi" => self.vi_bin.clone(),
            p if p.starts_with("./") || p.starts_with("../") => self.pipeline.dir.join(p),
            p => PathBuf::from(p),
        };
        let mut cmd = Command::new(&program);
        cmd.args(&stage.command[1..])
            .current_dir(&self.pipeline.dir)
            .env("VI_STAGE", name)
            .env("VI_OUT", &out)
            .env("VI_CHECKPOINT", &checkpoint)
            .env("VI_SETTINGS", &settings_path)
            .env("VI_CORPUS", &self.pipeline.corpus)
            .env("VI_BIN", &self.vi_bin)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for input in planned.inputs.keys() {
            let key = self.current(input).expect("upstream ran first");
            let var = format!("VI_INPUT_{}", input.to_uppercase().replace('-', "_"));
            cmd.env(var, self.run_dir(input, &key).join(OUT));
        }

        let log_path = tmp.join(LOG);
        let started = now();
        let status = {
            let log = Arc::new(Mutex::new(fs::File::create(&log_path).at(&log_path)?));
            let mut child = cmd.spawn().map_err(|e| Error::Stage {
                stage: name.into(),
                message: format!("could not start {}: {e}", program.display()),
            })?;
            let pumps = [
                pump(child.stdout.take().unwrap(), log.clone(), self.echo),
                pump(child.stderr.take().unwrap(), log.clone(), self.echo),
            ];
            let status = child.wait().at(&program)?;
            for p in pumps {
                let _ = p.join();
            }
            status
        };
        if !status.success() {
            return Err(Error::Stage {
                stage: name.into(),
                message: format!(
                    "{status}; partial output kept in {}, and a rerun resumes from {}\n{}",
                    tmp.display(),
                    checkpoint.display(),
                    tail(&log_path, 20)
                ),
            });
        }

        let mut files = BTreeMap::new();
        for rel in hash::walk(&out)? {
            let path = out.join(&rel);
            let sha = hash::sha256_file(&path)?;
            self.store(&path, &sha)?;
            files.insert(rel, sha);
        }
        let manifest = Manifest {
            format: FORMAT,
            stage: name.into(),
            key: planned.key.clone(),
            output_hash: hash::listing_hash(&files),
            version: stage.version,
            command: stage.command.clone(),
            settings: planned.settings.clone(),
            inputs: planned.inputs.clone(),
            files: planned.files.clone(),
            outputs: files,
            started,
            finished: now(),
        };
        let manifest_path = tmp.join(MANIFEST);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .at(&manifest_path)?;

        let dest = self.run_dir(name, &planned.key);
        if dest.exists() {
            fs::remove_dir_all(&dest).at(&dest)?;
        }
        fs::rename(&tmp, &dest).at(&dest)?;
        fs::remove_dir_all(&checkpoint).at(&checkpoint)?;
        Ok(manifest)
    }

    /// Moves an output file into the object store and hard-links it back,
    /// so identical outputs share one copy. Objects are read-only, which
    /// also protects the output (same inode) from accidental edits.
    fn store(&self, path: &Path, sha: &str) -> Result<(), Error> {
        let objects = self.pipeline.corpus.join("objects");
        fs::create_dir_all(&objects).at(&objects)?;
        let object = objects.join(sha);
        if object.exists() {
            fs::remove_file(path).at(path)?;
            if fs::hard_link(&object, path).is_err() {
                fs::copy(&object, path).at(path)?;
            }
        } else {
            if fs::hard_link(path, &object).is_err() {
                fs::copy(path, &object).at(&object)?;
            }
            let mut perms = fs::metadata(&object).at(&object)?.permissions();
            perms.set_readonly(true);
            fs::set_permissions(&object, perms).at(&object)?;
        }
        Ok(())
    }

    fn set_current(&self, stage: &str, key: &str) -> Result<(), Error> {
        let path = self.stage_dir(stage).join(CURRENT);
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, format!("{key}\n")).at(&tmp)?;
        fs::rename(&tmp, &path).at(&path)
    }

    /// Deletes runs that aren't current, failed partial runs, and objects
    /// no current run uses.
    pub fn gc(&mut self) -> Result<GcReport, Error> {
        let _lock = Lock::take(&self.pipeline.corpus)?;
        let mut report = GcReport::default();
        let mut keep = BTreeSet::new();
        let stages = self.pipeline.corpus.join("stages");
        let stage_names: Vec<String> = match fs::read_dir(&stages) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect(),
            Err(_) => Vec::new(),
        };
        for stage in stage_names {
            let current = self.current(&stage);
            if let Some(m) = current.as_deref().and_then(|k| self.manifest(&stage, k)) {
                keep.extend(m.outputs.into_values());
            }
            let dir = self.stage_dir(&stage);
            for entry in fs::read_dir(&dir).at(&dir)? {
                let entry = entry.at(&dir)?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if entry.path().is_dir() && current.as_deref() != Some(name.as_str()) {
                    fs::remove_dir_all(entry.path()).at(entry.path())?;
                    report.runs_removed += 1;
                }
            }
        }
        let objects = self.pipeline.corpus.join("objects");
        if let Ok(entries) = fs::read_dir(&objects) {
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !keep.contains(&name) {
                    report.bytes_freed += entry.metadata().map_or(0, |m| m.len());
                    fs::remove_file(entry.path()).at(entry.path())?;
                    report.objects_removed += 1;
                }
            }
        }
        Ok(report)
    }
}

struct Planned {
    key: String,
    settings: Value,
    inputs: BTreeMap<String, String>,
    files: BTreeMap<String, String>,
}

/// Keeps two runs from writing the same corpus at once.
struct Lock(PathBuf);

impl Lock {
    fn take(corpus: &Path) -> Result<Self, Error> {
        let path = corpus.join(".lock");
        match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => {
                let _ = writeln!(f, "{}", std::process::id());
                Ok(Lock(path))
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(Error::Invalid {
                path: path.clone(),
                message: "another vi dataset command is using this corpus; delete this file if none is running".into(),
            }),
            Err(e) => Err(Error::Io { path, source: e }),
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn pump(
    mut from: impl Read + Send + 'static,
    log: Arc<Mutex<fs::File>>,
    echo: bool,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while let Ok(n) = from.read(&mut buf) {
            if n == 0 {
                break;
            }
            let _ = log.lock().unwrap().write_all(&buf[..n]);
            if echo {
                let _ = std::io::stderr().write_all(&buf[..n]);
            }
        }
    })
}

fn tail(path: &Path, lines: usize) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

fn now() -> String {
    humantime::format_rfc3339_seconds(SystemTime::now()).to_string()
}

fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}
