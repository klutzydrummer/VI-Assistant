//! The capture's summary store: per line and captured layer, the mean of
//! the residual stream over the positions `vi chat` steers (`line_mean`)
//! and over the first few of them (`line_head`), as F16, plus per line the
//! token counts, the log-probability of her line and the residual norms.
//!
//! ```text
//! <store>/capture.json   what was captured, from which model and messages
//! <store>/lines.jsonl    one record per captured line, in capture order
//! <store>/line_mean.f16  lines x layers x n_embd
//! <store>/line_head.f16  lines x layers x n_embd
//! <store>/norms.f32      lines x layers: mean residual norm over line_mean's positions
//! ```
//!
//! `lines.jsonl` is the checkpoint: a line's record is appended only after
//! its rows are written, so a store left by a lost worker resumes after
//! its last record. No game text is stored, only ids, hashes and numbers.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::f16;

pub const META: &str = "capture.json";
pub const LINES: &str = "lines.jsonl";
pub const MEAN: &str = "line_mean.f16";
pub const HEAD: &str = "line_head.f16";
pub const NORMS: &str = "norms.f32";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub version: u32,
    pub model_file: String,
    pub model_sha256: String,
    pub architecture: String,
    /// The llama.cpp the capture ran on.
    pub llama_cpp: String,
    /// Where it ran ([`crate::llama::Llama::backend`]). GPU and CPU sums
    /// differ by rounding, so a store doesn't resume on another backend.
    /// Stores from before this field were all captured on the CPU.
    #[serde(default = "cpu")]
    pub backend: String,
    pub n_embd: usize,
    pub n_layer: usize,
    /// Captured layers, in store order: `1..=n_layer-2`.
    pub layers: Vec<usize>,
    /// The per-layer tensor read, `<layer_tensor>-<il>`.
    pub layer_tensor: String,
    pub chat_template_sha256: String,
    pub messages_sha256: String,
    /// Tokens of the shared persona prefix, decoded once and reused.
    pub prefix_tokens: usize,
    /// Her first tokens averaged into `line_head` (after the lead-in).
    pub head_tokens: usize,
    /// Her line sat behind an empty think block, and the summaries pool
    /// from `</think>` on; else from the last prompt token.
    #[serde(default)]
    pub think: bool,
    pub n_lines_total: usize,
    #[serde(default)]
    pub complete: bool,
    /// Preflight results on this model, when it ran first.
    #[serde(default)]
    pub preflight: Option<serde_json::Value>,
}

fn cpu() -> String {
    "cpu".into()
}

impl Meta {
    /// The part that must match for a capture to resume into a store.
    fn same_capture(&self, other: &Meta) -> bool {
        let strip = |m: &Meta| Meta {
            complete: false,
            preflight: None,
            ..m.clone()
        };
        strip(self) == strip(other)
    }

    pub fn row_floats(&self) -> usize {
        self.layers.len() * self.n_embd
    }

    /// Index of layer `il` in the store, if captured.
    pub fn slot(&self, il: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == il)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineRecord {
    pub line_id: String,
    /// SHA-256 of her line's text, to catch a relabelled id.
    pub text_sha256: String,
    pub n_prompt_tokens: usize,
    pub n_line_tokens: usize,
    /// Mean log-probability of her line's tokens, unsteered.
    pub logprob: f64,
}

pub struct StoreWriter {
    dir: PathBuf,
    meta: Meta,
    mean: File,
    head: File,
    norms: File,
    lines: File,
    done: Vec<LineRecord>,
}

fn io(path: &Path, e: std::io::Error) -> String {
    format!("{}: {e}", path.display())
}

fn write_meta(dir: &Path, meta: &Meta) -> Result<(), String> {
    let path = dir.join(META);
    let tmp = dir.join(format!("{META}.tmp"));
    let text = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    fs::write(&tmp, text + "\n").map_err(|e| io(&tmp, e))?;
    fs::rename(&tmp, &path).map_err(|e| io(&path, e))
}

fn read_records(path: &Path) -> Result<Vec<LineRecord>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path).map_err(|e| io(path, e))?;
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| io(path, e))?;
        if line.trim().is_empty() {
            continue;
        }
        // A torn last line (killed mid-write) is not a checkpoint.
        match serde_json::from_str(&line) {
            Ok(r) => out.push(r),
            Err(_) => break,
        }
    }
    Ok(out)
}

impl StoreWriter {
    /// Start a store in `dir`, or resume one made by the same capture.
    pub fn open(dir: &Path, meta: Meta) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
        let meta_path = dir.join(META);
        let mut done = Vec::new();
        if meta_path.exists() {
            let old: Meta = serde_json::from_str(
                &fs::read_to_string(&meta_path).map_err(|e| io(&meta_path, e))?,
            )
            .map_err(|e| format!("{}: {e}", meta_path.display()))?;
            if old.backend != meta.backend
                && old.same_capture(&Meta {
                    backend: old.backend.clone(),
                    ..meta.clone()
                })
            {
                return Err(format!(
                    "{} was captured on {}, this run is on {}; resume it there (--gpu-layers 0 for the CPU) \
                     or choose another --store",
                    dir.display(),
                    old.backend,
                    meta.backend
                ));
            }
            if !old.same_capture(&meta) {
                return Err(format!(
                    "{} holds a capture of another model, layer set or message file; \
                     choose another --store or delete it",
                    dir.display()
                ));
            }
            done = read_records(&dir.join(LINES))?;
        }
        let row = meta.row_floats() as u64;
        let n = done.len() as u64;
        let open = |name: &str, len: u64| -> Result<File, String> {
            let path = dir.join(name);
            let f = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&path)
                .map_err(|e| io(&path, e))?;
            if f.metadata().map_err(|e| io(&path, e))?.len() < len {
                return Err(format!(
                    "{} is shorter than its checkpoint says; delete the store and capture again",
                    path.display()
                ));
            }
            f.set_len(len).map_err(|e| io(&path, e))?;
            Ok(f)
        };
        let mut mean = open(MEAN, n * row * 2)?;
        let mut head = open(HEAD, n * row * 2)?;
        let mut norms = open(NORMS, n * meta.layers.len() as u64 * 4)?;
        for f in [&mut mean, &mut head, &mut norms] {
            f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        }
        // Rewrite the records file without any torn tail.
        let lines_path = dir.join(LINES);
        let mut text = String::new();
        for r in &done {
            text.push_str(&serde_json::to_string(r).unwrap());
            text.push('\n');
        }
        fs::write(&lines_path, text).map_err(|e| io(&lines_path, e))?;
        let lines = OpenOptions::new()
            .append(true)
            .open(&lines_path)
            .map_err(|e| io(&lines_path, e))?;
        let mut meta = meta;
        meta.complete = false;
        write_meta(dir, &meta)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            meta,
            mean,
            head,
            norms,
            lines,
            done,
        })
    }

    pub fn done(&self) -> &[LineRecord] {
        &self.done
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Append one line: its rows first, then its record (the checkpoint).
    pub fn push(
        &mut self,
        record: LineRecord,
        mean: &[f32],
        head: &[f32],
        norms: &[f32],
    ) -> Result<(), String> {
        let row = self.meta.row_floats();
        if mean.len() != row || head.len() != row || norms.len() != self.meta.layers.len() {
            return Err("capture row has the wrong size for the store".into());
        }
        let half = |xs: &[f32]| -> Vec<u8> {
            xs.iter()
                .flat_map(|&x| f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let err = |e: std::io::Error| format!("{}: {e}", self.dir.display());
        self.mean.write_all(&half(mean)).map_err(err)?;
        self.head.write_all(&half(head)).map_err(err)?;
        let nb: Vec<u8> = norms.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.norms.write_all(&nb).map_err(err)?;
        for f in [&self.mean, &self.head, &self.norms] {
            f.sync_data().map_err(err)?;
        }
        let mut text = serde_json::to_string(&record).unwrap();
        text.push('\n');
        self.lines.write_all(text.as_bytes()).map_err(err)?;
        self.lines.sync_data().map_err(err)?;
        self.done.push(record);
        Ok(())
    }

    /// Mark the store complete, with the preflight results if any.
    pub fn finish(mut self, preflight: Option<serde_json::Value>) -> Result<Meta, String> {
        if self.done.len() != self.meta.n_lines_total {
            return Err(format!(
                "store has {} of {} lines",
                self.done.len(),
                self.meta.n_lines_total
            ));
        }
        self.meta.complete = true;
        if preflight.is_some() {
            self.meta.preflight = preflight;
        }
        write_meta(&self.dir, &self.meta)?;
        Ok(self.meta)
    }
}

/// A finished store, read a line and a run of layers at a time.
pub struct Store {
    pub dir: PathBuf,
    pub meta: Meta,
    pub lines: Vec<LineRecord>,
    /// lines x layers.
    pub norms: Vec<f32>,
    mean: File,
    head: File,
    buf: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Summary {
    Mean,
    Head,
}

impl Store {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let meta_path = dir.join(META);
        let meta: Meta =
            serde_json::from_str(&fs::read_to_string(&meta_path).map_err(|e| io(&meta_path, e))?)
                .map_err(|e| format!("{}: {e}", meta_path.display()))?;
        if !meta.complete {
            return Err(format!(
                "{} is not a finished capture; run the capture again to resume it",
                dir.display()
            ));
        }
        let lines = read_records(&dir.join(LINES))?;
        let row = meta.row_floats() as u64;
        let n = lines.len() as u64;
        let open = |name: &str, len: u64| -> Result<File, String> {
            let path = dir.join(name);
            let f = File::open(&path).map_err(|e| io(&path, e))?;
            if f.metadata().map_err(|e| io(&path, e))?.len() != len {
                return Err(format!("{} has the wrong size", path.display()));
            }
            Ok(f)
        };
        let mean = open(MEAN, n * row * 2)?;
        let head = open(HEAD, n * row * 2)?;
        let mut nf = open(NORMS, n * meta.layers.len() as u64 * 4)?;
        let mut raw = Vec::new();
        nf.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let norms = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(Self {
            dir: dir.to_path_buf(),
            meta,
            lines,
            norms,
            mean,
            head,
            buf: Vec::new(),
        })
    }

    /// Read line `i`'s summary for store layers `slots`, as F32 into `out`
    /// (`slots.len() * n_embd` floats).
    pub fn read(
        &mut self,
        which: Summary,
        i: usize,
        slots: std::ops::Range<usize>,
        out: &mut [f32],
    ) -> Result<(), String> {
        let e = self.meta.n_embd;
        let n = slots.len() * e;
        assert_eq!(out.len(), n);
        let offset = ((i * self.meta.row_floats() + slots.start * e) * 2) as u64;
        self.buf.resize(n * 2, 0);
        let f = match which {
            Summary::Mean => &mut self.mean,
            Summary::Head => &mut self.head,
        };
        f.seek(SeekFrom::Start(offset))
            .and_then(|_| f.read_exact(&mut self.buf))
            .map_err(|e| format!("{}: {e}", self.dir.display()))?;
        for (o, c) in out.iter_mut().zip(self.buf.as_chunks::<2>().0) {
            *o = f16::to_f32(u16::from_le_bytes(*c));
        }
        Ok(())
    }

    /// [`Store::read`] without converting: the raw F16 bits.
    pub fn read_raw(
        &mut self,
        which: Summary,
        i: usize,
        slots: std::ops::Range<usize>,
        out: &mut [u16],
    ) -> Result<(), String> {
        let e = self.meta.n_embd;
        assert_eq!(out.len(), slots.len() * e);
        let offset = ((i * self.meta.row_floats() + slots.start * e) * 2) as u64;
        self.buf.resize(out.len() * 2, 0);
        let f = match which {
            Summary::Mean => &mut self.mean,
            Summary::Head => &mut self.head,
        };
        f.seek(SeekFrom::Start(offset))
            .and_then(|_| f.read_exact(&mut self.buf))
            .map_err(|e| format!("{}: {e}", self.dir.display()))?;
        for (o, c) in out.iter_mut().zip(self.buf.as_chunks::<2>().0) {
            *o = u16::from_le_bytes(*c);
        }
        Ok(())
    }

    /// Identifies the store's contents (its metadata and line records).
    pub fn hash(&self) -> Result<String, String> {
        let mut h = Sha256::new();
        for name in [META, LINES] {
            let path = self.dir.join(name);
            h.update(fs::read(&path).map_err(|e| io(&path, e))?);
        }
        Ok(crate::registry::hex(&h.finalize()))
    }

    /// Mean residual norm per store layer, over every line.
    pub fn mean_norms(&self) -> Vec<f64> {
        let k = self.meta.layers.len();
        let mut sum = vec![0.0f64; k];
        for row in self.norms.chunks(k) {
            for (s, x) in sum.iter_mut().zip(row) {
                *s += f64::from(*x);
            }
        }
        let n = self.lines.len().max(1) as f64;
        sum.iter().map(|s| s / n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(n_lines: usize) -> Meta {
        Meta {
            version: 1,
            model_file: "m.gguf".into(),
            model_sha256: "ab".into(),
            architecture: "llama".into(),
            llama_cpp: "test".into(),
            backend: "cpu".into(),
            n_embd: 3,
            n_layer: 4,
            layers: vec![1, 2],
            layer_tensor: "l_out".into(),
            chat_template_sha256: "t".into(),
            messages_sha256: "m".into(),
            prefix_tokens: 5,
            head_tokens: 8,
            think: true,
            n_lines_total: n_lines,
            complete: false,
            preflight: None,
        }
    }

    fn record(id: &str) -> LineRecord {
        LineRecord {
            line_id: id.into(),
            text_sha256: "x".into(),
            n_prompt_tokens: 10,
            n_line_tokens: 4,
            logprob: -1.5,
        }
    }

    #[test]
    fn round_trips_resumes_and_refuses_other_captures() {
        let dir = tempfile::tempdir().unwrap();
        let row = |k: f32| (0..6).map(|i| k + i as f32 * 0.25).collect::<Vec<_>>();
        let mut w = StoreWriter::open(dir.path(), meta(3)).unwrap();
        w.push(record("a"), &row(1.0), &row(2.0), &[3.0, 4.0])
            .unwrap();
        assert!(w.push(record("b"), &row(1.0), &[0.0], &[3.0, 4.0]).is_err());
        drop(w);
        // A torn record after "a" is ignored on resume.
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.path().join(LINES))
            .unwrap();
        f.write_all(b"{\"line_id\": \"b\", \"te").unwrap();
        drop(f);
        assert!(Store::open(dir.path()).is_err(), "not finished");

        let mut w = StoreWriter::open(dir.path(), meta(3)).unwrap();
        assert_eq!(w.done().len(), 1);
        w.push(record("b"), &row(5.0), &row(6.0), &[7.0, 8.0])
            .unwrap();
        w.push(record("c"), &row(-1.0), &row(-2.0), &[1.0, 2.0])
            .unwrap();
        w.finish(None).unwrap();

        let mut s = Store::open(dir.path()).unwrap();
        assert_eq!(
            s.lines
                .iter()
                .map(|r| r.line_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        let mut out = vec![0.0; 3];
        s.read(Summary::Mean, 1, 1..2, &mut out).unwrap();
        assert_eq!(out, row(5.0)[3..]);
        let mut out = vec![0.0; 6];
        s.read(Summary::Head, 2, 0..2, &mut out).unwrap();
        assert_eq!(out, row(-2.0));
        assert_eq!(s.mean_norms(), vec![11.0 / 3.0, 14.0 / 3.0]);

        let mut other = meta(3);
        other.model_sha256 = "cd".into();
        assert!(StoreWriter::open(dir.path(), other).is_err());
        let mut gpu = meta(3);
        gpu.backend = "Vulkan0 (test)".into();
        let err = StoreWriter::open(dir.path(), gpu).err().unwrap();
        assert!(err.contains("captured on cpu"), "{err}");
    }

    #[test]
    fn stores_from_before_the_backend_field_were_cpu() {
        let mut v = serde_json::to_value(meta(3)).unwrap();
        v.as_object_mut().unwrap().remove("backend");
        let m: Meta = serde_json::from_value(v).unwrap();
        assert_eq!(m.backend, "cpu");
    }
}
