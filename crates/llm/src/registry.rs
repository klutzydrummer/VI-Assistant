//! The dialogue models vectors can be built for, one entry each in
//! `models.toml` at the repository root:
//!
//! ```toml
//! ["qwen3.8-27b-q8"]          # quoted: ids contain dots
//! repo = "ggml-org/Qwen3.8-27B-GGUF"   # Hugging Face repo the file comes from
//! file = "Qwen3.8-27B-Q8_0.gguf"
//! sha256 = "aab65c67..."              # checked after download
//! mmproj = "mmproj-Qwen3.8-27B-Q8_0.gguf"
//! mtp = "mtp-Qwen3.8-27B-Q8_0.gguf"
//! # path = "/models/my-finetune.gguf" # a local file instead of repo + file
//! # layer_tensor = "l_out"            # the per-layer output llama.cpp names
//! ```
//!
//! Swapping models is adding an entry and running `vi vectors build <id>`:
//! everything built from a model lives under its file's SHA-256.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const REGISTRY: &str = "models.toml";

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub repo: Option<String>,
    pub file: Option<String>,
    pub sha256: Option<String>,
    pub mmproj: Option<String>,
    pub mtp: Option<String>,
    pub path: Option<PathBuf>,
    #[serde(default = "l_out")]
    pub layer_tensor: String,
    #[serde(default)]
    pub notes: Option<String>,
}

fn l_out() -> String {
    "l_out".into()
}

pub fn load(path: &Path) -> Result<BTreeMap<String, ModelEntry>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let models: BTreeMap<String, ModelEntry> =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    for (id, m) in &models {
        if m.path.is_none() && (m.repo.is_none() || m.file.is_none()) {
            return Err(format!("{id}: needs `path`, or `repo` and `file`"));
        }
    }
    Ok(models)
}

impl ModelEntry {
    /// Where the model file is: `path`, or `file` inside `models_dir`.
    pub fn local_path(&self, models_dir: &Path) -> PathBuf {
        match (&self.path, &self.file) {
            (Some(p), _) => p.clone(),
            (None, Some(f)) => models_dir.join(f),
            (None, None) => unreachable!("checked on load"),
        }
    }

    pub fn url(&self) -> Option<String> {
        Some(format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.repo.as_ref()?,
            self.file.as_ref()?
        ))
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn sidecar(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".sha256");
    PathBuf::from(s)
}

fn stamp(path: &Path) -> Result<String, String> {
    let md = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    Ok(format!("{} {mtime}", md.len()))
}

/// The file's SHA-256, cached in `<file>.sha256` next to it (keyed by size
/// and modification time) so a 29 GB model is hashed once.
pub fn sha256_cached(path: &Path) -> Result<String, String> {
    let side = sidecar(path);
    let stamp = stamp(path)?;
    if let Ok(text) = fs::read_to_string(&side) {
        if let Some((hash, rest)) = text.trim().split_once(' ') {
            if rest == stamp && hash.len() == 64 {
                return Ok(hash.to_string());
            }
        }
    }
    let hash = sha256_file(path)?;
    // Best effort: a read-only folder just means hashing again next time.
    let _ = fs::write(&side, format!("{hash} {stamp}\n"));
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_repository_registry() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(REGISTRY);
        let models = load(&path).unwrap();
        let m = &models["qwen3.5-0.8b-q8"];
        assert_eq!(m.layer_tensor, "l_out");
        assert!(m.url().unwrap().ends_with("/Qwen3.5-0.8B-Q8_0.gguf"));
        assert_eq!(
            m.local_path(Path::new("/m")),
            Path::new("/m/Qwen3.5-0.8B-Q8_0.gguf")
        );
        assert!(models
            .values()
            .all(|m| m.sha256.as_ref().is_some_and(|s| s.len() == 64)));
    }

    #[test]
    fn rejects_entries_without_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("models.toml");
        fs::write(&p, "[x]\nrepo = \"a/b\"\n").unwrap();
        assert!(load(&p).is_err());
        fs::write(&p, "[x]\npath = \"/m.gguf\"\nbogus = 1\n").unwrap();
        assert!(load(&p).is_err());
    }

    #[test]
    fn caches_hashes_until_the_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        fs::write(&p, b"abc").unwrap();
        let h = sha256_cached(&p).unwrap();
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // A stale sidecar is ignored once the size changes.
        fs::write(&p, b"abcd").unwrap();
        assert_ne!(sha256_cached(&p).unwrap(), h);
    }
}
