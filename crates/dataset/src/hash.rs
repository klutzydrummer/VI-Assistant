//! Content hashes for files, folders and stage outputs.
//!
//! Everything is plain SHA-256 so Python jobs can check hashes with
//! `hashlib`. A folder hashes to the SHA-256 of its `sha256sum`-style
//! listing, which is also how a stage's output hash is defined.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, IoContext};

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String, Error> {
    let mut file = fs::File::open(path).at(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf).at(path)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Hash of a listing of `relative path -> file hash`: the SHA-256 of the
/// lines `sha256sum` would print for it, sorted by path.
pub fn listing_hash(files: &BTreeMap<String, String>) -> String {
    let mut text = String::new();
    for (path, hash) in files {
        text.push_str(hash);
        text.push_str("  ");
        text.push_str(path);
        text.push('\n');
    }
    sha256_bytes(text.as_bytes())
}

/// Every regular file under `root`, as `/`-separated paths relative to it,
/// sorted. Symlinks are an error: outputs must be real files.
pub fn walk(root: &Path) -> Result<Vec<String>, Error> {
    let mut out = Vec::new();
    walk_into(root, "", &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_into(dir: &Path, prefix: &str, out: &mut Vec<String>) -> Result<(), Error> {
    for entry in fs::read_dir(dir).at(dir)? {
        let entry = entry.at(dir)?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = format!("{prefix}{name}");
        let kind = entry.file_type().at(entry.path())?;
        if kind.is_dir() {
            walk_into(&entry.path(), &format!("{rel}/"), out)?;
        } else if kind.is_file() {
            out.push(rel);
        } else {
            return Err(Error::Invalid {
                path: entry.path(),
                message: "only regular files and folders are allowed".into(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Stamp {
    size: u64,
    mtime_ns: u128,
    sha256: String,
}

/// How many bytes of newly hashed files may pile up before the cache is
/// saved, so an interrupted first pass over the game folder keeps most of
/// its progress.
const CHECKPOINT_BYTES: u64 = 1 << 30;

/// Remembers file hashes by path, size and modification time, so large
/// inputs such as the game folder are read in full only once.
#[derive(Debug, Default)]
pub struct HashCache {
    path: Option<PathBuf>,
    entries: BTreeMap<String, Stamp>,
    dirty: bool,
    /// Bytes hashed since the last save.
    unsaved_bytes: u64,
}

impl HashCache {
    /// Loads the cache stored at `path`, or starts empty if it is missing
    /// or unreadable (it is only a cache).
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let entries = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        HashCache {
            path: Some(path),
            entries,
            dirty: false,
            unsaved_bytes: 0,
        }
    }

    pub fn save(&mut self) -> Result<(), Error> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if !self.dirty {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).at(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_vec(&self.entries).expect("cache serializes");
        fs::write(&tmp, json).at(&tmp)?;
        fs::rename(&tmp, path).at(path)?;
        self.dirty = false;
        self.unsaved_bytes = 0;
        Ok(())
    }

    pub fn file(&mut self, path: &Path) -> Result<String, Error> {
        let meta = fs::metadata(path).at(path)?;
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        let key = fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned();
        if let Some(stamp) = self.entries.get(&key) {
            if stamp.size == meta.len() && stamp.mtime_ns == mtime_ns {
                return Ok(stamp.sha256.clone());
            }
        }
        let sha256 = sha256_file(path)?;
        self.entries.insert(
            key,
            Stamp {
                size: meta.len(),
                mtime_ns,
                sha256: sha256.clone(),
            },
        );
        self.dirty = true;
        self.unsaved_bytes += meta.len();
        if self.unsaved_bytes >= CHECKPOINT_BYTES {
            self.save()?;
        }
        Ok(sha256)
    }

    /// A file's hash, or for a folder the listing hash of everything in it.
    pub fn path(&mut self, path: &Path) -> Result<String, Error> {
        let meta = fs::metadata(path).at(path)?;
        if !meta.is_dir() {
            return self.file(path);
        }
        let mut files = BTreeMap::new();
        for rel in walk(path)? {
            let hash = self.file(&path.join(&rel))?;
            files.insert(rel, hash);
        }
        Ok(listing_hash(&files))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoints_large_passes() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("cache.json");
        let mut cache = HashCache::load(&store);
        let small = dir.path().join("small");
        fs::write(&small, b"x").unwrap();
        cache.file(&small).unwrap();
        assert!(!store.exists(), "small files wait for an explicit save");
        // Pretend a gigabyte was already hashed: the next file saves.
        cache.unsaved_bytes = CHECKPOINT_BYTES;
        let other = dir.path().join("other");
        fs::write(&other, b"y").unwrap();
        cache.file(&other).unwrap();
        assert!(store.exists());
        assert_eq!(HashCache::load(&store).entries.len(), 2);
    }

    #[test]
    fn matches_sha256sum() {
        // `printf abc | sha256sum`
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn folder_hash_tracks_content_and_names() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/a.txt"), "one").unwrap();
        fs::write(dir.path().join("b.txt"), "two").unwrap();
        let mut cache = HashCache::default();
        let first = cache.path(dir.path()).unwrap();
        assert_eq!(cache.path(dir.path()).unwrap(), first);

        fs::rename(dir.path().join("b.txt"), dir.path().join("c.txt")).unwrap();
        let renamed = cache.path(dir.path()).unwrap();
        assert_ne!(renamed, first);

        fs::write(dir.path().join("sub/a.txt"), "uno").unwrap();
        assert_ne!(cache.path(dir.path()).unwrap(), renamed);
    }

    #[test]
    fn cache_survives_a_reload() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f");
        fs::write(&file, "data").unwrap();
        let store = dir.path().join("cache.json");
        let mut cache = HashCache::load(&store);
        let hash = cache.file(&file).unwrap();
        cache.save().unwrap();
        let mut again = HashCache::load(&store);
        assert_eq!(again.entries.len(), 1);
        assert_eq!(again.file(&file).unwrap(), hash);
    }
}
