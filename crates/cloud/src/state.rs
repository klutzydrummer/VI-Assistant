//! What home remembers about the worker: `~/.local/state/vi/cloud.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::vast::Endpoint;
use crate::{io_err, Result};

/// The instance we rented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tracked {
    pub id: u64,
    pub machine_id: u64,
    pub gpu_name: String,
    /// Dollars per hour quoted when rented.
    pub dph: f64,
    /// Unix seconds.
    pub created: u64,
    /// Unix seconds of the last job, ssh session or `vi cloud touch`.
    pub last_used: u64,
    /// Set once ssh answers.
    pub endpoint: Option<Endpoint>,
}

pub struct StateDir {
    pub dir: PathBuf,
}

impl StateDir {
    /// `$VI_STATE_DIR`, else `$XDG_STATE_HOME/vi`, else `~/.local/state/vi`.
    pub fn default_dir() -> PathBuf {
        if let Some(p) = std::env::var_os("VI_STATE_DIR") {
            return p.into();
        }
        crate::config::xdg_dir("XDG_STATE_HOME", ".local/state").join("vi")
    }

    pub fn new(dir: impl Into<PathBuf>) -> Self {
        StateDir { dir: dir.into() }
    }

    fn file(&self) -> PathBuf {
        self.dir.join("cloud.json")
    }

    /// Per-instance ssh host keys, so a re-rented port on the same proxy
    /// doesn't trip a host key mismatch.
    pub fn known_hosts(&self, id: u64) -> PathBuf {
        self.dir.join(format!("known_hosts-{id}"))
    }

    pub fn load(&self) -> Result<Option<Tracked>> {
        let path = self.file();
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|e| {
                crate::Error::Worker(format!(
                    "{} is unreadable ({e}); check it on the Vast.ai console, then delete it",
                    path.display()
                ))
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_err(&path)(e)),
        }
    }

    pub fn save(&self, t: &Tracked) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(io_err(&self.dir))?;
        let path = self.file();
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(t).unwrap()).map_err(io_err(&tmp))?;
        std::fs::rename(&tmp, &path).map_err(io_err(&path))
    }

    pub fn clear(&self, id: u64) -> Result<()> {
        for path in [self.file(), self.known_hosts(id)] {
            remove(&path)?;
        }
        Ok(())
    }
}

fn remove(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io_err(path)(e)),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_clears() {
        let dir = tempfile::tempdir().unwrap();
        let state = StateDir::new(dir.path().join("vi"));
        assert_eq!(state.load().unwrap(), None);
        let t = Tracked {
            id: 9,
            machine_id: 3,
            gpu_name: "RTX 3090".into(),
            dph: 0.2,
            created: 100,
            last_used: 100,
            endpoint: None,
        };
        state.save(&t).unwrap();
        assert_eq!(state.load().unwrap(), Some(t));
        std::fs::write(state.known_hosts(9), "x").unwrap();
        state.clear(9).unwrap();
        assert_eq!(state.load().unwrap(), None);
        assert!(!state.known_hosts(9).exists());
    }
}
