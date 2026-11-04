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
    /// Account credit (dollars) when rented, for the spend report.
    #[serde(default)]
    pub credit: f64,
    /// Unix seconds.
    pub created: u64,
    /// Unix seconds of the last job, ssh session or `vi cloud touch`.
    pub last_used: u64,
    /// Set once ssh answers.
    pub endpoint: Option<Endpoint>,
}

/// A checkpoint `on-worker.sh` pushed home but nobody has verified yet. The
/// file lives at `<state dir>/unverified/<stage>` and holds the absolute home
/// checkpoint path on one line, so `vi cloud down` can refuse while it exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Marker {
    /// The stage name (`VI_STAGE`).
    pub stage: String,
    /// The absolute home checkpoint path recorded in the marker.
    pub path: PathBuf,
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

    /// The file holding the stamp of the newest VI-HOSTKEY batch accepted for
    /// this instance. A later rotation must present a newer stamp.
    pub fn hostkey_stamp(&self, id: u64) -> PathBuf {
        self.dir.join(format!("hostkey-{id}.stamp"))
    }

    /// `None` until the first rotation is accepted.
    pub fn last_hostkey_stamp(&self, id: u64) -> Result<Option<u64>> {
        let path = self.hostkey_stamp(id);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io_err(&path)(e)),
        };
        text.trim().parse().map(Some).map_err(|_| {
            crate::Error::Worker(format!(
                "{} does not hold a stamp; delete it to accept the next host-key rotation",
                path.display()
            ))
        })
    }

    pub fn save_hostkey_stamp(&self, id: u64, stamp: u64) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(io_err(&self.dir))?;
        let path = self.hostkey_stamp(id);
        let tmp = path.with_extension("stamp.tmp");
        std::fs::write(&tmp, format!("{stamp}\n")).map_err(io_err(&tmp))?;
        std::fs::rename(&tmp, &path).map_err(io_err(&path))
    }

    /// Where `on-worker.sh` drops a marker per unverified checkpoint.
    pub fn unverified_dir(&self) -> PathBuf {
        self.dir.join("unverified")
    }

    /// Every marker currently present, ordered by stage name.
    pub fn markers(&self) -> Result<Vec<Marker>> {
        let dir = self.unverified_dir();
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(io_err(&dir)(e)),
        };
        for entry in entries {
            let entry = entry.map_err(io_err(&dir))?;
            // Only regular files; a symlink must not be followed.
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let stage = entry.file_name().to_string_lossy().into_owned();
            let recorded = std::fs::read_to_string(&path).map_err(io_err(&path))?;
            out.push(Marker {
                stage,
                path: PathBuf::from(recorded.trim()),
            });
        }
        out.sort_by(|a, b| a.stage.cmp(&b.stage));
        Ok(out)
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
        for path in [self.file(), self.known_hosts(id), self.hostkey_stamp(id)] {
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
            credit: 12.0,
            created: 100,
            last_used: 100,
            endpoint: None,
        };
        state.save(&t).unwrap();
        assert_eq!(state.load().unwrap(), Some(t));
        std::fs::write(state.known_hosts(9), "x").unwrap();
        state.save_hostkey_stamp(9, 42).unwrap();
        assert_eq!(state.last_hostkey_stamp(9).unwrap(), Some(42));
        state.clear(9).unwrap();
        assert_eq!(state.load().unwrap(), None);
        assert!(!state.known_hosts(9).exists());
        assert!(!state.hostkey_stamp(9).exists());
        assert_eq!(state.last_hostkey_stamp(9).unwrap(), None);
    }
}
