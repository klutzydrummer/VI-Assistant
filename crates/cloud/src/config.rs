//! `~/.config/vi/cloud.toml`: what to rent and when to let it go.
//!
//! Every field has a default, so the file is optional. Example:
//!
//! ```toml
//! idle_minutes = 20
//! max_hours = 4
//! [search]
//! gpu_names = ["RTX 3090", "RTX 4090"]
//! max_dph = 0.35
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Docker image the worker boots.
    pub image: String,
    /// Disk to rent, in GB. Model weights and one stage's inputs must fit.
    pub disk_gb: f64,
    /// Release the worker after this long without a job or ssh session.
    pub idle_minutes: u64,
    /// Release the worker this long after boot even if busy. 0 turns it off.
    pub max_hours: f64,
    /// How long `vi cloud up` waits for a rented machine to accept ssh before
    /// giving it back and trying another.
    pub boot_minutes: u64,
    /// Vast.ai label marking instances as ours. `vi cloud down` only ever
    /// destroys instances carrying it.
    pub label: String,
    pub search: Search,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Search {
    /// Empty means any GPU that meets the other limits.
    pub gpu_names: Vec<String>,
    pub min_gpu_ram_gb: f64,
    /// Ceiling on the hourly price, in dollars.
    pub max_dph: f64,
    pub min_reliability: f64,
    pub min_inet_down_mbps: f64,
    pub min_cuda: f64,
    /// CUDA compute capability times 100. 800 (Ampere) and up runs bf16 and
    /// flash attention, which most current TTS models expect.
    pub min_compute_cap: u32,
    pub verified: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            image: "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime".into(),
            disk_gb: 60.0,
            idle_minutes: 20,
            max_hours: 4.0,
            boot_minutes: 15,
            label: "vi-worker".into(),
            search: Search::default(),
        }
    }
}

impl Default for Search {
    fn default() -> Self {
        Search {
            gpu_names: Vec::new(),
            min_gpu_ram_gb: 24.0,
            max_dph: 0.40,
            min_reliability: 0.98,
            min_inet_down_mbps: 200.0,
            min_cuda: 12.8,
            min_compute_cap: 800,
            verified: true,
        }
    }
}

impl Config {
    /// `$VI_CLOUD_CONFIG`, else `$XDG_CONFIG_HOME/vi/cloud.toml`, else
    /// `~/.config/vi/cloud.toml`.
    pub fn default_path() -> PathBuf {
        if let Some(p) = std::env::var_os("VI_CLOUD_CONFIG") {
            return p.into();
        }
        xdg_dir("XDG_CONFIG_HOME", ".config").join("vi/cloud.toml")
    }

    /// Reads the file, or the defaults if it doesn't exist.
    pub fn load(path: &Path) -> Result<Config> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
            Err(e) => return Err(crate::io_err(path)(e)),
        };
        let cfg: Config = toml::from_str(&text).map_err(|e| Error::Config {
            path: path.into(),
            message: e.to_string(),
        })?;
        if cfg.idle_minutes == 0 {
            return Err(Error::Config {
                path: path.into(),
                message: "idle_minutes must be at least 1".into(),
            });
        }
        Ok(cfg)
    }

    /// The Vast.ai offer search for these limits, cheapest first.
    pub fn offer_query(&self, limit: usize) -> Value {
        let s = &self.search;
        let mut q = json!({
            "rentable": {"eq": true},
            "rented": {"eq": false},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": s.min_gpu_ram_gb * 1000.0},
            "dph_total": {"lte": s.max_dph},
            "reliability2": {"gte": s.min_reliability},
            "inet_down": {"gte": s.min_inet_down_mbps},
            "cuda_max_good": {"gte": s.min_cuda},
            "compute_cap": {"gte": s.min_compute_cap},
            "disk_space": {"gte": self.disk_gb},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": limit,
        });
        if s.verified {
            q["verified"] = json!({"eq": true});
        }
        if !s.gpu_names.is_empty() {
            q["gpu_name"] = json!({"in": s.gpu_names});
        }
        q
    }

    pub fn max_minutes(&self) -> u64 {
        (self.max_hours * 60.0).round().max(0.0) as u64
    }
}

pub(crate) fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(p) if !p.is_empty() => p.into(),
        _ => PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into())).join(fallback),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_means_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load(&dir.path().join("cloud.toml")).unwrap();
        assert_eq!(cfg.idle_minutes, 20);
        assert_eq!(cfg.label, "vi-worker");
    }

    #[test]
    fn file_overrides_some_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cloud.toml");
        std::fs::write(
            &path,
            "idle_minutes = 5\n[search]\ngpu_names = [\"RTX 4090\"]\nmax_dph = 0.3\n",
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.idle_minutes, 5);
        assert_eq!(cfg.search.min_gpu_ram_gb, 24.0);
        let q = cfg.offer_query(3);
        assert_eq!(q["gpu_name"]["in"][0], "RTX 4090");
        assert_eq!(q["dph_total"]["lte"], 0.3);
        assert_eq!(q["limit"], 3);
        assert_eq!(q["verified"]["eq"], true);
    }

    #[test]
    fn typos_and_zero_idle_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cloud.toml");
        std::fs::write(&path, "idle_minute = 5\n").unwrap();
        assert!(Config::load(&path).is_err());
        std::fs::write(&path, "idle_minutes = 0\n").unwrap();
        assert!(Config::load(&path).is_err());
    }
}
