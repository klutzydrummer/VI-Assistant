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
//! countries = ["US", "CA"]
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::vast::Offer;
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
    /// Two-letter country codes the worker may be in (Vast.ai's
    /// geolocation). Empty means anywhere. Closer machines upload faster,
    /// since one rsync stream slows down with the round trip.
    pub countries: Vec<String>,
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
            // textscore's vLLM brings torch built for CUDA 13; older
            // drivers leave it no GPU. Newer drivers still run CUDA 12 jobs.
            min_cuda: 13.0,
            min_compute_cap: 800,
            verified: true,
            countries: Vec::new(),
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
        if !s.countries.is_empty() {
            let codes: Vec<String> = s
                .countries
                .iter()
                .map(|c| c.trim().to_uppercase())
                .collect();
            q["geolocation"] = json!({"in": codes});
        }
        q
    }

    pub fn max_minutes(&self) -> u64 {
        (self.max_hours * 60.0).round().max(0.0) as u64
    }

    /// Re-applies the search limits to a single offer. `vi cloud up --offer`
    /// fetches by id, which skips `offer_query`, so it must refuse an offer the
    /// search would have hidden. Returns one human-readable reason per limit the
    /// offer fails; empty when it fits.
    pub fn offer_violations(&self, o: &Offer) -> Vec<String> {
        let s = &self.search;
        let mut out = Vec::new();
        if !s.gpu_names.is_empty() && !s.gpu_names.iter().any(|n| n == &o.gpu_name) {
            out.push(format!("gpu {} is not in the allowed list", o.gpu_name));
        }
        if o.gpu_ram < s.min_gpu_ram_gb * 1000.0 {
            out.push(format!(
                "{:.0} GB of ram is under the {:.0} GB minimum",
                o.gpu_ram / 1000.0,
                s.min_gpu_ram_gb
            ));
        }
        if o.compute_cap < s.min_compute_cap {
            out.push(format!(
                "compute capability {} is under {}",
                o.compute_cap, s.min_compute_cap
            ));
        }
        if o.cuda_max_good < s.min_cuda {
            out.push(format!(
                "CUDA {:.1} is under {:.1}",
                o.cuda_max_good, s.min_cuda
            ));
        }
        if o.dph_total > s.max_dph {
            out.push(format!(
                "${:.3}/hr is over the ${:.3} maximum",
                o.dph_total, s.max_dph
            ));
        }
        out
    }
}

pub(crate) fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(p) if !p.is_empty() => p.into(),
        _ => PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into())).join(fallback),
    }
}

/// Parses the `--until` approval deadline: an RFC 3339 timestamp (for example
/// `2026-11-03T22:00:00Z` or `...-05:00`) or `HH:MM` in the local timezone.
/// Returns Unix seconds.
pub fn parse_until(input: &str, now: u64) -> std::result::Result<u64, String> {
    let input = input.trim();
    if let Some(ts) = rfc3339_unix(input) {
        return Ok(ts);
    }
    if let Some((hour, min)) = parse_hhmm(input) {
        return hhmm_today(hour, min, now);
    }
    Err(format!(
        "--until must be an RFC 3339 timestamp or HH:MM (got {input:?})"
    ))
}

fn parse_hhmm(input: &str) -> Option<(u32, u32)> {
    let (h, m) = input.split_once(':')?;
    if h.len() != 2 || m.len() != 2 {
        return None;
    }
    let hour: u32 = h.parse().ok()?;
    let min: u32 = m.parse().ok()?;
    (hour < 24 && min < 60).then_some((hour, min))
}

fn rfc3339_unix(input: &str) -> Option<u64> {
    let sep = input.find(['T', 't', ' '])?;
    let (date, rest) = input.split_at(sep);
    let time = &rest[1..];

    let mut d = date.split('-');
    let year: i64 = d.next()?.parse().ok()?;
    let month: u32 = d.next()?.parse().ok()?;
    let day: u32 = d.next()?.parse().ok()?;
    if d.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let bytes = time.as_bytes();
    let (hour, i) = two(bytes, 0)?;
    if bytes.get(i) != Some(&b':') {
        return None;
    }
    let (min, i) = two(bytes, i + 1)?;
    if bytes.get(i) != Some(&b':') {
        return None;
    }
    let (sec, mut i) = two(bytes, i + 1)?;
    if hour > 23 || min > 59 || sec > 59 {
        return None;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while bytes.get(i).is_some_and(|b| b.is_ascii_digit()) {
            i += 1;
        }
    }
    let offset_secs = match bytes.get(i)? {
        b'Z' | b'z' => {
            i += 1;
            0
        }
        b'+' => {
            let (o, j) = offset(bytes, i + 1)?;
            i = j;
            o
        }
        b'-' => {
            let (o, j) = offset(bytes, i + 1)?;
            i = j;
            -o
        }
        _ => return None,
    };
    if i != bytes.len() {
        return None;
    }
    let sod = hour * 3600 + min * 60 + sec - offset_secs;
    let days = days_from_civil(year, month, day);
    let total = days * 86400 + sod;
    (total >= 0).then_some(total as u64)
}

/// Two decimal digits at `i`: their value and the index just past them.
fn two(bytes: &[u8], i: usize) -> Option<(i64, usize)> {
    let digit = |b: u8| b.is_ascii_digit().then_some((b - b'0') as i64);
    let a = digit(*bytes.get(i)?)?;
    let b = digit(*bytes.get(i + 1)?)?;
    Some((a * 10 + b, i + 2))
}

/// `HH:MM` (or `HHMM`) after an offset sign, in seconds.
fn offset(bytes: &[u8], mut i: usize) -> Option<(i64, usize)> {
    let (h, j) = two(bytes, i)?;
    i = j;
    let m = if bytes.get(i) == Some(&b':') {
        let (m, j) = two(bytes, i + 1)?;
        i = j;
        m
    } else {
        let (m, j) = two(bytes, i)?;
        i = j;
        m
    };
    Some((h * 3600 + m * 60, i))
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// `HH:MM` today in the local timezone, as Unix seconds.
fn hhmm_today(hour: u32, min: u32, now: u64) -> std::result::Result<u64, String> {
    // SAFETY: localtime_r and mktime are called with a pointer to our own `tm`
    // and a valid timestamp; the result is read before the buffer is dropped.
    unsafe {
        let now = now as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&now, &mut tm).is_null() {
            return Err("could not read the local timezone".into());
        }
        tm.tm_hour = hour as libc::c_int;
        tm.tm_min = min as libc::c_int;
        tm.tm_sec = 0;
        let t = libc::mktime(&mut tm);
        if t < 0 {
            return Err("could not resolve HH:MM in the local timezone".into());
        }
        Ok(t as u64)
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
        assert!(q.get("geolocation").is_none());
    }

    #[test]
    fn countries_limit_where_the_worker_is() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cloud.toml");
        std::fs::write(&path, "[search]\ncountries = [\"us\", \"CA\"]\n").unwrap();
        let q = Config::load(&path).unwrap().offer_query(3);
        assert_eq!(q["geolocation"]["in"], json!(["US", "CA"]));
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

    #[test]
    fn parse_until_accepts_rfc3339() {
        assert_eq!(parse_until("1970-01-01T00:00:00Z", 0).unwrap(), 0);
        assert_eq!(parse_until("1970-01-01T01:00:00+01:00", 0).unwrap(), 0);
        assert_eq!(parse_until("1970-01-01T05:00:00-05:00", 0).unwrap(), 36000);
        assert_eq!(parse_until("2000-01-01T00:00:00Z", 0).unwrap(), 946684800);
        // A space separator, lowercase `z` and a fractional part are accepted.
        assert_eq!(parse_until("1970-01-02 00:00:00.5z", 0).unwrap(), 86400);
        assert!(parse_until("not a time", 0).is_err());
        assert!(parse_until("2026-13-01T00:00:00Z", 0).is_err());
    }

    #[test]
    fn parse_until_accepts_hhmm_in_the_local_timezone() {
        let now = 1_700_000_000;
        // Noon today must sit within a day either side of `now`, whatever the
        // machine's timezone is.
        let until = parse_until("12:00", now).unwrap();
        assert!(
            (now as i64 - until as i64).abs() < 86400,
            "{now} vs {until}"
        );
        assert!(parse_until("25:00", now).is_err());
        assert!(parse_until("12:60", now).is_err());
    }

    #[test]
    fn offer_violations_apply_the_search_limits() {
        let cfg = Config::default();
        let ok = Offer {
            id: 1,
            machine_id: 2,
            gpu_name: "RTX 3090".into(),
            gpu_ram: 24576.0,
            dph_total: 0.2,
            reliability2: 0.99,
            inet_down: 900.0,
            geolocation: None,
            compute_cap: 800,
            cuda_max_good: 13.0,
        };
        assert!(cfg.offer_violations(&ok).is_empty());
        let bad = Offer {
            dph_total: 0.5,
            compute_cap: 700,
            cuda_max_good: 12.0,
            ..ok
        };
        let reasons = cfg.offer_violations(&bad);
        assert!(reasons.iter().any(|r| r.contains("over")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.contains("compute")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.contains("CUDA")), "{reasons:?}");
    }
}
