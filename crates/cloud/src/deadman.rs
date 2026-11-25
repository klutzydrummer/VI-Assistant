//! The dead-man timer: layer L2 of the paid-run destroy layers (R32 WP2).
//!
//! Right after `vi cloud up` creates an instance it starts a transient systemd
//! *user* timer, `vi-deadman-<id>.timer`, that runs `vi cloud deadman fire
//! <id>` (destroy, then verify) at the instance's hard cap: the earliest of the
//! approval deadline (`--until`), `max_hours` after the rental and the dollar
//! cap at the rented price. The timer lives in the user manager, not in the
//! process that armed it, so it still fires when `paid_run` crashes, is killed
//! with SIGKILL or hangs.
//!
//! A heartbeat (`vi cloud deadman extend`) moves the fire time to now plus a
//! stall grace, never past the hard cap, so a run that stops making progress is
//! destroyed well before the cap. The timer is cancelled only after Vast's
//! instance list shows the instance gone; any unverified destroy leaves it armed.
//!
//! Decision D1: this is a transient timer only, with no unit file and no boot
//! unit. Transient unit definitions live under `$XDG_RUNTIME_DIR` (systemd.unit),
//! so we assume they do not survive a home reboot; a reboot is covered by the
//! worker's own watchdog (L3), not by this layer.
//!
//! The systemd facts these functions rely on are cited in the round-36 primer
//! `systemd-timer-primer.md`: a calendar time in the past never fires (so near
//! times use `--on-active`), stopping a unit that is not loaded exits 5, and
//! `systemctl --user` needs `XDG_RUNTIME_DIR`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::state::StateDir;
use crate::worker::Host;
use crate::{io_err, Error, Result};

/// Every dead-man unit name starts with this; `VI_DEADMAN_PREFIX` may only
/// lengthen it (the tests use `vi-deadman-test-<pid>-`).
pub const DEFAULT_PREFIX: &str = "vi-deadman-";
pub const PREFIX_ENV: &str = "VI_DEADMAN_PREFIX";
/// Where the fire command's output goes; `paid_run` points it at its run dir.
/// The state dir when unset.
pub const LOG_DIR_ENV: &str = "VI_DEADMAN_LOG_DIR";
/// A fire time this close, or already past, is armed relative to now
/// (`--on-active`), because a calendar time in the past never fires.
pub const NEAR_SECS: u64 = 5;
/// How long systemd waits before re-running a fire that exited nonzero (an
/// unverified destroy exits 9). Far above the default start rate limit of 5
/// starts in 10 s, so the retries never stop.
pub const RESTART_SECS: u64 = 60;

/// What home remembers about one armed timer: `<state dir>/deadman-<id>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: u64,
    /// The unit name without `.timer` / `.service`.
    pub unit: String,
    /// Unix seconds the instance was rented.
    pub start: u64,
    /// The hard cap (unix seconds). Nothing moves the fire time past it, and
    /// re-arming can only lower it.
    pub cap_at: u64,
    /// When the timer fires now (unix seconds); never after `cap_at`.
    pub fire_at: u64,
    /// The command the timer runs, so a re-arm runs exactly the same one.
    pub argv: Vec<String>,
    /// Paths the fire needs to find the same config, state and key file.
    /// Never the Vast key itself.
    pub env: Vec<(String, String)>,
    pub log: PathBuf,
}

/// The hard cap for an instance rented at `start`: the earliest of the
/// approval deadline, `max_hours` later, and the moment `cap` dollars run out
/// at `dph` dollars per hour.
pub fn cap_at(start: u64, until: u64, max_hours: f64, cap: f64, dph: f64) -> u64 {
    let mut at = until.min(start.saturating_add(secs(max_hours * 3600.0)));
    if dph > 0.0 {
        at = at.min(start.saturating_add(secs(cap / dph * 3600.0)));
    }
    at
}

/// Whole seconds, rounded down; NaN or a negative amount counts as none.
fn secs(x: f64) -> u64 {
    if x.is_nan() || x <= 0.0 {
        0
    } else {
        // `as` saturates, so an infinite amount is u64::MAX.
        x.floor() as u64
    }
}

/// An instance id as typed on a command line: ASCII digits only (no sign, no
/// spaces), so it can go into a unit name and a command line unquoted.
pub fn parse_id(s: &str) -> Result<u64> {
    if s.is_empty() || s.len() > 20 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::Worker(format!(
            "{s:?} is not an instance id (digits only)"
        )));
    }
    s.parse()
        .map_err(|_| Error::Worker(format!("{s:?} is not an instance id (too large)")))
}

/// `vi-deadman-<id>`, or `$VI_DEADMAN_PREFIX<id>`.
pub fn unit_name(id: u64) -> Result<String> {
    let prefix = std::env::var(PREFIX_ENV).unwrap_or_else(|_| DEFAULT_PREFIX.into());
    let ok = prefix.starts_with(DEFAULT_PREFIX)
        && prefix.len() <= 64
        && prefix
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if !ok {
        return Err(Error::Worker(format!(
            "{PREFIX_ENV}={prefix:?} must start with {DEFAULT_PREFIX:?} and hold only a-z, 0-9 and -"
        )));
    }
    Ok(format!("{prefix}{id}"))
}

/// A unit name this module could have built.
fn valid_unit(unit: &str) -> bool {
    unit.starts_with(DEFAULT_PREFIX)
        && unit.len() <= 96
        && unit
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Refuses a string that systemd might expand or that a reader could
/// misparse: `$` and `%` (systemd expands both in command lines), backslashes,
/// quotes, whitespace and control characters. Paths and arguments here never
/// need any of them.
fn check_plain(what: &str, s: &str) -> Result<()> {
    let bad = s.is_empty()
        || s.chars().any(|c| {
            matches!(c, '$' | '%' | '\\' | '\'' | '"' | '`') || c.is_whitespace() || c.is_control()
        });
    if bad {
        return Err(Error::Worker(format!(
            "{what} {s:?} is empty or holds a character the dead-man timer refuses \
             ($ % \\ quotes, whitespace or control characters)"
        )));
    }
    Ok(())
}

/// An absolute path with only plain characters.
fn plain_path(what: &str, p: &Path) -> Result<PathBuf> {
    let p = std::path::absolute(p).map_err(io_err(p))?;
    let s = p
        .to_str()
        .ok_or_else(|| Error::Worker(format!("{what} {} is not valid UTF-8", p.display())))?;
    check_plain(what, s)?;
    Ok(p)
}

/// The command the timer runs: `<vi> cloud deadman fire <id>`.
pub fn fire_argv(exe: &Path, id: u64) -> Result<Vec<String>> {
    let exe = plain_path("the vi executable", exe)?;
    Ok(vec![
        exe.to_string_lossy().into_owned(),
        "cloud".into(),
        "deadman".into(),
        "fire".into(),
        id.to_string(),
    ])
}

/// The environment the fire needs to find the same state, config and key
/// file as the `vi` that armed it. The key lookup is `vi cloud`'s own
/// (`VAST_API_KEY`, else the key file under `XDG_CONFIG_HOME`); the key itself
/// is never passed.
pub fn fire_env(state: &StateDir) -> Result<Vec<(String, String)>> {
    let mut env = vec![
        (
            "VI_STATE_DIR".to_string(),
            plain_path("the state dir", &state.dir)?,
        ),
        (
            "VI_CLOUD_CONFIG".to_string(),
            plain_path("the cloud config", &crate::Config::default_path())?,
        ),
    ];
    if let Some(p) = std::env::var_os("XDG_CONFIG_HOME") {
        env.push((
            "XDG_CONFIG_HOME".into(),
            plain_path("XDG_CONFIG_HOME", Path::new(&p))?,
        ));
    }
    let mut env: Vec<(String, String)> = env
        .into_iter()
        .map(|(k, p)| (k, p.to_string_lossy().into_owned()))
        .collect();
    for key in ["VAST_API_URL", PREFIX_ENV] {
        if let Ok(v) = std::env::var(key) {
            check_plain(key, &v)?;
            env.push((key.into(), v));
        }
    }
    Ok(env)
}

/// `<log dir>/deadman-<id>.log`; the log dir is created.
pub fn log_path(state: &StateDir, id: u64) -> Result<PathBuf> {
    let dir = match std::env::var_os(LOG_DIR_ENV) {
        Some(d) => PathBuf::from(d),
        None => state.dir.clone(),
    };
    let dir = plain_path(LOG_DIR_ENV, &dir)?;
    std::fs::create_dir_all(&dir).map_err(io_err(&dir))?;
    Ok(dir.join(format!("deadman-{id}.log")))
}

pub fn record_path(state: &StateDir, id: u64) -> PathBuf {
    state.dir.join(format!("deadman-{id}.json"))
}

pub fn load(state: &StateDir, id: u64) -> Result<Option<Record>> {
    let path = record_path(state, id);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|e| {
            Error::Worker(format!(
                "{} is unreadable ({e}); check `systemctl --user list-timers` and delete it",
                path.display()
            ))
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_err(&path)(e)),
    }
}

fn save(state: &StateDir, r: &Record) -> Result<()> {
    std::fs::create_dir_all(&state.dir).map_err(io_err(&state.dir))?;
    let path = record_path(state, r.id);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(r).unwrap()).map_err(io_err(&tmp))?;
    std::fs::rename(&tmp, &path).map_err(io_err(&path))
}

/// Checks before renting that a timer could be armed: the user manager
/// answers, and if the key comes from `VAST_API_KEY` (which the timer is never
/// given) the key file the fire would read exists. The file is not read.
pub fn preflight<H: Host>(host: &H) -> Result<()> {
    host.timer_ready().map_err(|e| {
        Error::Worker(format!(
            "the dead-man timer cannot be armed ({e}); refusing to rent without it"
        ))
    })?;
    if std::env::var_os("VAST_API_KEY").is_some() {
        let file = crate::config::xdg_dir("XDG_CONFIG_HOME", ".config").join("vastai/vast_api_key");
        if !file.is_file() {
            return Err(Error::Worker(format!(
                "VAST_API_KEY is set but {} does not exist; the dead-man timer never gets the \
                 key, so it could not destroy the instance. Refusing to rent",
                file.display()
            )));
        }
    }
    Ok(())
}

/// Arms the timer for instance `id` to fire at `cap_at`. Idempotent: if a
/// record already exists the cap can only go down, the fire time is kept (or
/// lowered to the new cap), and a timer that is already loaded at that time is
/// left alone.
pub fn arm<H: Host>(
    host: &H,
    state: &StateDir,
    argv: Vec<String>,
    id: u64,
    start: u64,
    cap_at: u64,
) -> Result<Record> {
    let unit = unit_name(id)?;
    for a in &argv {
        check_plain("a fire command argument", a)?;
    }
    let existing = load(state, id)?;
    let record = match &existing {
        Some(r) => Record {
            unit: unit.clone(),
            cap_at: r.cap_at.min(cap_at),
            fire_at: r.fire_at.min(cap_at),
            ..r.clone()
        },
        None => Record {
            id,
            unit: unit.clone(),
            start,
            cap_at,
            fire_at: cap_at,
            argv,
            env: fire_env(state)?,
            log: log_path(state, id)?,
        },
    };
    let loaded = host.timer_loaded(&unit)?;
    if loaded && existing.as_ref() == Some(&record) {
        return Ok(record);
    }
    if loaded {
        host.timer_stop(&unit, false)?;
    }
    host.timer_arm(&record)?;
    save(state, &record)?;
    Ok(record)
}

/// The heartbeat: moves the fire time to now + `grace`, clamped to the hard
/// cap. Refuses once the cap has been reached. If re-arming fails it puts the
/// old fire time back; if that fails too the error says the timer is gone.
pub fn extend<H: Host>(host: &H, state: &StateDir, id: u64, grace: u64) -> Result<Record> {
    let old = load(state, id)?
        .ok_or_else(|| Error::Worker(format!("no dead-man timer is recorded for instance {id}")))?;
    let now = host.now();
    if now >= old.cap_at {
        return Err(Error::Worker(format!(
            "instance {id} reached its hard cap at {}; the dead-man timer cannot be extended",
            old.cap_at
        )));
    }
    let new = Record {
        fire_at: now.saturating_add(grace).min(old.cap_at),
        ..old.clone()
    };
    let loaded = host.timer_loaded(&old.unit)?;
    if loaded && new.fire_at == old.fire_at {
        return Ok(old);
    }
    if loaded {
        host.timer_stop(&old.unit, false)?;
    }
    match host.timer_arm(&new) {
        Ok(()) => {
            save(state, &new)?;
            Ok(new)
        }
        Err(e) => match host.timer_arm(&old) {
            Ok(()) => Err(Error::Worker(format!(
                "could not move the dead-man timer for instance {id} ({e}); it still fires at {}",
                old.fire_at
            ))),
            Err(e2) => Err(Error::Worker(format!(
                "ALERT: the dead-man timer for instance {id} is NOT armed: moving it failed \
                 ({e}) and restoring it failed ({e2})"
            ))),
        },
    }
}

/// Stops the timer (and, if `service`, a fire already running) and forgets
/// the record. Call it only once the destroy is verified. Cancelling a timer
/// that is gone, or cancelling twice, does nothing.
pub fn cancel<H: Host>(host: &H, state: &StateDir, id: u64, service: bool) -> Result<()> {
    let unit = match load(state, id) {
        Ok(Some(r)) => r.unit,
        _ => unit_name(id)?,
    };
    host.timer_stop(&unit, service)?;
    let path = record_path(state, id);
    match std::fs::remove_file(&path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io_err(&path)(e)),
        _ => Ok(()),
    }
}

// ----- the real systemd side (SystemHost) -----------------------------------

fn user_command(program: &str) -> Command {
    let mut c = Command::new(program);
    c.arg("--user").stdin(Stdio::null());
    if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
        // SAFETY: getuid has no preconditions and cannot fail.
        let uid = unsafe { libc::getuid() };
        c.env("XDG_RUNTIME_DIR", format!("/run/user/{uid}"));
    }
    c
}

fn run(mut c: Command, what: &str) -> Result<(i32, String)> {
    let out = c
        .output()
        .map_err(|e| Error::Worker(format!("could not run {what}: {e}")))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Ok((out.status.code().unwrap_or(-1), text.trim().to_owned()))
}

/// The user manager answers.
pub fn systemd_ready() -> Result<()> {
    let mut c = user_command("systemctl");
    c.args(["show", "-p", "Version", "--value"]);
    match run(c, "systemctl --user")? {
        (0, v) if !v.is_empty() => Ok(()),
        (rc, text) => Err(Error::Worker(format!(
            "the systemd user manager does not answer (systemctl --user rc {rc}: {text})"
        ))),
    }
}

/// `<unit>.timer` is loaded (pending, or elapsed with its fire still running).
pub fn systemd_loaded(unit: &str) -> Result<bool> {
    if !valid_unit(unit) {
        return Err(Error::Worker(format!("refusing unit name {unit:?}")));
    }
    let mut c = user_command("systemctl");
    c.args([
        "show",
        &format!("{unit}.timer"),
        "-p",
        "LoadState",
        "--value",
    ]);
    match run(c, "systemctl --user show")? {
        (0, s) if s == "loaded" => Ok(true),
        (0, s) if s == "not-found" => Ok(false),
        (rc, text) => Err(Error::Worker(format!(
            "could not read {unit}.timer (rc {rc}: {text})"
        ))),
    }
}

/// The `systemd-run` arguments for `r`, given the current time.
pub fn systemd_run_args(r: &Record, now: u64) -> Result<Vec<String>> {
    if !valid_unit(&r.unit) {
        return Err(Error::Worker(format!("refusing unit name {:?}", r.unit)));
    }
    let log = r
        .log
        .to_str()
        .ok_or_else(|| Error::Worker("the dead-man log path is not UTF-8".into()))?;
    check_plain("the dead-man log path", log)?;
    if !r.log.is_absolute() {
        return Err(Error::Worker(format!(
            "the dead-man log {log} is not absolute"
        )));
    }
    let when = if r.fire_at <= now.saturating_add(NEAR_SECS) {
        format!("--on-active={}s", r.fire_at.saturating_sub(now).max(1))
    } else {
        format!("--on-calendar=@{}", r.fire_at)
    };
    let mut args = vec![
        format!("--unit={}", r.unit),
        when,
        "--timer-property=AccuracySec=1s".into(),
        "--timer-property=RemainAfterElapse=no".into(),
        "--collect".into(),
        "--quiet".into(),
        "-p".into(),
        "Restart=on-failure".into(),
        "-p".into(),
        format!("RestartSec={RESTART_SECS}s"),
        "-p".into(),
        format!("StandardOutput=append:{log}"),
        "-p".into(),
        format!("StandardError=append:{log}"),
    ];
    for (k, v) in &r.env {
        if k.is_empty() || !k.bytes().all(|b| b.is_ascii_uppercase() || b == b'_') {
            return Err(Error::Worker(format!("refusing environment name {k:?}")));
        }
        if k == "VAST_API_KEY" {
            return Err(Error::Worker(
                "the dead-man timer never gets the Vast key".into(),
            ));
        }
        check_plain(k, v)?;
        args.push(format!("--setenv={k}={v}"));
    }
    let Some(exe) = r.argv.first() else {
        return Err(Error::Worker("the dead-man fire command is empty".into()));
    };
    if !Path::new(exe).is_absolute() {
        return Err(Error::Worker(format!(
            "the dead-man fire command {exe} is not absolute"
        )));
    }
    args.push("--".into());
    for a in &r.argv {
        check_plain("a fire command argument", a)?;
        args.push(a.clone());
    }
    Ok(args)
}

pub fn systemd_arm(r: &Record, now: u64) -> Result<()> {
    let mut c = user_command("systemd-run");
    c.args(systemd_run_args(r, now)?);
    match run(c, "systemd-run --user")? {
        (0, _) => Ok(()),
        (rc, text) => Err(Error::Worker(format!(
            "systemd-run could not arm {}.timer (rc {rc}: {text})",
            r.unit
        ))),
    }
}

/// Stops `<unit>.timer`, and `<unit>.service` too if `service`. A unit that
/// is not loaded (exit 5) counts as stopped.
pub fn systemd_stop(unit: &str, service: bool) -> Result<()> {
    if !valid_unit(unit) {
        return Err(Error::Worker(format!("refusing unit name {unit:?}")));
    }
    let mut kinds = vec!["timer"];
    if service {
        kinds.push("service");
    }
    for kind in kinds {
        let mut c = user_command("systemctl");
        c.args(["stop", &format!("{unit}.{kind}")]);
        match run(c, "systemctl --user stop")? {
            (0 | 5, _) => {}
            (rc, text) => {
                return Err(Error::Worker(format!(
                    "could not stop {unit}.{kind} (rc {rc}: {text})"
                )))
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_is_the_earliest_of_until_wall_and_dollars() {
        // until wins
        assert_eq!(cap_at(1000, 2000, 4.0, 10.0, 0.2), 2000);
        // max_hours wins
        assert_eq!(cap_at(1000, 1_000_000, 1.0, 10.0, 0.2), 1000 + 3600);
        // dollars win: $0.10 at $0.20/h is half an hour
        assert_eq!(cap_at(1000, 1_000_000, 4.0, 0.1, 0.2), 1000 + 1800);
        // a zero price ignores the dollar term; NaN counts as no time at all
        assert_eq!(cap_at(1000, 1_000_000, 1.0, 10.0, 0.0), 1000 + 3600);
        assert_eq!(cap_at(1000, 1_000_000, f64::NAN, 10.0, 0.2), 1000);
    }

    #[test]
    fn ids_are_digits_only() {
        assert_eq!(parse_id("123").unwrap(), 123);
        for bad in [
            "",
            "+5",
            "-5",
            "1 2",
            "12a",
            "1;rm",
            "$(x)",
            "99999999999999999999999",
        ] {
            assert!(parse_id(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn plain_strings_refuse_expansion_characters() {
        for bad in ["a$b", "a%b", "a b", "a\nb", "a\\b", "a'b", "a\"b", ""] {
            assert!(check_plain("x", bad).is_err(), "{bad:?} accepted");
        }
        check_plain("x", "/home/me/.local/state/vi").unwrap();
    }

    fn record(fire_at: u64) -> Record {
        Record {
            id: 42,
            unit: "vi-deadman-42".into(),
            start: 1000,
            cap_at: 5000,
            fire_at,
            argv: vec![
                "/bin/vi".into(),
                "cloud".into(),
                "deadman".into(),
                "fire".into(),
                "42".into(),
            ],
            env: vec![("VI_STATE_DIR".into(), "/s".into())],
            log: "/s/deadman-42.log".into(),
        }
    }

    #[test]
    fn far_fire_times_are_calendar_and_near_ones_relative() {
        let args = systemd_run_args(&record(5000), 1000).unwrap();
        assert!(
            args.contains(&"--on-calendar=@5000".to_string()),
            "{args:?}"
        );
        assert!(args.contains(&"--unit=vi-deadman-42".to_string()));
        assert!(args.contains(&"--setenv=VI_STATE_DIR=/s".to_string()));
        assert!(args.contains(&"StandardOutput=append:/s/deadman-42.log".to_string()));
        let tail: Vec<&str> = args
            .iter()
            .skip_while(|a| *a != "--")
            .map(String::as_str)
            .collect();
        assert_eq!(tail, ["--", "/bin/vi", "cloud", "deadman", "fire", "42"]);
        // Already past: a calendar time would never fire.
        let args = systemd_run_args(&record(900), 1000).unwrap();
        assert!(args.contains(&"--on-active=1s".to_string()), "{args:?}");
        let args = systemd_run_args(&record(1003), 1000).unwrap();
        assert!(args.contains(&"--on-active=3s".to_string()), "{args:?}");
    }

    #[test]
    fn run_args_refuse_bad_units_env_and_the_key() {
        let mut r = record(5000);
        r.unit = "other-42".into();
        assert!(systemd_run_args(&r, 0).is_err());
        let mut r = record(5000);
        r.env.push(("VAST_API_KEY".into(), "x".into()));
        assert!(systemd_run_args(&r, 0).is_err());
        let mut r = record(5000);
        r.env.push(("X".into(), "$HOME".into()));
        assert!(systemd_run_args(&r, 0).is_err());
        let mut r = record(5000);
        r.argv[0] = "vi".into();
        assert!(systemd_run_args(&r, 0).is_err());
    }
}
