//! Renting, waiting for, and releasing the worker.

use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

use crate::state::{Marker, StateDir, Tracked};
use crate::vast::{Endpoint, Instance, Offer, Vast};
use crate::{deadman, onstart, Config, Error, Result};

const POLL_SECS: u64 = 10;

/// How long `down_verified` keeps listing instances, waiting for the
/// destroyed ones to disappear, before it reports the destroy unverified.
pub const VERIFY_SECS: u64 = 120;

/// While an instance is still listed, `down_verified` sends its DELETE again
/// at most this often.
const REDELETE_SECS: u64 = 30;

/// With hosts to exclude, `offers_excluding` asks Vast for this many times the rows it will
/// list, because the host filter can only be applied to the answer.
const HOST_EXCLUDE_WIDEN: usize = 4;

/// The outside world, so tests can fake time and ssh.
pub trait Host {
    /// Unix seconds.
    fn now(&self) -> u64;
    fn sleep(&self, secs: u64);
    /// Does `ssh root@endpoint true` succeed?
    fn reachable(&self, ep: &Endpoint, known_hosts: &Path) -> bool;
    fn log(&self, msg: &str);
    /// The host keys `ssh-keyscan` presents for `ep`, as known_hosts lines.
    /// Empty when the scan fails.
    fn keyscan(&self, _ep: &Endpoint) -> Vec<String> {
        Vec::new()
    }
    /// `ssh-keygen -lf -` output for `keyscan` lines, one line per key.
    fn key_fingerprints(&self, _lines: &[String]) -> Vec<String> {
        Vec::new()
    }
    /// The dead-man timer (see `deadman.rs`): does the systemd user manager
    /// answer? A host without timers says no, so renting refuses loudly.
    fn timer_ready(&self) -> Result<()> {
        Err(no_timers())
    }
    /// Is `<unit>.timer` loaded?
    fn timer_loaded(&self, _unit: &str) -> Result<bool> {
        Err(no_timers())
    }
    /// Starts the transient timer `r` describes.
    fn timer_arm(&self, _r: &deadman::Record) -> Result<()> {
        Err(no_timers())
    }
    /// Stops `<unit>.timer` (and `<unit>.service` if `service`); a unit that
    /// is not loaded counts as stopped.
    fn timer_stop(&self, _unit: &str, _service: bool) -> Result<()> {
        Err(no_timers())
    }
}

fn no_timers() -> Error {
    Error::Worker("this host has no systemd user timers".into())
}

pub struct SystemHost;

impl Host for SystemHost {
    fn now(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    }

    fn sleep(&self, secs: u64) {
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }

    fn reachable(&self, ep: &Endpoint, known_hosts: &Path) -> bool {
        Command::new("ssh")
            .args(ssh_args(ep, known_hosts))
            .args([
                "-o",
                "ConnectTimeout=10",
                &format!("root@{}", ep.host),
                "true",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    fn log(&self, msg: &str) {
        eprintln!("{msg}");
    }

    fn keyscan(&self, ep: &Endpoint) -> Vec<String> {
        let out = Command::new("ssh-keyscan")
            .args(["-T", "10", "-p", &ep.port.to_string(), &ep.host])
            .stdin(Stdio::null())
            .output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_owned)
                .collect(),
            _ => Vec::new(),
        }
    }

    fn key_fingerprints(&self, lines: &[String]) -> Vec<String> {
        use std::io::Write as _;

        let mut child = match Command::new("ssh-keygen")
            .args(["-lf", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => return Vec::new(),
        };
        if let Some(mut stdin) = child.stdin.take() {
            for line in lines {
                if writeln!(stdin, "{line}").is_err() {
                    return Vec::new();
                }
            }
        }
        match child.wait_with_output() {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect(),
            _ => Vec::new(),
        }
    }

    fn timer_ready(&self) -> Result<()> {
        deadman::systemd_ready()
    }

    fn timer_loaded(&self, unit: &str) -> Result<bool> {
        deadman::systemd_loaded(unit)
    }

    fn timer_arm(&self, r: &deadman::Record) -> Result<()> {
        deadman::systemd_arm(r, self.now())
    }

    fn timer_stop(&self, unit: &str, service: bool) -> Result<()> {
        deadman::systemd_stop(unit, service)
    }
}

/// Whether the worker is in use right now. `reap` consults this before
/// releasing a worker for idleness; tests inject a fixed answer.
pub trait Activity {
    /// `Ok(true)` when an interactive session or another ssh connection is
    /// using the worker, `Ok(false)` when it is idle. `Err` when that cannot be
    /// established, in which case `reap` keeps the worker.
    fn busy(&self, ep: &Endpoint, known_hosts: &Path) -> Result<bool>;
}

/// The real activity probe: one ssh command on the worker. `who` lists
/// interactive sessions (a non-interactive ssh command leaves no utmp entry,
/// so the probe's own connection does not count), and an established inbound
/// ssh connection beyond the probe's own also counts as use.
pub struct SshActivity;

/// The remote side of [`SshActivity`]: exit 10 when `who` shows an interactive
/// session, 11 when `ss` shows more than one established inbound ssh
/// connection (the probe's own is the one), 0 when idle. A missing `who` or
/// `ss` exits 3 or 4, which the probe treats as a failed check.
const ACTIVITY_CHECK: &str = r#"who_out=$(who) || exit 3
ss_out=$(ss -Htn state established '( sport = :22 )') || exit 4
[ -z "$who_out" ] || exit 10
newline='
'
case "$ss_out" in *"$newline"*) exit 11 ;; esac
exit 0"#;

impl Activity for SshActivity {
    fn busy(&self, ep: &Endpoint, known_hosts: &Path) -> Result<bool> {
        use std::time::{Duration, Instant};

        /// How long the whole probe may take before its ssh child is killed.
        const TIMEOUT: Duration = Duration::from_secs(15);

        // BatchMode=yes (from ssh_args) keeps the probe from prompting.
        let mut child = Command::new("ssh")
            .args(ssh_args(ep, known_hosts))
            .args(["-o", "ConnectTimeout=5"])
            .arg(format!("root@{}", ep.host))
            .arg(ACTIVITY_CHECK)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::Worker(format!("could not run the activity check: {e}")))?;

        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() >= TIMEOUT => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Error::Worker(
                        "the activity check timed out after 15 seconds".into(),
                    ));
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(e) => {
                    return Err(Error::Worker(format!(
                        "could not wait for the activity check: {e}"
                    )))
                }
            }
        };
        match status.code() {
            Some(0) => Ok(false),
            Some(10 | 11) => Ok(true),
            Some(code) => Err(Error::Worker(format!(
                "the activity check failed (ssh exited with status {code})"
            ))),
            None => Err(Error::Worker(
                "the activity check was killed by a signal".into(),
            )),
        }
    }
}

/// The ssh options every connection to the worker uses (no user or host).
pub fn ssh_args(ep: &Endpoint, known_hosts: &Path) -> Vec<String> {
    vec![
        "-p".into(),
        ep.port.to_string(),
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "StrictHostKeyChecking=accept-new".into(),
        "-o".into(),
        format!("UserKnownHostsFile={}", known_hosts.display()),
        "-o".into(),
        "ServerAliveInterval=30".into(),
        "-o".into(),
        "ForwardAgent=no".into(),
        "-o".into(),
        "ForwardX11=no".into(),
    ]
}

pub struct Cloud<V, H> {
    pub cfg: Config,
    pub vast: V,
    pub state: StateDir,
    pub host: H,
}

/// What `vi cloud status` shows.
pub struct Report {
    pub tracked: Option<Tracked>,
    /// The tracked instance as Vast.ai sees it; `None` if it's gone.
    pub instance: Option<Instance>,
    /// Instances carrying our label that home doesn't track.
    pub orphans: Vec<Instance>,
    /// Everything else on the account. Not ours to touch, but it may cost money.
    pub others: Vec<Instance>,
    pub credit: Option<f64>,
}

/// What `down`/`down_force` reports: the ids destroyed, any unverified markers
/// it overrode (left in place until a later `checkpoints.py verify` clears
/// them), and the account credit before and after for the spend report.
#[derive(Debug)]
pub struct Down {
    pub ids: Vec<u64>,
    pub overridden: Vec<Marker>,
    /// Account credit when the worker was rented.
    pub credit_before: Option<f64>,
    /// Account credit after the worker was released.
    pub credit_after: Option<f64>,
    /// `down_verified` only: true once Vast's instance list showed every id in
    /// `ids` absent (or `destroyed`). Always false for `down`/`down_force`,
    /// which trust the DELETE answer and never look.
    pub verified: bool,
    /// `down_verified` only: ids Vast still listed when the check gave up.
    pub still_listed: Vec<u64>,
    /// `down_verified` only: failures it carried on past (a DELETE or a listing
    /// that failed, unreadable state), so the caller can report them.
    pub errors: Vec<String>,
}

/// A host key the worker presents now, as a known_hosts line and the
/// fingerprint `ssh-keygen` reports for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentedKey {
    pub line: String,
    pub fingerprint: String,
    /// Uppercase, like `ED25519` in `ssh-keygen -lf` output.
    pub key_type: String,
}

/// What `vi cloud hostkey` accepted: the rotated keys and the boot they belong
/// to. Only a reattach continues after this; no inputs are sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rotation {
    pub endpoint: Endpoint,
    pub stamp: u64,
    /// The fingerprints of the keys written to known_hosts.
    pub fingerprints: Vec<String>,
}

/// The VI-HOSTKEY lines of one boot. Every line the boot script prints shares
/// the boot's timestamp, which orders rotations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostkeyBatch {
    pub stamp: u64,
    /// `(SHA256 fingerprint, uppercase key type)`.
    pub keys: Vec<(String, String)>,
}

/// Parse one `ssh-keygen -lf` line, like
/// `256 SHA256:abc... comment (ED25519)`, into the fingerprint and its key type.
pub fn parse_keygen_line(line: &str) -> Option<(String, String)> {
    let fingerprint = line
        .split_whitespace()
        .find(|t| t.starts_with("SHA256:"))?
        .to_owned();
    let open = line.rfind('(')?;
    let close = line[open..].find(')')? + open;
    let key_type = line[open + 1..close].trim().to_ascii_uppercase();
    (!key_type.is_empty()).then_some((fingerprint, key_type))
}

/// The newest VI-HOSTKEY batch in an instance log, if it is newer than the
/// last accepted stamp. The boot script prints
/// `VI-HOSTKEY <unix seconds> <ssh-keygen -lf line>` for every host key type,
/// so a batch is the group of lines sharing the highest stamp.
pub fn newest_hostkey_batch(log: &str, last: Option<u64>) -> Option<HostkeyBatch> {
    use std::collections::BTreeMap;

    let mut batches: BTreeMap<u64, Vec<(String, String)>> = BTreeMap::new();
    for line in log.lines() {
        // The line may carry a container-log timestamp before the marker.
        let Some((_, rest)) = line.split_once("VI-HOSTKEY ") else {
            continue;
        };
        let Some((stamp, details)) = rest.split_once(' ') else {
            continue;
        };
        let Ok(stamp) = stamp.parse::<u64>() else {
            continue;
        };
        if let Some(key) = parse_keygen_line(details) {
            batches.entry(stamp).or_default().push(key);
        }
    }
    let (stamp, keys) = batches.into_iter().next_back()?;
    if last.is_some_and(|last| stamp <= last) {
        return None;
    }
    Some(HostkeyBatch { stamp, keys })
}

impl<V: Vast, H: Host> Cloud<V, H> {
    pub fn offers(&self, limit: usize) -> Result<Vec<Offer>> {
        self.offers_excluding(limit, &[], &[])
    }

    /// Like `offers`, but never lists the given machines or hosts (r32 WP5: a host that failed
    /// its qualification or dropped is not rented again in the same run). Machines go into the
    /// Vast search itself. Vast documents no `host_id` search filter, so hosts are removed from
    /// the answer here; the search then asks for more rows so the excluded ones do not use up
    /// the `limit` slots. An offer without a host id is kept.
    pub fn offers_excluding(
        &self,
        limit: usize,
        machines: &[u64],
        hosts: &[u64],
    ) -> Result<Vec<Offer>> {
        if hosts.is_empty() {
            return self.vast.offers(&self.cfg.offer_query(limit, machines));
        }
        let wide = limit.saturating_mul(HOST_EXCLUDE_WIDEN).max(limit);
        let mut list = self.vast.offers(&self.cfg.offer_query(wide, machines))?;
        list.retain(|o| o.host_id.is_none_or(|h| !hosts.contains(&h)));
        list.truncate(limit);
        Ok(list)
    }

    /// Rents a worker (or reuses the one we have) and waits until ssh answers.
    /// `cap` and `until` are the approval the caller checked with Brandon: the
    /// most this run may cost, and when the approval expires. Before renting it
    /// refuses if the deadline has passed, the worst case (`max_hours` at the
    /// offer's price) exceeds `cap`, or the account holds less than `cap`.
    /// `confirm` sees the chosen offer and the account credit before anything
    /// is rented, and returns false to back out.
    pub fn up(
        &self,
        offer: Option<u64>,
        cap: f64,
        until: u64,
        confirm: impl FnOnce(&Offer, Option<f64>) -> bool,
    ) -> Result<Tracked> {
        if let Some(t) = self.state.load()? {
            match self.vast.instance(t.id)? {
                Some(inst) if !stopped(&inst) => {
                    // Never run a rental without the dead-man timer: re-arm it
                    // if it is gone (idempotent when it is still armed).
                    let cap_at = deadman::cap_at(t.created, until, self.cfg.max_hours, cap, t.dph);
                    self.arm_deadman(t.id, t.created, cap_at)?;
                    if t.endpoint.is_some() && inst.status() == "running" {
                        return self.touch_tracked(t);
                    }
                    self.host.log(&format!(
                        "worker {} is still starting ({})",
                        t.id,
                        inst.status()
                    ));
                    return self.wait(t);
                }
                Some(_) => {
                    let markers = self.state.markers()?;
                    if !markers.is_empty() {
                        return Err(Error::Worker(unverified_up_error(&markers)));
                    }
                    self.host.log(&format!(
                        "worker {} was stopped; destroying it and renting a fresh one",
                        t.id
                    ));
                    self.vast.destroy(t.id)?;
                    self.state.clear(t.id)?;
                }
                None => {
                    self.host
                        .log(&format!("worker {} is gone; renting a fresh one", t.id));
                    self.state.clear(t.id)?;
                }
            }
        }

        // The approval window must still be open before we look at any offer.
        if self.host.now() >= until {
            return Err(Error::Worker(
                "the approval deadline has passed; pass a later --until".into(),
            ));
        }

        let candidates = match offer {
            // Vast's search ignores `id`; an offer's id is its ask_contract_id.
            Some(id) => self.vast.offers(&json!({
                "ask_contract_id": {"eq": id}, "rentable": {"eq": true}, "limit": 1
            }))?,
            None => self.offers(10)?,
        };
        let Some(first) = candidates.first() else {
            return Err(Error::Worker(match offer {
                Some(id) => format!("offer {id} is no longer available"),
                None => "no offers match the search limits; raise search.max_dph or relax \
                         the others in the cloud config"
                    .into(),
            }));
        };
        // `--offer` bypasses the search, so re-apply its limits here.
        if offer.is_some() {
            if let Some(why) = self.cfg.offer_violations(first).first() {
                return Err(Error::Worker(format!(
                    "offer {} fails the search limits: {why}",
                    first.id
                )));
            }
        }
        let credit = self.check_budget(first, cap)?;
        // Refuse before renting if the dead-man timer could not be armed.
        deadman::preflight(&self.host)?;
        if !confirm(first, Some(credit)) {
            return Err(Error::Worker("nothing rented".into()));
        }

        let id = self.vast.create(first.id, &self.create_request())?;
        let now = self.host.now();
        let t = Tracked {
            id,
            machine_id: first.machine_id,
            gpu_name: first.gpu_name.clone(),
            dph: first.dph_total,
            credit,
            created: now,
            last_used: now,
            endpoint: None,
        };
        // Saved before waiting, so `vi cloud down` can release it even if
        // this process is interrupted.
        self.state.save(&t)?;
        // The dead-man timer (L2) is armed before anything else can go wrong,
        // so a crash, kill or hang from here on still destroys the instance.
        let cap_at = deadman::cap_at(now, until, self.cfg.max_hours, cap, t.dph);
        if let Err(e) = self.arm_deadman(id, now, cap_at) {
            self.host.log(&format!(
                "ALERT: could not arm the dead-man timer for instance {id} ({e}); destroying it"
            ));
            let gone = self.vast.destroy(id).and_then(|()| self.state.clear(id));
            return Err(Error::Worker(match gone {
                Ok(()) => format!(
                    "could not arm the dead-man timer ({e}); instance {id} was destroyed \
                     (run `vi cloud down --force --verify {id}` to confirm)"
                ),
                Err(d) => format!(
                    "could not arm the dead-man timer ({e}) and destroying instance {id} \
                     failed ({d}); run `vi cloud down --force --verify {id}` now"
                ),
            }));
        }
        self.host
            .log(&format!("rented instance {id}; waiting for it to boot"));
        match self.wait(t) {
            Ok(t) => Ok(t),
            Err(e) => {
                // One rental per invocation: a failed boot is given back and
                // `up` exits nonzero rather than renting a replacement.
                self.host.log(&format!(
                    "instance {id} failed to boot ({e}); giving it back without renting another"
                ));
                self.vast.destroy(id)?;
                self.state.clear(id)?;
                Err(e)
            }
        }
    }

    /// Arms (or confirms) the dead-man timer for `id`; it runs this same `vi`
    /// as `vi cloud deadman fire <id>`.
    fn arm_deadman(&self, id: u64, start: u64, cap_at: u64) -> Result<deadman::Record> {
        let exe = std::env::current_exe()
            .map_err(|e| Error::Worker(format!("cannot find the vi executable: {e}")))?;
        let r = deadman::arm(
            &self.host,
            &self.state,
            deadman::fire_argv(&exe, id)?,
            id,
            start,
            cap_at,
        )?;
        let now = self.host.now();
        self.host.log(&format!(
            "dead-man timer {}.timer fires at {} (in {} min; hard cap in {} min) unless \
             instance {id} is verified destroyed first",
            r.unit,
            r.fire_at,
            r.fire_at.saturating_sub(now) / 60,
            r.cap_at.saturating_sub(now) / 60
        ));
        Ok(r)
    }

    /// The heartbeat (`vi cloud deadman extend`): see [`deadman::extend`].
    pub fn deadman_extend(&self, id: u64, grace: u64) -> Result<deadman::Record> {
        deadman::extend(&self.host, &self.state, id, grace)
    }

    /// What the dead-man timer runs (`vi cloud deadman fire <id>`): destroy
    /// `id`, then verify. When `id` is the tracked worker this is exactly
    /// `down_verified(Some(id))`, labelled sweep included. When home tracks a
    /// different (newer) instance, or none, only `id` is destroyed, so a stale
    /// timer can never take down a later rental. On a verified destroy the
    /// timer is cancelled except for its running service, which is this
    /// process; a nonzero exit leaves it to be retried by systemd.
    pub fn deadman_fire(&self, id: u64) -> Result<Down> {
        let tracked = self.state.load().unwrap_or(None);
        let sweep = tracked.as_ref().is_some_and(|t| t.id == id);
        self.destroy_verified(Some(id), sweep, Some(id))
    }

    /// The money guards that need the chosen offer: the worst case (`max_hours`
    /// at this price) must stay under `cap`, and the account must already hold
    /// at least `cap`. Returns the credit for the confirmation and spend report.
    fn check_budget(&self, offer: &Offer, cap: f64) -> Result<f64> {
        let worst = self.cfg.max_hours * offer.dph_total;
        if worst > cap {
            return Err(Error::Worker(format!(
                "worst case ${worst:.2} ({:.1} h at ${:.3}/hr) exceeds the ${cap:.2} cap",
                self.cfg.max_hours, offer.dph_total
            )));
        }
        let credit = self.vast.credit()?;
        if credit < cap {
            return Err(Error::Worker(format!(
                "credit ${credit:.2} is below the ${cap:.2} cap"
            )));
        }
        Ok(credit)
    }

    /// Re-reads the tracked instance from the API and saves its current ssh
    /// endpoint. This is the same endpoint discovery `wait` uses, but without
    /// the ssh reachability probe: the home supervisor calls it while polls are
    /// failing or a host key changed, when ssh cannot answer yet. It never
    /// rents and never switches instances.
    pub fn refresh(&self) -> Result<Option<Tracked>> {
        let Some(mut t) = self.state.load()? else {
            return Ok(None);
        };
        let inst = self.vast.instance(t.id)?.ok_or_else(|| {
            Error::Worker(format!(
                "worker {} is gone; `vi cloud up` rents a new one",
                t.id
            ))
        })?;
        let ep = inst
            .endpoints()
            .into_iter()
            .next()
            .ok_or_else(|| Error::Worker(format!("worker {} has no ssh endpoint yet", t.id)))?;
        if t.endpoint.as_ref() != Some(&ep) {
            self.host.log(&format!(
                "worker {} endpoint is now {}:{}",
                t.id, ep.host, ep.port
            ));
            t.endpoint = Some(ep);
            self.state.save(&t)?;
        }
        Ok(Some(t))
    }

    /// Starts the tracked worker again (`vi cloud start`). The approval guards
    /// are `up`'s: the deadline must be open, the worst case (`max_hours` at
    /// the instance's price) must stay under `cap`, and the account must hold
    /// at least `cap`. Restarting does not reset the tracked `created` time, so
    /// `reap` still measures `max_hours` from the original rental.
    pub fn start(&self, cap: f64, until: u64) -> Result<Tracked> {
        let Some(mut t) = self.state.load()? else {
            return Err(Error::Worker(
                "no worker is rented; there is nothing to start".into(),
            ));
        };
        if self.host.now() >= until {
            return Err(Error::Worker(
                "the approval deadline has passed; pass a later --until".into(),
            ));
        }
        let inst = self.vast.instance(t.id)?.ok_or_else(|| {
            Error::Worker(format!(
                "worker {} is gone; `vi cloud up` rents a new one",
                t.id
            ))
        })?;
        let dph = if inst.dph_total > 0.0 {
            inst.dph_total
        } else {
            t.dph
        };
        let worst = self.cfg.max_hours * dph;
        if worst > cap {
            return Err(Error::Worker(format!(
                "worst case ${worst:.2} ({:.1} h at ${:.3}/hr) exceeds the ${cap:.2} cap",
                self.cfg.max_hours, dph
            )));
        }
        let credit = self.vast.credit()?;
        if credit < cap {
            return Err(Error::Worker(format!(
                "credit ${credit:.2} is below the ${cap:.2} cap"
            )));
        }
        self.vast.start(t.id)?;
        self.host.log(&format!(
            "asked Vast to start worker {} again; worst case ${worst:.2} at ${dph:.3}/hr",
            t.id
        ));
        if let Some(ep) = inst.endpoints().into_iter().next() {
            t.endpoint = Some(ep);
        }
        self.touch_tracked(t)
    }

    pub fn create_request(&self) -> Value {
        json!({
            "client_id": "me",
            "image": self.cfg.image,
            "disk": self.cfg.disk_gb,
            "label": self.cfg.label,
            "onstart": onstart::render(&self.cfg),
            "runtype": "ssh_direc ssh_proxy",
        })
    }

    fn wait(&self, mut t: Tracked) -> Result<Tracked> {
        let deadline = self.host.now() + self.cfg.boot_minutes * 60;
        let mut last_status = String::new();
        loop {
            let inst = self
                .vast
                .instance(t.id)?
                .ok_or_else(|| Error::Worker("the instance disappeared while booting".into()))?;
            let status = inst.status().to_owned();
            if status == "running" {
                let known_hosts = self.state.known_hosts(t.id);
                let ep = inst
                    .endpoints()
                    .into_iter()
                    .find(|ep| self.host.reachable(ep, &known_hosts));
                if let Some(ep) = ep {
                    t.endpoint = Some(ep);
                    return self.touch_tracked(t);
                }
            } else if let Some(why) = failed(&inst) {
                return Err(Error::Worker(format!("it failed to start: {why}")));
            }
            if self.host.now() >= deadline {
                return Err(Error::Worker(format!(
                    "no ssh after {} minutes (status {status})",
                    self.cfg.boot_minutes
                )));
            }
            if status != last_status {
                self.host.log(&format!("  {status}"));
                last_status = status;
            }
            self.host.sleep(POLL_SECS);
        }
    }

    /// Accepts the worker's new ssh host key after a rotation (R50). A new key
    /// is trusted only when `ssh-keyscan` presents it now *and* its fingerprint
    /// and type appear in the newest VI-HOSTKEY batch the instance logged after
    /// the last accepted stamp. A mismatch or a timeout refuses: the caller
    /// halts pushes and escalates instead of accepting blindly. Only this
    /// instance's `known_hosts-<id>` is rewritten. After a rotation only a
    /// reattach continues; no inputs are sent.
    pub fn accept_hostkey(&self, wait_secs: u64) -> Result<Rotation> {
        let Some(mut t) = self.state.load()? else {
            return Err(Error::Worker(
                "no worker is rented; there is nothing to check".into(),
            ));
        };
        let inst = self.vast.instance(t.id)?.ok_or_else(|| {
            Error::Worker(format!("worker {} is gone; refusing the host key", t.id))
        })?;
        if inst.status() != "running" {
            return Err(Error::Worker(format!(
                "worker {} is {}; refusing the host key",
                t.id,
                inst.status()
            )));
        }
        let ep = inst
            .endpoints()
            .into_iter()
            .next()
            .ok_or_else(|| Error::Worker(format!("worker {} has no ssh endpoint", t.id)))?;

        // R50 step 2: what the worker presents right now.
        let lines = self.host.keyscan(&ep);
        if lines.is_empty() {
            return Err(Error::Worker(format!(
                "ssh-keyscan found no host key at {}:{}; refusing",
                ep.host, ep.port
            )));
        }
        let fingerprints = self.host.key_fingerprints(&lines);
        if fingerprints.len() != lines.len() {
            return Err(Error::Worker(
                "could not fingerprint every presented host key; refusing".into(),
            ));
        }
        let mut presented = Vec::new();
        for (line, fp) in lines.iter().zip(&fingerprints) {
            let Some((fingerprint, key_type)) = parse_keygen_line(fp) else {
                return Err(Error::Worker(format!(
                    "unexpected ssh-keygen output {fp:?}; refusing"
                )));
            };
            if line.split_whitespace().count() < 3 {
                return Err(Error::Worker(format!(
                    "unexpected ssh-keyscan output {line:?}; refusing"
                )));
            }
            presented.push(PresentedKey {
                line: line.clone(),
                fingerprint,
                key_type,
            });
        }

        // R50 step 3: the boot script logs each host key it generated, stamped
        // with the boot time. Vast needs a moment to make the log available.
        // Only a batch newer than the last accepted stamp may be trusted, so a
        // stale log cannot re-authorise an old key.
        let last = self.state.last_hostkey_stamp(t.id)?;
        let deadline = self.host.now() + wait_secs;
        loop {
            let log = self.vast.logs(t.id)?;
            if let Some(batch) = newest_hostkey_batch(&log, last) {
                // R50 step 4: accept only a presented key whose type and
                // fingerprint are in the batch.
                let accepted: Vec<PresentedKey> = presented
                    .iter()
                    .filter(|key| {
                        batch
                            .keys
                            .iter()
                            .any(|(fp, ty)| fp == &key.fingerprint && ty == &key.key_type)
                    })
                    .cloned()
                    .collect();
                if accepted.is_empty() {
                    return Err(Error::Worker(format!(
                        "the worker presents {}, but the newest VI-HOSTKEY batch \
                         (stamp {}) does not match; refusing the new host key",
                        describe_keys(&presented),
                        batch.stamp
                    )));
                }
                self.rewrite_known_hosts(t.id, &accepted)?;
                self.state.save_hostkey_stamp(t.id, batch.stamp)?;
                if t.endpoint.as_ref() != Some(&ep) {
                    t.endpoint = Some(ep.clone());
                    self.state.save(&t)?;
                }
                self.host.log(&format!(
                    "accepted a host-key rotation for instance {} (stamp {}): {}; \
                     only a reattach continues, no inputs are sent",
                    t.id,
                    batch.stamp,
                    describe_keys(&accepted)
                ));
                return Ok(Rotation {
                    endpoint: ep,
                    stamp: batch.stamp,
                    fingerprints: accepted.iter().map(|k| k.fingerprint.clone()).collect(),
                });
            }
            if self.host.now() >= deadline {
                return Err(Error::Worker(format!(
                    "no new VI-HOSTKEY line from worker {} within {wait_secs} seconds; \
                     refusing the new host key",
                    t.id
                )));
            }
            self.host.sleep(30);
        }
    }

    /// Removes this instance's old entry for the scanned host and appends the
    /// accepted key lines. `ssh-keygen -R` never touches other hosts or other
    /// instances' files.
    fn rewrite_known_hosts(&self, id: u64, keys: &[PresentedKey]) -> Result<()> {
        use std::io::Write as _;

        let path = self.state.known_hosts(id);
        let host = keys[0]
            .line
            .split_whitespace()
            .next()
            .ok_or_else(|| Error::Worker("the ssh-keyscan line has no host".into()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(crate::io_err(parent))?;
        }
        if !path.exists() {
            std::fs::write(&path, b"").map_err(crate::io_err(&path))?;
        }
        let status = Command::new("ssh-keygen")
            .args(["-R", host, "-f"])
            .arg(&path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if !status.is_ok_and(|s| s.success()) {
            return Err(Error::Worker(format!(
                "could not remove the old host key for {host} from {}; refusing",
                path.display()
            )));
        }
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(crate::io_err(&path))?;
        for key in keys {
            writeln!(file, "{}", key.line).map_err(crate::io_err(&path))?;
        }
        Ok(())
    }

    /// Destroys the tracked worker and any other instance carrying our label.
    /// Refuses while an unverified checkpoint marker is present, so a pushed-home
    /// checkpoint isn't lost before it's been verified; `down_force` overrides
    /// that. Reports the credit before and after for the spend report.
    pub fn down(&self) -> Result<Down> {
        let markers = self.state.markers()?;
        if !markers.is_empty() {
            return Err(Error::Worker(unverified_error(&markers)));
        }
        self.release(Vec::new())
    }

    /// `down`, ignoring unverified markers. Returns the ids destroyed and the
    /// markers it overrode (they stay in place until a later verify clears them).
    pub fn down_force(&self) -> Result<Down> {
        let overridden = self.state.markers()?;
        self.release(overridden)
    }

    /// Destroys the worker and reports the credit before and after, so the CLI
    /// can show what the rental actually cost.
    fn release(&self, overridden: Vec<Marker>) -> Result<Down> {
        let credit_before = self.state.load()?.map(|t| t.credit);
        let ids = self.destroy()?;
        let credit_after = if ids.is_empty() {
            None
        } else {
            self.vast.credit().ok()
        };
        Ok(Down {
            ids,
            overridden,
            credit_before,
            credit_after,
            verified: false,
            still_listed: Vec::new(),
            errors: Vec::new(),
        })
    }

    /// The automated release: destroy first, then prove it. Unverified
    /// markers never stop it; they are returned in `overridden` so the caller
    /// can record them. Destroys the tracked worker, `id` if given (the id the
    /// caller cached when it rented) and every instance carrying our label,
    /// then lists the account's instances until none of them is listed (or
    /// each shows `destroyed`), for up to `VERIFY_SECS`. An instance still
    /// listed gets its DELETE again every `REDELETE_SECS`. Only a listing
    /// counts as proof, so if the API stays unreachable the result has
    /// `verified == false`. Failures along the way are collected in `errors`
    /// rather than returned, so one bad call never skips the rest.
    /// Once the destroy is verified, the dead-man timer of every destroyed id
    /// is cancelled; an unverified destroy leaves the timers armed.
    pub fn down_verified(&self, id: Option<u64>) -> Result<Down> {
        self.destroy_verified(id, true, None)
    }

    /// `down_verified`, optionally limited to `id` alone (`sweep == false`:
    /// no tracked worker, no labelled instances). `firing` is the id whose
    /// dead-man service is running this destroy; its service is not stopped.
    fn destroy_verified(&self, id: Option<u64>, sweep: bool, firing: Option<u64>) -> Result<Down> {
        let mut errors = Vec::new();
        let overridden = self.state.markers().unwrap_or_else(|e| {
            errors.push(format!("could not read the unverified markers: {e}"));
            Vec::new()
        });
        let tracked = if sweep {
            self.state.load().unwrap_or_else(|e| {
                errors.push(format!("could not read the tracked worker: {e}"));
                None
            })
        } else {
            None
        };
        let mut ids: Vec<u64> = tracked.iter().map(|t| t.id).collect();
        if let Some(id) = id {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        if sweep {
            match self.vast.instances() {
                Ok(list) => self.add_ours(&mut ids, &list),
                Err(e) => errors.push(format!("listing instances before the destroy: {e}")),
            }
        }
        for &i in &ids {
            self.delete_noting(i, &mut errors);
        }
        let mut last_delete = self.host.now();
        let deadline = last_delete + VERIFY_SECS;
        let mut still_listed = ids.clone();
        let mut verified = false;
        loop {
            match self.vast.instances() {
                Ok(list) => {
                    let before = ids.len();
                    if sweep {
                        self.add_ours(&mut ids, &list);
                    }
                    for &i in &ids[before..] {
                        // Appeared since the first listing (a rent that raced us).
                        self.delete_noting(i, &mut errors);
                    }
                    still_listed = ids
                        .iter()
                        .copied()
                        .filter(|i| {
                            list.iter()
                                .any(|inst| inst.id == *i && inst.status() != "destroyed")
                        })
                        .collect();
                    if still_listed.is_empty() {
                        verified = true;
                        break;
                    }
                }
                Err(e) => errors.push(format!("listing instances to verify the destroy: {e}")),
            }
            if self.host.now() >= deadline {
                break;
            }
            self.host.sleep(POLL_SECS);
            if self.host.now().saturating_sub(last_delete) >= REDELETE_SECS {
                for &i in &still_listed {
                    self.host.log(&format!(
                        "instance {i} is still listed; sending DELETE again"
                    ));
                    self.delete_noting(i, &mut errors);
                }
                last_delete = self.host.now();
            }
        }
        if verified {
            if let Some(t) = &tracked {
                if let Err(e) = self.state.clear(t.id) {
                    errors.push(format!("could not clear the tracked worker: {e}"));
                }
            }
            // Only now, with Vast's list as proof, may the timers go.
            for &i in &ids {
                if let Err(e) = deadman::cancel(&self.host, &self.state, i, firing != Some(i)) {
                    errors.push(format!("could not cancel the dead-man timer for {i}: {e}"));
                }
            }
        }
        let credit_after = if ids.is_empty() {
            None
        } else {
            self.vast.credit().ok()
        };
        Ok(Down {
            ids,
            overridden,
            credit_before: tracked.map(|t| t.credit),
            credit_after,
            verified,
            still_listed: if verified { Vec::new() } else { still_listed },
            errors,
        })
    }

    /// Appends to `ids` every instance in `list` that carries our label.
    fn add_ours(&self, ids: &mut Vec<u64>, list: &[Instance]) {
        for inst in list {
            if self.is_ours(inst) && !ids.contains(&inst.id) {
                ids.push(inst.id);
            }
        }
    }

    /// One DELETE (with the API client's own retries); a failure is logged
    /// and noted, never returned, so the verify still runs.
    fn delete_noting(&self, id: u64, errors: &mut Vec<String>) {
        if let Err(e) = self.vast.destroy(id) {
            self.host
                .log(&format!("DELETE for instance {id} failed: {e}"));
            errors.push(format!("DELETE instance {id}: {e}"));
        }
    }

    /// The actual destruction, without consulting markers. `reap` uses this so
    /// the idle/max-hours backstop can still release the worker no matter what.
    fn destroy(&self) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        let tracked = self.state.load()?;
        if let Some(t) = &tracked {
            self.vast.destroy(t.id)?;
            ids.push(t.id);
        }
        for inst in self.vast.instances()? {
            if self.is_ours(&inst) && !ids.contains(&inst.id) {
                self.vast.destroy(inst.id)?;
                ids.push(inst.id);
            }
        }
        if let Some(t) = tracked {
            self.state.clear(t.id)?;
        }
        Ok(ids)
    }

    /// Releases the worker if it has been idle for `idle_minutes` or up for
    /// `max_hours`. Returns why, if it did. An idle release only happens when
    /// `activity` says the worker is idle; a busy worker, or one whose activity
    /// can't be checked, is kept (the `max_hours` backstop still applies).
    pub fn reap(&self, activity: &dyn Activity) -> Result<Option<String>> {
        let Some(t) = self.state.load()? else {
            return Ok(None);
        };
        let now = self.host.now();
        let idle = now.saturating_sub(t.last_used) / 60;
        let up = now.saturating_sub(t.created) / 60;
        let max = self.cfg.max_minutes();
        let idle_release = idle >= self.cfg.idle_minutes && self.idle_confirmed(&t, activity);
        let why = if idle_release {
            format!("idle for {idle} minutes")
        } else if max > 0 && up >= max {
            format!("up for {up} minutes, the max_hours limit")
        } else {
            return Ok(None);
        };
        // The money backstop ignores markers: a worker with an unverified
        // checkpoint must still be released rather than kept billing.
        self.destroy()?;
        Ok(Some(why))
    }

    /// True only when the probe says the worker is idle. A busy worker, or one
    /// whose activity cannot be checked (including one with no endpoint yet),
    /// is never released for idleness: reaping on a guess can kill a live
    /// session.
    fn idle_confirmed(&self, t: &Tracked, activity: &dyn Activity) -> bool {
        let Some(ep) = &t.endpoint else {
            return false;
        };
        match activity.busy(ep, &self.state.known_hosts(t.id)) {
            Ok(false) => true,
            Ok(true) => false,
            Err(e) => {
                self.host.log(&format!(
                    "vi cloud reap: keeping worker {}; its activity could not be checked: {e}",
                    t.id
                ));
                false
            }
        }
    }

    /// Marks the worker as in use now. Returns it, if there is one.
    pub fn touch(&self) -> Result<Option<Tracked>> {
        match self.state.load()? {
            Some(t) => self.touch_tracked(t).map(Some),
            None => Ok(None),
        }
    }

    fn touch_tracked(&self, mut t: Tracked) -> Result<Tracked> {
        t.last_used = self.host.now();
        self.state.save(&t)?;
        Ok(t)
    }

    pub fn report(&self) -> Result<Report> {
        let tracked = self.state.load()?;
        let instance = match &tracked {
            Some(t) => self.vast.instance(t.id)?,
            None => None,
        };
        let tracked_id = tracked.as_ref().map(|t| t.id);
        let (orphans, others) = self
            .vast
            .instances()?
            .into_iter()
            .filter(|i| Some(i.id) != tracked_id)
            .partition(|i| self.is_ours(i));
        Ok(Report {
            tracked,
            instance,
            orphans,
            others,
            credit: self.vast.credit().ok(),
        })
    }

    /// Shell assignments `jobs/on-worker.sh` evals to reach the worker.
    pub fn env(&self) -> Result<String> {
        let t = self
            .touch()?
            .ok_or_else(|| Error::Worker("no worker is rented; run `vi cloud up` first".into()))?;
        let ep = t.endpoint.as_ref().ok_or_else(|| {
            Error::Worker(format!(
                "worker {} hasn't finished booting; run `vi cloud up` to wait for it",
                t.id
            ))
        })?;
        let vars = [
            ("VI_WORKER_ID", t.id.to_string()),
            ("VI_WORKER_HOST", ep.host.clone()),
            ("VI_WORKER_PORT", ep.port.to_string()),
            (
                "VI_WORKER_KNOWN_HOSTS",
                self.state.known_hosts(t.id).display().to_string(),
            ),
        ];
        Ok(vars
            .iter()
            .map(|(k, v)| format!("{k}={}\n", shell_quote(v)))
            .collect())
    }

    fn is_ours(&self, inst: &Instance) -> bool {
        inst.label.as_deref() == Some(self.cfg.label.as_str())
    }
}

/// Stopped instances still bill for disk, and a stateless worker loses
/// nothing by being replaced, so `up` destroys them instead of restarting.
fn stopped(inst: &Instance) -> bool {
    inst.status() == "exited" || inst.intended_status.as_deref() == Some("stopped")
}

fn describe_keys(keys: &[PresentedKey]) -> String {
    keys.iter()
        .map(|k| format!("{} {}", k.key_type, k.fingerprint))
        .collect::<Vec<_>>()
        .join(", ")
}

fn failed(inst: &Instance) -> Option<String> {
    let msg = inst.status_msg.as_deref().unwrap_or("").trim();
    if inst.status() == "exited" || stopped(inst) {
        Some(format!("status {}: {msg}", inst.status()))
    } else if msg.starts_with("Error") {
        Some(msg.lines().next().unwrap_or(msg).to_owned())
    } else {
        None
    }
}

pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// The message `down` returns when unverified markers are in the way.
fn unverified_error(markers: &[Marker]) -> String {
    let mut msg = String::from(
        "an unverified checkpoint is still at home; verify it before destroying the worker:\n",
    );
    for m in markers {
        let _ = writeln!(msg, "  {}: {}", m.stage, m.path.display());
    }
    msg.push_str(
        "run `python3 jobs/checkpoints.py verify <checkpoint> --expect N` for each, \
         or `vi cloud down --force` to destroy anyway",
    );
    msg
}

/// The message `up` returns when unverified markers would be destroyed with the
/// stopped worker it wants to replace.
fn unverified_up_error(markers: &[Marker]) -> String {
    let mut msg = String::from(
        "an unverified checkpoint is still at home; verify it before replacing the worker:\n",
    );
    for m in markers {
        let _ = writeln!(msg, "  {}: {}", m.stage, m.path.display());
    }
    msg.push_str(
        "run `python3 jobs/checkpoints.py verify <checkpoint> --expect N` for each, \
         or `vi cloud down --force` the old worker first",
    );
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// A cap large enough for the default test offers (dph 0.15–0.2 over
    /// `max_hours` 4) and below the fake account credit of $12, so the budget
    /// guards pass for the tests that don't exercise them.
    const CAP: f64 = 10.0;
    /// Far in the future, for the tests that don't exercise the deadline.
    const UNTIL: u64 = 10_000_000;

    struct FakeVast {
        offers: Vec<Offer>,
        /// Status each created instance reports, in creation order.
        boots: RefCell<Vec<&'static str>>,
        live: RefCell<Vec<Instance>>,
        created: RefCell<Vec<(u64, Value)>>,
        destroyed: RefCell<Vec<u64>>,
        credit: Cell<f64>,
    }

    impl Default for FakeVast {
        fn default() -> Self {
            FakeVast {
                offers: Vec::new(),
                boots: RefCell::new(Vec::new()),
                live: RefCell::new(Vec::new()),
                created: RefCell::new(Vec::new()),
                destroyed: RefCell::new(Vec::new()),
                credit: Cell::new(12.0),
            }
        }
    }

    impl Vast for &FakeVast {
        fn offers(&self, q: &Value) -> Result<Vec<Offer>> {
            Ok(match q["ask_contract_id"]["eq"].as_u64() {
                Some(id) => self.offers.iter().filter(|o| o.id == id).cloned().collect(),
                None => self.offers.clone(),
            })
        }
        fn create(&self, offer: u64, req: &Value) -> Result<u64> {
            let id = 100 + self.created.borrow().len() as u64;
            self.created.borrow_mut().push((offer, req.clone()));
            let status = self.boots.borrow_mut().remove(0);
            self.live.borrow_mut().push(Instance {
                id,
                label: req["label"].as_str().map(Into::into),
                actual_status: Some(status.into()),
                status_msg: (status == "loading").then(|| "Error: bad GPU".into()),
                ssh_host: Some("ssh1.vast.ai".into()),
                ssh_port: Some(2000 + id as u16),
                ..Instance::default()
            });
            Ok(id)
        }
        fn instance(&self, id: u64) -> Result<Option<Instance>> {
            Ok(self.live.borrow().iter().find(|i| i.id == id).cloned())
        }
        fn instances(&self) -> Result<Vec<Instance>> {
            Ok(self.live.borrow().clone())
        }
        fn destroy(&self, id: u64) -> Result<()> {
            self.live.borrow_mut().retain(|i| i.id != id);
            self.destroyed.borrow_mut().push(id);
            Ok(())
        }
        fn credit(&self) -> Result<f64> {
            Ok(self.credit.get())
        }
    }

    #[derive(Default)]
    struct FakeHost {
        now: Cell<u64>,
        /// Loaded timers: unit -> fire time.
        timers: RefCell<std::collections::BTreeMap<String, u64>>,
        /// Every timer call, in order, e.g. "arm vi-deadman-100 @5000".
        timer_calls: RefCell<Vec<String>>,
        /// When set, the user manager "does not answer" / systemd-run fails.
        timer_down: Cell<bool>,
        /// The next this many `systemd-run` calls fail.
        arm_failures: Cell<u32>,
    }

    impl Host for &FakeHost {
        fn now(&self) -> u64 {
            self.now.get()
        }
        fn sleep(&self, secs: u64) {
            self.now.set(self.now.get() + secs);
        }
        fn reachable(&self, _: &Endpoint, _: &Path) -> bool {
            true
        }
        fn log(&self, _: &str) {}
        fn timer_ready(&self) -> Result<()> {
            if self.timer_down.get() {
                return Err(Error::Worker("user manager down".into()));
            }
            Ok(())
        }
        fn timer_loaded(&self, unit: &str) -> Result<bool> {
            self.timer_ready()?;
            Ok(self.timers.borrow().contains_key(unit))
        }
        fn timer_arm(&self, r: &deadman::Record) -> Result<()> {
            self.timer_ready()?;
            // Exercise the real argument builder and its refusals.
            deadman::systemd_run_args(r, self.now.get())?;
            if self.arm_failures.get() > 0 || self.timers.borrow().contains_key(&r.unit) {
                self.arm_failures
                    .set(self.arm_failures.get().saturating_sub(1));
                return Err(Error::Worker("systemd-run failed".into()));
            }
            self.timer_calls
                .borrow_mut()
                .push(format!("arm {} @{}", r.unit, r.fire_at));
            self.timers.borrow_mut().insert(r.unit.clone(), r.fire_at);
            Ok(())
        }
        fn timer_stop(&self, unit: &str, service: bool) -> Result<()> {
            self.timer_ready()?;
            self.timer_calls
                .borrow_mut()
                .push(format!("stop {unit} service={service}"));
            self.timers.borrow_mut().remove(unit);
            Ok(())
        }
    }

    /// The fixed answer the reap tests inject instead of the real ssh probe.
    enum Answer {
        Idle,
        Busy,
        Failed,
    }

    struct FakeActivity(Answer);

    impl Activity for FakeActivity {
        fn busy(&self, _: &Endpoint, _: &Path) -> Result<bool> {
            match self.0 {
                Answer::Idle => Ok(false),
                Answer::Busy => Ok(true),
                Answer::Failed => Err(Error::Worker("probe unavailable".into())),
            }
        }
    }

    fn offer(id: u64, machine: u64, dph: f64) -> Offer {
        Offer {
            id,
            machine_id: machine,
            host_id: None,
            gpu_name: "RTX 3090".into(),
            gpu_ram: 24576.0,
            dph_total: dph,
            reliability: 0.99,
            reliability2: 0.99,
            inet_down: 900.0,
            inet_up: 500.0,
            cpu_cores_effective: 16.0,
            cpu_ram: 128000.0,
            disk_bw: 1000.0,
            disk_space: 100.0,
            geolocation: None,
            compute_cap: 800,
            cuda_max_good: 13.0,
        }
    }

    fn cloud<'a>(
        vast: &'a FakeVast,
        host: &'a FakeHost,
        dir: &Path,
    ) -> Cloud<&'a FakeVast, &'a FakeHost> {
        Cloud {
            cfg: Config::default(),
            vast,
            state: StateDir::new(dir),
            host,
        }
    }

    fn someone_elses() -> Instance {
        Instance {
            id: 5,
            label: None,
            actual_status: Some("created".into()),
            intended_status: Some("stopped".into()),
            ..Instance::default()
        }
    }

    #[test]
    fn up_rents_the_cheapest_and_down_releases_only_ours() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15), offer(2, 20, 0.2)],
            boots: RefCell::new(vec!["running"]),
            live: RefCell::new(vec![someone_elses()]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());

        let mut seen = None;
        let t = c
            .up(None, CAP, UNTIL, |o, credit| {
                seen = Some((o.id, credit));
                true
            })
            .unwrap();
        assert_eq!(seen, Some((1, Some(12.0))));
        assert_eq!(t.id, 100);
        assert_eq!(t.endpoint.as_ref().unwrap().port, 2100);
        let (rented, req) = &vast.created.borrow()[0];
        assert_eq!(*rented, 1);
        assert_eq!(req["label"], "vi-worker");
        assert!(req["onstart"].as_str().unwrap().contains("vi-watchdog"));

        // A second `up` reuses it without renting or asking.
        let again = c
            .up(None, CAP, UNTIL, |_, _| panic!("asked to rent again"))
            .unwrap();
        assert_eq!(again.id, 100);
        assert!(c.env().unwrap().contains("VI_WORKER_PORT='2100'"));

        assert_eq!(c.down().unwrap().ids, vec![100]);
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert_eq!(c.state.load().unwrap(), None);
        // The stopped instance from another project is reported, not touched.
        let report = c.report().unwrap();
        assert_eq!(report.others.len(), 1);
        assert!(report.tracked.is_none());
    }

    #[test]
    fn offers_excluding_skips_failed_machines_and_hosts() {
        let dir = tempfile::tempdir().unwrap();
        let on_host = |id, machine, host, dph| Offer {
            host_id: host,
            ..offer(id, machine, dph)
        };
        let vast = FakeVast {
            offers: vec![
                on_host(1, 10, Some(500), 0.10), // the failed host, another machine
                on_host(2, 20, Some(600), 0.12),
                on_host(3, 30, None, 0.13), // no host id: kept
                on_host(4, 40, Some(500), 0.14),
                on_host(5, 50, Some(700), 0.15),
            ],
            ..FakeVast::default()
        };
        let host = FakeHost::default();
        let c = cloud(&vast, &host, dir.path());

        // Default: nothing excluded, the whole answer.
        let all: Vec<u64> = c.offers(10).unwrap().iter().map(|o| o.id).collect();
        assert_eq!(all, vec![1, 2, 3, 4, 5]);
        // Every offer of host 500 is gone, the rest keep their order.
        let ids: Vec<u64> = c
            .offers_excluding(10, &[], &[500])
            .unwrap()
            .iter()
            .map(|o| o.id)
            .collect();
        assert_eq!(ids, vec![2, 3, 5]);
        // The limit counts what is listed after the host filter, not before it.
        let ids: Vec<u64> = c
            .offers_excluding(2, &[], &[500])
            .unwrap()
            .iter()
            .map(|o| o.id)
            .collect();
        assert_eq!(ids, vec![2, 3]);
        let ids: Vec<u64> = c
            .offers_excluding(10, &[], &[500, 600, 700])
            .unwrap()
            .iter()
            .map(|o| o.id)
            .collect();
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn a_failed_boot_is_returned_without_trying_another_offer() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15), offer(2, 20, 0.2)],
            boots: RefCell::new(vec!["loading"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        let err = c.up(None, CAP, UNTIL, |_, _| true).unwrap_err();
        assert!(err.to_string().contains("Error: bad GPU"), "{err}");
        // Only the cheapest offer was rented, and it was given back.
        let rented: Vec<u64> = vast.created.borrow().iter().map(|c| c.0).collect();
        assert_eq!(rented, vec![1]);
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert!(c.state.load().unwrap().is_none());
    }

    #[test]
    fn declining_rents_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        assert!(c.up(None, CAP, UNTIL, |_, _| false).is_err());
        assert!(vast.created.borrow().is_empty());
    }

    #[test]
    fn reap_waits_for_the_idle_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        host.now.set(1000 + 19 * 60);
        assert_eq!(c.reap(&FakeActivity(Answer::Idle)).unwrap(), None);
        c.touch().unwrap();
        host.now.set(1000 + 38 * 60);
        assert_eq!(c.reap(&FakeActivity(Answer::Idle)).unwrap(), None);
        host.now.set(1000 + 39 * 60);
        assert_eq!(
            c.reap(&FakeActivity(Answer::Idle)).unwrap().unwrap(),
            "idle for 20 minutes"
        );
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert_eq!(c.reap(&FakeActivity(Answer::Idle)).unwrap(), None);
    }

    #[test]
    fn reap_keeps_a_worker_the_probe_sees_busy() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        host.now.set(1000 + 21 * 60);
        assert_eq!(c.reap(&FakeActivity(Answer::Busy)).unwrap(), None);
        assert!(vast.destroyed.borrow().is_empty());
        assert!(c.state.load().unwrap().is_some());
    }

    #[test]
    fn reap_releases_a_worker_the_probe_sees_idle() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        host.now.set(1000 + 21 * 60);
        assert_eq!(
            c.reap(&FakeActivity(Answer::Idle)).unwrap().unwrap(),
            "idle for 21 minutes"
        );
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert_eq!(c.state.load().unwrap(), None);
    }

    #[test]
    fn reap_keeps_a_worker_when_the_probe_fails() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        host.now.set(1000 + 21 * 60);
        assert_eq!(c.reap(&FakeActivity(Answer::Failed)).unwrap(), None);
        assert!(vast.destroyed.borrow().is_empty());
        assert!(c.state.load().unwrap().is_some());
    }

    #[test]
    fn reap_enforces_max_hours_even_when_busy() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(0),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();
        for _ in 0..24 {
            host.now.set(host.now.get() + 10 * 60);
            c.touch().unwrap();
        }
        assert!(c
            .reap(&FakeActivity(Answer::Idle))
            .unwrap()
            .unwrap()
            .contains("max_hours"));
    }

    #[test]
    fn max_hours_releases_even_when_the_probe_sees_busy() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(0),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        host.now.set(24 * 60 * 60);
        assert!(c
            .reap(&FakeActivity(Answer::Busy))
            .unwrap()
            .unwrap()
            .contains("max_hours"));
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert_eq!(c.state.load().unwrap(), None);
    }

    #[test]
    fn quoting_survives_single_quotes() {
        assert_eq!(shell_quote("a'b"), r"'a'\''b'");
    }

    #[test]
    fn down_refuses_with_an_unverified_marker_and_force_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        let marker = dir.path().join("unverified").join("voicetune");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(
            &marker,
            "/home/brandon/corpus/stages/voicetune/.checkpoint-x\n",
        )
        .unwrap();

        // Refused, and nothing is destroyed on the fake Vast.
        let err = c.down().unwrap_err().to_string();
        assert!(err.contains("voicetune"), "{err}");
        assert!(
            err.contains("/home/brandon/corpus/stages/voicetune/.checkpoint-x"),
            "{err}"
        );
        assert!(err.contains("checkpoints.py verify"), "{err}");
        assert!(vast.destroyed.borrow().is_empty());

        // Force destroys and reports the overridden marker, leaving it in place.
        let down = c.down_force().unwrap();
        assert_eq!(down.ids, vec![100]);
        assert_eq!(down.overridden.len(), 1);
        assert_eq!(down.overridden[0].stage, "voicetune");
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert!(marker.exists());
    }

    #[test]
    fn down_is_allowed_once_the_marker_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        let marker = dir.path().join("unverified").join("voicetune");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, "/checkpoint\n").unwrap();
        assert!(c.down().is_err());

        std::fs::remove_file(&marker).unwrap();
        assert_eq!(c.down().unwrap().ids, vec![100]);
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
    }

    #[test]
    fn up_refuses_to_replace_a_stopped_worker_with_an_unverified_marker() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();

        // Stop the worker and leave an unverified marker: `up` must refuse to
        // destroy the stopped worker rather than rent over it.
        vast.live.borrow_mut()[0].actual_status = Some("exited".into());
        let marker = dir.path().join("unverified").join("voicetune");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(
            &marker,
            "/home/brandon/corpus/stages/voicetune/.checkpoint-x\n",
        )
        .unwrap();

        let err = c
            .up(None, CAP, UNTIL, |_, _| {
                panic!("must not rent over a marker")
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("voicetune"), "{err}");
        assert!(err.contains("checkpoints.py verify"), "{err}");
        assert!(err.contains("down --force"), "{err}");
        assert!(vast.destroyed.borrow().is_empty());
        assert_eq!(vast.created.borrow().len(), 1); // only the original rental
    }

    #[test]
    fn up_refuses_once_the_deadline_has_passed() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        let err = c
            .up(None, CAP, 999, |_, _| {
                panic!("must not ask after the deadline")
            })
            .unwrap_err();
        assert!(err.to_string().contains("deadline"), "{err}");
        assert!(vast.created.borrow().is_empty());
    }

    #[test]
    fn up_refuses_when_the_worst_case_exceeds_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        // max_hours (4) at $0.15/hr is $0.60, so a $0.50 cap is too small.
        let err = c
            .up(None, 0.50, UNTIL, |_, _| {
                panic!("must not ask over the cap")
            })
            .unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
        assert!(vast.created.borrow().is_empty());
    }

    #[test]
    fn up_refuses_when_the_account_holds_less_than_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            credit: Cell::new(2.0),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        let err = c
            .up(None, 10.0, UNTIL, |_, _| {
                panic!("must not ask on low credit")
            })
            .unwrap_err();
        assert!(err.to_string().contains("below"), "{err}");
        assert!(vast.created.borrow().is_empty());
    }

    #[test]
    fn up_refuses_an_offer_that_fails_the_search_limits() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![Offer {
                dph_total: 0.5,
                ..offer(1, 10, 0.15)
            }],
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        let err = c
            .up(Some(1), CAP, UNTIL, |_, _| {
                panic!("must not ask for a filtered offer")
            })
            .unwrap_err();
        assert!(err.to_string().contains("search limits"), "{err}");
        assert!(vast.created.borrow().is_empty());
    }

    #[test]
    fn ssh_args_disable_agent_and_x11_forwarding() {
        let ep = Endpoint {
            host: "ssh1.vast.ai".into(),
            port: 2100,
        };
        let args = ssh_args(&ep, Path::new("/tmp/known_hosts"));
        let has = |a: &str, b: &str| args.windows(2).any(|w| w[0] == a && w[1] == b);
        assert!(has("-o", "ForwardAgent=no"), "{args:?}");
        assert!(has("-o", "ForwardX11=no"), "{args:?}");
    }

    #[test]
    fn down_reports_the_credit_spent() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();
        // The account is billed while it ran.
        vast.credit.set(9.25);
        let down = c.down().unwrap();
        assert_eq!(down.ids, vec![100]);
        assert_eq!(down.credit_before, Some(12.0));
        assert_eq!(down.credit_after, Some(9.25));
    }

    #[test]
    fn up_confirms_before_replacing_a_gone_worker() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();
        // The instance vanishes, so the next `up` has to rent a fresh one.
        vast.live.borrow_mut().clear();

        // Reaching the confirmation proves the fresh rental still went through
        // the deadline and budget guards instead of being rented silently.
        let asked = Cell::new(false);
        let err = c
            .up(None, CAP, UNTIL, |_, _| {
                asked.set(true);
                false
            })
            .unwrap_err();
        assert!(err.to_string().contains("nothing rented"), "{err}");
        assert!(asked.get());
        assert_eq!(vast.created.borrow().len(), 1);
    }

    // ----- down_verified against a fake Vast HTTP API -------------------------
    //
    // These run the real `HttpVast` client (its retries included) over a fake
    // transport that answers like the Vast API, so nothing reaches the network.

    use crate::vast::{CallFailure, HttpVast, Transport};

    /// A stateful fake of the Vast API endpoints `down_verified` uses.
    #[derive(Default)]
    struct FakeApi {
        /// Instances the account holds, as (id, label).
        live: RefCell<Vec<(u64, Option<String>)>>,
        /// Status codes the next DELETEs fail with, in order.
        delete_failures: RefCell<Vec<u16>>,
        /// DELETEs (by count) that answer OK but leave the instance listed.
        sticky_deletes: Cell<u32>,
        /// When set, every call fails as if the API were unreachable.
        down: Cell<bool>,
        calls: RefCell<Vec<String>>,
    }

    impl Transport for &FakeApi {
        fn send(
            &self,
            method: &str,
            path: &str,
            _body: Option<&Value>,
        ) -> std::result::Result<Value, CallFailure> {
            self.calls.borrow_mut().push(format!("{method} {path}"));
            if self.down.get() {
                return Err(CallFailure::NotSent("connection refused".into()));
            }
            match (method, path) {
                ("GET", "/instances/") => Ok(json!({"instances": self
                    .live
                    .borrow()
                    .iter()
                    .map(|(id, label)| json!({"id": id, "label": label, "actual_status": "running"}))
                    .collect::<Vec<_>>()})),
                ("GET", "/users/current/") => Ok(json!({"credit": 11.5})),
                ("DELETE", p) => {
                    if !self.delete_failures.borrow().is_empty() {
                        let code = self.delete_failures.borrow_mut().remove(0);
                        return Err(CallFailure::Status(code, "server error".into()));
                    }
                    let id: u64 = p
                        .trim_start_matches("/instances/")
                        .trim_end_matches('/')
                        .parse()
                        .unwrap();
                    if self.sticky_deletes.get() > 0 {
                        self.sticky_deletes.set(self.sticky_deletes.get() - 1);
                    } else {
                        self.live.borrow_mut().retain(|(i, _)| *i != id);
                    }
                    Ok(json!({"success": true}))
                }
                _ => Err(CallFailure::Status(404, "not found".into())),
            }
        }
    }

    impl FakeApi {
        fn deletes(&self) -> usize {
            self.calls
                .borrow()
                .iter()
                .filter(|c| c.starts_with("DELETE"))
                .count()
        }
    }

    fn api_cloud<'a>(
        api: &'a FakeApi,
        host: &'a FakeHost,
        dir: &Path,
    ) -> Cloud<HttpVast<&'a FakeApi>, &'a FakeHost> {
        Cloud {
            cfg: Config::default(),
            vast: HttpVast::with_transport(api).with_backoff(vec![std::time::Duration::ZERO; 4]),
            state: StateDir::new(dir),
            host,
        }
    }

    /// A tracked worker 100 (our label) plus a stale unverified marker.
    fn rented_with_marker(dir: &Path, api: &FakeApi) -> std::path::PathBuf {
        let state = StateDir::new(dir);
        state
            .save(&Tracked {
                id: 100,
                machine_id: 10,
                gpu_name: "RTX 3090".into(),
                dph: 0.2,
                credit: 12.0,
                created: 1000,
                last_used: 1000,
                endpoint: None,
            })
            .unwrap();
        api.live
            .borrow_mut()
            .push((100, Some(Config::default().label)));
        let marker = dir.join("unverified").join("omnivoice_train");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, "/home/x/stages/omnivoice_train/.checkpoint-y\n").unwrap();
        marker
    }

    #[test]
    fn down_verified_survives_three_delete_500s_and_ignores_a_stale_marker() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        let marker = rented_with_marker(dir.path(), &api);
        *api.delete_failures.borrow_mut() = vec![500, 500, 500];
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());

        // Plain `down` still refuses for a human while the marker is there.
        assert!(c.down().is_err());
        assert_eq!(api.deletes(), 0);

        let down = c.down_verified(None).unwrap();
        assert!(down.verified, "{down:?}");
        assert_eq!(down.ids, vec![100]);
        assert!(down.still_listed.is_empty());
        // Three 500s then the OK, all inside the client's own retries.
        assert_eq!(api.deletes(), 4);
        assert!(api.live.borrow().is_empty());
        // The marker is recorded and left in place, and the state is cleared.
        assert_eq!(down.overridden.len(), 1);
        assert_eq!(down.overridden[0].stage, "omnivoice_train");
        assert!(marker.exists());
        assert!(c.state.load().unwrap().is_none());
        assert_eq!(down.credit_before, Some(12.0));
        assert_eq!(down.credit_after, Some(11.5));
    }

    #[test]
    fn down_verified_reissues_delete_while_the_instance_is_still_listed() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        rented_with_marker(dir.path(), &api);
        // The first two DELETEs answer OK but Vast keeps listing the instance.
        api.sticky_deletes.set(2);
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());

        let down = c.down_verified(Some(100)).unwrap();
        assert!(down.verified, "{down:?}");
        assert_eq!(api.deletes(), 3);
        assert!(api.live.borrow().is_empty());
        // Verified within the window, not at its end.
        assert!(host.now.get() - 1000 < VERIFY_SECS, "{}", host.now.get());
    }

    #[test]
    fn down_verified_destroys_a_cached_id_and_an_unlabelled_one_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        // Nothing tracked at home: the caller passes the id it cached at create.
        api.live.borrow_mut().push((7, None));
        api.live.borrow_mut().push((5, None));
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());

        let down = c.down_verified(Some(7)).unwrap();
        assert!(down.verified, "{down:?}");
        assert_eq!(down.ids, vec![7]);
        assert_eq!(*api.live.borrow(), vec![(5, None)]);
    }

    #[test]
    fn down_verified_is_unverified_when_the_api_is_down() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        rented_with_marker(dir.path(), &api);
        api.down.set(true);
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());

        let down = c.down_verified(None).unwrap();
        assert!(!down.verified);
        assert_eq!(down.still_listed, vec![100]);
        assert!(!down.errors.is_empty());
        // It kept trying for the whole window, then gave up.
        assert!(host.now.get() - 1000 >= VERIFY_SECS);
        // The tracked worker is kept, so a later `down` can still find it.
        assert_eq!(c.state.load().unwrap().map(|t| t.id), Some(100));
    }

    #[test]
    fn down_verified_with_nothing_rented_is_verified_by_an_empty_listing() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());
        let down = c.down_verified(None).unwrap();
        assert!(down.verified);
        assert!(down.ids.is_empty());
        assert_eq!(api.deletes(), 0);
    }

    // ----- the dead-man timer (L2) ------------------------------------------

    fn one_offer_running() -> FakeVast {
        FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        }
    }

    #[test]
    fn up_arms_the_dead_man_timer_at_the_hard_cap_and_rearming_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let vast = one_offer_running();
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        let t = c.up(None, CAP, UNTIL, |_, _| true).unwrap();
        assert_eq!(t.id, 100);
        let cap_at = deadman::cap_at(1000, UNTIL, c.cfg.max_hours, CAP, 0.15);
        assert_eq!(cap_at, 1000 + (c.cfg.max_hours * 3600.0) as u64);
        assert_eq!(
            *host.timer_calls.borrow(),
            vec![format!("arm vi-deadman-100 @{cap_at}")]
        );
        let r = deadman::load(&c.state, 100).unwrap().unwrap();
        assert_eq!((r.start, r.cap_at, r.fire_at), (1000, cap_at, cap_at));
        assert_eq!(r.argv[1..], ["cloud", "deadman", "fire", "100"]);
        assert!(
            r.env.iter().all(|(k, _)| k != "VAST_API_KEY"),
            "{:?}",
            r.env
        );
        assert_eq!(r.log, dir.path().join("deadman-100.log"));

        // A second `up` reuses the worker and leaves the armed timer alone.
        c.up(None, CAP, UNTIL, |_, _| panic!("asked to rent again"))
            .unwrap();
        assert_eq!(host.timer_calls.borrow().len(), 1);

        // If the timer is gone (it was never meant to survive a reboot), the
        // next `up` re-arms it at the recorded time, not a later one.
        host.timers.borrow_mut().clear();
        host.now.set(5000);
        c.up(None, CAP, UNTIL, |_, _| panic!("asked to rent again"))
            .unwrap();
        assert_eq!(
            host.timer_calls.borrow().last().unwrap(),
            &format!("arm vi-deadman-100 @{cap_at}")
        );
        // A shorter approval lowers the cap; nothing raises it.
        c.up(None, CAP, 6000, |_, _| panic!("asked to rent again"))
            .unwrap();
        let r = deadman::load(&c.state, 100).unwrap().unwrap();
        assert_eq!((r.cap_at, r.fire_at), (6000, 6000));
        c.up(None, CAP, UNTIL, |_, _| panic!("asked to rent again"))
            .unwrap();
        assert_eq!(deadman::load(&c.state, 100).unwrap().unwrap().cap_at, 6000);
    }

    #[test]
    fn a_failed_arm_aborts_up_and_destroys_the_new_instance() {
        let dir = tempfile::tempdir().unwrap();
        let vast = one_offer_running();
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        host.arm_failures.set(1);
        let c = cloud(&vast, &host, dir.path());
        let err = c.up(None, CAP, UNTIL, |_, _| true).unwrap_err();
        assert!(err.to_string().contains("dead-man timer"), "{err}");
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert!(c.state.load().unwrap().is_none());
        assert!(deadman::load(&c.state, 100).unwrap().is_none());
    }

    #[test]
    fn no_user_manager_refuses_before_renting() {
        let dir = tempfile::tempdir().unwrap();
        let vast = one_offer_running();
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        host.timer_down.set(true);
        let c = cloud(&vast, &host, dir.path());
        let err = c.up(None, CAP, UNTIL, |_, _| true).unwrap_err();
        assert!(err.to_string().contains("refusing to rent"), "{err}");
        assert!(vast.created.borrow().is_empty());
    }

    #[test]
    fn heartbeats_extend_the_timer_but_never_past_the_hard_cap() {
        let dir = tempfile::tempdir().unwrap();
        let vast = one_offer_running();
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();
        let cap_at = deadman::load(&c.state, 100).unwrap().unwrap().cap_at;

        // The first heartbeat brings the fire time in to now + grace.
        host.now.set(2000);
        assert_eq!(c.deadman_extend(100, 1800).unwrap().fire_at, 3800);
        assert_eq!(host.timers.borrow()["vi-deadman-100"], 3800);
        // Near the cap, now + grace is clamped to the cap.
        host.now.set(cap_at - 100);
        assert_eq!(c.deadman_extend(100, 1800).unwrap().fire_at, cap_at);
        assert_eq!(host.timers.borrow()["vi-deadman-100"], cap_at);
        // At the cap nothing can extend it, and the timer stays armed.
        host.now.set(cap_at);
        let err = c.deadman_extend(100, 1800).unwrap_err();
        assert!(err.to_string().contains("hard cap"), "{err}");
        assert_eq!(host.timers.borrow()["vi-deadman-100"], cap_at);
        // An unknown instance has nothing to extend.
        assert!(c.deadman_extend(7, 1800).is_err());
    }

    #[test]
    fn a_failed_extend_puts_the_old_fire_time_back() {
        let dir = tempfile::tempdir().unwrap();
        let vast = one_offer_running();
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, CAP, UNTIL, |_, _| true).unwrap();
        host.now.set(2000);
        c.deadman_extend(100, 1800).unwrap();
        host.now.set(2500);
        host.arm_failures.set(1);
        let err = c.deadman_extend(100, 1800).unwrap_err();
        assert!(err.to_string().contains("still fires at 3800"), "{err}");
        assert_eq!(host.timers.borrow()["vi-deadman-100"], 3800);
        assert_eq!(deadman::load(&c.state, 100).unwrap().unwrap().fire_at, 3800);
        // Both failing is reported as an alert.
        host.arm_failures.set(2);
        let err = c.deadman_extend(100, 1800).unwrap_err();
        assert!(err.to_string().contains("NOT armed"), "{err}");
    }

    #[test]
    fn the_timer_is_cancelled_only_after_a_verified_destroy() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        rented_with_marker(dir.path(), &api);
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());
        c.arm_deadman(100, 1000, 9000).unwrap();

        // An unverified destroy leaves the timer and its record.
        api.down.set(true);
        assert!(!c.down_verified(Some(100)).unwrap().verified);
        assert!(host.timers.borrow().contains_key("vi-deadman-100"));
        assert!(deadman::load(&c.state, 100).unwrap().is_some());

        // A verified one cancels both units and forgets the record.
        api.down.set(false);
        let down = c.down_verified(Some(100)).unwrap();
        assert!(down.verified, "{down:?}");
        assert!(down.errors.is_empty(), "{down:?}");
        assert!(host.timers.borrow().is_empty());
        assert!(deadman::load(&c.state, 100).unwrap().is_none());
        assert!(host
            .timer_calls
            .borrow()
            .contains(&"stop vi-deadman-100 service=true".to_string()));
        // Cancelling again, or a timer that never existed, is a no-op.
        deadman::cancel(&c.host, &c.state, 100, true).unwrap();
        deadman::cancel(&c.host, &c.state, 12345, true).unwrap();
    }

    #[test]
    fn a_fire_for_a_stale_id_never_touches_the_tracked_worker() {
        let dir = tempfile::tempdir().unwrap();
        let api = FakeApi::default();
        let label = Config::default().label;
        rented_with_marker(dir.path(), &api);
        api.live.borrow_mut().push((7, Some(label.clone())));
        let host = FakeHost {
            now: Cell::new(1000),
            ..FakeHost::default()
        };
        let c = api_cloud(&api, &host, dir.path());
        c.arm_deadman(7, 1000, 9000).unwrap();

        // Home tracks 100, so the timer for 7 destroys 7 alone.
        let down = c.deadman_fire(7).unwrap();
        assert!(down.verified, "{down:?}");
        assert_eq!(down.ids, vec![7]);
        assert_eq!(*api.live.borrow(), vec![(100, Some(label.clone()))]);
        assert_eq!(c.state.load().unwrap().map(|t| t.id), Some(100));
        // Its own service (this process) is left to finish.
        assert!(host
            .timer_calls
            .borrow()
            .contains(&"stop vi-deadman-7 service=false".to_string()));
        assert!(deadman::load(&c.state, 7).unwrap().is_none());

        // The timer for the tracked worker is the full verified down.
        api.live.borrow_mut().push((8, Some(label)));
        let down = c.deadman_fire(100).unwrap();
        assert!(down.verified, "{down:?}");
        assert_eq!(down.ids, vec![100, 8]);
        assert!(api.live.borrow().is_empty());
        assert!(c.state.load().unwrap().is_none());
    }
}
