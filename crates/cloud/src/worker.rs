//! Renting, waiting for, and releasing the worker.

use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

use crate::state::{StateDir, Tracked};
use crate::vast::{Endpoint, Instance, Offer, Vast};
use crate::{onstart, Config, Error, Result};

/// How many machines `up` tries before giving up, when one fails to boot.
const ATTEMPTS: usize = 3;
const POLL_SECS: u64 = 10;

/// The outside world, so tests can fake time and ssh.
pub trait Host {
    /// Unix seconds.
    fn now(&self) -> u64;
    fn sleep(&self, secs: u64);
    /// Does `ssh root@endpoint true` succeed?
    fn reachable(&self, ep: &Endpoint, known_hosts: &Path) -> bool;
    fn log(&self, msg: &str);
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

impl<V: Vast, H: Host> Cloud<V, H> {
    pub fn offers(&self, limit: usize) -> Result<Vec<Offer>> {
        self.vast.offers(&self.cfg.offer_query(limit))
    }

    /// Rents a worker (or reuses the one we have) and waits until ssh answers.
    /// `confirm` sees the cheapest offer and the account credit before anything
    /// is rented, and returns false to back out.
    pub fn up(
        &self,
        offer: Option<u64>,
        confirm: impl FnOnce(&Offer, Option<f64>) -> bool,
    ) -> Result<Tracked> {
        if let Some(t) = self.state.load()? {
            match self.vast.instance(t.id)? {
                Some(inst) if !stopped(&inst) => {
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
        if !confirm(first, self.vast.credit().ok()) {
            return Err(Error::Worker("nothing rented".into()));
        }

        let mut tried = Vec::new();
        let mut last_err = None;
        for offer in candidates {
            if tried.len() == ATTEMPTS {
                break;
            }
            if tried.contains(&offer.machine_id) {
                continue;
            }
            tried.push(offer.machine_id);
            if tried.len() > 1 {
                self.host.log(&format!(
                    "trying offer {} ({}, ${:.3}/hr) instead",
                    offer.id, offer.gpu_name, offer.dph_total
                ));
            }
            let id = self.vast.create(offer.id, &self.create_request())?;
            let now = self.host.now();
            let t = Tracked {
                id,
                machine_id: offer.machine_id,
                gpu_name: offer.gpu_name.clone(),
                dph: offer.dph_total,
                created: now,
                last_used: now,
                endpoint: None,
            };
            // Saved before waiting, so `vi cloud down` can release it even if
            // this process is interrupted.
            self.state.save(&t)?;
            self.host
                .log(&format!("rented instance {id}; waiting for it to boot"));
            match self.wait(t) {
                Ok(t) => return Ok(t),
                Err(e) => {
                    self.host
                        .log(&format!("instance {id} failed: {e}; giving it back"));
                    self.vast.destroy(id)?;
                    self.state.clear(id)?;
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Worker("no usable offers".into())))
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

    /// Destroys the tracked worker and any other instance carrying our label.
    /// Returns the ids destroyed.
    pub fn down(&self) -> Result<Vec<u64>> {
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
    /// `max_hours`. Returns why, if it did.
    pub fn reap(&self) -> Result<Option<String>> {
        let Some(t) = self.state.load()? else {
            return Ok(None);
        };
        let now = self.host.now();
        let idle = now.saturating_sub(t.last_used) / 60;
        let up = now.saturating_sub(t.created) / 60;
        let max = self.cfg.max_minutes();
        let why = if idle >= self.cfg.idle_minutes {
            format!("idle for {idle} minutes")
        } else if max > 0 && up >= max {
            format!("up for {up} minutes, the max_hours limit")
        } else {
            return Ok(None);
        };
        self.down()?;
        Ok(Some(why))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[derive(Default)]
    struct FakeVast {
        offers: Vec<Offer>,
        /// Status each created instance reports, in creation order.
        boots: RefCell<Vec<&'static str>>,
        live: RefCell<Vec<Instance>>,
        created: RefCell<Vec<(u64, Value)>>,
        destroyed: RefCell<Vec<u64>>,
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
            Ok(12.0)
        }
    }

    struct FakeHost {
        now: Cell<u64>,
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
    }

    fn offer(id: u64, machine: u64, dph: f64) -> Offer {
        Offer {
            id,
            machine_id: machine,
            gpu_name: "RTX 3090".into(),
            gpu_ram: 24576.0,
            dph_total: dph,
            reliability2: 0.99,
            inet_down: 900.0,
            geolocation: None,
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
        };
        let c = cloud(&vast, &host, dir.path());

        let mut seen = None;
        let t = c
            .up(None, |o, credit| {
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
        let again = c.up(None, |_, _| panic!("asked to rent again")).unwrap();
        assert_eq!(again.id, 100);
        assert!(c.env().unwrap().contains("VI_WORKER_PORT='2100'"));

        assert_eq!(c.down().unwrap(), vec![100]);
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert_eq!(c.state.load().unwrap(), None);
        // The stopped instance from another project is reported, not touched.
        let report = c.report().unwrap();
        assert_eq!(report.others.len(), 1);
        assert!(report.tracked.is_none());
    }

    #[test]
    fn a_machine_that_fails_to_boot_is_returned_and_the_next_tried() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15), offer(3, 10, 0.16), offer(2, 20, 0.2)],
            boots: RefCell::new(vec!["loading", "running"]),
            ..FakeVast::default()
        };
        let host = FakeHost {
            now: Cell::new(1000),
        };
        let c = cloud(&vast, &host, dir.path());
        let t = c.up(None, |_, _| true).unwrap();
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        // Offer 3 is on the machine that just failed, so it's skipped.
        let rented: Vec<u64> = vast.created.borrow().iter().map(|c| c.0).collect();
        assert_eq!(rented, vec![1, 2]);
        assert_eq!(t.id, 101);
        assert_eq!(c.state.load().unwrap().unwrap().id, 101);
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
        };
        let c = cloud(&vast, &host, dir.path());
        assert!(c.up(None, |_, _| false).is_err());
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
        };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, |_, _| true).unwrap();

        host.now.set(1000 + 19 * 60);
        assert_eq!(c.reap().unwrap(), None);
        c.touch().unwrap();
        host.now.set(1000 + 38 * 60);
        assert_eq!(c.reap().unwrap(), None);
        host.now.set(1000 + 39 * 60);
        assert_eq!(c.reap().unwrap().unwrap(), "idle for 20 minutes");
        assert_eq!(*vast.destroyed.borrow(), vec![100]);
        assert_eq!(c.reap().unwrap(), None);
    }

    #[test]
    fn reap_enforces_max_hours_even_when_busy() {
        let dir = tempfile::tempdir().unwrap();
        let vast = FakeVast {
            offers: vec![offer(1, 10, 0.15)],
            boots: RefCell::new(vec!["running"]),
            ..FakeVast::default()
        };
        let host = FakeHost { now: Cell::new(0) };
        let c = cloud(&vast, &host, dir.path());
        c.up(None, |_, _| true).unwrap();
        for _ in 0..24 {
            host.now.set(host.now.get() + 10 * 60);
            c.touch().unwrap();
        }
        assert!(c.reap().unwrap().unwrap().contains("max_hours"));
    }

    #[test]
    fn quoting_survives_single_quotes() {
        assert_eq!(shell_quote("a'b"), r"'a'\''b'");
    }
}
