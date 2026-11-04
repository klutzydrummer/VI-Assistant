//! The home half of worker recovery, against a fake Vast API and a fake ssh:
//! endpoint refresh (row 44), restart (R49) and authenticated host-key rotation
//! (R50). Nothing here touches the network or rents anything.

use std::cell::{Cell, RefCell};
use std::path::Path;

use cloud::worker::{newest_hostkey_batch, parse_keygen_line, Cloud, Host};
use cloud::{Config, Endpoint, Instance, Offer, StateDir, Tracked, Vast};
use serde_json::Value;

/// Enough for the default `max_hours` at the fake instance's price, and below
/// the fake account credit.
const CAP: f64 = 10.0;
/// Far in the future, for the tests that do not exercise the deadline.
const UNTIL: u64 = 10_000_000;

struct FakeVast {
    instance: RefCell<Option<Instance>>,
    /// Successive answers for `logs`; the last one repeats.
    logs: RefCell<Vec<String>>,
    log_calls: Cell<usize>,
    started: RefCell<Vec<u64>>,
    created: RefCell<Vec<u64>>,
    destroyed: RefCell<Vec<u64>>,
    credit: f64,
}

impl FakeVast {
    fn new(instance: Option<Instance>) -> Self {
        FakeVast {
            instance: RefCell::new(instance),
            logs: RefCell::new(Vec::new()),
            log_calls: Cell::new(0),
            started: RefCell::new(Vec::new()),
            created: RefCell::new(Vec::new()),
            destroyed: RefCell::new(Vec::new()),
            credit: 12.0,
        }
    }

    fn with_log(self, log: String) -> Self {
        self.logs.borrow_mut().push(log);
        self
    }
}

impl Vast for &FakeVast {
    fn offers(&self, _query: &Value) -> cloud::Result<Vec<Offer>> {
        Ok(Vec::new())
    }

    fn create(&self, offer: u64, _request: &Value) -> cloud::Result<u64> {
        self.created.borrow_mut().push(offer);
        Ok(999)
    }

    fn instance(&self, id: u64) -> cloud::Result<Option<Instance>> {
        Ok(self
            .instance
            .borrow()
            .clone()
            .filter(|instance| instance.id == id))
    }

    fn instances(&self) -> cloud::Result<Vec<Instance>> {
        Ok(self.instance.borrow().clone().into_iter().collect())
    }

    fn destroy(&self, id: u64) -> cloud::Result<()> {
        self.destroyed.borrow_mut().push(id);
        Ok(())
    }

    fn credit(&self) -> cloud::Result<f64> {
        Ok(self.credit)
    }

    fn start(&self, id: u64) -> cloud::Result<()> {
        self.started.borrow_mut().push(id);
        Ok(())
    }

    fn logs(&self, _id: u64) -> cloud::Result<String> {
        let n = self.log_calls.get();
        self.log_calls.set(n + 1);
        let logs = self.logs.borrow();
        Ok(logs
            .get(n)
            .or_else(|| logs.last())
            .cloned()
            .unwrap_or_default())
    }
}

struct FakeHost {
    now: Cell<u64>,
    /// What `ssh-keyscan` presents.
    scan: Vec<String>,
    /// What `ssh-keygen -lf` reports for those lines.
    keygen: Vec<String>,
    logged: RefCell<Vec<String>>,
}

impl FakeHost {
    fn new(scan: &[&str], keygen: &[&str]) -> Self {
        FakeHost {
            now: Cell::new(5000),
            scan: scan.iter().map(|s| s.to_string()).collect(),
            keygen: keygen.iter().map(|s| s.to_string()).collect(),
            logged: RefCell::new(Vec::new()),
        }
    }
}

impl Host for &FakeHost {
    fn now(&self) -> u64 {
        self.now.get()
    }

    fn sleep(&self, secs: u64) {
        self.now.set(self.now.get() + secs);
    }

    fn reachable(&self, _ep: &Endpoint, _known_hosts: &Path) -> bool {
        true
    }

    fn log(&self, msg: &str) {
        self.logged.borrow_mut().push(msg.to_string());
    }

    fn keyscan(&self, _ep: &Endpoint) -> Vec<String> {
        self.scan.clone()
    }

    fn key_fingerprints(&self, _lines: &[String]) -> Vec<String> {
        self.keygen.clone()
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

fn running(id: u64) -> Instance {
    Instance {
        id,
        label: Some("vi-worker".into()),
        actual_status: Some("running".into()),
        ssh_host: Some("ssh1.vast.ai".into()),
        ssh_port: Some(41022),
        dph_total: 0.2,
        ..Instance::default()
    }
}

fn save_tracked(dir: &Path, id: u64, port: u16) -> StateDir {
    let state = StateDir::new(dir);
    state
        .save(&Tracked {
            id,
            machine_id: 7,
            gpu_name: "RTX 3090".into(),
            dph: 0.2,
            credit: 12.0,
            created: 1000,
            last_used: 1000,
            endpoint: Some(Endpoint {
                host: "ssh1.vast.ai".into(),
                port,
            }),
        })
        .unwrap();
    state
}

fn log_line(stamp: u64, fingerprint: &str) -> String {
    format!("VI-HOSTKEY {stamp} 256 {fingerprint} root@worker (ED25519)\n")
}

#[test]
fn refresh_updates_endpoint_for_same_instance_only() {
    let dir = tempfile::tempdir().unwrap();
    let state = save_tracked(dir.path(), 42, 40000);
    let vast = FakeVast::new(Some(running(42)));
    let host = FakeHost::new(&[], &[]);
    let c = cloud(&vast, &host, dir.path());

    let t = c.refresh().unwrap().expect("the tracked worker exists");
    assert_eq!(t.id, 42);
    assert_eq!(
        t.endpoint,
        Some(Endpoint {
            host: "ssh1.vast.ai".into(),
            port: 41022
        })
    );
    // It saved the new endpoint and never rented or released anything.
    assert_eq!(state.load().unwrap().unwrap().endpoint, t.endpoint);
    assert!(vast.created.borrow().is_empty(), "refresh must not rent");
    assert!(
        vast.destroyed.borrow().is_empty(),
        "refresh must not destroy"
    );
    assert!(vast.started.borrow().is_empty(), "refresh must not start");

    // A gone instance is an error, not a reason to rent a replacement.
    *vast.instance.borrow_mut() = None;
    let err = c.refresh().unwrap_err().to_string();
    assert!(err.contains("is gone"), "{err}");
    assert!(vast.created.borrow().is_empty());
    assert!(vast.destroyed.borrow().is_empty());
}

#[test]
fn start_puts_running() {
    let dir = tempfile::tempdir().unwrap();
    let state = save_tracked(dir.path(), 42, 40000);
    let stopped = Instance {
        actual_status: Some("exited".into()),
        intended_status: Some("stopped".into()),
        ..running(42)
    };
    let vast = FakeVast::new(Some(stopped));
    let host = FakeHost::new(&[], &[]);
    let c = cloud(&vast, &host, dir.path());

    let t = c.start(CAP, UNTIL).unwrap();
    assert_eq!(*vast.started.borrow(), vec![42]);
    assert!(vast.created.borrow().is_empty(), "start must not rent");
    assert!(vast.destroyed.borrow().is_empty(), "start must not destroy");
    // The explicit start counts as use, and the endpoint comes from the API.
    assert_eq!(state.load().unwrap().unwrap().last_used, 5000);
    assert_eq!(t.endpoint.as_ref().unwrap().port, 41022);

    // The same approval guards as `up`: a cap below the worst case refuses.
    let err = c.start(0.1, UNTIL).unwrap_err().to_string();
    assert!(err.contains("exceeds the $0.10 cap"), "{err}");
    // An expired approval refuses before Vast is asked to start anything.
    let err = c.start(CAP, 1000).unwrap_err().to_string();
    assert!(err.contains("deadline has passed"), "{err}");
    assert_eq!(*vast.started.borrow(), vec![42], "no start after a refusal");
}

#[test]
fn hostkey_matching_parses_keygen_and_log_lines() {
    assert_eq!(
        parse_keygen_line("256 SHA256:abc123 root@host (ED25519)"),
        Some(("SHA256:abc123".into(), "ED25519".into()))
    );
    assert_eq!(
        parse_keygen_line("3072 SHA256:def456 [ssh1.vast.ai]:41022 (RSA)"),
        Some(("SHA256:def456".into(), "RSA".into()))
    );
    assert_eq!(parse_keygen_line("no fingerprint here"), None);
    assert_eq!(parse_keygen_line("256 SHA256:abc root@host (ED25519"), None);

    let log = concat!(
        "some other boot output\n",
        "2026-11-04T10:00:00Z VI-HOSTKEY 1600 256 SHA256:old root@worker (ED25519)\n",
        "VI-HOSTKEY 1700 256 SHA256:new root@worker (ED25519)\n",
        "2026-11-04T10:01:00Z VI-HOSTKEY 1700 3072 SHA256:new-rsa root@worker (RSA)\n",
    );
    let batch = newest_hostkey_batch(log, None).unwrap();
    assert_eq!(batch.stamp, 1700);
    assert_eq!(
        batch.keys,
        vec![
            ("SHA256:new".into(), "ED25519".into()),
            ("SHA256:new-rsa".into(), "RSA".into()),
        ]
    );
    // A batch at or below the last accepted stamp is stale, even if it is the
    // newest one in the log.
    assert_eq!(newest_hostkey_batch(log, Some(1700)), None);
    assert_eq!(newest_hostkey_batch(log, Some(1500)).unwrap().stamp, 1700);
    assert_eq!(newest_hostkey_batch("nothing to see", None), None);
}

#[test]
fn match_accepts_and_rewrites_known_hosts_only_for_that_instance() {
    let dir = tempfile::tempdir().unwrap();
    let state = save_tracked(dir.path(), 42, 40000);
    let known_hosts = state.known_hosts(42);
    std::fs::write(
        &known_hosts,
        concat!(
            "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOLDKEY\n",
            "other.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOTHERKEY\n",
        ),
    )
    .unwrap();
    // Another instance's file must not be touched.
    std::fs::write(state.known_hosts(8), "other instance\n").unwrap();

    let scan = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINEWKEY";
    let keygen = "256 SHA256:newfp root@worker (ED25519)";
    let vast = FakeVast::new(Some(running(42))).with_log(log_line(1700, "SHA256:newfp"));
    let host = FakeHost::new(&[scan], &[keygen]);
    let c = cloud(&vast, &host, dir.path());

    let rotation = c.accept_hostkey(0).unwrap();
    assert_eq!(rotation.stamp, 1700);
    assert_eq!(rotation.fingerprints, vec!["SHA256:newfp".to_string()]);

    let rewritten = std::fs::read_to_string(&known_hosts).unwrap();
    assert!(rewritten.contains(scan), "{rewritten}");
    assert!(
        !rewritten.contains("OLDKEY"),
        "the old key is gone: {rewritten}"
    );
    assert!(
        rewritten.contains("other.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOTHERKEY"),
        "other hosts are kept: {rewritten}"
    );
    assert_eq!(
        std::fs::read_to_string(state.known_hosts(8)).unwrap(),
        "other instance\n"
    );
    assert_eq!(state.last_hostkey_stamp(42).unwrap(), Some(1700));
    let logged = host.logged.borrow().join("\n");
    assert!(logged.contains("accepted a host-key rotation"), "{logged}");
    assert!(logged.contains("no inputs are sent"), "{logged}");
    assert_eq!(state.load().unwrap().unwrap().endpoint.unwrap().port, 41022);
}

#[test]
fn mismatch_refuses_and_halts_pushes() {
    let dir = tempfile::tempdir().unwrap();
    let state = save_tracked(dir.path(), 42, 40000);
    let known_hosts = state.known_hosts(42);
    let before = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOLDKEY\n";
    std::fs::write(&known_hosts, before).unwrap();

    let scan = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPRESENTED";
    let keygen = "256 SHA256:presented root@worker (ED25519)";
    let vast = FakeVast::new(Some(running(42))).with_log(log_line(1700, "SHA256:somebodyelse"));
    let host = FakeHost::new(&[scan], &[keygen]);
    let c = cloud(&vast, &host, dir.path());

    let err = c.accept_hostkey(0).unwrap_err().to_string();
    assert!(err.contains("does not match"), "{err}");
    assert!(err.contains("refusing"), "{err}");
    assert_eq!(std::fs::read_to_string(&known_hosts).unwrap(), before);
    assert_eq!(state.last_hostkey_stamp(42).unwrap(), None);
}

#[test]
fn stale_batch_older_than_last_accepted_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let state = save_tracked(dir.path(), 42, 40000);
    state.save_hostkey_stamp(42, 1700).unwrap();
    let known_hosts = state.known_hosts(42);
    let before = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOLDKEY\n";
    std::fs::write(&known_hosts, before).unwrap();

    // The worker presents a key the stale batch names; it is still refused,
    // because that batch was already accepted for an earlier boot.
    let scan = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOLDKEY";
    let keygen = "256 SHA256:oldfp root@worker (ED25519)";
    let vast = FakeVast::new(Some(running(42))).with_log(log_line(1700, "SHA256:oldfp"));
    let host = FakeHost::new(&[scan], &[keygen]);
    let c = cloud(&vast, &host, dir.path());

    let err = c.accept_hostkey(0).unwrap_err().to_string();
    assert!(err.contains("no new VI-HOSTKEY"), "{err}");
    assert!(err.contains("refusing"), "{err}");
    assert_eq!(std::fs::read_to_string(&known_hosts).unwrap(), before);
    assert_eq!(state.last_hostkey_stamp(42).unwrap(), Some(1700));
}

#[test]
fn timeout_without_line_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let state = save_tracked(dir.path(), 42, 40000);
    let known_hosts = state.known_hosts(42);
    let before = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOLDKEY\n";
    std::fs::write(&known_hosts, before).unwrap();

    let scan = "[ssh1.vast.ai]:41022 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINEWKEY";
    let keygen = "256 SHA256:newfp root@worker (ED25519)";
    // The log never carries a VI-HOSTKEY line (Vast has not made it available).
    let vast = FakeVast::new(Some(running(42))).with_log("booting\n".into());
    let host = FakeHost::new(&[scan], &[keygen]);
    let c = cloud(&vast, &host, dir.path());

    let err = c.accept_hostkey(0).unwrap_err().to_string();
    assert!(err.contains("no new VI-HOSTKEY"), "{err}");
    assert!(err.contains("refusing"), "{err}");
    assert_eq!(std::fs::read_to_string(&known_hosts).unwrap(), before);
    assert_eq!(state.last_hostkey_stamp(42).unwrap(), None);
}
