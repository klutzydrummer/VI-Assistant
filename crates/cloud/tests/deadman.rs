//! The dead-man timer (L2) against the real systemd user manager.
//!
//! Every test here talks to `systemctl --user`, so each one first checks that
//! the user manager answers and skips (printing why) when it does not, as in a
//! container without one. Nothing here touches Vast: the fire talks to a fake
//! Vast API on 127.0.0.1, reads a fake key from a temporary `XDG_CONFIG_HOME`,
//! and the instance ids are made up. Units are named
//! `vi-deadman-test-<pid>-<id>` and every test stops its units on the way out.
//!
//! The "paid run" that arms the timer is a child copy of this test binary
//! (`child_entry`), so it can be killed with SIGKILL or frozen with SIGSTOP.
//! The timer then runs `fire_entry`, another copy of this binary, which does
//! what `vi cloud deadman fire <id>` does.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cloud::{deadman, Cloud, Config, Host, HttpVast, StateDir, SystemHost};

const KEY: &str = "fake-test-key-not-a-vast-key";

/// Skips the test when there is no systemd user manager to talk to.
macro_rules! need_user_manager {
    () => {
        if let Err(e) = deadman::systemd_ready() {
            eprintln!("SKIPPED: no systemd user manager ({e})");
            return;
        }
    };
}

/// A tiny fake Vast API: instances listed until DELETEd. Records every
/// request line and whether the fake key was sent.
#[derive(Clone)]
struct FakeVast {
    url: String,
    live: Arc<Mutex<Vec<u64>>>,
    requests: Arc<Mutex<Vec<String>>>,
    bad_auth: Arc<Mutex<u32>>,
}

impl FakeVast {
    fn start(live: Vec<u64>) -> FakeVast {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let fake = FakeVast {
            url: format!("http://{}", listener.local_addr().unwrap()),
            live: Arc::new(Mutex::new(live)),
            requests: Arc::default(),
            bad_auth: Arc::default(),
        };
        let server = fake.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                server.answer(stream);
            }
        });
        fake
    }

    fn answer(&self, stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        let mut length = 0;
        let mut authed = false;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                break;
            }
            let lower = h.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                length = v.trim().parse().unwrap_or(0);
            }
            if lower.starts_with("authorization:") {
                authed = h.trim_end().ends_with(&format!("Bearer {KEY}"));
            }
        }
        let mut body = vec![0; length];
        let _ = reader.read_exact(&mut body);
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        if !authed {
            *self.bad_auth.lock().unwrap() += 1;
        }
        self.requests
            .lock()
            .unwrap()
            .push(format!("{method} {path}"));
        let reply = match (method, path) {
            ("GET", "/instances/") => {
                let list: Vec<String> = self
                    .live
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|id| {
                        format!(r#"{{"id":{id},"label":"vi-worker","actual_status":"running"}}"#)
                    })
                    .collect();
                format!(r#"{{"instances":[{}]}}"#, list.join(","))
            }
            ("GET", "/users/current/") => r#"{"credit":5.0}"#.to_string(),
            ("DELETE", p) => {
                let id = p
                    .trim_start_matches("/instances/")
                    .trim_end_matches('/')
                    .parse::<u64>()
                    .ok();
                self.live.lock().unwrap().retain(|i| Some(*i) != id);
                r#"{"success":true}"#.to_string()
            }
            _ => r#"{"instances":[]}"#.to_string(),
        };
        let mut out = stream;
        let _ = write!(
            out,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{reply}",
            reply.len()
        );
    }

    fn saw(&self, request: &str) -> bool {
        self.requests.lock().unwrap().iter().any(|r| r == request)
    }

    /// A listing of instances after the first DELETE of `id`: the verify.
    fn verified_after_delete(&self, id: u64) -> bool {
        let r = self.requests.lock().unwrap();
        let delete = format!("DELETE /instances/{id}/");
        r.iter()
            .position(|x| *x == delete)
            .is_some_and(|i| r[i + 1..].iter().any(|x| x == "GET /instances/"))
    }
}

/// One test's world: temp state and config dirs, the fake key file, the unit
/// name, and teardown that always stops the units.
struct World {
    _tmp: tempfile::TempDir,
    state: PathBuf,
    config_home: PathBuf,
    id: u64,
    unit: String,
    child: Option<Child>,
}

fn prefix() -> String {
    format!("vi-deadman-test-{}-", std::process::id())
}

impl World {
    fn new(id: u64) -> World {
        let tmp = tempfile::tempdir().unwrap();
        let state = tmp.path().join("state");
        let config_home = tmp.path().join("config");
        std::fs::create_dir_all(config_home.join("vastai")).unwrap();
        std::fs::write(config_home.join("vastai/vast_api_key"), KEY).unwrap();
        World {
            unit: format!("{}{id}", prefix()),
            _tmp: tmp,
            state,
            config_home,
            id,
            child: None,
        }
    }

    /// Runs `child_entry` in a copy of this binary, as `role`.
    fn command(&self, role: &str, vast: &FakeVast, fire_in: u64) -> Command {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", "child_entry", "--ignored", "--nocapture"])
            .env("VI_DEADMAN_TEST_ROLE", role)
            .env("VI_DEADMAN_TEST_ID", self.id.to_string())
            .env("VI_DEADMAN_TEST_FIRE_IN", fire_in.to_string())
            .env("VI_STATE_DIR", &self.state)
            .env("VI_CLOUD_CONFIG", self.config_home.join("vi/cloud.toml"))
            .env("XDG_CONFIG_HOME", &self.config_home)
            .env("VAST_API_URL", &vast.url)
            .env(deadman::PREFIX_ENV, prefix())
            .env_remove("VAST_API_KEY")
            .env_remove(deadman::LOG_DIR_ENV)
            .stdin(Stdio::null());
        c
    }

    /// Starts the stand-in paid run and waits until it has armed the timer.
    fn arm_and_hang(&mut self, vast: &FakeVast, fire_in: u64) {
        let mut child = self
            .command("arm_and_hang", vast, fire_in)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            line.clear();
            assert!(
                out.read_line(&mut line).unwrap() > 0,
                "the stand-in paid run exited before arming"
            );
            if line.trim() == "ARMED" {
                break;
            }
            assert!(Instant::now() < deadline, "never armed");
        }
        self.child = Some(child);
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.state.join(format!("deadman-{}.log", self.id)))
            .unwrap_or_default()
    }

    fn record_exists(&self) -> bool {
        deadman::record_path(&StateDir::new(&self.state), self.id).exists()
    }

    /// Waits for the fire: the fake DELETE, a verifying listing after it, the
    /// record removed and the timer unloaded.
    fn wait_for_fire(&self, vast: &FakeVast) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let done = vast.verified_after_delete(self.id)
                && !self.record_exists()
                && !deadman::systemd_loaded(&self.unit).unwrap();
            if done {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the dead-man timer did not fire and verify within 60 s; requests {:?}; log:\n{}",
                vast.requests.lock().unwrap(),
                self.log()
            );
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

impl Drop for World {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // SIGKILL also ends a stopped process.
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Err(e) = deadman::systemd_stop(&self.unit, true) {
            eprintln!("teardown could not stop {}: {e}", self.unit);
        }
    }
}

/// Test ids: digits, unique per test within this process.
fn test_id(n: u64) -> u64 {
    9_000_000_000 + u64::from(std::process::id()) * 10 + n
}

#[test]
fn kill_9_of_the_paid_run_still_fires_the_destroy_and_verifies_it() {
    need_user_manager!();
    let mut w = World::new(test_id(1));
    let vast = FakeVast::start(vec![w.id]);
    w.arm_and_hang(&vast, 2);
    assert!(deadman::systemd_loaded(&w.unit).unwrap());
    w.child.as_mut().unwrap().kill().unwrap(); // SIGKILL
    w.child.as_mut().unwrap().wait().unwrap();
    w.wait_for_fire(&vast);
    assert!(vast.saw(&format!("DELETE /instances/{}/", w.id)));
    assert!(vast.live.lock().unwrap().is_empty());
    assert_eq!(
        *vast.bad_auth.lock().unwrap(),
        0,
        "the fire must use the key file"
    );
    let log = w.log();
    assert!(log.contains("dead-man fire"), "{log}");
    assert!(log.contains("VERIFIED"), "{log}");
}

#[test]
fn a_hung_sigstopped_paid_run_still_fires_the_destroy_and_verifies_it() {
    need_user_manager!();
    let mut w = World::new(test_id(2));
    let vast = FakeVast::start(vec![w.id]);
    w.arm_and_hang(&vast, 2);
    let pid = w.child.as_ref().unwrap().id() as libc::pid_t;
    // SAFETY: kill(2) on our own child's pid.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0);
    w.wait_for_fire(&vast);
    assert!(vast.saw(&format!("DELETE /instances/{}/", w.id)));
    assert!(vast.live.lock().unwrap().is_empty());
    assert!(w.log().contains("VERIFIED"), "{}", w.log());
}

#[test]
fn arm_extend_cancel_against_the_real_user_manager() {
    need_user_manager!();
    let w = World::new(test_id(3));
    // Nothing listens here; the timer never fires within the test.
    let vast = FakeVast::start(vec![]);
    let out = w.command("lifecycle", &vast, 3600).output().unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("LIFECYCLE OK"));
    assert!(!deadman::systemd_loaded(&w.unit).unwrap());
    assert!(vast.requests.lock().unwrap().is_empty());
}

#[test]
fn invalid_instance_ids_are_refused_before_systemd_is_asked() {
    for bad in ["", "12a", "1 2", "1;rm -rf ~", "$(id)", "-5", "１２"] {
        assert!(deadman::parse_id(bad).is_err(), "{bad:?}");
    }
    assert!(deadman::systemd_loaded("vi-deadman-1;true").is_err());
    assert!(deadman::systemd_stop("../vi-deadman-1", true).is_err());
}

// ----- entries run as child processes ------------------------------------------

fn env_u64(name: &str) -> u64 {
    std::env::var(name).unwrap().parse().unwrap()
}

/// The stand-in paid run (`arm_and_hang`) or an in-process lifecycle check.
/// Ignored, so a plain `cargo test` never runs it.
#[test]
#[ignore]
fn child_entry() {
    let Ok(role) = std::env::var("VI_DEADMAN_TEST_ROLE") else {
        return;
    };
    let id = env_u64("VI_DEADMAN_TEST_ID");
    let state = StateDir::new(std::env::var_os("VI_STATE_DIR").unwrap());
    let host = SystemHost;
    let now = host.now();
    let cap_at = now + env_u64("VI_DEADMAN_TEST_FIRE_IN");
    let exe = std::env::current_exe().unwrap();
    let mut argv = vec![exe.to_string_lossy().into_owned()];
    argv.extend(
        ["--exact", "fire_entry", "--ignored", "--nocapture"]
            .iter()
            .map(|s| s.to_string()),
    );
    let unit = deadman::unit_name(id).unwrap();
    match role.as_str() {
        "arm_and_hang" => {
            deadman::arm(&host, &state, argv, id, now, cap_at).unwrap();
            println!("ARMED");
            std::io::stdout().flush().unwrap();
            std::thread::sleep(Duration::from_secs(600));
        }
        "lifecycle" => {
            let r = deadman::arm(&host, &state, argv.clone(), id, now, cap_at).unwrap();
            assert_eq!((r.cap_at, r.fire_at), (cap_at, cap_at));
            assert!(r.env.iter().all(|(k, _)| k != "VAST_API_KEY"));
            assert!(deadman::systemd_loaded(&unit).unwrap());
            // Re-arming the same instance changes nothing.
            assert_eq!(
                deadman::arm(&host, &state, argv.clone(), id, now, cap_at).unwrap(),
                r
            );
            assert!(deadman::systemd_loaded(&unit).unwrap());
            // A heartbeat moves it to now + grace; a long one stops at the cap.
            let e = deadman::extend(&host, &state, id, 600).unwrap();
            assert!(e.fire_at >= now + 600 && e.fire_at < cap_at, "{e:?}");
            assert!(deadman::systemd_loaded(&unit).unwrap());
            let e = deadman::extend(&host, &state, id, 1_000_000).unwrap();
            assert_eq!(e.fire_at, cap_at);
            assert!(deadman::systemd_loaded(&unit).unwrap());
            // Cancel unloads it; again, or for a timer that never was, is a no-op.
            deadman::cancel(&host, &state, id, true).unwrap();
            assert!(!deadman::systemd_loaded(&unit).unwrap());
            assert!(deadman::load(&state, id).unwrap().is_none());
            deadman::cancel(&host, &state, id, true).unwrap();
            deadman::cancel(&host, &state, id + 1, true).unwrap();
            println!("LIFECYCLE OK");
        }
        other => panic!("unknown role {other}"),
    }
}

/// What the timer runs here instead of `vi cloud deadman fire <id>`: the same
/// `Cloud::deadman_fire`, with the same exit 9 when not verified.
#[test]
#[ignore]
fn fire_entry() {
    let Some(dir) = std::env::var_os("VI_STATE_DIR") else {
        return;
    };
    let id = only_record(Path::new(&dir));
    let c = Cloud {
        cfg: Config::default(),
        vast: HttpVast::from_env(),
        state: StateDir::new(&dir),
        host: SystemHost,
    };
    println!("dead-man fire for instance {id}");
    match c.deadman_fire(id) {
        Ok(down) if down.verified => println!("VERIFIED destroyed {:?}", down.ids),
        other => {
            println!("DESTROY NOT VERIFIED: {other:?}");
            std::process::exit(9);
        }
    }
}

/// The id in the single `deadman-<id>.json` of a test's state dir.
fn only_record(dir: &Path) -> u64 {
    let ids: Vec<u64> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_prefix("deadman-")?
                .strip_suffix(".json")?
                .parse()
                .ok()
        })
        .collect();
    assert_eq!(ids.len(), 1, "{ids:?}");
    ids[0]
}
