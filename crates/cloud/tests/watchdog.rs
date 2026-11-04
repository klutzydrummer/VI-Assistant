//! The rendered worker watchdog is a shell script; these tests run it with a
//! fake `curl` that records each call and a fast fake `sleep`, against a
//! temporary job root (`R` replaces `/root/vi-jobs` in production).

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cloud::onstart;
use cloud::Config;

/// Everything one watchdog run needs: the rendered script, a job root, and the
/// fake commands that record what it did.
struct Rig {
    _dir: tempfile::TempDir,
    root: PathBuf,
    log: PathBuf,
    calls: PathBuf,
    put_fails: PathBuf,
    config: Config,
}

impl Rig {
    fn new(config: Config) -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("vi-jobs");
        fs::create_dir(&root).unwrap();
        let bin = dir.path().join("bin");
        fs::create_dir(&bin).unwrap();
        let real_sleep = which("sleep");
        write_exe(
            &bin.join("sleep"),
            &format!("#!/bin/sh\nexec {real_sleep} 0.02\n"),
        );
        write_exe(
            &bin.join("pgrep"),
            "#!/bin/sh\nexit 1\n", // no ssh session unless a test says so
        );
        write_exe(
            &bin.join("curl"),
            "#!/bin/sh\necho \"$*\" >>\"$CALLS\"\ncase \"$*\" in\n  *\"-X PUT\"*) [ -e \"$PUT_FAILS\" ] && exit 22 ;;\nesac\nexit 0\n",
        );
        let rig = Rig {
            calls: dir.path().join("calls"),
            put_fails: dir.path().join("put-fails"),
            log: dir.path().join("watchdog.log"),
            root,
            config,
            _dir: dir,
        };
        fs::write(rig.root.join(".active"), "").unwrap();
        fs::write(rig.root.join(".boot"), now().to_string()).unwrap();
        fs::write(rig.root.join(".this-boot"), now().to_string()).unwrap();
        rig
    }

    /// Run the watchdog until it exits or `ms` passes, then return the fake
    /// curl calls it made, one per line.
    fn run(&self, ms: u64) -> String {
        let rendered = onstart::render(&self.config);
        let body = rendered
            .split_once("cat >/root/vi-watchdog.sh <<'EOF'\n")
            .expect("the watchdog heredoc")
            .1
            .split_once("\nEOF\n")
            .expect("the heredoc end")
            .0;
        let script = self.root.parent().unwrap().join("watchdog.sh");
        fs::write(
            &script,
            body.replace("/var/log/vi-watchdog.log", &self.log.display().to_string()),
        )
        .unwrap();
        let path = format!(
            "{}:{}",
            self.root.parent().unwrap().join("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let mut child = Command::new("bash")
            .arg(&script)
            .env("R", &self.root)
            .env("CALLS", &self.calls)
            .env("PUT_FAILS", &self.put_fails)
            .env("PATH", path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        wait_or_kill(&mut child, ms);
        fs::read_to_string(&self.calls).unwrap_or_default()
    }

    fn job(&self) -> PathBuf {
        let p = self.root.join("vi-jobs.job.1.2");
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn backdate(&self, path: &Path, seconds: u64) {
        set_mtime(path, now() - seconds);
    }

    fn backdate_active(&self, seconds: u64) {
        self.backdate(&self.root.join(".active"), seconds);
    }

    /// The watchdog reads `.boot` and `.this-boot` as timestamps, not mtimes.
    fn set_stamp(&self, name: &str, unix: u64) {
        fs::write(self.root.join(name), unix.to_string()).unwrap();
    }
}

fn which(name: &str) -> String {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name}"))
        .output()
        .unwrap();
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(!path.is_empty(), "{name} is not on PATH");
    path
}

fn write_exe(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn set_mtime(path: &Path, unix: u64) {
    let status = Command::new("touch")
        .arg("-d")
        .arg(format!("@{unix}"))
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success(), "touch failed for {}", path.display());
}

fn wait_or_kill(child: &mut Child, ms: u64) {
    let deadline = Instant::now() + Duration::from_millis(ms);
    while Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn methods(calls: &str) -> Vec<&str> {
    calls
        .lines()
        .map(|line| {
            if line.contains("-X PUT") {
                "PUT"
            } else if line.contains("-X DELETE") {
                "DELETE"
            } else {
                "OTHER"
            }
        })
        .collect()
}

fn liveness_config() -> Config {
    Config {
        idle_minutes: 30,
        max_hours: 0.0, // max_hours is tested separately
        ..Config::default()
    }
}

fn hold_lock(path: &Path) -> Child {
    Command::new("flock")
        .arg(path)
        .arg("sleep")
        .arg("30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

fn sleeper() -> Child {
    Command::new("sleep")
        .arg("30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn lock_held_with_recent_change_does_not_release() {
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    let lock = job.join("job.lock");
    let mut holder = hold_lock(&lock);
    fs::write(job.join("progress"), "step").unwrap();
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
    stop(&mut holder);
}

#[test]
fn trainer_alive_with_recent_change_does_not_release() {
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    let mut trainer = sleeper();
    fs::write(job.join("train.pid"), trainer.id().to_string()).unwrap();
    fs::write(job.join("progress"), "step").unwrap();
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
    stop(&mut trainer);
}

#[test]
fn lock_held_without_change_releases() {
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    let lock = job.join("job.lock");
    let mut holder = hold_lock(&lock);
    rig.backdate(&lock, 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
    stop(&mut holder);
}

#[test]
fn pending_only_holds_until_pending_hold() {
    // Before PENDING_HOLD (120 min by default) the pending epochs hold the
    // worker even with no lock, trainer or recent change. The job ended 31
    // minutes ago: past the idle window, inside PENDING_HOLD.
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    let rc = job.join("rc");
    fs::write(&rc, "0").unwrap();
    set_mtime(&rc, now() - 31 * 60);
    rig.backdate(&job.join("pending"), 3600);
    rig.backdate(&job.join("pending/0"), 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());

    // After it, release stops (never destroys) with work still pending.
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    let rc = job.join("rc");
    fs::write(&rc, "0").unwrap();
    set_mtime(&rc, now() - 121 * 60);
    rig.backdate(&job.join("pending"), 3600);
    rig.backdate(&job.join("pending/0"), 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["PUT"]);
    assert!(rig.root.join(".stopped-at").is_file());
}

#[test]
fn idle_with_pending_stops() {
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    rig.backdate(&job.join("pending"), 3600);
    rig.backdate(&job.join("pending/0"), 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["PUT"]);
}

#[test]
fn nothing_releases() {
    let rig = Rig::new(liveness_config());
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
}

#[test]
fn platter_clips_count_as_progress() {
    // job.log is silent and only a new listening clip appears: that is
    // progress, so a job with pending work keeps the worker. Every directory
    // is backdated, so the clip is the one recent entry.
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    fs::write(job.join("job.log"), "").unwrap();
    let platter = job.join("out/platter");
    fs::create_dir_all(&platter).unwrap();
    fs::write(platter.join("clip.wav"), "wav").unwrap();
    rig.backdate(&job.join("pending"), 3600);
    rig.backdate(&job.join("pending/0"), 3600);
    rig.backdate(&job.join("job.log"), 3600);
    rig.backdate(&job.join("out"), 3600);
    rig.backdate(&platter, 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
}

#[test]
fn hung_job_with_pending_stops() {
    // A hung trainer (pid alive, no file changes) with a pending epoch is let
    // go: stop, never destroy, so the un-acked work keeps its disk.
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    let lock = job.join("job.lock");
    let mut holder = hold_lock(&lock);
    let mut trainer = sleeper();
    fs::write(job.join("train.pid"), trainer.id().to_string()).unwrap();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    rig.backdate(&lock, 3600);
    rig.backdate(&job.join("train.pid"), 3600);
    rig.backdate(&job.join("pending"), 3600);
    rig.backdate(&job.join("pending/0"), 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["PUT"]);
    stop(&mut holder);
    stop(&mut trainer);
}

#[test]
fn max_hours_with_pending_issues_put_stop_no_delete() {
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    rig.set_stamp(".boot", now() - 7200);
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    assert_eq!(methods(&rig.run(400)), vec!["PUT"]);
    assert!(rig.root.join(".stopped-at").is_file());
    let log = fs::read_to_string(&rig.log).unwrap();
    assert!(log.contains("stopped"), "{log}");
}

#[test]
fn max_hours_with_pending_stop_fails_then_delete() {
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    rig.set_stamp(".boot", now() - 7200);
    fs::write(&rig.put_fails, "").unwrap();
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    assert_eq!(methods(&rig.run(1000)), vec!["PUT", "PUT", "PUT", "DELETE"]);
}

#[test]
fn max_hours_without_pending_deletes() {
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    rig.set_stamp(".boot", now() - 7200);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
}

#[test]
fn recovery_window_after_stopped_at_then_stop_again() {
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    fs::write(rig.root.join(".stopped-at"), "").unwrap();
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    // Inside the recovery window the new boot has all of RECOVERY_MINUTES.
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
    // This boot started more than RECOVERY_MINUTES (30) ago.
    rig.set_stamp(".this-boot", now() - 31 * 60);
    assert_eq!(methods(&rig.run(400)), vec!["PUT"]);
}
