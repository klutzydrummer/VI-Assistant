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
    del_fails: PathBuf,
    sshd: PathBuf,
    kids: PathBuf,
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
            // `-f` lists the sshd session pids in $SSHD (none unless a test
            // writes some); `-P PID` succeeds when PID is a line of $KIDS,
            // that is, when the session runs a command.
            "#!/bin/sh\ncase \"$1\" in\n  -f) [ -s \"$SSHD\" ] && exec cat \"$SSHD\" ;;\n  -P) grep -qx \"$2\" \"$KIDS\" 2>/dev/null && exit 0 ;;\nesac\nexit 1\n",
        );
        write_exe(
            &bin.join("curl"),
            "#!/bin/sh\necho \"$*\" >>\"$CALLS\"\ncase \"$*\" in\n  *\"-X PUT\"*) [ -e \"$PUT_FAILS\" ] && exit 22 ;;\n  *\"-X DELETE\"*) [ -e \"$DEL_FAILS\" ] && { echo 'curl: (22) The requested URL returned error: 403' >&2; exit 22; } ;;\nesac\nexit 0\n",
        );
        let rig = Rig {
            calls: dir.path().join("calls"),
            put_fails: dir.path().join("put-fails"),
            del_fails: dir.path().join("del-fails"),
            sshd: dir.path().join("sshd-pids"),
            kids: dir.path().join("sshd-kids"),
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
        self.run_as(ms, false, &[])
    }

    /// Like `run`, with extra environment variables for the watchdog.
    fn run_env(&self, ms: u64, env: &[(&str, &str)]) -> String {
        self.run_as(ms, false, env)
    }

    /// Like `run`, but the way on-worker.sh restarts a missing watchdog over
    /// ssh: no `R` and no Vast `CONTAINER_*` vars in the environment. The
    /// production paths point at this rig's root and env file.
    fn run_over_ssh(&self, ms: u64) -> String {
        self.run_as(ms, true, &[])
    }

    fn env_file(&self) -> PathBuf {
        self.root.parent().unwrap().join("vi-env")
    }

    fn run_as(&self, ms: u64, over_ssh: bool, env: &[(&str, &str)]) -> String {
        let rendered = onstart::render(&self.config);
        let body = rendered
            .split_once("cat >/root/vi-watchdog.sh <<'EOF'\n")
            .expect("the watchdog heredoc")
            .1
            .split_once("\nEOF\n")
            .expect("the heredoc end")
            .0;
        let script = self.root.parent().unwrap().join("watchdog.sh");
        let body = body
            .replace("/var/log/vi-watchdog.log", &self.log.display().to_string())
            .replace("/root/.vi-env", &self.env_file().display().to_string())
            .replace("/root/vi-jobs", &self.root.display().to_string());
        fs::write(&script, body).unwrap();
        let path = format!(
            "{}:{}",
            self.root.parent().unwrap().join("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let mut cmd = Command::new("bash");
        cmd.arg(&script)
            .env_remove("CONTAINER_ID")
            .env_remove("CONTAINER_API_KEY");
        if over_ssh {
            cmd.env_remove("R");
        } else {
            cmd.env("R", &self.root);
        }
        cmd.env_remove("PENDING_HOLD")
            .env_remove("RECOVERY_MINUTES");
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd
            .env("CALLS", &self.calls)
            .env("PUT_FAILS", &self.put_fails)
            .env("DEL_FAILS", &self.del_fails)
            .env("SSHD", &self.sshd)
            .env("KIDS", &self.kids)
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

    fn read_log(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// An sshd session; `runs_command` is false for home's tunnel (`ssh -N`).
    fn ssh_session(&self, runs_command: bool) {
        fs::write(&self.sshd, "4100\n").unwrap();
        if runs_command {
            fs::write(&self.kids, "4100\n").unwrap();
        }
    }

    /// A job dir with one un-acked pending epoch, all of it backdated.
    fn pending_job(&self) -> PathBuf {
        let job = self.job();
        fs::create_dir(job.join("pending")).unwrap();
        fs::write(job.join("pending/0"), "").unwrap();
        self.backdate(&job.join("pending"), 3600);
        self.backdate(&job.join("pending/0"), 3600);
        job
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
    // Before the hold ends the pending epochs keep the worker even with no
    // lock, trainer or recent change. PENDING_HOLD defaults to 120 min but
    // is capped at 90, so a job that ended 89 minutes ago is still held.
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    let rc = job.join("rc");
    fs::write(&rc, "0").unwrap();
    set_mtime(&rc, now() - 89 * 60);
    rig.backdate(&job.join("pending"), 3600);
    rig.backdate(&job.join("pending/0"), 3600);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());

    // After it (91 min, still inside the old 120), release stops (never
    // destroys) with work still pending.
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    fs::create_dir(job.join("pending")).unwrap();
    fs::write(job.join("pending/0"), "").unwrap();
    let rc = job.join("rc");
    fs::write(&rc, "0").unwrap();
    set_mtime(&rc, now() - 91 * 60);
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
    // DELETE succeeds, so the stop fallback never runs.
    let rig = Rig::new(liveness_config());
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
    let log = rig.read_log();
    assert!(log.contains("release: idle"), "{log}");
    assert!(log.contains("destroyed"), "{log}");
    assert!(!log.contains("fallback"), "{log}");
    assert!(!rig.root.join(".stopped-at").exists());
}

#[test]
fn pending_hold_shorter_than_cap_is_kept() {
    // PENDING_HOLD below the 90-minute cap still applies as set. The idle
    // window is shorter than rc's age, so rc is not a recent change.
    let rig = Rig::new(Config {
        idle_minutes: 5,
        max_hours: 0.0,
        ..Config::default()
    });
    let job = rig.pending_job();
    let rc = job.join("rc");
    fs::write(&rc, "0").unwrap();
    set_mtime(&rc, now() - 11 * 60);
    rig.backdate_active(3600);
    assert_eq!(
        methods(&rig.run_env(400, &[("PENDING_HOLD", "10")])),
        vec!["PUT"]
    );
    assert!(rig.read_log().contains("release: pending"));
}

#[test]
fn pending_hold_cap_reached_proceeds_to_destroy_when_stop_fails() {
    // Even a huge PENDING_HOLD ends at 90 minutes; with stop refused the
    // watchdog goes on to DELETE.
    let rig = Rig::new(liveness_config());
    let job = rig.pending_job();
    let rc = job.join("rc");
    fs::write(&rc, "0").unwrap();
    set_mtime(&rc, now() - 91 * 60);
    rig.backdate_active(3600);
    fs::write(&rig.put_fails, "").unwrap();
    assert_eq!(
        methods(&rig.run_env(1000, &[("PENDING_HOLD", "100000")])),
        vec!["PUT", "PUT", "PUT", "DELETE"]
    );
    let log = rig.read_log();
    assert!(log.contains("stop refused; fallback DELETE"), "{log}");
    assert!(log.contains("destroyed"), "{log}");
}

/// A job dir whose every file is old and whose `.active` is old: idle unless a liveness signal
/// says otherwise. Returns the job dir.
fn idle_job(rig: &Rig) -> PathBuf {
    let job = rig.job();
    fs::write(job.join("progress"), "step").unwrap();
    rig.backdate(&job.join("progress"), 3600);
    rig.backdate_active(3600);
    job
}

#[test]
fn checkpoint_push_longer_than_idle_stays_live() {
    // l=t=p=0, no ssh, and no job file changed for far longer than idle_minutes: only the
    // push loop's push.alive says the rsync is still working.
    let rig = Rig::new(liveness_config());
    let job = idle_job(&rig);
    fs::write(job.join("push.pid"), "1").unwrap();
    fs::write(job.join("push.alive"), "").unwrap();
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
}

#[test]
fn stale_push_markers_release_idle() {
    // A dead push leaves push.pid behind and a push.alive nobody refreshes.
    let rig = Rig::new(liveness_config());
    let job = idle_job(&rig);
    fs::write(job.join("push.pid"), "1").unwrap();
    fs::write(job.join("push.alive"), "").unwrap();
    rig.backdate(&job.join("push.alive"), 3600);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
    assert!(rig.read_log().contains("release: idle"));
}

#[test]
fn debug_pass_alone_stays_live() {
    // A debug pass holds no job.lock, train.pid or pending marker; debug.alive keeps it live.
    let rig = Rig::new(liveness_config());
    let job = idle_job(&rig);
    fs::write(job.join("debug.alive"), "").unwrap();
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
}

#[test]
fn stale_debug_alive_releases_idle() {
    let rig = Rig::new(liveness_config());
    let job = idle_job(&rig);
    fs::write(job.join("debug.alive"), "").unwrap();
    rig.backdate(&job.join("debug.alive"), 3600);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
}

#[test]
fn stage_pid_alive_with_recent_change_stays_live() {
    let rig = Rig::new(liveness_config());
    let job = rig.job();
    let mut stage = sleeper();
    fs::write(job.join("stage.pid"), stage.id().to_string()).unwrap();
    fs::write(job.join("progress"), "step").unwrap();
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
    stop(&mut stage);
}

#[test]
fn stale_stage_pid_releases_idle() {
    // A pid that is not running (and no change) is idle.
    let rig = Rig::new(liveness_config());
    let job = idle_job(&rig);
    fs::write(job.join("stage.pid"), "999999999").unwrap();
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
}

#[test]
fn true_idle_still_releases() {
    let rig = Rig::new(liveness_config());
    idle_job(&rig);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
}

#[test]
fn tunnel_only_ssh_still_releases_idle() {
    // Home's reverse tunnel is an sshd session with no command under it.
    let rig = Rig::new(liveness_config());
    rig.ssh_session(false);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
    assert!(rig.read_log().contains("release: idle"));
}

#[test]
fn ssh_session_running_a_command_keeps_worker() {
    let rig = Rig::new(liveness_config());
    rig.ssh_session(true);
    rig.backdate_active(3600);
    assert_eq!(methods(&rig.run(400)), Vec::<&str>::new());
    // It counts as activity, so the idle clock restarts.
    let active = fs::metadata(rig.root.join(".active")).unwrap();
    let age = SystemTime::now()
        .duration_since(active.modified().unwrap())
        .unwrap_or_default();
    assert!(age < Duration::from_secs(60), "{age:?}");
}

#[test]
fn delete_refused_falls_back_to_stop() {
    let rig = Rig::new(liveness_config());
    rig.backdate_active(3600);
    fs::write(&rig.del_fails, "").unwrap();
    assert_eq!(
        methods(&rig.run(1000)),
        vec!["DELETE", "DELETE", "DELETE", "PUT"]
    );
    assert!(rig.root.join(".stopped-at").is_file());
    let log = rig.read_log();
    assert!(log.contains("destroy-failed"), "{log}");
    assert!(log.contains("403"), "{log}");
    assert!(log.contains("DELETE refused; fallback stop"), "{log}");
    assert!(log.contains("stopped; storage bills"), "{log}");
}

#[test]
fn hard_cap_destroys_with_ssh_live_and_pending() {
    // Past max_hours + 10 min a live command session and pending work no
    // longer matter: DELETE, not the max-hours stop.
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    rig.set_stamp(".boot", now() - 3600 - 601);
    rig.ssh_session(true);
    rig.pending_job();
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
    let log = rig.read_log();
    assert!(log.contains("release: hard"), "{log}");
}

#[test]
fn hard_cap_ignores_recovery_window_after_stop() {
    // A restart after a stop gets RECOVERY_MINUTES, but never past the cap.
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    rig.set_stamp(".boot", now() - 7200);
    fs::write(rig.root.join(".stopped-at"), "").unwrap();
    rig.pending_job();
    assert_eq!(methods(&rig.run(400)), vec!["DELETE"]);
}

#[test]
fn between_cap_and_hard_cap_pending_still_stops() {
    // From max_hours to max_hours + 10 min the D2 checkpoint hold still
    // stops rather than destroys.
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    rig.set_stamp(".boot", now() - 3600 - 300);
    rig.ssh_session(true);
    rig.pending_job();
    assert_eq!(methods(&rig.run(400)), vec!["PUT"]);
    assert!(rig.read_log().contains("release: max-hours"));
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
    rig.set_stamp(".boot", now() - 3600 - 300);
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
    rig.set_stamp(".boot", now() - 3600 - 300);
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

#[test]
fn ssh_restarted_watchdog_reads_saved_root_and_vast_env() {
    let rig = Rig::new(Config {
        idle_minutes: 30,
        max_hours: 1.0,
        ..Config::default()
    });
    // The boot script saves `export -p` lines for the CONTAINER_* vars.
    fs::write(
        rig.env_file(),
        "declare -x CONTAINER_API_KEY=\"k\"\ndeclare -x CONTAINER_ID=\"4242\"\n",
    )
    .unwrap();
    // A fresh boot must not be released: an empty R used to read /.boot as
    // empty, which counts as 0 and released on max-hours at once.
    assert_eq!(rig.run_over_ssh(400), "");
    rig.set_stamp(".boot", now() - 7200);
    let calls = rig.run_over_ssh(400);
    assert_eq!(methods(&calls), vec!["DELETE"]);
    assert!(calls.contains("/instances/4242/"), "{calls}");
    assert!(calls.contains("Authorization: Bearer k"), "{calls}");
}
