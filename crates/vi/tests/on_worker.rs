//! `jobs/on-worker.sh` through `vi dataset run`, with the "worker" in a local
//! folder (VI_WORKER_HOST=local) so no GPU or rented machine is needed.

use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::time::{Duration, Instant};

const VI: &str = env!("CARGO_BIN_EXE_vi");

fn jobs_file(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../jobs")
        .join(name)
        .canonicalize()
        .unwrap()
}

/// Test-only pids: `kill -0` checks liveness without signalling anything else.
fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success()
}

fn wait_until(what: &str, seconds: u64, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while !cond() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The stage session dying signals its whole process group, as sshd does.
fn hup_group(pgid: u32) {
    let ok = Command::new("kill")
        .args(["-s", "HUP", "--", &format!("-{pgid}")])
        .status()
        .unwrap()
        .success();
    assert!(ok, "could not HUP process group {pgid}");
}

fn run_worker_job(dir: &Path, run: &Path, cmd: &[&str], envs: &[(&str, String)]) -> Output {
    let work = dir.join("work");
    let mut c = Command::new("bash");
    c.arg(jobs_file("worker-job.sh"))
        .args([
            "--work",
            work.to_str().unwrap(),
            "--run",
            run.to_str().unwrap(),
            "--",
        ])
        .args(cmd)
        .envs(envs.iter().map(|(k, v)| (*k, v.as_str())))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    c.output().unwrap()
}

fn on_worker() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../jobs/on-worker.sh")
        .canonicalize()
        .unwrap()
        .display()
        .to_string()
}

fn have_rsync() -> bool {
    let ok = Command::new("rsync").arg("--version").output().is_ok();
    if !ok {
        eprintln!("skipping: rsync is not installed");
    }
    ok
}

fn run(dir: &Path, job: &str) -> Output {
    run_with(dir, job, &[])
}

fn run_with(dir: &Path, job: &str, envs: &[(&str, String)]) -> Output {
    // The runner always gives the stage a checkpoint dir, so a stage that
    // publishes no epochs must say so: on-worker.sh fails closed when a
    // durable dir exists and expect_epochs is missing.
    run_settings(dir, job, "judge = \"gemma\", expect_epochs = 0", envs)
}

/// The `vi dataset run` command for a stage with a custom `settings` table.
fn command_settings(dir: &Path, job: &str, settings: &str, envs: &[(&str, String)]) -> Command {
    let pipeline = dir.join("pipeline.toml");
    fs::write(dir.join("extra.txt"), "extra\n").unwrap();
    fs::write(
        &pipeline,
        format!(
            r#"
corpus = "corpus"

[stage.gen]
command = ["sh", "-c", "echo hello > \"$VI_OUT/a.txt\""]

[stage.on-gpu]
command = [{:?}, "--put", "extra.txt", "sh", "-c", {job:?}]
inputs = ["gen"]
settings = {{ {settings} }}
"#,
            on_worker()
        ),
    )
    .unwrap();
    let mut cmd = Command::new(VI);
    cmd.args(["dataset", "run", pipeline.to_str().unwrap()])
        .env("VI_WORKER_HOST", "local")
        .env("VI_WORKER_ROOT", dir.join("worker"))
        .env("VI_STATE_DIR", dir.join("state"))
        .env("HOME", dir.join("home"))
        .envs(envs.iter().map(|(k, v)| (*k, v.as_str())))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Like `run_with`, but with a custom `settings` table for the on-gpu stage.
fn run_settings(dir: &Path, job: &str, settings: &str, envs: &[(&str, String)]) -> Output {
    command_settings(dir, job, settings, envs).output().unwrap()
}

/// Everything the job left on the worker besides the watchdog's activity file.
fn leftovers(dir: &Path) -> Vec<String> {
    fs::read_dir(dir.join("worker"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n != ".active")
        .collect()
}

/// The one `on-gpu.*` work dir a failed run kept on the "worker".
fn kept_work(dir: &Path) -> PathBuf {
    let mut paths: Vec<_> = fs::read_dir(dir.join("worker"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("on-gpu.")
        })
        .collect();
    assert_eq!(paths.len(), 1, "{paths:?}");
    paths.remove(0)
}

/// The single durable publish dir the on-gpu stage created for its attempt.
fn durable_dir(dir: &Path) -> PathBuf {
    let root = dir.join("home/.cache/vi/checkpoints/on-gpu");
    let mut entries: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    entries.remove(0)
}

/// A shell job that publishes `n` valid epochs (tagged `tag`) into the
/// vi-publish module, then writes a small output file.
fn publish_job(n: usize, tag: &str) -> String {
    let mut s = String::from("set -e\np=pub\n");
    for k in 0..n {
        s += &format!(
            "mkdir -p \"$p/epoch-{k}\"\nprintf '{tag}{k}' > \"$p/epoch-{k}/talker.safetensors\"\nprintf '{{}}' > \"$p/epoch-{k}/tune.json\"\n( cd \"$p/epoch-{k}\" && sha256sum talker.safetensors tune.json > SHA256SUMS )\n"
        );
    }
    s += "jobs/worker-return.sh push \"$p\" vi-publish\necho ok > \"$VI_OUT/ok.txt\"\n";
    s
}

/// The `*.tar.zst` bundles in `dir`, ignoring the packer's `.lock` file.
fn bundle_tars(dir: &Path) -> Vec<String> {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tar.zst"))
        .collect()
}

/// Wait until on-worker.sh has created the stage's work dir and the job has
/// written its receiver port into `$VI_OUT/port`.
fn wait_for_work(dir: &Path) -> PathBuf {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(entries) = fs::read_dir(dir.join("worker")) {
            for e in entries.flatten() {
                let p = e.path();
                if p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("on-gpu.")
                    && p.join("out/port").is_file()
                {
                    return p;
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "the job never reached its wait loop"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn jobs_dir() -> PathBuf {
    jobs_file("on-worker.sh").parent().unwrap().to_path_buf()
}

fn unique_token(prefix: &str) -> String {
    format!(
        "viow-{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

/// Poll every second so a test never waits for the production 30 s cadence.
fn poll_env() -> Vec<(&'static str, String)> {
    vec![("VI_POLL_SECONDS", "1".to_string())]
}

/// Start `vi dataset run` with its output in files, so the test can read the
/// supervisor's log while the run is still going.
fn start_stage(
    dir: &Path,
    name: &str,
    job: &str,
    settings: &str,
    envs: &[(&str, String)],
) -> (Child, PathBuf) {
    let err = dir.join(format!("{name}.err"));
    let out = dir.join(format!("{name}.out"));
    let child = command_settings(dir, job, settings, envs)
        .stdout(Stdio::from(fs::File::create(&out).unwrap()))
        .stderr(Stdio::from(fs::File::create(&err).unwrap()))
        .spawn()
        .unwrap();
    (child, err)
}

fn err_text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

/// Wait for the run, killing it (and failing the test) if it hangs.
fn wait_stage(child: &mut Child, secs: u64) -> ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the stage run did not finish within {secs}s");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn pgrep_f(pattern: &str) -> Vec<u32> {
    let out = Command::new("pgrep")
        .args(["-f", pattern])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// The home supervisor pid for one test run, named by a token only this test's
/// job text carries.
fn supervisor_pid(token: &str) -> u32 {
    let pattern = format!("on-worker\\.sh.*{token}");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(pid) = pgrep_f(&pattern).into_iter().next() {
            return pid;
        }
        assert!(
            Instant::now() < deadline,
            "on-worker.sh carrying {token} was not found"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn send_signal(pid: u32, sig: &str) -> bool {
    Command::new("kill")
        .args([sig, &pid.to_string()])
        .status()
        .unwrap()
        .success()
}

/// A SIGSTOPped supervisor is resumed even when the test fails early.
struct Paused(u32);

impl Drop for Paused {
    fn drop(&mut self) {
        send_signal(self.0, "-CONT");
    }
}

fn pause(pid: u32) -> Paused {
    assert!(send_signal(pid, "-STOP"), "could not SIGSTOP {pid}");
    Paused(pid)
}

/// Live children a test started and must reap.
struct Kids(Vec<Child>);

impl Drop for Kids {
    fn drop(&mut self) {
        for c in &mut self.0 {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// `on-gpu.<key>.<attempt>` -> (key, attempt).
fn split_attempt(work: &Path) -> (String, String) {
    let name = work.file_name().unwrap().to_string_lossy().into_owned();
    let (rest, attempt) = name.rsplit_once('.').expect("no attempt suffix");
    let key = rest.split_once('.').expect("no stage prefix").1;
    (key.to_string(), attempt.to_string())
}

/// The attempt's durable dir once on-worker.sh has made it, without failing
/// while the run is still starting.
fn maybe_durable(dir: &Path) -> Option<PathBuf> {
    let root = dir.join("home/.cache/vi/checkpoints/on-gpu");
    let entries: Vec<_> = fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    (entries.len() == 1).then(|| entries[0].clone())
}

/// The home checkpoint dir the runner gave this stage.
fn home_checkpoint(dir: &Path) -> PathBuf {
    let stage = dir.join("corpus/stages/on-gpu");
    let mut found: Vec<PathBuf> = fs::read_dir(&stage)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".checkpoint-")
        })
        .collect();
    assert_eq!(found.len(), 1, "{found:?}");
    found.remove(0)
}

fn progress_lines(cp: &Path) -> usize {
    fs::read_to_string(cp.join("progress.jsonl"))
        .map(|s| s.lines().count())
        .unwrap_or(0)
}

/// The receiver pid recorded in this run's return state dir.
fn receiver_pid(dir: &Path) -> Option<u32> {
    for e in fs::read_dir(dir.join("state/return")).ok()?.flatten() {
        if let Ok(s) = fs::read_to_string(e.path().join("rsyncd.pid")) {
            if let Ok(pid) = s.trim().parse() {
                return Some(pid);
            }
        }
    }
    None
}

/// A synthetic `on-gpu.<key>.<attempt>` work dir with a live stage pid and
/// fresh activity, so `pick_attempt` counts it as LIVE.
fn fake_live(root: &Path, key: &str, attempt: &str, kids: &mut Kids) -> PathBuf {
    let d = root.join(format!("on-gpu.{key}.{attempt}"));
    fs::create_dir_all(d.join("out")).unwrap();
    let c = Command::new("sleep")
        .arg("600")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    fs::write(d.join("stage.pid"), c.id().to_string()).unwrap();
    fs::write(d.join("activity"), "x").unwrap();
    kids.0.push(c);
    d
}

/// The unverified marker on-worker.sh writes before a stage launches.
fn write_marker(dir: &Path, key: &str, attempt: &str) -> PathBuf {
    let mdir = dir.join("state/unverified");
    fs::create_dir_all(&mdir).unwrap();
    let m = mdir.join(format!("on-gpu-{key}-{attempt}"));
    let durable = dir.join(format!("home/.cache/vi/checkpoints/on-gpu/{key}-{attempt}"));
    fs::write(&m, format!("{}\n", durable.display())).unwrap();
    m
}

/// Every regular file under `root` with its bytes, sorted by relative path.
fn tree_files(root: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        let mut paths: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        for p in paths {
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                out.push((
                    p.strip_prefix(root).unwrap().to_string_lossy().into_owned(),
                    fs::read(&p).unwrap(),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out
}

#[test]
fn ctrl_c_mid_job_keeps_worker_dir() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let tmp = dir.path().join("tmp");
    fs::create_dir(&tmp).unwrap();
    let token = format!(
        "viow-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    // The job records the receiver's port and waits, so the signal arrives
    // while it is running. The test releases the wait once the supervisor has
    // stopped, which lets the detached stage finish and exit.
    let job = format!(
        r#"# {token}
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
while [ ! -e "$VI_OUT/release" ]; do sleep 0.1; done"#
    );
    let envs = [
        ("TMPDIR", tmp.display().to_string()),
        // A short poll interval keeps the deferred INT trap a fraction of a
        // second away instead of the 30 s default.
        ("VI_POLL_SECONDS", "1".to_string()),
    ];
    let mut child = command_settings(dir.path(), &job, "judge = \"gemma\"", &envs)
        .spawn()
        .unwrap();
    let work = wait_for_work(dir.path());
    let port: u16 = fs::read_to_string(work.join("out/port"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    // SIGINT the home supervisor while the job runs (row 9). Command
    // substitutions and sleeps defer the trap until the foreground command
    // returns, so give it a few polls to take effect.
    let pgrep = Command::new("pgrep")
        .args(["-f", &format!("on-worker\\.sh.*{token}")])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&pgrep.stdout).into_owned();
    let pid = stdout
        .split_whitespace()
        .next()
        .unwrap_or_else(|| {
            panic!(
                "on-worker.sh not found: {}",
                String::from_utf8_lossy(&pgrep.stderr)
            )
        })
        .to_string();
    assert!(Command::new("kill")
        .args(["-INT", &pid])
        .status()
        .unwrap()
        .success());

    let deadline = Instant::now() + Duration::from_secs(15);
    let mut stopped = None;
    while stopped.is_none() {
        stopped = child.try_wait().unwrap();
        assert!(Instant::now() < deadline, "the supervisor ignored SIGINT");
        std::thread::sleep(Duration::from_millis(100));
    }
    // The stage is detached, so it must be released separately; then it can
    // finish and write its exit code.
    fs::write(work.join("out/release"), "").unwrap();
    wait_until("the detached stage did not finish", 15, || {
        work.join("stage.rc").is_file()
    });

    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("keeping"),
        "the runner did not keep the worker dir: {stderr}"
    );

    // The un-pushed worker dir survives the signal.
    assert!(work.is_dir(), "{} was deleted", work.display());
    // The receiver is stopped: its state dir (under TMPDIR) is gone and the
    // loopback port refuses connections. Local mode has no SSH tunnel.
    assert_eq!(
        fs::read_dir(&tmp).unwrap().count(),
        0,
        "the receiver state was left behind"
    );
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    assert!(
        TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_err(),
        "the receiver is still listening on {port}"
    );
}

#[test]
fn gpu_stage_gate_uses_reserve() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // The gate stats $HOME, so it must exist before the stage runs.
    fs::create_dir(dir.path().join("home")).unwrap();
    // Free space on any real machine is far below 1e9 GiB, so the gate must
    // refuse, and its message must name the reserve. Without --reserve-gb it
    // would only name min_home_gb.
    let out = run_settings(
        dir.path(),
        "echo should-not-run > \"$VI_OUT/ran.txt\"",
        "min_home_gb = 0, home_gate_reserve_gb = 1000000000",
        &[],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("plus a 1000000000.0 GiB reserve"),
        "{stderr}"
    );
    // The gate runs before anything touches the worker.
    assert!(!dir.path().join("worker").exists(), "{stderr}");
}

#[test]
fn stale_bundle_part_does_not_block_rerun() {
    if !have_rsync() {
        return;
    }
    if Command::new("zstd").arg("--version").output().is_err() {
        eprintln!("skipping: zstd is not installed");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let bundle_dir = dir.path().join("bundles");
    let envs = [
        ("VI_PACK_MIN_KB", "0".to_string()),
        ("VI_BUNDLE_DIR", bundle_dir.display().to_string()),
    ];
    let job = r#"cat "$VI_INPUT_GEN/a.txt" > "$VI_OUT/b.txt""#;
    let first = run_with(dir.path(), job, &envs);
    let stderr = String::from_utf8_lossy(&first.stderr);
    assert!(first.status.success(), "{stderr}");
    let packed = bundle_tars(&bundle_dir);
    assert_eq!(packed.len(), 1, "{packed:?}");

    // A killed pack left its `.part` behind and no finished bundle.
    let bundle = bundle_dir.join(&packed[0]);
    fs::remove_file(&bundle).unwrap();
    let part = PathBuf::from(format!("{}.part", bundle.display()));
    fs::write(&part, b"stale").unwrap();

    let second = run_with(dir.path(), &format!("{job}; true"), &envs);
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(second.status.success(), "{stderr}");
    assert!(!part.exists(), "the stale .part was not replaced");
    assert_eq!(bundle_tars(&bundle_dir), packed);
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
}

#[test]
fn durable_dir_survives_runner_error_path() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let failing = format!("{}exit 3\n", publish_job(1, "w"));
    let first = run_settings(dir.path(), &failing, "expect_epochs = 1", &[]);
    assert!(!first.status.success());
    let d = durable_dir(dir.path());
    assert!(d.join("epoch-0/talker.safetensors").is_file());
    let before = tree_files(&d);

    // A rerun would normally reattach to the newest unverified marker; this
    // test is about a genuinely new attempt, so clear the marker first (the
    // same thing `checkpoints.py abandon` does when a human gives up on one).
    for e in fs::read_dir(dir.path().join("state/unverified"))
        .unwrap()
        .flatten()
    {
        fs::remove_file(e.path()).unwrap();
    }
    // The rerun has the same stage key, so it deletes its own `.tmp-` scratch
    // dir; the first attempt's durable dir lives outside corpus/ and must not
    // change. Sleep past the durable name's one-second resolution so this
    // attempt gets its own dir.
    std::thread::sleep(Duration::from_millis(1100));
    let second = run_settings(dir.path(), &failing, "expect_epochs = 1", &[]);
    assert!(!second.status.success());
    assert_eq!(
        tree_files(&d),
        before,
        "the first attempt's durable dir changed"
    );
    let root = dir.path().join("home/.cache/vi/checkpoints/on-gpu");
    assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
}

/// A rerun after a FAILED stage starts a fresh attempt (and keeps the old
/// marker, so the old work stays unverified); a finished-ok stage whose output
/// never came home is still reattached.
#[test]
fn rerun_after_failed_stage_is_fresh_but_finished_ok_reattaches() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let failing = format!("{}exit 3\n", publish_job(1, "w"));
    let first = run_settings(dir.path(), &failing, "expect_epochs = 1", &[]);
    assert!(!first.status.success());
    let markers = || {
        fs::read_dir(dir.path().join("state/unverified"))
            .unwrap()
            .count()
    };
    assert_eq!(markers(), 1);

    std::thread::sleep(Duration::from_millis(1100));
    let second = run_settings(dir.path(), &failing, "expect_epochs = 1", &[]);
    let err = String::from_utf8_lossy(&second.stderr);
    assert!(!second.status.success(), "{err}");
    assert!(!err.contains("reattaching to"), "{err}");
    assert_eq!(markers(), 2, "the old unverified marker must stay");

    // Pretend the newest attempt's stage finished fine but its output never
    // came home: that one must be reattached, not restarted.
    let newest = fs::read_dir(dir.path().join("worker"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("stage.rc").is_file())
        .max()
        .expect("a worker dir with stage.rc");
    fs::write(newest.join("stage.rc"), "0\n").unwrap();
    // The second run's marker is already the newest (>1 s younger than the first).
    let third = run_settings(dir.path(), &failing, "expect_epochs = 1", &[]);
    let err = String::from_utf8_lossy(&third.stderr);
    assert!(
        err.contains(&format!("reattaching to {}", newest.display())),
        "{err}"
    );
}

#[test]
fn stage_runs_on_the_worker_and_its_output_comes_home() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run(
        dir.path(),
        r#"set -e
test "$VI_WORKER" = 1
test -x jobs/on-worker.sh
cat "$VI_INPUT_GEN/a.txt" files/extra.txt > "$VI_OUT/b.txt"
cp "$VI_SETTINGS" "$VI_OUT/settings.json"
pwd > "$VI_OUT/where.txt""#,
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert_eq!(
        fs::read_to_string(got.join("b.txt")).unwrap(),
        "hello\nextra\n"
    );
    assert!(fs::read_to_string(got.join("settings.json"))
        .unwrap()
        .contains("gemma"));
    let worker = dir.path().join("worker").canonicalize().unwrap();
    let ran_in = fs::read_to_string(got.join("where.txt")).unwrap();
    assert!(Path::new(ran_in.trim()).starts_with(&worker), "{ran_in}");
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
}

/// Row 34: a cheap check fails on the worker before any input is uploaded, so
/// a bad machine is refused while nothing but the rental is billed.
#[test]
fn cheap_checks_run_before_any_input_upload() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // No machine has a million effective cores, so this always fails.
    let job = "echo ran > \"$VI_OUT/ran.txt\"";
    let out = run_settings(dir.path(), job, "min_cpu_cores = 1000000", &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("worker preflight failed; nothing was uploaded"),
        "{stderr}"
    );
    assert!(stderr.contains("effective CPU cores"), "{stderr}");
    let work = kept_work(dir.path());
    assert_eq!(
        fs::read_dir(work.join("inputs")).unwrap().count(),
        0,
        "an input reached the worker before the cheap checks passed"
    );
    assert_eq!(fs::read_dir(work.join("files")).unwrap().count(), 0);
    assert!(!work.join("out/ran.txt").exists(), "the stage ran anyway");
    assert!(
        !work.join("out/preflight.json").exists(),
        "a failed preflight must write no report"
    );
    assert!(!work.join(".preflight-disk").exists());
    assert!(!work.join(".speedtest").exists());
}

/// Row 34: the dummy push rate is a gate, and it runs before any input.
#[test]
fn slow_push_fails_before_upload() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // 100 MiB in a simulated 40 s is 2.62 MB/s, under the 7 MB/s floor.
    let envs = [("VI_FAKE_PUSH_SECONDS", "40".to_string())];
    let job = "echo ran > \"$VI_OUT/ran.txt\"";
    let out = run_settings(dir.path(), job, "min_push_mb_s = 7", &envs);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("worker push (MB/s)"), "{stderr}");
    assert!(
        stderr.contains("worker preflight failed; nothing was uploaded"),
        "{stderr}"
    );
    let work = kept_work(dir.path());
    assert_eq!(fs::read_dir(work.join("inputs")).unwrap().count(), 0);
    assert!(!work.join("out/ran.txt").exists(), "the stage ran anyway");
}

/// Row 34: the speed test file is home only long enough to time the push; the
/// runner's stage output must never carry it.
#[test]
fn speedtest_dummy_never_reaches_stage_output() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run(dir.path(), "echo ok > \"$VI_OUT/ok.txt\"");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert!(got.join("ok.txt").is_file());
    let files = tree_files(&dir.path().join("corpus"));
    assert!(
        files.iter().all(|(name, _)| !name.ends_with(".speedtest")),
        "the speed test dummy was kept in the stage output: {:?}",
        files
            .iter()
            .map(|(name, _)| name)
            .filter(|n| n.contains("speedtest"))
            .collect::<Vec<_>>()
    );
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
}

/// The fail-closed gap: a stage that publishes a durable checkpoint but does
/// not say how many epochs to expect has nothing verified home, so the worker
/// copy must be kept and the stage must fail.
#[test]
fn durable_without_expect_epochs_keeps_worker_copy_and_fails() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run_settings(dir.path(), &publish_job(1, "w"), "judge = \"gemma\"", &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("sets no expect_epochs"), "{stderr}");
    assert!(stderr.contains("keeping the worker copy"), "{stderr}");
    let work = kept_work(dir.path());
    assert!(
        work.join("out/ok.txt").is_file(),
        "the stage output was not kept"
    );
    // The published epoch did reach home (the output push ran first), so the
    // failure is about verification, not about a missing result.
    let d = durable_dir(dir.path());
    assert!(d.join("epoch-0/talker.safetensors").is_file());
}

#[test]
fn a_packed_input_arrives_whole_and_its_bundle_is_reused() {
    if !have_rsync() {
        return;
    }
    if Command::new("zstd").arg("--version").output().is_err() {
        eprintln!("skipping: zstd is not installed");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let bundles = dir.path().join("bundles");
    // Every input is packed, however small.
    let envs = [
        ("VI_PACK_MIN_KB", "0".to_string()),
        ("VI_BUNDLE_DIR", bundles.display().to_string()),
    ];
    let job = r#"cat "$VI_INPUT_GEN/a.txt" > "$VI_OUT/b.txt""#;
    let out = run_with(dir.path(), job, &envs);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("packed GEN"), "{stderr}");
    assert!(stderr.contains("upload: 1/1"), "{stderr}");
    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert_eq!(fs::read_to_string(got.join("b.txt")).unwrap(), "hello\n");
    assert_eq!(bundle_tars(&bundles).len(), 1);
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());

    // A changed job reruns the stage; the unchanged input's bundle is reused.
    let out = run_with(dir.path(), &format!("{job}; true"), &envs);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("GEN already packed"), "{stderr}");
    assert_eq!(bundle_tars(&bundles).len(), 1);
}

#[test]
fn a_failing_job_fails_the_stage_and_keeps_the_worker_dir() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run(dir.path(), "echo partial > \"$VI_OUT/b.txt\"; exit 3");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("exited with 3"));
    assert!(dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").is_none());
    // The worker's job dir is kept on failure, so nothing is lost; `vi cloud
    // down` removes it with the instance.
    assert_eq!(leftovers(dir.path()).len(), 1);
}

#[test]
fn a_failing_job_still_brings_its_checkpoint_home() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run(
        dir.path(),
        r#"case "$VI_CHECKPOINT" in "$PWD"/*) ;; *) exit 9 ;; esac
echo clip-1 >> "$VI_CHECKPOINT/step.jsonl"; exit 3"#,
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("exited with 3"));
    let stage = dir.path().join("corpus/stages/on-gpu");
    let saved: Vec<_> = fs::read_dir(&stage)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".checkpoint-")
        })
        .collect();
    assert_eq!(saved.len(), 1, "{saved:?}");
    assert_eq!(
        fs::read_to_string(saved[0].join("step.jsonl")).unwrap(),
        "clip-1\n"
    );
    // The worker's copy is kept on failure, so a re-push stays possible.
    assert_eq!(leftovers(dir.path()).len(), 1);
    // The unverified guard wrote a per-attempt marker into the temp state dir
    // (never ~/.local/state/vi), naming the durable publish dir — not the
    // checkpoint scratch the runner still owns.
    let markers: Vec<_> = fs::read_dir(dir.path().join("state/unverified"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(markers.len(), 1);
    assert!(markers[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("on-gpu-"));
    assert_eq!(
        fs::read_to_string(&markers[0]).unwrap().trim(),
        durable_dir(dir.path())
            .canonicalize()
            .unwrap()
            .to_string_lossy()
    );
}

#[test]
fn published_epochs_are_verified_and_survive_runner_cleanup() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run_settings(dir.path(), &publish_job(2, "w"), "expect_epochs = 2", &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let d = durable_dir(dir.path());
    assert!(d.join("VERIFIED").is_file());
    assert!(d.join("epoch-0").is_dir());
    assert!(d.join("epoch-1").is_dir());
    // The runner deleted its own `.checkpoint-<key>` on success; the durable
    // dir lives outside corpus/, so it must still hold the epochs.
    let stage = dir.path().join("corpus/stages/on-gpu");
    let checkpoints: Vec<_> = fs::read_dir(&stage)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".checkpoint-"))
        .collect();
    assert_eq!(checkpoints, Vec::<String>::new(), "{checkpoints:?}");
    assert_eq!(
        fs::read_to_string(d.join("epoch-0/talker.safetensors")).unwrap(),
        "w0"
    );
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
    // Verify cleared the per-attempt marker.
    assert_eq!(
        fs::read_dir(dir.path().join("state/unverified"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn a_partial_publish_fails_the_stage_and_keeps_every_copy() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run_settings(dir.path(), &publish_job(1, "w"), "expect_epochs = 2", &[]);
    assert!(!out.status.success());
    let d = durable_dir(dir.path());
    assert!(d.join("epoch-0").is_dir());
    assert!(!d.join("VERIFIED").exists());
    // The worker's job dir is kept on failure, so a re-push stays possible.
    assert_eq!(leftovers(dir.path()).len(), 1);
    // The per-attempt marker is kept and names the durable dir.
    let markers: Vec<_> = fs::read_dir(dir.path().join("state/unverified"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(markers.len(), 1);
    assert!(markers[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .starts_with("on-gpu-"));
    assert_eq!(
        fs::read_to_string(&markers[0]).unwrap().trim(),
        d.canonicalize().unwrap().to_string_lossy()
    );
}

#[test]
fn a_rerun_gets_a_fresh_durable_dir() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let first = run_settings(dir.path(), &publish_job(1, "a"), "expect_epochs = 1", &[]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    // The second run changes the job, so the stage reruns; its epochs must land
    // in a fresh dir, not mix into the first attempt's.
    let second = run_settings(dir.path(), &publish_job(1, "b"), "expect_epochs = 1", &[]);
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    let root = dir.path().join("home/.cache/vi/checkpoints/on-gpu");
    let dirs: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(dirs.len(), 2, "{dirs:?}");
    let (mut a, mut b) = (None, None);
    for d in dirs {
        assert!(d.join("VERIFIED").is_file());
        let weight = fs::read_to_string(d.join("epoch-0/talker.safetensors")).unwrap();
        match weight.as_str() {
            "a0" => a = Some(d),
            "b0" => b = Some(d),
            other => panic!("unexpected epoch weight {other:?}"),
        }
    }
    assert!(a.is_some(), "first attempt's epoch is missing");
    assert!(b.is_some(), "second attempt's epoch is missing");
}

#[test]
fn check_worker_round_trips_a_token() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../jobs/check-worker.sh");
    let out = Command::new(script)
        .env("VI_WORKER_HOST", "local")
        .env("VI_WORKER_ROOT", dir.path().join("worker"))
        .env("VI_STATE_DIR", dir.path().join("state"))
        .env("HOME", dir.path().join("home"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("check-worker: OK"), "{stdout}");
    assert!(stdout.contains("gpu: "), "{stdout}");
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
}

/// Row 1: the stage session's sshd dies mid-job. The trainer runs in its own
/// session (and the wrapper in another), so both survive the group HUP and the
/// wrapper still records the trainer's exit code.
#[test]
fn detached_job_survives_stage_session_kill() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let job = r#"w=$(dirname "$VI_OUT")
jobs/worker-job.sh --work "$w/wj" --run "$w/run" -- sh -c 'sleep 3'"#;
    let mut child = command_settings(dir.path(), job, "judge = \"gemma\"", &[])
        .process_group(0)
        .spawn()
        .unwrap();
    let (work, trainer) = {
        let mut found = None;
        let mut scan = || {
            if let Ok(entries) = fs::read_dir(dir.path().join("worker")) {
                for e in entries.flatten() {
                    let p = e.path();
                    if !p
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with("on-gpu.")
                    {
                        continue;
                    }
                    if let Ok(s) = fs::read_to_string(p.join("wj/train.pid")) {
                        if let Ok(pid) = s.trim().parse::<u32>() {
                            found = Some((p, pid));
                        }
                    }
                }
            }
            found.is_some()
        };
        wait_until("the detached job never wrote train.pid", 30, &mut scan);
        found.unwrap()
    };

    hup_group(child.id());
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        pid_alive(trainer),
        "the trainer died with the stage session"
    );
    let worker: u32 = fs::read_to_string(work.join("wj/worker.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(
        pid_alive(worker),
        "the detached runner died with the stage session"
    );

    let rc_file = work.join("wj/rc");
    wait_until("the detached runner never wrote rc", 20, || {
        rc_file.is_file()
    });
    assert_eq!(fs::read_to_string(&rc_file).unwrap().trim(), "0");
    let _ = child.wait();
}

/// Row 12: abort.json means the supervisor stopped the run; never restart it.
#[test]
fn wj_does_not_retry_after_abort_json() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("run");
    fs::create_dir_all(&run).unwrap();
    fs::write(run.join("abort.json"), "{}").unwrap();
    let count = dir.path().join("count");
    let out = run_worker_job(
        dir.path(),
        &run,
        &[
            "sh",
            "-c",
            "echo x >> \"$1\"; exit 1",
            "sh",
            count.to_str().unwrap(),
        ],
        &[],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert_eq!(fs::read_to_string(&count).unwrap().lines().count(), 1);
    assert_eq!(
        fs::read_to_string(dir.path().join("work/rc"))
            .unwrap()
            .trim(),
        "1"
    );
}

/// Row 12: a lock collision (exit 75) is another job, not a crash; the command
/// must not run and must not be retried.
#[test]
fn wj_does_not_retry_lock_collision_exit_75() {
    let dir = tempfile::tempdir().unwrap();
    let work = dir.path().join("work");
    let run = dir.path().join("run");
    fs::create_dir_all(&work).unwrap();
    fs::create_dir_all(&run).unwrap();
    let lock = work.join("job.lock");
    fs::write(&lock, "").unwrap();
    let mut holder = Command::new("flock")
        .arg(&lock)
        .arg("sleep")
        .arg("30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_until("the test could not take the job lock", 10, || {
        !Command::new("flock")
            .args(["-n", lock.to_str().unwrap(), "true"])
            .status()
            .unwrap()
            .success()
    });

    let marker = dir.path().join("ran");
    let out = run_worker_job(
        dir.path(),
        &run,
        &[
            "sh",
            "-c",
            "echo ran > \"$1\"",
            "sh",
            marker.to_str().unwrap(),
        ],
        &[],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(75), "{stderr}");
    assert!(!marker.exists(), "the command ran despite the held lock");
    assert_eq!(fs::read_to_string(work.join("rc")).unwrap().trim(), "75");
    let _ = holder.kill();
    let _ = holder.wait();
}

/// Row 14: rc 137 alone is not OOM. Only a rise in cgroup v2's oom_kill
/// counter lowers VI_NUM_WORKERS; without the counter, retry unchanged once.
#[test]
fn wj_reduces_workers_only_when_oom_kill_counter_rises() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("run");
    fs::create_dir_all(&run).unwrap();
    let events = dir.path().join("memory.events");
    let seen = dir.path().join("workers");
    // The first attempt (unset) fakes an OOM kill and fails; only the retry
    // with VI_NUM_WORKERS=0 succeeds.
    fs::write(&events, "oom_kill 1\n").unwrap();
    let script = "echo \"${VI_NUM_WORKERS-unset}\" >> \"$1\"\n\
                  [ \"${VI_NUM_WORKERS-}\" = 0 ] && exit 0\n\
                  echo 'oom_kill 2' > \"$2\"\nexit 137";
    let out = run_worker_job(
        dir.path(),
        &run,
        &[
            "sh",
            "-c",
            script,
            "sh",
            seen.to_str().unwrap(),
            events.to_str().unwrap(),
        ],
        &[("VI_MEMORY_EVENTS", events.display().to_string())],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert_eq!(
        fs::read_to_string(&seen)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        vec!["unset", "0"]
    );

    // The same failure with a flat counter retries unchanged.
    let flat = dir.path().join("flat.events");
    fs::write(&flat, "oom_kill 1\n").unwrap();
    let seen_flat = dir.path().join("workers-flat");
    let script = "echo \"${VI_NUM_WORKERS-unset}\" >> \"$1\"\nexit 137";
    let work2 = dir.path().join("work2");
    let out = Command::new("bash")
        .arg(jobs_file("worker-job.sh"))
        .args([
            "--work",
            work2.to_str().unwrap(),
            "--run",
            run.to_str().unwrap(),
            "--retries",
            "1",
            "--",
            "sh",
            "-c",
            script,
            "sh",
            seen_flat.to_str().unwrap(),
        ])
        .env("VI_MEMORY_EVENTS", &flat)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(137), "{stderr}");
    assert_eq!(
        fs::read_to_string(&seen_flat)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        vec!["unset", "unset"]
    );

    // No cgroup v2 memory.events at all: retry unchanged, but only once.
    let seen_none = dir.path().join("workers-none");
    let work3 = dir.path().join("work3");
    let out = Command::new("bash")
        .arg(jobs_file("worker-job.sh"))
        .args([
            "--work",
            work3.to_str().unwrap(),
            "--run",
            run.to_str().unwrap(),
            "--retries",
            "2",
            "--",
            "sh",
            "-c",
            script,
            "sh",
            seen_none.to_str().unwrap(),
        ])
        .env("VI_MEMORY_EVENTS", dir.path().join("absent"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(137));
    assert_eq!(
        fs::read_to_string(&seen_none)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        vec!["unset", "unset"]
    );
}

/// Row 38: voicetune-fresh.sh dying (HUP from the stage session, then its own
/// abort/teardown) must not kill the trainer; worker-job.sh keeps it running
/// and records the exit. The home supervisor's relaunch is a later task.
#[test]
fn trainer_survives_vf_kill() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let vf = jobs_file("voicetune-fresh.sh");
    let text = fs::read_to_string(&vf).unwrap();
    let rev = text
        .lines()
        .find_map(|l| l.strip_prefix("qwen_rev="))
        .expect("voicetune-fresh.sh names the qwen revision")
        .trim();
    let ft = home.join(format!(".cache/vi/Qwen3-TTS-{rev}/finetuning"));
    fs::create_dir_all(&ft).unwrap();
    fs::write(ft.join("dataset.py"), "").unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let work = dir.path().join("work");
    let out_dir = work.join("out");
    fs::create_dir_all(&out_dir).unwrap();

    // A stand-in for the qwen-tts venv: passes the check gate, then is the
    // long-running trainer.
    let py = dir.path().join("py");
    fs::write(
        &py,
        "#!/bin/sh\ncase \"$*\" in\n\
         *'voicetune.py train'*) exec sleep 30 ;;\n\
         *'voicetune.py check'*)\n\
         out=; prev=; for a in \"$@\"; do [ \"$prev\" = --out ] && out=$a; prev=$a; done\n\
         mkdir -p \"$out\"; echo '{\"vram_gb\": 80, \"free_disk_gb\": 500}' > \"$out/check.json\"\n\
         exit 0 ;;\n\
         *) exit 0 ;;\nesac\n",
    )
    .unwrap();
    fs::set_permissions(&py, fs::Permissions::from_mode(0o755)).unwrap();

    let mut child = Command::new("bash")
        .arg(&vf)
        .arg("train")
        .env("VI_OUT", &out_dir)
        .env("VI_RETURN_PORT", "1")
        .env("VI_INPUT_VOICETUNE_DATA", &data)
        .env("QWEN_TTS_PYTHON", &py)
        .env("VI_TRAIN_RETRIES", "0")
        .env("HOME", &home)
        .process_group(0)
        .spawn()
        .unwrap();
    let trainer = {
        let mut pid = 0;
        wait_until("voicetune-fresh.sh never started the trainer", 30, || {
            pid = fs::read_to_string(work.join("train.pid"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            pid != 0
        });
        pid
    };
    assert!(pid_alive(trainer));

    hup_group(child.id());
    let deadline = Instant::now() + Duration::from_secs(15);
    while child.try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "voicetune-fresh.sh did not exit after the session HUP"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(143));
    assert!(
        pid_alive(trainer),
        "the trainer died with voicetune-fresh.sh"
    );
    let worker: u32 = fs::read_to_string(work.join("worker.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(
        pid_alive(worker),
        "the detached runner died with the session"
    );

    // A trainer killed by its own failure ends this wrapper attempt (no
    // retries); the wrapper exits and the home supervisor can relaunch later.
    let _ = Command::new("kill")
        .args(["-KILL", &trainer.to_string()])
        .status();
    wait_until("the wrapper never recorded the trainer's exit", 20, || {
        work.join("rc").is_file()
    });
    assert_eq!(fs::read_to_string(work.join("rc")).unwrap().trim(), "137");
    wait_until("the detached runner did not exit", 10, || {
        !pid_alive(worker)
    });
}

/// Row 38: a relaunched voicetune-fresh.sh that loses the job lock (exit 75)
/// must not kill the attempt that holds it. The wrapper holding the lock wrote
/// `$work/train.pid`, so the dying script's EXIT trap must leave that trainer
/// alone; the lock holder stands in for the surviving attempt's trainer.
#[test]
fn vf_exit_75_does_not_kill_the_surviving_trainer() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let vf = jobs_file("voicetune-fresh.sh");
    let text = fs::read_to_string(&vf).unwrap();
    let rev = text
        .lines()
        .find_map(|l| l.strip_prefix("qwen_rev="))
        .expect("voicetune-fresh.sh names the qwen revision")
        .trim();
    let ft = home.join(format!(".cache/vi/Qwen3-TTS-{rev}/finetuning"));
    fs::create_dir_all(&ft).unwrap();
    fs::write(ft.join("dataset.py"), "").unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    let work = dir.path().join("work");
    let out_dir = work.join("out");
    fs::create_dir_all(&out_dir).unwrap();

    let py = dir.path().join("py");
    fs::write(
        &py,
        "#!/bin/sh\ncase \"$*\" in\n\
         *'voicetune.py train'*) exec sleep 60 ;;\n\
         *'voicetune.py check'*)\n\
         out=; prev=; for a in \"$@\"; do [ \"$prev\" = --out ] && out=$a; prev=$a; done\n\
         mkdir -p \"$out\"; echo '{\"vram_gb\": 80, \"free_disk_gb\": 500}' > \"$out/check.json\"\n\
         exit 0 ;;\n\
         *) exit 0 ;;\nesac\n",
    )
    .unwrap();
    fs::set_permissions(&py, fs::Permissions::from_mode(0o755)).unwrap();

    // Another attempt is alive and holds the job lock; its wrapper recorded
    // its trainer's pid before this relaunch arrived.
    let lock = work.join("job.lock");
    fs::write(&lock, "").unwrap();
    let mut holder = Command::new("flock")
        .arg(&lock)
        .arg("sleep")
        .arg("60")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_until("the test could not take the job lock", 10, || {
        !Command::new("flock")
            .args(["-n", lock.to_str().unwrap(), "true"])
            .status()
            .unwrap()
            .success()
    });
    fs::write(work.join("train.pid"), holder.id().to_string()).unwrap();

    let out = Command::new("bash")
        .arg(&vf)
        .arg("train")
        .env("VI_OUT", &out_dir)
        .env("VI_RETURN_PORT", "1")
        .env("VI_INPUT_VOICETUNE_DATA", &data)
        .env("QWEN_TTS_PYTHON", &py)
        .env("VI_TRAIN_RETRIES", "0")
        .env("HOME", &home)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(75), "{stderr}");
    assert!(
        pid_alive(holder.id()),
        "voicetune-fresh.sh killed the attempt that holds the job lock: {stderr}"
    );
    let _ = holder.kill();
    let _ = holder.wait();
}

/// Row 39: the cursor decides only what is printed. Deleting it reprints the
/// log from the start and changes no action: the epoch is verified and acked
/// once, and nothing is pushed or repaired twice.
#[test]
fn deleting_cursor_reprints_log_but_changes_no_action() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let job = r#"set -e
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
echo hello-from-job
p=voicetune-fresh/publish
mkdir -p "$p/epoch-0"
printf 'v0' > "$p/epoch-0/talker.safetensors"
printf '{}' > "$p/epoch-0/tune.json"
( cd "$p/epoch-0" && sha256sum talker.safetensors tune.json > SHA256SUMS )
jobs/worker-return.sh push "$p" vi-publish
while [ ! -e "$VI_OUT/release" ]; do sleep 0.1; done
echo after-release
echo ok > "$VI_OUT/ok.txt""#;
    let (mut child, err_path) =
        start_stage(dir.path(), "cursor", job, "expect_epochs = 1", &poll_env());
    let work = wait_for_work(dir.path());
    let (_, attempt) = split_attempt(&work);
    wait_until("the epoch was never acked", 60, || {
        work.join("acked/epoch-0").is_file()
    });
    let cursor = dir
        .path()
        .join("state/hs")
        .join(format!("{attempt}.offset"));
    assert!(cursor.is_file(), "no log cursor at {}", cursor.display());
    fs::remove_file(&cursor).unwrap();
    wait_until("deleting the cursor did not reprint the log", 30, || {
        err_text(&err_path).matches("hello-from-job").count() >= 2
    });
    fs::write(work.join("out/release"), "").unwrap();
    let status = wait_stage(&mut child, 120);
    let err = err_text(&err_path);
    assert!(status.success(), "{err}");
    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert_eq!(fs::read_to_string(got.join("ok.txt")).unwrap(), "ok\n");
    assert_eq!(
        err.matches("epoch 0 verified home and acked").count(),
        1,
        "the ack was repeated after the cursor was deleted: {err}"
    );
    assert!(!err.contains("failed verification"), "{err}");
    assert!(cursor.is_file(), "the cursor was not rewritten");
}

/// Row 19: the checkpoint copy loop lives in worker-job.sh, so it keeps
/// running while the home supervisor is stopped, and the loss is not noticed.
#[test]
fn checkpoint_sync_continues_with_home_supervisor_paused() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let token = unique_token("cpsync");
    let job = format!(
        r#"# {token}
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
w=$(dirname "$VI_OUT")
jobs/worker-job.sh --work "$w/wj" --run "$w/run" -- sh -c 'i=0; while [ $i -lt 20 ]; do i=$((i+1)); echo line-$i >> "$VI_CHECKPOINT/progress.jsonl"; sleep 1; done'"#
    );
    let mut envs = poll_env();
    envs.push(("VI_SYNC_SECONDS", "1".to_string()));
    let (mut child, _err_path) = start_stage(
        dir.path(),
        "cpsync",
        &job,
        "judge = \"gemma\", expect_epochs = 0",
        &envs,
    );
    wait_for_work(dir.path());
    let cp = home_checkpoint(dir.path());
    wait_until("the checkpoint never reached home", 30, || {
        progress_lines(&cp) >= 2
    });
    let supervisor = supervisor_pid(&token);
    let paused = pause(supervisor);
    // Let any push that was already in flight land before taking the baseline.
    std::thread::sleep(Duration::from_secs(2));
    let at_stop = progress_lines(&cp);
    wait_until(
        "worker-job.sh stopped copying the checkpoint home",
        30,
        || progress_lines(&cp) > at_stop,
    );
    drop(paused);
    let status = wait_stage(&mut child, 180);
    let err = err_text(&_err_path);
    assert!(status.success(), "{err}");
    // The grow-while-paused check above is the assertion; the scratch
    // checkpoint is moved to the durable dir once the run succeeds, so it
    // cannot be counted again here.
}

/// Row 6: a home supervisor stopped long enough to miss the whole run resumes
/// and reattaches: the return receiver is restarted and re-proved, the stage
/// is not relaunched, and the output still comes home. The pause is 10
/// seconds of wall clock rather than 10 minutes; the code keys on state, not
/// on how long the process was stopped.
#[test]
fn sigstop_home_supervisor_10min_then_sigcont_reattaches() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let token = unique_token("sigstop");
    let job = format!(
        r#"# {token}
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
echo "$BASHPID" > "$VI_OUT/jobpid"
echo run >> "$VI_OUT/runs"
sleep 8
echo done > "$VI_OUT/done.txt""#
    );
    let (mut child, err_path) = start_stage(
        dir.path(),
        "sigstop",
        &job,
        "judge = \"gemma\", expect_epochs = 0",
        &poll_env(),
    );
    let work = wait_for_work(dir.path());
    wait_until("the job never wrote jobpid", 30, || {
        work.join("out/jobpid").is_file()
    });
    let supervisor = supervisor_pid(&token);
    let paused = pause(supervisor);
    std::thread::sleep(Duration::from_secs(1));
    let receiver = receiver_pid(dir.path()).expect("the receiver never wrote its pid");
    assert!(
        send_signal(receiver, "-TERM"),
        "could not stop the receiver"
    );
    wait_until("the receiver did not die", 10, || !pid_alive(receiver));
    std::thread::sleep(Duration::from_secs(4));
    drop(paused);
    let status = wait_stage(&mut child, 120);
    let err = err_text(&err_path);
    assert!(status.success(), "{err}");
    assert!(
        err.contains("the return receiver is down; restarting it"),
        "{err}"
    );
    assert!(
        err.matches("return path re-proved").count() >= 2,
        "the restarted return path was not proved again: {err}"
    );
    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert_eq!(fs::read_to_string(got.join("done.txt")).unwrap(), "done\n");
    assert_eq!(
        fs::read_to_string(got.join("runs"))
            .unwrap()
            .lines()
            .count(),
        1,
        "the job was launched more than once"
    );
}

/// Row 7: a home SIGKILL leaves the detached stage running. The rerun picks
/// that live attempt, reuses its durable dir, sends no inputs, and collects
/// the output the first run never got.
#[test]
fn rerun_after_home_kill9_reattaches_same_durable_dir() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let token = unique_token("kill9");
    let job = format!(
        r#"# {token}
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
p=voicetune-fresh/publish
mkdir -p "$p/epoch-0"
printf 'w0' > "$p/epoch-0/talker.safetensors"
printf '{{}}' > "$p/epoch-0/tune.json"
( cd "$p/epoch-0" && sha256sum talker.safetensors tune.json > SHA256SUMS )
jobs/worker-return.sh push "$p" vi-publish
while [ ! -e "$VI_OUT/release" ]; do sleep 0.1; done
echo done > "$VI_OUT/ok.txt""#
    );
    let (mut first, _first_err) = start_stage(
        dir.path(),
        "kill9-first",
        &job,
        "expect_epochs = 1",
        &poll_env(),
    );
    let work = wait_for_work(dir.path());
    wait_until("the epoch was never acked", 60, || {
        work.join("acked/epoch-0").is_file()
    });
    let durable_before = durable_dir(dir.path());
    let receiver = receiver_pid(dir.path());
    let supervisor = supervisor_pid(&token);
    assert!(send_signal(supervisor, "-KILL"));
    let status = wait_stage(&mut first, 30);
    assert!(!status.success(), "the killed run reported success");
    if let Some(pid) = receiver {
        wait_until("the receiver outlived its owner", 10, || !pid_alive(pid));
    }

    let (mut second, err_path) = start_stage(
        dir.path(),
        "kill9-second",
        &job,
        "expect_epochs = 1",
        &poll_env(),
    );
    wait_until("the rerun did not reattach", 30, || {
        err_text(&err_path).contains("reattaching to")
    });
    let err = err_text(&err_path);
    assert!(
        err.contains(&format!("reattaching to {}", work.display())),
        "{err}"
    );
    assert!(err.contains("no inputs will be sent"), "{err}");
    fs::write(work.join("out/release"), "").unwrap();
    let status = wait_stage(&mut second, 180);
    let err = err_text(&err_path);
    assert!(status.success(), "{err}");
    assert_eq!(
        durable_dir(dir.path()).canonicalize().unwrap(),
        durable_before.canonicalize().unwrap(),
        "the rerun used a different durable dir"
    );
    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert_eq!(fs::read_to_string(got.join("ok.txt")).unwrap(), "done\n");
}

/// Row 7: two live attempts are a human decision, never a silent pick.
#[test]
fn reattach_refuses_when_two_live_candidates() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let probe = "echo \"$VI_RETURN_PORT\" > \"$VI_OUT/port\"; exit 3";
    let first = run_settings(dir.path(), probe, "expect_epochs = 1", &poll_env());
    assert!(!first.status.success());
    let work = wait_for_work(dir.path());
    let (key, _attempt) = split_attempt(&work);
    let root = dir.path().join("worker");
    let mut kids = Kids(Vec::new());
    let a = fake_live(&root, &key, "20260101T000001Z", &mut kids);
    let b = fake_live(&root, &key, "20260101T000002Z", &mut kids);
    write_marker(dir.path(), &key, "20260101T000001Z");
    write_marker(dir.path(), &key, "20260101T000002Z");

    let err_path = dir.path().join("refuse.err");
    let out = command_settings(dir.path(), probe, "expect_epochs = 1", &poll_env())
        .stderr(Stdio::from(fs::File::create(&err_path).unwrap()))
        .output()
        .unwrap();
    let err = err_text(&err_path);
    assert!(!out.status.success(), "{err}");
    assert!(
        err.contains("refusing to choose between 2 live attempts"),
        "{err}"
    );
    assert!(
        err.contains("checkpoints.py") && err.contains("abandon"),
        "{err}"
    );
    assert!(err.contains("nothing was chosen or deleted"), "{err}");
    assert!(a.is_dir(), "the refusing run deleted {}", a.display());
    assert!(b.is_dir(), "the refusing run deleted {}", b.display());
    assert!(a.join("stage.pid").is_file() && b.join("stage.pid").is_file());
}

/// Row 7: a live attempt wins over a newer unverified marker, because the
/// marker only names a dir whose attempt is already dead.
#[test]
fn reattach_prefers_live_over_newest_marker() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let probe = "echo \"$VI_RETURN_PORT\" > \"$VI_OUT/port\"; exit 3";
    let first = run_settings(dir.path(), probe, "expect_epochs = 1", &poll_env());
    assert!(!first.status.success());
    let probe_work = wait_for_work(dir.path());
    let (key, _attempt) = split_attempt(&probe_work);
    let root = dir.path().join("worker");

    // The live attempt: a fake stage that finishes by itself, a jobs/ folder
    // for the fresh-token proof, and output for the final push.
    let live_attempt = "20260101T000003Z";
    let live = root.join(format!("on-gpu.{key}.{live_attempt}"));
    fs::create_dir_all(live.join("out")).unwrap();
    fs::write(live.join("out/ok.txt"), "live\n").unwrap();
    std::os::unix::fs::symlink(jobs_dir(), live.join("jobs")).unwrap();
    fs::write(live.join("stage.pid"), "").unwrap();
    let stage = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "sleep 5; printf '0\\n' > '{}'",
            live.join("stage.rc").display()
        ))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    fs::write(live.join("stage.pid"), stage.id().to_string()).unwrap();
    // A zombie still answers `kill -0`, so reap the fake stage the moment it
    // exits; otherwise the supervisor waits forever on a stage it believes is
    // alive, even though stage.rc is already there.
    let reaper = std::thread::spawn(move || {
        let mut stage = stage;
        let _ = stage.wait();
    });

    // The reattached run still verifies the durable checkpoint home, so give
    // the live attempt one valid epoch there.
    let durable = dir.path().join(format!(
        "home/.cache/vi/checkpoints/on-gpu/{key}-{live_attempt}"
    ));
    let epoch = durable.join("epoch-0");
    fs::create_dir_all(&epoch).unwrap();
    fs::write(epoch.join("talker.safetensors"), "w0").unwrap();
    fs::write(epoch.join("tune.json"), "{}").unwrap();
    assert!(Command::new("sh")
        .arg("-c")
        .arg("sha256sum talker.safetensors tune.json > SHA256SUMS")
        .current_dir(&epoch)
        .status()
        .unwrap()
        .success());

    // A newer marker names a dead attempt; the live one must still win.
    let dead_attempt = "20260101T000004Z";
    let dead = root.join(format!("on-gpu.{key}.{dead_attempt}"));
    fs::create_dir_all(&dead).unwrap();
    fs::write(dead.join("activity"), "x").unwrap();
    write_marker(dir.path(), &key, live_attempt);
    std::thread::sleep(Duration::from_millis(1100));
    write_marker(dir.path(), &key, dead_attempt);

    let (mut child, err_path) = start_stage(
        dir.path(),
        "prefer",
        probe,
        "expect_epochs = 1",
        &poll_env(),
    );
    wait_until("the rerun did not reattach", 30, || {
        err_text(&err_path).contains("reattaching to")
    });
    let err = err_text(&err_path);
    assert!(
        err.contains(&format!("reattaching to {}", live.display())),
        "{err}"
    );
    assert!(
        !err.contains(&format!("reattaching to {}", dead.display())),
        "{err}"
    );
    let status = wait_stage(&mut child, 120);
    let err = err_text(&err_path);
    assert!(status.success(), "{err}");
    let got = dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").unwrap();
    assert_eq!(fs::read_to_string(got.join("ok.txt")).unwrap(), "live\n");
    assert!(dead.is_dir(), "the run touched the dir a marker named");
    let _ = reaper.join();
}

/// Row 18/40: a same-size, same-mtime byte flip survives rsync's quick check,
/// so only the checksum repair push can fix it. The home supervisor notices
/// the failed verify, asks for a re-send, acks the repaired epoch, and keeps
/// the corrupt copy under `.rsync-replaced-1/`.
#[test]
fn corrupt_same_size_epoch_is_repaired_and_acked() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let job = r#"set -e
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
p=voicetune-fresh/publish
mkdir -p "$p/epoch-0"
printf 'w0' > "$p/epoch-0/talker.safetensors"
printf '{}' > "$p/epoch-0/tune.json"
jobs/worker-return.sh push "$p" vi-publish
while [ ! -e "$VI_OUT/release" ]; do sleep 0.1; done
( cd "$p/epoch-0" && sha256sum talker.safetensors tune.json > SHA256SUMS )
jobs/worker-return.sh push "$p" vi-publish
echo done > "$VI_OUT/done.txt""#;
    let mut envs = poll_env();
    envs.push(("VI_PUSH_SECONDS", "1".to_string()));
    let (mut child, err_path) = start_stage(dir.path(), "repair", job, "expect_epochs = 1", &envs);
    let work = wait_for_work(dir.path());
    wait_until("the epoch never arrived home", 60, || {
        maybe_durable(dir.path())
            .and_then(|d| fs::read_to_string(d.join("epoch-0/talker.safetensors")).ok())
            .is_some_and(|s| s == "w0")
    });
    let durable = maybe_durable(dir.path()).unwrap();
    let target = durable.join("epoch-0/talker.safetensors");
    let keep = dir.path().join("original-copy");
    fs::copy(&target, &keep).unwrap();
    assert!(Command::new("touch")
        .arg("-r")
        .arg(&target)
        .arg(&keep)
        .status()
        .unwrap()
        .success());
    let mut bytes = fs::read(&target).unwrap();
    bytes[0] ^= 0xff;
    fs::write(&target, &bytes).unwrap();
    assert_eq!(fs::metadata(&target).unwrap().len(), 2);
    assert!(Command::new("touch")
        .arg("-r")
        .arg(&keep)
        .arg(&target)
        .status()
        .unwrap()
        .success());
    fs::write(work.join("out/release"), "").unwrap();

    let status = wait_stage(&mut child, 180);
    let err = err_text(&err_path);
    assert!(status.success(), "{err}");
    assert!(
        err.contains("epoch 0 failed verification; asked the worker to re-send it"),
        "{err}"
    );
    assert!(err.contains("epoch 0 re-sent"), "{err}");
    assert_eq!(
        err.matches("epoch 0 verified home and acked").count(),
        1,
        "{err}"
    );
    assert_eq!(fs::read_to_string(&target).unwrap(), "w0");
    assert_eq!(
        fs::read_to_string(durable.join("epoch-0/tune.json")).unwrap(),
        "{}"
    );
    assert!(durable.join("VERIFIED").is_file());
    let files = tree_files(&durable);
    let backup = files
        .iter()
        .find(|(p, _)| {
            p.starts_with(".rsync-replaced-1/") && p.ends_with("epoch-0/talker.safetensors")
        })
        .unwrap_or_else(|| {
            panic!(
                "no corrupt copy under .rsync-replaced-1: {:?}",
                files.iter().map(|(p, _)| p).collect::<Vec<_>>()
            )
        });
    assert_eq!(backup.1, bytes, "the replaced copy is not the corrupt one");
}

/// Row 21 (home supervisor side): while home is below its disk floor the
/// supervisor pauses acks and pushes, but the stage keeps running and the run
/// completes once space returns. A fake python3 fails only the gate, so the
/// pre-flight gate passes and the floor can drop mid-run.
#[test]
fn hs_pauses_acks_below_disk_floor_and_resumes() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("home")).unwrap();
    let gate_fail = dir.path().join("gate-fail");
    let real_python = Command::new("python3")
        .args(["-c", "import sys; print(sys.executable)"])
        .output()
        .unwrap();
    assert!(real_python.status.success());
    let real_python = String::from_utf8_lossy(&real_python.stdout)
        .trim()
        .to_string();
    let bin = dir.path().join("bin");
    fs::create_dir(&bin).unwrap();
    let py = bin.join("python3");
    fs::write(
        &py,
        format!(
            "#!/bin/sh\n\
             script= cmd=\n\
             for a in \"$@\"; do\n\
             case $a in */checkpoints.py) script=1;; gate) cmd=1;; esac\n\
             done\n\
             [ \"$script$cmd\" = 11 ] && [ -e {gate} ] && exit 1\n\
             exec {real} \"$@\"\n",
            gate = gate_fail.display(),
            real = real_python,
        ),
    )
    .unwrap();
    fs::set_permissions(&py, fs::Permissions::from_mode(0o755)).unwrap();

    let job = format!(
        r#"set -e
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
: > {gate}
p=voicetune-fresh/publish
mkdir -p "$p/epoch-0"
printf 'w0' > "$p/epoch-0/talker.safetensors"
printf '{{}}' > "$p/epoch-0/tune.json"
( cd "$p/epoch-0" && sha256sum talker.safetensors tune.json > SHA256SUMS )
jobs/worker-return.sh push "$p" vi-publish
echo done > "$VI_OUT/stage-done.txt""#,
        gate = gate_fail.display(),
    );
    let mut envs = poll_env();
    envs.push(("VI_PUSH_SECONDS", "1".to_string()));
    envs.push((
        "PATH",
        format!(
            "{}:{}",
            bin.display(),
            std::env::var("PATH").unwrap_or_default()
        ),
    ));
    let (mut child, err_path) = start_stage(
        dir.path(),
        "disk-floor",
        &job,
        "expect_epochs = 1, min_home_gb = 0",
        &envs,
    );
    let work = wait_for_work(dir.path());
    wait_until(
        "the stage never finished while the floor was low",
        60,
        || work.join("out/stage-done.txt").is_file(),
    );
    wait_until("the supervisor never reported the paused gate", 20, || {
        err_text(&err_path).contains("ALERT: home disk is below the floor; training continues")
    });
    assert!(
        child.try_wait().unwrap().is_none(),
        "the run ended while acks were paused"
    );
    assert!(
        !work.join("acked/epoch-0").exists(),
        "the supervisor acked an epoch below the disk floor"
    );

    fs::remove_file(&gate_fail).unwrap();
    let status = wait_stage(&mut child, 60);
    let err = err_text(&err_path);
    assert!(status.success(), "{err}");
    assert!(
        err.contains("home disk is above the floor again; resuming acks and pushes"),
        "{err}"
    );
    assert!(err.contains("epoch 0 verified home and acked"), "{err}");
    let durable = maybe_durable(dir.path()).expect("no durable dir after the run");
    assert!(durable.join("VERIFIED").is_file(), "{err}");
    assert_eq!(
        fs::read_to_string(durable.join("epoch-0/talker.safetensors")).unwrap(),
        "w0"
    );
}

/// Row 3: a worker that cannot be polled is retried; only after
/// VI_LOST_MINUTES does the supervisor give up, and it keeps the worker copy
/// so a reattach is still possible.
#[test]
fn poll_failures_retry_then_give_up_keeping_worker_copy() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let job = r#"echo "$VI_RETURN_PORT" > "$VI_OUT/port"
while [ ! -e "$VI_OUT/release" ]; do sleep 0.1; done
echo done > "$VI_OUT/done.txt""#;
    let mut envs = poll_env();
    envs.push(("VI_LOST_MINUTES", "1".to_string()));
    let (mut child, err_path) = start_stage(dir.path(), "giveup", job, "judge = \"gemma\"", &envs);
    let work = wait_for_work(dir.path());
    // Make the worker side of every poll impossible: the supervisor's `cd`
    // into the work dir fails, so no status block is printed.
    fs::set_permissions(&work, fs::Permissions::from_mode(0o000)).unwrap();
    wait_until("the supervisor never gave up", 100, || {
        err_text(&err_path).contains("giving up after 1 min")
    });
    fs::set_permissions(&work, fs::Permissions::from_mode(0o755)).unwrap();
    let status = wait_stage(&mut child, 30);
    let err = err_text(&err_path);
    assert!(!status.success(), "{err}");
    assert!(
        err.matches("poll failed; will retry").count() >= 2,
        "the supervisor did not retry a failed poll: {err}"
    );
    assert!(err.contains("keeping"), "{err}");
    assert!(work.is_dir(), "the worker copy was deleted");
    assert!(work.join("stage.pid").is_file());
    // The stage itself was never stopped and still finishes.
    fs::write(work.join("out/release"), "").unwrap();
    wait_until("the detached stage did not finish", 20, || {
        work.join("stage.rc").is_file()
    });
    assert_eq!(
        fs::read_to_string(work.join("stage.rc")).unwrap().trim(),
        "0"
    );
    // Cleanup stopped the receiver and removed its state dir.
    assert_eq!(
        fs::read_dir(dir.path().join("state/return"))
            .unwrap()
            .count(),
        0
    );
}
