//! `jobs/on-worker.sh` through `vi dataset run`, with the "worker" in a local
//! folder (VI_WORKER_HOST=local) so no GPU or rented machine is needed.

use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
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
        .args(["-HUP", &format!("-{pgid}")])
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
    run_settings(dir, job, "judge = \"gemma\"", envs)
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
    // The job records the receiver's port and waits. The test releases the
    // wait after signalling, because bash runs a trap only once the foreground
    // command returns (a real Ctrl-C also stops that command).
    let job = format!(
        r#"# {token}
echo "$VI_RETURN_PORT" > "$VI_OUT/port"
while [ ! -e "$VI_OUT/release" ]; do sleep 0.1; done"#
    );
    let envs = [("TMPDIR", tmp.display().to_string())];
    let child = command_settings(dir.path(), &job, "judge = \"gemma\"", &envs)
        .spawn()
        .unwrap();
    let work = wait_for_work(dir.path());
    let port: u16 = fs::read_to_string(work.join("out/port"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();

    // SIGINT the home runner while the job runs (row 9).
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

    fs::write(work.join("out/release"), "").unwrap();
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
