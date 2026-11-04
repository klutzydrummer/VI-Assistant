//! `jobs/on-worker.sh` through `vi dataset run`, with the "worker" in a local
//! folder (VI_WORKER_HOST=local) so no GPU or rented machine is needed.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const VI: &str = env!("CARGO_BIN_EXE_vi");

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

/// Like `run_with`, but with a custom `settings` table for the on-gpu stage.
fn run_settings(dir: &Path, job: &str, settings: &str, envs: &[(&str, String)]) -> Output {
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
    Command::new(VI)
        .args(["dataset", "run", pipeline.to_str().unwrap()])
        .env("VI_WORKER_HOST", "local")
        .env("VI_WORKER_ROOT", dir.join("worker"))
        .env("VI_STATE_DIR", dir.join("state"))
        .env("HOME", dir.join("home"))
        .envs(envs.iter().map(|(k, v)| (*k, v.as_str())))
        .output()
        .unwrap()
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
    assert_eq!(fs::read_dir(&bundles).unwrap().count(), 1);
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());

    // A changed job reruns the stage; the unchanged input's bundle is reused.
    let out = run_with(dir.path(), &format!("{job}; true"), &envs);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("GEN already packed"), "{stderr}");
    assert_eq!(fs::read_dir(&bundles).unwrap().count(), 1);
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
