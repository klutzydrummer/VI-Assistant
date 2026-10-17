//! `jobs/on-worker.sh` through `vi dataset run`, with the "worker" in a local
//! folder (VI_WORKER_HOST=local) so no GPU or rented machine is needed.

use std::fs;
use std::path::Path;
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
settings = {{ judge = "gemma" }}
"#,
            on_worker()
        ),
    )
    .unwrap();
    Command::new(VI)
        .args(["dataset", "run", pipeline.to_str().unwrap()])
        .env("VI_WORKER_HOST", "local")
        .env("VI_WORKER_ROOT", dir.join("worker"))
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
fn a_failing_job_fails_the_stage_and_still_cleans_up() {
    if !have_rsync() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run(dir.path(), "echo partial > \"$VI_OUT/b.txt\"; exit 3");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("exited with 3"));
    assert!(dataset::runner::current_out(&dir.path().join("corpus"), "on-gpu").is_none());
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
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
    assert_eq!(leftovers(dir.path()), Vec::<String>::new());
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
