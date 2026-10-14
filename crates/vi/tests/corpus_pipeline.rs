//! The corpus pipeline (docs/design/corpus-pipeline.md) end to end on
//! synthetic clips: prep at home, then clean and embed with fake models
//! through `jobs/on-worker.sh` and a local "worker". The real models only run
//! on a rented GPU.

use std::fs;
use std::path::Path;
use std::process::Command;

const VI: &str = env!("CARGO_BIN_EXE_vi");

fn repo(rel: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
        .canonicalize()
        .unwrap()
        .display()
        .to_string()
}

/// The home stages need numpy, scipy and soundfile; the worker needs rsync.
fn have_tools() -> bool {
    let py = Command::new("python3")
        .args(["-c", "import tomllib, numpy, scipy, soundfile"])
        .output()
        .is_ok_and(|o| o.status.success());
    let rsync = Command::new("rsync").arg("--version").output().is_ok();
    if !(py && rsync) {
        eprintln!("skipping: needs python3 >= 3.11 with numpy, scipy and soundfile, and rsync");
    }
    py && rsync
}

fn lines(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn prep_clean_and_embed_with_fake_models() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let on_worker = repo("jobs/on-worker.sh");
    let prep = repo("jobs/prep.py");
    let pipeline = dir.path().join("dataset.toml");
    fs::write(
        &pipeline,
        format!(
            r#"
corpus = "corpus"

[stage.harvest]
command = ["vi", "corpus", "synth", "--lines", "24"]

[stage.prep]
command = ["python3", {prep:?}]
inputs = ["harvest"]

[stage.clean]
command = [{on_worker:?}, "python3", "jobs/clean.py"]
inputs = ["harvest", "prep"]
settings = {{ backend = "fake" }}

[stage.embed]
command = [{on_worker:?}, "python3", "jobs/embed.py"]
inputs = ["harvest", "prep"]
settings = {{ backend = "fake", checks = "fail", wavlm_layers = [6, 18] }}
"#
        ),
    )
    .unwrap();
    let out = Command::new(VI)
        .args(["dataset", "run", pipeline.to_str().unwrap()])
        .env("VI_WORKER_HOST", "local")
        .env("VI_WORKER_ROOT", dir.path().join("worker"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let corpus = dir.path().join("corpus");
    let prep = dataset::runner::current_out(&corpus, "prep").unwrap();
    let prepped = lines(&prep.join("prep.jsonl"));
    assert_eq!(prepped.len(), 24);
    for r in &prepped {
        let path = r["audio16k"]["path"].as_str().unwrap();
        assert!(prep.join(path).is_file(), "no {path}");
        assert!(r["gain_db"].is_number());
    }
    assert!(prep.join("prep.json").is_file());

    // Clean judged every prepped clip, at prep's trimmed length.
    let clean = dataset::runner::current_out(&corpus, "clean").unwrap();
    let cleaned = lines(&clean.join("clips.jsonl"));
    assert_eq!(cleaned.len(), 24);
    for (p, c) in prepped.iter().zip(&cleaned) {
        assert_eq!(p["id"], c["id"]);
        assert_eq!(p["duration_s"], c["audio"]["duration_s"]);
    }
    let report: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(clean.join("clean.json")).unwrap()).unwrap();
    assert!(report["speaker_by_length"].is_object());

    // Embed wrote every candidate for every clip, plus the scores.
    let embed = dataset::runner::current_out(&corpus, "embed").unwrap();
    for name in [
        "wavlm_L06",
        "wavlm_L18",
        "emotion2vec_plus_large",
        "emotion2vec_base",
        "bude_whisper",
    ] {
        let ids = fs::read_to_string(embed.join(format!("embeddings/{name}.ids"))).unwrap();
        assert_eq!(ids.lines().count(), 24, "{name}");
        assert!(embed.join(format!("embeddings/{name}.npy")).is_file());
    }
    assert_eq!(lines(&embed.join("scores.jsonl")).len(), 24);
    assert!(embed.join("embed.json").is_file());
}
