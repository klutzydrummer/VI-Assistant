//! The discover and label jobs end to end with fake models: synthetic clips,
//! through `jobs/on-worker.sh` with a local "worker", read back by the review
//! page's loader. The real models only run on a rented GPU.

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

/// The jobs need Python 3.11 (tomllib) and rsync; skip without them.
fn have_tools() -> bool {
    let py = Command::new("python3")
        .args(["-c", "import tomllib"])
        .output()
        .is_ok_and(|o| o.status.success());
    let rsync = Command::new("rsync").arg("--version").output().is_ok();
    if !(py && rsync) {
        eprintln!("skipping: needs python3 >= 3.11 and rsync");
    }
    py && rsync
}

#[test]
fn discover_and_label_run_on_the_worker_with_fake_models() {
    if !have_tools() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    fs::copy(
        repo("characters/tali/states.toml"),
        dir.path().join("states.toml"),
    )
    .unwrap();
    let on_worker = repo("jobs/on-worker.sh");
    let pipeline = dir.path().join("dataset.toml");
    fs::write(
        &pipeline,
        format!(
            r#"
corpus = "corpus"

[stage.harvest]
command = ["vi", "corpus", "synth", "--lines", "24"]

[stage.clean]
command = [{on_worker:?}, "python3", "jobs/clean.py"]
inputs = ["harvest"]
settings = {{ backend = "fake" }}

[stage.discover]
command = [{on_worker:?}, "--put", "states.toml", "python3", "jobs/discover.py"]
inputs = ["harvest", "clean"]
files = ["states.toml"]
settings = {{ backend = "fake", compare = 6 }}

[stage.label]
command = [{on_worker:?}, "--put", "states.toml", "python3", "jobs/label.py"]
inputs = ["harvest", "clean", "discover"]
files = ["states.toml"]
settings = {{ backend = "fake" }}
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
    let clean = dataset::runner::current_out(&corpus, "clean").unwrap();
    assert_eq!(
        fs::read_to_string(clean.join("clips.jsonl"))
            .unwrap()
            .lines()
            .count(),
        24
    );
    assert!(clean.join("clean.json").is_file());
    assert!(clean.join("embeddings/speaker.npy").is_file());
    let discover = dataset::runner::current_out(&corpus, "discover").unwrap();
    for f in [
        "captions.jsonl",
        "clusters.jsonl",
        "prosody.jsonl",
        "clusters.md",
        "discover.json",
    ] {
        assert!(discover.join(f).is_file(), "discover wrote no {f}");
    }
    let clusters = fs::read_to_string(discover.join("clusters.jsonl")).unwrap();
    assert_eq!(clusters.lines().count(), 24);
    let ids = |stage: &Path, name: &str| {
        assert!(stage.join(format!("embeddings/{name}.npy")).is_file());
        fs::read_to_string(stage.join(format!("embeddings/{name}.ids")))
            .unwrap()
            .lines()
            .count()
    };
    assert_eq!(ids(&discover, "emotion"), 24);
    assert!(ids(&discover, "text") > 0);

    let label = dataset::runner::current_out(&corpus, "label").unwrap();
    assert!(label.join("fusion.json").is_file());
    assert_eq!(ids(&label, "emotion"), 24);

    // The review page reads the result.
    let loaded = review::corpus::Corpus::load(&corpus).unwrap();
    assert_eq!(loaded.labels.len(), 24);
    let axes: Vec<_> = loaded.axes.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(
        axes,
        [
            "joy",
            "sadness",
            "anger",
            "worry",
            "playfulness",
            "intoxication"
        ]
    );
    for label in loaded.labels.values() {
        for (axis, score) in &label.scores {
            assert!((0.0..=1.0).contains(&score.value), "{axis}: {score:?}");
            assert!((0.0..=1.0).contains(&score.confidence), "{axis}: {score:?}");
            assert!(score.signals.contains_key("audio_llm"));
        }
        // No acoustic model hears intoxication.
        assert_eq!(label.scores["intoxication"].signals["audio"], None);
    }
}
