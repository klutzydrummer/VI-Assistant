//! End-to-end runs of small shell pipelines.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use dataset::runner::{Outcome, Status};
use dataset::{Pipeline, Runner};

const PIPELINE: &str = r#"
corpus = "corpus"

[stage.harvest]
command = ["sh", "-c", "mkdir -p \"$VI_OUT/audio\" && cat \"$VI_SETTINGS\" > \"$VI_OUT/audio/clip.txt\" && echo harvested"]
settings = { take = 3 }

[stage.count]
command = ["sh", "./count.sh"]
inputs = ["harvest"]
files = ["count.sh", "corpus:reviews.jsonl"]
settings = { label = "lines" }
"#;

const COUNT: &str = "wc -c < \"$VI_INPUT_HARVEST/audio/clip.txt\" > \"$VI_OUT/count.txt\"\n";

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        fs::write(root.join("pipeline.toml"), PIPELINE).unwrap();
        fs::write(root.join("count.sh"), COUNT).unwrap();
        Fixture { _dir: dir, root }
    }

    fn edit(&self, from: &str, to: &str) {
        let path = self.root.join("pipeline.toml");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains(from), "{from}");
        fs::write(&path, text.replacen(from, to, 1)).unwrap();
    }

    fn runner(&self) -> Runner {
        let pipeline = Pipeline::load(&self.root.join("pipeline.toml")).unwrap();
        Runner::new(pipeline, PathBuf::from("/nonexistent/vi")).unwrap()
    }

    fn run(&self) -> Vec<(String, Outcome)> {
        self.runner()
            .run(None, false, |_| {})
            .unwrap()
            .into_iter()
            .map(|r| (r.stage, r.outcome))
            .collect()
    }

    fn current_out(&self, stage: &str) -> PathBuf {
        let runner = self.runner();
        let key = runner.current(stage).unwrap();
        runner.run_dir(stage, &key).join("out")
    }
}

fn outcomes(pairs: &[(&str, Outcome)]) -> Vec<(String, Outcome)> {
    pairs
        .iter()
        .map(|(s, o)| (s.to_string(), o.clone()))
        .collect()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

#[test]
fn runs_once_then_caches() {
    let f = Fixture::new();
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Ran), ("count", Outcome::Ran)])
    );
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Cached), ("count", Outcome::Cached)])
    );
    assert!(read(&f.current_out("harvest").join("audio/clip.txt")).contains("\"take\": 3"));
    let runner = f.runner();
    let key = runner.current("harvest").unwrap();
    assert_eq!(
        read(&runner.run_dir("harvest", &key).join("log.txt")),
        "harvested\n"
    );
    let manifest = runner
        .manifest("count", &runner.current("count").unwrap())
        .unwrap();
    assert_eq!(manifest.inputs.len(), 1);
    assert!(manifest.files.contains_key("count.sh"));
    assert!(manifest.outputs.contains_key("count.txt"));
}

#[test]
fn changes_rerun_only_what_they_affect() {
    let f = Fixture::new();
    f.run();

    // A downstream setting reruns only that stage.
    f.edit("label = \"lines\"", "label = \"bytes\"");
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Cached), ("count", Outcome::Ran)])
    );

    // An upstream change that alters its output reruns everything after it.
    f.edit("take = 3", "take = 30");
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Ran), ("count", Outcome::Ran)])
    );

    // An upstream rerun with byte-identical output stops there.
    f.edit("[stage.harvest]\n", "[stage.harvest]\nversion = 2\n");
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Ran), ("count", Outcome::Cached)])
    );

    // Editing a listed file reruns its stage.
    fs::write(f.root.join("count.sh"), format!("# tweak\n{COUNT}")).unwrap();
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Cached), ("count", Outcome::Ran)])
    );

    // Reverting a setting finds the earlier run still on disk.
    f.edit("take = 30", "take = 3");
    f.edit("version = 2\n", "");
    let results = f.run();
    assert_eq!(results[0], ("harvest".to_string(), Outcome::Cached));
}

#[test]
fn status_reports_without_running() {
    let f = Fixture::new();
    let status = f.runner().status().unwrap();
    assert!(matches!(status[0].1, Status::Stale { .. }));
    assert_eq!(
        status[1].1,
        Status::Waiting {
            on: "harvest".into()
        }
    );
    f.run();
    let status = f.runner().status().unwrap();
    assert!(status
        .iter()
        .all(|(_, s)| matches!(s, Status::Cached { .. })));
    f.edit("label = \"lines\"", "label = \"x\"");
    let status = f.runner().status().unwrap();
    assert!(matches!(status[0].1, Status::Cached { .. }));
    assert!(matches!(status[1].1, Status::Stale { .. }));
}

#[test]
fn a_target_runs_only_what_it_needs_and_force_reruns_it() {
    let f = Fixture::new();
    let results = f.runner().run(Some("harvest"), false, |_| {}).unwrap();
    assert_eq!(results.len(), 1);
    let results = f.runner().run(Some("count"), true, |_| {}).unwrap();
    let got: Vec<_> = results.into_iter().map(|r| (r.stage, r.outcome)).collect();
    assert_eq!(
        got,
        outcomes(&[("harvest", Outcome::Cached), ("count", Outcome::Ran)])
    );
}

#[test]
fn failure_keeps_the_previous_run_current() {
    let f = Fixture::new();
    f.run();
    let before = f.runner().current("count").unwrap();
    fs::write(f.root.join("count.sh"), "echo broken >&2\nexit 3\n").unwrap();
    let err = f.runner().run(None, false, |_| {}).unwrap_err().to_string();
    assert!(err.contains("stage count failed"), "{err}");
    assert!(err.contains("broken"), "{err}");
    assert_eq!(f.runner().current("count").unwrap(), before);
    // The lock is released after a failure.
    fs::write(f.root.join("count.sh"), COUNT).unwrap();
    f.run();
}

#[test]
fn a_failed_run_leaves_its_checkpoint_for_the_rerun() {
    let f = Fixture::new();
    // Saves a checkpoint, then fails unless the previous run already saved one.
    let job = "test -s \"$VI_CHECKPOINT/done\" && wc -c < \"$VI_INPUT_HARVEST/audio/clip.txt\" > \"$VI_OUT/count.txt\" && exit 0\necho step1 > \"$VI_CHECKPOINT/done\"\nexit 3\n";
    fs::write(f.root.join("count.sh"), job).unwrap();
    assert!(f.runner().run(None, false, |_| {}).is_err());
    let stage = f.runner().stage_dir("count");
    let saved: Vec<_> = fs::read_dir(&stage)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(".checkpoint-"))
        .collect();
    assert_eq!(saved.len(), 1, "{saved:?}");
    assert_eq!(read(&stage.join(&saved[0]).join("done")), "step1\n");

    // Same key, so the rerun sees the checkpoint, and it's gone once it works.
    f.run();
    assert!(f.current_out("count").join("count.txt").is_file());
    assert!(!stage.join(&saved[0]).exists());
}

#[test]
fn identical_outputs_share_one_object_and_gc_frees_old_runs() {
    let f = Fixture::new();
    f.run();
    let clip = f.current_out("harvest").join("audio/clip.txt");
    let meta = fs::metadata(&clip).unwrap();
    assert!(meta.permissions().readonly());
    assert_eq!(meta.nlink(), 2, "output and its object");

    f.edit("[stage.harvest]\n", "[stage.harvest]\nversion = 2\n");
    f.run();
    let clip2 = f.current_out("harvest").join("audio/clip.txt");
    assert_ne!(clip, clip2);
    assert_eq!(fs::metadata(&clip2).unwrap().ino(), meta.ino());

    f.edit("take = 3", "take = 4");
    f.run();
    let report = f.runner().gc().unwrap();
    assert_eq!(report.runs_removed, 3, "{report:?}");
    // Only the old clip goes: both counts were "15", so they shared an object.
    assert_eq!(report.objects_removed, 1, "{report:?}");
    assert_eq!(
        f.run().iter().filter(|(_, o)| *o == Outcome::Ran).count(),
        0
    );
    let objects = fs::read_dir(f.root.join("corpus/objects")).unwrap().count();
    assert_eq!(objects, 2, "one clip and one count");
}

#[test]
fn missing_listed_file_is_an_error() {
    let f = Fixture::new();
    fs::remove_file(f.root.join("count.sh")).unwrap();
    let err = f.runner().run(None, false, |_| {}).unwrap_err().to_string();
    assert!(err.contains("does not exist"), "{err}");
}

#[test]
fn a_corpus_file_appearing_reruns_its_stage() {
    let f = Fixture::new();
    f.run();
    fs::write(f.root.join("corpus/reviews.jsonl"), "{}\n").unwrap();
    assert_eq!(
        f.run(),
        outcomes(&[("harvest", Outcome::Cached), ("count", Outcome::Ran)])
    );
    assert_eq!(
        dataset::runner::current_out(&f.root.join("corpus"), "count"),
        Some(f.current_out("count"))
    );
}
