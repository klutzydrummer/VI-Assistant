//! `vi dataset` end to end, with `vi corpus synth` standing in for the
//! extractor.

use std::fs;
use std::process::Command;

const VI: &str = env!("CARGO_BIN_EXE_vi");

#[test]
fn synthetic_pipeline_runs_and_caches() {
    let dir = tempfile::tempdir().unwrap();
    let pipeline = dir.path().join("pipeline.toml");
    fs::write(
        &pipeline,
        r#"
corpus = "corpus"

[stage.harvest]
command = ["vi", "corpus", "synth", "--lines", "6"]

[stage.check]
command = ["sh", "-c", "\"$VI_BIN\" corpus check \"$VI_INPUT_HARVEST\" > \"$VI_OUT/check.txt\""]
inputs = ["harvest"]
"#,
    )
    .unwrap();

    let run = |args: &[&str]| {
        let out = Command::new(VI).args(args).output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let p = pipeline.to_str().unwrap();
    assert!(run(&["dataset", "status", p]).contains("waiting  on harvest"));
    assert_eq!(run(&["dataset", "run", p]), "2 ran, 0 cached\n");
    assert_eq!(run(&["dataset", "run", p]), "0 ran, 2 cached\n");
    assert_eq!(
        run(&["dataset", "run", p, "check", "--force"]),
        "1 ran, 1 cached\n"
    );

    let corpus = dir.path().join("corpus");
    let check = dataset::runner::current_out(&corpus, "check").unwrap();
    assert!(fs::read_to_string(check.join("check.txt"))
        .unwrap()
        .contains("6 lines"));
    let harvest = dataset::runner::current_out(&corpus, "harvest").unwrap();
    assert!(run(&["corpus", "check", harvest.to_str().unwrap()]).contains("0 problems"));
}
