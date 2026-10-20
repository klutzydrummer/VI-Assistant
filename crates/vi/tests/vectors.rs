//! `vi vectors` end to end on the tiny random model: build, check, layers.

use std::fs;
use std::path::Path;
use std::process::Command;

fn vi(dir: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_vi"))
        .current_dir(dir)
        .arg("vectors")
        .args(args)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "vi vectors {args:?}:\n{text}");
    text
}

macro_rules! jsonl {
    ($path:expr, $rows:expr) => {
        fs::write(
            $path,
            $rows
                .iter()
                .map(|r| serde_json::to_string(r).unwrap() + "\n")
                .collect::<String>(),
        )
        .unwrap()
    };
}

#[test]
fn builds_checks_and_relayers_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    llm::testmodel::write_layers(&d.join("tiny.gguf"), 5, 6).unwrap();
    fs::write(
        d.join("models.toml"),
        format!("[tiny]\npath = \"{}\"\n", d.join("tiny.gguf").display()),
    )
    .unwrap();
    jsonl!(&d.join("messages.jsonl"), &llm::selftest::messages());
    jsonl!(&d.join("labels.jsonl"), &llm::selftest::labels());
    jsonl!(&d.join("pairs.jsonl"), &llm::selftest::pairs());

    let out = vi(
        d,
        &[
            "build",
            "tiny",
            "--registry",
            "models.toml",
            "--messages",
            "messages.jsonl",
            "--labels",
            "labels.jsonl",
            "--pairs",
            "pairs.jsonl",
            "--store",
            "store",
            "--out",
            "built",
            "--threads",
            "2",
        ],
    );
    assert!(
        out.contains("preflight") || out.contains("layer test"),
        "{out}"
    );
    assert!(out.contains("Captured 60 lines"), "{out}");
    for f in [
        "joint/loud.gguf",
        "joint/natural/loud.gguf",
        "joint/vectors.toml",
        "mean_diff/quiet.gguf",
        "report.json",
        "report.md",
        "cosines.json",
    ] {
        assert!(d.join("built").join(f).exists(), "{f}");
    }
    let toml = fs::read_to_string(d.join("built/joint/vectors.toml")).unwrap();
    assert!(toml.contains("layers = [3, 3]"), "{toml}");

    // Rerunning the capture resumes into the finished store.
    let again = vi(
        d,
        &[
            "capture",
            "tiny",
            "--registry",
            "models.toml",
            "--messages",
            "messages.jsonl",
            "--store",
            "store",
            "--skip-preflight",
            "--threads",
            "2",
        ],
    );
    assert!(again.contains("Captured 60 lines"), "{again}");

    let plan = [
        serde_json::json!({"id": "plain", "messages": [{"role": "user", "content": "Hi"}],
                           "max_tokens": 8}),
        serde_json::json!({"id": "loud-band", "messages": [{"role": "user", "content": "Hi"}],
                           "steer": {"loud": 1.0}, "scheme": "band:2-4", "max_tokens": 8, "thinking": false,
                           "judge": [{"role": "user", "content": "Rate: {reply}"}],
                           "judge_max_tokens": 4}),
        serde_json::json!({"id": "loud-toml", "messages": [{"role": "user", "content": "Hi"}],
                           "steer": {"loud": 1.0, "nope": 0.5}, "max_tokens": 8}),
        serde_json::json!({"id": "loud-off", "messages": [{"role": "user", "content": "Hi"}],
                           "steer": {"loud": 1.0}, "max_tokens": 8, "thinking": "off"}),
        serde_json::json!({"id": "loud-all", "messages": [{"role": "user", "content": "Hi"}],
                           "steer": {"loud": 1.0}, "max_tokens": 8, "scope": "all"}),
    ];
    jsonl!(&d.join("plan.jsonl"), &plan);
    vi(
        d,
        &[
            "check",
            "tiny",
            "--registry",
            "models.toml",
            "--vectors",
            "built/joint",
            "--plan",
            "plan.jsonl",
            "--results",
            "results.jsonl",
            "--thinking",
            "on",
            "--steer",
            "reply",
            "--threads",
            "2",
        ],
    );
    let results: Vec<serde_json::Value> = fs::read_to_string(d.join("results.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(results.len(), 5);
    // Thinking on, reply scope: the random model never closes its thinking,
    // so nothing was steered.
    assert_eq!(results[2]["think"], "on");
    assert_eq!(results[2]["scope"], "reply");
    assert_eq!(results[2]["steer_start"], "never");
    // ...and ran out of tokens doing it: no reply, not judged.
    assert_eq!(results[2]["outcome"], "truncated");
    assert_eq!(results[3]["outcome"], "reply");
    // Thinking off: the prompt's own </think> starts it.
    assert_eq!(results[3]["think"], "off");
    assert_eq!(results[3]["steer_start"], "prompt");
    assert_eq!(results[4]["scope"], "all");
    assert_eq!(results[4]["steer_start"], "prompt");
    assert_eq!(results[1]["scheme"], "band:2-4");
    assert!(results[1]["judge"].is_string());
    assert_eq!(results[2]["missing"][0], "nope");
    assert!(results[2]["applied"]["loud"].is_number());
    // A rerun skips finished trials.
    let rerun = vi(
        d,
        &[
            "check",
            "tiny",
            "--registry",
            "models.toml",
            "--vectors",
            "built/joint",
            "--plan",
            "plan.jsonl",
            "--results",
            "results.jsonl",
            "--threads",
            "2",
        ],
    );
    assert!(rerun.contains("0 trials run"), "{rerun}");

    let l = vi(d, &["layers", "built/joint", "band:2-4", "--only", "loud"]);
    assert!(l.contains("1 vectors now use band:2-4"), "{l}");
    let toml = fs::read_to_string(d.join("built/joint/vectors.toml")).unwrap();
    assert!(toml.contains("layers = [2, 4]"), "{toml}");

    // A full-decode build of the same lines, compared with the reuse one.
    let full = vi(
        d,
        &[
            "build",
            "tiny",
            "--registry",
            "models.toml",
            "--messages",
            "messages.jsonl",
            "--labels",
            "labels.jsonl",
            "--pairs",
            "pairs.jsonl",
            "--store",
            "store-full",
            "--out",
            "built-full",
            "--no-reuse",
            "--skip-preflight",
            "--threads",
            "2",
        ],
    );
    assert!(full.contains("Captured 60 lines"), "{full}");
    let c = vi(d, &["compare", "store", "store-full"]);
    assert!(
        c.contains("layer   1:") && c.contains("worst noise vs spread"),
        "{c}"
    );
    let c = vi(d, &["compare", "built/joint", "built-full/joint"]);
    assert!(c.contains("loud layer 1: cosine"), "{c}");
}
