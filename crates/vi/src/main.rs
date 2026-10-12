use std::path::{Path, PathBuf};
use std::process::ExitCode;

use charpack::Pack;

mod chat;
mod dataset;

const USAGE: &str = "usage:
  vi pack check <character-folder>
  vi chat <character-folder>
  vi dataset run <pipeline.toml> [stage] [--force]
  vi dataset status <pipeline.toml>
  vi dataset gc <pipeline.toml>
  vi corpus check <harvest-output-folder>
  vi corpus synth [folder] [--lines N]";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let force = take_flag(&mut args, "--force");
    let lines = take_option(&mut args, "--lines");
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["pack", "check", dir] => pack_check(PathBuf::from(dir)),
        ["chat", dir] => chat::run(PathBuf::from(dir)),
        ["dataset", "run", p] => dataset::run(Path::new(p), None, force),
        ["dataset", "run", p, stage] => dataset::run(Path::new(p), Some(stage), force),
        ["dataset", "status", p] => dataset::status(Path::new(p)),
        ["dataset", "gc", p] => dataset::gc(Path::new(p)),
        ["corpus", "check", dir] => dataset::check(Path::new(dir)),
        ["corpus", "synth"] => dataset::synth(None, lines.as_deref()),
        ["corpus", "synth", dir] => dataset::synth(Some(dir), lines.as_deref()),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|a| a != flag);
    args.len() != before
}

fn take_option(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.remove(i);
    (i < args.len()).then(|| args.remove(i))
}

fn pack_check(dir: PathBuf) -> ExitCode {
    let pack = match Pack::load(&dir) {
        Ok(pack) => pack,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let m = &pack.manifest;
    println!("{} ({}) from {}", m.name, m.id, m.source);
    println!(
        "emotions: {}",
        pack.emotions().into_iter().collect::<Vec<_>>().join(", ")
    );

    if let Some(spec) = &pack.machine_spec {
        let ids = |v: Vec<&str>| v.join(", ");
        println!(
            "states: {}",
            ids(spec.states.iter().map(|st| st.id.as_str()).collect())
        );
        println!(
            "variables: {}",
            ids(spec.variables.iter().map(|v| v.id.as_str()).collect())
        );
        println!(
            "events: {}",
            ids(spec.events.iter().map(|e| e.id.as_str()).collect())
        );
        for target in spec
            .mixes
            .iter()
            .map(|m| m.target.as_str())
            .collect::<std::collections::BTreeSet<_>>()
        {
            let vectors: Vec<_> = spec
                .mixes
                .iter()
                .filter(|m| m.target == target)
                .map(|m| m.vector.as_str())
                .collect();
            println!("{target} vectors: {}", vectors.join(", "));
        }
    }
    if !pack.agents.is_empty() {
        let names: Vec<_> = pack.agents.iter().map(|a| a.name.as_str()).collect();
        println!("agents: {}", names.join(", "));
    }

    let report = pack.check();
    for e in &report.errors {
        println!("error: {e}");
    }
    for a in &report.missing_audio {
        println!("missing audio: {}", a.display());
    }
    if report.is_ready() {
        println!("ready");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
