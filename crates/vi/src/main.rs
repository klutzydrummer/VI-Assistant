use std::path::PathBuf;
use std::process::ExitCode;

use charpack::Pack;

mod chat;

const USAGE: &str = "usage:
  vi pack check <character-folder>
  vi chat <character-folder>
  vi review <corpus-folder> [--port N]
  vi review --demo <empty-folder> [--port N]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["pack", "check", dir] => pack_check(PathBuf::from(dir)),
        ["chat", dir] => chat::run(PathBuf::from(dir)),
        ["review", rest @ ..] => review(rest),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
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

/// Serve the label review page for a corpus on localhost.
fn review(args: &[&str]) -> ExitCode {
    let mut port = 8737u16;
    let mut demo = false;
    let mut dir = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match *arg {
            "--demo" => demo = true,
            "--port" => match args.next().and_then(|p| p.parse().ok()) {
                Some(p) => port = p,
                None => {
                    eprintln!("--port needs a number");
                    return ExitCode::from(2);
                }
            },
            d if dir.is_none() => dir = Some(PathBuf::from(d)),
            _ => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };

    if demo {
        let empty = std::fs::read_dir(&dir).map_or(true, |mut d| d.next().is_none());
        if !empty {
            eprintln!(
                "error: {} is not empty; demo data needs a fresh folder",
                dir.display()
            );
            return ExitCode::FAILURE;
        }
        if let Err(e) = review::fixture::write(&dir, 120) {
            eprintln!("error: could not write demo corpus: {e}");
            return ExitCode::FAILURE;
        }
    }

    let corpus = match review::Corpus::load(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let addr = format!("127.0.0.1:{port}");
    println!(
        "{} clips, {} labelled, {} reviewed so far",
        corpus.lines.len(),
        corpus.labels.len(),
        corpus.latest().len()
    );
    println!("review at http://{addr}/  (Ctrl+C to stop; every verdict is saved as you go)");
    match review::server::serve(corpus, &addr) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
