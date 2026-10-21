use std::path::{Path, PathBuf};
use std::process::ExitCode;

use charpack::Pack;

mod chat;
mod cloud;
mod dataset;
mod harvest;
mod vectors;

pub(crate) const USAGE: &str = "usage:
  vi pack check <character-folder>
  vi chat <character-folder> [--model <file.gguf>] [--vectors <folder>] [--gpu-layers N] [--ctx N]
  vi dataset run <pipeline.toml> [stage] [--force]
  vi dataset status <pipeline.toml>
  vi dataset gc <pipeline.toml>
  vi corpus check <harvest-output-folder>
  vi corpus tag <harvest-output-folder> --character <character-folder>
  vi corpus synth [folder] [--lines N]
  vi harvest scan <LE-install-folder> [--open]
  vi harvest package <file.pcc>
  vi harvest tlk <file.tlk | LE1-file.pcc> [--id N]
  vi extract --character <character-folder> <LE-install-folder> [--out <folder>]
  vi review <corpus-folder> [--moods <moods.toml>] [--port N]   (the hub: every task in one place)
  vi review --demo <empty-folder> [--port N]
  vi review --labels <corpus-folder> [--port N]
  vi review --labels --demo <empty-folder> [--port N]
  vi review --purity <corpus-folder> [--port N]
  vi review --purity --demo <empty-folder> [--port N]
  vi review --gold <corpus-folder> [--moods <moods.toml>] [--port N]
  vi review --channel <corpus-folder> [--port N]
  (every review mode also takes --host ADDR; default 127.0.0.1)
  vi cloud offers
  vi cloud up [--offer ID] [--yes]
  vi cloud down
  vi cloud status
  vi cloud reap
  vi cloud ssh [command]
  vi cloud env | touch
  vi vectors <fetch|preflight|capture|fit|build|check|layers|selftest> ... (vi vectors help lists them)";

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "vectors") {
        return vectors::run(&args[1..]);
    }
    let force = take_flag(&mut args, "--force");
    let lines = take_option(&mut args, "--lines");
    let model = chat::ModelArgs {
        model: take_option(&mut args, "--model").map(PathBuf::from),
        vectors: take_option(&mut args, "--vectors").map(PathBuf::from),
        gpu_layers: take_option(&mut args, "--gpu-layers"),
        ctx: take_option(&mut args, "--ctx"),
    };
    let open = take_flag(&mut args, "--open");
    let id = take_option(&mut args, "--id");
    let character = take_option(&mut args, "--character");
    let out = take_option(&mut args, "--out");
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["pack", "check", dir] => pack_check(PathBuf::from(dir)),
        ["chat", dir] => chat::run(PathBuf::from(dir), model),
        ["dataset", "run", p] => dataset::run(Path::new(p), None, force),
        ["dataset", "run", p, stage] => dataset::run(Path::new(p), Some(stage), force),
        ["dataset", "status", p] => dataset::status(Path::new(p)),
        ["dataset", "gc", p] => dataset::gc(Path::new(p)),
        ["corpus", "check", dir] => dataset::check(Path::new(dir)),
        ["corpus", "tag", dir] => match &character {
            Some(c) => harvest::retag(Path::new(c), Path::new(dir)),
            None => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        ["corpus", "synth"] => dataset::synth(None, lines.as_deref()),
        ["corpus", "synth", dir] => dataset::synth(Some(dir), lines.as_deref()),
        ["harvest", "scan", dir] => harvest::scan(Path::new(dir), open),
        ["harvest", "package", file] => harvest::package(Path::new(file)),
        ["harvest", "tlk", file] => harvest::tlk(Path::new(file), id.as_deref()),
        ["extract", root] => match &character {
            Some(c) => harvest::extract(Path::new(c), Path::new(root), out.as_deref()),
            None => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        ["review", rest @ ..] => review(rest),
        ["cloud", rest @ ..] => cloud::run(rest),
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

/// Serve the hub (every page Brandon works in, with what needs him), or one
/// page on its own: `--labels` the label review page, `--purity` the cluster
/// purity review, `--gold` the gold labelling page, `--channel` the channel
/// marks page (radio and other effects, by ear), for a corpus. On localhost
/// unless `--host` names another address (a reverse proxy's bridge, say).
fn review(args: &[&str]) -> ExitCode {
    let mut port = 8737u16;
    let mut host = String::from("127.0.0.1");
    let mut demo = false;
    let mut purity = false;
    let mut gold = false;
    let mut channel = false;
    let mut labels = false;
    let mut moods = None;
    let mut dir = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match *arg {
            "--demo" => demo = true,
            "--purity" => purity = true,
            "--gold" => gold = true,
            "--channel" => channel = true,
            "--labels" => labels = true,
            "--moods" => match args.next() {
                Some(m) => moods = Some(PathBuf::from(m)),
                None => {
                    eprintln!("--moods needs a path");
                    return ExitCode::from(2);
                }
            },
            "--host" => match args.next() {
                Some(h) => host = h.to_string(),
                None => {
                    eprintln!("--host needs an address");
                    return ExitCode::from(2);
                }
            },
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
        let written = review::fixture::write(&dir, 120).and_then(|()| {
            let hub = !(purity || gold || channel || labels);
            if purity || hub {
                review::fixture::write_clusters(&dir, 4)?;
            }
            if hub {
                review::fixture::write_listening(&dir)?;
            }
            Ok(())
        });
        if let Err(e) = written {
            eprintln!("error: could not write demo corpus: {e}");
            return ExitCode::FAILURE;
        }
    }
    let addr = format!("{host}:{port}");

    if channel {
        let mut c = match review::channel::Channel::load(&dir) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!(
            "{} clips to mark, {} marked so far",
            c.clips.len(),
            c.marked()
        );
        println!(
            "channel marks at http://{addr}/  (Ctrl+C to stop; every mark goes to {} as you go)",
            dir.join(review::channel::MARKS).display()
        );
        return match review::server::serve_with(&addr, |post, url, body| {
            review::channel::handle(&mut c, post, url, body)
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    // characters/<name>/moods.toml next to corpus/<name>, unless given.
    let moods = moods.unwrap_or_else(|| {
        let name = dir.file_name().unwrap_or_default();
        dir.join("../../characters").join(name).join("moods.toml")
    });

    if !(purity || gold || channel || labels) {
        let mut hub = match review::hub::Hub::load(&dir, &moods) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let view = hub.view();
        let count = |k: &str| view[k].as_array().map_or(0, Vec::len);
        println!(
            "{} tasks need you, {} done, {} not ready",
            count("pending"),
            count("done"),
            count("not_ready")
        );
        println!("review hub at http://{addr}/  (Ctrl+C to stop; every answer is saved as you go)");
        return match review::server::serve_with(&addr, |post, url, body| {
            review::hub::handle(&mut hub, post, url, body)
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if gold {
        let mut g = match review::gold::Gold::load(&dir, &moods) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!(
            "{} lines to label, {} labelled so far ({} moods, {} knobs from {})",
            g.lines.len(),
            g.labelled(),
            g.moods.len(),
            g.knobs.len(),
            moods.display()
        );
        println!(
            "gold labels at http://{addr}/  (Ctrl+C to stop; every save goes to {} as you go)",
            dir.join(review::gold::GOLD).display()
        );
        return match review::server::serve_with(&addr, |post, url, body| {
            review::gold::handle(&mut g, post, url, body)
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if purity {
        let mut p = match review::purity::Purity::load(&dir) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!(
            "{} clusters, {} decided, {} answers so far",
            p.classes.len(),
            p.decided(),
            p.records.len()
        );
        println!(
            "purity review at http://{addr}/  (Ctrl+C to stop; every answer is saved to {} as you go)",
            dir.join(review::purity::PURITY).display()
        );
        return match review::server::serve_with(&addr, |post, url, body| {
            review::purity::handle(&mut p, post, url, body)
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let corpus = match review::Corpus::load(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
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
