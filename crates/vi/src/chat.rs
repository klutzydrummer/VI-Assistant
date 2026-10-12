//! `vi chat`: talk to a character through the full turn pipeline. Until
//! the models are connected, the dialogue is a placeholder and the voice
//! prints each sentence with its delivery, so you can watch events, state
//! and control vector strengths move.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use charpack::Pack;
use engine::director::RulesDirector;
use engine::pipeline::Session;
use engine::prompt::mood_line;
use engine::stub::{PrintVoice, ScriptedDialogue};

const HELP: &str = "Type to talk. Commands:
  /event <id>   apply an event directly, e.g. /event drink
  /state <id>   force a state, e.g. /state sober
  /status       show state, variables and vector strengths
  /quit         leave";

pub fn run(dir: PathBuf) -> ExitCode {
    let pack = match Pack::load(&dir) {
        Ok(pack) => pack,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let machine = match pack.machine() {
        Some(Ok(machine)) => machine,
        Some(Err(errors)) => {
            for e in errors {
                eprintln!("error: {e}");
            }
            return ExitCode::FAILURE;
        }
        None => {
            eprintln!("error: {} has no states.toml", dir.display());
            return ExitCode::FAILURE;
        }
    };
    let director = RulesDirector::new(machine.spec());
    let mut dialogue = ScriptedDialogue::new(Vec::<String>::new());
    dialogue.word_delay = Duration::from_millis(40);
    let voice = PrintVoice;
    let mut session = Session::new(pack, machine);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("tokio runtime");

    println!(
        "Talking to {} (placeholder dialogue, printed voice).\n{HELP}\n",
        session.pack.manifest.short_name
    );
    let stdin = std::io::stdin();
    loop {
        print!("you> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        match line.split_once(' ').unwrap_or((line, "")) {
            ("", _) => continue,
            ("/quit", _) => break,
            ("/help", _) => println!("{HELP}"),
            ("/status", _) => status(&session),
            ("/event", id) => {
                match session.machine.apply(&mut session.snapshot, id.trim()) {
                    Ok(changes) => {
                        for c in changes {
                            println!("  state: {} -> {}", c.from, c.to);
                        }
                    }
                    Err(e) => println!("  {e}"),
                }
                status(&session);
            }
            ("/state", id) => {
                if let Err(e) = session.machine.force(&mut session.snapshot, id.trim()) {
                    println!("  {e}");
                }
                status(&session);
            }
            _ => {
                let report = rt.block_on(session.turn(line, &director, &dialogue, &voice));
                match report {
                    Ok(r) => {
                        if !r.events.is_empty() {
                            println!("  events: {}", r.events.join(", "));
                        }
                        for e in &r.rejected {
                            println!("  rejected: {e}");
                        }
                        for c in &r.changes {
                            println!("  state: {} -> {}", c.from, c.to);
                        }
                        println!("  llm vectors: {}", strengths(&r.llm_vectors));
                        status(&session);
                    }
                    Err(e) => println!("  turn failed, nothing saved: {e}"),
                }
            }
        }
    }
    ExitCode::SUCCESS
}

fn status(session: &Session) {
    let snap = &session.snapshot;
    let vars: Vec<String> = snap
        .vars
        .iter()
        .filter(|(_, v)| **v > 0.005)
        .map(|(k, v)| format!("{k} {v:.2}"))
        .collect();
    println!(
        "  now: {} ({} turns) | {}",
        snap.state,
        snap.turns_in_state,
        if vars.is_empty() {
            "all variables at rest".into()
        } else {
            vars.join(", ")
        }
    );
    if let Some(line) = mood_line(&session.machine, snap) {
        println!("  mood line: {line}");
    }
    println!(
        "  tts vectors: {}",
        strengths(&session.machine.mix(snap, "tts"))
    );
}

fn strengths(v: &std::collections::BTreeMap<String, f32>) -> String {
    let mut v: Vec<_> = v.iter().filter(|(_, w)| **w >= 0.01).collect();
    v.sort_by(|a, b| b.1.total_cmp(a.1));
    if v.is_empty() {
        return "none".into();
    }
    v.iter()
        .map(|(n, w)| format!("{n} {w:.2}"))
        .collect::<Vec<_>>()
        .join(", ")
}
