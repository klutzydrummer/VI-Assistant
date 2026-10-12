//! `vi chat`: talk to a character through the full turn pipeline. With
//! `--model`, the dialogue comes from a GGUF model run in-process by
//! llama.cpp and steered by the control vectors in `--vectors`; without it,
//! from a placeholder. The voice prints each sentence with its delivery, so
//! you can watch events, state and control vector strengths move.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use charpack::Pack;
use engine::director::RulesDirector;
use engine::pipeline::{Dialogue, Session};
use engine::prompt::mood_line;
use engine::stub::{PrintVoice, ScriptedDialogue};
use llm::llama::{Llama, ModelConfig};
use llm::vectors::VectorBank;
use llm::LlamaDialogue;

const HELP: &str = "Type to talk. Commands:
  /event <id>   apply an event directly, e.g. /event drink
  /state <id>   force a state, e.g. /state sober
  /status       show state, variables and vector strengths
  /quit         leave";

pub struct ModelArgs {
    pub model: Option<PathBuf>,
    pub vectors: Option<PathBuf>,
    pub gpu_layers: Option<String>,
    pub ctx: Option<String>,
}

/// Something that can report what the last turn did beyond the pipeline's
/// own report.
trait Explain {
    fn explain(&self) {}
}

impl Explain for ScriptedDialogue {}

impl Explain for LlamaDialogue {
    fn explain(&self) {
        let info = self.last_turn();
        println!(
            "  steering: {} | prompt cache {} tokens, reply {} tokens",
            strengths(&info.applied),
            info.reused_tokens,
            info.generated_tokens
        );
        if !info.missing.is_empty() {
            println!("  no vector yet for: {}", info.missing.join(", "));
        }
    }
}

fn load_model(args: &ModelArgs, path: &PathBuf) -> Result<LlamaDialogue, String> {
    let mut config = ModelConfig::new(path);
    if let Some(n) = &args.gpu_layers {
        config.n_gpu_layers = n.parse().map_err(|_| format!("--gpu-layers {n}?"))?;
    }
    if let Some(n) = &args.ctx {
        config.n_ctx = n.parse().map_err(|_| format!("--ctx {n}?"))?;
    }
    println!("Loading {}...", path.display());
    let llama = Llama::load(&config)?;
    let bank = match &args.vectors {
        Some(dir) => VectorBank::load_dir(dir, llama.n_embd(), llama.n_layer())?,
        None => VectorBank::empty(llama.n_embd(), llama.n_layer()),
    };
    let names: Vec<_> = bank.names().collect();
    println!(
        "Control vectors: {}",
        if names.is_empty() {
            "none".to_string()
        } else {
            names.join(", ")
        }
    );
    Ok(LlamaDialogue::new(llama, bank))
}

pub fn run(dir: PathBuf, args: ModelArgs) -> ExitCode {
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
    let session = Session::new(pack, machine);
    match &args.model {
        Some(path) => match load_model(&args, path) {
            Ok(dialogue) => talk(session, &dialogue, "printed voice"),
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        None => {
            let mut dialogue = ScriptedDialogue::new(Vec::<String>::new());
            dialogue.word_delay = Duration::from_millis(40);
            talk(session, &dialogue, "placeholder dialogue, printed voice")
        }
    }
}

fn talk<L: Dialogue + Explain>(mut session: Session, dialogue: &L, setup: &str) -> ExitCode {
    let director = RulesDirector::new(session.machine.spec());
    let voice = PrintVoice;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("tokio runtime");

    println!(
        "Talking to {} ({setup}).\n{HELP}\n",
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
                let report = rt.block_on(session.turn(line, &director, dialogue, &voice));
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
                        dialogue.explain();
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
