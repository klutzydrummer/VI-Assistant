//! The real llama.cpp path, on a tiny random model written at test time:
//! no GPU, no downloads.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use charpack::Pack;
use engine::director::{Delivery, RulesDirector};
use engine::pipeline::{Session, Voice};
use llm::llama::{Llama, ModelConfig, Sampling, Scope, SteerStart, Stop, EMPTY_THINK};
use llm::testmodel::{self, N_EMBD, N_LAYER};
use llm::vectors::{uniform_gguf, Limits, Vector, VectorBank};
use llm::LlamaDialogue;

fn model_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vi-llm-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.gguf"));
    testmodel::write(&path, 7).unwrap();
    path
}

fn load(name: &str) -> Llama {
    load_with_ctx(name, 1024)
}

fn load_with_ctx(name: &str, n_ctx: u32) -> Llama {
    let mut config = ModelConfig::new(model_path(name));
    config.n_ctx = n_ctx;
    config.n_threads = Some(2);
    Llama::load(&config).unwrap()
}

/// A direction strong enough to flip a random model's choices.
fn direction(sign: f32) -> Vec<f32> {
    (0..N_EMBD)
        .map(|i| sign * if i % 3 == 0 { 3.0 } else { -1.5 })
        .collect()
}

fn bank() -> VectorBank {
    let mut bank = VectorBank::empty(N_EMBD, N_LAYER);
    for (name, sign) in [("drunk", 1.0), ("affectionate", -1.0)] {
        let v = Vector::from_gguf(
            &uniform_gguf(&direction(sign), N_LAYER),
            N_EMBD,
            N_LAYER,
            Limits::default(),
        )
        .unwrap();
        bank.insert(name, v).unwrap();
    }
    bank
}

fn run(llama: &mut Llama, prompt: &str, steer: Option<&[f32]>) -> llm::llama::Generation {
    llama
        .generate(prompt, steer, &Sampling::greedy(24), |_| true)
        .unwrap()
}

#[test]
fn reads_the_model_and_formats_chats() {
    let llama = load("shape");
    assert_eq!(llama.n_embd(), N_EMBD);
    assert_eq!(llama.n_layer(), N_LAYER);
    let prompt = llama
        .format_chat(&[("system", "You are Tali."), ("user", "Hi")])
        .unwrap();
    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are Tali.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn steering_changes_the_reply_and_keeps_the_prompt_cache_valid() {
    let mut llama = load("steer");
    let prompt = llama
        .format_chat(&[("system", "You are Tali."), ("user", "Have a drink!")])
        .unwrap();
    let plain = run(&mut llama, &prompt, None);
    assert_eq!(plain.reused, 0);
    assert_eq!(plain.stop, Stop::MaxTokens, "{plain:?}");

    let blend = bank().blend(&BTreeMap::from([("drunk".to_string(), 0.8)]));
    let drunk = run(&mut llama, &prompt, blend.data.as_deref());
    assert_ne!(drunk.tokens, plain.tokens);
    // Everything but the last prompt token came from the cache.
    let n_prompt = llama.tokenize(&prompt).unwrap().len();
    assert_eq!(drunk.reused, n_prompt - 1);

    // Back to unsteered, still from the cache, and identical to a fresh run.
    let again = run(&mut llama, &prompt, None);
    assert_eq!(again.reused, n_prompt - 1);
    assert_eq!(again.tokens, plain.tokens);
    let fresh = run(&mut load("steer-fresh"), &prompt, None);
    assert_eq!(fresh.tokens, plain.tokens);

    // Same blend, same reply.
    let drunk_again = run(&mut llama, &prompt, blend.data.as_deref());
    assert_eq!(drunk_again.tokens, drunk.tokens);
}

#[test]
fn reply_scope_steers_from_think_end_on() {
    let mut llama = load("scope");
    let think_end = llama.think_end().unwrap();
    let chat = llama
        .format_chat(&[("system", "You are Tali."), ("user", "Have a drink!")])
        .unwrap();
    let blend = bank().blend(&BTreeMap::from([("drunk".to_string(), 0.8)]));
    let steer = blend.data.as_deref();
    let greedy = Sampling::greedy(24);
    let plain = run(&mut llama, &chat, None);

    // Thinking on: until `</think>` comes, nothing is steered.
    let open = llama
        .generate_scoped(&chat, steer, Scope::Reply, &greedy, |_| true)
        .unwrap();
    assert!(
        !open.tokens.contains(&think_end),
        "the random model said </think>"
    );
    assert_eq!(open.steer_start, SteerStart::Never);
    assert_eq!(open.tokens, plain.tokens);

    // When it comes, steering starts at it: the same reply up to and
    // including it, then a steered one. The random model never says
    // `</think>`, so a token it does say stands in for it.
    let k = 5;
    let stand_in = plain.tokens[k];
    let k = plain.tokens.iter().position(|&t| t == stand_in).unwrap();
    let switched = llama
        .generate_switching(&chat, steer, Some(stand_in), false, &greedy, &mut |_| true)
        .unwrap();
    assert_eq!(switched.steer_start, SteerStart::Reply(k));
    assert_eq!(switched.tokens[..=k], plain.tokens[..=k]);
    assert_ne!(switched.tokens, plain.tokens);

    // Thinking off: the prompt closes the think block, and steering starts
    // at its `</think>`.
    let closed = format!("{chat}{EMPTY_THINK}");
    let at = llama.tokenize(&closed).unwrap();
    let pos = at.iter().rposition(|&t| t == think_end).unwrap();
    let plain_closed = run(&mut llama, &closed, None);
    let reply = llama
        .generate_scoped(&closed, steer, Scope::Reply, &greedy, |_| true)
        .unwrap();
    assert_eq!(reply.steer_start, SteerStart::Prompt(pos));
    assert_ne!(reply.tokens, plain_closed.tokens);
    let all = llama
        .generate_scoped(&closed, steer, Scope::All, &greedy, |_| true)
        .unwrap();
    assert_eq!(all.steer_start, SteerStart::Prompt(at.len() - 1));

    // The cache still holds only unsteered tokens.
    assert_eq!(run(&mut llama, &closed, None).tokens, plain_closed.tokens);
    assert_eq!(run(&mut llama, &chat, None).tokens, plain.tokens);
}

#[test]
fn a_longer_conversation_reuses_its_prefix() {
    let mut llama = load("prefix");
    let first = llama
        .format_chat(&[("system", "You are Tali."), ("user", "Hi")])
        .unwrap();
    run(&mut llama, &first, None);
    let second = llama
        .format_chat(&[
            ("system", "You are Tali."),
            ("user", "Hi"),
            ("assistant", "Hello."),
            ("user", "How's the ship?"),
        ])
        .unwrap();
    let g = run(&mut llama, &second, None);
    let n_system = llama
        .tokenize("<|im_start|>system\nYou are Tali.<|im_end|>\n")
        .unwrap()
        .len();
    assert!(g.reused >= n_system, "reused {} of {n_system}", g.reused);
}

#[test]
fn stops_when_the_listener_does() {
    let mut llama = load("cancel");
    let prompt = llama.format_chat(&[("user", "Talk")]).unwrap();
    let mut chunks = 0;
    let g = llama
        .generate(&prompt, None, &Sampling::greedy(50), |_| {
            chunks += 1;
            chunks < 3
        })
        .unwrap();
    assert_eq!(g.stop, Stop::Cancelled);
    assert_eq!(chunks, 3);
}

#[test]
fn rejects_a_buffer_for_another_model() {
    let mut llama = load("shape-check");
    let prompt = llama.format_chat(&[("user", "Hi")]).unwrap();
    let wrong = vec![0.0; 5];
    assert!(llama
        .generate(&prompt, Some(&wrong), &Sampling::greedy(4), |_| true)
        .is_err());
}

struct Mute;

impl Voice for Mute {
    async fn speak(&self, _: &str, _: &Delivery) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn drives_a_turn_in_the_pipeline() {
    let pack =
        Pack::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../characters/tali")).unwrap();
    let machine = pack.machine().unwrap().unwrap();
    let director = RulesDirector::new(machine.spec());
    let mut session = Session::new(pack, machine);
    // The test model's vocabulary is bytes, so Tali's persona is ~13k tokens.
    let mut dialogue = LlamaDialogue::new(load_with_ctx("pipeline", 16384), bank());
    dialogue.sampling = Sampling::greedy(16);

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let sober = rt
        .block_on(session.turn("Hello!", &director, &dialogue, &Mute))
        .unwrap();
    assert!(!sober.reply.is_empty());
    assert_eq!(dialogue.last_turn().generated_tokens, 16);
    assert!(dialogue.last_turn().applied.is_empty());

    let report = rt
        .block_on(session.turn("Have a drink. Cheers!", &director, &dialogue, &Mute))
        .unwrap();
    assert_eq!(report.events, vec!["drink"]);
    let info = dialogue.last_turn();
    assert!(info.applied["affectionate"] > 0.0, "{info:?}");
    assert_eq!(session.history.len(), 4);
}
