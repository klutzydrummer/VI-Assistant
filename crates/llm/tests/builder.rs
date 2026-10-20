//! The vector builder on the tiny random model: no GPU, no downloads.

use llm::capture::{self, CaptureOptions, Capturer, LineMessages};
use llm::llama::{Llama, ModelConfig, EMPTY_THINK};
use llm::testmodel;

fn model(name: &str, layers: usize) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("vi-builder-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.gguf"));
    testmodel::write_layers(&path, 3, layers).unwrap();
    path
}

fn capturer(path: &std::path::Path, tensor: &str, lines: &[LineMessages]) -> Capturer {
    capturer_think(path, tensor, lines, true)
}

fn capturer_think(
    path: &std::path::Path,
    tensor: &str,
    lines: &[LineMessages],
    think: bool,
) -> Capturer {
    let mut mc = ModelConfig::new(path);
    mc.n_threads = Some(2);
    let probe = Llama::load(&mc).unwrap();
    let (max_tokens, max_rest) = capture::sizes(&probe, lines, think).unwrap();
    let mut opts = CaptureOptions::new(mc);
    opts.layer_tensor = tensor.into();
    opts.think = think;
    Capturer::load(&opts, max_tokens, max_rest).unwrap()
}

#[test]
fn selftest_passes_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let r = llm::selftest::run(dir.path(), 0, |_| {}).unwrap();
    assert!(r.preflight.passed, "{:#?}", r.preflight);
    assert_eq!(r.preflight.layer_test.len(), 3);
    assert!(r
        .preflight
        .layer_test
        .iter()
        .all(|c| c.leak_before.is_some()));
    assert!(r.preflight.reuse_test.iter().all(|c| c.ok));
    assert!(r.gpu_vs_cpu_ok, "{:?}", r.gpu_vs_cpu);
    assert_eq!(r.lines, llm::selftest::LINES);
    assert!(r.steered_differs);
    assert!(r.passed);
}

#[test]
fn capture_prompts_are_vi_chats_prompts() {
    let path = model("prompts", 4);
    let lines = llm::selftest::messages();
    let llama = Llama::load(&ModelConfig::new(&path)).unwrap();
    let think_end = llama.think_end().unwrap();
    for think in [true, false] {
        let mut cap = capturer_think(&path, "l_out", &lines, think);
        let prepared = cap.prepare(&lines).unwrap();
        for (l, p) in lines.iter().zip(&prepared) {
            // What `vi chat` renders for the same exchange, through the same code.
            let req = capture::request(&l.system, l.user.as_deref());
            let prompt = llama.format_chat(&llm::chat_messages(&req)).unwrap();
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
            let chat = llama.tokenize(&prompt).unwrap();
            assert!(p.tokens.starts_with(&chat), "{}", l.line_id);
            let full = if think {
                format!("{prompt}{EMPTY_THINK}")
            } else {
                prompt.clone()
            };
            let tokens = llama.tokenize(&full).unwrap();
            assert_eq!(&p.tokens[..p.n_prompt], tokens.as_slice(), "{}", l.line_id);
            let line = llama.tokenize(&format!("{full}{}", l.assistant)).unwrap();
            assert_eq!(p.tokens, line);
            if think {
                // Pooling starts at `</think>`, after vi chat's prompt.
                assert_eq!(p.tokens[p.lead_in], think_end);
                assert!(p.lead_in >= chat.len() && p.lead_in < p.n_prompt - 1);
            } else {
                assert_eq!(p.lead_in, p.n_prompt - 1);
            }
        }
        // The persona is shared and comes first.
        assert!(cap.prefix_tokens() > 10);
    }
}

#[test]
fn the_think_block_changes_what_is_pooled() {
    let path = model("think", 4);
    let lines = llm::selftest::messages();
    let mut on = capturer_think(&path, "l_out", &lines, true);
    let mut off = capturer_think(&path, "l_out", &lines, false);
    let p_on = on.prepare(&lines).unwrap();
    let p_off = off.prepare(&lines).unwrap();
    let a = on.capture(&p_on[2], true).unwrap();
    let b = off.capture(&p_off[2], true).unwrap();
    assert_ne!(a.mean, b.mean);
    // Her line's log-probability covers her tokens only, either way.
    assert!(a.logprob.is_finite() && b.logprob.is_finite());
}

#[test]
fn the_capture_refuses_missing_layers_and_mixed_personas() {
    let path = model("refuse", 4);
    let lines = llm::selftest::messages();
    let mut cap = capturer(&path, "no_such_tensor", &lines);
    let prepared = cap.prepare(&lines).unwrap();
    let err = cap.capture(&prepared[0], true).unwrap_err();
    assert!(err.contains("arrived 0 times"), "{err}");

    // Layer 0 and the last layer aren't captured: l_out-1..=n_layer-2 only.
    let mut cap = capturer(&path, "l_out", &lines);
    assert_eq!(cap.layers, vec![1, 2]);
    let prepared = cap.prepare(&lines).unwrap();
    let s = cap.capture(&prepared[1], true).unwrap();
    assert_eq!(s.mean.len(), 2 * testmodel::N_EMBD);
    assert!(s.logprob < 0.0 && s.logprob.is_finite());

    let mut mixed = lines.clone();
    mixed[3].system = "Someone else.".into();
    assert!(cap
        .prepare(&mixed)
        .unwrap_err()
        .contains("different persona"));
}
