//! The whole builder on a tiny random model, end to end: preflight,
//! capture (on the GPU when there is one, and on the CPU to compare), fit,
//! write, and load the result into a steered generation. CI runs it, the
//! CUDA workflow runs it inside the worker image, and on a rental it's the
//! first thing to run (preflight step 3, GPU against CPU).

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use crate::capture::{self, CaptureOptions, CaptureRun, Capturer, LineMessages, Preflight};
use crate::fit::{self, FitOptions, LabelRow, Pair};
use crate::llama::{Llama, ModelConfig, Sampling};
use crate::store::Store;
use crate::testmodel;
use crate::vectors::{Scheme, VectorBank};

pub const LAYERS: usize = 6;
pub const LINES: usize = 60;

#[derive(Debug, Clone, Serialize)]
pub struct SelfTest {
    pub gpu_offload_supported: bool,
    pub preflight: Preflight,
    /// Worst cosine and relative error, capture on `gpu_layers` against CPU.
    pub gpu_vs_cpu: (f64, f64),
    pub gpu_vs_cpu_ok: bool,
    pub lines: usize,
    pub vectors: usize,
    pub steered_differs: bool,
    pub passed: bool,
}

/// Synthetic exchanges: one persona, a short user line, her line. Lines
/// marked "a" are loud, "b" quiet, so the fit has something to find.
pub fn messages() -> Vec<LineMessages> {
    (0..LINES)
        .map(|i| LineMessages {
            line_id: format!("t{i:03}"),
            system: "You are Tali, a quarian engineer.".into(),
            user: (i % 7 != 0).then(|| format!("Line {i}?")),
            assistant: match i % 3 {
                0 => format!("KEELAH! Engines {i} ONLINE!"),
                1 => format!("hm. quiet {i}... ok."),
                _ => format!("Diagnostics show {i} nominal."),
            },
        })
        .collect()
}

pub fn labels() -> Vec<LabelRow> {
    (0..LINES)
        .map(|i| LabelRow {
            line_id: format!("t{i:03}"),
            conversation: format!("c{}", i / 4),
            game: if i % 2 == 0 { "ME2" } else { "ME3" }.into(),
            audio_energy: Some((i % 5) as f64),
            scores: [
                ("loud".to_string(), if i % 3 == 0 { 3.0 } else { 0.0 }),
                (
                    "quiet".to_string(),
                    if i % 3 == 1 { 3.0 } else { (i % 2) as f64 },
                ),
            ]
            .into(),
            knobs: [("energy".to_string(), (i % 4) as f64 / 3.0)].into(),
            pass1_agree: Default::default(),
        })
        .collect()
}

pub fn pairs() -> Vec<Pair> {
    let ids: Vec<String> = (0..LINES).map(|i| format!("t{i:03}")).collect();
    let mut out = Vec::new();
    for (set, r) in [("loud", 0), ("quiet", 1)] {
        let pos: Vec<usize> = (0..LINES).filter(|i| i % 3 == r).collect();
        let zero: Vec<usize> = (0..LINES).filter(|i| i % 3 == 2).collect();
        for (k, &p) in pos.iter().enumerate() {
            out.push(Pair {
                set: set.into(),
                pos: ids[p].clone(),
                zero: ids[zero[k % zero.len()]].clone(),
            });
        }
    }
    out
}

fn load(model: &Path, gpu_layers: i32, lines: &[LineMessages]) -> Result<Capturer, String> {
    let mut mc = ModelConfig::new(model);
    mc.n_gpu_layers = gpu_layers;
    mc.n_threads = Some(2);
    let opts = CaptureOptions::new(mc.clone());
    let probe = Llama::load(&mc)?;
    let (max_tokens, max_rest) = capture::sizes(&probe, lines)?;
    drop(probe);
    Capturer::load(&opts, max_tokens, max_rest)
}

/// Run everything in `dir` (made if needed). `gpu_layers` as for `vi chat`.
pub fn run(dir: &Path, gpu_layers: i32, mut log: impl FnMut(&str)) -> Result<SelfTest, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let model = dir.join("tiny.gguf");
    testmodel::write_layers(&model, 11, LAYERS).map_err(|e| e.to_string())?;
    let msgs = messages();

    log("preflight: layer and reuse tests");
    let mut cap = load(&model, gpu_layers, &msgs)?;
    let prepared = cap.prepare(&msgs)?;
    let preflight = cap.preflight(&prepared)?;

    log("capture");
    let store_dir = dir.join("store");
    let _ = std::fs::remove_dir_all(&store_dir);
    let sha = crate::registry::sha256_file(&model)?;
    let run = CaptureRun {
        model_file: &model,
        model_sha256: &sha,
        messages_sha256: "selftest",
        store: &store_dir,
        reuse: true,
    };
    capture::capture_all(&mut cap, &prepared, &run, Some(&preflight), |_, _| {})?;
    // The same lines on the CPU only.
    log("capture again on the CPU, to compare");
    let reference: Vec<_> = prepared
        .iter()
        .take(5)
        .map(|p| cap.capture(p, true))
        .collect::<Result<_, _>>()?;
    drop(cap);
    let mut cpu = load(&model, 0, &msgs)?;
    let prepared_cpu = cpu.prepare(&msgs)?;
    let (mut min_cos, mut max_rel) = (1.0f64, 0.0f64);
    for (p, r) in prepared_cpu.iter().zip(&reference) {
        let c = cpu.capture(p, true)?;
        let (a, b) = capture::compare(&c.mean, &r.mean, cpu.n_embd());
        min_cos = min_cos.min(a);
        max_rel = max_rel.max(b);
    }
    drop(cpu);
    let gpu_vs_cpu_ok = min_cos >= capture::REUSE_MIN_COS && max_rel <= capture::REUSE_MAX_REL;

    log("fit");
    let mut store = Store::open(&store_dir)?;
    let fitted = fit::fit(
        &mut store,
        &labels(),
        &pairs(),
        ("selftest", "selftest"),
        &FitOptions::default(),
        |_| {},
    )?;
    let out = dir.join("vectors");
    let scheme = Scheme::Single(fitted.report.provisional_layer as u32);
    fit::write_method(
        &out.join("joint"),
        "joint",
        &fitted.joint,
        scheme,
        &store.meta,
        &fitted.report,
    )?;
    fit::write_method(
        &out.join("mean_diff"),
        "mean_diff",
        &fitted.mean_diff,
        scheme,
        &store.meta,
        &fitted.report,
    )?;

    log("steer a generation with a fitted vector");
    let mut mc = ModelConfig::new(&model);
    mc.n_gpu_layers = gpu_layers;
    mc.n_ctx = 2048;
    mc.n_threads = Some(2);
    let mut llama = Llama::load(&mc)?;
    let bank = VectorBank::load_dir(&out.join("joint"), llama.n_embd(), llama.n_layer())?;
    let prompt = llama.format_chat(&[("system", "You are Tali."), ("user", "Hi")])?;
    let plain = llama.generate(&prompt, None, &Sampling::greedy(12), |_| true)?;
    // The natural size of a random model's vector is tiny; scale it up so
    // the test sees the steering path change the reply.
    let blend = bank.blend(&BTreeMap::from([("loud".to_string(), 1.0)]));
    let data: Vec<f32> = blend
        .data
        .ok_or("the fitted vector didn't blend")?
        .iter()
        .map(|x| x * 200.0)
        .collect();
    let steered = llama.generate(&prompt, Some(&data), &Sampling::greedy(12), |_| true)?;
    let steered_differs = steered.tokens != plain.tokens;

    let passed = preflight.passed && gpu_vs_cpu_ok && steered_differs;
    Ok(SelfTest {
        gpu_offload_supported: unsafe { llama_cpp_sys_2::llama_supports_gpu_offload() },
        preflight,
        gpu_vs_cpu: (min_cos, max_rel),
        gpu_vs_cpu_ok,
        lines: store.lines.len(),
        vectors: fitted.joint.len() + fitted.mean_diff.len(),
        steered_differs,
        passed,
    })
}
