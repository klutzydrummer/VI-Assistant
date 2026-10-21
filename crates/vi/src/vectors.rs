//! `vi vectors`: build the dialogue model's control vectors from her own
//! lines (docs/design/steering-builder-design.md).
//!
//! - `fetch <model-id>` downloads a registry model and checks its SHA-256.
//! - `preflight <model-id>` plants vectors and tests prefix reuse on the real
//!   model, and with `--character` checks the persona is `vi chat`'s.
//! - `capture <model-id>` decodes every line once into a summary store
//!   (resumable).
//! - `fit` makes the vectors and the report from a store, on the CPU.
//! - `build <model-id>` is preflight, capture and fit in one.
//! - `check <model-id>` runs the chat check's trials on `vi chat`'s
//!   steering path.
//! - `layers <method-folder> <scheme>` sets the layer scheme the chat check
//!   settled (single layer or telescoping band).
//! - `compare <a> <b>` compares two captures of the same lines (say with
//!   and without persona reuse), or two fits' vectors.
//! - `selftest` runs all of it on a tiny random model.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use charpack::Pack;
use llm::capture::{self, CaptureOptions, CaptureRun, Capturer, LineMessages};
use llm::fit::{self, FitOptions, LabelRow, Pair};
use llm::llama::{Llama, ModelConfig};
use llm::registry::{self, ModelEntry};
use llm::store::{Store, Summary};
use llm::vectors::Scheme;

pub const USAGE: &str = "  vi vectors fetch <model-id>
  vi vectors preflight <model-id> --messages <messages.jsonl> [--character <folder>] [--no-think]
      [--labels <l.jsonl> --pairs <p.jsonl> [--check-dir <folder>]]
  vi vectors capture <model-id> --messages <messages.jsonl> --store <folder> [--skip-preflight] [--no-reuse] [--no-think]
      [--labels <l.jsonl> --pairs <p.jsonl>] (needed to reuse the persona after the preflight)
  vi vectors fit --store <folder> --labels <labels.jsonl> --pairs <pairs.jsonl> --out <folder> [--head]
  vi vectors build <model-id> --messages <m.jsonl> --labels <l.jsonl> --pairs <p.jsonl> [--store <folder>] [--out <folder>]
      [--no-think] [--skip-preflight] [--no-reuse]
  vi vectors check <model-id> --vectors <method-folder> --plan <plan.jsonl> --results <results.jsonl>
      [--thinking on|off] [--steer reply|all]
  vi vectors layers <method-folder> <single:L | band:A-B | band-k:A-B | band-sqrtk:A-B> [--only a,b]
  vi vectors compare <store-a> <store-b> | <method-folder-a> <method-folder-b>
  vi vectors selftest [--dir <folder>]
  (model commands also take --models <folder> (default $VI_MODELS or models/), --registry <models.toml>,
   --gpu-layers N, --threads N; fit and build take --mem-mb N; the fit-level reuse check takes
   --check-sets a,b (default: the first sets, to 100+ lines) and --check-layers A-B (default: 20-80%
   of the model's depth))";

struct Args {
    pos: Vec<String>,
    opts: BTreeMap<String, String>,
    flags: Vec<String>,
}

const FLAGS: [&str; 4] = ["--head", "--skip-preflight", "--no-reuse", "--no-think"];

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        pos: Vec::new(),
        opts: BTreeMap::new(),
        flags: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(x) = it.next() {
        if FLAGS.contains(&x.as_str()) {
            a.flags.push(x.clone());
        } else if let Some(name) = x.strip_prefix("--") {
            let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
            a.opts.insert(name.to_string(), v.clone());
        } else {
            a.pos.push(x.clone());
        }
    }
    Ok(a)
}

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.opts.get(k).map(String::as_str)
    }

    fn need(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("needs --{k}"))
    }

    fn path(&self, k: &str) -> Result<PathBuf, String> {
        self.need(k).map(PathBuf::from)
    }

    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        self.get(k)
            .map(|v| v.parse().map_err(|_| format!("--{k} {v}?")))
            .transpose()
    }

    fn flag(&self, f: &str) -> bool {
        self.flags.iter().any(|x| x == f)
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let result = parse(args).and_then(|a| {
        let cmd = a.pos.first().map(String::as_str).unwrap_or("");
        match cmd {
            "fetch" => fetch_cmd(&a),
            "preflight" => preflight_cmd(&a),
            "capture" => capture_cmd(&a),
            "fit" => fit_cmd(&a),
            "build" => build_cmd(&a),
            "check" => check_cmd(&a),
            "layers" => layers_cmd(&a),
            "compare" => compare_cmd(&a),
            "selftest" => selftest_cmd(&a),
            _ => Err(format!("usage:\n{USAGE}")),
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ------------------------------------------------------------------ models

struct Model {
    id: String,
    entry: ModelEntry,
    path: PathBuf,
}

fn models_dir(a: &Args) -> PathBuf {
    a.get("models")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("VI_MODELS").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models"))
}

fn model(a: &Args) -> Result<Model, String> {
    let id = a
        .pos
        .get(1)
        .ok_or("needs a model id from models.toml")?
        .clone();
    let reg = a.get("registry").unwrap_or(registry::REGISTRY);
    let models = registry::load(Path::new(reg))?;
    let entry = models.get(&id).cloned().ok_or_else(|| {
        format!(
            "{id} isn't in {reg}; it has: {}",
            models.keys().cloned().collect::<Vec<_>>().join(", ")
        )
    })?;
    let path = entry.local_path(&models_dir(a));
    Ok(Model { id, entry, path })
}

fn config(a: &Args, m: &Model) -> Result<ModelConfig, String> {
    if !m.path.exists() {
        return Err(format!(
            "{} isn't there yet; run: vi vectors fetch {}",
            m.path.display(),
            m.id
        ));
    }
    let mut c = ModelConfig::new(&m.path);
    if let Some(n) = a.num("gpu-layers")? {
        c.n_gpu_layers = n;
    }
    if let Some(n) = a.num("threads")? {
        c.n_threads = Some(n);
    }
    Ok(c)
}

/// The model's SHA-256, checked against the registry when it has one.
fn model_sha(m: &Model) -> Result<String, String> {
    println!(
        "Hashing {} (cached after the first time)...",
        m.path.display()
    );
    let sha = registry::sha256_cached(&m.path)?;
    if let Some(want) = &m.entry.sha256 {
        if want != &sha {
            return Err(format!(
                "{} has SHA-256 {sha}, the registry says {want}",
                m.path.display()
            ));
        }
    }
    Ok(sha)
}

fn fetch_cmd(a: &Args) -> Result<(), String> {
    let m = model(a)?;
    fetch(&m)?;
    model_sha(&m)?;
    println!("{} is ready at {}", m.id, m.path.display());
    Ok(())
}

/// Download the model file if it's missing, resuming a partial download.
fn fetch(m: &Model) -> Result<(), String> {
    if m.path.exists() {
        return Ok(());
    }
    let url = m
        .entry
        .url()
        .ok_or_else(|| format!("{} is a local file and isn't there", m.path.display()))?;
    if let Some(dir) = m.path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let part = m.path.with_extension("gguf.part");
    let have = fs::metadata(&part).map(|md| md.len()).unwrap_or(0);
    println!("Downloading {url}");
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
    let mut req = agent.get(&url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| format!("{url}: {e}"))?;
    let resumed = resp.status() == 206;
    let total: Option<u64> = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .map(|n: u64| n + if resumed { have } else { 0 });
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(resumed)
        .write(true)
        .truncate(!resumed)
        .open(&part)
        .map_err(|e| format!("{}: {e}", part.display()))?;
    let mut got = if resumed { have } else { 0 };
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut last = Instant::now();
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("{}: {e}", part.display()))?;
        got += n as u64;
        if last.elapsed().as_secs() >= 10 {
            last = Instant::now();
            match total {
                Some(t) => println!("  {:.1} of {:.1} GB", got as f64 / 1e9, t as f64 / 1e9),
                None => println!("  {:.1} GB", got as f64 / 1e9),
            }
        }
    }
    file.sync_all().map_err(|e| e.to_string())?;
    if total.is_some_and(|t| t != got) {
        return Err(format!(
            "download stopped at {got} bytes; run the same command again to resume"
        ));
    }
    fs::rename(&part, &m.path).map_err(|e| format!("{}: {e}", m.path.display()))?;
    Ok(())
}

// ----------------------------------------------------------------- capture

fn load_capturer(
    a: &Args,
    m: &Model,
    lines: &[LineMessages],
) -> Result<(Capturer, Vec<capture::Prepared>), String> {
    let mc = config(a, m)?;
    println!("Loading {}...", m.path.display());
    let probe = Llama::load(&mc)?;
    // Her line behind an empty think block, unless --no-think (comparison only).
    let think = !a.flag("--no-think");
    let (max_tokens, max_rest) = capture::sizes(&probe, lines, think)?;
    drop(probe);
    let mut opts = CaptureOptions::new(mc);
    opts.layer_tensor = m.entry.layer_tensor.clone();
    opts.think = think;
    let mut cap = Capturer::load(&opts, max_tokens, max_rest)?;
    let prepared = cap.prepare(lines)?;
    println!(
        "{} lines; persona {} tokens decoded once; longest prompt {max_tokens} tokens; layers {}..={}; {}",
        lines.len(),
        cap.prefix_tokens(),
        cap.layers[0],
        cap.layers[cap.layers.len() - 1],
        if think {
            "her line behind an empty think block, pooled from </think>"
        } else {
            "no think block (--no-think), pooled from the last prompt token"
        }
    );
    Ok((cap, prepared))
}

/// Compare the messages' persona with what `vi chat` builds for the
/// character at rest.
fn persona_check(character: &Path, lines: &[LineMessages]) -> Result<(), String> {
    let pack = Pack::load(character).map_err(|e| e.to_string())?;
    let machine = match pack.machine() {
        Some(Ok(m)) => m,
        Some(Err(errs)) => {
            return Err(errs
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; "))
        }
        None => return Err(format!("{} has no states.toml", character.display())),
    };
    let want = engine::prompt::system_prompt(&pack, &machine, &machine.start());
    let got = &lines[0].system;
    if *got == want {
        println!("persona: byte-equal to vi chat's system prompt at rest");
        return Ok(());
    }
    let at = got
        .bytes()
        .zip(want.bytes())
        .take_while(|(x, y)| x == y)
        .count();
    Err(format!(
        "the messages' persona differs from vi chat's system prompt at byte {at} \
         (messages {} bytes, vi chat {} bytes)",
        got.len(),
        want.len()
    ))
}

fn print_preflight(p: &capture::Preflight) {
    for c in &p.layer_test {
        println!(
            "  layer test at {}: planted shift error {:.2e}, leak into layer before {} -> {}",
            c.layer,
            c.rel_error,
            c.leak_before.map_or("-".into(), |l| format!("{l:.2e}")),
            if c.ok { "ok" } else { "FAILED" }
        );
    }
    for c in &p.copy_test {
        println!(
            "  copy test {}: persona copied against the same decodes without the copy, \
             worst relative error {:.2e} -> {}",
            c.line_id,
            c.max_rel,
            if c.ok { "ok" } else { "FAILED" }
        );
    }
    let worst = p.reuse_test.iter().map(|c| c.max_rel).fold(0.0, f64::max);
    println!(
        "  reuse against whole decodes ({} lines, reported only): worst relative error {worst:.2e}{}",
        p.reuse_test.len(),
        if p.reuse_warning {
            " (over 1e-3: rounding, judged by the fit-level check)"
        } else {
            ""
        }
    );
    for d in &p.reuse_spread {
        println!(
            "    layer {:>3}: noise against the lines' spread {:.2e}, shared by every line {:.2}",
            d.layer, d.noise_vs_spread, d.shared
        );
    }
}

fn print_fit_check(f: &capture::FitCheck) {
    println!(
        "  fit-level reuse check on {} lines of {}: gate covers layers {}-{}, worst cosine there {:.6}, worst norm change {:.2}% -> {}",
        f.lines,
        f.sets.join(", "),
        f.layers.0,
        f.layers.1,
        f.worst_cos,
        f.worst_norm * 100.0,
        if f.passed { "ok" } else { "FAILED" }
    );
    if !f.skipped.is_empty() {
        println!(
            "    {} other vectors not gated (not among the checked sets)",
            f.skipped.len()
        );
    }
}

/// The fit-level reuse gate, on the lines of the first contrast sets.
fn fit_check(
    a: &Args,
    cap: &mut Capturer,
    prepared: &[capture::Prepared],
    dir: &Path,
) -> Result<capture::FitCheck, String> {
    let (labels, _) = fit::read_jsonl::<LabelRow>(&a.path("labels")?)?;
    let (pairs, _) = fit::read_jsonl::<Pair>(&a.path("pairs")?)?;
    let sets: Option<Vec<String>> = a
        .get("check-sets")
        .map(|s| s.split(',').map(str::to_string).collect());
    let (sets, lines) = capture::fit_check_lines(prepared, &pairs, sets.as_deref());
    if lines.len() < 2 {
        return Err("the fit-level reuse check found no lines in the contrast sets".into());
    }
    let range = match a.get("check-layers") {
        Some(r) => {
            let (x, y) = r
                .split_once('-')
                .ok_or_else(|| format!("--check-layers {r}? (like 6-18)"))?;
            (
                x.parse().map_err(|_| format!("--check-layers {r}?"))?,
                y.parse().map_err(|_| format!("--check-layers {r}?"))?,
            )
        }
        None => capture::default_check_range(&cap.layers, cap.llama().n_layer()),
    };
    let f = capture::reuse_fit_check(cap, &lines, sets, &labels, &pairs, range, dir, |s| {
        println!("  fit-level reuse check: capturing {s}")
    })?;
    print_fit_check(&f);
    Ok(f)
}

fn preflight_cmd(a: &Args) -> Result<(), String> {
    let m = model(a)?;
    let lines = capture::read_messages(&a.path("messages")?)?;
    if let Some(c) = a.get("character") {
        persona_check(Path::new(c), &lines)?;
    }
    let (mut cap, prepared) = load_capturer(a, &m, &lines)?;
    let t = Instant::now();
    let p = cap.preflight(&prepared)?;
    print_preflight(&p);
    if !p.passed {
        return Err("preflight failed; nothing should be captured on this model".into());
    }
    if a.get("labels").is_some() {
        let dir = a
            .get("check-dir")
            .map_or_else(|| std::env::temp_dir().join("vi-fit-check"), PathBuf::from);
        if !fit_check(a, &mut cap, &prepared, &dir)?.passed {
            println!("preflight took {:.1}s", t.elapsed().as_secs_f64());
            return Err("reuse moves the vectors on this model: capture with --no-reuse".into());
        }
    } else {
        println!(
            "  (no --labels/--pairs: the fit-level reuse check didn't run; capture needs it to reuse)"
        );
    }
    println!("preflight took {:.1}s", t.elapsed().as_secs_f64());
    println!("preflight passed");
    Ok(())
}

fn capture_into(a: &Args, m: &Model, messages: &Path, store: &Path) -> Result<(), String> {
    let lines = capture::read_messages(messages)?;
    let messages_sha = registry::sha256_file(messages)?;
    let sha = model_sha(m)?;
    let (mut cap, prepared) = load_capturer(a, m, &lines)?;
    let reuse = !a.flag("--no-reuse");
    let pre = if a.flag("--skip-preflight") {
        None
    } else {
        let p = cap.preflight(&prepared)?;
        print_preflight(&p);
        let layers_ok = p.layer_test.iter().all(|c| c.ok);
        if !layers_ok || (!p.passed && reuse) {
            return Err(if layers_ok {
                "the copy test failed: the reused persona isn't an exact copy; the capture \
                 didn't start (--no-reuse decodes every prompt whole instead)"
                    .into()
            } else {
                "the layer test failed; the capture didn't start".into()
            });
        }
        if reuse {
            if a.get("labels").is_none() || a.get("pairs").is_none() {
                return Err(
                    "reusing the persona needs the fit-level check: pass --labels and \
                     --pairs, or --no-reuse"
                        .into(),
                );
            }
            let dir = PathBuf::from(format!("{}-fit-check", store.display()));
            let f = fit_check(a, &mut cap, &prepared, &dir)?;
            let _ = fs::write(
                dir.join("fit-check.json"),
                serde_json::to_string_pretty(&f).unwrap(),
            );
            if !f.passed {
                return Err(
                    "reuse moves the vectors on this model; the capture didn't start \
                     (--no-reuse decodes every prompt whole instead)"
                        .into(),
                );
            }
        }
        Some(p)
    };
    let run = CaptureRun {
        model_file: &m.path,
        model_sha256: &sha,
        messages_sha256: &messages_sha,
        store,
        reuse,
    };
    let t = Instant::now();
    let mut last = Instant::now();
    let mut first: Option<usize> = None;
    let meta = capture::capture_all(&mut cap, &prepared, &run, pre.as_ref(), |done, total| {
        let start = *first.get_or_insert(done - 1);
        if last.elapsed().as_secs() >= 15 || done == total {
            last = Instant::now();
            let rate = (done - start) as f64 / t.elapsed().as_secs_f64().max(1e-9);
            println!(
                "  {done}/{total} lines, {rate:.1} lines/s, about {:.0} min left",
                (total - done) as f64 / rate.max(1e-9) / 60.0
            );
        }
    })?;
    println!(
        "Captured {} lines x {} layers into {} ({:.0}s)",
        meta.n_lines_total,
        meta.layers.len(),
        store.display(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn capture_cmd(a: &Args) -> Result<(), String> {
    let m = model(a)?;
    capture_into(a, &m, &a.path("messages")?, &a.path("store")?)
}

// --------------------------------------------------------------------- fit

fn fit_into(a: &Args, store: &Path, out: &Path) -> Result<(), String> {
    let mut store = Store::open(store)?;
    let (labels, lsha) = fit::read_jsonl::<LabelRow>(&a.path("labels")?)?;
    let (pairs, psha) = fit::read_jsonl::<Pair>(&a.path("pairs")?)?;
    let mut opts = FitOptions::default();
    if let Some(n) = a.num("mem-mb")? {
        opts.mem_mb = n;
    }
    if a.flag("--head") {
        opts.summary = Summary::Head;
    }
    let t = Instant::now();
    let f = fit::fit(&mut store, &labels, &pairs, (&lsha, &psha), &opts, |s| {
        println!("  {s}")
    })?;
    let scheme = Scheme::Single(f.report.provisional_layer as u32);
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    fit::write_method(
        &out.join("joint"),
        "joint",
        &f.joint,
        scheme,
        &store.meta,
        &f.report,
    )?;
    fit::write_method(
        &out.join("mean_diff"),
        "mean_diff",
        &f.mean_diff,
        scheme,
        &store.meta,
        &f.report,
    )?;
    let write = |name: &str, text: String| -> Result<(), String> {
        let p = out.join(name);
        fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))
    };
    write(
        "report.json",
        serde_json::to_string_pretty(&f.report).unwrap(),
    )?;
    write("report.md", fit::report_markdown(&f.report))?;
    write(
        "cosines.json",
        serde_json::to_string(&serde_json::json!({"names": f.cosines.0, "layers": f.cosines.1}))
            .unwrap(),
    )?;
    println!(
        "{} joint and {} mean_diff vectors in {} ({:.0}s). Joint is the default; \
         vectors.toml starts every vector at layer {} (single layer, strength 1.0) \
         until the chat check picks the layer scheme. Report: {}",
        f.joint.len(),
        f.mean_diff.len(),
        out.display(),
        t.elapsed().as_secs_f64(),
        f.report.provisional_layer,
        out.join("report.md").display()
    );
    Ok(())
}

fn fit_cmd(a: &Args) -> Result<(), String> {
    fit_into(a, &a.path("store")?, &a.path("out")?)
}

fn build_cmd(a: &Args) -> Result<(), String> {
    let m = model(a)?;
    fetch(&m)?;
    let sha = model_sha(&m)?;
    let name = a.get("name").unwrap_or("tali");
    let base = PathBuf::from("vectors").join(name).join("llm").join(&sha);
    let store = a
        .get("store")
        .map_or_else(|| base.join("capture"), PathBuf::from);
    let out = a.get("out").map_or(base, PathBuf::from);
    capture_into(a, &m, &a.path("messages")?, &store)?;
    fit_into(a, &store, &out)
}

// ------------------------------------------------------------------- check

fn check_cmd(a: &Args) -> Result<(), String> {
    let m = model(a)?;
    let sha = model_sha(&m)?;
    let mut mc = config(a, &m)?;
    if let Some(n) = a.num("ctx")? {
        mc.n_ctx = n;
    }
    println!("Loading {}...", m.path.display());
    let llama = Llama::load(&mc)?;
    let mut defaults = llm::check::CheckDefaults::default();
    if let Some(t) = a.get("thinking") {
        defaults.thinking = llm::check::Thinking::parse(t)?;
    }
    if let Some(s) = a.get("steer") {
        defaults.scope = llm::llama::Scope::parse(s)?;
    }
    println!(
        "Thinking {}, steering {} (trials may override)",
        defaults.thinking.name(),
        defaults.scope.name()
    );
    let mut checker = llm::check::Checker::new(llama, &a.path("vectors")?, &sha, defaults)?;
    let n = llm::check::run_plan(&mut checker, &a.path("plan")?, &a.path("results")?, |r| {
        println!(
            "  {} [{}, thinking {}, steer {} from {}] {} tokens in {:.1}s",
            r.id, r.scheme, r.think, r.scope, r.steer_start, r.reply_tokens, r.seconds
        )
    })?;
    println!("{n} trials run");
    // Trials with no steered reply aren't steering results; show how often
    // that happens, so max_tokens can be raised if it's common.
    let all = llm::check::read_results(&a.path("results")?)?;
    println!("Per mood: trials, truncated (thinking ran out of tokens), never steered:");
    for (mood, c) in llm::check::summarize(&all) {
        println!(
            "  {mood}: {} trials, {} truncated, {} never steered",
            c.trials, c.truncated, c.never_steered
        );
    }
    Ok(())
}

fn layers_cmd(a: &Args) -> Result<(), String> {
    let dir = a.pos.get(1).ok_or("needs a method folder")?;
    let scheme = Scheme::parse(a.pos.get(2).ok_or("needs a scheme")?)?;
    let only: Option<Vec<String>> = a
        .get("only")
        .map(|s| s.split(',').map(str::to_string).collect());
    let done = fit::relayer(Path::new(dir), scheme, only.as_deref())?;
    println!("{} vectors now use {}", done.len(), scheme.name());
    Ok(())
}

fn compare_cmd(a: &Args) -> Result<(), String> {
    let (Some(x), Some(y)) = (a.pos.get(1), a.pos.get(2)) else {
        return Err("needs two stores or two method folders".into());
    };
    let (x, y) = (Path::new(x), Path::new(y));
    if x.join(llm::store::META).exists() {
        let diffs = llm::compare::stores(&mut Store::open(x)?, &mut Store::open(y)?)?;
        println!(
            "line_mean, per layer: relative error per line (median, worst), worst cosine; \
             the difference against the lines' spread around their mean, the part every line \
             shares (about 1/sqrt(lines) for noise), and the spread ratio without that part"
        );
        for d in &diffs {
            println!(
                "  layer {:>3}: {:.2e} / {:.2e}, cos {:.6}; vs spread {:.2e}, shared {:.2}, noise vs spread {:.2e}",
                d.layer, d.median_rel, d.worst_rel, d.min_cos, d.vs_spread, d.shared, d.noise_vs_spread
            );
        }
        let worst = diffs.iter().map(|d| d.noise_vs_spread).fold(0.0, f64::max);
        println!("worst noise vs spread: {worst:.2e}");
    } else {
        let diffs = llm::compare::vectors(x, y)?;
        for d in &diffs {
            println!(
                "  {} layer {}: cosine {:.6}, norm ratio {:.4}",
                d.vector, d.layer, d.cos, d.norm_ratio
            );
        }
        if let Some(w) = diffs.iter().min_by(|p, q| p.cos.total_cmp(&q.cos)) {
            println!(
                "lowest cosine: {} layer {} at {:.6}",
                w.vector, w.layer, w.cos
            );
        }
    }
    Ok(())
}

fn selftest_cmd(a: &Args) -> Result<(), String> {
    let dir = a.get("dir").map_or_else(
        || std::env::temp_dir().join("vi-vectors-selftest"),
        PathBuf::from,
    );
    let gpu: i32 = a.num("gpu-layers")?.unwrap_or(-1);
    let r = llm::selftest::run(&dir, gpu, |s| println!("  {s}"))?;
    print_preflight(&r.preflight);
    print_fit_check(&r.fit_check);
    println!(
        "  GPU offload {}; capture with --gpu-layers {gpu} against CPU: worst cosine {:.6}, \
         worst relative error {:.2e} -> {}",
        if r.gpu_offload_supported {
            "built in"
        } else {
            "not built in (CPU build)"
        },
        r.gpu_vs_cpu.0,
        r.gpu_vs_cpu.1,
        if r.gpu_vs_cpu_ok { "ok" } else { "FAILED" }
    );
    println!(
        "  {} lines captured, {} vectors fitted, steered reply differs: {}",
        r.lines, r.vectors, r.steered_differs
    );
    if !r.passed {
        return Err("selftest failed".into());
    }
    println!("selftest passed");
    Ok(())
}
