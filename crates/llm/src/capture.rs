//! Stage 1 of the vector builder: decode each of her lines once, in the
//! prompt `vi chat` would build, and keep per-layer summaries of the
//! residual stream in a [`crate::store`]. See
//! `docs/design/steering-builder-design.md`.
//!
//! - Prompts are rendered with [`crate::chat_messages`] and
//!   [`Llama::format_chat`], the same code `vi chat` runs, followed by an
//!   empty think block (`<think>\n\n</think>\n\n`) so her line sits where
//!   a reply sits after thinking, and tokenized whole. Her line is the tokens after the prompt's own tokens; the
//!   capture refuses a line whose prompt doesn't tokenize identically on
//!   its own, and refuses a persona prefix that doesn't either.
//! - The persona is decoded once into sequence 0. Each line copies it into
//!   sequence 1, decodes the rest there in one micro-batch, and drops
//!   sequence 1 again.
//! - An eval callback reads `<layer_tensor>-<il>` for layers
//!   `1..=n_layer-2` and fails the run unless each arrives exactly once
//!   per decode as a 2-D F32 tensor of `n_embd` x the batch's tokens.
//! - Summaries: `line_mean` over `</think>`, the blank line after it and
//!   her line (the positions the `reply` steering scope steers),
//!   `line_head` over that lead-in and her first `head_tokens`, the mean
//!   residual norm over line_mean's positions, and her line's mean token
//!   log-probability. Without the think block (`think: false`, kept for
//!   comparison) the lead-in is the last prompt token, as the `all` scope
//!   steers.

use std::ffi::{c_void, CStr};
use std::fs;
use std::path::Path;

use engine::pipeline::{DialogueRequest, Message, Role};
use llama_cpp_sys_2 as sys;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sys::llama_token;

use crate::llama::{Llama, ModelConfig};
use crate::registry::hex;
use crate::store::{LineRecord, Meta, StoreWriter};

/// The llama.cpp this crate embeds, recorded in every store.
pub const LLAMA_CPP: &str = "llama-cpp-sys-2 0.1.159";

/// Her first tokens averaged into `line_head`.
pub const HEAD_TOKENS: usize = 8;

/// Preflight tolerances: prefix reuse against a fresh decode.
pub const REUSE_MIN_COS: f64 = 0.9999;
pub const REUSE_MAX_REL: f64 = 1e-3;
/// The copy test's limit, per line and layer.
pub const COPY_MAX_REL: f64 = 1e-3;
/// Lines in the reported full-decode comparison.
pub const REUSE_WARNING_LINES: usize = 20;

/// One line to capture: her persona, the line she answers, her line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineMessages {
    pub line_id: String,
    pub system: String,
    #[serde(default)]
    pub user: Option<String>,
    pub assistant: String,
}

pub fn read_messages(path: &Path) -> Result<Vec<LineMessages>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out: Vec<LineMessages> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let m: LineMessages =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if !seen.insert(m.line_id.clone()) {
            return Err(format!(
                "{}:{}: line {} appears twice",
                path.display(),
                i + 1,
                m.line_id
            ));
        }
        out.push(m);
    }
    if out.is_empty() {
        return Err(format!("{} has no lines", path.display()));
    }
    Ok(out)
}

pub fn sha256_text(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

#[derive(Debug, Clone)]
pub struct CaptureOptions {
    pub model: ModelConfig,
    /// `<layer_tensor>-<il>` is read per layer.
    pub layer_tensor: String,
    pub head_tokens: usize,
    /// Put her line behind an empty think block (`<think>\n\n</think>\n\n`),
    /// where a reply sits after thinking, and pool from `</think>` on: the
    /// positions the `reply` steering scope steers. Off pools from the last
    /// prompt token, as the `all` scope steers; kept for comparison.
    pub think: bool,
}

impl CaptureOptions {
    pub fn new(model: ModelConfig) -> Self {
        Self {
            model,
            layer_tensor: "l_out".into(),
            head_tokens: HEAD_TOKENS,
            think: true,
        }
    }
}

/// A line's prompt, tokenized.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub line_id: String,
    /// Prompt and her line.
    pub tokens: Vec<llama_token>,
    /// Tokens of the prompt alone; her line is `tokens[n_prompt..]`.
    pub n_prompt: usize,
    /// Where the pooled lead-in starts: `</think>`, or the last prompt token.
    pub lead_in: usize,
    pub text_sha256: String,
}

/// What one decode left behind.
#[derive(Debug, Clone, PartialEq)]
pub struct LineSummary {
    /// layers x n_embd.
    pub mean: Vec<f32>,
    pub head: Vec<f32>,
    /// Per layer, the mean L2 norm over line_mean's positions.
    pub norms: Vec<f32>,
    pub logprob: f64,
}

/// The eval callback's state. Lives in a `Box` so its address is stable
/// for llama.cpp.
struct Tap {
    active: bool,
    prefix: String,
    n_embd: usize,
    /// Layer index to store slot.
    slot_of: Vec<Option<usize>>,
    n_tokens: usize,
    mean_rows: (usize, usize),
    head_rows: (usize, usize),
    seen: Vec<u32>,
    mean: Vec<f32>,
    head: Vec<f32>,
    norms: Vec<f32>,
    scratch: Vec<f32>,
    error: Option<String>,
}

impl Tap {
    fn slot(&self, name: &str) -> Option<usize> {
        let il: usize = name.strip_prefix(&self.prefix)?.parse().ok()?;
        *self.slot_of.get(il)?
    }

    fn fail(&mut self, msg: String) {
        if self.error.is_none() {
            self.error = Some(msg);
        }
    }

    unsafe fn receive(&mut self, t: *mut sys::ggml_tensor, slot: usize, name: &str) {
        let tensor = &*t;
        self.seen[slot] += 1;
        if self.seen[slot] > 1 {
            return self.fail(format!("{name} arrived more than once in one decode"));
        }
        let dims = sys::ggml_n_dims(t);
        if dims != 2 {
            return self.fail(format!(
                "{name} is {dims}-D, not 2-D: this architecture's layer output isn't one the capture can read"
            ));
        }
        if tensor.type_ != sys::GGML_TYPE_F32 {
            return self.fail(format!("{name} is not F32"));
        }
        let (ne0, ne1) = (tensor.ne[0] as usize, tensor.ne[1] as usize);
        if ne0 != self.n_embd || ne1 != self.n_tokens {
            return self.fail(format!(
                "{name} is {ne0} x {ne1}, expected {} x {} (n_embd x tokens in the batch)",
                self.n_embd, self.n_tokens
            ));
        }
        if tensor.nb[0] != 4 || tensor.nb[1] != 4 * ne0 {
            return self.fail(format!("{name} is not stored row by row"));
        }
        let (start, end) = self.mean_rows;
        let e = self.n_embd;
        self.scratch.resize((end - start) * e, 0.0);
        sys::ggml_backend_tensor_get(
            t,
            self.scratch.as_mut_ptr().cast(),
            start * tensor.nb[1],
            (end - start) * tensor.nb[1],
        );
        let mean = &mut self.mean[slot * e..(slot + 1) * e];
        let head = &mut self.head[slot * e..(slot + 1) * e];
        let mut norm = 0.0f64;
        for (r, row) in self.scratch.chunks(e).enumerate() {
            let pos = start + r;
            let in_head = pos >= self.head_rows.0 && pos < self.head_rows.1;
            let mut sq = 0.0f64;
            for j in 0..e {
                mean[j] += row[j];
                if in_head {
                    head[j] += row[j];
                }
                sq += f64::from(row[j]) * f64::from(row[j]);
            }
            norm += sq.sqrt();
        }
        let n_mean = (end - start) as f32;
        let n_head = (self.head_rows.1 - self.head_rows.0) as f32;
        mean.iter_mut().for_each(|x| *x /= n_mean);
        head.iter_mut().for_each(|x| *x /= n_head);
        self.norms[slot] = (norm / f64::from(n_mean)) as f32;
    }
}

unsafe extern "C" fn tap_cb(t: *mut sys::ggml_tensor, ask: bool, user: *mut c_void) -> bool {
    let tap = &mut *(user as *mut Tap);
    if !tap.active {
        return !ask;
    }
    let name = CStr::from_ptr((*t).name.as_ptr()).to_string_lossy();
    let slot = tap.slot(&name);
    if ask {
        return slot.is_some();
    }
    if let Some(slot) = slot {
        let name = name.into_owned();
        tap.receive(t, slot, &name);
    }
    // Keep computing; errors are reported after the decode.
    true
}

/// Where a capture's persona comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Persona {
    /// Copied from sequence 0 with `seq_cp`.
    Reused,
    /// Decoded again into sequence 1, beside sequence 0's.
    Fresh,
    /// Decoded into sequence 1 of an empty cache, in two decodes like the
    /// reused path: the same arithmetic in the same cells, without the copy.
    Alone,
}

pub struct Capturer {
    // Dropped first: the context must go before the tap it points at.
    llama: Llama,
    tap: Box<Tap>,
    pub layers: Vec<usize>,
    n_ubatch: usize,
    head_tokens: usize,
    think: bool,
    prefix: Vec<llama_token>,
    prefix_ready: bool,
}

// SAFETY: as `Llama`; the tap is only touched during `&mut self` decodes.
unsafe impl Send for Capturer {}

impl Capturer {
    /// Load the model for capturing prompts of up to `max_tokens` tokens,
    /// whose part after the persona is at most `max_rest` tokens.
    pub fn load(opts: &CaptureOptions, max_tokens: usize, max_rest: usize) -> Result<Self, String> {
        let mut tap = Box::new(Tap {
            active: false,
            prefix: format!("{}-", opts.layer_tensor),
            n_embd: 0,
            slot_of: Vec::new(),
            n_tokens: 0,
            mean_rows: (0, 0),
            head_rows: (0, 0),
            seen: Vec::new(),
            mean: Vec::new(),
            head: Vec::new(),
            norms: Vec::new(),
            scratch: Vec::new(),
            error: None,
        });
        let n_ubatch = max_rest.max(32).next_multiple_of(32);
        let user: *mut Tap = &mut *tap;
        let mut model = opts.model.clone();
        // Room for the persona twice (the fresh decode the reuse test
        // compares against) and the longest rest.
        model.n_ctx = (2 * max_tokens + 64) as u32;
        let cpu_only = model.n_gpu_layers == 0;
        let llama = Llama::load_with(&model, |cp| {
            cp.n_seq_max = 2;
            cp.kv_unified = true;
            cp.n_batch = n_ubatch as u32;
            cp.n_ubatch = n_ubatch as u32;
            cp.cb_eval = Some(tap_cb);
            cp.cb_eval_user_data = user.cast();
            if cpu_only {
                cp.op_offload = false;
            }
        })?;
        let n_layer = llama.n_layer();
        if n_layer < 3 {
            return Err("the model needs at least three layers to capture".into());
        }
        let layers: Vec<usize> = (1..=n_layer - 2).collect();
        tap.n_embd = llama.n_embd();
        tap.slot_of = vec![None; n_layer];
        for (s, &il) in layers.iter().enumerate() {
            tap.slot_of[il] = Some(s);
        }
        Ok(Self {
            llama,
            tap,
            layers,
            n_ubatch,
            head_tokens: opts.head_tokens,
            think: opts.think,
            prefix: Vec::new(),
            prefix_ready: false,
        })
    }

    pub fn llama(&self) -> &Llama {
        &self.llama
    }

    pub fn n_embd(&self) -> usize {
        self.llama.n_embd()
    }

    pub fn row_floats(&self) -> usize {
        self.layers.len() * self.llama.n_embd()
    }

    /// Store slot of layer `il`.
    pub fn slot(&self, il: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == il)
    }

    /// Render and tokenize every line, and find the persona prefix they
    /// share. Fails on any line whose tokens don't extend its prompt's and
    /// the prefix's.
    pub fn prepare(&mut self, lines: &[LineMessages]) -> Result<Vec<Prepared>, String> {
        let (prefix_text, prepared) = prepare(&self.llama, lines, self.think)?;
        let prefix = self.llama.tokenize(&prefix_text)?;
        for p in &prepared {
            if !p.tokens.starts_with(&prefix) || p.lead_in < prefix.len() {
                return Err(format!(
                    "line {}: the persona prefix doesn't tokenize the same on its own, \
                     so it can't be decoded once and reused",
                    p.line_id
                ));
            }
            if p.tokens.len() - prefix.len() > self.n_ubatch {
                return Err(format!(
                    "line {} needs {} tokens after the persona; the capture was loaded for {}",
                    p.line_id,
                    p.tokens.len() - prefix.len(),
                    self.n_ubatch
                ));
            }
        }
        if prefix != self.prefix {
            self.prefix = prefix;
            self.prefix_ready = false;
        }
        Ok(prepared)
    }

    pub fn prefix_tokens(&self) -> usize {
        self.prefix.len()
    }

    pub fn think(&self) -> bool {
        self.think
    }

    fn decode(
        &mut self,
        tokens: &[llama_token],
        pos0: usize,
        seq: i32,
        outputs: std::ops::Range<usize>,
    ) -> Result<(), String> {
        for (c, chunk) in tokens.chunks(self.n_ubatch).enumerate() {
            let base = c * self.n_ubatch;
            unsafe {
                let mut batch = sys::llama_batch_init(chunk.len() as i32, 0, 1);
                batch.n_tokens = chunk.len() as i32;
                for (j, &tok) in chunk.iter().enumerate() {
                    *batch.token.add(j) = tok;
                    *batch.pos.add(j) = (pos0 + base + j) as i32;
                    *batch.n_seq_id.add(j) = 1;
                    *(*batch.seq_id.add(j)) = seq;
                    *batch.logits.add(j) = i8::from(outputs.contains(&(base + j)));
                }
                let rc = sys::llama_decode(self.llama.ctx(), batch);
                sys::llama_batch_free(batch);
                if rc != 0 {
                    return Err(format!("llama_decode failed ({rc})"));
                }
            }
        }
        Ok(())
    }

    fn clear_seq(&mut self, seq: i32) -> Result<(), String> {
        unsafe {
            let mem = sys::llama_get_memory(self.llama.ctx());
            if !sys::llama_memory_seq_rm(mem, seq, -1, -1) {
                return Err(format!("llama.cpp could not drop sequence {seq}"));
            }
        }
        Ok(())
    }

    /// Decode the persona once into sequence 0.
    fn ensure_prefix(&mut self) -> Result<(), String> {
        if self.prefix_ready {
            return Ok(());
        }
        self.clear_seq(0)?;
        self.clear_seq(1)?;
        let prefix = self.prefix.clone();
        self.decode(&prefix, 0, 0, 0..0)?;
        self.prefix_ready = true;
        Ok(())
    }

    /// Capture one prepared line. With `reuse`, the persona comes from
    /// sequence 0; without, it's decoded afresh (the full-decode
    /// comparison, and the fallback).
    pub fn capture(&mut self, p: &Prepared, reuse: bool) -> Result<LineSummary, String> {
        self.capture_with(
            p,
            if reuse {
                Persona::Reused
            } else {
                Persona::Fresh
            },
        )
    }

    fn capture_with(&mut self, p: &Prepared, persona: Persona) -> Result<LineSummary, String> {
        let n_prefix = self.prefix.len();
        if !p.tokens.starts_with(&self.prefix) {
            return Err(format!(
                "line {} wasn't prepared with this prefix",
                p.line_id
            ));
        }
        if persona == Persona::Alone {
            // An empty cache, so the persona lands in the cells sequence 0
            // gets when it's decoded for reuse.
            self.clear_seq(0)?;
            self.prefix_ready = false;
        } else {
            self.ensure_prefix()?;
        }
        self.clear_seq(1)?;
        match persona {
            Persona::Reused => unsafe {
                let mem = sys::llama_get_memory(self.llama.ctx());
                sys::llama_memory_seq_cp(mem, 0, 1, -1, -1);
            },
            Persona::Fresh | Persona::Alone => {
                let prefix = self.prefix.clone();
                self.decode(&prefix, 0, 1, 0..0)?;
            }
        }
        let rest = &p.tokens[n_prefix..];
        // Indexes in `rest`: the lead-in, the prompt's last token, her line.
        let lead = p.lead_in - n_prefix;
        let a = p.n_prompt - 1 - n_prefix;
        let len = rest.len();
        let rows = self.row_floats();
        {
            let tap = &mut *self.tap;
            tap.n_tokens = len;
            tap.mean_rows = (lead, len);
            tap.head_rows = (lead, (a + 1 + self.head_tokens).min(len));
            tap.seen = vec![0; self.layers.len()];
            tap.mean = vec![0.0; rows];
            tap.head = vec![0.0; rows];
            tap.norms = vec![0.0; self.layers.len()];
            tap.error = None;
            tap.active = true;
        }
        let rest = rest.to_vec();
        let result = self.decode(&rest, n_prefix, 1, a..len - 1);
        self.tap.active = false;
        result?;
        if let Some(e) = self.tap.error.take() {
            return Err(e);
        }
        if let Some(s) = self.tap.seen.iter().position(|&n| n != 1) {
            return Err(format!(
                "{}{} arrived {} times in one decode (expected once per layer {}..={})",
                self.tap.prefix,
                self.layers[s],
                self.tap.seen[s],
                self.layers[0],
                self.layers[self.layers.len() - 1]
            ));
        }
        let logprob = self.logprob(&rest, a);
        self.clear_seq(1)?;
        Ok(LineSummary {
            mean: std::mem::take(&mut self.tap.mean),
            head: std::mem::take(&mut self.tap.head),
            norms: std::mem::take(&mut self.tap.norms),
            logprob,
        })
    }

    /// Mean log-probability of her line, `rest[a+1..]`, from the logits at
    /// `a..len-1`.
    fn logprob(&self, rest: &[llama_token], a: usize) -> f64 {
        let n_vocab = unsafe { sys::llama_vocab_n_tokens(self.llama.vocab()) } as usize;
        let mut total = 0.0f64;
        for j in a..rest.len() - 1 {
            let logits = unsafe {
                std::slice::from_raw_parts(
                    sys::llama_get_logits_ith(self.llama.ctx(), j as i32),
                    n_vocab,
                )
            };
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
            let target = rest[j + 1] as usize;
            total += f64::from(logits[target] - max) - sum.ln();
        }
        total / (rest.len() - 1 - a) as f64
    }

    /// Steer layer `il` alone with `v` during later captures, or stop.
    pub fn plant(&mut self, il: Option<(usize, &[f32])>) -> Result<(), String> {
        match il {
            None => self.llama.set_steering_layers(None, 1, 1),
            Some((il, v)) => {
                let e = self.n_embd();
                let mut buf = vec![0.0f32; self.llama.steer_len()];
                buf[(il - 1) * e..il * e].copy_from_slice(v);
                self.llama
                    .set_steering_layers(Some(&buf), il as i32, il as i32)
            }
        }
    }
}

/// Render every line's prompt the way `vi chat` does (plus the empty think
/// block when `think`), tokenize it, and find the persona text all prompts
/// start with.
fn prepare(
    llama: &Llama,
    lines: &[LineMessages],
    think: bool,
) -> Result<(String, Vec<Prepared>), String> {
    let system = &lines[0].system;
    if let Some(l) = lines.iter().find(|l| &l.system != system) {
        return Err(format!(
            "line {} has a different persona (system) from line {}; \
             the capture decodes one persona and reuses it",
            l.line_id, lines[0].line_id
        ));
    }
    let render = |user: Option<&str>| -> Result<String, String> {
        llama.format_chat(&crate::chat_messages(&request(system, user)))
    };
    // The persona prefix: what every rendering starts with, whatever the
    // user says (or whether there's a user turn at all).
    let ra = render(Some("a"))?;
    let mut prefix = common_prefix(&ra, &render(Some("b"))?).to_string();
    if lines.iter().any(|l| l.user.is_none()) {
        prefix = common_prefix(&prefix, &render(None)?).to_string();
    }
    let think_end = if think {
        Some(llama.think_end()?)
    } else {
        None
    };
    let mut out = Vec::with_capacity(lines.len());
    for l in lines {
        if l.assistant.is_empty() {
            return Err(format!("line {} is empty", l.line_id));
        }
        let prompt = capture_prompt(&render(l.user.as_deref())?, think);
        let prompt_tokens = llama.tokenize(&prompt)?;
        let lead_in = match think_end {
            None => prompt_tokens.len() - 1,
            Some(t) => prompt_tokens.iter().rposition(|&p| p == t).ok_or_else(|| {
                format!(
                    "line {}: the empty think block doesn't tokenize to </think>",
                    l.line_id
                )
            })?,
        };
        let tokens = llama.tokenize(&format!("{prompt}{}", l.assistant))?;
        if !tokens.starts_with(&prompt_tokens) || tokens.len() == prompt_tokens.len() {
            return Err(format!(
                "line {}: her line merges with the prompt's last token when tokenized",
                l.line_id
            ));
        }
        out.push(Prepared {
            line_id: l.line_id.clone(),
            tokens,
            n_prompt: prompt_tokens.len(),
            lead_in,
            text_sha256: sha256_text(&l.assistant),
        });
    }
    Ok((prefix.to_string(), out))
}

/// A capture prompt: `vi chat`'s prompt for the exchange, then the empty
/// think block when `think`. Her line follows it.
pub fn capture_prompt(chat_prompt: &str, think: bool) -> String {
    if think {
        format!("{chat_prompt}{}", crate::llama::EMPTY_THINK)
    } else {
        chat_prompt.to_string()
    }
}

/// The request `vi chat` would send for this exchange.
pub fn request(system: &str, user: Option<&str>) -> DialogueRequest {
    DialogueRequest {
        system: system.to_string(),
        messages: user
            .map(|u| Message {
                role: Role::User,
                text: u.to_string(),
            })
            .into_iter()
            .collect(),
        vectors: Default::default(),
    }
}

fn common_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let mut n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    while !a.is_char_boundary(n) {
        n -= 1;
    }
    &a[..n]
}

/// Token counts the capture needs to size its context.
pub fn sizes(llama: &Llama, lines: &[LineMessages], think: bool) -> Result<(usize, usize), String> {
    let (prefix, prepared) = prepare(llama, lines, think)?;
    let n_prefix = llama.tokenize(&prefix)?.len();
    let max_tokens = prepared.iter().map(|p| p.tokens.len()).max().unwrap_or(0);
    let max_rest = prepared
        .iter()
        .map(|p| p.tokens.len().saturating_sub(n_prefix))
        .max()
        .unwrap_or(0);
    Ok((max_tokens, max_rest))
}

// ---------------------------------------------------------------- preflight

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerCheck {
    pub layer: usize,
    /// |shift at the planted layer - planted| / |planted|.
    pub rel_error: f64,
    /// |shift one layer earlier| / |residual there|, if that layer is captured.
    pub leak_before: Option<f64>,
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReuseCheck {
    pub line_id: String,
    pub min_cos: f64,
    pub max_rel: f64,
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Preflight {
    pub layer_test: Vec<LayerCheck>,
    /// The gate on reuse: the copied persona against the same two decodes
    /// without the copy. `ok` is a relative error within [`COPY_MAX_REL`].
    pub copy_test: Vec<ReuseCheck>,
    /// Reported, not a gate: each line with the persona reused against
    /// decoded whole, which also differs by rounding (quantized matmuls
    /// amplify last-bit order differences).
    pub reuse_test: Vec<ReuseCheck>,
    /// The same, per layer, against the lines' own spread.
    pub reuse_spread: Vec<crate::compare::LayerDiff>,
    /// Some line misses the old per-line tolerance.
    pub reuse_warning: bool,
    pub passed: bool,
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// Worst cosine and relative error between two summaries, over layers.
pub fn compare(a: &[f32], b: &[f32], n_embd: usize) -> (f64, f64) {
    let mut min_cos = 1.0f64;
    let mut max_rel = 0.0f64;
    for (x, y) in a.chunks(n_embd).zip(b.chunks(n_embd)) {
        let dot: f64 = x
            .iter()
            .zip(y)
            .map(|(&p, &q)| f64::from(p) * f64::from(q))
            .sum();
        let (nx, ny) = (norm(x), norm(y));
        let cos = if nx == 0.0 && ny == 0.0 {
            1.0
        } else {
            dot / (nx * ny).max(1e-30)
        };
        let diff: Vec<f32> = x.iter().zip(y).map(|(p, q)| p - q).collect();
        let rel = norm(&diff) / nx.max(1e-30);
        min_cos = min_cos.min(cos);
        max_rel = max_rel.max(rel);
    }
    (min_cos, max_rel)
}

impl Capturer {
    /// The layer test: plant a known vector at layer k, and check the shift
    /// shows up at layer k's output, and not at k-1's.
    pub fn layer_test(&mut self, p: &Prepared, seed: u64) -> Result<Vec<LayerCheck>, String> {
        let n = self.layers.len();
        let mut ks = vec![
            self.layers[n.min(2) - 1],
            self.layers[n / 2],
            self.layers[n - 1],
        ];
        ks.dedup();
        let plain = self.capture(p, false)?;
        let e = self.n_embd();
        let mut rng = seed | 1;
        let mut out = Vec::new();
        for k in ks {
            let s = self.slot(k).unwrap();
            let scale = norm(&plain.mean[s * e..(s + 1) * e]).max(1.0) * 0.5;
            let mut v: Vec<f32> = (0..e)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect();
            let nv = norm(&v);
            v.iter_mut().for_each(|x| *x *= (scale / nv) as f32);
            self.plant(Some((k, &v)))?;
            let steered = self.capture(p, false);
            self.plant(None)?;
            let steered = steered?;
            let shift: Vec<f32> = (0..e)
                .map(|j| steered.mean[s * e + j] - plain.mean[s * e + j] - v[j])
                .collect();
            let rel_error = norm(&shift) / norm(&v);
            let leak_before = self.slot(k - 1).map(|b| {
                let d: Vec<f32> = (0..e)
                    .map(|j| steered.mean[b * e + j] - plain.mean[b * e + j])
                    .collect();
                norm(&d) / norm(&plain.mean[b * e..(b + 1) * e]).max(1e-30)
            });
            let ok = rel_error <= 1e-3 && leak_before.is_none_or(|l| l <= 1e-5);
            out.push(LayerCheck {
                layer: k,
                rel_error,
                leak_before,
                ok,
            });
        }
        Ok(out)
    }

    /// The copy test: each line with the persona copied by `seq_cp`
    /// against the same two decodes without the copy.
    pub fn copy_test(&mut self, lines: &[Prepared]) -> Result<Vec<ReuseCheck>, String> {
        let e = self.n_embd();
        let mut out = Vec::new();
        for p in lines {
            let alone = self.capture_with(p, Persona::Alone)?;
            let reused = self.capture_with(p, Persona::Reused)?;
            let (c1, r1) = compare(&alone.mean, &reused.mean, e);
            let (c2, r2) = compare(&alone.head, &reused.head, e);
            let max_rel = r1.max(r2);
            out.push(ReuseCheck {
                line_id: p.line_id.clone(),
                min_cos: c1.min(c2),
                max_rel,
                ok: max_rel <= COPY_MAX_REL,
            });
        }
        Ok(out)
    }

    /// The reported comparison: lines captured with and without prefix
    /// reuse, per line and per layer against the lines' spread.
    pub fn reuse_test(
        &mut self,
        lines: &[Prepared],
    ) -> Result<(Vec<ReuseCheck>, Vec<crate::compare::LayerDiff>), String> {
        let e = self.n_embd();
        let mut out = Vec::new();
        let mut diffs = crate::compare::Diffs::new(&self.layers, e);
        for p in lines {
            let fresh = self.capture(p, false)?;
            let reused = self.capture(p, true)?;
            diffs.add(&fresh.mean, &reused.mean);
            let (c1, r1) = compare(&fresh.mean, &reused.mean, e);
            let (c2, r2) = compare(&fresh.head, &reused.head, e);
            let (min_cos, max_rel) = (c1.min(c2), r1.max(r2));
            out.push(ReuseCheck {
                line_id: p.line_id.clone(),
                min_cos,
                max_rel,
                ok: min_cos >= REUSE_MIN_COS && max_rel <= REUSE_MAX_REL,
            });
        }
        Ok((out, diffs.finish()))
    }

    /// The layer test, the copy test on a few lines, and the reported
    /// full-decode comparison. Reuse also needs [`reuse_fit_check`] to pass.
    pub fn preflight(&mut self, lines: &[Prepared]) -> Result<Preflight, String> {
        let first = lines.first().ok_or("no lines to test")?;
        let layer_test = self.layer_test(first, 0x5eed)?;
        let copy_test = self.copy_test(&lines[..lines.len().min(5)])?;
        let (reuse_test, reuse_spread) =
            self.reuse_test(&lines[..lines.len().min(REUSE_WARNING_LINES)])?;
        let reuse_warning = reuse_test.iter().any(|c| !c.ok);
        let passed = layer_test.iter().all(|c| c.ok) && copy_test.iter().all(|c| c.ok);
        Ok(Preflight {
            layer_test,
            copy_test,
            reuse_test,
            reuse_spread,
            reuse_warning,
            passed,
        })
    }
}

/// The fit-level reuse gate's result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FitCheck {
    pub lines: usize,
    pub sets: Vec<String>,
    /// Layers gated; every layer is reported.
    pub layers: (u32, u32),
    /// Joint and mean_diff, prefixed `joint:` and `mean_diff:`.
    pub vectors: Vec<crate::compare::VectorDiff>,
    pub worst_cos: f64,
    pub worst_norm: f64,
    pub passed: bool,
}

pub const FIT_MIN_COS: f64 = 0.999;
pub const FIT_MAX_NORM: f64 = 0.01;
pub const FIT_MIN_LINES: usize = 100;

/// The lines of the first contrast sets (by name, or `sets`) until there
/// are at least [`FIT_MIN_LINES`].
pub fn fit_check_lines<'a>(
    lines: &'a [Prepared],
    pairs: &[crate::fit::Pair],
    sets: Option<&[String]>,
) -> (Vec<String>, Vec<&'a Prepared>) {
    let mut by_set: std::collections::BTreeMap<&str, std::collections::BTreeSet<&str>> =
        Default::default();
    for p in pairs {
        let s = by_set.entry(p.set.as_str()).or_default();
        s.insert(p.pos.as_str());
        s.insert(p.zero.as_str());
    }
    let mut chosen = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for (set, members) in &by_set {
        let take = match sets {
            Some(want) => want.iter().any(|w| w == set),
            None => chosen.len() < 2 || ids.len() < FIT_MIN_LINES,
        };
        if take {
            chosen.push(set.to_string());
            ids.extend(members.iter().copied());
        }
    }
    let subset = lines
        .iter()
        .filter(|p| ids.contains(p.line_id.as_str()))
        .collect();
    (chosen, subset)
}

/// The fit-level reuse gate: capture `lines` with the persona reused and
/// decoded whole, fit both with the same labels and pairs, and require
/// every vector at every layer in `range` to agree at cosine
/// [`FIT_MIN_COS`] with norms within [`FIT_MAX_NORM`]. Stores go in `dir`.
#[allow(clippy::too_many_arguments)]
pub fn reuse_fit_check(
    cap: &mut Capturer,
    lines: &[&Prepared],
    sets: Vec<String>,
    labels: &[crate::fit::LabelRow],
    pairs: &[crate::fit::Pair],
    range: (u32, u32),
    dir: &Path,
    mut progress: impl FnMut(&str),
) -> Result<FitCheck, String> {
    let owned: Vec<Prepared> = lines.iter().map(|p| (*p).clone()).collect();
    let mut fits = Vec::new();
    for (name, reuse) in [("reused", true), ("whole", false)] {
        progress(&format!("{} lines, persona {name}", owned.len()));
        let store = dir.join(name);
        let _ = fs::remove_dir_all(&store);
        let run = CaptureRun {
            model_file: Path::new("fit-check"),
            model_sha256: "fit-check",
            messages_sha256: "fit-check",
            store: &store,
            reuse,
        };
        capture_all(cap, &owned, &run, None, |_, _| {})?;
        let mut s = crate::store::Store::open(&store)?;
        fits.push(crate::fit::fit(
            &mut s,
            labels,
            pairs,
            ("fit-check", "fit-check"),
            &crate::fit::FitOptions::default(),
            |_| {},
        )?);
    }
    let mut vectors = Vec::new();
    for (method, a, b) in [
        ("joint", &fits[0].joint, &fits[1].joint),
        ("mean_diff", &fits[0].mean_diff, &fits[1].mean_diff),
    ] {
        for mut d in crate::compare::vector_maps(a, b)? {
            d.vector = format!("{method}:{}", d.vector);
            vectors.push(d);
        }
    }
    let gated = || {
        vectors
            .iter()
            .filter(|d| d.layer >= range.0 && d.layer <= range.1)
    };
    let worst_cos = gated().map(|d| d.cos).fold(1.0, f64::min);
    let worst_norm = gated()
        .map(|d| (d.norm_ratio - 1.0).abs())
        .fold(0.0, f64::max);
    let passed = gated().next().is_some() && worst_cos >= FIT_MIN_COS && worst_norm <= FIT_MAX_NORM;
    Ok(FitCheck {
        lines: owned.len(),
        sets,
        layers: range,
        vectors,
        worst_cos,
        worst_norm,
        passed,
    })
}

/// The layers the fit-level gate checks by default: the middle half of the
/// captured ones, where chat vectors are applied.
pub fn default_check_range(layers: &[usize]) -> (u32, u32) {
    let n = layers.len();
    (layers[n / 4] as u32, layers[(3 * n / 4).min(n - 1)] as u32)
}

// ------------------------------------------------------------- the capture

pub struct CaptureRun<'a> {
    pub model_file: &'a Path,
    pub model_sha256: &'a str,
    pub messages_sha256: &'a str,
    pub store: &'a Path,
    /// Reuse the decoded persona (the default). Without, every line's
    /// prompt is decoded whole: the fallback if the reuse test fails.
    pub reuse: bool,
}

/// Capture every line into the store at `run.store`, resuming after its
/// last checkpoint. `progress` gets (done, total) after each line.
pub fn capture_all(
    cap: &mut Capturer,
    lines: &[Prepared],
    run: &CaptureRun,
    preflight: Option<&Preflight>,
    mut progress: impl FnMut(usize, usize),
) -> Result<Meta, String> {
    let llama = cap.llama();
    let meta = Meta {
        version: 1,
        model_file: run
            .model_file
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        model_sha256: run.model_sha256.to_string(),
        architecture: llama.architecture(),
        llama_cpp: LLAMA_CPP.to_string(),
        n_embd: llama.n_embd(),
        n_layer: llama.n_layer(),
        layers: cap.layers.clone(),
        layer_tensor: cap.tap.prefix.trim_end_matches('-').to_string(),
        chat_template_sha256: sha256_text(&llama.chat_template()),
        messages_sha256: run.messages_sha256.to_string(),
        prefix_tokens: cap.prefix_tokens(),
        head_tokens: cap.head_tokens,
        think: cap.think,
        n_lines_total: lines.len(),
        complete: false,
        preflight: None,
    };
    let mut w = StoreWriter::open(run.store, meta)?;
    let done = w.done().len();
    for (i, (r, p)) in w.done().iter().zip(lines).enumerate() {
        if r.line_id != p.line_id || r.text_sha256 != p.text_sha256 {
            return Err(format!(
                "store line {i} is {}, the messages say {}; delete the store and capture again",
                r.line_id, p.line_id
            ));
        }
    }
    for p in &lines[done..] {
        let s = cap.capture(p, run.reuse)?;
        w.push(
            LineRecord {
                line_id: p.line_id.clone(),
                text_sha256: p.text_sha256.clone(),
                n_prompt_tokens: p.n_prompt,
                n_line_tokens: p.tokens.len() - p.n_prompt,
                logprob: s.logprob,
            },
            &s.mean,
            &s.head,
            &s.norms,
        )?;
        progress(w.done().len(), lines.len());
    }
    let preflight = preflight.map(|p| serde_json::to_value(p).unwrap());
    w.finish(preflight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tap() -> Tap {
        Tap {
            active: true,
            prefix: "l_out-".into(),
            n_embd: 4,
            slot_of: vec![None, Some(0), Some(1), None],
            n_tokens: 5,
            mean_rows: (1, 5),
            head_rows: (1, 3),
            seen: vec![0, 0],
            mean: vec![0.0; 8],
            head: vec![0.0; 8],
            norms: vec![0.0; 2],
            scratch: Vec::new(),
            error: None,
        }
    }

    fn tensor(ne: [i64; 4]) -> sys::ggml_tensor {
        // SAFETY: a plain C struct; all-zero is a valid (empty) tensor.
        let mut t: sys::ggml_tensor = unsafe { std::mem::zeroed() };
        t.type_ = sys::GGML_TYPE_F32;
        t.ne = ne;
        t.nb = [4, 4 * ne[0] as usize, 0, 0];
        t
    }

    #[test]
    fn the_tap_refuses_3d_wrong_shapes_and_repeats() {
        let mut t = tap();
        assert_eq!(t.slot("l_out-2"), Some(1));
        assert_eq!(t.slot("l_out-0"), None);
        assert_eq!(t.slot("l_out-3"), None);
        assert_eq!(t.slot("l_last-1"), None);

        let mut x = tensor([4, 5, 2, 1]);
        unsafe { t.receive(&mut x, 0, "l_out-1") };
        assert!(t.error.take().unwrap().contains("3-D"));

        let mut t = tap();
        let mut x = tensor([4, 6, 1, 1]);
        unsafe { t.receive(&mut x, 0, "l_out-1") };
        assert!(t.error.take().unwrap().contains("4 x 6"));

        let mut t = tap();
        t.seen[1] = 1;
        let mut x = tensor([4, 5, 1, 1]);
        unsafe { t.receive(&mut x, 1, "l_out-2") };
        assert!(t.error.take().unwrap().contains("more than once"));
    }

    #[test]
    fn common_prefix_stays_on_char_boundaries() {
        assert_eq!(common_prefix("abcé", "abcè"), "abc");
        assert_eq!(common_prefix("ab", "abc"), "ab");
    }
}
