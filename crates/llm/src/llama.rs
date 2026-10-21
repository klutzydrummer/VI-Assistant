//! A thin safe wrapper over the parts of llama.cpp we use: load a model,
//! format a chat, and generate with an optional control vector.
//!
//! Steering applies to generation only. The prompt is decoded with the
//! control vector off, so its KV cache doesn't depend on the blend and the
//! next turn reuses the common prefix whatever the mood. The one exception
//! is the prompt's last token: its logits pick the first word of the reply,
//! so it's decoded with the vector on, and like every generated token it is
//! dropped from the cache when the turn ends.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Mutex, Once};

use llama_cpp_sys_2 as sys;
use sys::llama_token;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub path: PathBuf,
    /// Context length in tokens; 0 means the model's training length.
    pub n_ctx: u32,
    /// Layers to offload to the GPU; -1 for all. Ignored on CPU builds.
    pub n_gpu_layers: i32,
    /// CPU threads; `None` lets llama.cpp choose.
    pub n_threads: Option<i32>,
}

impl ModelConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            n_ctx: 8192,
            n_gpu_layers: -1,
            n_threads: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    /// 0 for greedy decoding.
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
    pub seed: u32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            max_tokens: 400,
            seed: 0xFFFF_FFFF, // LLAMA_DEFAULT_SEED: random
        }
    }
}

impl Sampling {
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            repeat_penalty: 1.0,
            max_tokens,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The model ended its turn.
    EndOfTurn,
    MaxTokens,
    /// The caller stopped listening.
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub tokens: Vec<llama_token>,
    pub text: String,
    /// Prompt tokens served from the cache instead of being decoded.
    pub reused: usize,
    pub stop: Stop,
    /// Where steering switched on.
    pub steer_start: SteerStart,
}

/// Where a generation starts steering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// From the last prompt token on, thinking included (`vi chat` today).
    #[default]
    All,
    /// From `</think>` on: the prompt and any thinking stay unsteered.
    Reply,
}

impl Scope {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "all" => Ok(Self::All),
            "reply" => Ok(Self::Reply),
            _ => Err(format!("steering scope is reply or all, not {s:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Reply => "reply",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerStart {
    /// At this prompt token.
    Prompt(usize),
    /// At this reply token (the sampled `</think>`).
    Reply(usize),
    /// `</think>` never came.
    Never,
}

/// Closes Qwen's think block.
pub const THINK_END: &str = "</think>";
/// An empty think block: what a prompt ends with for thinking off.
pub const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";

pub struct Llama {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    n_embd: usize,
    n_layer: usize,
    n_ctx: usize,
    n_batch: usize,
    template: Option<CString>,
    backend: String,
    /// Tokens whose KV entries are in the cache, all decoded unsteered.
    cached: Vec<llama_token>,
}

// SAFETY: llama.cpp contexts aren't tied to a thread; `&mut self` on every
// call that touches them keeps access exclusive.
unsafe impl Send for Llama {}

impl Drop for Llama {
    fn drop(&mut self) {
        let _guard = SETUP.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

static INIT: Once = Once::new();

/// Held while a model and context are made or freed. The Vulkan backend's
/// device setup isn't thread-safe: two contexts made at once in different
/// threads crash in `ggml_vk_init`.
static SETUP: Mutex<()> = Mutex::new(());

fn init_backend() {
    INIT.call_once(|| unsafe {
        sys::llama_log_set(Some(log), ptr::null_mut());
        sys::llama_backend_init();
    });
}

/// [`Llama::backend`]: the first GPU device llama.cpp offloads to.
fn backend_name(n_gpu_layers: i32, n_layer: i32) -> String {
    if n_gpu_layers == 0 {
        return "cpu".into();
    }
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            let ty = sys::ggml_backend_dev_type(dev);
            if ty != sys::GGML_BACKEND_DEVICE_TYPE_GPU && ty != sys::GGML_BACKEND_DEVICE_TYPE_IGPU {
                continue;
            }
            let text = |p: *const c_char| {
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            let mut name = format!(
                "{} ({})",
                text(sys::ggml_backend_dev_name(dev)),
                text(sys::ggml_backend_dev_description(dev))
            );
            if n_gpu_layers > 0 && n_gpu_layers < n_layer {
                name += &format!(", {n_gpu_layers} of {n_layer} layers");
            }
            return name;
        }
    }
    "cpu".into()
}

/// Pass llama.cpp's warnings and errors through, or everything with
/// `VI_LLAMA_LOG=1`.
unsafe extern "C" fn log(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    let verbose = std::env::var_os("VI_LLAMA_LOG").is_some_and(|v| v == "1");
    if text.is_null() || !(verbose || level == sys::GGML_LOG_LEVEL_ERROR) {
        return;
    }
    eprint!("{}", CStr::from_ptr(text).to_string_lossy());
}

impl Llama {
    pub fn load(config: &ModelConfig) -> Result<Self, String> {
        Self::load_with(config, |_| {})
    }

    /// [`Llama::load`], with a last say over the context's parameters (the
    /// capture's sequences, batch sizes and eval callback).
    pub(crate) fn load_with(
        config: &ModelConfig,
        tweak: impl FnOnce(&mut sys::llama_context_params),
    ) -> Result<Self, String> {
        init_backend();
        let path = path_cstring(&config.path)?;
        let _guard = SETUP.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let mut mp = sys::llama_model_default_params();
            mp.n_gpu_layers = config.n_gpu_layers;
            let model = sys::llama_model_load_from_file(path.as_ptr(), mp);
            if model.is_null() {
                return Err(format!("could not load {}", config.path.display()));
            }
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = config.n_ctx;
            cp.no_perf = true;
            if let Some(n) = config.n_threads {
                cp.n_threads = n;
                cp.n_threads_batch = n;
            }
            tweak(&mut cp);
            let ctx = sys::llama_init_from_model(model, cp);
            if ctx.is_null() {
                sys::llama_model_free(model);
                return Err("could not create a llama.cpp context".into());
            }
            let template = sys::llama_model_chat_template(model, ptr::null());
            let template = (!template.is_null()).then(|| CStr::from_ptr(template).to_owned());
            Ok(Self {
                model,
                ctx,
                vocab: sys::llama_model_get_vocab(model),
                n_embd: sys::llama_model_n_embd(model) as usize,
                n_layer: sys::llama_model_n_layer(model) as usize,
                n_ctx: sys::llama_n_ctx(ctx) as usize,
                n_batch: sys::llama_n_batch(ctx) as usize,
                template,
                backend: backend_name(config.n_gpu_layers, sys::llama_model_n_layer(model)),
                cached: Vec::new(),
            })
        }
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Where the layers run: `cpu`, or the GPU device, e.g.
    /// `Vulkan0 (AMD Radeon RX 480 Graphics (RADV POLARIS10))`, with the
    /// offloaded layer count when only some are.
    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    pub(crate) fn ctx(&self) -> *mut sys::llama_context {
        self.ctx
    }

    pub(crate) fn vocab(&self) -> *const sys::llama_vocab {
        self.vocab
    }

    /// `general.architecture` from the model file.
    pub fn architecture(&self) -> String {
        let mut buf = [0u8; 128];
        let n = unsafe {
            sys::llama_model_meta_val_str(
                self.model,
                c"general.architecture".as_ptr(),
                buf.as_mut_ptr().cast(),
                buf.len(),
            )
        };
        if n <= 0 {
            return String::new();
        }
        String::from_utf8_lossy(&buf[..(n as usize).min(buf.len() - 1)]).into_owned()
    }

    /// The chat template `format_chat` uses: the model's own, or "chatml".
    pub fn chat_template(&self) -> String {
        self.template
            .as_deref()
            .map_or_else(|| "chatml".into(), |t| t.to_string_lossy().into_owned())
    }

    /// Length of the steering buffer [`Llama::generate`] expects.
    pub fn steer_len(&self) -> usize {
        self.n_embd * (self.n_layer - 1)
    }

    /// Render `(role, content)` messages with the model's chat template
    /// (ChatML if it has none), ending with the assistant's turn opening.
    pub fn format_chat(&self, messages: &[(&str, &str)]) -> Result<String, String> {
        let owned: Vec<(CString, CString)> = messages
            .iter()
            .map(|(r, c)| Ok((CString::new(*r)?, CString::new(*c)?)))
            .collect::<Result<_, std::ffi::NulError>>()
            .map_err(|e| e.to_string())?;
        let chat: Vec<sys::llama_chat_message> = owned
            .iter()
            .map(|(r, c)| sys::llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();
        let chatml = c"chatml";
        let tmpl = self.template.as_deref().unwrap_or(chatml);
        let mut buf = vec![0u8; 4096];
        loop {
            let n = unsafe {
                sys::llama_chat_apply_template(
                    tmpl.as_ptr(),
                    chat.as_ptr(),
                    chat.len(),
                    true,
                    buf.as_mut_ptr().cast(),
                    buf.len() as i32,
                )
            };
            if n < 0 {
                return Err("the model's chat template isn't one llama.cpp knows".into());
            }
            if n as usize <= buf.len() {
                buf.truncate(n as usize);
                return String::from_utf8(buf).map_err(|e| e.to_string());
            }
            buf.resize(n as usize, 0);
        }
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<llama_token>, String> {
        self.tokenize_with(text, true)
    }

    fn tokenize_with(&self, text: &str, add_special: bool) -> Result<Vec<llama_token>, String> {
        let len = i32::try_from(text.len()).map_err(|_| "prompt too long")?;
        let mut tokens = vec![0; text.len() + 2];
        loop {
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast(),
                    len,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    true,
                )
            };
            if n >= 0 {
                tokens.truncate(n as usize);
                return Ok(tokens);
            }
            if n == i32::MIN {
                return Err("prompt too long to tokenize".into());
            }
            tokens.resize((-n) as usize, 0);
        }
    }

    fn piece(&self, token: llama_token, out: &mut Vec<u8>) {
        let mut buf = [0u8; 256];
        let n = unsafe {
            sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr().cast(),
                buf.len() as i32,
                0,
                false,
            )
        };
        if n > 0 {
            out.extend_from_slice(&buf[..n as usize]);
        }
    }

    fn set_steering(&mut self, steer: Option<&[f32]>) -> Result<(), String> {
        self.set_steering_layers(steer, 1, self.n_layer as i32 - 1)
    }

    /// Steer only layers `first..=last` with `steer` (`steer_len()` floats).
    pub(crate) fn set_steering_layers(
        &mut self,
        steer: Option<&[f32]>,
        first: i32,
        last: i32,
    ) -> Result<(), String> {
        let (data, len) = match steer {
            Some(s) => (s.as_ptr(), s.len()),
            None => (ptr::null(), 0),
        };
        let rc = unsafe {
            sys::llama_set_adapter_cvec(self.ctx, data, len, self.n_embd as i32, first, last)
        };
        if rc != 0 {
            return Err("llama.cpp rejected the control vector".into());
        }
        Ok(())
    }

    fn decode(&mut self, tokens: &[llama_token]) -> Result<(), String> {
        for chunk in tokens.chunks(self.n_batch.max(1)) {
            let mut chunk = chunk.to_vec();
            let rc = unsafe {
                sys::llama_decode(
                    self.ctx,
                    sys::llama_batch_get_one(chunk.as_mut_ptr(), chunk.len() as i32),
                )
            };
            if rc != 0 {
                return Err(format!("llama_decode failed ({rc})"));
            }
        }
        Ok(())
    }

    /// Drop KV entries from position `from` on. Falls back to clearing
    /// everything for models whose memory can't be truncated.
    fn truncate_cache(&mut self, from: usize) {
        unsafe {
            let mem = sys::llama_get_memory(self.ctx);
            if !sys::llama_memory_seq_rm(mem, 0, from as i32, -1) {
                sys::llama_memory_clear(mem, true);
                self.cached.clear();
                return;
            }
        }
        self.cached.truncate(from);
    }

    /// Generate a reply to `prompt` (already formatted with
    /// [`Llama::format_chat`]), steered by `steer` (`steer_len()` floats,
    /// or `None`) from the last prompt token on ([`Scope::All`]). `on_text`
    /// gets the reply as it streams, in valid UTF-8 pieces, and returns
    /// false to stop.
    pub fn generate(
        &mut self,
        prompt: &str,
        steer: Option<&[f32]>,
        sampling: &Sampling,
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<Generation, String> {
        self.generate_scoped(prompt, steer, Scope::All, sampling, on_text)
    }

    /// [`Llama::generate`] with a steering scope. [`Scope::Reply`] leaves
    /// the prompt and any thinking unsteered and steers from `</think>` on:
    /// from the prompt's `</think>` when the prompt closes the think block
    /// itself, else from the moment the model samples it. Refuses a model
    /// where `</think>` isn't a single token.
    pub fn generate_scoped(
        &mut self,
        prompt: &str,
        steer: Option<&[f32]>,
        scope: Scope,
        sampling: &Sampling,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<Generation, String> {
        let switch = match scope {
            Scope::All => None,
            Scope::Reply => Some(self.think_end()?),
        };
        let closed = prompt.trim_end().ends_with(THINK_END);
        self.generate_switching(prompt, steer, switch, closed, sampling, &mut on_text)
    }

    /// The `</think>` token, or why there isn't one.
    pub fn think_end(&self) -> Result<llama_token, String> {
        self.single_token(THINK_END).ok_or_else(|| {
            format!("{THINK_END} isn't a single token in this model, so steering can't start there")
        })
    }

    /// `text`'s token, if it tokenizes (without BOS) as exactly one.
    pub fn single_token(&self, text: &str) -> Option<llama_token> {
        match self.tokenize_with(text, false).ok()?.as_slice() {
            [t] => Some(*t),
            _ => None,
        }
    }

    /// The reply scope's machinery, with the switch token given: steering
    /// starts at `switch` (the prompt's last one when `closed`, else the
    /// first one sampled), or at the last prompt token when `switch` is
    /// `None`. Public for tests, which can't make a random model say
    /// `</think>`.
    #[doc(hidden)]
    pub fn generate_switching(
        &mut self,
        prompt: &str,
        steer: Option<&[f32]>,
        switch: Option<llama_token>,
        closed: bool,
        sampling: &Sampling,
        on_text: &mut impl FnMut(&str) -> bool,
    ) -> Result<Generation, String> {
        if let Some(s) = steer {
            if s.len() != self.steer_len() {
                return Err(format!(
                    "steering buffer has {} floats, the model needs {}",
                    s.len(),
                    self.steer_len()
                ));
            }
        }
        let prompt = self.tokenize(prompt)?;
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() + sampling.max_tokens > self.n_ctx {
            return Err(format!(
                "prompt ({} tokens) plus reply ({}) doesn't fit the context ({})",
                prompt.len(),
                sampling.max_tokens,
                self.n_ctx
            ));
        }
        // Where the steered part of the prompt starts.
        let (start, switch) = match switch {
            None => (prompt.len() - 1, None),
            Some(t) if closed => match prompt.iter().rposition(|&p| p == t) {
                Some(i) => (i, None),
                None => return Err("the prompt ends with </think> but doesn't tokenize it".into()),
            },
            Some(t) => (prompt.len(), Some(t)),
        };
        let result = self.generate_inner(&prompt, start, switch, steer, sampling, on_text);
        match &result {
            // Keep the unsteered prompt, drop the rest.
            Ok(_) => self.truncate_cache(start),
            Err(_) => self.truncate_cache(0),
        }
        result
    }

    fn generate_inner(
        &mut self,
        prompt: &[llama_token],
        start: usize,
        switch: Option<llama_token>,
        steer: Option<&[f32]>,
        sampling: &Sampling,
        on_text: &mut impl FnMut(&str) -> bool,
    ) -> Result<Generation, String> {
        let head = &prompt[..start];
        let reused = self
            .cached
            .iter()
            .zip(head)
            .take_while(|(a, b)| a == b)
            .count();
        // Sampling needs the last prompt token's logits, so it's decoded
        // even when the whole prompt is cached.
        self.truncate_cache(reused.min(prompt.len() - 1));
        let reused = self.cached.len();

        self.set_steering(None)?;
        let mut steer_start = SteerStart::Never;
        if start < prompt.len() {
            self.decode(&prompt[reused..start])?;
            self.cached = head.to_vec();
            self.set_steering(steer)?;
            self.decode(&prompt[start..])?;
            steer_start = SteerStart::Prompt(start);
        } else {
            self.decode(&prompt[reused..])?;
            self.cached = prompt.to_vec();
        }

        let sampler = Sampler::new(self, sampling);
        let mut tokens = Vec::new();
        let mut pending = Vec::new();
        let mut text = String::new();
        let stop = loop {
            let token = unsafe { sys::llama_sampler_sample(sampler.0, self.ctx, -1) };
            if unsafe { sys::llama_vocab_is_eog(self.vocab, token) } {
                break Stop::EndOfTurn;
            }
            if steer_start == SteerStart::Never && Some(token) == switch {
                self.set_steering(steer)?;
                steer_start = SteerStart::Reply(tokens.len());
            }
            tokens.push(token);
            self.piece(token, &mut pending);
            let ready = take_utf8(&mut pending, false);
            if !ready.is_empty() {
                text.push_str(&ready);
                if !on_text(&ready) {
                    break Stop::Cancelled;
                }
            }
            if tokens.len() >= sampling.max_tokens {
                break Stop::MaxTokens;
            }
            self.decode(&[token])?;
        };
        let rest = take_utf8(&mut pending, true);
        if !rest.is_empty() {
            text.push_str(&rest);
            if stop != Stop::Cancelled {
                on_text(&rest);
            }
        }
        Ok(Generation {
            tokens,
            text,
            reused,
            stop,
            steer_start,
        })
    }
}

struct Sampler(*mut sys::llama_sampler);

impl Sampler {
    fn new(llama: &Llama, s: &Sampling) -> Self {
        unsafe {
            let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
            let add = |smpl| sys::llama_sampler_chain_add(chain, smpl);
            if s.repeat_penalty != 1.0 {
                let n_vocab = sys::llama_vocab_n_tokens(llama.vocab);
                add(sys::llama_sampler_init_penalties(
                    n_vocab,
                    64,
                    s.repeat_penalty,
                    0.0,
                    0.0,
                ));
            }
            if s.temperature <= 0.0 {
                add(sys::llama_sampler_init_greedy());
            } else {
                add(sys::llama_sampler_init_top_p(s.top_p, 1));
                add(sys::llama_sampler_init_min_p(s.min_p, 1));
                add(sys::llama_sampler_init_temp(s.temperature));
                add(sys::llama_sampler_init_dist(s.seed));
            }
            Self(chain)
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { sys::llama_sampler_free(self.0) }
    }
}

/// Take the longest valid UTF-8 prefix of `bytes`, replacing invalid
/// sequences. An incomplete character at the end waits for more bytes,
/// unless `flush`.
fn take_utf8(bytes: &mut Vec<u8>, flush: bool) -> String {
    let mut out = String::new();
    loop {
        match std::str::from_utf8(bytes) {
            Ok(s) => {
                out.push_str(s);
                bytes.clear();
                return out;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                out.push_str(std::str::from_utf8(&bytes[..valid]).unwrap());
                match e.error_len() {
                    Some(bad) => {
                        out.push(char::REPLACEMENT_CHARACTER);
                        bytes.drain(..valid + bad);
                    }
                    None if flush => {
                        out.push(char::REPLACEMENT_CHARACTER);
                        bytes.clear();
                        return out;
                    }
                    None => {
                        bytes.drain(..valid);
                        return out;
                    }
                }
            }
        }
    }
}

fn path_cstring(path: &Path) -> Result<CString, String> {
    let s = path
        .to_str()
        .ok_or_else(|| format!("{} is not valid UTF-8", path.display()))?;
    CString::new(s).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::take_utf8;

    #[test]
    fn utf8_waits_for_split_characters() {
        let mut b = "Keelah".as_bytes().to_vec();
        b.extend_from_slice(&"é".as_bytes()[..1]);
        assert_eq!(take_utf8(&mut b, false), "Keelah");
        b.extend_from_slice(&"é".as_bytes()[1..]);
        assert_eq!(take_utf8(&mut b, false), "é");
        assert!(b.is_empty());
    }

    #[test]
    fn utf8_replaces_garbage() {
        let mut b = vec![b'a', 0xff, b'b', 0xe2];
        assert_eq!(take_utf8(&mut b, false), "a\u{fffd}b");
        assert_eq!(take_utf8(&mut b, true), "\u{fffd}");
    }
}
