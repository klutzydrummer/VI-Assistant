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
use std::sync::Once;

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
}

pub struct Llama {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    n_embd: usize,
    n_layer: usize,
    n_ctx: usize,
    n_batch: usize,
    template: Option<CString>,
    /// Tokens whose KV entries are in the cache, all decoded unsteered.
    cached: Vec<llama_token>,
}

// SAFETY: llama.cpp contexts aren't tied to a thread; `&mut self` on every
// call that touches them keeps access exclusive.
unsafe impl Send for Llama {}

impl Drop for Llama {
    fn drop(&mut self) {
        unsafe {
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

static INIT: Once = Once::new();

fn init_backend() {
    INIT.call_once(|| unsafe {
        sys::llama_log_set(Some(log), ptr::null_mut());
        sys::llama_backend_init();
    });
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
        init_backend();
        let path = path_cstring(&config.path)?;
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
                cached: Vec::new(),
            })
        }
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
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
                    true,
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
        let (data, len) = match steer {
            Some(s) => (s.as_ptr(), s.len()),
            None => (ptr::null(), 0),
        };
        let rc = unsafe {
            sys::llama_set_adapter_cvec(
                self.ctx,
                data,
                len,
                self.n_embd as i32,
                1,
                self.n_layer as i32 - 1,
            )
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
    /// or `None`). `on_text` gets the reply as it streams, in valid UTF-8
    /// pieces, and returns false to stop.
    pub fn generate(
        &mut self,
        prompt: &str,
        steer: Option<&[f32]>,
        sampling: &Sampling,
        mut on_text: impl FnMut(&str) -> bool,
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
        let Some((&last, head)) = prompt.split_last() else {
            return Err("empty prompt".into());
        };
        if prompt.len() + sampling.max_tokens > self.n_ctx {
            return Err(format!(
                "prompt ({} tokens) plus reply ({}) doesn't fit the context ({})",
                prompt.len(),
                sampling.max_tokens,
                self.n_ctx
            ));
        }
        let result = self.generate_inner(head, last, steer, sampling, &mut on_text);
        match &result {
            // Keep the unsteered prompt, drop the rest.
            Ok(_) => self.truncate_cache(head.len()),
            Err(_) => self.truncate_cache(0),
        }
        result
    }

    fn generate_inner(
        &mut self,
        head: &[llama_token],
        last: llama_token,
        steer: Option<&[f32]>,
        sampling: &Sampling,
        on_text: &mut impl FnMut(&str) -> bool,
    ) -> Result<Generation, String> {
        let reused = self
            .cached
            .iter()
            .zip(head)
            .take_while(|(a, b)| a == b)
            .count();
        self.truncate_cache(reused);
        let reused = self.cached.len();

        self.set_steering(None)?;
        self.decode(&head[reused..])?;
        self.cached = head.to_vec();
        self.set_steering(steer)?;
        self.decode(&[last])?;

        let sampler = Sampler::new(self, sampling);
        let mut tokens = Vec::new();
        let mut pending = Vec::new();
        let mut text = String::new();
        let stop = loop {
            let token = unsafe { sys::llama_sampler_sample(sampler.0, self.ctx, -1) };
            if unsafe { sys::llama_vocab_is_eog(self.vocab, token) } {
                break Stop::EndOfTurn;
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
