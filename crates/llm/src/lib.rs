//! The dialogue model: llama.cpp embedded in-process, steered per turn by
//! a blend of control vectors.
//!
//! [`LlamaDialogue`] plugs into the turn pipeline as its
//! [`engine::pipeline::Dialogue`]. Each request carries the mixing table's
//! `llm` strengths; the [`vectors::VectorBank`] turns them into one blended
//! buffer, and [`llama::Llama`] applies it while generating the reply.
//!
//! llama.cpp is built from source by `llama-cpp-sys-2`, so building this
//! crate needs cmake and a C++ compiler. CPU only by default; the `cuda`
//! and `vulkan` features add those backends.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use engine::pipeline::{Dialogue, DialogueRequest, Role};
use tokio::sync::mpsc;

pub mod capture;
pub mod check;
pub mod f16;
pub mod fit;
pub mod gguf;
mod linalg;
pub mod llama;
pub mod registry;
pub mod selftest;
pub mod store;
pub mod testmodel;
pub mod vectors;

use llama::{Llama, Sampling};
use vectors::VectorBank;

/// What the last turn actually did, for logs and `vi chat`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnInfo {
    /// Vector name to the strength applied, after each vector's limits.
    pub applied: BTreeMap<String, f32>,
    /// Vectors the mixing table asked for that aren't in the bank.
    pub missing: Vec<String>,
    /// Prompt tokens reused from the cache.
    pub reused_tokens: usize,
    pub generated_tokens: usize,
}

struct Inner {
    llama: Llama,
    bank: VectorBank,
    last: TurnInfo,
}

/// The real dialogue model. Generation runs on a blocking thread and
/// streams text back as it's produced.
pub struct LlamaDialogue {
    inner: Arc<Mutex<Inner>>,
    pub sampling: Sampling,
}

impl LlamaDialogue {
    pub fn new(llama: Llama, bank: VectorBank) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                llama,
                bank,
                last: TurnInfo::default(),
            })),
            sampling: Sampling::default(),
        }
    }

    pub fn last_turn(&self) -> TurnInfo {
        self.inner.lock().unwrap().last.clone()
    }
}

/// The `(role, content)` messages a turn's prompt is rendered from. The
/// vector capture builds its prompts through this too, so they match
/// `vi chat`'s byte for byte.
pub fn chat_messages(req: &DialogueRequest) -> Vec<(&'static str, &str)> {
    let mut messages = vec![("system", req.system.as_str())];
    messages.extend(req.messages.iter().map(|m| {
        let role = match m.role {
            Role::User => "user",
            Role::Character => "assistant",
        };
        (role, m.text.as_str())
    }));
    messages
}

impl Dialogue for LlamaDialogue {
    async fn stream(&self, req: &DialogueRequest, out: mpsc::Sender<String>) -> Result<(), String> {
        let inner = self.inner.clone();
        let req = req.clone();
        let sampling = self.sampling.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock().map_err(|_| "dialogue model poisoned")?;
            let inner = &mut *guard;
            let blend = inner.bank.blend(&req.vectors);
            let prompt = inner.llama.format_chat(&chat_messages(&req))?;
            let mut receiver_gone = false;
            let generation =
                inner
                    .llama
                    .generate(&prompt, blend.data.as_deref(), &sampling, |text| {
                        receiver_gone = out.blocking_send(text.to_string()).is_err();
                        !receiver_gone
                    })?;
            inner.last = TurnInfo {
                applied: blend.applied,
                missing: blend.missing,
                reused_tokens: generation.reused,
                generated_tokens: generation.tokens.len(),
            };
            if receiver_gone {
                return Err("the reply's listener went away".to_string());
            }
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }
}
