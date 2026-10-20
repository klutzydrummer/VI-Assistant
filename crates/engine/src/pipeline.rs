//! One conversational turn, end to end:
//!
//! 1. The director reads the user's message and picks events.
//! 2. The state machine applies them.
//! 3. The dialogue model streams a plain-text reply, steered by the
//!    mixing table's `llm` vectors. A leading `<think>…</think>` block is
//!    kept apart: it's never voiced or stored as her line.
//! 4. Each sentence, as soon as it's complete, is cleaned up, directed and
//!    voiced with the `tts` vectors, while the rest is still streaming.
//! 5. Time passes for the state machine, and the turn is committed.
//!
//! Nothing in the session changes until the turn succeeds, so a failed turn
//! (a rented GPU vanishing mid-reply) can simply be retried.

use std::collections::BTreeMap;
use std::future::Future;

use charpack::machine::{Change, Machine, Snapshot};
use charpack::Pack;
use tokio::sync::mpsc;

use crate::director::{Delivery, Director, SentenceContext, TurnContext};
use crate::prompt::system_prompt;
use crate::text::{normalize, SentenceSplitter, ThinkSplitter};

/// How many past messages go to the dialogue model.
const HISTORY_WINDOW: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Character,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

/// Everything the dialogue model needs for one reply. Self-contained, so a
/// stateless worker can serve it.
#[derive(Debug, Clone)]
pub struct DialogueRequest {
    pub system: String,
    pub messages: Vec<Message>,
    /// Control vector name to strength for the LLM.
    pub vectors: BTreeMap<String, f32>,
}

pub trait Dialogue {
    /// Stream the reply as text chunks into `out`.
    fn stream(
        &self,
        req: &DialogueRequest,
        out: mpsc::Sender<String>,
    ) -> impl Future<Output = Result<(), String>>;
}

pub trait Voice {
    /// Synthesize and queue one sentence for playback.
    fn speak(
        &self,
        sentence: &str,
        delivery: &Delivery,
    ) -> impl Future<Output = Result<(), String>>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnReport {
    /// Events the director picked and the machine accepted.
    pub events: Vec<String>,
    /// Events the machine rejected, with the reason.
    pub rejected: Vec<String>,
    pub changes: Vec<Change>,
    pub llm_vectors: BTreeMap<String, f32>,
    pub reply: String,
    /// The model's thinking before its reply, for debugging; `None` if it
    /// didn't think.
    pub thinking: Option<String>,
    pub sentences: Vec<(String, Delivery)>,
}

pub struct Session {
    pub pack: Pack,
    pub machine: Machine,
    pub snapshot: Snapshot,
    pub history: Vec<Message>,
}

impl Session {
    pub fn new(pack: Pack, machine: Machine) -> Self {
        let snapshot = machine.start();
        Self {
            pack,
            machine,
            snapshot,
            history: Vec::new(),
        }
    }

    pub async fn turn<D: Director, L: Dialogue, V: Voice>(
        &mut self,
        user_message: &str,
        director: &D,
        dialogue: &L,
        voice: &V,
    ) -> Result<TurnReport, String> {
        // Work on a copy; commit only at the end.
        let mut snap = self.snapshot.clone();

        let picked = director
            .on_user_turn(&TurnContext {
                user_message,
                machine: &self.machine,
                snapshot: &snap,
            })
            .await;
        let mut events = Vec::new();
        let mut rejected = Vec::new();
        let mut changes = Vec::new();
        for event in picked {
            match self.machine.apply(&mut snap, &event) {
                Ok(c) => {
                    changes.extend(c);
                    events.push(event);
                }
                Err(e) => rejected.push(e.to_string()),
            }
        }

        let mut messages: Vec<Message> = self
            .history
            .iter()
            .rev()
            .take(HISTORY_WINDOW - 1)
            .rev()
            .cloned()
            .collect();
        messages.push(Message {
            role: Role::User,
            text: user_message.to_string(),
        });
        let req = DialogueRequest {
            system: system_prompt(&self.pack, &self.machine, &snap),
            messages,
            vectors: self.machine.mix(&snap, "llm"),
        };
        let base = self.machine.mix(&snap, "tts");
        let default_emotion = self.pack.manifest.voice.default_emotion.clone();

        let (tx, mut rx) = mpsc::channel::<String>(64);
        let produce = dialogue.stream(&req, tx);
        let consume = async {
            let mut think = ThinkSplitter::default();
            let mut splitter = SentenceSplitter::default();
            let mut reply = String::new();
            let mut sentences = Vec::new();
            let mut index = 0;
            let mut voice_one = |raw: String| {
                let sentence = normalize(&raw);
                let i = index;
                index += 1;
                let base = &base;
                let default_emotion = default_emotion.as_str();
                async move {
                    if sentence.is_empty() {
                        return Ok(None);
                    }
                    let ctx = SentenceContext {
                        sentence: &sentence,
                        index: i,
                        base,
                        default_emotion,
                    };
                    let delivery = director.on_sentence(&ctx).await;
                    voice.speak(&sentence, &delivery).await?;
                    Ok::<_, String>(Some((sentence, delivery)))
                }
            };
            while let Some(chunk) = rx.recv().await {
                let chunk = think.push(&chunk);
                reply.push_str(&chunk);
                for raw in splitter.push(&chunk) {
                    sentences.extend(voice_one(raw).await?);
                }
            }
            let chunk = think.finish();
            reply.push_str(&chunk);
            for raw in splitter.push(&chunk) {
                sentences.extend(voice_one(raw).await?);
            }
            if let Some(raw) = splitter.finish() {
                sentences.extend(voice_one(raw).await?);
            }
            Ok::<_, String>((reply, think.thinking(), sentences))
        };
        let (produced, consumed) = tokio::join!(produce, consume);
        produced?;
        let (reply, thinking, sentences) = consumed?;

        changes.extend(self.machine.end_turn(&mut snap));

        // Commit.
        self.snapshot = snap;
        self.history.push(Message {
            role: Role::User,
            text: user_message.to_string(),
        });
        self.history.push(Message {
            role: Role::Character,
            text: normalize(&reply),
        });

        Ok(TurnReport {
            events,
            rejected,
            changes,
            llm_vectors: req.vectors,
            reply: normalize(&reply),
            thinking,
            sentences,
        })
    }
}
