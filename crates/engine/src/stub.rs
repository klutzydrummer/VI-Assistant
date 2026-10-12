//! Placeholder dialogue and voice, for running the pipeline before the
//! real models exist.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::director::Delivery;
use crate::pipeline::{Dialogue, DialogueRequest, Voice};

/// Streams canned replies word by word, like a model would. When the
/// script runs out it says so in character.
#[derive(Debug, Default)]
pub struct ScriptedDialogue {
    replies: Mutex<VecDeque<String>>,
    /// Pause between words, to make streaming visible.
    pub word_delay: Duration,
}

impl ScriptedDialogue {
    pub fn new(replies: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().map(Into::into).collect()),
            word_delay: Duration::ZERO,
        }
    }
}

impl Dialogue for ScriptedDialogue {
    async fn stream(
        &self,
        _req: &DialogueRequest,
        out: mpsc::Sender<String>,
    ) -> Result<(), String> {
        let reply = self.replies.lock().unwrap().pop_front().unwrap_or_else(|| {
            "Keelah, my link to the language model isn't connected yet. You'll have to imagine something clever.".into()
        });
        for (i, word) in reply.split(' ').enumerate() {
            let chunk = if i == 0 {
                word.to_string()
            } else {
                format!(" {word}")
            };
            out.send(chunk).await.map_err(|e| e.to_string())?;
            if !self.word_delay.is_zero() {
                tokio::time::sleep(self.word_delay).await;
            }
        }
        Ok(())
    }
}

/// Prints each sentence with its delivery instead of speaking it.
#[derive(Debug, Default)]
pub struct PrintVoice;

impl Voice for PrintVoice {
    async fn speak(&self, sentence: &str, delivery: &Delivery) -> Result<(), String> {
        println!("  voice {}: {sentence}", describe(delivery));
        Ok(())
    }
}

/// "[drunk 0.62, happy 0.30]", strongest first, skipping near-zero ones.
pub fn describe(delivery: &Delivery) -> String {
    let mut v: Vec<_> = delivery
        .vectors
        .iter()
        .filter(|(_, w)| **w >= 0.01)
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(a.1));
    let parts: Vec<String> = v.iter().map(|(n, w)| format!("{n} {w:.2}")).collect();
    if parts.is_empty() {
        format!("[{}]", delivery.emotion)
    } else {
        format!("[{}]", parts.join(", "))
    }
}
