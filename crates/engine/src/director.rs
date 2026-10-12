//! Directors decide what happened (events for the state machine) and how
//! each sentence should sound (delivery for the voice). They never write or
//! change dialogue.

use std::collections::BTreeMap;
use std::future::Future;

use charpack::machine::{Machine, MachineSpec, Snapshot};

/// Voice direction for one sentence, engine-neutral. Voice adapters turn
/// it into whatever their engine takes.
#[derive(Debug, Clone, PartialEq)]
pub struct Delivery {
    /// Control vector name to strength, from the mixing table.
    pub vectors: BTreeMap<String, f32>,
    /// Strongest emotion, for engines that pick a reference clip instead.
    pub emotion: String,
    /// Free-text style hint, e.g. "slurred, giggling".
    pub style: Option<String>,
    /// Speaking rate, 1.0 is normal.
    pub pace: f32,
}

pub struct TurnContext<'a> {
    pub user_message: &'a str,
    pub machine: &'a Machine,
    pub snapshot: &'a Snapshot,
}

pub struct SentenceContext<'a> {
    pub sentence: &'a str,
    /// Position in the reply, from 0.
    pub index: usize,
    /// The turn's voice strengths from the mixing table.
    pub base: &'a BTreeMap<String, f32>,
    /// Emotion to use when nothing stands out.
    pub default_emotion: &'a str,
}

pub trait Director {
    /// Event ids the user's message triggers. Ids the pack doesn't define
    /// are rejected by the state machine, so a director can't break it.
    fn on_user_turn(&self, ctx: &TurnContext<'_>) -> impl Future<Output = Vec<String>>;

    /// How one sentence of the reply should sound.
    fn on_sentence(&self, ctx: &SentenceContext<'_>) -> impl Future<Output = Delivery>;
}

/// Below this, a vector doesn't count as the sentence's emotion.
const EMOTION_FLOOR: f32 = 0.15;

/// The model-free director: matches each event's cues against the user's
/// message, and voices every sentence with the turn's mixing table output.
/// Always available, and the fallback when a model director times out.
#[derive(Debug, Clone)]
pub struct RulesDirector {
    /// (event id, cue normalized for matching)
    cues: Vec<(String, String)>,
}

impl RulesDirector {
    pub fn new(spec: &MachineSpec) -> Self {
        let cues = spec
            .events
            .iter()
            .flat_map(|e| e.cues.iter().map(|c| (e.id.clone(), words(c))))
            .filter(|(_, cue)| cue.trim() != "")
            .collect();
        Self { cues }
    }
}

impl Director for RulesDirector {
    async fn on_user_turn(&self, ctx: &TurnContext<'_>) -> Vec<String> {
        let message = words(ctx.user_message);
        let mut events: Vec<String> = Vec::new();
        for (event, cue) in &self.cues {
            if message.contains(cue.as_str()) && !events.contains(event) {
                events.push(event.clone());
            }
        }
        events
    }

    async fn on_sentence(&self, ctx: &SentenceContext<'_>) -> Delivery {
        let emotion = ctx
            .base
            .iter()
            .filter(|(_, w)| **w >= EMOTION_FLOOR)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| ctx.default_emotion.to_string());
        Delivery {
            vectors: ctx.base.clone(),
            emotion,
            style: None,
            pace: 1.0,
        }
    }
}

/// Lowercase words separated and surrounded by single spaces, so a cue
/// matches whole words only: " a drink " is found in " have a drink now "
/// but " rael " is not found in " israel ".
fn words(s: &str) -> String {
    let cleaned: String = s
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect();
    format!(
        " {} ",
        cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = r#"
initial = "calm"

[[variable]]
id = "joy"
description = "Happiness"

[[event]]
id = "drink"
description = "Has a drink"
cues = ["a drink", "brandy"]

[[event]]
id = "homesick"
description = "Thinks of home"
cues = ["Rael", "the fleet"]

[[state]]
id = "calm"
description = "Calm"
"#;

    fn ctx_events(message: &str) -> Vec<String> {
        let spec: MachineSpec = toml::from_str(SPEC).unwrap();
        let director = RulesDirector::new(&spec);
        let machine = Machine::new(spec).unwrap();
        let snapshot = machine.start();
        let ctx = TurnContext {
            user_message: message,
            machine: &machine,
            snapshot: &snapshot,
        };
        futures_block(director.on_user_turn(&ctx))
    }

    fn futures_block<F: Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }

    #[test]
    fn cues_match_whole_words_case_insensitively() {
        assert_eq!(ctx_events("Here, have a drink!"), vec!["drink"]);
        assert_eq!(
            ctx_events("Tell me about RAEL and the Fleet."),
            vec!["homesick"]
        );
        assert!(ctx_events("I visited Israel.").is_empty());
        assert_eq!(
            ctx_events("Brandy, then a drink, then the fleet"),
            vec!["drink", "homesick"]
        );
    }

    #[test]
    fn sentence_emotion_is_the_strongest_vector_above_the_floor() {
        let spec: MachineSpec = toml::from_str(SPEC).unwrap();
        let director = RulesDirector::new(&spec);
        let base = BTreeMap::from([
            ("sad".to_string(), 0.4),
            ("drunk".to_string(), 0.7),
            ("happy".to_string(), 0.1),
        ]);
        let ctx = SentenceContext {
            sentence: "Hi.",
            index: 0,
            base: &base,
            default_emotion: "neutral",
        };
        let d = futures_block(director.on_sentence(&ctx));
        assert_eq!(d.emotion, "drunk");
        assert_eq!(d.vectors, base);

        let quiet = BTreeMap::from([("sad".to_string(), 0.05)]);
        let ctx = SentenceContext {
            base: &quiet,
            ..ctx
        };
        assert_eq!(futures_block(director.on_sentence(&ctx)).emotion, "neutral");
    }
}
