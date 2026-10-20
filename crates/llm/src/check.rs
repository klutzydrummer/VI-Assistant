//! Stage 3's harness: the chat check, run on `vi chat`'s own steering path
//! ([`Llama::generate_scoped`] and [`VectorBank::blend`]: unsteered
//! prompt, then steered, each vector on its own layers), so the strengths
//! it settles mean the same in `vi chat`. llama-server steers differently
//! (every token, one layer range for all vectors) and isn't used.
//!
//! Thinking is on by default: the prompt ends where the model starts its
//! think block. Off appends an empty one (`<think>\n\n</think>\n\n`).
//! The steering scope is `reply` by default: the prompt and the thinking
//! stay unsteered and steering starts at `</think>`, where the vectors were
//! measured; `all` steers from the last prompt token on, thinking included,
//! as `vi chat` does today. Both can be set per trial.
//!
//! The step 5 thread owns what is checked: the prompts, the judge's
//! questions and the pass rule. This runs a plan of trials, each a chat
//! with a blend and a layer scheme, optionally followed by the same model
//! unsteered answering a judge prompt about the reply. When the model
//! thinks first, the thinking is kept apart and the judge sees only what
//! she says after `</think>`. Blends and schemes
//! change per trial without reloading the model.
//!
//! A plan is JSON lines:
//!
//! ```json
//! {"id": "warmth-1", "messages": [{"role": "system", "content": "..."},
//!   {"role": "user", "content": "..."}], "steer": {"warmth": 1.0},
//!  "scheme": "single:32", "max_tokens": 600, "thinking": "on", "scope": "reply",
//!  "judge": [{"role": "user", "content": "Rate the warmth of: {reply}"}]}
//! ```
//!
//! `scheme` is `single:L`, `band:A-B` (telescoping), `band-k:A-B` or
//! `band-sqrtk:A-B`, built from the folder's `natural/` vectors; without
//! it the folder's own bank and `vectors.toml` are used, exactly as
//! `vi chat` would load them. `max_tokens` covers the thinking too.
//! `thinking` takes `"on"`/`"off"` or `true`/`false`; `steer_scope` (or
//! `scope`) takes `"reply"`/`"all"`. The judge thinks only with
//! `judge_thinking` (off by default), and sees only the text after
//! `</think>`.
//!
//! A trial whose thinking never closed (out of tokens) has no reply, and in
//! the `reply` scope was never steered: its `outcome` is `truncated` and it
//! isn't judged. One in the `reply` scope that never said `</think>` for any
//! other reason is `never_steered`, also unjudged. Neither counts as a
//! steering result; [`summarize`] gives their rate per mood. Results are
//! appended as JSON lines, and a rerun skips ids already there.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::gguf::Gguf;
use crate::llama::{Llama, Sampling, Scope, SteerStart, Stop, EMPTY_THINK};
use crate::vectors::{directions, Limits, Scheme, Vector, VectorBank, MODEL_SHA_KEY};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Whether the model thinks before it replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Thinking {
    #[default]
    On,
    /// The prompt closes an empty think block.
    Off,
}

impl Thinking {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            _ => Err(format!("thinking is on or off, not {s:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

impl<'de> Deserialize<'de> for Thinking {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Text(String),
        }
        match Raw::deserialize(d)? {
            Raw::Bool(true) => Ok(Self::On),
            Raw::Bool(false) => Ok(Self::Off),
            Raw::Text(s) => Self::parse(&s).map_err(serde::de::Error::custom),
        }
    }
}

/// The run's settings, which a trial may override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckDefaults {
    pub thinking: Thinking,
    pub scope: Scope,
}

impl Default for CheckDefaults {
    fn default() -> Self {
        Self {
            thinking: Thinking::On,
            scope: Scope::Reply,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub steer: BTreeMap<String, f32>,
    #[serde(default)]
    pub scheme: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Greedy when absent.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    /// Asked of the same model, unsteered, with `{reply}` replaced.
    #[serde(default)]
    pub judge: Option<Vec<ChatMessage>>,
    #[serde(default)]
    pub judge_max_tokens: Option<usize>,
    #[serde(default)]
    pub thinking: Option<Thinking>,
    /// Steering scope, `reply` or `all`.
    #[serde(default, alias = "steer_scope")]
    pub scope: Option<Scope>,
    /// Whether the judge thinks first; off by default.
    #[serde(default)]
    pub judge_thinking: Option<Thinking>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialResult {
    pub id: String,
    pub scheme: String,
    pub applied: BTreeMap<String, f32>,
    pub missing: Vec<String>,
    /// The reply after any `</think>`: what she says, and what the judge sees.
    pub reply: String,
    /// The thinking before `</think>`, when the model thought first.
    #[serde(default)]
    pub thinking: Option<String>,
    pub reply_tokens: usize,
    pub stopped: String,
    pub judge: Option<String>,
    pub seconds: f64,
    /// `on` or `off`.
    #[serde(default)]
    pub think: String,
    /// `reply` or `all`.
    #[serde(default)]
    pub scope: String,
    /// Where steering started: `prompt` (at the token named by
    /// `steer_start_token`), `reply` (at the sampled `</think>`, reply token
    /// `steer_start_token`), or `never` (`</think>` didn't come).
    #[serde(default)]
    pub steer_start: String,
    #[serde(default)]
    pub steer_start_token: Option<usize>,
    /// `reply`, or `truncated` (thinking never closed) or `never_steered`
    /// (`reply` scope, no `</think>`): those two aren't judged and aren't
    /// steering results.
    #[serde(default)]
    pub outcome: String,
    #[serde(default)]
    pub judge_think: String,
}

pub struct Checker {
    llama: Llama,
    dir: PathBuf,
    natural: BTreeMap<String, BTreeMap<u32, Vec<f32>>>,
    banks: HashMap<String, VectorBank>,
    defaults: CheckDefaults,
}

fn file_model_sha(g: &Gguf) -> Option<String> {
    g.metadata
        .get(MODEL_SHA_KEY)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

impl Checker {
    /// `dir` is a method folder written by the fit (`.../joint`). Refuses
    /// vectors built for another model file.
    pub fn new(
        llama: Llama,
        dir: &Path,
        model_sha256: &str,
        defaults: CheckDefaults,
    ) -> Result<Self, String> {
        let mut natural = BTreeMap::new();
        let nat_dir = dir.join("natural");
        if nat_dir.is_dir() {
            let mut paths: Vec<_> = fs::read_dir(&nat_dir)
                .map_err(|e| format!("{}: {e}", nat_dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
                .collect();
            paths.sort();
            for p in paths {
                let bytes = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                let g = Gguf::parse(&bytes).map_err(|e| format!("{}: {e}", p.display()))?;
                if let Some(sha) = file_model_sha(&g) {
                    if sha != model_sha256 {
                        return Err(format!(
                            "{} was built for model {sha}, not this one ({model_sha256})",
                            p.display()
                        ));
                    }
                }
                let name = p.file_stem().unwrap().to_string_lossy().into_owned();
                natural.insert(name, directions(&g)?);
            }
        }
        Ok(Self {
            llama,
            dir: dir.to_path_buf(),
            natural,
            banks: HashMap::new(),
            defaults,
        })
    }

    fn bank(&mut self, scheme: Option<&str>) -> Result<&VectorBank, String> {
        let key = scheme.unwrap_or("vectors.toml").to_string();
        if !self.banks.contains_key(&key) {
            let (e, l) = (self.llama.n_embd(), self.llama.n_layer());
            let bank = match scheme {
                None => VectorBank::load_dir(&self.dir, e, l)?,
                Some(s) => {
                    let scheme = Scheme::parse(s)?;
                    let mut bank = VectorBank::empty(e, l);
                    for (name, dirs) in &self.natural {
                        let applied = scheme.apply(dirs).map_err(|e| format!("{name}: {e}"))?;
                        let g = crate::vectors::vector_gguf(&applied, l, "", &[]);
                        let limits = Limits {
                            scale: 1.0,
                            cap: f32::INFINITY,
                            layers: Some(scheme.layers()),
                        };
                        let v = Vector::from_gguf(&g, e, l, limits)
                            .map_err(|e| format!("{name}: {e}"))?;
                        bank.insert(name.clone(), v)?;
                    }
                    bank
                }
            };
            self.banks.insert(key.clone(), bank);
        }
        Ok(&self.banks[&key])
    }

    pub fn run(&mut self, t: &Trial) -> Result<TrialResult, String> {
        let start = Instant::now();
        let blend = self.bank(t.scheme.as_deref())?.blend(&t.steer);
        let messages: Vec<(&str, &str)> = t
            .messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        let think = t.thinking.unwrap_or(self.defaults.thinking);
        let scope = t.scope.unwrap_or(self.defaults.scope);
        let prompt = with_thinking(self.llama.format_chat(&messages)?, think);
        let max = t.max_tokens.unwrap_or(match think {
            Thinking::On => 1000,
            Thinking::Off => 200,
        });
        let sampling = match t.temperature {
            Some(temp) if temp > 0.0 => Sampling {
                temperature: temp,
                max_tokens: max,
                seed: t.seed.unwrap_or(0xFFFF_FFFF),
                ..Sampling::default()
            },
            _ => Sampling::greedy(max),
        };
        let g =
            self.llama
                .generate_scoped(&prompt, blend.data.as_deref(), scope, &sampling, |_| true)?;
        let (thinking, reply) = split_thinking(&g.text);
        let open_thinking =
            g.text.trim_start().starts_with("<think>") && !g.text.contains("</think>");
        let outcome = if open_thinking && g.stop == Stop::MaxTokens {
            "truncated"
        } else if scope == Scope::Reply && g.steer_start == SteerStart::Never {
            if g.stop == Stop::MaxTokens {
                "truncated"
            } else {
                "never_steered"
            }
        } else {
            "reply"
        };
        let judge_think = t.judge_thinking.unwrap_or(Thinking::Off);
        let judge = match &t.judge {
            Some(_) if outcome != "reply" => None,
            None => None,
            Some(js) => {
                let filled: Vec<(String, String)> = js
                    .iter()
                    .map(|m| (m.role.clone(), m.content.replace("{reply}", &reply)))
                    .collect();
                let refs: Vec<(&str, &str)> = filled
                    .iter()
                    .map(|(r, c)| (r.as_str(), c.as_str()))
                    .collect();
                let jp = with_thinking(self.llama.format_chat(&refs)?, judge_think);
                let sampling = Sampling::greedy(t.judge_max_tokens.unwrap_or(max));
                let text = self.llama.generate(&jp, None, &sampling, |_| true)?.text;
                Some(split_thinking(&text).1)
            }
        };
        Ok(TrialResult {
            id: t.id.clone(),
            scheme: t.scheme.clone().unwrap_or_else(|| "vectors.toml".into()),
            applied: blend.applied,
            missing: blend.missing,
            reply,
            thinking,
            reply_tokens: g.tokens.len(),
            stopped: match g.stop {
                Stop::EndOfTurn => "end_of_turn",
                Stop::MaxTokens => "max_tokens",
                Stop::Cancelled => "cancelled",
            }
            .into(),
            judge,
            seconds: start.elapsed().as_secs_f64(),
            think: think.name().into(),
            scope: scope.name().into(),
            steer_start: match g.steer_start {
                SteerStart::Prompt(_) => "prompt",
                SteerStart::Reply(_) => "reply",
                SteerStart::Never => "never",
            }
            .into(),
            steer_start_token: match g.steer_start {
                SteerStart::Prompt(i) | SteerStart::Reply(i) => Some(i),
                SteerStart::Never => None,
            },
            outcome: outcome.into(),
            judge_think: judge_think.name().into(),
        })
    }
}

fn with_thinking(prompt: String, thinking: Thinking) -> String {
    match thinking {
        Thinking::On => prompt,
        Thinking::Off => prompt + EMPTY_THINK,
    }
}

/// Split a reply into the thinking before `</think>` and what she says.
/// Thinking that never closed (out of tokens) leaves no reply.
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    match text.split_once("</think>") {
        Some((think, reply)) => {
            let think = think.trim_start().trim_start_matches("<think>").trim();
            (Some(think.to_string()), reply.trim().to_string())
        }
        None => match text.trim_start().strip_prefix("<think>") {
            Some(think) => (Some(think.trim().to_string()), String::new()),
            None => (None, text.to_string()),
        },
    }
}

/// Per mood (the trial's `steer` names; `(unsteered)` for none), how many
/// trials there were and how many had no steered reply.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MoodOutcomes {
    pub trials: usize,
    pub truncated: usize,
    pub never_steered: usize,
}

pub fn summarize(results: &[TrialResult]) -> BTreeMap<String, MoodOutcomes> {
    let mut out: BTreeMap<String, MoodOutcomes> = BTreeMap::new();
    for r in results {
        let mut moods: Vec<&String> = r.applied.keys().chain(&r.missing).collect();
        moods.sort();
        moods.dedup();
        let unsteered = "(unsteered)".to_string();
        if moods.is_empty() {
            moods.push(&unsteered);
        }
        for m in moods {
            let c = out.entry(m.clone()).or_default();
            c.trials += 1;
            match r.outcome.as_str() {
                "truncated" => c.truncated += 1,
                "never_steered" => c.never_steered += 1,
                _ => {}
            }
        }
    }
    out
}

/// Every result in a results file.
pub fn read_results(path: &Path) -> Result<Vec<TrialResult>, String> {
    Ok(crate::fit::read_jsonl::<TrialResult>(path)?.0)
}

/// Run every trial in `plan` not already in `out`, appending results.
pub fn run_plan(
    checker: &mut Checker,
    plan: &Path,
    out: &Path,
    mut progress: impl FnMut(&TrialResult),
) -> Result<usize, String> {
    use std::io::Write;
    let (trials, _) = crate::fit::read_jsonl::<Trial>(plan)?;
    let done: std::collections::HashSet<String> = if out.exists() {
        crate::fit::read_jsonl::<TrialResult>(out)?
            .0
            .into_iter()
            .map(|r| r.id)
            .collect()
    } else {
        Default::default()
    };
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let mut n = 0;
    for t in trials.iter().filter(|t| !done.contains(&t.id)) {
        let r = checker.run(t).map_err(|e| format!("trial {}: {e}", t.id))?;
        let mut line = serde_json::to_string(&r).unwrap();
        line.push('\n');
        file.write_all(line.as_bytes())
            .and_then(|_| file.sync_data())
            .map_err(|e| format!("{}: {e}", out.display()))?;
        progress(&r);
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::{split_thinking, Thinking, Trial};

    #[test]
    fn reads_step_5s_field_names() {
        let t: Trial = serde_json::from_str(
            r#"{"id": "a", "messages": [], "thinking": true, "steer_scope": "all",
                "judge_thinking": false}"#,
        )
        .unwrap();
        assert_eq!(t.thinking, Some(Thinking::On));
        assert_eq!(t.scope, Some(crate::llama::Scope::All));
        assert_eq!(t.judge_thinking, Some(Thinking::Off));
        let t: Trial = serde_json::from_str(
            r#"{"id": "b", "messages": [], "thinking": "off", "scope": "reply"}"#,
        )
        .unwrap();
        assert_eq!(t.thinking, Some(Thinking::Off));
        assert_eq!(t.scope, Some(crate::llama::Scope::Reply));
    }

    #[test]
    fn splits_thinking_from_the_reply() {
        assert_eq!(
            split_thinking("<think>\nShe'd be glad.\n</think>\n\nKeelah!"),
            (Some("She'd be glad.".into()), "Keelah!".into())
        );
        assert_eq!(split_thinking("Keelah!"), (None, "Keelah!".into()));
        assert_eq!(
            split_thinking("<think>\nStill thinking"),
            (Some("Still thinking".into()), String::new())
        );
    }
}
