//! Stage 3's harness: the chat check, run on `vi chat`'s own steering path
//! ([`Llama::generate`] and [`VectorBank::blend`]: unsteered prompt, steered
//! last prompt token and reply, each vector on its own layers), so the
//! strengths it settles mean the same in `vi chat`. llama-server steers
//! differently (every token, one layer range for all vectors) and isn't
//! used.
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
//!  "scheme": "single:32", "max_tokens": 200,
//!  "judge": [{"role": "user", "content": "Rate the warmth of: {reply}"}]}
//! ```
//!
//! `scheme` is `single:L`, `band:A-B` (telescoping), `band-k:A-B` or
//! `band-sqrtk:A-B`, built from the folder's `natural/` vectors; without
//! it the folder's own bank and `vectors.toml` are used, exactly as
//! `vi chat` would load them. Results are appended as JSON lines, and a
//! rerun skips ids already there.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::gguf::Gguf;
use crate::llama::{Llama, Sampling, Stop};
use crate::vectors::{directions, Limits, Scheme, Vector, VectorBank, MODEL_SHA_KEY};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
}

pub struct Checker {
    llama: Llama,
    dir: PathBuf,
    natural: BTreeMap<String, BTreeMap<u32, Vec<f32>>>,
    banks: HashMap<String, VectorBank>,
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
    pub fn new(llama: Llama, dir: &Path, model_sha256: &str) -> Result<Self, String> {
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
        let prompt = self.llama.format_chat(&messages)?;
        let max = t.max_tokens.unwrap_or(200);
        let sampling = match t.temperature {
            Some(temp) if temp > 0.0 => Sampling {
                temperature: temp,
                max_tokens: max,
                seed: t.seed.unwrap_or(0xFFFF_FFFF),
                ..Sampling::default()
            },
            _ => Sampling::greedy(max),
        };
        let g = self
            .llama
            .generate(&prompt, blend.data.as_deref(), &sampling, |_| true)?;
        let (thinking, reply) = split_thinking(&g.text);
        let judge = match &t.judge {
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
                let jp = self.llama.format_chat(&refs)?;
                let sampling = Sampling::greedy(t.judge_max_tokens.unwrap_or(200));
                Some(self.llama.generate(&jp, None, &sampling, |_| true)?.text)
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
        })
    }
}

/// Split a reply into the thinking before `</think>` and what she says.
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    match text.split_once("</think>") {
        Some((think, reply)) => {
            let think = think.trim_start().trim_start_matches("<think>").trim();
            (Some(think.to_string()), reply.trim().to_string())
        }
        None => (None, text.to_string()),
    }
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
    use super::split_thinking;

    #[test]
    fn splits_thinking_from_the_reply() {
        assert_eq!(
            split_thinking("<think>\nShe'd be glad.\n</think>\n\nKeelah!"),
            (Some("She'd be glad.".into()), "Keelah!".into())
        );
        assert_eq!(split_thinking("Keelah!"), (None, "Keelah!".into()));
    }
}
