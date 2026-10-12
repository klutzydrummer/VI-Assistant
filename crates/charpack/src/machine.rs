//! A character's state machine: continuous variables plus discrete states.
//!
//! Variables (like intoxication) move continuously: events nudge them and
//! they drift back toward a resting value every turn. A mixing table turns
//! the variables into control vector strengths for each model (the LLM,
//! the TTS engine), so behavior and voice blend smoothly. Discrete states
//! (sober, tipsy, drunk) are a readout derived from the variables, events
//! and timers through explicit transitions, and carry only what has to be
//! discrete: prompt overlays and flags that code enforces.
//!
//! Nothing here parses model output. A director (rules, a small model, or a
//! second LLM) picks events from the pack's vocabulary, and the machine
//! rejects anything the pack doesn't define.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Most transitions one settle can chain before stopping, so a badly
/// written pack can't loop forever.
const MAX_CHAIN: usize = 8;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineSpec {
    pub initial: String,
    #[serde(default, rename = "variable")]
    pub variables: Vec<VariableSpec>,
    #[serde(default, rename = "event")]
    pub events: Vec<EventSpec>,
    #[serde(rename = "state")]
    pub states: Vec<StateSpec>,
    #[serde(default, rename = "transition")]
    pub transitions: Vec<TransitionSpec>,
    #[serde(default, rename = "mix")]
    pub mixes: Vec<MixSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariableSpec {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub min: f32,
    #[serde(default = "one")]
    pub max: f32,
    /// Where the variable drifts back to when nothing pushes it.
    #[serde(default)]
    pub rest: f32,
    /// Starting value; defaults to `rest`.
    pub initial: Option<f32>,
    /// How far it moves toward `rest` at the end of each turn.
    #[serde(default)]
    pub decay_per_turn: f32,
}

/// One row of the mixing table: how strongly one reference control vector
/// is applied to one model, as a function of the variables.
///
/// strength = curve(clamp(bias + sum(input weight * variable), 0, 1)) * max
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixSpec {
    /// Which model receives the vector, e.g. "llm" or "tts".
    pub target: String,
    /// Reference control vector name, computed per character and model.
    pub vector: String,
    /// Variable id to its weight. Negative weights suppress.
    pub inputs: BTreeMap<String, f32>,
    #[serde(default)]
    pub bias: f32,
    #[serde(default)]
    pub curve: Curve,
    /// Strength at full input. Caps how hard a vector can push, since too
    /// much steering makes models incoherent.
    #[serde(default = "one")]
    pub max: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Curve {
    #[default]
    Linear,
    /// Eases in and out: small inputs barely register, large ones saturate.
    Smoothstep,
}

impl Curve {
    fn apply(self, x: f32) -> f32 {
        match self {
            Curve::Linear => x,
            Curve::Smoothstep => x * x * (3.0 - 2.0 * x),
        }
    }
}

fn one() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventSpec {
    pub id: String,
    /// Shown to the director so it knows when the event applies.
    pub description: String,
    /// Variable id to the amount added when the event happens.
    #[serde(default)]
    pub effects: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateSpec {
    pub id: String,
    pub description: String,
    /// Extra persona instructions while in this state, relative to the
    /// pack folder.
    pub prompt: Option<PathBuf>,
    /// Voice clips to use while in this state, for engines that can't take
    /// control vectors.
    pub voice: Option<String>,
    /// Checked by code, e.g. `no_risky_tools` blocks tools that send,
    /// delete or spend.
    #[serde(default)]
    pub flags: Vec<String>,
}

/// Fires when every trigger it lists holds. At least one is required.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionSpec {
    pub from: Vec<String>,
    pub to: String,
    /// An event that just happened.
    pub on: Option<String>,
    /// A variable condition.
    pub when: Option<Condition>,
    /// Turns spent in the current state.
    pub after_turns: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub var: String,
    /// Holds when the variable is at or above this.
    pub above: Option<f32>,
    /// Holds when the variable is below this.
    pub below: Option<f32>,
}

/// The live state of one conversation. Save it with the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub state: String,
    pub vars: BTreeMap<String, f32>,
    pub turns_in_state: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub from: String,
    pub to: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Rejected {
    #[error("unknown event {0:?}")]
    UnknownEvent(String),
    #[error("unknown state {0:?}")]
    UnknownState(String),
}

/// A validated [`MachineSpec`], ready to run.
#[derive(Debug, Clone)]
pub struct Machine {
    spec: MachineSpec,
}

impl Machine {
    pub fn new(spec: MachineSpec) -> Result<Self, Vec<String>> {
        let errors = validate(&spec);
        if errors.is_empty() {
            Ok(Self { spec })
        } else {
            Err(errors)
        }
    }

    pub fn spec(&self) -> &MachineSpec {
        &self.spec
    }

    pub fn start(&self) -> Snapshot {
        let vars = self
            .spec
            .variables
            .iter()
            .map(|v| (v.id.clone(), v.initial.unwrap_or(v.rest)))
            .collect();
        let mut snap = Snapshot {
            state: self.spec.initial.clone(),
            vars,
            turns_in_state: 0,
        };
        self.settle(&mut snap, None);
        snap
    }

    /// Apply an event's effects, then follow any transitions it causes.
    pub fn apply(&self, snap: &mut Snapshot, event: &str) -> Result<Vec<Change>, Rejected> {
        let spec = self
            .spec
            .events
            .iter()
            .find(|e| e.id == event)
            .ok_or_else(|| Rejected::UnknownEvent(event.into()))?;
        for (var, delta) in &spec.effects {
            let v = self.variable(var);
            let value = snap.vars.get(var).copied().unwrap_or(v.rest) + delta;
            snap.vars.insert(var.clone(), value.clamp(v.min, v.max));
        }
        Ok(self.settle(snap, Some(event)))
    }

    /// Jump straight to a state, e.g. when the user overrides it.
    /// Variables are left alone.
    pub fn force(&self, snap: &mut Snapshot, state: &str) -> Result<(), Rejected> {
        if self.state(state).is_none() {
            return Err(Rejected::UnknownState(state.into()));
        }
        snap.state = state.into();
        snap.turns_in_state = 0;
        Ok(())
    }

    /// Call once after each of the character's replies: variables drift
    /// toward rest, timers advance, and transitions are re-checked.
    pub fn end_turn(&self, snap: &mut Snapshot) -> Vec<Change> {
        for v in &self.spec.variables {
            let value = snap.vars.entry(v.id.clone()).or_insert(v.rest);
            *value = if *value > v.rest {
                (*value - v.decay_per_turn).max(v.rest)
            } else {
                (*value + v.decay_per_turn).min(v.rest)
            };
        }
        snap.turns_in_state += 1;
        self.settle(snap, None)
    }

    pub fn state(&self, id: &str) -> Option<&StateSpec> {
        self.spec.states.iter().find(|s| s.id == id)
    }

    pub fn flags(&self, snap: &Snapshot) -> BTreeSet<&str> {
        self.state(&snap.state)
            .map(|s| s.flags.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Models the mixing table sends vectors to, e.g. "llm" and "tts".
    pub fn targets(&self) -> BTreeSet<&str> {
        self.spec.mixes.iter().map(|m| m.target.as_str()).collect()
    }

    /// Reference vector name to strength for one model, from the current
    /// variables. Zero-strength vectors are included so callers can clear
    /// them.
    pub fn mix(&self, snap: &Snapshot, target: &str) -> BTreeMap<String, f32> {
        self.spec
            .mixes
            .iter()
            .filter(|m| m.target == target)
            .map(|m| {
                let sum: f32 = m
                    .inputs
                    .iter()
                    .map(|(var, w)| w * snap.vars.get(var).copied().unwrap_or_default())
                    .sum();
                let x = (m.bias + sum).clamp(0.0, 1.0);
                (m.vector.clone(), m.curve.apply(x) * m.max)
            })
            .collect()
    }

    fn variable(&self, id: &str) -> &VariableSpec {
        self.spec
            .variables
            .iter()
            .find(|v| v.id == id)
            .expect("validated: effects only name known variables")
    }

    fn settle(&self, snap: &mut Snapshot, mut event: Option<&str>) -> Vec<Change> {
        let mut changes = Vec::new();
        for _ in 0..MAX_CHAIN {
            let Some(t) = self
                .spec
                .transitions
                .iter()
                .find(|t| t.from.contains(&snap.state) && self.fires(t, snap, event))
            else {
                break;
            };
            changes.push(Change {
                from: std::mem::replace(&mut snap.state, t.to.clone()),
                to: t.to.clone(),
            });
            snap.turns_in_state = 0;
            // An event triggers at most one transition; chains after it
            // follow conditions only.
            event = None;
        }
        changes
    }

    fn fires(&self, t: &TransitionSpec, snap: &Snapshot, event: Option<&str>) -> bool {
        let on = t.on.as_deref().is_none_or(|e| event == Some(e));
        let when = t.when.as_ref().is_none_or(|c| {
            let value = snap.vars.get(&c.var).copied().unwrap_or_default();
            c.above.is_none_or(|a| value >= a) && c.below.is_none_or(|b| value < b)
        });
        let after = t.after_turns.is_none_or(|n| snap.turns_in_state >= n);
        on && when && after
    }
}

fn validate(spec: &MachineSpec) -> Vec<String> {
    let mut errors = Vec::new();
    let mut unique = |kind: &str, ids: Vec<&str>| -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        for id in ids {
            if !seen.insert(id.to_string()) {
                errors.push(format!("{kind} {id:?} is defined twice"));
            }
        }
        seen
    };
    let states = unique("state", spec.states.iter().map(|s| s.id.as_str()).collect());
    let events = unique("event", spec.events.iter().map(|e| e.id.as_str()).collect());
    let vars = unique(
        "variable",
        spec.variables.iter().map(|v| v.id.as_str()).collect(),
    );

    if !states.contains(&spec.initial) {
        errors.push(format!("initial state {:?} is not defined", spec.initial));
    }
    for v in &spec.variables {
        if v.min >= v.max {
            errors.push(format!("variable {:?}: min must be below max", v.id));
        }
        let in_range = |x: f32| (v.min..=v.max).contains(&x);
        if !in_range(v.rest) || !v.initial.is_none_or(in_range) {
            errors.push(format!(
                "variable {:?}: rest and initial must be within min..max",
                v.id
            ));
        }
        if v.decay_per_turn < 0.0 {
            errors.push(format!("variable {:?}: decay_per_turn is negative", v.id));
        }
    }
    for e in &spec.events {
        for var in e.effects.keys() {
            if !vars.contains(var) {
                errors.push(format!("event {:?} changes unknown variable {var:?}", e.id));
            }
        }
    }
    for (i, t) in spec.transitions.iter().enumerate() {
        let name = format!("transition {} ({} -> {})", i + 1, t.from.join("|"), t.to);
        for s in t.from.iter().chain([&t.to]) {
            if !states.contains(s) {
                errors.push(format!("{name}: unknown state {s:?}"));
            }
        }
        if t.on.is_none() && t.when.is_none() && t.after_turns.is_none() {
            errors.push(format!("{name}: needs on, when or after_turns"));
        }
        if let Some(e) = &t.on {
            if !events.contains(e) {
                errors.push(format!("{name}: unknown event {e:?}"));
            }
        }
        if let Some(c) = &t.when {
            if !vars.contains(&c.var) {
                errors.push(format!("{name}: unknown variable {:?}", c.var));
            }
            if c.above.is_none() && c.below.is_none() {
                errors.push(format!("{name}: when needs above or below"));
            }
        }
    }

    let mut mixes = BTreeSet::new();
    for m in &spec.mixes {
        let name = format!("mix {}/{}", m.target, m.vector);
        if !mixes.insert((m.target.as_str(), m.vector.as_str())) {
            errors.push(format!("{name} is defined twice"));
        }
        if m.inputs.is_empty() {
            errors.push(format!("{name} has no inputs"));
        }
        for var in m.inputs.keys() {
            if !vars.contains(var) {
                errors.push(format!("{name}: unknown variable {var:?}"));
            }
        }
        if m.max <= 0.0 {
            errors.push(format!("{name}: max must be positive"));
        }
    }

    // Every state should be reachable from the initial one.
    let mut reached = BTreeSet::from([spec.initial.clone()]);
    loop {
        let before = reached.len();
        for t in &spec.transitions {
            if t.from.iter().any(|s| reached.contains(s)) {
                reached.insert(t.to.clone());
            }
        }
        if reached.len() == before {
            break;
        }
    }
    for s in &states {
        if !reached.contains(s) {
            errors.push(format!("state {s:?} can never be reached"));
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = r#"
initial = "sober"

[[variable]]
id = "intoxication"
description = "How drunk"
decay_per_turn = 0.1

[[variable]]
id = "joy"
description = "Happiness"

[[event]]
id = "drink"
description = "Has a drink"
effects = { intoxication = 0.35 }

[[event]]
id = "sleep"
description = "Sleeps it off"

[[state]]
id = "sober"
description = "Clear-headed"

[[state]]
id = "tipsy"
description = "A little drunk"
flags = ["impaired"]

[[state]]
id = "drunk"
description = "Very drunk"
flags = ["impaired", "no_risky_tools"]

[[state]]
id = "hungover"
description = "Paying for it"

[[transition]]
from = ["sober"]
when = { var = "intoxication", above = 0.25 }
to = "tipsy"

[[transition]]
from = ["tipsy"]
when = { var = "intoxication", above = 0.6 }
to = "drunk"

[[transition]]
from = ["tipsy"]
when = { var = "intoxication", below = 0.15 }
to = "sober"

[[transition]]
from = ["drunk"]
when = { var = "intoxication", below = 0.35 }
to = "hungover"

[[transition]]
from = ["hungover"]
after_turns = 3
to = "sober"

[[transition]]
from = ["hungover"]
on = "sleep"
to = "sober"

[[mix]]
target = "tts"
vector = "drunk"
inputs = { intoxication = 1.0 }

[[mix]]
target = "llm"
vector = "drunk"
inputs = { intoxication = 1.0 }
max = 0.8

[[mix]]
target = "llm"
vector = "affectionate"
inputs = { intoxication = 0.5, joy = 0.5 }
curve = "smoothstep"
"#;

    fn machine() -> Machine {
        Machine::new(toml::from_str(SPEC).unwrap()).unwrap()
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn drinks_raise_state_and_voice_weight() {
        let m = machine();
        let mut s = m.start();
        assert_eq!(s.state, "sober");

        let changes = m.apply(&mut s, "drink").unwrap();
        assert_eq!(
            changes,
            vec![Change {
                from: "sober".into(),
                to: "tipsy".into()
            }]
        );
        assert!(approx(m.mix(&s, "tts")["drunk"], 0.35));

        m.apply(&mut s, "drink").unwrap();
        assert_eq!(s.state, "drunk");
        assert!(m.flags(&s).contains("no_risky_tools"));
    }

    #[test]
    fn mixing_table_blends_variables_per_target() {
        let m = machine();
        let mut s = m.start();
        assert_eq!(m.targets(), BTreeSet::from(["llm", "tts"]));
        assert!(approx(m.mix(&s, "llm")["affectionate"], 0.0));

        s.vars.insert("intoxication".into(), 0.5);
        s.vars.insert("joy".into(), 0.5);
        let llm = m.mix(&s, "llm");
        assert!(approx(llm["drunk"], 0.4)); // 0.5, capped by max 0.8
        assert!(approx(llm["affectionate"], 0.5)); // smoothstep(0.5) = 0.5

        s.vars.insert("joy".into(), 1.0);
        let x: f32 = 0.75;
        assert!(approx(
            m.mix(&s, "llm")["affectionate"],
            x * x * (3.0 - 2.0 * x)
        ));
        assert!(m.mix(&s, "nope").is_empty());
    }

    #[test]
    fn intoxication_caps_at_max() {
        let m = machine();
        let mut s = m.start();
        for _ in 0..5 {
            m.apply(&mut s, "drink").unwrap();
        }
        assert!(approx(s.vars["intoxication"], 1.0));
    }

    #[test]
    fn sobering_up_goes_through_hangover_then_times_out() {
        let m = machine();
        let mut s = m.start();
        m.apply(&mut s, "drink").unwrap();
        m.apply(&mut s, "drink").unwrap();
        assert_eq!(s.state, "drunk");

        // 0.7 decays by 0.1 a turn: below 0.35 after four turns.
        for _ in 0..3 {
            m.end_turn(&mut s);
            assert_eq!(s.state, "drunk");
        }
        m.end_turn(&mut s);
        assert_eq!(s.state, "hungover");

        m.end_turn(&mut s);
        m.end_turn(&mut s);
        assert_eq!(s.state, "hungover");
        m.end_turn(&mut s);
        assert_eq!(s.state, "sober");
    }

    #[test]
    fn events_trigger_their_transitions() {
        let m = machine();
        let mut s = m.start();
        m.force(&mut s, "hungover").unwrap();
        m.apply(&mut s, "sleep").unwrap();
        assert_eq!(s.state, "sober");
    }

    #[test]
    fn rejects_unknown_events_and_states() {
        let m = machine();
        let mut s = m.start();
        let before = s.clone();
        assert_eq!(
            m.apply(&mut s, "dance"),
            Err(Rejected::UnknownEvent("dance".into()))
        );
        assert_eq!(
            m.force(&mut s, "flying"),
            Err(Rejected::UnknownState("flying".into()))
        );
        assert_eq!(s, before);
    }

    #[test]
    fn validation_catches_broken_graphs() {
        let broken = SPEC
            .replace("on = \"sleep\"", "on = \"nap\"")
            .replace(", above = 0.6", "")
            .replace("joy = 0.5 }", "joy = 0.5, mood = 1.0 }")
            + "\n[[state]]\nid = \"lost\"\ndescription = \"Nowhere\"\n";
        let mut spec: MachineSpec = toml::from_str(&broken).unwrap();
        spec.transitions[0].to = "tipsy".into();
        let errors = Machine::new(spec).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("unknown event \"nap\"")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.contains("when needs above or below")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.contains("\"lost\" can never be reached")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.contains("mix llm/affectionate: unknown variable \"mood\"")),
            "{errors:?}"
        );
    }

    #[test]
    fn snapshot_round_trips() {
        let m = machine();
        let mut s = m.start();
        m.apply(&mut s, "drink").unwrap();
        let restored: Snapshot = toml::from_str(&toml::to_string(&s).unwrap()).unwrap();
        assert_eq!(restored, s);
    }
}
