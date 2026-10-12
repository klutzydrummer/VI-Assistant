//! Assembling the dialogue model's system prompt from the pack and the
//! current state. State reaches the model only as plain language.

use charpack::machine::{Machine, Snapshot};
use charpack::Pack;

/// Feelings weaker than this are left out of the mood line.
const FEELING_FLOOR: f32 = 0.15;

pub fn system_prompt(pack: &Pack, machine: &Machine, snap: &Snapshot) -> String {
    let mut parts = vec![pack.persona.trim().to_string()];

    if !pack.agents.is_empty() {
        let mut drones = String::from("## Your drones\n");
        for a in &pack.agents {
            drones.push_str(&format!(
                "- {} ({}): {} {}\n",
                a.name, a.full_name, a.role, a.temperament
            ));
        }
        parts.push(drones.trim_end().to_string());
    }

    if let Some(overlay) = pack.state_prompts.get(&snap.state) {
        parts.push(overlay.trim().to_string());
    }
    if let Some(line) = mood_line(machine, snap) {
        parts.push(format!("## Right now\n{line}"));
    }
    if !pack.examples.trim().is_empty() {
        parts.push(pack.examples.trim().to_string());
    }
    parts.join("\n\n")
}

/// "You feel a little sad and quite playful." from variables that name a
/// feeling, strongest first.
pub fn mood_line(machine: &Machine, snap: &Snapshot) -> Option<String> {
    let mut feelings: Vec<(f32, &str)> = machine
        .spec()
        .variables
        .iter()
        .filter_map(|v| {
            let feeling = v.feeling.as_deref()?;
            if v.max <= v.rest {
                return None;
            }
            let value = snap.vars.get(&v.id).copied().unwrap_or(v.rest);
            let strength = (value - v.rest) / (v.max - v.rest);
            (strength >= FEELING_FLOOR).then_some((strength, feeling))
        })
        .collect();
    if feelings.is_empty() {
        return None;
    }
    feelings.sort_by(|a, b| b.0.total_cmp(&a.0));
    let phrases: Vec<String> = feelings
        .iter()
        .map(|(s, f)| {
            let degree = match s {
                s if *s < 0.35 => "a little",
                s if *s < 0.65 => "quite",
                _ => "very",
            };
            format!("{degree} {f}")
        })
        .collect();
    let list = match phrases.as_slice() {
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
        [] => unreachable!(),
    };
    Some(format!("You feel {list}."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use charpack::machine::MachineSpec;

    const SPEC: &str = r#"
initial = "calm"

[[variable]]
id = "sadness"
description = "Sad"
feeling = "sad"

[[variable]]
id = "playfulness"
description = "Playful"
feeling = "playful"

[[variable]]
id = "intoxication"
description = "No feeling named, so never in the mood line"

[[state]]
id = "calm"
description = "Calm"
"#;

    #[test]
    fn mood_line_names_feelings_by_strength() {
        let machine = Machine::new(toml::from_str::<MachineSpec>(SPEC).unwrap()).unwrap();
        let mut snap = machine.start();
        assert_eq!(mood_line(&machine, &snap), None);

        snap.vars.insert("sadness".into(), 0.2);
        snap.vars.insert("playfulness".into(), 0.7);
        snap.vars.insert("intoxication".into(), 0.9);
        assert_eq!(
            mood_line(&machine, &snap).as_deref(),
            Some("You feel very playful and a little sad.")
        );
    }
}
