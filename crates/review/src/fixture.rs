//! A synthetic corpus for trying the page and for tests: invented lines and
//! generated tones, so no game material is needed or ever committed.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::corpus::{
    write_jsonl, Audio, Axis, AxisScore, Clip, ContextLine, Label, Line, Source, AXES as AXES_FILE,
    CLIPS, LABELS, LINES,
};

/// The label axes from Tali's pack variables, in the pack's order.
pub const AXES: [(&str, &str); 6] = [
    ("joy", "Happiness and excitement"),
    ("sadness", "Sadness and homesickness"),
    ("anger", "Irritation and anger"),
    ("worry", "Concern, usually for the user"),
    ("playfulness", "In the mood to tease"),
    ("intoxication", "How much turian brandy is in her system"),
];

/// Every 25th clip is one clean dropped.
const DROP_EVERY: usize = 25;

const TEXTS: [&str; 12] = [
    "The coolant loop is holding, for now.",
    "I told you that would work. Admit it.",
    "Please be careful out there.",
    "I miss the sound of the ship at night.",
    "Do not touch that console again.",
    "One more round. Just one. Maybe two.",
    "That is the best news I have heard all week!",
    "We lost contact with the scout team.",
    "If you break it, you are fixing it.",
    "I am fine. Really. Mostly fine.",
    "Hand me the calibrator, would you?",
    "Everyone is singing. Why is everyone singing?",
];

const SCENES: [&str; 4] = ["engineering", "combat", "citadel_party", "normandy"];

/// Write a corpus of `count` synthetic clips into `dir`, laid out like the
/// stage runner's output: harvest, clean (dropping a few) and label.
pub fn write(dir: &Path, count: usize) -> std::io::Result<()> {
    let harvest = stage(dir, "harvest")?;
    let clean = stage(dir, "clean")?;
    let label = stage(dir, "label")?;
    let mut rng = Rng(0x5eed);
    let mut lines = Vec::new();
    let mut clips = Vec::new();
    let mut labels = Vec::new();
    for i in 0..count {
        let game = format!("le{}", 1 + i % 3);
        let id = format!("{game}.s9{i:05}");
        let text = TEXTS[i % TEXTS.len()];
        let scene = SCENES[rng.below(SCENES.len())];
        let party = scene == "citadel_party";
        let path = format!("audio/{id}.wav");
        let seconds = 1.0 + rng.unit() * 2.5;
        let wav = tone(220.0 + 40.0 * (i % 7) as f32, seconds);
        fs::write(harvest.join(&path), &wav)?;
        // Real clean writes FLAC; a copy of the WAV stands in for it.
        fs::write(clean.join(&path), &wav)?;

        let mut scores = BTreeMap::new();
        let lead = AXES[rng.below(AXES.len())].0;
        for (axis, _) in AXES {
            let base = if axis == lead {
                0.4 + 0.6 * rng.unit()
            } else {
                0.25 * rng.unit()
            };
            let base = if axis == "intoxication" && party {
                (base + 0.4).min(1.0)
            } else {
                base
            };
            let audio_says = (base + (rng.unit() - 0.5) * 0.3).clamp(0.0, 1.0);
            let text_says = (base + (rng.unit() - 0.5) * 0.6 * rng.unit()).clamp(0.0, 1.0);
            let agreement = 1.0 - (audio_says - text_says).abs();
            let mut signals = BTreeMap::from([
                ("audio".to_string(), Some(round(audio_says))),
                ("text".to_string(), Some(round(text_says))),
                ("prior".to_string(), None),
            ]);
            if axis == "intoxication" && party {
                signals.insert("prior".into(), Some(0.5));
            }
            scores.insert(
                axis.to_string(),
                AxisScore {
                    value: round((audio_says + text_says) / 2.0),
                    confidence: round((agreement * (0.6 + 0.4 * rng.unit())).clamp(0.05, 1.0)),
                    signals,
                },
            );
        }

        let audio = Audio {
            path,
            duration_s: Some(round(seconds)),
        };
        let keep = i % DROP_EVERY != DROP_EVERY - 1;
        clips.push(Clip {
            id: id.clone(),
            keep,
            audio: audio.clone(),
            transcript: Some(text.to_string()),
            flags: if keep { vec![] } else { vec!["music".into()] },
        });
        lines.push(Line {
            id: id.clone(),
            speaker: "fixture".into(),
            text: text.to_string(),
            flags: vec![],
            audio,
            source: Source {
                game: Some(game),
                conversation: Some(format!("synthetic_{scene}_{}", i / 4)),
            },
            context: vec![
                ContextLine {
                    speaker: "commander".into(),
                    text: "How are we looking?".into(),
                },
                ContextLine {
                    speaker: "pilot".into(),
                    text: "Better than last time.".into(),
                },
            ],
            scene_tags: vec![scene.to_string()],
            stage_direction: (i % 9 == 0).then(|| "quietly, a little tired".to_string()),
        });
        let tags = if scene == "combat" {
            vec!["combat_bark".to_string()]
        } else {
            vec![]
        };
        labels.push(Label { id, scores, tags });
    }
    write_jsonl(&harvest.join(LINES), &lines)?;
    write_jsonl(&clean.join(CLIPS), &clips)?;
    write_jsonl(&label.join(LABELS), &labels)?;
    let axes: Vec<Axis> = AXES
        .iter()
        .map(|(id, description)| Axis {
            id: id.to_string(),
            description: description.to_string(),
        })
        .collect();
    fs::write(
        label.join(AXES_FILE),
        serde_json::to_string_pretty(&axes).expect("axes serialize"),
    )
}

/// Make `stages/<name>/synthetic/out/audio` and point `current` at it.
fn stage(dir: &Path, name: &str) -> std::io::Result<std::path::PathBuf> {
    let stage_dir = dir.join("stages").join(name);
    let out = stage_dir.join("synthetic").join("out");
    fs::create_dir_all(out.join("audio"))?;
    fs::write(stage_dir.join("current"), "synthetic\n")?;
    Ok(out)
}

fn round(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

/// A 24 kHz mono 16-bit WAV of a fading tone.
fn tone(hz: f32, seconds: f32) -> Vec<u8> {
    const RATE: u32 = 24_000;
    let n = (RATE as f32 * seconds) as u32;
    let mut wav = Vec::with_capacity(44 + 2 * n as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + 2 * n).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&RATE.to_le_bytes());
    wav.extend_from_slice(&(RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(2 * n).to_le_bytes());
    for i in 0..n {
        let t = i as f32 / RATE as f32;
        let envelope = (t * 20.0).min(1.0) * (1.0 - t / seconds);
        let s = (t * hz * std::f32::consts::TAU).sin() * envelope * 0.3;
        wav.extend_from_slice(&((s * i16::MAX as f32) as i16).to_le_bytes());
    }
    wav
}

/// Small deterministic generator, so fixtures are the same every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
