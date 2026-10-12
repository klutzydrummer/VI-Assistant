//! A synthetic corpus for trying the page and for tests: invented lines and
//! generated tones, so no game material is needed or ever committed.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::corpus::{write_jsonl, AxisScore, ContextLine, Label, Line, LABELS, LINES};

/// The label axes from Tali's pack variables.
pub const AXES: [&str; 6] = [
    "joy",
    "sadness",
    "anger",
    "worry",
    "playfulness",
    "intoxication",
];

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

const SCENES: [&str; 4] = ["engineering", "combat", "party", "conversation"];

/// Write `count` synthetic clips with labels into `dir`.
pub fn write(dir: &Path, count: usize) -> std::io::Result<()> {
    fs::create_dir_all(dir.join("audio"))?;
    let mut rng = Rng(0x5eed);
    let mut lines = Vec::new();
    let mut labels = Vec::new();
    for i in 0..count {
        let id = format!("fx{i:04}");
        let text = TEXTS[i % TEXTS.len()];
        let scene = SCENES[rng.below(SCENES.len())];
        let audio = format!("audio/{id}.wav");
        let seconds = 1.0 + rng.unit() * 2.5;
        fs::write(
            dir.join(&audio),
            tone(220.0 + 40.0 * (i % 7) as f32, seconds),
        )?;

        let mut scores = BTreeMap::new();
        let lead = AXES[rng.below(AXES.len())];
        for axis in AXES {
            let base = if axis == lead {
                0.4 + 0.6 * rng.unit()
            } else {
                0.25 * rng.unit()
            };
            let base = if axis == "intoxication" && scene == "party" {
                (base + 0.4).min(1.0)
            } else {
                base
            };
            let audio_says = (base + (rng.unit() - 0.5) * 0.3).clamp(0.0, 1.0);
            let text_says = (base + (rng.unit() - 0.5) * 0.6 * rng.unit()).clamp(0.0, 1.0);
            let agreement = 1.0 - (audio_says - text_says).abs();
            let mut signals = BTreeMap::from([
                ("audio".to_string(), round(audio_says)),
                ("text".to_string(), round(text_says)),
            ]);
            if axis == "intoxication" && scene == "party" {
                signals.insert("prior".into(), 0.5);
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

        lines.push(Line {
            id: id.clone(),
            audio,
            text: text.to_string(),
            speaker: "Fixture".into(),
            game: Some(format!("SYN{}", 1 + i % 3)),
            conversation: Some(format!("synthetic_{scene}_{}", i / 4)),
            context: vec![
                ContextLine {
                    speaker: "Commander".into(),
                    text: "How are we looking?".into(),
                },
                ContextLine {
                    speaker: "Pilot".into(),
                    text: "Better than last time.".into(),
                },
            ],
            tags: vec![scene.to_string()],
        });
        labels.push(Label { id, scores });
    }
    write_jsonl(&dir.join(LINES), &lines)?;
    write_jsonl(&dir.join(LABELS), &labels)
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
