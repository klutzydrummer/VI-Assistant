//! Which clips to review first.
//!
//! An hour covers a few hundred clips out of a few thousand, so the order
//! matters more than anything else on the page. First comes a calibration
//! sample: `CALIBRATION` clips picked at random, so the accuracy the
//! calibrate stage measures on them isn't biased toward hard cases. Then a
//! clip ranks high when the label stage is unsure of an axis *and* the clip
//! is likely to land in that axis's contrast set, because a wrong positive
//! moves the vector most.

use std::collections::HashMap;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::corpus::{Corpus, Label, Verdict};

/// Contrast sets take positives from lines scoring about this high.
pub const POSITIVE_FROM: f32 = 0.6;

/// Size of the calibration sample reviewed first.
pub const CALIBRATION: usize = 150;

/// The calibration sample: of the given ids, the `n` whose lowercase hex
/// SHA-256 sorts first, in that order. A uniform random sample that any tool
/// can recompute from the ids alone (the calibrate stage does).
pub fn calibration<'a>(ids: impl IntoIterator<Item = &'a str>, n: usize) -> Vec<&'a str> {
    let mut keyed: Vec<(String, &str)> = ids
        .into_iter()
        .map(|id| {
            let hex: String = Sha256::digest(id.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            (hex, id)
        })
        .collect();
    keyed.sort();
    keyed.into_iter().take(n).map(|(_, id)| id).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct Item {
    pub id: String,
    pub priority: f32,
    /// The axis that ranked it, for the "why is this here" note.
    pub axis: String,
    pub reason: String,
}

/// How unsure the label stage is about an axis: low confidence, or signals
/// that disagree with each other.
pub fn uncertainty(score: &crate::corpus::AxisScore) -> f32 {
    let lo = score.scored().fold(f32::INFINITY, f32::min);
    let hi = score.scored().fold(f32::NEG_INFINITY, f32::max);
    let spread = if score.scored().count() > 1 {
        hi - lo
    } else {
        0.0
    };
    (1.0 - score.confidence).max(spread).clamp(0.0, 1.0)
}

/// How much a mistake on this axis would move its vector. Likely positives
/// matter most; clearly neutral lines still count, since they are the
/// negatives.
pub fn stake(value: f32) -> f32 {
    let near_positive = 1.0 - ((value - POSITIVE_FROM).abs() / POSITIVE_FROM).min(1.0);
    0.4 + 0.6 * value.max(near_positive)
}

pub fn rank(label: &Label) -> Option<Item> {
    label
        .scores
        .iter()
        .map(|(axis, s)| (axis, s, uncertainty(s) * stake(s.value)))
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(axis, s, priority)| {
            let spread = uncertainty(s) > 1.0 - s.confidence;
            let why = if spread {
                "signals disagree"
            } else {
                "low confidence"
            };
            Item {
                id: label.id.clone(),
                priority,
                axis: axis.clone(),
                reason: format!(
                    "{axis} {:.2}, {why} (confidence {:.2})",
                    s.value, s.confidence
                ),
            }
        })
}

/// Clips still to review, most useful first: the calibration sample, then
/// by uncertainty. Skipped clips go to the back.
pub fn build(corpus: &Corpus) -> Vec<Item> {
    build_with(corpus, CALIBRATION)
}

pub fn build_with(corpus: &Corpus, calibration_size: usize) -> Vec<Item> {
    let labelled = corpus
        .lines
        .iter()
        .filter(|l| corpus.labels.contains_key(&l.id))
        .map(|l| l.id.as_str());
    let sample: HashMap<&str, usize> = calibration(labelled, calibration_size)
        .into_iter()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    let latest = corpus.latest();
    let mut fresh = Vec::new();
    let mut skipped = Vec::new();
    for line in &corpus.lines {
        let Some(mut item) = corpus.labels.get(&line.id).and_then(rank) else {
            continue;
        };
        if let Some(&i) = sample.get(line.id.as_str()) {
            // Above every uncertainty priority (at most 1), in sample order.
            item.priority = 2.0 - i as f32 / sample.len() as f32;
            item.reason = format!("calibration sample (random), {}", item.reason);
        }
        match latest.get(line.id.as_str()).map(|r| r.verdict) {
            None => fresh.push(item),
            Some(Verdict::Skip) => skipped.push(item),
            Some(Verdict::Keep | Verdict::Flag) => {}
        }
    }
    for list in [&mut fresh, &mut skipped] {
        list.sort_by(|a, b| b.priority.total_cmp(&a.priority));
    }
    fresh.extend(skipped);
    fresh
}
