//! Which clips to review first.
//!
//! An hour covers a few hundred clips out of a few thousand, so the order
//! matters more than anything else on the page. A clip ranks high when the
//! label stage is unsure of an axis *and* the clip is likely to land in that
//! axis's contrast set, because a wrong positive moves the vector most.

use serde::Serialize;

use crate::corpus::{Corpus, Label, Verdict};

/// Contrast sets take positives from lines scoring about this high.
pub const POSITIVE_FROM: f32 = 0.6;

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

/// Clips still to review, most useful first. Skipped clips go to the back.
pub fn build(corpus: &Corpus) -> Vec<Item> {
    let latest = corpus.latest();
    let mut fresh = Vec::new();
    let mut skipped = Vec::new();
    for line in &corpus.lines {
        let Some(item) = corpus.labels.get(&line.id).and_then(rank) else {
            continue;
        };
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
