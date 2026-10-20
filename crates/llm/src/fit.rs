//! Stage 2 of the vector builder: fit the vectors from a capture's summary
//! store and the label table, on the CPU, with no model. See
//! `docs/design/steering-builder-design.md`, "Stage 2: fit".
//!
//! **joint** (the default): per layer, least squares over every labelled
//! line, `h = μ + Σ s_k β_k + Σ c_j γ_j`, with `s_k` the line's score for
//! mood k divided by 3, the knobs at their levels, and nuisance columns for
//! the game (one level dropped), log line length and measured audio energy.
//! A small fixed ridge, `λ = 1e-4 x mean diag(CᵀC)`, sits on the mood and
//! knob columns only. A column the variance-inflation check flags is shrunk
//! toward its own `mean_diff` instead of toward 0.
//!
//! **mean_diff**: per contrast set, the mean of its pairs' differences
//! (positive minus matched zero), so a zero used by three pairs counts three
//! times.
//!
//! Every estimate here is a fixed weighting of the lines' summaries, with
//! weights that depend only on the labels: `β = (CᵀC + Λ)⁻¹ CᵀH = WᵀH`. So
//! the fit computes all the weights first (joint, mean_diff, and the
//! report's variants: masked scores, conversation fixed effects, split
//! halves and cross-validation folds), then goes through the store a block
//! of layers at a time: it reads the block's slice of every needed line
//! once (front to back), adds `Wᵀ` times each chunk of lines, and from the
//! same in-memory copy projects lines for held-out AUC and takes each
//! pair's difference for the pair consistency. The store is read once in
//! all; memory is set by `mem_mb` (the block size), whatever the number of
//! lines.
//!
//! Nothing is dropped or lowered on the report's flags; thin and flagged
//! moods are built at full strength like the rest (Brandon, PR #62).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::gguf::Value;
use crate::linalg::{Cholesky, Square};
use crate::store::{Store, Summary};
use crate::vectors::{vector_gguf, Scheme, LIMITS_FILE, MODEL_SHA_KEY};

/// One line of the label table (`labels.jsonl`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelRow {
    pub line_id: String,
    pub conversation: String,
    pub game: String,
    /// Measured audio energy of the clip (any consistent unit, e.g. dB).
    #[serde(default)]
    pub audio_energy: Option<f64>,
    /// Pass-2 score 0-3 for every mood family.
    pub scores: BTreeMap<String, f64>,
    /// Knob levels (energy, intoxication, hungover), 0 to 1.
    #[serde(default)]
    pub knobs: BTreeMap<String, f64>,
    /// Whether pass 1 agreed with pass 2, per mood; absent means agreed.
    #[serde(default)]
    pub pass1_agree: BTreeMap<String, bool>,
}

/// One pair of a contrast set (`pairs.jsonl`): her positive line and its
/// matched zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pair {
    pub set: String,
    pub pos: String,
    pub zero: String,
}

pub fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<(Vec<T>, String), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?,
        );
    }
    Ok((out, crate::registry::hex(&Sha256::digest(text.as_bytes()))))
}

#[derive(Debug, Clone)]
pub struct FitOptions {
    /// λ as a fraction of the mean diagonal of CᵀC.
    pub ridge: f64,
    pub folds: usize,
    /// Variance inflation above this flags a column as ill-conditioned.
    pub vif_flag: f64,
    /// Rough cap on the accumulators' memory.
    pub mem_mb: usize,
    pub summary: Summary,
    /// Positives below this count are flagged thin.
    pub thin: usize,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            ridge: 1e-4,
            folds: 5,
            vif_flag: 30.0,
            mem_mb: 768,
            summary: Summary::Mean,
            thin: 30,
        }
    }
}

/// Per vector and layer, what the report shows. Flags only.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LayerStats {
    pub layer: usize,
    pub joint_norm: Option<f64>,
    pub mean_diff_norm: Option<f64>,
    /// |joint| / |mean_diff|: shrinkage left after the ridge shows here.
    pub joint_over_mean_diff: Option<f64>,
    /// Mean residual norm at this layer over the steered positions.
    pub resid_norm: f64,
    /// |vector| / residual norm (joint, else mean_diff).
    pub norm_over_resid: f64,
    /// Held-out AUC on the set, folds grouped by conversation.
    pub auc: Option<f64>,
    /// Cosine between joint and joint with conversation fixed effects.
    pub fe_cos: Option<f64>,
    /// Cosine between joint fits on two halves of the conversations.
    pub split_half_cos: Option<f64>,
    /// Cosine between joint and the masked-score variant, and their norm ratio.
    pub masked_cos: Option<f64>,
    pub masked_norm_ratio: Option<f64>,
    /// Cosine between joint and mean_diff.
    pub joint_mean_diff_cos: Option<f64>,
    /// Mean cosine among the set's pair differences.
    pub pair_cos: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct VectorReport {
    pub name: String,
    pub kind: String,
    pub has_joint: bool,
    pub has_set: bool,
    /// Lines with a score of 3 (moods) or above 0.5 (knobs).
    pub n_strong: usize,
    pub n_pairs: usize,
    /// Distinct conversations the set's positives come from, and the share
    /// of the largest.
    pub n_conversations: usize,
    pub top_conversation_share: f64,
    pub vif: Option<f64>,
    pub shrunk_toward_mean_diff: bool,
    pub flags: Vec<String>,
    pub layers: Vec<LayerStats>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Report {
    pub store: String,
    pub model_sha256: String,
    pub labels_sha256: String,
    pub pairs_sha256: String,
    pub summary: String,
    pub lambda: f64,
    pub ridge: f64,
    pub n_lines: usize,
    pub n_labelled: usize,
    pub unlabelled_store_lines: usize,
    pub labels_not_in_store: usize,
    pub pairs_dropped: usize,
    pub audio_energy_missing: usize,
    /// The layer `vectors.toml` starts on: the middle of the model's depth,
    /// until the chat check picks one.
    pub provisional_layer: usize,
    pub columns: Vec<String>,
    pub vectors: Vec<VectorReport>,
    pub notes: Vec<String>,
}

/// What the fit makes: per method, per vector, natural per-layer directions.
pub struct Fitted {
    pub report: Report,
    pub joint: BTreeMap<String, BTreeMap<u32, Vec<f32>>>,
    pub mean_diff: BTreeMap<String, BTreeMap<u32, Vec<f32>>>,
    /// Joint cosines between vectors per layer (names, layer, matrix).
    pub cosines: (Vec<String>, BTreeMap<usize, Vec<Vec<f32>>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Variant {
    Joint,
    Masked,
    MeanDiff,
    Fe,
    Half(u8),
    Fold(u8),
}

fn bucket(conv: &str, byte: usize, n: usize) -> usize {
    Sha256::digest(conv.as_bytes())[byte] as usize % n
}

/// Design rows for the labelled lines.
struct Design {
    cols: Vec<String>,
    /// Which columns the ridge penalises (moods and knobs).
    penalised: Vec<bool>,
    /// Labelled store line -> row.
    x: Vec<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn design(
    rows: &[(usize, &LabelRow)],
    outs: &[String],
    n_moods: usize,
    log_len: &[f64],
    groups: &[String],
    group_name: &str,
    energy_mean: f64,
    energy_missing: bool,
) -> Design {
    let levels: BTreeSet<&String> = groups.iter().collect();
    let levels: Vec<&String> = levels.into_iter().skip(1).collect();
    let mut cols = vec!["intercept".to_string()];
    cols.extend(outs.iter().cloned());
    cols.extend(levels.iter().map(|l| format!("{group_name}={l}")));
    cols.push("log_length".into());
    cols.push("audio_energy".into());
    if energy_missing {
        cols.push("audio_energy_missing".into());
    }
    let mut penalised = vec![false; cols.len()];
    penalised[1..=outs.len()].fill(true);
    let x = rows
        .iter()
        .zip(groups)
        .map(|(&(i, r), g)| {
            let mut v = vec![1.0];
            for (k, name) in outs.iter().enumerate() {
                v.push(if k < n_moods {
                    r.scores[name] / 3.0
                } else {
                    r.knobs.get(name).copied().unwrap_or(0.0)
                });
            }
            v.extend(levels.iter().map(|l| f64::from(u8::from(*l == g))));
            v.push(log_len[i]);
            v.push(r.audio_energy.unwrap_or(energy_mean));
            if energy_missing {
                v.push(f64::from(u8::from(r.audio_energy.is_none())));
            }
            v
        })
        .collect();
    Design { cols, penalised, x }
}

/// `(XᵀX + Λ)` over the rows in `mask`, with `lambda` on the penalised columns.
fn normal_matrix(d: &Design, mask: &[bool], lambda: f64) -> Square {
    let mut a = Square::zeros(d.cols.len());
    for (x, &m) in d.x.iter().zip(mask) {
        if m {
            a.add_outer(x, 1.0);
        }
    }
    for (j, &p) in d.penalised.iter().enumerate() {
        if p {
            *a.at_mut(j, j) += lambda;
        }
    }
    a
}

/// Weights over rows giving the ridge estimate of column `col`, from the
/// rows in `mask`; plus, for each shrink target `(column, weights)`,
/// `λ (A⁻¹)[col, f]` times that target's weights.
fn ridge_weights(
    d: &Design,
    chol: &Cholesky,
    mask: &[bool],
    col: usize,
    lambda: f64,
    shrink: &[(usize, &[f64])],
) -> Vec<f64> {
    let g = chol.inverse_column(col);
    let mut w: Vec<f64> =
        d.x.iter()
            .zip(mask)
            .map(|(x, &m)| {
                if m {
                    x.iter().zip(&g).map(|(a, b)| a * b).sum()
                } else {
                    0.0
                }
            })
            .collect();
    for &(f, target) in shrink {
        let c = lambda * g[f];
        for (wi, ti) in w.iter_mut().zip(target) {
            *wi += c * ti;
        }
    }
    w
}

/// Variance inflation of each column (intercept excluded) over all rows.
fn vif(d: &Design) -> Vec<f64> {
    let p = d.cols.len() - 1;
    let n = d.x.len() as f64;
    let mut mean = vec![0.0; p];
    for x in &d.x {
        for j in 0..p {
            mean[j] += x[j + 1] / n;
        }
    }
    let mut cov = Square::zeros(p);
    for x in &d.x {
        let c: Vec<f64> = (0..p).map(|j| x[j + 1] - mean[j]).collect();
        cov.add_outer(&c, 1.0 / n);
    }
    let var: Vec<f64> = (0..p).map(|j| cov.at(j, j)).collect();
    let trace: f64 = var.iter().sum();
    for j in 0..p {
        *cov.at_mut(j, j) += 1e-12 * trace.max(1e-300);
    }
    let Ok(chol) = Cholesky::new(&cov) else {
        return vec![f64::INFINITY; d.cols.len()];
    };
    let mut out = vec![f64::NAN];
    for (j, &v) in var.iter().enumerate() {
        out.push(if v <= 1e-12 * trace / p as f64 {
            f64::INFINITY
        } else {
            v * chol.inverse_column(j)[j]
        });
    }
    out
}

fn auc(pos: &[f32], neg: &[f32]) -> Option<f64> {
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut all: Vec<(f32, bool)> = pos
        .iter()
        .map(|&x| (x, true))
        .chain(neg.iter().map(|&x| (x, false)))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Mann-Whitney with average ranks for ties.
    let mut rank_sum = 0.0f64;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        rank_sum += r * all[i..=j].iter().filter(|x| x.1).count() as f64;
        i = j + 1;
    }
    let (np, nn) = (pos.len() as f64, neg.len() as f64);
    Some((rank_sum - np * (np + 1.0) / 2.0) / (np * nn))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cos(a: &[f64], b: &[f64]) -> Option<f64> {
    let (na, nb) = (dot(a, a).sqrt(), dot(b, b).sqrt());
    (na > 0.0 && nb > 0.0).then(|| dot(a, b) / (na * nb))
}

/// Fit every vector from `store`, `labels` and `pairs`.
pub fn fit(
    store: &mut Store,
    labels: &[LabelRow],
    pairs: &[Pair],
    hashes: (&str, &str),
    opts: &FitOptions,
    mut progress: impl FnMut(&str),
) -> Result<Fitted, String> {
    let meta = store.meta.clone();
    let e = meta.n_embd;
    let n_store = store.lines.len();
    let index: HashMap<&str, usize> = store
        .lines
        .iter()
        .enumerate()
        .map(|(i, r)| (r.line_id.as_str(), i))
        .collect();

    // ---- labels
    let first = labels.first().ok_or("the label table is empty")?;
    let moods: Vec<String> = first.scores.keys().cloned().collect();
    let knobs: BTreeSet<String> = labels
        .iter()
        .flat_map(|r| r.knobs.keys().cloned())
        .collect();
    let mut outs: Vec<String> = moods.clone();
    for k in &knobs {
        if first.scores.contains_key(k) {
            return Err(format!("{k} is both a mood and a knob"));
        }
        outs.push(k.clone());
    }
    let n_moods = moods.len();
    let mut rows: Vec<(usize, &LabelRow)> = Vec::new();
    let mut labelled = vec![None; n_store];
    let mut labels_not_in_store = 0;
    for r in labels {
        if r.scores.len() != n_moods || !moods.iter().all(|m| r.scores.contains_key(m)) {
            return Err(format!(
                "line {} doesn't score the same moods as line {}",
                r.line_id, first.line_id
            ));
        }
        if let Some((m, s)) = r.scores.iter().find(|(_, s)| !(0.0..=3.0).contains(*s)) {
            return Err(format!("line {}: {m} score {s} is outside 0-3", r.line_id));
        }
        match index.get(r.line_id.as_str()) {
            Some(&i) => {
                if labelled[i].is_some() {
                    return Err(format!("line {} is labelled twice", r.line_id));
                }
                labelled[i] = Some(rows.len());
                rows.push((i, r));
            }
            None => labels_not_in_store += 1,
        }
    }
    if rows.len() < outs.len() + 10 {
        return Err(format!(
            "only {} labelled lines are in the store, too few for {} columns",
            rows.len(),
            outs.len()
        ));
    }
    let log_len: Vec<f64> = store
        .lines
        .iter()
        .map(|r| (r.n_line_tokens.max(1) as f64).ln())
        .collect();
    let energies: Vec<f64> = rows.iter().filter_map(|(_, r)| r.audio_energy).collect();
    let energy_missing = rows.len() - energies.len();
    let energy_mean = if energies.is_empty() {
        0.0
    } else {
        energies.iter().sum::<f64>() / energies.len() as f64
    };

    // ---- sets
    let mut sets: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    let mut pairs_dropped = 0;
    for p in pairs {
        match (index.get(p.pos.as_str()), index.get(p.zero.as_str())) {
            (Some(&a), Some(&b)) => sets.entry(p.set.clone()).or_default().push((a, b)),
            _ => pairs_dropped += 1,
        }
    }
    for s in sets.keys() {
        if !outs.contains(s) {
            outs.push(s.clone());
        }
    }
    let n_joint = n_moods + knobs.len();

    // ---- designs and weights
    let games: Vec<String> = rows.iter().map(|(_, r)| r.game.clone()).collect();
    let convs: Vec<String> = rows.iter().map(|(_, r)| r.conversation.clone()).collect();
    let joint_outs = &outs[..n_joint];
    let d = design(
        &rows,
        joint_outs,
        n_moods,
        &log_len,
        &games,
        "game",
        energy_mean,
        energy_missing > 0,
    );
    let all = vec![true; rows.len()];
    let a0 = normal_matrix(&d, &all, 0.0);
    let lambda = opts.ridge * (0..a0.n).map(|j| a0.at(j, j)).sum::<f64>() / a0.n as f64;
    let chol = Cholesky::new(&normal_matrix(&d, &all, lambda)).map_err(|e| {
        format!("the joint design is singular ({e}); a nuisance column may be constant")
    })?;
    let vifs = vif(&d);

    // mean_diff weights over store lines.
    let mut md_w: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (s, ps) in &sets {
        let mut w = vec![0.0; n_store];
        let n = ps.len() as f64;
        for &(a, b) in ps {
            w[a] += 1.0 / n;
            w[b] -= 1.0 / n;
        }
        md_w.insert(s, w);
    }
    // The same over labelled rows, for shrinking toward mean_diff.
    let md_rows = |s: &str| -> Option<Vec<f64>> {
        let w = md_w.get(s)?;
        Some(rows.iter().map(|&(i, _)| w[i]).collect())
    };

    let mut flagged: Vec<usize> = Vec::new();
    for k in 0..n_joint {
        if vifs[k + 1] > opts.vif_flag && md_w.contains_key(outs[k].as_str()) {
            flagged.push(k);
        }
    }
    // Weights for shrinking flagged columns toward their mean_diff need
    // mean_diff over labelled rows only: a pair line without a label
    // can't enter the joint system, so its weight is carried separately.
    let flagged_targets: Vec<(usize, Vec<f64>)> = flagged
        .iter()
        .map(|&k| (k + 1, md_rows(&outs[k]).unwrap()))
        .collect();
    // Columns of the weight matrix: (variant, out) -> column index.
    let mut cols: Vec<(Variant, usize)> = Vec::new();
    let mut w_cols: Vec<Vec<f64>> = Vec::new(); // over store lines
    let to_store = |w_rows: Vec<f64>| -> Vec<f64> {
        let mut w = vec![0.0; n_store];
        for (&(i, _), x) in rows.iter().zip(w_rows) {
            w[i] = x;
        }
        w
    };

    progress("weights: joint");
    for k in 0..n_joint {
        let shrink: Vec<(usize, &[f64])> = if flagged.contains(&k) {
            flagged_targets
                .iter()
                .filter(|(c, _)| *c == k + 1)
                .map(|(c, w)| (*c, w.as_slice()))
                .collect()
        } else {
            Vec::new()
        };
        let mut w = to_store(ridge_weights(&d, &chol, &all, k + 1, lambda, &shrink));
        if !shrink.is_empty() {
            // Pair lines outside the label table still belong in the target.
            let c = lambda * chol.inverse_column(k + 1)[k + 1];
            let md = &md_w[outs[k].as_str()];
            for (i, wi) in w.iter_mut().enumerate() {
                if labelled[i].is_none() {
                    *wi += c * md[i];
                }
            }
        }
        cols.push((Variant::Joint, k));
        w_cols.push(w);
    }
    for (s, w) in &md_w {
        let o = outs.iter().position(|x| x == s).unwrap();
        cols.push((Variant::MeanDiff, o));
        w_cols.push(w.clone());
    }

    progress("weights: masked scores");
    for (k, m) in moods.iter().enumerate() {
        let mask: Vec<bool> = rows
            .iter()
            .map(|(_, r)| {
                let s = r.scores[m];
                (s == 0.0 || s == 3.0) && r.pass1_agree.get(m).copied().unwrap_or(true)
            })
            .collect();
        let Ok(c) = Cholesky::new(&normal_matrix(&d, &mask, lambda)) else {
            continue;
        };
        cols.push((Variant::Masked, k));
        w_cols.push(to_store(ridge_weights(&d, &c, &mask, k + 1, lambda, &[])));
    }

    progress("weights: conversation fixed effects");
    let dfe = design(
        &rows,
        joint_outs,
        n_moods,
        &log_len,
        &convs,
        "conversation",
        energy_mean,
        energy_missing > 0,
    );
    let mut notes = Vec::new();
    match Cholesky::new(&normal_matrix(&dfe, &all, lambda)) {
        Ok(c) => {
            for k in 0..n_joint {
                cols.push((Variant::Fe, k));
                w_cols.push(to_store(ridge_weights(&dfe, &c, &all, k + 1, lambda, &[])));
            }
        }
        Err(e) => notes.push(format!("conversation fixed-effects fit skipped: {e}")),
    }

    progress("weights: split halves and folds");
    let mut subsets: Vec<(Variant, Vec<bool>)> = Vec::new();
    for h in 0..2u8 {
        subsets.push((
            Variant::Half(h),
            convs
                .iter()
                .map(|c| bucket(c, 1, 2) == h as usize)
                .collect(),
        ));
    }
    let fold_of: Vec<usize> = convs.iter().map(|c| bucket(c, 0, opts.folds)).collect();
    let with_sets: Vec<usize> = (0..n_joint)
        .filter(|&k| md_w.contains_key(outs[k].as_str()))
        .collect();
    for f in 0..opts.folds {
        subsets.push((
            Variant::Fold(f as u8),
            fold_of.iter().map(|&x| x != f).collect(),
        ));
    }
    for (v, mask) in &subsets {
        let Ok(c) = Cholesky::new(&normal_matrix(&d, mask, lambda)) else {
            notes.push(format!("{v:?} fit skipped: singular"));
            continue;
        };
        let ks: Vec<usize> = match v {
            Variant::Fold(_) => with_sets.clone(),
            _ => (0..n_joint).collect(),
        };
        for k in ks {
            cols.push((*v, k));
            w_cols.push(to_store(ridge_weights(&d, &c, mask, k + 1, lambda, &[])));
        }
    }
    let col_of: HashMap<(Variant, usize), usize> =
        cols.iter().enumerate().map(|(c, &v)| (v, c)).collect();
    let nc = cols.len();
    // Lines that matter for any estimate or diagnostic.
    let in_sets: BTreeSet<usize> = sets.values().flatten().flat_map(|&(a, b)| [a, b]).collect();
    let needed: Vec<usize> = (0..n_store)
        .filter(|&i| in_sets.contains(&i) || w_cols.iter().any(|w| w[i] != 0.0))
        .collect();
    let cache_at: HashMap<usize, usize> = needed.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    // Line-major weights for the chunked product.
    let mut wl = vec![0.0f64; n_store * nc];
    for (c, w) in w_cols.iter().enumerate() {
        for (i, &x) in w.iter().enumerate() {
            wl[i * nc + c] = x;
        }
    }
    drop(w_cols);

    // ---- passes over the store, a block of layers at a time
    let n_slots = meta.layers.len();
    // Per layer in a block: the accumulators (F64), an F16 copy of the
    // block for every needed line (read once, used by both passes), and
    // one chunk of lines in F64.
    let per_layer = nc * e * 8 + needed.len() * e * 2 + needed.len().min(64) * e * 8;
    let block = ((opts.mem_mb << 20) / per_layer.max(1)).clamp(1, n_slots);
    let resid = store.mean_norms();
    let mut beta_joint: BTreeMap<String, BTreeMap<u32, Vec<f32>>> = BTreeMap::new();
    let mut beta_md: BTreeMap<String, BTreeMap<u32, Vec<f32>>> = BTreeMap::new();
    let mut stats: Vec<Vec<LayerStats>> = vec![Vec::new(); outs.len()];
    let mut cosines: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
    let set_lines: BTreeMap<&str, (Vec<usize>, Vec<usize>)> = sets
        .iter()
        .map(|(s, ps)| {
            let pos: BTreeSet<usize> = ps.iter().map(|p| p.0).collect();
            let neg: BTreeSet<usize> = ps.iter().map(|p| p.1).collect();
            (
                s.as_str(),
                (pos.into_iter().collect(), neg.into_iter().collect()),
            )
        })
        .collect();
    let mut s0 = 0;
    while s0 < n_slots {
        let s1 = (s0 + block).min(n_slots);
        let nl = s1 - s0;
        let dim = nl * e;
        progress(&format!(
            "layers {}..={}: reading and summing {} lines",
            meta.layers[s0],
            meta.layers[s1 - 1],
            needed.len()
        ));
        // In line order, so the store is read front to back.
        let mut cache = vec![0u16; needed.len() * dim];
        for (k, &i) in needed.iter().enumerate() {
            store.read_raw(opts.summary, i, s0..s1, &mut cache[k * dim..(k + 1) * dim])?;
        }
        let row = |i: usize, out: &mut [f32]| {
            let k = cache_at[&i];
            for (o, &x) in out.iter_mut().zip(&cache[k * dim..(k + 1) * dim]) {
                *o = crate::f16::to_f32(x);
            }
        };
        let mut acc = vec![0.0f64; nc * dim];
        let chunk = 64usize;
        let mut h32 = vec![0.0f32; dim];
        let mut h = vec![0.0f64; chunk * dim];
        let mut wt = vec![0.0f64; nc * chunk];
        for lines in needed.chunks(chunk) {
            for (b, &i) in lines.iter().enumerate() {
                row(i, &mut h32);
                for (dst, &x) in h[b * dim..(b + 1) * dim].iter_mut().zip(&h32) {
                    *dst = f64::from(x);
                }
                for c in 0..nc {
                    wt[c * chunk + b] = wl[i * nc + c];
                }
            }
            let nb = lines.len();
            // acc (nc x dim) += wt (nc x nb) . h (nb x dim)
            unsafe {
                matrixmultiply::dgemm(
                    nc,
                    nb,
                    dim,
                    1.0,
                    wt.as_ptr(),
                    chunk as isize,
                    1,
                    h.as_ptr(),
                    dim as isize,
                    1,
                    1.0,
                    acc.as_mut_ptr(),
                    dim as isize,
                    1,
                );
            }
        }
        let vecof = |v: Variant, o: usize, sl: usize| -> Option<&[f64]> {
            let c = *col_of.get(&(v, o))?;
            let off = c * dim + (sl - s0) * e;
            Some(&acc[off..off + e])
        };

        // Second pass: held-out projections and pair differences.
        progress(&format!(
            "layers {}..={}: held-out AUC and pair consistency",
            meta.layers[s0],
            meta.layers[s1 - 1]
        ));
        // (out, slot) -> (positive projections, zero projections)
        let mut proj: HashMap<(usize, usize), (Vec<f32>, Vec<f32>)> = HashMap::new();
        let mut line_sets: BTreeMap<usize, Vec<(usize, bool)>> = BTreeMap::new();
        for &k in &with_sets {
            let (pos, neg) = &set_lines[outs[k].as_str()];
            for &i in pos {
                line_sets.entry(i).or_default().push((k, true));
            }
            for &i in neg {
                line_sets.entry(i).or_default().push((k, false));
            }
        }
        for (&i, uses) in &line_sets {
            let Some(r) = labelled[i] else { continue };
            let f = fold_of[r];
            row(i, &mut h32);
            for &(k, is_pos) in uses {
                for sl in s0..s1 {
                    let Some(b) = vecof(Variant::Fold(f as u8), k, sl) else {
                        continue;
                    };
                    let x = &h32[(sl - s0) * e..(sl - s0 + 1) * e];
                    let p: f64 = b.iter().zip(x).map(|(a, &y)| a * f64::from(y)).sum();
                    let ent = proj.entry((k, sl)).or_default();
                    if is_pos {
                        ent.0.push(p as f32);
                    } else {
                        ent.1.push(p as f32);
                    }
                }
            }
        }
        let mut pair_cos: HashMap<(usize, usize), f64> = HashMap::new();
        let mut hb = vec![0.0f32; dim];
        for (s, ps) in &sets {
            let o = outs.iter().position(|x| x == s).unwrap();
            let mut sum = vec![0.0f64; dim];
            let mut n = vec![0usize; nl];
            for &(a, b) in ps {
                row(a, &mut h32);
                row(b, &mut hb);
                for j in 0..nl {
                    let d: Vec<f64> = (j * e..(j + 1) * e)
                        .map(|t| f64::from(h32[t]) - f64::from(hb[t]))
                        .collect();
                    let nd = dot(&d, &d).sqrt();
                    if nd > 0.0 {
                        n[j] += 1;
                        for (acc, x) in sum[j * e..(j + 1) * e].iter_mut().zip(&d) {
                            *acc += x / nd;
                        }
                    }
                }
            }
            for j in 0..nl {
                if n[j] > 1 {
                    let u = &sum[j * e..(j + 1) * e];
                    let nn = n[j] as f64;
                    pair_cos.insert((o, s0 + j), (dot(u, u) - nn) / (nn * (nn - 1.0)));
                }
            }
        }

        // Collect.
        #[allow(clippy::needless_range_loop)]
        for sl in s0..s1 {
            let il = meta.layers[sl];
            for (o, name) in outs.iter().enumerate() {
                let j = vecof(Variant::Joint, o, sl);
                let m = vecof(Variant::MeanDiff, o, sl);
                let jn = j.map(|v| dot(v, v).sqrt());
                let mn = m.map(|v| dot(v, v).sqrt());
                let main = jn.or(mn).unwrap_or(0.0);
                let pair = |v1, v2| match (vecof(v1, o, sl), vecof(v2, o, sl)) {
                    (Some(a), Some(b)) => cos(a, b),
                    _ => None,
                };
                let masked = vecof(Variant::Masked, o, sl);
                let st = LayerStats {
                    layer: il,
                    joint_norm: jn,
                    mean_diff_norm: mn,
                    joint_over_mean_diff: jn.zip(mn).and_then(|(a, b)| (b > 0.0).then(|| a / b)),
                    resid_norm: resid[sl],
                    norm_over_resid: main / resid[sl].max(1e-30),
                    auc: proj.get(&(o, sl)).and_then(|(p, n)| auc(p, n)),
                    fe_cos: pair(Variant::Joint, Variant::Fe),
                    split_half_cos: pair(Variant::Half(0), Variant::Half(1)),
                    masked_cos: pair(Variant::Joint, Variant::Masked),
                    masked_norm_ratio: masked
                        .zip(jn)
                        .and_then(|(v, n)| (n > 0.0).then(|| dot(v, v).sqrt() / n)),
                    joint_mean_diff_cos: pair(Variant::Joint, Variant::MeanDiff),
                    pair_cos: pair_cos.get(&(o, sl)).copied(),
                };
                stats[o].push(st);
                if let Some(v) = j {
                    beta_joint
                        .entry(name.clone())
                        .or_default()
                        .insert(il as u32, v.iter().map(|&x| x as f32).collect());
                }
                if let Some(v) = m {
                    beta_md
                        .entry(name.clone())
                        .or_default()
                        .insert(il as u32, v.iter().map(|&x| x as f32).collect());
                }
            }
            let mat: Vec<Vec<f32>> = (0..n_joint)
                .map(|a| {
                    (0..n_joint)
                        .map(|b| {
                            cos(
                                vecof(Variant::Joint, a, sl).unwrap(),
                                vecof(Variant::Joint, b, sl).unwrap(),
                            )
                            .map_or(f32::NAN, |c| (c * 1000.0).round() as f32 / 1000.0)
                        })
                        .collect()
                })
                .collect();
            cosines.insert(il, mat);
        }
        s0 = s1;
    }

    // ---- report
    let conv_of = |i: usize| labelled[i].map(|r| rows[r].1.conversation.as_str());
    let mut vectors = Vec::new();
    for (o, name) in outs.iter().enumerate() {
        let is_mood = o < n_moods;
        let has_joint = o < n_joint;
        let strong = if !has_joint {
            0
        } else {
            rows.iter()
                .filter(|(_, r)| {
                    if is_mood {
                        r.scores[name] >= 3.0
                    } else {
                        r.knobs.get(name).copied().unwrap_or(0.0) > 0.5
                    }
                })
                .count()
        };
        let (n_pairs, n_conv, top) = match sets.get(name) {
            Some(ps) => {
                let mut by: BTreeMap<&str, usize> = BTreeMap::new();
                let pos: BTreeSet<usize> = ps.iter().map(|p| p.0).collect();
                for &i in &pos {
                    *by.entry(conv_of(i).unwrap_or("?")).or_default() += 1;
                }
                let top = by.values().max().copied().unwrap_or(0) as f64 / pos.len().max(1) as f64;
                (ps.len(), by.len(), top)
            }
            None => (0, 0, 0.0),
        };
        let v = has_joint.then(|| vifs[o + 1]);
        let mut flags = Vec::new();
        if has_joint && is_mood && strong < opts.thin {
            flags.push(format!("thin: {strong} lines scored 3"));
        }
        if top > 0.5 {
            flags.push(format!(
                "concentrated: {:.0}% of positives from one conversation",
                top * 100.0
            ));
        }
        if v.is_some_and(|v| v > opts.vif_flag) {
            flags.push(format!("ill-conditioned: VIF {:.1}", v.unwrap()));
        }
        if !sets.contains_key(name) {
            flags.push("no contrast set: no mean_diff, AUC or pair consistency".into());
        }
        if !has_joint {
            flags.push("set only: no joint column in the label table".into());
        }
        vectors.push(VectorReport {
            name: name.clone(),
            kind: if is_mood {
                "mood".into()
            } else if has_joint {
                "knob".into()
            } else {
                "set".into()
            },
            has_joint,
            has_set: sets.contains_key(name),
            n_strong: strong,
            n_pairs,
            n_conversations: n_conv,
            top_conversation_share: top,
            vif: v,
            shrunk_toward_mean_diff: flagged.contains(&o),
            flags,
            layers: std::mem::take(&mut stats[o]),
        });
    }
    let mid = (meta.n_layer / 2).clamp(meta.layers[0], *meta.layers.last().unwrap());
    if energy_missing > 0 {
        notes.push(format!(
            "{energy_missing} labelled lines have no audio energy; they get the mean and an indicator column"
        ));
    }
    let report = Report {
        store: store.hash()?,
        model_sha256: meta.model_sha256.clone(),
        labels_sha256: hashes.0.to_string(),
        pairs_sha256: hashes.1.to_string(),
        summary: match opts.summary {
            Summary::Mean => "line_mean".into(),
            Summary::Head => "line_head".into(),
        },
        lambda,
        ridge: opts.ridge,
        n_lines: n_store,
        n_labelled: rows.len(),
        unlabelled_store_lines: n_store - rows.len(),
        labels_not_in_store,
        pairs_dropped,
        audio_energy_missing: energy_missing,
        provisional_layer: mid,
        columns: d.cols.clone(),
        vectors,
        notes,
    };
    Ok(Fitted {
        report,
        joint: beta_joint,
        mean_diff: beta_md,
        cosines: (outs[..n_joint].to_vec(), cosines),
    })
}

#[derive(Serialize)]
struct LimitsOut {
    scale: f32,
    layers: [u32; 2],
}

/// Write one method's vectors: `natural/<name>.gguf` with every layer's
/// direction, and `<name>.gguf` plus `vectors.toml` for `scheme`.
pub fn write_method(
    dir: &Path,
    method: &str,
    vectors: &BTreeMap<String, BTreeMap<u32, Vec<f32>>>,
    scheme: Scheme,
    meta: &crate::store::Meta,
    report: &Report,
) -> Result<(), String> {
    let natural = dir.join("natural");
    for d in [dir, natural.as_path()] {
        fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        // Generated files only: clear old vectors so the bank holds exactly these.
        for entry in fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))? {
            let p = entry.map_err(|e| e.to_string())?.path();
            if p.extension().is_some_and(|x| x == "gguf") {
                fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            }
        }
    }
    let extra = |layers: String| {
        vec![
            ("vi.method", Value::Str(method.into())),
            (MODEL_SHA_KEY, Value::Str(meta.model_sha256.clone())),
            ("vi.store_sha256", Value::Str(report.store.clone())),
            ("vi.summary", Value::Str(report.summary.clone())),
            ("vi.lambda", Value::F64(report.lambda)),
            ("vi.layers", Value::Str(layers)),
        ]
    };
    let mut limits = BTreeMap::new();
    for (name, dirs) in vectors {
        let path = natural.join(format!("{name}.gguf"));
        vector_gguf(
            dirs,
            meta.n_layer,
            &meta.architecture,
            &extra("natural".into()),
        )
        .write(fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())?;
        let applied = scheme.apply(dirs)?;
        let path = dir.join(format!("{name}.gguf"));
        vector_gguf(
            &applied,
            meta.n_layer,
            &meta.architecture,
            &extra(scheme.name()),
        )
        .write(fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())?;
        limits.insert(
            name.clone(),
            LimitsOut {
                scale: 1.0,
                layers: scheme.layers(),
            },
        );
    }
    let text = format!(
        "# Written by `vi vectors`: {} vectors, {} (full strength = 1.0).\n{}",
        method,
        scheme.name(),
        toml::to_string(&limits).map_err(|e| e.to_string())?
    );
    let path = dir.join(LIMITS_FILE);
    fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Rewrite a method folder's bank for a new layer scheme, from its
/// `natural/` files. `only` limits it to some vectors.
pub fn relayer(dir: &Path, scheme: Scheme, only: Option<&[String]>) -> Result<Vec<String>, String> {
    use crate::gguf::Gguf;
    let natural = dir.join("natural");
    let limits_path = dir.join(LIMITS_FILE);
    let mut limits: BTreeMap<String, toml::Value> = match fs::read_to_string(&limits_path) {
        Ok(t) => toml::from_str(&t).map_err(|e| format!("{}: {e}", limits_path.display()))?,
        Err(_) => BTreeMap::new(),
    };
    let mut done = Vec::new();
    let mut paths: Vec<_> = fs::read_dir(&natural)
        .map_err(|e| format!("{}: {e}", natural.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .collect();
    paths.sort();
    for p in paths {
        let name = p.file_stem().unwrap().to_string_lossy().into_owned();
        if only.is_some_and(|o| !o.contains(&name)) {
            continue;
        }
        let bytes = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let mut g = Gguf::parse(&bytes).map_err(|e| format!("{}: {e}", p.display()))?;
        let dirs = crate::vectors::directions(&g)?;
        let applied = scheme.apply(&dirs).map_err(|e| format!("{name}: {e}"))?;
        g.tensors.clear();
        g.metadata
            .insert("vi.layers".into(), Value::Str(scheme.name()));
        for (l, d) in &applied {
            g.add_f32(format!("direction.{l}"), vec![d.len() as u64], d.clone());
        }
        let out = dir.join(format!("{name}.gguf"));
        g.write(fs::File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?)
            .map_err(|e| e.to_string())?;
        let entry = limits
            .entry(name.clone())
            .or_insert_with(|| toml::Value::Table(Default::default()));
        if let toml::Value::Table(t) = entry {
            let [a, b] = scheme.layers();
            t.insert(
                "layers".into(),
                toml::Value::Array(vec![i64::from(a).into(), i64::from(b).into()]),
            );
        }
        done.push(name);
    }
    let text = format!(
        "# Written by `vi vectors layers`: last scheme {} (full strength = 1.0).\n{}",
        scheme.name(),
        toml::to_string(&limits).map_err(|e| e.to_string())?
    );
    fs::write(&limits_path, text).map_err(|e| format!("{}: {e}", limits_path.display()))?;
    Ok(done)
}

/// A short readable summary of the report, at the provisional layer.
pub fn report_markdown(r: &Report) -> String {
    let mut s = format!(
        "# Vector fit report\n\nStore {}, model {}, {} of {} store lines labelled, λ = {:.3e} \
         ({} x mean diag). Summary: {}. Numbers below are at layer {} (provisional: the \
         middle of the model until the chat check picks a layer); report.json has every layer.\n\n",
        &r.store[..12.min(r.store.len())],
        &r.model_sha256[..12.min(r.model_sha256.len())],
        r.n_labelled,
        r.n_lines,
        r.lambda,
        r.ridge,
        r.summary,
        r.provisional_layer,
    );
    for n in &r.notes {
        s.push_str(&format!("- {n}\n"));
    }
    s.push_str(
        "\n| vector | kind | strong | pairs | convs | top conv | AUC | best-AUC layer | split-half | FE cos | joint/md | norm/resid | pair cos | flags |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    let f = |x: Option<f64>| x.map_or("-".to_string(), |v| format!("{v:.2}"));
    for v in &r.vectors {
        let at = v.layers.iter().find(|l| l.layer == r.provisional_layer);
        let best = v
            .layers
            .iter()
            .filter(|l| l.auc.is_some())
            .max_by(|a, b| a.auc.unwrap().total_cmp(&b.auc.unwrap()))
            .map_or("-".to_string(), |l| {
                format!("{} ({:.2})", l.layer, l.auc.unwrap())
            });
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.0}% | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            v.name,
            v.kind,
            v.n_strong,
            v.n_pairs,
            v.n_conversations,
            v.top_conversation_share * 100.0,
            f(at.and_then(|l| l.auc)),
            best,
            f(at.and_then(|l| l.split_half_cos)),
            f(at.and_then(|l| l.fe_cos)),
            f(at.and_then(|l| l.joint_over_mean_diff)),
            at.map_or("-".to_string(), |l| format!("{:.3}", l.norm_over_resid)),
            f(at.and_then(|l| l.pair_cos)),
            v.flags.join("; "),
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{LineRecord, Meta, StoreWriter};

    #[test]
    fn auc_counts_ties_as_half() {
        assert_eq!(auc(&[2.0, 3.0], &[1.0, 0.0]), Some(1.0));
        assert_eq!(auc(&[1.0], &[1.0]), Some(0.5));
        assert_eq!(auc(&[], &[1.0]), None);
    }

    /// Unit vector `i` of `n`, scaled.
    fn axis(n: usize, i: usize, s: f32) -> Vec<f32> {
        let mut v = vec![0.0; n];
        v[i] = s;
        v
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    /// A store of synthetic summaries with planted directions: mood "a" on
    /// axis 0, mood "b" on axis 1, a game offset on axis 2, layer 2 twice
    /// layer 1. "b" mostly comes with "a".
    fn synthetic(dir: &Path) -> (Vec<LabelRow>, Vec<Pair>) {
        let e = 6;
        let meta = Meta {
            version: 1,
            model_file: "m".into(),
            model_sha256: "abc".into(),
            architecture: "llama".into(),
            llama_cpp: "t".into(),
            n_embd: e,
            n_layer: 4,
            layers: vec![1, 2],
            layer_tensor: "l_out".into(),
            chat_template_sha256: "t".into(),
            messages_sha256: "m".into(),
            prefix_tokens: 1,
            head_tokens: 8,
            think: true,
            n_lines_total: 400,
            complete: false,
            preflight: None,
        };
        let mut w = StoreWriter::open(dir, meta).unwrap();
        let mut rng = Rng(42);
        let mut labels = Vec::new();
        for i in 0..400 {
            let a = (i % 4) as f64; // 0..3
            let b = if i % 4 == 3 && i % 8 == 3 {
                3.0
            } else if i % 5 == 0 {
                2.0
            } else {
                0.0
            };
            let game = if i % 3 == 0 { "ME2" } else { "ME3" };
            let mut h = vec![0.0f32; 2 * e];
            for l in 0..2 {
                let s = (l + 1) as f32;
                for j in 0..e {
                    h[l * e + j] = 1.0
                        + 0.02 * rng.next()
                        + s * (axis(e, 0, 1.0)[j] * (a / 3.0) as f32
                            + axis(e, 1, 0.5)[j] * (b / 3.0) as f32
                            + axis(e, 2, 0.7)[j] * f32::from(u8::from(game == "ME3")));
                }
            }
            w.push(
                LineRecord {
                    line_id: format!("l{i}"),
                    text_sha256: "x".into(),
                    n_prompt_tokens: 5,
                    n_line_tokens: 3 + i % 7,
                    logprob: -1.0,
                },
                &h,
                &h,
                &[1.0, 2.0],
            )
            .unwrap();
            labels.push(LabelRow {
                line_id: format!("l{i}"),
                conversation: format!("c{}", i / 10),
                game: game.into(),
                audio_energy: Some(f64::from(rng.next())),
                scores: [("a".to_string(), a), ("b".to_string(), b)].into(),
                knobs: Default::default(),
                pass1_agree: Default::default(),
            });
        }
        w.finish(None).unwrap();
        // Set "b": its positives (b=3, also a=3) against a=0, b=0 zeros,
        // so a per-set mean mixes "a" into "b".
        let pos: Vec<usize> = (0..400).filter(|i| i % 8 == 3).collect();
        let zero: Vec<usize> = (0..400).filter(|i| i % 4 == 0 && i % 5 != 0).collect();
        let mut pairs: Vec<Pair> = pos
            .iter()
            .enumerate()
            .map(|(k, &p)| Pair {
                set: "b".into(),
                pos: format!("l{p}"),
                zero: format!("l{}", zero[k % 7]),
            })
            .collect();
        pairs.push(Pair {
            set: "b".into(),
            pos: "gone".into(),
            zero: "l0".into(),
        });
        (labels, pairs)
    }

    #[test]
    fn joint_separates_correlated_moods_and_removes_the_game() {
        let dir = tempfile::tempdir().unwrap();
        let (labels, pairs) = synthetic(dir.path());
        let mut store = Store::open(dir.path()).unwrap();
        // One layer per block, to cover the blocking.
        let mut opts = FitOptions {
            mem_mb: 0,
            ..FitOptions::default()
        };
        let f = fit(&mut store, &labels, &pairs, ("l", "p"), &opts, |_| {}).unwrap();
        let close = |v: &[f32], want: &[f32], tol: f32| {
            v.iter().zip(want).all(|(x, y)| (x - y).abs() < tol)
        };
        // Joint: "a" is axis 0 (x layer), "b" axis 1 at half size, no game.
        for (l, s) in [(1u32, 1.0f32), (2, 2.0)] {
            let a = &f.joint["a"][&l];
            assert!(close(a, &axis(6, 0, s), 0.02), "a at {l}: {a:?}");
            let b = &f.joint["b"][&l];
            assert!(close(b, &axis(6, 1, 0.5 * s), 0.02), "b at {l}: {b:?}");
        }
        // mean_diff of set "b" carries "a" along (its positives are a=3 too).
        let md = &f.mean_diff["b"][&1];
        assert!(md[0] > 0.9, "{md:?}");
        assert!((md[1] - 0.5).abs() < 0.02, "{md:?}");
        let r = &f.report;
        assert_eq!(r.pairs_dropped, 1);
        assert_eq!(r.n_labelled, 400);
        let b = r.vectors.iter().find(|v| v.name == "b").unwrap();
        assert!(b.has_set && b.has_joint);
        let l1 = &b.layers[0];
        assert!(l1.auc.unwrap() > 0.95, "{l1:?}");
        assert!(l1.split_half_cos.unwrap() > 0.99, "{l1:?}");
        assert!(l1.pair_cos.unwrap() > 0.3, "{l1:?}");
        assert!(l1.joint_mean_diff_cos.unwrap() < 0.6, "{l1:?}");
        let a = r.vectors.iter().find(|v| v.name == "a").unwrap();
        assert!(a.flags.iter().any(|f| f.starts_with("no contrast set")));
        assert_eq!(f.cosines.1.len(), 2);
        // Head and mean are equal in this store, so the head fit agrees.
        opts.summary = Summary::Head;
        let g = fit(&mut store, &labels, &pairs, ("l", "p"), &opts, |_| {}).unwrap();
        assert_eq!(g.joint["a"][&2], f.joint["a"][&2]);
    }

    #[test]
    fn mean_diff_is_the_mean_of_pair_differences() {
        let dir = tempfile::tempdir().unwrap();
        let (labels, _) = synthetic(dir.path());
        let mut store = Store::open(dir.path()).unwrap();
        // One zero (l0) used by three pairs.
        let pairs: Vec<Pair> = [("l3", "l0"), ("l7", "l0"), ("l11", "l0"), ("l15", "l4")]
            .iter()
            .map(|(p, z)| Pair {
                set: "x".into(),
                pos: (*p).into(),
                zero: (*z).into(),
            })
            .collect();
        let f = fit(
            &mut store,
            &labels,
            &pairs,
            ("l", "p"),
            &FitOptions::default(),
            |_| {},
        )
        .unwrap();
        let mut want = vec![0.0f64; 6];
        let mut a = vec![0.0f32; 6];
        let mut b = vec![0.0f32; 6];
        for p in &pairs {
            let i = |id: &str| id[1..].parse::<usize>().unwrap();
            store.read(Summary::Mean, i(&p.pos), 0..1, &mut a).unwrap();
            store.read(Summary::Mean, i(&p.zero), 0..1, &mut b).unwrap();
            for j in 0..6 {
                want[j] += f64::from(a[j] - b[j]) / 4.0;
            }
        }
        let got = &f.mean_diff["x"][&1];
        for j in 0..6 {
            assert!(
                (f64::from(got[j]) - want[j]).abs() < 1e-5,
                "{got:?} {want:?}"
            );
        }
        let x = f.report.vectors.iter().find(|v| v.name == "x").unwrap();
        assert_eq!(x.kind, "set");
    }

    #[test]
    fn writes_banks_that_load_and_relayer() {
        let dir = tempfile::tempdir().unwrap();
        let (labels, pairs) = synthetic(dir.path());
        let mut store = Store::open(dir.path()).unwrap();
        let f = fit(
            &mut store,
            &labels,
            &pairs,
            ("l", "p"),
            &FitOptions::default(),
            |_| {},
        )
        .unwrap();
        let out = dir.path().join("joint");
        write_method(
            &out,
            "joint",
            &f.joint,
            Scheme::Single(2),
            &store.meta,
            &f.report,
        )
        .unwrap();
        let bank = crate::vectors::VectorBank::load_dir(&out, 6, 4).unwrap();
        assert_eq!(bank.names().collect::<Vec<_>>(), ["a", "b"]);
        // `vi chat` refuses these for any other model file.
        assert!(bank.knows_its_model());
        assert!(bank.built_for_other("abc").is_empty());
        assert_eq!(bank.built_for_other("other").len(), 2);
        let blend = bank.blend(&[("a".to_string(), 1.0)].into());
        let data = blend.data.unwrap();
        // Layer 2 only, at the natural size.
        assert!(data[..6].iter().all(|&x| x == 0.0));
        assert_eq!(&data[6..12], f.joint["a"][&2].as_slice());

        relayer(&out, Scheme::Telescoping(1, 2), Some(&["a".to_string()])).unwrap();
        let bank = crate::vectors::VectorBank::load_dir(&out, 6, 4).unwrap();
        let data = bank.blend(&[("a".to_string(), 1.0)].into()).data.unwrap();
        assert_eq!(&data[..6], f.joint["a"][&1].as_slice());
        let inc: Vec<f32> = f.joint["a"][&2]
            .iter()
            .zip(&f.joint["a"][&1])
            .map(|(x, y)| x - y)
            .collect();
        assert_eq!(&data[6..12], inc.as_slice());
        let text = fs::read_to_string(out.join(LIMITS_FILE)).unwrap();
        assert!(text.contains("layers = [1, 2]"), "{text}");
        assert!(text.contains("layers = [2, 2]"), "{text}");
    }
}
