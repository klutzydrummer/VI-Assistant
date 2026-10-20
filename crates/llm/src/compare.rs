//! Compare two captures of the same lines, or two fits, to judge whether a
//! difference between them matters: the persona-reuse capture against a
//! full decode, a GPU against the CPU, or a model against its quantization.
//!
//! - [`stores`] measures, per layer, how far one store's `line_mean`s are
//!   from the other's. It sets that against what the fit works with, the
//!   line-to-line variation, and splits it into the part shared by every
//!   line (a bias, which a contrast or the fit's intercept cancels) and the
//!   rest (noise, which averaging over lines shrinks).
//! - [`vectors`] compares two fits' natural vectors, per vector and layer.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::gguf::Gguf;
use crate::store::{Store, Summary};
use crate::vectors::directions;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerDiff {
    pub layer: usize,
    /// Per line, |a - b| / |a|: the median and the worst.
    pub median_rel: f64,
    pub worst_rel: f64,
    pub min_cos: f64,
    /// RMS of the per-line difference over the RMS of the lines' spread
    /// around their mean (in `a`): the difference against what the fit
    /// works with.
    pub vs_spread: f64,
    /// |mean difference| / RMS difference: about 1/sqrt(lines) for noise,
    /// near 1 when every line moves the same way.
    pub shared: f64,
    /// [`LayerDiff::vs_spread`] without the shared part, which the fit's
    /// intercept and every contrast cancel.
    pub noise_vs_spread: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Compare `line_mean` in two stores of the same lines and layers.
pub fn stores(a: &mut Store, b: &mut Store) -> Result<Vec<LayerDiff>, String> {
    if a.meta.layers != b.meta.layers || a.meta.n_embd != b.meta.n_embd {
        return Err("the stores captured different layers or models".into());
    }
    if a.lines.len() != b.lines.len()
        || a.lines
            .iter()
            .zip(&b.lines)
            .any(|(x, y)| x.line_id != y.line_id || x.text_sha256 != y.text_sha256)
    {
        return Err("the stores hold different lines".into());
    }
    let (e, k, n) = (a.meta.n_embd, a.meta.layers.len(), a.lines.len());
    if n < 2 {
        return Err("needs at least two lines".into());
    }
    let mut ra = vec![0.0f32; k * e];
    let mut rb = vec![0.0f32; k * e];
    let mut sum_a = vec![0.0f64; k * e];
    let mut sum_aa = vec![0.0f64; k];
    let mut sum_d = vec![0.0f64; k * e];
    let mut sum_dd = vec![0.0f64; k];
    let mut rel: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k];
    let mut min_cos = vec![1.0f64; k];
    for i in 0..n {
        a.read(Summary::Mean, i, 0..k, &mut ra)?;
        b.read(Summary::Mean, i, 0..k, &mut rb)?;
        for s in 0..k {
            let x: Vec<f64> = ra[s * e..(s + 1) * e].iter().map(|&v| v.into()).collect();
            let y: Vec<f64> = rb[s * e..(s + 1) * e].iter().map(|&v| v.into()).collect();
            let d: Vec<f64> = x.iter().zip(&y).map(|(p, q)| p - q).collect();
            let (xx, yy, dd) = (dot(&x, &x), dot(&y, &y), dot(&d, &d));
            rel[s].push(dd.sqrt() / xx.sqrt().max(1e-30));
            let cos = if xx == 0.0 && yy == 0.0 {
                1.0
            } else {
                dot(&x, &y) / (xx * yy).sqrt().max(1e-30)
            };
            min_cos[s] = min_cos[s].min(cos);
            sum_aa[s] += xx;
            sum_dd[s] += dd;
            for j in 0..e {
                sum_a[s * e + j] += x[j];
                sum_d[s * e + j] += d[j];
            }
        }
    }
    let nf = n as f64;
    Ok((0..k)
        .map(|s| {
            let mean_a = &sum_a[s * e..(s + 1) * e];
            let mean_d = &sum_d[s * e..(s + 1) * e];
            let mean_aa = dot(mean_a, mean_a) / (nf * nf);
            let spread = (sum_aa[s] / nf - mean_aa).max(0.0);
            let mean_dd = dot(mean_d, mean_d) / (nf * nf);
            let msd = sum_dd[s] / nf;
            let noise = (msd - mean_dd).max(0.0);
            let mut r = std::mem::take(&mut rel[s]);
            r.sort_by(f64::total_cmp);
            LayerDiff {
                layer: a.meta.layers[s],
                median_rel: r[r.len() / 2],
                worst_rel: r[r.len() - 1],
                min_cos: min_cos[s],
                vs_spread: (msd / spread.max(1e-30)).sqrt(),
                shared: (mean_dd / msd.max(1e-300)).sqrt(),
                noise_vs_spread: (noise / spread.max(1e-30)).sqrt(),
            }
        })
        .collect())
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VectorDiff {
    pub vector: String,
    pub layer: u32,
    pub cos: f64,
    /// |b| / |a|.
    pub norm_ratio: f64,
}

fn natural(dir: &Path) -> Result<BTreeMap<String, BTreeMap<u32, Vec<f32>>>, String> {
    let nat = dir.join("natural");
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(&nat).map_err(|e| format!("{}: {e}", nat.display()))? {
        let p = entry.map_err(|e| e.to_string())?.path();
        if p.extension().is_some_and(|x| x == "gguf") {
            let bytes = fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let g = Gguf::parse(&bytes).map_err(|e| format!("{}: {e}", p.display()))?;
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            out.insert(name, directions(&g)?);
        }
    }
    Ok(out)
}

/// Compare the natural vectors of two method folders (`.../joint`).
pub fn vectors(a: &Path, b: &Path) -> Result<Vec<VectorDiff>, String> {
    let (va, vb) = (natural(a)?, natural(b)?);
    let mut out = Vec::new();
    for (name, layers) in &va {
        let Some(other) = vb.get(name) else {
            return Err(format!("{} has no {name}", b.display()));
        };
        for (&l, x) in layers {
            let Some(y) = other.get(&l) else {
                return Err(format!("{} has no {name} at layer {l}", b.display()));
            };
            let x: Vec<f64> = x.iter().map(|&v| v.into()).collect();
            let y: Vec<f64> = y.iter().map(|&v| v.into()).collect();
            let (xx, yy) = (dot(&x, &x), dot(&y, &y));
            out.push(VectorDiff {
                vector: name.clone(),
                layer: l,
                cos: dot(&x, &y) / (xx * yy).sqrt().max(1e-30),
                norm_ratio: (yy / xx.max(1e-30)).sqrt(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{LineRecord, Meta, StoreWriter};

    fn meta(n: usize) -> Meta {
        Meta {
            version: 1,
            model_file: "m".into(),
            model_sha256: "s".into(),
            architecture: "a".into(),
            llama_cpp: "l".into(),
            n_embd: 4,
            n_layer: 3,
            layers: vec![1],
            layer_tensor: "l_out".into(),
            chat_template_sha256: "t".into(),
            messages_sha256: "m".into(),
            prefix_tokens: 1,
            head_tokens: 8,
            think: true,
            n_lines_total: n,
            complete: false,
            preflight: None,
        }
    }

    fn store(dir: &Path, rows: &[[f32; 4]]) -> Store {
        let mut w = StoreWriter::open(dir, meta(rows.len())).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let rec = LineRecord {
                line_id: format!("l{i}"),
                text_sha256: "x".into(),
                n_prompt_tokens: 1,
                n_line_tokens: 1,
                logprob: -1.0,
            };
            w.push(rec, r, r, &[1.0]).unwrap();
        }
        w.finish(None).unwrap();
        Store::open(dir).unwrap()
    }

    #[test]
    fn tells_a_shared_shift_from_noise() {
        let tmp = tempfile::tempdir().unwrap();
        let lines: Vec<[f32; 4]> = (0..64)
            .map(|i| [i as f32, (i % 5) as f32, 1.0, -(i as f32) / 2.0])
            .collect();
        let shifted: Vec<[f32; 4]> = lines
            .iter()
            .map(|r| [r[0], r[1], r[2] + 0.5, r[3]])
            .collect();
        let noisy: Vec<[f32; 4]> = lines
            .iter()
            .enumerate()
            .map(|(i, r)| {
                [
                    r[0],
                    r[1] + if i % 2 == 0 { 0.25 } else { -0.25 },
                    r[2],
                    r[3],
                ]
            })
            .collect();
        let mut a = store(&tmp.path().join("a"), &lines);
        let mut b = store(&tmp.path().join("b"), &shifted);
        let mut c = store(&tmp.path().join("c"), &noisy);
        let s = &stores(&mut a, &mut b).unwrap()[0];
        assert!(s.shared > 0.99, "{s:?}");
        assert!(s.noise_vs_spread < 1e-3, "{s:?}");
        let n = &stores(&mut a, &mut c).unwrap()[0];
        assert!(n.shared < 0.05, "{n:?}");
        assert!(n.noise_vs_spread > 1e-2, "{n:?}");
        let mut same = store(&tmp.path().join("d"), &lines);
        assert_eq!(stores(&mut a, &mut same).unwrap()[0].vs_spread, 0.0);
    }
}
