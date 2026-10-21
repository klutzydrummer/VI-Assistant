//! Control vectors for the dialogue model, and blending them per turn.
//!
//! A vector file is what llama.cpp's `cvector-generator` writes: a GGUF
//! with one F32 tensor `direction.<layer>` of `n_embd` floats per layer,
//! layers counted from 1. A folder holds one file per vector, named after
//! it (`drunk.gguf`), plus an optional `vectors.toml` with limits:
//!
//! ```toml
//! [drunk]
//! scale = 1.5       # strength 1.0 from the mixing table means this much
//! cap = 1.2         # never apply more than this, whatever the table says
//! layers = [10, 20] # only steer these layers (inclusive)
//! ```
//!
//! The blend is sum(strength x vector) over every vector, laid out the way
//! `llama_set_adapter_cvec` wants it: layer 1 first, `n_embd` floats each,
//! one block per layer from 1 to `n_layer - 1`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::gguf::Gguf;

/// File in a vector folder holding per-vector limits.
pub const LIMITS_FILE: &str = "vectors.toml";

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    #[serde(default = "one")]
    pub scale: f32,
    #[serde(default = "one")]
    pub cap: f32,
    /// First and last layer to steer, inclusive.
    pub layers: Option<[u32; 2]>,
}

fn one() -> f32 {
    1.0
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            scale: 1.0,
            cap: 1.0,
            layers: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Vector {
    /// `n_embd * (n_layer - 1)` floats, zero outside the steered layers.
    data: Vec<f32>,
    pub limits: Limits,
    /// SHA-256 of the file it came from, so a worker can cache by content.
    pub hash: String,
}

impl Vector {
    /// Build from a parsed `cvector-generator` file for a model with
    /// `n_embd` and `n_layer`.
    pub fn from_gguf(
        g: &Gguf,
        n_embd: usize,
        n_layer: usize,
        limits: Limits,
    ) -> Result<Self, String> {
        if n_layer < 2 {
            return Err("the model needs at least two layers to steer".into());
        }
        let mut data = vec![0.0; n_embd * (n_layer - 1)];
        let mut found = 0;
        for t in &g.tensors {
            let Some(layer) = t.name.strip_prefix("direction.") else {
                continue;
            };
            let layer: usize = layer
                .parse()
                .map_err(|_| format!("bad layer in tensor name {:?}", t.name))?;
            if layer == 0 || layer >= n_layer {
                return Err(format!(
                    "{:?} is outside the model's layers 1..{}",
                    t.name,
                    n_layer - 1
                ));
            }
            let dir = t
                .data
                .as_ref()
                .ok_or_else(|| format!("{:?} is not F32", t.name))?;
            if dir.len() != n_embd || t.dims.len() != 1 {
                return Err(format!(
                    "{:?} has {} values, the model's n_embd is {n_embd}",
                    t.name,
                    dir.len()
                ));
            }
            if let Some([first, last]) = limits.layers {
                if layer < first as usize || layer > last as usize {
                    continue;
                }
            }
            let block = &mut data[(layer - 1) * n_embd..layer * n_embd];
            for (d, x) in block.iter_mut().zip(dir) {
                *d += x;
            }
            found += 1;
        }
        if found == 0 {
            return Err("no direction tensors in the steered layers".into());
        }
        Ok(Self {
            data,
            limits,
            hash: String::new(),
        })
    }

    /// The strength actually applied for a mixing-table strength.
    pub fn effective(&self, strength: f32) -> f32 {
        let cap = self.limits.cap.abs();
        (strength * self.limits.scale).clamp(-cap, cap)
    }
}

/// Every vector available for one model.
#[derive(Debug, Clone)]
pub struct VectorBank {
    n_embd: usize,
    n_layer: usize,
    vectors: BTreeMap<String, Vector>,
    /// Vector name to the model SHA-256 its file says it was built for.
    built_for: BTreeMap<String, String>,
}

/// One turn's blended vector.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Blend {
    /// The buffer for `llama_set_adapter_cvec`, or `None` when nothing
    /// steers this turn.
    pub data: Option<Vec<f32>>,
    /// Vector name to the strength applied, after scale and cap.
    pub applied: BTreeMap<String, f32>,
    /// Names the mixing table asked for that the bank doesn't have yet.
    pub missing: Vec<String>,
}

impl VectorBank {
    pub fn empty(n_embd: usize, n_layer: usize) -> Self {
        Self {
            n_embd,
            n_layer,
            vectors: BTreeMap::new(),
            built_for: BTreeMap::new(),
        }
    }

    /// Load every `*.gguf` in `dir`, with limits from its `vectors.toml`.
    pub fn load_dir(dir: &Path, n_embd: usize, n_layer: usize) -> Result<Self, String> {
        let mut bank = Self::empty(n_embd, n_layer);
        let limits_path = dir.join(LIMITS_FILE);
        let mut limits: BTreeMap<String, Limits> = if limits_path.exists() {
            let text = fs::read_to_string(&limits_path)
                .map_err(|e| format!("{}: {e}", limits_path.display()))?;
            toml::from_str(&text).map_err(|e| format!("{}: {e}", limits_path.display()))?
        } else {
            BTreeMap::new()
        };
        let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .collect();
        paths.sort();
        for path in paths {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let g = Gguf::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut v = Vector::from_gguf(
                &g,
                n_embd,
                n_layer,
                limits.remove(&name).unwrap_or_default(),
            )
            .map_err(|e| format!("{}: {e}", path.display()))?;
            v.hash = hex(&Sha256::digest(&bytes));
            if let Some(sha) = g.metadata.get(MODEL_SHA_KEY).and_then(|v| v.as_str()) {
                bank.built_for.insert(name.clone(), sha.to_string());
            }
            bank.vectors.insert(name, v);
        }
        if let Some(name) = limits.keys().next() {
            return Err(format!(
                "{} sets limits for {name:?}, which has no {name}.gguf",
                limits_path.display()
            ));
        }
        Ok(bank)
    }

    pub fn insert(&mut self, name: impl Into<String>, vector: Vector) -> Result<(), String> {
        if vector.data.len() != self.n_embd * (self.n_layer - 1) {
            return Err("vector was built for a different model shape".into());
        }
        self.vectors.insert(name.into(), vector);
        Ok(())
    }

    /// Vectors whose file names a model other than `model_sha256`.
    pub fn built_for_other(&self, model_sha256: &str) -> Vec<(&str, &str)> {
        self.built_for
            .iter()
            .filter(|(_, sha)| *sha != model_sha256)
            .map(|(n, sha)| (n.as_str(), sha.as_str()))
            .collect()
    }

    /// Whether any vector says which model it was built for.
    pub fn knows_its_model(&self) -> bool {
        !self.built_for.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.vectors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&Vector> {
        self.vectors.get(name)
    }

    /// Add `name`: vector `src` with each layer's direction swapped for a
    /// random one of the same length, the same on every run for a `seed`.
    /// A control for the chat check: a push this size in no particular
    /// direction.
    pub fn insert_random_like(&mut self, name: &str, src: &str, seed: u64) -> Result<(), String> {
        let v = self
            .vectors
            .get(src)
            .ok_or_else(|| format!("{name}: there's no vector {src:?} to match"))?;
        let mut rng = SplitMix(seed);
        let mut data = vec![0.0f32; v.data.len()];
        for (from, to) in v.data.chunks(self.n_embd).zip(data.chunks_mut(self.n_embd)) {
            let norm = from
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm == 0.0 {
                continue;
            }
            let g: Vec<f64> = (0..self.n_embd).map(|_| rng.gaussian()).collect();
            let gn = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            for (d, x) in to.iter_mut().zip(&g) {
                *d = (x * norm / gn) as f32;
            }
        }
        let random = Vector {
            data,
            limits: v.limits.clone(),
            hash: String::new(),
        };
        self.vectors.insert(name.to_string(), random);
        Ok(())
    }

    pub fn blend(&self, strengths: &BTreeMap<String, f32>) -> Blend {
        let mut blend = Blend::default();
        let mut data = vec![0.0f32; self.n_embd * (self.n_layer - 1)];
        for (name, &strength) in strengths {
            let Some(v) = self.vectors.get(name) else {
                if strength != 0.0 {
                    blend.missing.push(name.clone());
                }
                continue;
            };
            let s = v.effective(strength);
            if s == 0.0 {
                continue;
            }
            for (d, x) in data.iter_mut().zip(&v.data) {
                *d += s * x;
            }
            blend.applied.insert(name.clone(), s);
        }
        if !blend.applied.is_empty() {
            blend.data = Some(data);
        }
        blend
    }
}

/// SplitMix64, enough randomness for a control vector without a crate.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    fn gaussian(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A `cvector-generator`-style file with the same direction on every layer.
/// For tests and for hand-made vectors.
pub fn uniform_gguf(direction: &[f32], n_layer: usize) -> Gguf {
    use crate::gguf::Value;
    let mut g = Gguf::default();
    g.metadata.insert(
        "general.architecture".into(),
        Value::Str("controlvector".into()),
    );
    g.metadata.insert(
        "controlvector.layer_count".into(),
        Value::I32(n_layer as i32 - 1),
    );
    for layer in 1..n_layer {
        g.add_f32(
            format!("direction.{layer}"),
            vec![direction.len() as u64],
            direction.to_vec(),
        );
    }
    g
}

/// How a vector's per-layer directions are applied across layers. The
/// builder measures a direction `d_L` at every layer's output; "full
/// strength" is the best single layer at 1.0 (Brandon, 2026-10-20).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// `d_L` at layer L only.
    Single(u32),
    /// `d_A` at A, then each later layer's own extra gap `d_L - d_(L-1)`:
    /// the stream carries each layer's measured gap, at full strength.
    Telescoping(u32, u32),
    /// `d_L / K` on each of the K layers (comparison only).
    OverK(u32, u32),
    /// `d_L / sqrt(K)` on each of the K layers (comparison only).
    OverSqrtK(u32, u32),
}

impl Scheme {
    /// `single:32`, `band:24-40` (telescoping), `band-k:24-40`, `band-sqrtk:24-40`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let bad = || format!("{s:?}: expected single:L, band:A-B, band-k:A-B or band-sqrtk:A-B");
        let (kind, arg) = s.split_once(':').ok_or_else(bad)?;
        if kind == "single" {
            return Ok(Self::Single(arg.parse().map_err(|_| bad())?));
        }
        let (a, b) = arg.split_once('-').ok_or_else(bad)?;
        let (a, b): (u32, u32) = (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
        if a == 0 || b < a {
            return Err(bad());
        }
        match kind {
            "band" => Ok(Self::Telescoping(a, b)),
            "band-k" => Ok(Self::OverK(a, b)),
            "band-sqrtk" => Ok(Self::OverSqrtK(a, b)),
            _ => Err(bad()),
        }
    }

    pub fn layers(&self) -> [u32; 2] {
        match *self {
            Self::Single(l) => [l, l],
            Self::Telescoping(a, b) | Self::OverK(a, b) | Self::OverSqrtK(a, b) => [a, b],
        }
    }

    /// Apply to a vector's natural per-layer directions.
    pub fn apply(
        &self,
        natural: &BTreeMap<u32, Vec<f32>>,
    ) -> Result<BTreeMap<u32, Vec<f32>>, String> {
        let [a, b] = self.layers();
        let get = |l: u32| {
            natural
                .get(&l)
                .ok_or_else(|| format!("no direction measured at layer {l}"))
        };
        let k = (b - a + 1) as f32;
        let mut out = BTreeMap::new();
        for l in a..=b {
            let d = get(l)?;
            let v = match *self {
                Self::Single(_) => d.clone(),
                Self::Telescoping(..) if l == a => d.clone(),
                Self::Telescoping(..) => {
                    let prev = get(l - 1)?;
                    d.iter().zip(prev).map(|(x, p)| x - p).collect()
                }
                Self::OverK(..) => d.iter().map(|x| x / k).collect(),
                Self::OverSqrtK(..) => d.iter().map(|x| x / k.sqrt()).collect(),
            };
            out.insert(l, v);
        }
        Ok(out)
    }

    pub fn name(&self) -> String {
        match *self {
            Self::Single(l) => format!("single:{l}"),
            Self::Telescoping(a, b) => format!("band:{a}-{b}"),
            Self::OverK(a, b) => format!("band-k:{a}-{b}"),
            Self::OverSqrtK(a, b) => format!("band-sqrtk:{a}-{b}"),
        }
    }
}

/// A vector file's `direction.<layer>` tensors.
pub fn directions(g: &Gguf) -> Result<BTreeMap<u32, Vec<f32>>, String> {
    let mut out = BTreeMap::new();
    for t in &g.tensors {
        if let Some(l) = t.name.strip_prefix("direction.") {
            let l = l
                .parse()
                .map_err(|_| format!("bad layer in tensor name {:?}", t.name))?;
            let d = t
                .data
                .clone()
                .ok_or_else(|| format!("{:?} is not F32", t.name))?;
            out.insert(l, d);
        }
    }
    Ok(out)
}

/// A control-vector file holding `dirs`, with extra metadata.
pub fn vector_gguf(
    dirs: &BTreeMap<u32, Vec<f32>>,
    n_layer: usize,
    model_hint: &str,
    extra: &[(&str, crate::gguf::Value)],
) -> Gguf {
    use crate::gguf::Value;
    let mut g = Gguf::default();
    g.metadata.insert(
        "general.architecture".into(),
        Value::Str("controlvector".into()),
    );
    g.metadata.insert(
        "controlvector.model_hint".into(),
        Value::Str(model_hint.into()),
    );
    g.metadata.insert(
        "controlvector.layer_count".into(),
        Value::I32(n_layer as i32 - 1),
    );
    for (k, v) in extra {
        g.metadata.insert((*k).into(), v.clone());
    }
    for (l, d) in dirs {
        g.add_f32(format!("direction.{l}"), vec![d.len() as u64], d.clone());
    }
    g
}

/// The model SHA-256 a vector file was built for, if it says.
pub const MODEL_SHA_KEY: &str = "vi.model_sha256";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemes_telescope_and_compare() {
        let nat: BTreeMap<u32, Vec<f32>> = [
            (1, vec![1.0]),
            (2, vec![3.0]),
            (3, vec![4.0]),
            (4, vec![8.0]),
        ]
        .into();
        let t = Scheme::parse("band:2-4").unwrap().apply(&nat).unwrap();
        assert_eq!(t, [(2, vec![3.0]), (3, vec![1.0]), (4, vec![4.0])].into());
        // The increments add back up to the last layer's full gap.
        assert_eq!(t.values().map(|v| v[0]).sum::<f32>(), 8.0);
        let s = Scheme::parse("single:3").unwrap().apply(&nat).unwrap();
        assert_eq!(s, [(3, vec![4.0])].into());
        let k = Scheme::parse("band-k:1-2").unwrap().apply(&nat).unwrap();
        assert_eq!(k, [(1, vec![0.5]), (2, vec![1.5])].into());
        assert!(Scheme::parse("band:4-2").is_err());
        assert!(Scheme::parse("band:0-2").is_err());
        assert!(Scheme::parse("band:3-5").unwrap().apply(&nat).is_err());
        assert_eq!(
            Scheme::parse("band-sqrtk:2-4").unwrap().name(),
            "band-sqrtk:2-4"
        );
    }

    fn bank() -> VectorBank {
        // n_embd 2, 3 layers: blocks for layers 1 and 2.
        let mut bank = VectorBank::empty(2, 3);
        let drunk = Vector::from_gguf(&uniform_gguf(&[1.0, 0.0], 3), 2, 3, Limits::default());
        bank.insert("drunk", drunk.unwrap()).unwrap();
        let sad = Vector::from_gguf(
            &uniform_gguf(&[0.0, 1.0], 3),
            2,
            3,
            Limits {
                scale: 2.0,
                cap: 0.5,
                layers: Some([2, 2]),
            },
        );
        bank.insert("sad", sad.unwrap()).unwrap();
        bank
    }

    fn strengths(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn blends_scaled_and_capped_vectors_per_layer() {
        let b = bank().blend(&strengths(&[("drunk", 0.4), ("sad", 0.2)]));
        // sad: 0.2 x scale 2 = 0.4, under the cap, layer 2 only.
        assert_eq!(b.data.unwrap(), vec![0.4, 0.0, 0.4, 0.4]);
        assert_eq!(b.applied, strengths(&[("drunk", 0.4), ("sad", 0.4)]));
        assert!(b.missing.is_empty());

        let b = bank().blend(&strengths(&[("sad", 0.9)]));
        assert_eq!(b.applied["sad"], 0.5);
    }

    #[test]
    fn random_controls_match_each_layers_length() {
        let mut b = VectorBank::empty(8, 4);
        let dirs: BTreeMap<u32, Vec<f32>> = [(1, vec![3.0; 8]), (3, vec![0.5; 8])].into();
        let v = Vector::from_gguf(&vector_gguf(&dirs, 4, "", &[]), 8, 4, Limits::default());
        b.insert("warmth", v.unwrap()).unwrap();
        b.insert_random_like("random:warmth", "warmth", 7).unwrap();
        let w = b.get("warmth").unwrap().data.clone();
        let r = b.get("random:warmth").unwrap().data.clone();
        let norm = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>().sqrt();
        for (a, c) in w.chunks(8).zip(r.chunks(8)) {
            assert!((norm(a) - norm(c)).abs() < 1e-4, "{a:?} vs {c:?}");
        }
        // Layer 2 had no direction and stays empty; the others point elsewhere.
        assert_eq!(norm(&r[8..16]), 0.0);
        assert!(r[..8].iter().zip(&w[..8]).any(|(x, y)| (x - y).abs() > 0.1));
        // The same seed gives the same vector.
        b.insert_random_like("again", "warmth", 7).unwrap();
        assert_eq!(b.get("again").unwrap().data, r);
        assert!(b.insert_random_like("random:nope", "nope", 7).is_err());
    }

    #[test]
    fn nothing_to_steer_means_no_buffer() {
        let b = bank().blend(&strengths(&[("drunk", 0.0), ("melancholy", 0.3)]));
        assert_eq!(b.data, None);
        assert_eq!(b.missing, vec!["melancholy"]);
    }

    #[test]
    fn rejects_vectors_for_another_model() {
        let g = uniform_gguf(&[1.0, 0.0, 0.0], 3);
        assert!(Vector::from_gguf(&g, 2, 3, Limits::default()).is_err());
        let g = uniform_gguf(&[1.0, 0.0], 5);
        assert!(Vector::from_gguf(&g, 2, 3, Limits::default()).is_err());
    }

    #[test]
    fn loads_a_folder_with_limits() {
        let dir = std::env::temp_dir().join(format!("vi-vectors-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        uniform_gguf(&[1.0, 2.0], 3)
            .write(fs::File::create(dir.join("drunk.gguf")).unwrap())
            .unwrap();
        fs::write(dir.join(LIMITS_FILE), "[drunk]\ncap = 0.3\n").unwrap();
        let bank = VectorBank::load_dir(&dir, 2, 3).unwrap();
        assert_eq!(bank.names().collect::<Vec<_>>(), vec!["drunk"]);
        assert_eq!(bank.get("drunk").unwrap().effective(1.0), 0.3);
        assert_eq!(bank.get("drunk").unwrap().hash.len(), 64);

        fs::write(dir.join(LIMITS_FILE), "[sober]\ncap = 0.3\n").unwrap();
        assert!(VectorBank::load_dir(&dir, 2, 3).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
