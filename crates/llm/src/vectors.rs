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

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.vectors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&Vector> {
        self.vectors.get(name)
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

#[cfg(test)]
mod tests {
    use super::*;

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
