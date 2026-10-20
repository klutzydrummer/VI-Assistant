//! A tiny llama-architecture model with random weights, written on the fly
//! so tests run the real llama.cpp code path without downloading anything.
//! It speaks gibberish, but deterministically, which is all a test needs.

use std::path::Path;

use crate::gguf::{Gguf, Value};

pub const N_EMBD: usize = 64;
pub const N_LAYER: usize = 3;
const N_HEAD: usize = 4;
const N_FF: usize = 128;
const N_CTX: u32 = 16384;

/// `<unk>`, `<s>`, `</s>`, one token per byte, then `<think>` and
/// `</think>` (user-defined, as in Qwen's vocabulary).
const N_VOCAB: usize = 3 + 256 + 2;

const CHATML: &str = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

pub fn write(path: &Path, seed: u64) -> std::io::Result<()> {
    build(seed).write(std::fs::File::create(path)?)
}

/// [`write`] with `n_layer` layers instead of [`N_LAYER`].
pub fn write_layers(path: &Path, seed: u64, n_layer: usize) -> std::io::Result<()> {
    build_layers(seed, n_layer).write(std::fs::File::create(path)?)
}

pub fn build(seed: u64) -> Gguf {
    build_layers(seed, N_LAYER)
}

pub fn build_layers(seed: u64, n_layer: usize) -> Gguf {
    let mut g = Gguf::default();
    let mut kv = |k: &str, v: Value| {
        g.metadata.insert(k.to_string(), v);
    };
    kv("general.architecture", Value::Str("llama".into()));
    kv("general.name", Value::Str("vi-test-tiny".into()));
    kv("llama.context_length", Value::U32(N_CTX));
    kv("llama.embedding_length", Value::U32(N_EMBD as u32));
    kv("llama.block_count", Value::U32(n_layer as u32));
    kv("llama.feed_forward_length", Value::U32(N_FF as u32));
    kv("llama.attention.head_count", Value::U32(N_HEAD as u32));
    kv("llama.attention.head_count_kv", Value::U32(N_HEAD as u32));
    kv(
        "llama.rope.dimension_count",
        Value::U32((N_EMBD / N_HEAD) as u32),
    );
    kv("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5));
    kv("tokenizer.ggml.model", Value::Str("llama".into()));
    let mut tokens = vec![
        Value::Str("<unk>".into()),
        Value::Str("<s>".into()),
        Value::Str("</s>".into()),
    ];
    // Token types: 2 unknown, 3 control, 4 user-defined, 6 byte.
    let mut types = vec![Value::I32(2), Value::I32(3), Value::I32(3)];
    for b in 0..=255u8 {
        tokens.push(Value::Str(format!("<0x{b:02X}>")));
        types.push(Value::I32(6));
    }
    for t in ["<think>", "</think>"] {
        tokens.push(Value::Str(t.into()));
        types.push(Value::I32(4));
    }
    kv(
        "tokenizer.ggml.scores",
        Value::Array(vec![Value::F32(0.0); N_VOCAB]),
    );
    kv("tokenizer.ggml.tokens", Value::Array(tokens));
    kv("tokenizer.ggml.token_type", Value::Array(types));
    kv("tokenizer.ggml.bos_token_id", Value::U32(1));
    kv("tokenizer.ggml.eos_token_id", Value::U32(2));
    kv("tokenizer.ggml.unknown_token_id", Value::U32(0));
    kv("tokenizer.chat_template", Value::Str(CHATML.into()));

    let mut rng = Rng(seed | 1);
    let (e, f, v) = (N_EMBD as u64, N_FF as u64, N_VOCAB as u64);
    let mut add = |g: &mut Gguf, name: String, dims: Vec<u64>, scale: f32| {
        let n = dims.iter().product::<u64>() as usize;
        let data = if scale == 0.0 {
            vec![1.0; n] // norm weights
        } else {
            (0..n).map(|_| rng.uniform() * scale).collect()
        };
        g.add_f32(name, dims, data);
    };
    let s = 1.0 / (N_EMBD as f32).sqrt();
    add(&mut g, "token_embd.weight".into(), vec![e, v], 1.0);
    add(&mut g, "output_norm.weight".into(), vec![e], 0.0);
    add(&mut g, "output.weight".into(), vec![e, v], s);
    for i in 0..n_layer {
        add(&mut g, format!("blk.{i}.attn_norm.weight"), vec![e], 0.0);
        for w in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            add(&mut g, format!("blk.{i}.{w}.weight"), vec![e, e], s);
        }
        add(&mut g, format!("blk.{i}.ffn_norm.weight"), vec![e], 0.0);
        add(&mut g, format!("blk.{i}.ffn_gate.weight"), vec![e, f], s);
        add(&mut g, format!("blk.{i}.ffn_up.weight"), vec![e, f], s);
        add(&mut g, format!("blk.{i}.ffn_down.weight"), vec![f, e], s);
    }
    g
}

/// xorshift64*, enough for reproducible weights.
struct Rng(u64);

impl Rng {
    /// Uniform in [-1, 1).
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        x as f32 / (1u64 << 23) as f32 - 1.0
    }
}
