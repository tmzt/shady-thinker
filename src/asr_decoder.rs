//! GPU-accelerated Qwen3-ASR decoder (bf16 causal LLM).
//!
//! Takes encoder output embeddings + prompt tokens, runs prefill + autoregressive decode.
//! Weights are bf16, computation in f32 (same as encoder).

use std::path::{Path, PathBuf};
use safetensors::SafeTensors;
use crate::gpu::GpuContext;

mod shaders {
    pub const BF16_MATVEC: &str = include_str!("shaders/bf16_matvec.wgsl");
    pub const BF16_GEMM: &str = include_str!("shaders/bf16_gemm.wgsl");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.wgsl");
    pub const ADD_RMSNORM: &str = include_str!("shaders/add_rmsnorm.wgsl");
    pub const SILU_MUL: &str = include_str!("shaders/silu_mul.wgsl");
    pub const EMBEDDING: &str = include_str!("shaders/embedding.wgsl");
    pub const ARGMAX: &str = include_str!("shaders/argmax.wgsl");
    pub const GQA_ATTN: &str = include_str!("shaders/gqa_attention_head.wgsl");
    pub const GQA_REDUCE: &str = include_str!("shaders/gqa_reduce.wgsl");
    pub const FUSED_QKNORM: &str = include_str!("shaders/fused_split_qknorm_kvstore.wgsl");
    pub const SIGMOID_MUL: &str = include_str!("shaders/sigmoid_mul.wgsl");
}

#[derive(Debug, Clone)]
pub struct AsrDecoderConfig {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,  // Q heads
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
}

/// Per-layer bf16 weights on GPU.
struct DecoderLayer {
    // Attention (bf16 packed as u32)
    q_proj: wgpu::Buffer,   // [num_q_heads * head_dim, hidden]
    k_proj: wgpu::Buffer,   // [num_kv_heads * head_dim, hidden]
    v_proj: wgpu::Buffer,   // [num_kv_heads * head_dim, hidden]
    o_proj: wgpu::Buffer,   // [hidden, num_q_heads * head_dim]
    // Q/K norms (bf16 packed)
    q_norm: wgpu::Buffer,   // [head_dim]
    k_norm: wgpu::Buffer,   // [head_dim]
    // MLP (bf16 packed)
    gate_proj: wgpu::Buffer, // [intermediate, hidden]
    up_proj: wgpu::Buffer,   // [intermediate, hidden]
    down_proj: wgpu::Buffer, // [hidden, intermediate]
    // Norms (bf16 packed)
    input_layernorm: wgpu::Buffer,
    post_attn_layernorm: wgpu::Buffer,
}

/// GPU ASR decoder.
pub struct AsrDecoder {
    gpu: GpuContext,
    layers: Vec<DecoderLayer>,
    embed_tokens: Vec<wgpu::Buffer>, // chunked embedding table
    final_norm: wgpu::Buffer,
    lm_head: Vec<wgpu::Buffer>,      // chunked lm_head
    pub config: AsrDecoderConfig,
    // Chunk info for embedding/lm_head
    embed_chunk_size: u32,  // tokens per chunk
}

impl AsrDecoder {
    pub fn new(model_dir: &Path, max_seq_len: u32) -> Self {
        let gpu = GpuContext::new();
        Self::load(gpu, model_dir, max_seq_len)
    }

    pub fn load(gpu: GpuContext, model_dir: &Path, max_seq_len: u32) -> Self {
        log::info!("[asr-decoder] loading from {:?}", model_dir);

        let config = Self::parse_config(&model_dir.join("config.json"), max_seq_len);
        log::info!("[asr-decoder] config: {:?}", config);

        let shard_data = Self::load_shards(model_dir);
        let shards: Vec<SafeTensors> = shard_data.iter()
            .map(|d| SafeTensors::deserialize(d).expect("parse safetensors"))
            .collect();

        let get = |name: &str| -> &[u8] {
            for st in &shards {
                if let Ok(t) = st.tensor(name) { return t.data(); }
            }
            panic!("[asr-decoder] tensor not found: {name}");
        };

        let upload = |label: &str, name: &str| -> wgpu::Buffer {
            gpu.upload_buffer(label, get(name))
        };

        // Load layers
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("thinker.model.layers.{i}");
            layers.push(DecoderLayer {
                q_proj: upload(&format!("{p}.q"), &format!("{p}.self_attn.q_proj.weight")),
                k_proj: upload(&format!("{p}.k"), &format!("{p}.self_attn.k_proj.weight")),
                v_proj: upload(&format!("{p}.v"), &format!("{p}.self_attn.v_proj.weight")),
                o_proj: upload(&format!("{p}.o"), &format!("{p}.self_attn.o_proj.weight")),
                q_norm: upload(&format!("{p}.qn"), &format!("{p}.self_attn.q_norm.weight")),
                k_norm: upload(&format!("{p}.kn"), &format!("{p}.self_attn.k_norm.weight")),
                gate_proj: upload(&format!("{p}.gate"), &format!("{p}.mlp.gate_proj.weight")),
                up_proj: upload(&format!("{p}.up"), &format!("{p}.mlp.up_proj.weight")),
                down_proj: upload(&format!("{p}.down"), &format!("{p}.mlp.down_proj.weight")),
                input_layernorm: upload(&format!("{p}.in"), &format!("{p}.input_layernorm.weight")),
                post_attn_layernorm: upload(&format!("{p}.pa"), &format!("{p}.post_attention_layernorm.weight")),
            });
            if (i + 1) % 7 == 0 {
                log::info!("[asr-decoder] loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        // Embedding table — chunk into 128MB pieces
        let embed_data = get("thinker.model.embed_tokens.weight");
        let bytes_per_token = config.hidden_size as usize; // bf16 packed = hidden/2 u32 = hidden bytes
        let max_chunk_bytes = 120 * 1024 * 1024; // 120MB per chunk (under 128MB limit)
        let tokens_per_chunk = max_chunk_bytes / bytes_per_token;
        let n_chunks = (config.vocab_size as usize + tokens_per_chunk - 1) / tokens_per_chunk;

        let mut embed_tokens = Vec::new();
        for c in 0..n_chunks {
            let start = c * tokens_per_chunk * bytes_per_token;
            let end = ((c + 1) * tokens_per_chunk * bytes_per_token).min(embed_data.len());
            embed_tokens.push(gpu.upload_buffer(
                &format!("embed.{c}"), &embed_data[start..end]));
        }
        log::info!("[asr-decoder] embedding: {} chunks of {} tokens", n_chunks, tokens_per_chunk);

        // lm_head — same chunking (or tied to embedding)
        let lm_head_data = get("thinker.lm_head.weight");
        let mut lm_head = Vec::new();
        for c in 0..n_chunks {
            let start = c * tokens_per_chunk * bytes_per_token;
            let end = ((c + 1) * tokens_per_chunk * bytes_per_token).min(lm_head_data.len());
            lm_head.push(gpu.upload_buffer(
                &format!("lm_head.{c}"), &lm_head_data[start..end]));
        }

        let final_norm = upload("dec.fn", "thinker.model.norm.weight");

        log::info!("[asr-decoder] loaded {} layers + {} embed chunks", layers.len(), n_chunks);

        Self {
            gpu, layers, embed_tokens, final_norm, lm_head, config,
            embed_chunk_size: tokens_per_chunk as u32,
        }
    }

    /// Run a single autoregressive decode step.
    /// Returns the predicted token ID.
    pub fn forward_token(&mut self, token_id: u32) -> u32 {
        // TODO: implement — needs KV cache allocation, embedding lookup,
        // layer loop with bf16_matvec, attention, MLP, lm_head, argmax
        log::warn!("[asr-decoder] forward_token not yet implemented");
        0
    }

    /// Prefill a sequence of embeddings (from encoder output + prompt).
    /// Sets up the KV cache for subsequent autoregressive decode.
    pub fn prefill(&mut self, _embeddings: &[f32], _seq_len: u32) {
        // TODO: implement — batched bf16_gemm through all layers
        log::warn!("[asr-decoder] prefill not yet implemented");
    }

    /// Generate tokens autoregressively until EOS or max_tokens.
    pub fn generate(&mut self, _prompt_embeddings: &[f32], _seq_len: u32, _max_tokens: u32) -> Vec<u32> {
        // TODO: implement — prefill + decode loop
        log::warn!("[asr-decoder] generate not yet implemented");
        Vec::new()
    }

    fn parse_config(path: &Path, max_seq_len: u32) -> AsrDecoderConfig {
        let text = std::fs::read_to_string(path).expect("config.json");
        let v: serde_json::Value = serde_json::from_str(&text).expect("parse json");

        let dec = if v["thinker_config"]["text_config"].is_object() {
            &v["thinker_config"]["text_config"]
        } else {
            &v
        };

        AsrDecoderConfig {
            hidden_size: dec["hidden_size"].as_u64().unwrap_or(1024) as u32,
            num_layers: dec["num_hidden_layers"].as_u64().unwrap_or(28) as u32,
            num_attention_heads: dec["num_attention_heads"].as_u64().unwrap_or(16) as u32,
            num_kv_heads: dec["num_key_value_heads"].as_u64().unwrap_or(8) as u32,
            head_dim: dec["head_dim"].as_u64().unwrap_or(128) as u32,
            intermediate_size: dec["intermediate_size"].as_u64().unwrap_or(3072) as u32,
            vocab_size: dec["vocab_size"].as_u64().unwrap_or(151936) as u32,
            max_seq_len,
        }
    }

    fn load_shards(model_dir: &Path) -> Vec<Vec<u8>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(model_dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        files.sort();
        log::info!("[asr-decoder] loading {} shard(s)", files.len());
        files.iter().map(|p| std::fs::read(p).expect("read shard")).collect()
    }
}
