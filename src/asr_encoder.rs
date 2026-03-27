//! GPU-accelerated Qwen-ASR encoder transformer.
//!
//! Takes conv-stem output (token embeddings) from the C pipeline,
//! runs the transformer stack on GPU, returns encoder output.
//! The C pipeline handles mel spectrogram + conv stem (CPU)
//! and decoder (CPU). This module handles the encoder transformer
//! which is the main bottleneck.

use std::path::{Path, PathBuf};
use safetensors::SafeTensors;
use crate::gpu::GpuContext;

mod shaders {
    pub const BF16_GEMM: &str = include_str!("shaders/bf16_gemm.wgsl");
    pub const LAYERNORM: &str = include_str!("shaders/layernorm.wgsl");
    pub const GELU_MUL: &str = include_str!("shaders/gelu_mul.wgsl");
    pub const BIDIR_ATTN: &str = include_str!("shaders/qwen_asr_bidir_attn.wgsl");
    pub const ADD: &str = include_str!("shaders/add.wgsl");
}

/// Encoder configuration (matches qwen_asr_enc_config_t in C).
#[derive(Debug, Clone)]
pub struct AsrEncoderConfig {
    pub d_model: u32,       // 1024 (0.6B) or 896 (1.7B)
    pub num_layers: u32,    // 24 (0.6B) or 18 (1.7B)
    pub num_heads: u32,     // 16 (0.6B) or 14 (1.7B)
    pub head_dim: u32,      // 64
    pub ffn_dim: u32,       // 4096 (0.6B) or 3584 (1.7B)
    pub output_dim: u32,    // 2048 (0.6B) or 1024 (1.7B)
}

/// Per-layer weights on GPU.
struct EncoderLayer {
    // Attention weights (bf16 packed as u32)
    wq: wgpu::Buffer,  // [d_model, d_model] bf16
    wk: wgpu::Buffer,
    wv: wgpu::Buffer,
    wo: wgpu::Buffer,
    // Attention biases (f32)
    bq: wgpu::Buffer,
    bk: wgpu::Buffer,
    bv: wgpu::Buffer,
    bo: wgpu::Buffer,
    // Attention norm (f32)
    attn_norm_w: wgpu::Buffer,
    attn_norm_b: wgpu::Buffer,
    // FFN weights (bf16 packed)
    fc1: wgpu::Buffer,  // [ffn_dim, d_model] bf16
    fc2: wgpu::Buffer,  // [d_model, ffn_dim] bf16
    // FFN biases (f32)
    fc1_bias: wgpu::Buffer,
    fc2_bias: wgpu::Buffer,
    // FFN norm (f32)
    ffn_norm_w: wgpu::Buffer,
    ffn_norm_b: wgpu::Buffer,
}

/// GPU ASR encoder.
pub struct AsrEncoder {
    gpu: GpuContext,
    layers: Vec<EncoderLayer>,
    // Final layer norm
    ln_post_w: wgpu::Buffer,
    ln_post_b: wgpu::Buffer,
    // Output projections (bf16)
    proj1_w: wgpu::Buffer,
    proj1_b: wgpu::Buffer,
    proj2_w: wgpu::Buffer,
    proj2_b: wgpu::Buffer,
    pub config: AsrEncoderConfig,
}

impl AsrEncoder {
    /// Create a new encoder with its own GPU device.
    pub fn new(model_dir: &Path) -> Self {
        let gpu = GpuContext::new();
        Self::load(gpu, model_dir)
    }

    /// Load encoder weights from safetensors into GPU buffers.
    pub fn load(gpu: GpuContext, model_dir: &Path) -> Self {
        log::info!("[asr-encoder] loading weights from {:?}", model_dir);

        let config_path = model_dir.join("config.json");
        let config = Self::parse_config(&config_path);
        log::info!("[asr-encoder] config: {:?}", config);

        // Find and mmap all safetensors shards
        let shard_data = Self::load_shards(model_dir);
        let shards: Vec<SafeTensors> = shard_data.iter()
            .map(|data| SafeTensors::deserialize(data).expect("failed to parse safetensors"))
            .collect();

        // Helper: find tensor across shards
        let get_tensor = |name: &str| -> &[u8] {
            for st in &shards {
                if let Ok(t) = st.tensor(name) {
                    return t.data();
                }
            }
            panic!("[asr-encoder] tensor not found: {name}");
        };

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.encoder.layers.{i}");
            let layer = EncoderLayer {
                wq: gpu.upload_buffer(&format!("{p}.q"), get_tensor(&format!("{p}.self_attn.q_proj.weight"))),
                wk: gpu.upload_buffer(&format!("{p}.k"), get_tensor(&format!("{p}.self_attn.k_proj.weight"))),
                wv: gpu.upload_buffer(&format!("{p}.v"), get_tensor(&format!("{p}.self_attn.v_proj.weight"))),
                wo: gpu.upload_buffer(&format!("{p}.o"), get_tensor(&format!("{p}.self_attn.out_proj.weight"))),
                bq: gpu.upload_buffer(&format!("{p}.bq"), get_tensor(&format!("{p}.self_attn.q_proj.bias"))),
                bk: gpu.upload_buffer(&format!("{p}.bk"), get_tensor(&format!("{p}.self_attn.k_proj.bias"))),
                bv: gpu.upload_buffer(&format!("{p}.bv"), get_tensor(&format!("{p}.self_attn.v_proj.bias"))),
                bo: gpu.upload_buffer(&format!("{p}.bo"), get_tensor(&format!("{p}.self_attn.out_proj.bias"))),
                attn_norm_w: gpu.upload_buffer(&format!("{p}.an_w"), get_tensor(&format!("{p}.self_attn_layer_norm.weight"))),
                attn_norm_b: gpu.upload_buffer(&format!("{p}.an_b"), get_tensor(&format!("{p}.self_attn_layer_norm.bias"))),
                fc1: gpu.upload_buffer(&format!("{p}.fc1"), get_tensor(&format!("{p}.fc1.weight"))),
                fc2: gpu.upload_buffer(&format!("{p}.fc2"), get_tensor(&format!("{p}.fc2.weight"))),
                fc1_bias: gpu.upload_buffer(&format!("{p}.fc1b"), get_tensor(&format!("{p}.fc1.bias"))),
                fc2_bias: gpu.upload_buffer(&format!("{p}.fc2b"), get_tensor(&format!("{p}.fc2.bias"))),
                ffn_norm_w: gpu.upload_buffer(&format!("{p}.fn_w"), get_tensor(&format!("{p}.final_layer_norm.weight"))),
                ffn_norm_b: gpu.upload_buffer(&format!("{p}.fn_b"), get_tensor(&format!("{p}.final_layer_norm.bias"))),
            };
            layers.push(layer);
            if (i + 1) % 6 == 0 {
                log::info!("[asr-encoder] loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        let ln_post_w = gpu.upload_buffer("enc.ln_w", get_tensor("model.encoder.layer_norm.weight"));
        let ln_post_b = gpu.upload_buffer("enc.ln_b", get_tensor("model.encoder.layer_norm.bias"));
        let proj1_w = gpu.upload_buffer("enc.p1_w", get_tensor("model.encoder.proj1.weight"));
        let proj1_b = gpu.upload_buffer("enc.p1_b", get_tensor("model.encoder.proj1.bias"));
        let proj2_w = gpu.upload_buffer("enc.p2_w", get_tensor("model.encoder.proj2.weight"));
        let proj2_b = gpu.upload_buffer("enc.p2_b", get_tensor("model.encoder.proj2.bias"));

        log::info!("[asr-encoder] loaded {} layers onto GPU", layers.len());

        Self { gpu, layers, ln_post_w, ln_post_b, proj1_w, proj1_b, proj2_w, proj2_b, config }
    }

    fn load_shards(model_dir: &Path) -> Vec<Vec<u8>> {
        let mut shard_files: Vec<PathBuf> = std::fs::read_dir(model_dir)
            .expect("can't read model dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        shard_files.sort();
        log::info!("[asr-encoder] loading {} safetensors shard(s)", shard_files.len());
        shard_files.iter()
            .map(|p| std::fs::read(p).expect(&format!("failed to read {:?}", p)))
            .collect()
    }

    /// Run the encoder transformer on GPU.
    /// Input: token embeddings from conv stem [seq_len, d_model] as f32.
    /// Output: encoder output [seq_len, output_dim] as f32.
    pub fn forward(&mut self, token_embeddings: &[f32], seq_len: u32) -> Vec<f32> {
        let d = self.config.d_model;
        let t0 = std::time::Instant::now();

        // Upload input to GPU
        let mut x = self.gpu.upload_buffer("enc_input",
            bytemuck::cast_slice(token_embeddings));

        // Allocate scratch buffers
        let buf_size = (seq_len * d) as u64 * 4;
        let mut x_norm = self.gpu.create_storage_buffer("enc_x_norm", buf_size);
        let mut residual = self.gpu.create_storage_buffer("enc_residual", buf_size);

        let qkv_size = buf_size; // same as d_model for MHA
        let mut q_buf = self.gpu.create_storage_buffer("enc_q", qkv_size);
        let mut k_buf = self.gpu.create_storage_buffer("enc_k", qkv_size);
        let mut v_buf = self.gpu.create_storage_buffer("enc_v", qkv_size);
        let mut attn_out = self.gpu.create_storage_buffer("enc_attn", qkv_size);
        let mut o_out = self.gpu.create_storage_buffer("enc_o", buf_size);

        let ffn_size = (seq_len * self.config.ffn_dim) as u64 * 4;
        let mut ffn_mid = self.gpu.create_storage_buffer("enc_ffn_mid", ffn_size);
        let mut ffn_act = self.gpu.create_storage_buffer("enc_ffn_act", ffn_size);
        let mut ffn_out = self.gpu.create_storage_buffer("enc_ffn_out", buf_size);

        let prefill_ms = t0.elapsed().as_millis();
        log::info!("[asr-encoder] buffers allocated in {}ms, seq_len={}", prefill_ms, seq_len);

        // TODO: Run transformer layers
        // For each layer:
        //   1. LayerNorm(x) → x_norm
        //   2. bf16_gemm(x_norm, wq) + bq → q
        //   3. bf16_gemm(x_norm, wk) + bk → k
        //   4. bf16_gemm(x_norm, wv) + bv → v
        //   5. bidir_attn(q, k, v) → attn_out
        //   6. bf16_gemm(attn_out, wo) + bo → o_out
        //   7. x = x + o_out (residual)
        //   8. LayerNorm(x) → x_norm
        //   9. bf16_gemm(x_norm, fc1) + fc1_bias → ffn_mid
        //  10. gelu(ffn_mid) → ffn_act
        //  11. bf16_gemm(ffn_act, fc2) + fc2_bias → ffn_out
        //  12. x = x + ffn_out (residual)

        // TODO: Output projection
        //  13. LayerNorm(x) → x_norm
        //  14. bf16_gemm(x_norm, proj1) + proj1_bias → ffn_mid (reuse buffer)
        //  15. gelu(ffn_mid) → ffn_act
        //  16. bf16_gemm(ffn_act, proj2) + proj2_bias → output

        let total_ms = t0.elapsed().as_millis();
        log::info!("[asr-encoder] forward: {}ms for {} tokens", total_ms, seq_len);

        // Readback output
        // TODO: read from output buffer
        vec![0.0; (seq_len * self.config.output_dim) as usize]
    }

    fn parse_config(path: &Path) -> AsrEncoderConfig {
        let text = std::fs::read_to_string(path).expect("config.json not found");
        let v: serde_json::Value = serde_json::from_str(&text).expect("invalid config.json");

        // Detect 0.6B vs 1.7B from encoder config
        let enc = &v["encoder"];
        let d_model = enc["d_model"].as_u64().unwrap_or(1024) as u32;
        let num_layers = enc["encoder_layers"].as_u64().unwrap_or(24) as u32;
        let num_heads = enc["encoder_attention_heads"].as_u64().unwrap_or(16) as u32;
        let ffn_dim = enc["encoder_ffn_dim"].as_u64().unwrap_or(4096) as u32;
        let output_dim = enc.get("output_dim").and_then(|v| v.as_u64()).unwrap_or(2048) as u32;

        AsrEncoderConfig {
            d_model,
            num_layers,
            num_heads,
            head_dim: 64,
            ffn_dim,
            output_dim,
        }
    }
}
