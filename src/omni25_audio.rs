//! GPU-accelerated Qwen2.5-Omni audio tower.
//!
//! Whisper-style encoder: conv1 + conv2 + 32 transformer layers + ln_post + proj.
//! Takes mel spectrogram (128 bins × n_frames), outputs [seq_len, output_dim=3584]
//! embeddings that are injected into the thinker's token sequence.
//!
//! Weight tensors: `thinker.audio_tower.*` from Qwen2.5-Omni safetensors.

use std::path::Path;
use safetensors::SafeTensors;
use crate::gpu::GpuContext;

mod shaders {
    pub const BF16_GEMM: &str = include_str!("shaders/bf16_gemm.wgsl");
    pub const LAYERNORM: &str = include_str!("shaders/layernorm.wgsl");
    pub const GELU_MUL: &str = include_str!("shaders/gelu_mul.wgsl");
    pub const BIDIR_ATTN: &str = include_str!("shaders/qwen_asr_bidir_attn.wgsl");
    pub const ADD: &str = include_str!("shaders/add.wgsl");
}

/// Audio tower configuration (from config.json audio_config).
#[derive(Debug, Clone)]
pub struct Omni25AudioConfig {
    pub d_model: u32,         // 1280
    pub num_layers: u32,      // 32
    pub num_heads: u32,       // 20
    pub head_dim: u32,        // 64 (d_model / num_heads)
    pub ffn_dim: u32,         // 5120 (encoder_ffn_dim)
    pub output_dim: u32,      // 3584 (projects to thinker hidden size)
    pub num_mel_bins: u32,    // 128
}

impl Default for Omni25AudioConfig {
    fn default() -> Self {
        Self {
            d_model: 1280,
            num_layers: 32,
            num_heads: 20,
            head_dim: 64,
            ffn_dim: 5120,
            output_dim: 3584,
            num_mel_bins: 128,
        }
    }
}

/// Per-layer transformer weights on GPU.
struct AudioLayer {
    // Self-attention weights (bf16 packed as u32)
    wq: wgpu::Buffer,  // [d_model, d_model] bf16
    wk: wgpu::Buffer,
    wv: wgpu::Buffer,
    wo: wgpu::Buffer,
    // Self-attention biases (f32) — k_proj has no bias in Omni
    bq: wgpu::Buffer,
    bv: wgpu::Buffer,
    bo: wgpu::Buffer,
    // Pre-attention layer norm
    attn_norm_w: wgpu::Buffer,
    attn_norm_b: wgpu::Buffer,
    // FFN weights (bf16 packed)
    fc1: wgpu::Buffer,  // [ffn_dim, d_model] bf16
    fc2: wgpu::Buffer,  // [d_model, ffn_dim] bf16
    // FFN biases (f32)
    fc1_bias: wgpu::Buffer,
    fc2_bias: wgpu::Buffer,
    // Post-FFN layer norm
    ffn_norm_w: wgpu::Buffer,
    ffn_norm_b: wgpu::Buffer,
}

/// GPU Qwen2.5-Omni audio tower encoder.
pub struct Omni25AudioEncoder {
    layers: Vec<AudioLayer>,
    // Conv stem
    conv1_w: wgpu::Buffer, conv1_b: wgpu::Buffer, // [d_model, mel_bins, 3]
    conv2_w: wgpu::Buffer, conv2_b: wgpu::Buffer, // [d_model, d_model, 3]
    // Final layer norm
    ln_post_w: wgpu::Buffer,
    ln_post_b: wgpu::Buffer,
    // Output projection to thinker hidden size
    proj_w: wgpu::Buffer,  // [output_dim, d_model] bf16
    proj_b: wgpu::Buffer,  // [output_dim] f32
    // Audio BOS/EOS token embeddings
    audio_bos_eos: wgpu::Buffer, // [2, output_dim] — row 0 = BOS, row 1 = EOS
    pub config: Omni25AudioConfig,
}

impl Omni25AudioEncoder {
    /// Parse audio config from the model's config.json.
    pub fn parse_config(model_dir: &Path) -> Omni25AudioConfig {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .unwrap_or_else(|_| panic!("cannot read {:?}", config_path));
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .expect("invalid config.json");

        // Navigate: config.thinker_config.audio_config
        let ac = config
            .pointer("/thinker_config/audio_config")
            .or_else(|| config.pointer("/audio_config"))
            .expect("no audio_config in config.json");

        let d_model = ac["d_model"].as_u64().unwrap_or(1280) as u32;
        let num_heads = ac["encoder_attention_heads"].as_u64().unwrap_or(20) as u32;
        let num_layers = ac["encoder_layers"].as_u64()
            .or_else(|| ac["num_hidden_layers"].as_u64())
            .unwrap_or(32) as u32;
        let ffn_dim = ac["encoder_ffn_dim"].as_u64().unwrap_or(5120) as u32;
        let output_dim = ac["output_dim"].as_u64().unwrap_or(3584) as u32;
        let num_mel_bins = ac["num_mel_bins"].as_u64().unwrap_or(128) as u32;

        Omni25AudioConfig {
            d_model,
            num_layers,
            num_heads,
            head_dim: d_model / num_heads,
            ffn_dim,
            output_dim,
            num_mel_bins,
        }
    }

    /// Load audio tower weights from safetensors into GPU buffers.
    pub fn load(gpu: &mut GpuContext, model_dir: &Path) -> Self {
        log::info!("[omni25-audio] loading weights from {:?}", model_dir);

        let config = Self::parse_config(model_dir);
        log::info!("[omni25-audio] config: {:?}", config);

        // Find and mmap all safetensors shards
        let mut shard_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
            .expect("can't read model dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();
        shard_files.sort();
        log::info!("[omni25-audio] loading {} safetensors shard(s)", shard_files.len());

        let shard_data: Vec<Vec<u8>> = shard_files.iter()
            .map(|p| std::fs::read(p).expect("failed to read safetensors shard"))
            .collect();
        let shards: Vec<SafeTensors> = shard_data.iter()
            .map(|data| SafeTensors::deserialize(data).expect("failed to parse safetensors"))
            .collect();

        // Tensor lookup helper
        let get_tensor = |name: &str| -> &[u8] {
            for st in &shards {
                if let Ok(t) = st.tensor(name) {
                    return t.data();
                }
            }
            panic!("[omni25-audio] tensor not found: {name}");
        };

        let _d = config.d_model as usize;

        // Conv stem
        let conv1_w = gpu.upload_buffer("omni_conv1_w",
            get_tensor("thinker.audio_tower.conv1.weight"));
        let conv1_b = gpu.upload_buffer("omni_conv1_b",
            get_tensor("thinker.audio_tower.conv1.bias"));
        let conv2_w = gpu.upload_buffer("omni_conv2_w",
            get_tensor("thinker.audio_tower.conv2.weight"));
        let conv2_b = gpu.upload_buffer("omni_conv2_b",
            get_tensor("thinker.audio_tower.conv2.bias"));

        // Final layer norm
        let ln_post_w = gpu.upload_buffer("omni_ln_post_w",
            get_tensor("thinker.audio_tower.ln_post.weight"));
        let ln_post_b = gpu.upload_buffer("omni_ln_post_b",
            get_tensor("thinker.audio_tower.ln_post.bias"));

        // Output projection
        let proj_w = gpu.upload_buffer("omni_proj_w",
            get_tensor("thinker.audio_tower.proj.weight"));
        let proj_b = gpu.upload_buffer("omni_proj_b",
            get_tensor("thinker.audio_tower.proj.bias"));

        // Audio BOS/EOS embeddings
        let audio_bos_eos = gpu.upload_buffer("omni_audio_bos_eos",
            get_tensor("thinker.audio_tower.audio_bos_eos_token.weight"));

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let prefix = format!("thinker.audio_tower.layers.{i}");

            let wq = gpu.upload_buffer(&format!("omni_l{i}_wq"),
                get_tensor(&format!("{prefix}.self_attn.q_proj.weight")));
            let wk = gpu.upload_buffer(&format!("omni_l{i}_wk"),
                get_tensor(&format!("{prefix}.self_attn.k_proj.weight")));
            let wv = gpu.upload_buffer(&format!("omni_l{i}_wv"),
                get_tensor(&format!("{prefix}.self_attn.v_proj.weight")));
            let wo = gpu.upload_buffer(&format!("omni_l{i}_wo"),
                get_tensor(&format!("{prefix}.self_attn.out_proj.weight")));

            let bq = gpu.upload_buffer(&format!("omni_l{i}_bq"),
                get_tensor(&format!("{prefix}.self_attn.q_proj.bias")));
            let bv = gpu.upload_buffer(&format!("omni_l{i}_bv"),
                get_tensor(&format!("{prefix}.self_attn.v_proj.bias")));
            let bo = gpu.upload_buffer(&format!("omni_l{i}_bo"),
                get_tensor(&format!("{prefix}.self_attn.out_proj.bias")));

            let attn_norm_w = gpu.upload_buffer(&format!("omni_l{i}_an_w"),
                get_tensor(&format!("{prefix}.self_attn_layer_norm.weight")));
            let attn_norm_b = gpu.upload_buffer(&format!("omni_l{i}_an_b"),
                get_tensor(&format!("{prefix}.self_attn_layer_norm.bias")));

            let fc1 = gpu.upload_buffer(&format!("omni_l{i}_fc1"),
                get_tensor(&format!("{prefix}.fc1.weight")));
            let fc2 = gpu.upload_buffer(&format!("omni_l{i}_fc2"),
                get_tensor(&format!("{prefix}.fc2.weight")));
            let fc1_bias = gpu.upload_buffer(&format!("omni_l{i}_fc1b"),
                get_tensor(&format!("{prefix}.fc1.bias")));
            let fc2_bias = gpu.upload_buffer(&format!("omni_l{i}_fc2b"),
                get_tensor(&format!("{prefix}.fc2.bias")));

            let ffn_norm_w = gpu.upload_buffer(&format!("omni_l{i}_fn_w"),
                get_tensor(&format!("{prefix}.final_layer_norm.weight")));
            let ffn_norm_b = gpu.upload_buffer(&format!("omni_l{i}_fn_b"),
                get_tensor(&format!("{prefix}.final_layer_norm.bias")));

            layers.push(AudioLayer {
                wq, wk, wv, wo, bq, bv, bo,
                attn_norm_w, attn_norm_b,
                fc1, fc2, fc1_bias, fc2_bias,
                ffn_norm_w, ffn_norm_b,
            });

            if (i + 1) % 8 == 0 {
                log::info!("[omni25-audio] loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        log::info!("[omni25-audio] loaded {} layers onto GPU", layers.len());

        Self {
            layers,
            conv1_w, conv1_b, conv2_w, conv2_b,
            ln_post_w, ln_post_b,
            proj_w, proj_b,
            audio_bos_eos,
            config,
        }
    }

    /// Encode mel spectrogram to audio embeddings.
    ///
    /// Input: mel_data [num_mel_bins × n_frames] (mel-bin-major f32)
    /// Output: GPU buffer [seq_len, output_dim] f32 embeddings ready for thinker injection.
    ///
    /// Returns (output_buffer, seq_len).
    pub fn encode_mel(
        &self,
        gpu: &mut GpuContext,
        mel_data: &[f32],
        mel_frames: u32,
    ) -> (wgpu::Buffer, u32) {
        let t0 = std::time::Instant::now();
        let d = self.config.d_model;
        let out_dim = self.config.output_dim;

        // TODO: implement conv stem + transformer + ln_post + proj on GPU
        // For now, placeholder that returns zeros
        let seq_len = mel_frames / 2; // conv stride=2 halves the sequence
        let output = gpu.create_storage_buffer(
            "omni_audio_output",
            seq_len as u64 * out_dim as u64 * 4,
        );

        log::info!("[omni25-audio] encode_mel: {} frames → {} output tokens ({:.1}ms)",
            mel_frames, seq_len, t0.elapsed().as_secs_f64() * 1000.0);

        (output, seq_len)
    }
}
