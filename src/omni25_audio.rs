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

/// Convert BF16 raw bytes to f32 bytes for shaders that read f32.
fn bf16_bytes_to_f32(data: &[u8]) -> Vec<u8> {
    let bf16: &[u16] = bytemuck::cast_slice(data);
    let f32_vals: Vec<f32> = bf16.iter()
        .map(|&bits| f32::from_bits((bits as u32) << 16))
        .collect();
    bytemuck::cast_slice(&f32_vals).to_vec()
}

mod shaders {
    pub const BF16_GEMM: &str = include_str!("shaders/bf16_gemm.wgsl");
    pub const LAYERNORM: &str = include_str!("shaders/layernorm.wgsl");
    pub const GELU_MUL: &str = include_str!("shaders/gelu_mul.wgsl");
    pub const BIDIR_ATTN: &str = include_str!("shaders/qwen_asr_bidir_attn.wgsl");
    pub const ADD: &str = include_str!("shaders/add.wgsl");
    pub const CONV1D_GELU_BF16: &str = include_str!("shaders/conv1d_gelu_bf16.wgsl");
    pub const CONV1D_BF16: &str = include_str!("shaders/conv1d_bf16.wgsl");
    pub const GELU: &str = include_str!("shaders/gelu.wgsl");
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
            &bf16_bytes_to_f32(get_tensor("thinker.audio_tower.ln_post.weight")));
        let ln_post_b = gpu.upload_buffer("omni_ln_post_b",
            &bf16_bytes_to_f32(get_tensor("thinker.audio_tower.ln_post.bias")));

        // Output projection
        let proj_w = gpu.upload_buffer("omni_proj_w",
            get_tensor("thinker.audio_tower.proj.weight"));
        let proj_b = gpu.upload_buffer("omni_proj_b",
            &bf16_bytes_to_f32(get_tensor("thinker.audio_tower.proj.bias")));

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
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.self_attn.q_proj.bias"))));
            let bv = gpu.upload_buffer(&format!("omni_l{i}_bv"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.self_attn.v_proj.bias"))));
            let bo = gpu.upload_buffer(&format!("omni_l{i}_bo"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.self_attn.out_proj.bias"))));

            let attn_norm_w = gpu.upload_buffer(&format!("omni_l{i}_an_w"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.self_attn_layer_norm.weight"))));
            let attn_norm_b = gpu.upload_buffer(&format!("omni_l{i}_an_b"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.self_attn_layer_norm.bias"))));

            let fc1 = gpu.upload_buffer(&format!("omni_l{i}_fc1"),
                get_tensor(&format!("{prefix}.fc1.weight")));
            let fc2 = gpu.upload_buffer(&format!("omni_l{i}_fc2"),
                get_tensor(&format!("{prefix}.fc2.weight")));
            let fc1_bias = gpu.upload_buffer(&format!("omni_l{i}_fc1b"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.fc1.bias"))));
            let fc2_bias = gpu.upload_buffer(&format!("omni_l{i}_fc2b"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.fc2.bias"))));

            let ffn_norm_w = gpu.upload_buffer(&format!("omni_l{i}_fn_w"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.final_layer_norm.weight"))));
            let ffn_norm_b = gpu.upload_buffer(&format!("omni_l{i}_fn_b"),
                &bf16_bytes_to_f32(get_tensor(&format!("{prefix}.final_layer_norm.bias"))));

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
    /// For multi-device: this buffer lives on the audio tower's GPU device.
    /// When colocated with the text decoder, it's zero-copy. For split devices,
    /// copy [seq_len × output_dim × 4] bytes between devices at this boundary.
    ///
    /// Returns (output_buffer, seq_len).
    pub fn encode_mel(
        &self,
        gpu: &mut GpuContext,
        mel_data: &[f32],
        mel_frames: u32,
    ) -> (wgpu::Buffer, u32) {
        use crate::gpu::bind;
        let t0 = std::time::Instant::now();
        let d = self.config.d_model;
        let mel_bins = self.config.num_mel_bins;
        let out_dim = self.config.output_dim;

        // Upload mel to GPU
        let mel_buf = gpu.upload_buffer("omni_mel", bytemuck::cast_slice(mel_data));

        // Conv params uniform
        let params = gpu.create_buffer("omni_conv_params", 32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        // Conv1: [mel_bins, mel_frames] → [d_model, mel_frames] (stride=1, pad=1)
        let conv1_out_len = mel_frames; // stride=1
        let c1_size = (d * conv1_out_len) as u64 * 4;
        let c1_buf = gpu.create_storage_buffer("omni_conv1_out", c1_size);
        gpu.write_buffer(&params, 0, bytemuck::cast_slice(&[
            mel_bins, d, mel_frames, 3u32, 1u32, 1u32, conv1_out_len, 0u32,
        ]));
        gpu.dispatch("omni_conv1", shaders::CONV1D_GELU_BF16, &[
            bind(0, &mel_buf), bind(1, &self.conv1_w), bind(2, &self.conv1_b),
            bind(3, &c1_buf), bind(4, &params),
        ], ((d * conv1_out_len).div_ceil(256), 1, 1));

        let conv1_ms = t0.elapsed().as_millis();

        // Conv2: [d_model, mel_frames] → [d_model, out_len] (stride=2, pad=1)
        let conv2_out_len = (mel_frames + 2 * 1 - 3) / 2 + 1; // stride=2, kernel=3, pad=1
        let c2_size = (d * conv2_out_len) as u64 * 4;
        let c2_buf = gpu.create_storage_buffer("omni_conv2_out", c2_size);
        gpu.flush();
        gpu.write_buffer(&params, 0, bytemuck::cast_slice(&[
            d, d, conv1_out_len, 3u32, 2u32, 1u32, conv2_out_len, 0u32,
        ]));
        gpu.dispatch("omni_conv2", shaders::CONV1D_BF16, &[
            bind(0, &c1_buf), bind(1, &self.conv2_w), bind(2, &self.conv2_b),
            bind(3, &c2_buf), bind(4, &params),
        ], ((d * conv2_out_len).div_ceil(256), 1, 1));
        gpu.flush_and_wait();

        let conv_ms = t0.elapsed().as_millis();
        let seq_len = conv2_out_len;
        log::info!("[omni25-audio] conv stem: {} frames → {} tokens (conv1={conv1_ms}ms total={conv_ms}ms)",
            mel_frames, seq_len);

        // ── Transformer layers (32 layers of bidirectional self-attention) ──
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let ffn_dim = self.config.ffn_dim;

        let buf_size = (seq_len * d) as u64 * 4;
        let x_norm = gpu.create_storage_buffer("omni_enc_xn", buf_size);
        let q_buf = gpu.create_storage_buffer("omni_enc_q", buf_size);
        let k_buf = gpu.create_storage_buffer("omni_enc_k", buf_size);
        let v_buf = gpu.create_storage_buffer("omni_enc_v", buf_size);
        let attn_out = gpu.create_storage_buffer("omni_enc_attn", buf_size);
        let o_out = gpu.create_storage_buffer("omni_enc_o", buf_size);
        let ffn_size = (seq_len * ffn_dim) as u64 * 4;
        let ffn_mid = gpu.create_storage_buffer("omni_enc_ffn1", ffn_size);
        let ffn_act = gpu.create_storage_buffer("omni_enc_ffn2", ffn_size);
        let ffn_out = gpu.create_storage_buffer("omni_enc_ffno", buf_size);
        let enc_params = gpu.create_buffer("omni_enc_p", 64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        // Dummy bias for k_proj (which has no bias in Omni)
        let dummy_bias = gpu.create_storage_buffer("omni_enc_dummy", d as u64 * 2);

        // Conv2 output needs to be reshaped from [d_model, seq_len] to [seq_len, d_model]
        // TODO: add transpose shader or handle layout in GEMM
        // For now: read back and re-upload transposed
        gpu.flush_and_wait();
        let c2_bytes = gpu.read_buffer(&c2_buf, (d * seq_len) as u64 * 4);
        let c2_f32: &[f32] = bytemuck::cast_slice(&c2_bytes);
        let mut transposed = vec![0f32; (seq_len * d) as usize];
        for ch in 0..d as usize {
            for t in 0..seq_len as usize {
                transposed[t * d as usize + ch] = c2_f32[ch * seq_len as usize + t];
            }
        }
        // Debug: conv2 output comparison with HF reference
        {
            let norm: f32 = transposed.iter().map(|x| x*x).sum::<f32>().sqrt();
            let t0 = &transposed[0..8.min(d as usize)];
            let t0_norm: f32 = transposed[..d as usize].iter().map(|x| x*x).sum::<f32>().sqrt();
            log::info!("[omni25-audio] conv2 transposed: norm={:.2} token[0] norm={:.4} first8={:.6?}",
                norm, t0_norm, t0);
        }

        // Sinusoidal positional encoding (matches HF SinusoidsPositionEmbedding)
        {
            let max_timescale: f32 = 10000.0;
            let channels = d as usize;
            let half = channels / 2;
            let log_inc = (max_timescale.ln()) / (half as f32 - 1.0);
            for t in 0..seq_len as usize {
                for c in 0..half {
                    let inv_ts = (-log_inc * c as f32).exp();
                    let angle = t as f32 * inv_ts;
                    transposed[t * d as usize + c] += angle.sin();
                    transposed[t * d as usize + half + c] += angle.cos();
                }
            }
        }
        let x_cur = gpu.upload_buffer("omni_enc_x", bytemuck::cast_slice(&transposed));

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];

            // 1. LayerNorm
            dispatch_layernorm(gpu, &x_cur, &layer.attn_norm_w, &layer.attn_norm_b,
                &x_norm, &enc_params, seq_len, d);

            // 2-4. Q/K/V projections (k has no bias)
            dispatch_bf16_gemm(gpu, &x_norm, &layer.wq, &layer.bq, &q_buf,
                &enc_params, seq_len, d, d, true);
            dispatch_bf16_gemm(gpu, &x_norm, &layer.wk, &dummy_bias, &k_buf,
                &enc_params, seq_len, d, d, false);
            dispatch_bf16_gemm(gpu, &x_norm, &layer.wv, &layer.bv, &v_buf,
                &enc_params, seq_len, d, d, true);

            // 5. Bidirectional attention (full sequence, no window)
            dispatch_bidir_attn(gpu, &q_buf, &k_buf, &v_buf, &attn_out,
                &enc_params, seq_len, num_heads, head_dim, 0, seq_len);

            // 6. Output projection
            dispatch_bf16_gemm(gpu, &attn_out, &layer.wo, &layer.bo, &o_out,
                &enc_params, seq_len, d, d, true);

            // 7. Residual add
            dispatch_add(gpu, &x_cur, &o_out, &enc_params, seq_len * d);

            // 8. FFN LayerNorm
            dispatch_layernorm(gpu, &x_cur, &layer.ffn_norm_w, &layer.ffn_norm_b,
                &x_norm, &enc_params, seq_len, d);

            // 9-11. FFN: fc1 + GELU + fc2
            dispatch_bf16_gemm(gpu, &x_norm, &layer.fc1, &layer.fc1_bias, &ffn_mid,
                &enc_params, seq_len, d, ffn_dim, true);
            dispatch_gelu(gpu, &ffn_mid, &ffn_act, &enc_params, seq_len * ffn_dim);
            dispatch_bf16_gemm(gpu, &ffn_act, &layer.fc2, &layer.fc2_bias, &ffn_out,
                &enc_params, seq_len, ffn_dim, d, true);

            // 12. Residual add
            dispatch_add(gpu, &x_cur, &ffn_out, &enc_params, seq_len * d);

            if (layer_idx + 1) % 8 == 0 {
                gpu.flush_and_wait();
                log::info!("[omni25-audio] transformer layer {}/{}", layer_idx + 1, self.layers.len());
            }
        }

        // ── AvgPool1d(2, stride=2): reduce seq_len by 2x ──
        // Pool along the time dimension: output[t, d] = (input[2t, d] + input[2t+1, d]) / 2
        let pooled_len = seq_len / 2;
        {
            gpu.flush_and_wait();
            let x_bytes = gpu.read_buffer(&x_cur, (seq_len * d) as u64 * 4);
            let x_f32: &[f32] = bytemuck::cast_slice(&x_bytes);
            let mut pooled = vec![0f32; (pooled_len * d) as usize];
            for t in 0..pooled_len as usize {
                for c in 0..d as usize {
                    let a = x_f32[t * 2 * d as usize + c];
                    let b = x_f32[(t * 2 + 1) * d as usize + c];
                    pooled[t * d as usize + c] = (a + b) * 0.5;
                }
            }
            gpu.write_buffer(&x_cur, 0, bytemuck::cast_slice(&pooled));
        }
        let seq_len = pooled_len;
        log::info!("[omni25-audio] avg_pool: {} → {} tokens", pooled_len * 2, seq_len);

        // ── Final LayerNorm + output projection ──
        dispatch_layernorm(gpu, &x_cur, &self.ln_post_w, &self.ln_post_b,
            &x_norm, &enc_params, seq_len, d);

        // proj: [d_model → output_dim] linear
        let output = gpu.create_storage_buffer("omni_audio_output",
            seq_len as u64 * out_dim as u64 * 4);
        dispatch_bf16_gemm(gpu, &x_norm, &self.proj_w, &self.proj_b, &output,
            &enc_params, seq_len, d, out_dim, true);

        gpu.flush_and_wait();
        let total_ms = t0.elapsed().as_millis();
        log::info!("[omni25-audio] encode_mel: {} frames → {} tokens ({total_ms}ms, conv={conv_ms}ms)",
            mel_frames, seq_len);

        (output, seq_len)
    }
}

// ── GPU dispatch helpers (shared with asr_encoder) ──────────────────────

fn dispatch_layernorm(
    gpu: &mut GpuContext, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer,
    output: &wgpu::Buffer, params: &wgpu::Buffer, seq_len: u32, dim: u32,
) {
    use crate::gpu::bind;
    #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { n: u32, eps: f32, seq_len: u32, _pad: u32 }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P { n: dim, eps: 1e-5, seq_len, _pad: 0 }));
    gpu.dispatch("layernorm", shaders::LAYERNORM, &[
        bind(0, input), bind(1, weight), bind(2, bias), bind(3, output), bind(4, params),
    ], (seq_len, 1, 1));
}

fn dispatch_bf16_gemm(
    gpu: &mut GpuContext, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer,
    output: &wgpu::Buffer, params: &wgpu::Buffer,
    seq_len: u32, d_in: u32, d_out: u32, has_bias: bool,
) {
    use crate::gpu::bind;
    #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { d_in: u32, d_out: u32, seq_len: u32, has_bias: u32 }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P { d_in, d_out, seq_len, has_bias: has_bias as u32 }));
    gpu.dispatch("bf16_gemm", shaders::BF16_GEMM, &[
        bind(0, input), bind(1, weight), bind(2, bias), bind(3, output), bind(4, params),
    ], (d_out.div_ceil(32), seq_len, 1));
}

fn dispatch_bidir_attn(
    gpu: &mut GpuContext, q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer,
    output: &wgpu::Buffer, params: &wgpu::Buffer,
    seq_len: u32, num_heads: u32, head_dim: u32, window_start: u32, window_end: u32,
) {
    use crate::gpu::bind;
    #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { seq_len: u32, head_dim: u32, num_heads: u32, win_s: u32, win_e: u32, _p: [u32; 3] }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P {
        seq_len, head_dim, num_heads, win_s: window_start, win_e: window_end, _p: [0; 3],
    }));
    gpu.dispatch("bidir_attn", shaders::BIDIR_ATTN, &[
        bind(0, q), bind(1, k), bind(2, v), bind(3, output), bind(4, params),
    ], (num_heads, seq_len, 1));
}

fn dispatch_gelu(
    gpu: &mut GpuContext, input: &wgpu::Buffer, output: &wgpu::Buffer,
    params: &wgpu::Buffer, n: u32,
) {
    use crate::gpu::bind;
    #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { n: u32, _p: [u32; 3] }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P { n, _p: [0; 3] }));
    gpu.dispatch("gelu", shaders::GELU, &[
        bind(0, input), bind(1, output), bind(2, params),
    ], (n.div_ceil(256), 1, 1));
}

fn dispatch_add(
    gpu: &mut GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer,
    params: &wgpu::Buffer, n: u32,
) {
    use crate::gpu::bind;
    #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { n: u32, _p: [u32; 3] }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P { n, _p: [0; 3] }));
    gpu.dispatch("add", shaders::ADD, &[
        bind(0, a), bind(1, b), bind(2, params),
    ], (n.div_ceil(256), 1, 1));
}
