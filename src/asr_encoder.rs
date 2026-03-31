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
const CONV_HIDDEN: u32 = 480;

/// Conv stem weights on GPU (f32, not quantized).
struct ConvStemGpu {
    conv1_w: wgpu::Buffer, conv1_b: wgpu::Buffer, // [480, 1, 3, 3] + [480]
    conv2_w: wgpu::Buffer, conv2_b: wgpu::Buffer, // [480, 480, 3, 3] + [480]
    conv3_w: wgpu::Buffer, conv3_b: wgpu::Buffer, // [480, 480, 3, 3] + [480]
    proj_w: wgpu::Buffer,                           // [d_model, 480*16] f32
}

pub struct AsrEncoder {
    pub gpu: GpuContext,
    layers: Vec<EncoderLayer>,
    conv: ConvStemGpu,
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

        // Helper: find tensor across shards. Tries both HF and MLX prefixes.
        let get_tensor = |name: &str| -> &[u8] {
            for st in &shards {
                if let Ok(t) = st.tensor(name) {
                    return t.data();
                }
            }
            // Try without "thinker." prefix (MLX models)
            if name.starts_with("thinker.") {
                let alt = &name["thinker.".len()..];
                for st in &shards {
                    if let Ok(t) = st.tensor(alt) {
                        return t.data();
                    }
                }
            }
            panic!("[asr-encoder] tensor not found: {name}");
        };

        // Convert bf16 bytes → f32 bytes for biases and norm weights.
        // GEMM weights stay as bf16 packed (shader unpacks them).
        // Biases/norms are read as f32 by the shaders.
        let bf16_to_f32 = |data: &[u8]| -> Vec<u8> {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n * 4);
            for i in 0..n {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                let f = f32::from_bits((bits as u32) << 16);
                out.extend_from_slice(&f.to_le_bytes());
            }
            out
        };

        // Upload raw bf16 packed as u32 (shader unpacks per-multiply)
        let upload_bf16 = |gpu: &GpuContext, label: &str, name: &str| -> wgpu::Buffer {
            gpu.upload_buffer(label, get_tensor(name))
        };
        // Upload bf16→f32 converted (for biases/norms that shaders read as f32)
        let upload_f32 = |gpu: &GpuContext, label: &str, name: &str| -> wgpu::Buffer {
            let f32_data = bf16_to_f32(get_tensor(name));
            gpu.upload_buffer(label, &f32_data)
        };

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("thinker.audio_tower.layers.{i}");
            let layer = EncoderLayer {
                // Weights: raw bf16 packed as u32 (shader unpacks per-multiply)
                wq: upload_bf16(&gpu, &format!("{p}.q"), &format!("{p}.self_attn.q_proj.weight")),
                wk: upload_bf16(&gpu, &format!("{p}.k"), &format!("{p}.self_attn.k_proj.weight")),
                wv: upload_bf16(&gpu, &format!("{p}.v"), &format!("{p}.self_attn.v_proj.weight")),
                wo: upload_bf16(&gpu, &format!("{p}.o"), &format!("{p}.self_attn.out_proj.weight")),
                fc1: upload_bf16(&gpu, &format!("{p}.fc1"), &format!("{p}.fc1.weight")),
                fc2: upload_bf16(&gpu, &format!("{p}.fc2"), &format!("{p}.fc2.weight")),
                // Biases + norms: bf16→f32 (shaders read as f32)
                bq: upload_f32(&gpu, &format!("{p}.bq"), &format!("{p}.self_attn.q_proj.bias")),
                bk: upload_f32(&gpu, &format!("{p}.bk"), &format!("{p}.self_attn.k_proj.bias")),
                bv: upload_f32(&gpu, &format!("{p}.bv"), &format!("{p}.self_attn.v_proj.bias")),
                bo: upload_f32(&gpu, &format!("{p}.bo"), &format!("{p}.self_attn.out_proj.bias")),
                attn_norm_w: upload_f32(&gpu, &format!("{p}.an_w"), &format!("{p}.self_attn_layer_norm.weight")),
                attn_norm_b: upload_f32(&gpu, &format!("{p}.an_b"), &format!("{p}.self_attn_layer_norm.bias")),
                fc1_bias: upload_f32(&gpu, &format!("{p}.fc1b"), &format!("{p}.fc1.bias")),
                fc2_bias: upload_f32(&gpu, &format!("{p}.fc2b"), &format!("{p}.fc2.bias")),
                ffn_norm_w: upload_f32(&gpu, &format!("{p}.fn_w"), &format!("{p}.final_layer_norm.weight")),
                ffn_norm_b: upload_f32(&gpu, &format!("{p}.fn_b"), &format!("{p}.final_layer_norm.bias")),
            };
            layers.push(layer);
            if (i + 1) % 6 == 0 {
                log::info!("[asr-encoder] loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        let ln_post_w = upload_f32(&gpu, "enc.ln_w", "thinker.audio_tower.ln_post.weight");
        let ln_post_b = upload_f32(&gpu, "enc.ln_b", "thinker.audio_tower.ln_post.bias");
        let proj1_w = upload_bf16(&gpu, "enc.p1_w", "thinker.audio_tower.proj1.weight");
        let proj1_b = upload_f32(&gpu, "enc.p1_b", "thinker.audio_tower.proj1.bias");
        let proj2_w = upload_bf16(&gpu, "enc.p2_w", "thinker.audio_tower.proj2.weight");
        let proj2_b = upload_f32(&gpu, "enc.p2_b", "thinker.audio_tower.proj2.bias");

        log::info!("[asr-encoder] loaded {} layers onto GPU", layers.len());

        // Load conv stem weights → GPU (f32, not quantized)
        let upload_conv_weight = |gpu: &GpuContext, label: &str, name: &str, c_in: usize| -> wgpu::Buffer {
            let data = get_tensor(name);
            let f32_bytes = bf16_to_f32(data);
            let w: &[f32] = bytemuck::cast_slice(&f32_bytes);
            let c_out = w.len() / (c_in * 3 * 3);
            // Detect and transpose MLX layout [out, kH, kW, in] → PyTorch [out, in, kH, kW]
            let w_pt = transpose_conv2d_if_mlx(w, c_out, c_in, 3, 3);
            gpu.upload_buffer(label, bytemuck::cast_slice(&w_pt))
        };

        let conv = ConvStemGpu {
            conv1_w: upload_conv_weight(&gpu, "conv1_w", "thinker.audio_tower.conv2d1.weight", 1),
            conv1_b: upload_f32(&gpu, "conv1_b", "thinker.audio_tower.conv2d1.bias"),
            conv2_w: upload_conv_weight(&gpu, "conv2_w", "thinker.audio_tower.conv2d2.weight", CONV_HIDDEN as usize),
            conv2_b: upload_f32(&gpu, "conv2_b", "thinker.audio_tower.conv2d2.bias"),
            conv3_w: upload_conv_weight(&gpu, "conv3_w", "thinker.audio_tower.conv2d3.weight", CONV_HIDDEN as usize),
            conv3_b: upload_f32(&gpu, "conv3_b", "thinker.audio_tower.conv2d3.bias"),
            proj_w: { let d = bf16_to_f32(get_tensor("thinker.audio_tower.conv_out.weight"));
                       gpu.upload_buffer("conv_proj_w", &d) },
        };

        Self { gpu, layers, conv, ln_post_w, ln_post_b, proj1_w, proj1_b, proj2_w, proj2_b, config }
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

    /// Full pipeline: mel → conv stem (GPU) → transformer (GPU).
    /// Input: mel data `[128, mel_frames]` as f32.
    /// Returns `(output, n_tokens, conv_ms, enc_ms)`.
    pub fn forward_mel(&mut self, mel: &[f32], mel_frames: u32) -> (Vec<f32>, u32, u128, u128) {
        // Invalidate bind group cache — temporary buffers from previous call may
        // have been deallocated and their addresses reused by the allocator.
        self.gpu.invalidate_bind_groups();
        let t0 = std::time::Instant::now();
        let (conv_out, n_tokens) = self.conv_stem_gpu(mel, mel_frames);
        let conv_ms = t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();
        let output = self.forward(&conv_out, n_tokens);
        let enc_ms = t1.elapsed().as_millis();
        (output, n_tokens, conv_ms, enc_ms)
    }

    /// GPU conv stem: mel [128, frames] → token embeddings [n_tokens, d_model].
    /// 3× Conv2D(3×3, stride=2, pad=1) + GELU → reshape → linear proj → sinusoidal PE.
    fn conv_stem_gpu(&mut self, mel: &[f32], mel_frames: u32) -> (Vec<f32>, u32) {
        use crate::gpu::bind;
        let d_model = self.config.d_model;
        let mf = mel_frames as usize;

        // Spatial dims after each conv layer (stride=2, pad=1, kernel=3)
        let h0 = 128u32; let w0 = mel_frames;
        let h1 = (h0 + 2 - 3) / 2 + 1; // 64
        let w1 = (w0 + 2 - 3) / 2 + 1;
        let h2 = (h1 + 2 - 3) / 2 + 1; // 32
        let w2 = (w1 + 2 - 3) / 2 + 1;
        let h3 = (h2 + 2 - 3) / 2 + 1; // 16
        let w3 = (w2 + 2 - 3) / 2 + 1;
        let n_tokens = w3;
        let proj_dim = CONV_HIDDEN * h3; // 480 * 16 = 7680

        // Upload mel to GPU
        let mel_buf = self.gpu.upload_buffer("conv_mel", bytemuck::cast_slice(mel));

        // Params uniform for conv dispatches
        let params = self.gpu.create_buffer("conv_params", 32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        // Conv1: [1, 128, w0] → [480, 64, w1]
        let c1_size = (CONV_HIDDEN * h1 * w1) as u64 * 4;
        let c1_buf = self.gpu.create_storage_buffer("conv1_out", c1_size);
        #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct ConvParams { c_in: u32, h_in: u32, w_in: u32, h_out: u32, w_out: u32, _pad: [u32; 3] }
        self.gpu.flush();
        self.gpu.write_buffer(&params, 0, bytemuck::bytes_of(&ConvParams {
            c_in: 1, h_in: h0, w_in: w0, h_out: h1, w_out: w1, _pad: [0; 3]
        }));
        let shader1 = build_conv2d_gelu_shader(1, CONV_HIDDEN);
        self.gpu.dispatch("conv_stem_1", &shader1, &[
            bind(0, &mel_buf), bind(1, &self.conv.conv1_w), bind(2, &self.conv.conv1_b),
            bind(3, &c1_buf), bind(4, &params),
        ], ((CONV_HIDDEN * h1 * w1).div_ceil(256), 1, 1));

        // Conv2: [480, h1, w1] → [480, h2, w2]
        let c2_size = (CONV_HIDDEN * h2 * w2) as u64 * 4;
        let c2_buf = self.gpu.create_storage_buffer("conv2_out", c2_size);
        self.gpu.flush();
        self.gpu.write_buffer(&params, 0, bytemuck::bytes_of(&ConvParams {
            c_in: CONV_HIDDEN, h_in: h1, w_in: w1, h_out: h2, w_out: w2, _pad: [0; 3]
        }));
        let shader2 = build_conv2d_gelu_shader(CONV_HIDDEN, CONV_HIDDEN);
        self.gpu.dispatch("conv_stem_2", &shader2, &[
            bind(0, &c1_buf), bind(1, &self.conv.conv2_w), bind(2, &self.conv.conv2_b),
            bind(3, &c2_buf), bind(4, &params),
        ], ((CONV_HIDDEN * h2 * w2).div_ceil(256), 1, 1));

        // Conv3: [480, h2, w2] → [480, h3, w3]
        let c3_size = (CONV_HIDDEN * h3 * w3) as u64 * 4;
        let c3_buf = self.gpu.create_storage_buffer("conv3_out", c3_size);
        self.gpu.flush();
        self.gpu.write_buffer(&params, 0, bytemuck::bytes_of(&ConvParams {
            c_in: CONV_HIDDEN, h_in: h2, w_in: w2, h_out: h3, w_out: w3, _pad: [0; 3]
        }));
        self.gpu.dispatch("conv_stem_3", &shader2, &[
            bind(0, &c2_buf), bind(1, &self.conv.conv3_w), bind(2, &self.conv.conv3_b),
            bind(3, &c3_buf), bind(4, &params),
        ], ((CONV_HIDDEN * h3 * w3).div_ceil(256), 1, 1));

        // Reshape [480, h3, w3] → [w3, 480*h3] + Linear proj → [w3, d_model] + sinusoidal PE
        // Fused into one shader: reads conv3 output, reshapes, does matvec with proj_w, adds PE
        let out_size = (n_tokens * d_model) as u64 * 4;
        let out_buf = self.gpu.create_storage_buffer("conv_stem_out", out_size);
        self.gpu.flush();
        #[repr(C)] #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct ProjParams { h3: u32, w3: u32, d_model: u32, _pad: u32 }
        let proj_params = self.gpu.create_buffer("proj_params", 16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        self.gpu.write_buffer(&proj_params, 0, bytemuck::bytes_of(&ProjParams {
            h3, w3, d_model, _pad: 0
        }));
        let proj_shader = build_reshape_proj_pe_shader();
        self.gpu.dispatch("conv_proj_pe", &proj_shader, &[
            bind(0, &c3_buf), bind(1, &self.conv.proj_w),
            bind(2, &out_buf), bind(3, &proj_params),
        ], (n_tokens, d_model.div_ceil(32), 1));

        // Read back
        self.gpu.flush();
        let bytes = self.gpu.read_buffer(&out_buf, out_size);
        let result: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        (result, n_tokens)
    }

    /// Run the encoder transformer on GPU.
    /// Input: token embeddings from conv stem [seq_len, d_model] as f32.
    /// Output: encoder output [seq_len, output_dim] as f32.
    pub fn forward(&mut self, token_embeddings: &[f32], seq_len: u32) -> Vec<f32> {
        use crate::gpu::bind;

        let d = self.config.d_model;
        let ffn_d = self.config.ffn_dim;
        let out_d = self.config.output_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let t0 = std::time::Instant::now();

        // Upload input to GPU
        let x = self.gpu.upload_buffer("enc_input",
            bytemuck::cast_slice(token_embeddings));

        // Allocate scratch buffers
        let buf_size = (seq_len * d) as u64 * 4;
        let x_norm = self.gpu.create_storage_buffer("enc_x_norm", buf_size);
        let q_buf = self.gpu.create_storage_buffer("enc_q", buf_size);
        let k_buf = self.gpu.create_storage_buffer("enc_k", buf_size);
        let v_buf = self.gpu.create_storage_buffer("enc_v", buf_size);
        let attn_out = self.gpu.create_storage_buffer("enc_attn", buf_size);
        let o_out = self.gpu.create_storage_buffer("enc_o", buf_size);

        let ffn_size = (seq_len * ffn_d) as u64 * 4;
        let ffn_mid = self.gpu.create_storage_buffer("enc_ffn_mid", ffn_size);
        let ffn_act = self.gpu.create_storage_buffer("enc_ffn_act", ffn_size);
        let ffn_out = self.gpu.create_storage_buffer("enc_ffn_out", buf_size);

        let out_size = (seq_len * out_d) as u64 * 4;
        let output_buf = self.gpu.create_storage_buffer("enc_output", out_size);

        // Params uniform buffer (overwritten between dispatches)
        let params_buf = self.gpu.create_buffer("enc_params", 64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        let alloc_ms = t0.elapsed().as_millis();
        log::info!("[asr-encoder] buffers allocated in {}ms, seq_len={}", alloc_ms, seq_len);

        // ── Split borrows for transformer layers + output projection ──
        let Self { gpu, layers, conv: _, ln_post_w, ln_post_b, proj1_w, proj1_b, proj2_w, proj2_b, config: _ } = self;

        let x_cur = gpu.create_storage_buffer("enc_x_cur", buf_size);
        gpu.copy_buffer(&x, &x_cur, buf_size);

        for layer_idx in 0..layers.len() {
            let layer = &layers[layer_idx];

            // 1. LayerNorm(x) → x_norm
            dispatch_layernorm(gpu, &x_cur, &layer.attn_norm_w, &layer.attn_norm_b,
                &x_norm, &params_buf, seq_len, d);

            // 2-4. Q/K/V projections via bf16 GEMM
            dispatch_bf16_gemm(gpu, &x_norm, &layer.wq, &layer.bq, &q_buf, &params_buf,
                seq_len, d, d, true);
            dispatch_bf16_gemm(gpu, &x_norm, &layer.wk, &layer.bk, &k_buf, &params_buf,
                seq_len, d, d, true);
            dispatch_bf16_gemm(gpu, &x_norm, &layer.wv, &layer.bv, &v_buf, &params_buf,
                seq_len, d, d, true);

            // 5. Bidirectional windowed attention
            dispatch_bidir_attn(gpu, &q_buf, &k_buf, &v_buf, &attn_out, &params_buf,
                seq_len, num_heads, head_dim, 0, seq_len);

            // 6. Output projection
            dispatch_bf16_gemm(gpu, &attn_out, &layer.wo, &layer.bo, &o_out, &params_buf,
                seq_len, d, d, true);

            // 7. Residual: x = x + o_out
            dispatch_add(gpu, &x_cur, &o_out, &params_buf, seq_len * d);

            // 8. LayerNorm(x) → x_norm
            dispatch_layernorm(gpu, &x_cur, &layer.ffn_norm_w, &layer.ffn_norm_b,
                &x_norm, &params_buf, seq_len, d);

            // 9. FFN fc1
            dispatch_bf16_gemm(gpu, &x_norm, &layer.fc1, &layer.fc1_bias, &ffn_mid,
                &params_buf, seq_len, d, ffn_d, true);

            // 10. GELU
            dispatch_gelu(gpu, &ffn_mid, &ffn_act, &params_buf, seq_len * ffn_d);

            // 11. FFN fc2
            dispatch_bf16_gemm(gpu, &ffn_act, &layer.fc2, &layer.fc2_bias, &ffn_out,
                &params_buf, seq_len, ffn_d, d, true);

            // 12. Residual: x = x + ffn_out
            dispatch_add(gpu, &x_cur, &ffn_out, &params_buf, seq_len * d);
        }

        // ── Output projection ──
        let num_layers = layers.len();

        // 13. Final LayerNorm
        dispatch_layernorm(gpu, &x_cur, ln_post_w, ln_post_b, &x_norm, &params_buf, seq_len, d);

        // 14. proj1 + GELU
        dispatch_bf16_gemm(gpu, &x_norm, proj1_w, proj1_b, &ffn_mid, &params_buf, seq_len, d, d, true);
        dispatch_gelu(gpu, &ffn_mid, &ffn_act, &params_buf, seq_len * d);

        // 15. proj2
        dispatch_bf16_gemm(gpu, &ffn_act, proj2_w, proj2_b, &output_buf, &params_buf, seq_len, d, out_d, true);

        // Readback
        let result_bytes = gpu.read_buffer(&output_buf, out_size);
        let result: &[f32] = bytemuck::cast_slice(&result_bytes);

        let total_ms = t0.elapsed().as_millis();
        log::info!("[asr-encoder] forward: {}ms for {} tokens ({} layers)",
            total_ms, seq_len, num_layers);

        result.to_vec()
    }

    fn parse_config(path: &Path) -> AsrEncoderConfig {
        let text = std::fs::read_to_string(path).expect("config.json not found");
        let v: serde_json::Value = serde_json::from_str(&text).expect("invalid config.json");

        // Config path: thinker_config.audio_config or audio_config (Qwen3-ASR format)
        let enc = if v["thinker_config"]["audio_config"].is_object() {
            &v["thinker_config"]["audio_config"]
        } else if v["audio_config"].is_object() {
            &v["audio_config"]
        } else {
            panic!("no audio_config found in config.json");
        };
        let d_model = enc["d_model"].as_u64().unwrap_or(1024) as u32;
        let num_layers = enc["encoder_layers"].as_u64().unwrap_or(24) as u32;
        let num_heads = enc["encoder_attention_heads"].as_u64().unwrap_or(16) as u32;
        let ffn_dim = enc["encoder_ffn_dim"].as_u64().unwrap_or(4096) as u32;
        // output_dim is d_model for the projection — check for explicit value
        let output_dim = enc.get("output_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(d_model as u64 * 2) as u32;  // typically 2x d_model

        AsrEncoderConfig { d_model, num_layers, num_heads, head_dim: 64, ffn_dim, output_dim }
    }
}

// ── Free-standing dispatch helpers (avoid borrow conflicts with self.layers + self.gpu) ──

fn dispatch_layernorm(
    gpu: &mut GpuContext, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer,
    output: &wgpu::Buffer, params: &wgpu::Buffer, seq_len: u32, dim: u32,
) {
    use crate::gpu::bind;
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
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
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { d_in: u32, d_out: u32, seq_len: u32, has_bias: u32 }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P {
        d_in, d_out, seq_len, has_bias: has_bias as u32,
    }));
    gpu.dispatch("bf16_gemm", shaders::BF16_GEMM, &[
        bind(0, input), bind(1, weight), bind(2, bias), bind(3, output), bind(4, params),
    ], (d_out.div_ceil(32), seq_len, 1));
}

fn dispatch_gelu(
    gpu: &mut GpuContext, input: &wgpu::Buffer, output: &wgpu::Buffer,
    params: &wgpu::Buffer, n: u32,
) {
    use crate::gpu::bind;
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { n: u32, _pad: [u32; 3] }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P { n, _pad: [0; 3] }));
    gpu.dispatch("gelu_mul", shaders::GELU_MUL, &[
        bind(0, input), bind(1, output), bind(2, params),
    ], (n.div_ceil(256), 1, 1));
}

fn dispatch_bidir_attn(
    gpu: &mut GpuContext, q: &wgpu::Buffer, k: &wgpu::Buffer, v: &wgpu::Buffer,
    output: &wgpu::Buffer, params: &wgpu::Buffer,
    seq_len: u32, num_heads: u32, head_dim: u32,
    window_start: u32, window_end: u32,
) {
    use crate::gpu::bind;
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { seq_len: u32, head_dim: u32, num_heads: u32,
               window_start: u32, window_end: u32, _pad: [u32; 3] }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P {
        seq_len, head_dim, num_heads, window_start, window_end, _pad: [0; 3],
    }));
    gpu.dispatch("qwen_asr_bidir_attn", shaders::BIDIR_ATTN, &[
        bind(0, q), bind(1, k), bind(2, v), bind(3, output), bind(4, params),
    ], (num_heads, seq_len, 1));
}

// ── Conv stem shader builders ───────────────────────────────────────────

/// Conv2D + GELU shader. c_in and c_out are baked as constants.
/// Spatial dims (h_in, w_in, h_out, w_out) come from uniform params.
/// Kernel: 3×3, stride=2, padding=1.
pub fn build_conv2d_gelu_shader(c_in: u32, c_out: u32) -> String {
    format!("\
const C_IN: u32 = {c_in}u;
const C_OUT: u32 = {c_out}u;

struct Params {{ c_in: u32, h_in: u32, w_in: u32, h_out: u32, w_out: u32, }}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let total = C_OUT * p.h_out * p.w_out;
    if (idx >= total) {{ return; }}
    let oc = idx / (p.h_out * p.w_out);
    let rem = idx % (p.h_out * p.w_out);
    let oh = rem / p.w_out;
    let ow = rem % p.w_out;
    var sum = bias[oc];
    let w_base = oc * C_IN * 9u;
    for (var ic: u32 = 0u; ic < C_IN; ic++) {{
        let ic_base = w_base + ic * 9u;
        let in_base = ic * p.h_in * p.w_in;
        for (var kh: u32 = 0u; kh < 3u; kh++) {{
            let ih = oh * 2u + kh;
            if (ih == 0u || ih > p.h_in) {{ continue; }}
            let ih_adj = ih - 1u;
            let row_base = in_base + ih_adj * p.w_in;
            for (var kw: u32 = 0u; kw < 3u; kw++) {{
                let iw = ow * 2u + kw;
                if (iw == 0u || iw > p.w_in) {{ continue; }}
                sum += input[row_base + iw - 1u] * weight[ic_base + kh * 3u + kw];
            }}
        }}
    }}
    // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let x = sum;
    let cdf = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    output[idx] = x * cdf;
}}", c_in=c_in, c_out=c_out)
}

/// Fused reshape + linear projection + sinusoidal PE.
/// Reads conv3 output [480, h3, w3], reshapes to [w3, 480*h3],
/// multiplies by proj_w [d_model, 480*h3], adds sinusoidal PE.
/// Dispatch: (w3, ceil(d_model/32), 1) — one thread per (token, d_model_chunk).
pub fn build_reshape_proj_pe_shader() -> String {
    format!("\
const CONV_H: u32 = {ch}u;

struct Params {{ h3: u32, w3: u32, d_model: u32, }}

@group(0) @binding(0) var<storage, read> conv3: array<f32>;
@group(0) @binding(1) var<storage, read> proj_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let t = wg.x;    // token index (0..w3)
    let d = wg.y * 32u + lid.x;  // d_model index
    if (t >= p.w3 || d >= p.d_model) {{ return; }}
    let proj_dim = CONV_H * p.h3;
    // Reshape: conv3[ch, f, w3] → reshaped[t, ch*h3+f]
    // Then dot product with proj_w[d, :]
    var sum: f32 = 0.0;
    let w_base = d * proj_dim;
    for (var ch: u32 = 0u; ch < CONV_H; ch++) {{
        for (var f: u32 = 0u; f < p.h3; f++) {{
            let k = ch * p.h3 + f;
            let conv_idx = ch * p.h3 * p.w3 + f * p.w3 + t;
            sum += conv3[conv_idx] * proj_w[w_base + k];
        }}
    }}
    // Sinusoidal PE: pe[t, d] = sin/cos(t / 10000^(2*floor(d/2)/d_model))
    let half_d = d / 2u;
    let freq = 1.0 / pow(10000.0, f32(half_d * 2u) / f32(p.d_model));
    let angle = f32(t) * freq;
    let pe = select(cos(angle), sin(angle), d % 2u == 0u);
    output[t * p.d_model + d] = sum + pe;
}}", ch=CONV_HIDDEN)
}

/// Transpose conv2d weights from MLX [out,kH,kW,in] to PyTorch [out,in,kH,kW] if needed.
fn transpose_conv2d_if_mlx(w: &[f32], c_out: usize, c_in: usize, kh: usize, kw: usize) -> Vec<f32> {
    if c_in <= 1 { return w.to_vec(); } // can't detect for depthwise
    // MLX layout: inner stride is c_in (adjacent values span input channels)
    // PyTorch: inner stride is 1 (adjacent values span kW)
    // Heuristic: compare variance at c_in stride vs unit stride
    let stride = c_in;
    if stride > kw && w.len() > stride * 2 {
        let unit_var: f32 = (0..kw.min(3)).map(|i| (w[i] - w[0]).abs()).sum();
        let strided_var: f32 = (0..kw.min(3)).map(|i| (w[i * stride] - w[0]).abs()).sum();
        if strided_var >= unit_var { return w.to_vec(); } // already PyTorch layout
    } else {
        return w.to_vec();
    }
    log::info!("encoder: transposing conv2d weights from MLX layout ({}x{}x{}x{})", c_out, c_in, kh, kw);
    let mut out = vec![0.0f32; c_out * c_in * kh * kw];
    for oc in 0..c_out {
        for ic in 0..c_in {
            for h in 0..kh {
                for wi in 0..kw {
                    out[oc * c_in * kh * kw + ic * kh * kw + h * kw + wi] =
                        w[oc * kh * kw * c_in + h * kw * c_in + wi * c_in + ic];
                }
            }
        }
    }
    out
}

/// In-place add: a[i] += b[i].
fn dispatch_add(
    gpu: &mut GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer,
    params: &wgpu::Buffer, n: u32,
) {
    use crate::gpu::bind;
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct P { n: u32 }
    gpu.flush();
    gpu.write_buffer(params, 0, bytemuck::bytes_of(&P { n }));
    gpu.dispatch("add", shaders::ADD, &[
        bind(0, a), bind(1, b), bind(2, params),
    ], (n.div_ceil(256), 1, 1));
}

