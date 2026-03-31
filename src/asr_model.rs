//! AsrModel: zero-write GPU-resident ASR decoder for MLX INT4.
//!
//! All shader parameters are baked as WGSL `const` declarations at construction.
//! Per-token decode dispatches zero CPU→GPU writes. The GPU-side sequence counter,
//! argmax result, and token ring buffer keep the decode loop fully on-device.
//!
//! Separate from `Model` which handles general LLM inference (Qwen3.5 thinker, etc).

use crate::gpu::{self, GpuContext};
use crate::weights::{ModelConfig, ModelWeights, QuantConfig};

// ── State ─────────────────────────────────────────────────────────────

/// GPU buffers for ASR decoder inference.
pub struct AsrState {
    pub hidden: wgpu::Buffer,
    pub residual: wgpu::Buffer,
    pub normed: wgpu::Buffer,
    pub q_out: wgpu::Buffer,
    pub k_out: wgpu::Buffer,
    pub v_out: wgpu::Buffer,
    pub q_proj: wgpu::Buffer,     // Q after qknorm+RoPE (separate from attn_output)
    pub attn_output: wgpu::Buffer,
    pub o_proj_out: wgpu::Buffer,
    pub gate_out: wgpu::Buffer,
    pub up_out: wgpu::Buffer,
    pub mlp_output: wgpu::Buffer,
    pub k_cache: Vec<wgpu::Buffer>,
    pub v_cache: Vec<wgpu::Buffer>,
    pub logits: wgpu::Buffer,
    pub qknorm_params: Vec<wgpu::Buffer>,

    // GPU-resident decode state — no CPU writes per token
    /// Sequence position counter (atomic<u32>). Incremented on GPU each token.
    pub seq_counter: wgpu::Buffer,
    /// Argmax result {idx: u32, val: f32}. Written by GPU argmax, read by embedding.
    pub argmax_result: wgpu::Buffer,
    /// Generated token IDs ring buffer (max_decode_tokens entries).
    pub token_ring: wgpu::Buffer,
    /// Winning logit value per token (parallel to token_ring).
    pub logit_ring: wgpu::Buffer,
    /// Number of tokens written to token_ring (atomic<u32>).
    pub token_count: wgpu::Buffer,
    /// EOS flag (atomic<u32>). Set by store_token_eos shader.
    pub eos_flag: wgpu::Buffer,
    /// Embedding params uniform buffer (reused, not re-created per call).
    pub embed_params: wgpu::Buffer,
}

const MAX_DECODE_TOKENS: u32 = 448;

// ── Model ─────────────────────────────────────────────────────────────

pub struct AsrModel {
    pub config: ModelConfig,
    pub quant_config: QuantConfig,
    pub weights: ModelWeights,
    pub state: AsrState,
    pub seq_len: u32,
    pub generated_tokens: Vec<u32>,

    // Const-specialized shaders (all dims baked, no params uniform)
    s_matvec_q: String,
    s_matvec_kv: String,
    s_matvec_o: String,
    s_matvec_gate: String,
    s_matvec_up: String,
    s_silu_down: String,
    s_rmsnorm: String,
    s_add_rmsnorm: String,
    s_qknorm: String,
    s_gqa: String,
    pub s_lm_head: Vec<String>,  // one per embed chunk
    pub s_argmax: String,
    s_embed_from_argmax: String,
    s_fused_gate_up_silu: Option<String>, // 8-bit fused gate+up+SiLU (None for 4-bit)
    s_embedding: String, // int4 or int8 embedding shader
    s_inc_seq: String,
    s_store_eos: String,
}

impl AsrModel {
    pub fn new(gpu: &GpuContext, config: ModelConfig, quant_config: QuantConfig,
               weights: ModelWeights, max_seq_len: u32) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let nl = config.num_hidden_layers;
        let gs = quant_config.group_size;
        let f = 4u64;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;

        // ── Build const-specialized shaders (4-bit or 8-bit) ──
        let bits = quant_config.bits;
        let build_mv = |i: u32, o: u32| -> String {
            if bits == 8 { build_int8_matvec_const(i, o, gs) }
            else { build_int4_matvec_const(i, o, gs) }
        };
        let build_mv_off = |i: u32, o: u32, off: u32| -> String {
            if bits == 8 { build_int8_matvec_const_offset(i, o, gs, off) }
            else { build_int4_matvec_const_offset(i, o, gs, off) }
        };

        let s_matvec_q = build_mv(h, q_dim);
        let s_matvec_kv = build_mv(h, kv_dim);
        let s_matvec_o = build_mv(q_dim, h);
        let s_matvec_gate = build_mv(h, inter);  // unused for 8-bit (fused)
        let s_matvec_up = build_mv(h, inter);    // unused for 8-bit (fused)
        let s_silu_down = if bits == 8 {
            // 8-bit: down_proj is a plain matvec (gate+up+SiLU fused separately)
            build_int8_matvec_const(inter, h, gs)
        } else {
            build_fused_silu_int4_const(inter, h, gs)
        };
        let s_fused_gate_up_silu = if bits == 8 {
            Some(build_fused_gate_up_silu_int8_const(h, inter, gs))
        } else {
            None
        };
        let s_rmsnorm = build_rmsnorm_direct_const(h, config.rms_norm_eps);
        let s_add_rmsnorm = build_add_rmsnorm_direct_const(h, config.rms_norm_eps);
        let s_gqa = build_gqa_attention_const(hd, nkv, nh, nh / nkv);
        let s_argmax = build_argmax_const(config.vocab_size);

        let s_qknorm = {
            let partial_dim = (hd as f32 * config.partial_rotary_factor) as u32;
            build_qknorm_const(&config, partial_dim)
        };

        // LM head: quantized matvec with output offset for chunking.
        let s_lm_head = if !weights.embed_chunks.is_empty() {
            let cs = weights.embed_chunk_size;
            (0..weights.embed_chunks.len()).map(|ci| {
                let start = ci as u32 * cs;
                let end = ((ci as u32 + 1) * cs).min(config.vocab_size);
                build_mv_off(h, end - start, start)
            }).collect()
        } else {
            vec![build_mv_off(h, config.vocab_size, 0)]
        };

        let chunk_boundary = if !weights.embed_chunks.is_empty() {
            weights.embed_chunk_size
        } else {
            config.vocab_size // no chunking: boundary beyond vocab
        };
        let s_embed_from_argmax = if bits == 8 {
            build_embed_from_argmax_int8_const(h, gs, chunk_boundary)
        } else {
            build_embed_from_argmax_int4_const(h, gs, chunk_boundary)
        };
        let s_embedding = if bits == 8 {
            include_str!("shaders/int8_embedding_mlx.wgsl").to_string()
        } else {
            include_str!("shaders/int4_embedding_mlx.wgsl").to_string()
        };
        let s_inc_seq = include_str!("shaders/increment_seq_counter_const.wgsl").to_string();
        let s_store_eos = build_store_token_eos_const();

        // ── QK norm params ──
        let qknorm_buf_size = 32 + 320 * 16;
        let qknorm_params: Vec<wgpu::Buffer> = (0..nl)
            .map(|i| gpu.create_buffer(&format!("asr_qknorm_{i}"), qknorm_buf_size as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST))
            .collect();

        // ── Allocate state buffers ──
        let state = AsrState {
            hidden: gpu.create_storage_buffer("asr_hidden", h as u64 * f),
            residual: gpu.create_storage_buffer("asr_residual", h as u64 * f),
            normed: gpu.create_storage_buffer("asr_normed", h as u64 * f),
            q_out: gpu.create_storage_buffer("asr_q", q_dim as u64 * f),
            k_out: gpu.create_storage_buffer("asr_k", kv_dim as u64 * f),
            v_out: gpu.create_storage_buffer("asr_v", kv_dim as u64 * f),
            q_proj: gpu.create_storage_buffer("asr_qproj", q_dim as u64 * f),
            attn_output: gpu.create_storage_buffer("asr_attn", q_dim as u64 * f),
            o_proj_out: gpu.create_storage_buffer("asr_o", h as u64 * f),
            gate_out: gpu.create_storage_buffer("asr_gate", inter as u64 * f),
            up_out: gpu.create_storage_buffer("asr_up", inter as u64 * f),
            mlp_output: gpu.create_storage_buffer("asr_mlp", h as u64 * f),
            k_cache: (0..nl).map(|i| gpu.create_storage_buffer(
                &format!("asr_kc_{i}"), max_seq_len as u64 * kv_dim as u64 * f)).collect(),
            v_cache: (0..nl).map(|i| gpu.create_storage_buffer(
                &format!("asr_vc_{i}"), max_seq_len as u64 * kv_dim as u64 * f)).collect(),
            logits: gpu.create_storage_buffer("asr_logits", config.vocab_size as u64 * f),
            qknorm_params,
            seq_counter: gpu.create_storage_buffer("asr_seq_ctr", 4),
            argmax_result: gpu.create_storage_buffer("asr_argmax", 8),
            token_ring: gpu.create_storage_buffer("asr_tok_ring", MAX_DECODE_TOKENS as u64 * 4),
            logit_ring: gpu.create_storage_buffer("asr_logit_ring", MAX_DECODE_TOKENS as u64 * 4),
            token_count: gpu.create_storage_buffer("asr_tok_cnt", 4),
            eos_flag: gpu.create_storage_buffer("asr_eos", 4),
            embed_params: gpu.create_buffer("asr_emb_params", 16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
        };

        Self {
            config, quant_config, weights, state,
            seq_len: 0, generated_tokens: Vec::new(),
            s_matvec_q, s_matvec_kv, s_matvec_o,
            s_matvec_gate, s_matvec_up, s_silu_down,
            s_rmsnorm, s_add_rmsnorm, s_qknorm, s_gqa,
            s_lm_head, s_argmax, s_embed_from_argmax,
            s_fused_gate_up_silu,
            s_embedding,
            s_inc_seq, s_store_eos,
        }
    }

    pub fn init_qknorm_params(&self, gpu: &GpuContext, layer: usize, q_norm: &[u8], k_norm: &[u8]) {
        let data = crate::model::build_qknorm_params(&self.config, q_norm, k_norm);
        gpu.write_buffer(&self.state.qknorm_params[layer], 0, &data);
    }

    /// MLX INT4 embedding lookup (handles chunked embeddings for 128MB binding).
    pub fn embedding(&self, gpu: &mut GpuContext, token_id: u32) {
        let sc = self.weights.mlx_embed_scales.as_ref().expect("mlx_embed_scales");
        let bi = self.weights.mlx_embed_biases.as_ref().expect("mlx_embed_biases");
        let (embed_buf, local_id) = if !self.weights.embed_chunks.is_empty() {
            let cs = self.weights.embed_chunk_size;
            (&self.weights.embed_chunks[(token_id / cs) as usize], token_id % cs)
        } else {
            (&self.weights.embed_tokens, token_id)
        };
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { token_id: u32, dim: u32, group_size: u32, token_id_global: u32 }
        // Must flush before writing to the shared params buffer — previous dispatches
        // may still be reading the old value.
        gpu.flush();
        gpu.queue.write_buffer(&self.state.embed_params, 0, bytemuck::bytes_of(&P {
            token_id: local_id, dim: self.config.hidden_size,
            group_size: self.quant_config.group_size, token_id_global: token_id,
        }));
        gpu.dispatch("asr_emb", &self.s_embedding, &[
            gpu::bind(0, embed_buf), gpu::bind(1, sc), gpu::bind(2, bi),
            gpu::bind(3, &self.state.hidden), gpu::bind(4, &self.state.embed_params),
        ], (self.config.hidden_size.div_ceil(256), 1, 1));
    }

    // ── Zero-write forward ────────────────────────────────────────────

    /// Single-token forward pass. Zero CPU writes — reads seq_counter from GPU.
    /// Must be preceded by increment_seq_counter dispatch.
    /// Assumes hidden+residual already contain the token embedding.
    pub fn forward_layers(&self, gpu: &mut GpuContext) {
        let h = self.config.hidden_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let inter = self.config.intermediate_size;

        for i in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[i];
            let biases = &self.weights.mlx_biases[i];

            // ── Norm ──
            if i == 0 {
                gpu.dispatch("asr_norm_const", &self.s_rmsnorm, &[
                    gpu::bind(0, &self.state.hidden),
                    gpu::bind(1, &layer.input_layernorm),
                    gpu::bind(2, &self.state.normed),
                ], (1, 1, 1));
            } else {
                gpu.dispatch("asr_addnorm_const", &self.s_add_rmsnorm, &[
                    gpu::bind(0, &self.state.residual),
                    gpu::bind(1, &self.state.mlp_output),
                    gpu::bind(2, &layer.input_layernorm),
                    gpu::bind(3, &self.state.normed),
                ], (1, 1, 1));
            }

            // ── QKV projections ──
            if let Some(sa) = layer.self_attn() {
                gpu.dispatch("asr_q_const", &self.s_matvec_q, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &sa.q_proj_qweight),
                    gpu::bind(2, &sa.q_proj_scales), gpu::bind(3, &biases[0]),
                    gpu::bind(4, &self.state.q_out),
                ], ((nh * hd).div_ceil(32), 1, 1));

                gpu.dispatch("asr_k_const", &self.s_matvec_kv, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &sa.k_proj_qweight),
                    gpu::bind(2, &sa.k_proj_scales), gpu::bind(3, &biases[1]),
                    gpu::bind(4, &self.state.k_out),
                ], ((nkv * hd).div_ceil(32), 1, 1));

                gpu.dispatch("asr_v_const", &self.s_matvec_kv, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &sa.v_proj_qweight),
                    gpu::bind(2, &sa.v_proj_scales), gpu::bind(3, &biases[2]),
                    gpu::bind(4, &self.state.v_out),
                ], ((nkv * hd).div_ceil(32), 1, 1));

                // ── QK norm + RoPE + KV cache write (reads seq_counter at binding 8) ──
                gpu.dispatch("asr_qknorm_const", &self.s_qknorm, &[
                    gpu::bind(0, &self.state.q_out),
                    gpu::bind(1, &self.state.k_out),
                    gpu::bind(2, &self.state.v_out),
                    gpu::bind(3, &self.state.q_proj),      // q_proj output (separate from attn_output)
                    gpu::bind(4, &self.state.gate_out),    // q_gate (unused, non-gated)
                    gpu::bind(5, &self.state.k_cache[i]),
                    gpu::bind(6, &self.state.v_cache[i]),
                    gpu::bind(7, &self.state.qknorm_params[i]),
                    gpu::bind(8, &self.state.seq_counter),
                ], (nh + nkv, 1, 1));

                // ── GQA attention (reads seq_counter for seq_len) ──
                gpu.dispatch("asr_gqa_const", &self.s_gqa, &[
                    gpu::bind(0, &self.state.q_proj),
                    gpu::bind(1, &self.state.k_cache[i]),
                    gpu::bind(2, &self.state.v_cache[i]),
                    gpu::bind(3, &self.state.attn_output),
                    gpu::bind(4, &self.state.seq_counter),
                ], (nh, 1, 1));

                // ── O projection ──
                gpu.dispatch("asr_o_const", &self.s_matvec_o, &[
                    gpu::bind(0, &self.state.attn_output), gpu::bind(1, &sa.o_proj_qweight),
                    gpu::bind(2, &sa.o_proj_scales), gpu::bind(3, &biases[3]),
                    gpu::bind(4, &self.state.o_proj_out),
                ], (h.div_ceil(32), 1, 1));
            }

            // ── Post-attention norm ──
            gpu.dispatch("asr_postnorm_const", &self.s_add_rmsnorm, &[
                gpu::bind(0, &self.state.residual),
                gpu::bind(1, &self.state.o_proj_out),
                gpu::bind(2, &layer.post_attn_layernorm),
                gpu::bind(3, &self.state.normed),
            ], (1, 1, 1));

            // ── MLP ──
            if let Some(ref fused) = self.s_fused_gate_up_silu {
                // 8-bit: fused gate+up+SiLU → gate_out, then plain matvec down
                gpu.dispatch("asr_gate_up_silu_const", fused, &[
                    gpu::bind(0, &self.state.normed),
                    gpu::bind(1, &layer.gate_proj_qweight),
                    gpu::bind(2, &layer.gate_proj_scales), gpu::bind(3, &biases[4]),
                    gpu::bind(4, &layer.up_proj_qweight),
                    gpu::bind(5, &layer.up_proj_scales), gpu::bind(6, &biases[5]),
                    gpu::bind(7, &self.state.gate_out),
                ], (inter.div_ceil(8), 1, 1)); // 8 cols per workgroup with 4-thread lanes

                gpu.dispatch("asr_down_const", &self.s_silu_down, &[
                    gpu::bind(0, &self.state.gate_out),
                    gpu::bind(1, &layer.down_proj_qweight),
                    gpu::bind(2, &layer.down_proj_scales), gpu::bind(3, &biases[6]),
                    gpu::bind(4, &self.state.mlp_output),
                ], (h.div_ceil(32), 1, 1));
            } else {
                // 4-bit: separate gate + up + fused SiLU+down
                gpu.dispatch("asr_gate_const", &self.s_matvec_gate, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &layer.gate_proj_qweight),
                    gpu::bind(2, &layer.gate_proj_scales), gpu::bind(3, &biases[4]),
                    gpu::bind(4, &self.state.gate_out),
                ], (inter.div_ceil(32), 1, 1));

                gpu.dispatch("asr_up_const", &self.s_matvec_up, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &layer.up_proj_qweight),
                    gpu::bind(2, &layer.up_proj_scales), gpu::bind(3, &biases[5]),
                    gpu::bind(4, &self.state.up_out),
                ], (inter.div_ceil(32), 1, 1));

                gpu.dispatch("asr_silu_down_const", &self.s_silu_down, &[
                    gpu::bind(0, &self.state.gate_out), gpu::bind(1, &self.state.up_out),
                    gpu::bind(2, &layer.down_proj_qweight),
                    gpu::bind(3, &layer.down_proj_scales), gpu::bind(4, &biases[6]),
                    gpu::bind(5, &self.state.mlp_output),
                ], (h.div_ceil(32), 1, 1));
            }
        }

        // ── Final norm ──
        gpu.dispatch("asr_final_norm_const", &self.s_add_rmsnorm, &[
            gpu::bind(0, &self.state.residual),
            gpu::bind(1, &self.state.mlp_output),
            gpu::bind(2, &self.weights.final_norm),
            gpu::bind(3, &self.state.normed),
        ], (1, 1, 1));

        // ── LM head: INT4 matvec with tied embed weights (bind normed directly) ──
        let sc = self.weights.mlx_embed_scales.as_ref().unwrap();
        let bi = self.weights.mlx_embed_biases.as_ref().unwrap();
        let chunks = if !self.weights.embed_chunks.is_empty() {
            &self.weights.embed_chunks[..]
        } else {
            std::slice::from_ref(&self.weights.embed_tokens)
        };
        let cs = if !self.weights.embed_chunks.is_empty() { self.weights.embed_chunk_size } else { self.config.vocab_size };
        for (ci, (chunk, shader)) in chunks.iter().zip(self.s_lm_head.iter()).enumerate() {
            let n = ((ci as u32 + 1) * cs).min(self.config.vocab_size) - ci as u32 * cs;
            gpu.dispatch(&format!("asr_lmh_const_{ci}"), shader, &[
                gpu::bind(0, &self.state.normed),
                gpu::bind(1, chunk),
                gpu::bind(2, sc),
                gpu::bind(3, bi),
                gpu::bind(4, &self.state.logits),
            ], (n.div_ceil(32), 1, 1));
        }

        // ── GPU argmax with repetition penalty ──
        gpu.dispatch("asr_argmax_const", &self.s_argmax, &[
            gpu::bind(0, &self.state.logits),
            gpu::bind(1, &self.state.argmax_result),
            gpu::bind(2, &self.state.token_ring),
            gpu::bind(3, &self.state.token_count),
        ], (1, 1, 1));
    }

    /// Full zero-write token generation step. Zero CPU writes.
    /// Dispatches: inc_seq → embed_from_argmax → copy → layers → lm_head → argmax → store_eos.
    pub fn forward_zero_write(&self, gpu: &mut GpuContext) {
        let h = self.config.hidden_size;

        // 1. Increment sequence counter
        gpu.dispatch("asr_inc_seq_const", &self.s_inc_seq, &[
            gpu::bind(0, &self.state.seq_counter),
        ], (1, 1, 1));

        // 2. Embed from previous argmax result (GPU-side, INT4 dequant)
        let sc = self.weights.mlx_embed_scales.as_ref().unwrap();
        let bi = self.weights.mlx_embed_biases.as_ref().unwrap();
        let chunk0 = if !self.weights.embed_chunks.is_empty() {
            &self.weights.embed_chunks[0]
        } else {
            &self.weights.embed_tokens
        };
        let chunk1 = if self.weights.embed_chunks.len() > 1 {
            &self.weights.embed_chunks[1]
        } else {
            chunk0
        };
        gpu.dispatch("asr_embed_argmax_const", &self.s_embed_from_argmax, &[
            gpu::bind(0, chunk0), gpu::bind(1, chunk1),
            gpu::bind(2, sc), gpu::bind(3, bi),
            gpu::bind(4, &self.state.hidden),
            gpu::bind(5, &self.state.argmax_result),
        ], (h.div_ceil(256), 1, 1));

        // 3. Copy hidden → residual
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);

        // 4. Forward through all layers + LM head + argmax
        self.forward_layers(gpu);

        // 5. Store token and check EOS
        gpu.dispatch("asr_store_eos_const", &self.s_store_eos, &[
            gpu::bind(0, &self.state.argmax_result),
            gpu::bind(1, &self.state.eos_flag),
            gpu::bind(2, &self.state.token_ring),
            gpu::bind(3, &self.state.token_count),
            gpu::bind(4, &self.state.logit_ring),
        ], (1, 1, 1));
    }

    /// Initialize decode state: write initial token to argmax_result, set seq_counter.
    /// Called once after prefill, before the zero-write decode loop.
    pub fn init_decode(&self, gpu: &mut GpuContext, first_token: u32, initial_seq_len: u32) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct AR { idx: u32, val: f32 }
        gpu.write_buffer(&self.state.argmax_result, 0,
            bytemuck::bytes_of(&AR { idx: first_token, val: 0.0 }));
        // seq_counter = initial_seq_len - 1 because first forward_zero_write increments it
        gpu.write_buffer(&self.state.seq_counter, 0,
            bytemuck::cast_slice(&[initial_seq_len.wrapping_sub(1)]));
        gpu.write_buffer(&self.state.eos_flag, 0, &[0u8; 4]);
        gpu.write_buffer(&self.state.token_count, 0, &[0u8; 4]);
    }

    /// Read decode results after EOS or max tokens. Returns generated token IDs.
    pub fn read_generated_tokens(&self, gpu: &mut GpuContext) -> Vec<u32> {
        gpu.flush();
        let count_bytes = gpu.read_buffer(&self.state.token_count, 4);
        let count = u32::from_le_bytes(count_bytes[..4].try_into().unwrap()) as usize;
        let count = count.min(MAX_DECODE_TOKENS as usize);
        if count == 0 { return Vec::new(); }
        let ring_bytes = gpu.read_buffer(&self.state.token_ring, count as u64 * 4);
        bytemuck::cast_slice::<u8, u32>(&ring_bytes[..count * 4]).to_vec()
    }

    /// Check if EOS was reached (4-byte read).
    pub fn check_eos(&self, gpu: &mut GpuContext) -> bool {
        gpu.flush();
        let bytes = gpu.read_buffer(&self.state.eos_flag, 4);
        u32::from_le_bytes(bytes[..4].try_into().unwrap()) != 0
    }

    /// Simple forward + CPU argmax (for testing / prefill bootstrap).
    /// Writes seq_len to seq_counter so qknorm/gqa read correct position.
    pub fn forward_argmax_simple(&mut self, gpu: &mut GpuContext, token_id: u32) -> u32 {
        self.embedding(gpu, token_id);
        gpu.flush();
        gpu.copy_buffer(&self.state.hidden, &self.state.residual,
            self.config.hidden_size as u64 * 4);
        // Write current seq_len to GPU seq_counter so qknorm/gqa read correct position
        gpu.write_buffer(&self.state.seq_counter, 0,
            bytemuck::cast_slice(&[self.seq_len]));
        self.forward_layers(gpu);
        self.seq_len += 1;
        // CPU argmax
        let logits_bytes = gpu.read_buffer(&self.state.logits, self.config.vocab_size as u64 * 4);
        let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let (max_idx, _) = logits.iter().enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        let token = max_idx as u32;
        self.generated_tokens.push(token);
        token
    }
}

// ── Const-specialized shader builders ──────────────────────────────────

fn build_int4_matvec_const(in_dim: u32, out_dim: u32, gs: u32) -> String {
    format!("\
const OUT_DIM: u32 = {out_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x * 32u + lid.x;
    if (row >= OUT_DIM) {{ return; }}
    let w_row_off = row * PACKED_COLS;
    let sg_row_off = row * N_GROUPS;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < N_GROUPS; g++) {{
        let sb_idx = sg_row_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let group_start = g * PACKED_PER_GROUP;
        let input_base = g * GROUP_SIZE;
        for (var p: u32 = 0u; p < PACKED_PER_GROUP; p++) {{
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 8u;
            sum += (f32((packed) & 0xFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 4u) & 0xFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 8u) & 0xFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 12u) & 0xFu) * scale + bias) * input[ib + 3u];
            sum += (f32((packed >> 16u) & 0xFu) * scale + bias) * input[ib + 4u];
            sum += (f32((packed >> 20u) & 0xFu) * scale + bias) * input[ib + 5u];
            sum += (f32((packed >> 24u) & 0xFu) * scale + bias) * input[ib + 6u];
            sum += (f32((packed >> 28u) & 0xFu) * scale + bias) * input[ib + 7u];
        }}
    }}
    output[row] = sum;
}}",
        out_dim=out_dim, gs=gs, pc=in_dim/8, ng=in_dim/gs, ppg=gs/8,
    )
}

/// INT4 matvec with output offset — for chunked LM head where chunk N writes at offset N*chunk_size.
fn build_int4_matvec_const_offset(in_dim: u32, out_dim: u32, gs: u32, output_offset: u32) -> String {
    format!("\
const OUT_DIM: u32 = {out_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;
const OUTPUT_OFFSET: u32 = {offset}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x * 32u + lid.x;
    if (row >= OUT_DIM) {{ return; }}
    let w_row_off = row * PACKED_COLS;
    let global_row = OUTPUT_OFFSET + row;  // for scales/biases which are not chunked
    let sg_row_off = global_row * N_GROUPS;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < N_GROUPS; g++) {{
        let sb_idx = sg_row_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let group_start = g * PACKED_PER_GROUP;
        let input_base = g * GROUP_SIZE;
        for (var p: u32 = 0u; p < PACKED_PER_GROUP; p++) {{
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 8u;
            sum += (f32((packed) & 0xFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 4u) & 0xFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 8u) & 0xFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 12u) & 0xFu) * scale + bias) * input[ib + 3u];
            sum += (f32((packed >> 16u) & 0xFu) * scale + bias) * input[ib + 4u];
            sum += (f32((packed >> 20u) & 0xFu) * scale + bias) * input[ib + 5u];
            sum += (f32((packed >> 24u) & 0xFu) * scale + bias) * input[ib + 6u];
            sum += (f32((packed >> 28u) & 0xFu) * scale + bias) * input[ib + 7u];
        }}
    }}
    output[global_row] = sum;
}}",
        out_dim=out_dim, gs=gs, pc=in_dim/8, ng=in_dim/gs, ppg=gs/8, offset=output_offset,
    )
}

fn build_fused_silu_int4_const(in_dim: u32, out_dim: u32, gs: u32) -> String {
    format!("\
const OUT_DIM: u32 = {out_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> qweight: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read> biases: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x * 32u + lid.x;
    if (row >= OUT_DIM) {{ return; }}
    let w_off = row * PACKED_COLS;
    let sg_off = row * N_GROUPS;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < N_GROUPS; g++) {{
        let sb_idx = sg_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let g_start = g * PACKED_PER_GROUP;
        let ib = g * GROUP_SIZE;
        for (var p: u32 = 0u; p < PACKED_PER_GROUP; p++) {{
            let packed = qweight[w_off + g_start + p];
            let base = ib + p * 8u;
            for (var n: u32 = 0u; n < 8u; n++) {{
                let nibble = (packed >> (n * 4u)) & 0xFu;
                let w = f32(nibble) * scale + bias;
                let gv = gate[base + n];
                let silu = gv / (1.0 + exp(-gv));
                sum += w * silu * up[base + n];
            }}
        }}
    }}
    output[row] = sum;
}}",
        out_dim=out_dim, gs=gs, pc=in_dim/8, ng=in_dim/gs, ppg=gs/8,
    )
}

fn build_rmsnorm_direct_const(n: u32, eps: f32) -> String {
    format!("\
const N: u32 = {n}u;
const EPS: f32 = {eps};
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> wg_temp: array<f32, 256>;
fn unpack_bf16(packed: u32, idx: u32) -> f32 {{ let bits = (packed >> (idx * 16u)) & 0xFFFFu; return bitcast<f32>(bits << 16u); }}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < N) {{ let v = input[i]; sum_sq += v * v; i += 256u; }}
    wg_temp[tid] = sum_sq;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{ if (tid < stride) {{ wg_temp[tid] = wg_temp[tid] + wg_temp[tid + stride]; }} workgroupBarrier(); stride = stride >> 1u; }}
    let rms = 1.0 / sqrt(wg_temp[0] / f32(N) + EPS);
    i = tid;
    while (i < N) {{ let w = unpack_bf16(weight[i / 2u], i % 2u); output[i] = input[i] * rms * w; i += 256u; }}
}}", n=n, eps=eps)
}

fn build_add_rmsnorm_direct_const(n: u32, eps: f32) -> String {
    format!("\
const N: u32 = {n}u;
const EPS: f32 = {eps};
@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> wg_temp: array<f32, 256>;
fn unpack_bf16(packed: u32, idx: u32) -> f32 {{ let bits = (packed >> (idx * 16u)) & 0xFFFFu; return bitcast<f32>(bits << 16u); }}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < N) {{ let v = hidden[i] + addend[i]; hidden[i] = v; sum_sq += v * v; i += 256u; }}
    wg_temp[tid] = sum_sq;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{ if (tid < stride) {{ wg_temp[tid] = wg_temp[tid] + wg_temp[tid + stride]; }} workgroupBarrier(); stride = stride >> 1u; }}
    let rms = 1.0 / sqrt(wg_temp[0] / f32(N) + EPS);
    i = tid;
    while (i < N) {{ let w = unpack_bf16(weight[i / 2u], i % 2u); output[i] = hidden[i] * rms * w; i += 256u; }}
}}", n=n, eps=eps)
}

fn build_gqa_attention_const(hd: u32, nkv: u32, nh: u32, hpk: u32) -> String {
    format!("\
const HEAD_DIM: u32 = {hd}u;
const NUM_KV_HEADS: u32 = {nkv}u;
const NUM_Q_HEADS: u32 = {nh}u;
const HEADS_PER_KV: u32 = {hpk}u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> seq_counter: array<atomic<u32>>;

var<workgroup> shared_reduce: array<f32, 256>;
var<workgroup> shared_acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tid = lid.x;
    let q_head = wid.x;
    let seq_len = atomicLoad(&seq_counter[0]) + 1u;
    let kv_head = q_head / HEADS_PER_KV;
    let scale = 1.0 / sqrt(f32(HEAD_DIM));
    if (tid < HEAD_DIM) {{ shared_acc[tid] = 0.0; }}
    workgroupBarrier();
    var running_max: f32 = -3.402823e+38;
    var running_sum: f32 = 0.0;
    let q_base = q_head * HEAD_DIM;
    var pos: u32 = 0u;
    while (pos < seq_len) {{
        let kv_base = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
        var local_dot: f32 = 0.0;
        var d = tid;
        while (d < HEAD_DIM) {{ local_dot += q[q_base + d] * k_cache[kv_base + d]; d += 256u; }}
        shared_reduce[tid] = local_dot;
        workgroupBarrier();
        var stride = 128u;
        while (stride > 0u) {{ if (tid < stride) {{ shared_reduce[tid] = shared_reduce[tid] + shared_reduce[tid + stride]; }} workgroupBarrier(); stride = stride >> 1u; }}
        let score = shared_reduce[0] * scale;
        let new_max = max(running_max, score);
        let correction = exp(running_max - new_max);
        let exp_score = exp(score - new_max);
        running_sum = running_sum * correction + exp_score;
        if (tid < HEAD_DIM) {{ shared_acc[tid] = shared_acc[tid] * correction + exp_score * v_cache[kv_base + tid]; }}
        workgroupBarrier();
        running_max = new_max;
        pos += 1u;
    }}
    if (tid < HEAD_DIM) {{ output[q_head * HEAD_DIM + tid] = shared_acc[tid] / running_sum; }}
}}",
        hd=hd, nkv=nkv, nh=nh, hpk=hpk,
    )
}

fn build_qknorm_const(config: &ModelConfig, partial_dim: u32) -> String {
    // Build the qknorm shader that reads position from seq_counter storage buffer
    // instead of uniform params fields
    let s_limit = partial_dim / 2;
    format!(
        "const ROPE_THETA: f32 = {theta:.1};\n\
         const MROPE_S1_LIMIT: u32 = {s1}u;\n\
         const MROPE_S2_LIMIT: u32 = {s2}u;\n\
         const PARTIAL_DIM: u32 = {pd}u;\n\
         const MROPE_INTERLEAVED: bool = {interleaved};\n\
         const Q_GATED: bool = false;\n\
         const NORM_OFFSET: f32 = 0.0;\n\n\
         @group(0) @binding(8) var<storage, read> seq_counter: array<atomic<u32>>;\n\n\
         {body}",
        theta = config.rope_theta,
        s1 = s_limit.min(partial_dim / 6 + 1),
        s2 = (partial_dim / 6 * 2 + 1).min(partial_dim),
        pd = partial_dim,
        interleaved = config.mrope_interleaved(),
        body = include_str!("shaders/fused_split_qknorm_kvstore.wgsl")
            .lines().skip(8)
            .collect::<Vec<_>>().join("\n")
            .replace("params.cache_position", "atomicLoad(&seq_counter[0])")
            .replace("params.position_w", "atomicLoad(&seq_counter[0])")
            .replace("params.position_h", "atomicLoad(&seq_counter[0])")
            .replace("params.position", "atomicLoad(&seq_counter[0])"),
    )
}

fn build_argmax_const(vocab_size: u32) -> String {
    // EOS_PENALTY: logit bias subtracted from EOS/IM_END tokens before argmax.
    // REPEAT_PENALTY: logit bias subtracted from any token in the recent window.
    // REPEAT_WINDOW: how many recent tokens to penalize.
    format!("\
const N: u32 = {n}u;
const TOKEN_ENDOFTEXT: u32 = 151643u;
const TOKEN_IM_END: u32 = 151645u;
const EOS_PENALTY: f32 = 5.0;
const REPEAT_PENALTY: f32 = 3.0;
const REPEAT_WINDOW: u32 = 4u;
struct Result {{ idx: u32, val: f32, }}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: Result;
@group(0) @binding(2) var<storage, read> token_ring: array<u32>;
@group(0) @binding(3) var<storage, read> token_count: array<u32>;
var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;
var<workgroup> recent: array<u32, 4>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    let tc = token_count[0];
    // Load recent tokens into workgroup shared memory
    if (tid < REPEAT_WINDOW) {{
        let w = REPEAT_WINDOW - 1u - tid;
        recent[tid] = select(0xFFFFFFFFu, token_ring[tc - 1u - w], tc > w);
    }}
    workgroupBarrier();
    var best_val: f32 = -3.402823e+38;
    var best_idx: u32 = 0u;
    var i = tid;
    while (i < N) {{
        var v = logits[i];
        if (i == TOKEN_ENDOFTEXT || i == TOKEN_IM_END) {{ v -= EOS_PENALTY; }}
        for (var w: u32 = 0u; w < REPEAT_WINDOW; w++) {{
            if (i == recent[w]) {{ v -= REPEAT_PENALTY; break; }}
        }}
        if (v > best_val) {{ best_val = v; best_idx = i; }}
        i += 256u;
    }}
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{
        if (tid < stride) {{
            if (shared_val[tid + stride] > shared_val[tid]) {{
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }}
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}
    if (tid == 0u) {{ result.idx = shared_idx[0]; result.val = shared_val[0]; }}
}}", n=vocab_size)
}

fn build_bf16_matvec_const(k: u32, n: u32) -> String {
    format!("\
const K: u32 = {k}u;
const N: u32 = {n}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
fn unpack_bf16(packed: u32, idx: u32) -> f32 {{ let bits = (packed >> (idx * 16u)) & 0xFFFFu; return bitcast<f32>(bits << 16u); }}
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x * 32u + lid.x;
    if (row >= N) {{ return; }}
    let half_k = K / 2u;
    let w_base = row * half_k;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < half_k; i++) {{
        let p = weight[w_base + i];
        let ki = i * 2u;
        sum += unpack_bf16(p, 0u) * input[ki];
        sum += unpack_bf16(p, 1u) * input[ki + 1u];
    }}
    output[row] = sum;
}}", k=k, n=n)
}

fn build_embed_from_argmax_int4_const(dim: u32, gs: u32, chunk_boundary: u32) -> String {
    format!("\
const DIM: u32 = {dim}u;
const GROUP_SIZE: u32 = {gs}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const CHUNK_BOUNDARY: u32 = {cb}u;

struct ArgmaxResult {{ idx: u32, val: f32, }}

@group(0) @binding(0) var<storage, read> chunk0: array<u32>;
@group(0) @binding(1) var<storage, read> chunk1: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> argmax_result: ArgmaxResult;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= DIM) {{ return; }}
    let tok = argmax_result.idx;
    let local_tok = select(tok - CHUNK_BOUNDARY, tok, tok < CHUNK_BOUNDARY);
    let packed_idx = i / 8u;
    let nibble_idx = i % 8u;
    let row_off = local_tok * PACKED_COLS + packed_idx;
    let packed = select(chunk1[row_off], chunk0[row_off], tok < CHUNK_BOUNDARY);
    let nibble = (packed >> (nibble_idx * 4u)) & 0xFu;
    let group = i / GROUP_SIZE;
    let sb_idx = tok * N_GROUPS + group;
    let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
    let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
    output[i] = f32(nibble) * scale + bias;
}}",
        dim=dim, gs=gs, pc=dim/8, ng=dim/gs, cb=chunk_boundary,
    )
}

fn build_store_token_eos_const() -> String {
    include_str!("shaders/store_token_eos_const.wgsl")
        .replace("// const TOKEN_ENDOFTEXT", "const TOKEN_ENDOFTEXT")
        .replace("// const TOKEN_IM_END", "const TOKEN_IM_END")
        .replace("// const MIN_TOKENS_BEFORE_EOS", "const MIN_TOKENS_BEFORE_EOS")
}

// ── 8-bit MLX shader builders ──────────────────────────────────────────
// Same dequant formula (byte * scale + bias), 4 bytes per u32 instead of 8 nibbles.

fn build_int8_matvec_const(in_dim: u32, out_dim: u32, gs: u32) -> String {
    format!("\
const OUT_DIM: u32 = {out_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x * 32u + lid.x;
    if (row >= OUT_DIM) {{ return; }}
    let w_row_off = row * PACKED_COLS;
    let sg_row_off = row * N_GROUPS;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < N_GROUPS; g++) {{
        let sb_idx = sg_row_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let group_start = g * PACKED_PER_GROUP;
        let input_base = g * GROUP_SIZE;
        for (var p: u32 = 0u; p < PACKED_PER_GROUP; p++) {{
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 4u;
            sum += (f32((packed) & 0xFFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 8u) & 0xFFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 16u) & 0xFFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 24u) & 0xFFu) * scale + bias) * input[ib + 3u];
        }}
    }}
    output[row] = sum;
}}",
        out_dim=out_dim, gs=gs, pc=in_dim/4, ng=in_dim/gs, ppg=gs/4,
    )
}

fn build_int8_matvec_const_offset(in_dim: u32, out_dim: u32, gs: u32, output_offset: u32) -> String {
    format!("\
const OUT_DIM: u32 = {out_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;
const OUTPUT_OFFSET: u32 = {offset}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x * 32u + lid.x;
    if (row >= OUT_DIM) {{ return; }}
    let w_row_off = row * PACKED_COLS;
    let global_row = OUTPUT_OFFSET + row;
    let sg_row_off = global_row * N_GROUPS;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < N_GROUPS; g++) {{
        let sb_idx = sg_row_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let group_start = g * PACKED_PER_GROUP;
        let input_base = g * GROUP_SIZE;
        for (var p: u32 = 0u; p < PACKED_PER_GROUP; p++) {{
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 4u;
            sum += (f32((packed) & 0xFFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 8u) & 0xFFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 16u) & 0xFFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 24u) & 0xFFu) * scale + bias) * input[ib + 3u];
        }}
    }}
    output[global_row] = sum;
}}",
        out_dim=out_dim, gs=gs, pc=in_dim/4, ng=in_dim/gs, ppg=gs/4, offset=output_offset,
    )
}

/// Fused gate + up + SiLU for 8-bit MLX with 4-thread lane parallelism.
/// output[col] = SiLU(gate_sum) * up_sum
/// Dispatch: (ceil(inter / 8), 1, 1) — 8 columns per workgroup, 4 threads per column.
fn build_fused_gate_up_silu_int8_const(in_dim: u32, out_dim: u32, gs: u32) -> String {
    format!("\
const IN_DIM: u32 = {in_dim}u;
const OUT_DIM: u32 = {out_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qw_gate: array<u32>;
@group(0) @binding(2) var<storage, read> sc_gate: array<u32>;
@group(0) @binding(3) var<storage, read> bi_gate: array<u32>;
@group(0) @binding(4) var<storage, read> qw_up: array<u32>;
@group(0) @binding(5) var<storage, read> sc_up: array<u32>;
@group(0) @binding(6) var<storage, read> bi_up: array<u32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

var<workgroup> scratch: array<f32, 64>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    let lane = tid & 3u;
    let col_local = tid >> 2u;
    let col = wg_id.x * 8u + col_local;

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    if (col < OUT_DIM) {{
        let rows_per_lane = PACKED_COLS / 4u;
        let pr_start = lane * rows_per_lane;
        let pr_end = select(pr_start + rows_per_lane, PACKED_COLS, lane == 3u);
        let w_off = col * PACKED_COLS;

        var pr: u32 = pr_start;
        while (pr < pr_end) {{
            let k_base = pr * 4u;
            let group = k_base / GROUP_SIZE;
            let sb_idx = col * N_GROUPS + group;
            let sg = bitcast<f32>(((sc_gate[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
            let bg = bitcast<f32>(((bi_gate[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
            let su = bitcast<f32>(((sc_up[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
            let bu = bitcast<f32>(((bi_up[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);

            let pg = qw_gate[w_off + pr];
            let pu = qw_up[w_off + pr];

            for (var n: u32 = 0u; n < 4u; n++) {{
                let byte_g = (pg >> (n * 8u)) & 0xFFu;
                let byte_u = (pu >> (n * 8u)) & 0xFFu;
                let inp = input[k_base + n];
                gate_sum += (f32(byte_g) * sg + bg) * inp;
                up_sum += (f32(byte_u) * su + bu) * inp;
            }}
            pr += 1u;
        }}
    }}

    scratch[tid] = gate_sum;
    scratch[32u + tid] = up_sum;
    workgroupBarrier();

    if (lane == 0u && col < OUT_DIM) {{
        let b = col_local * 4u;
        let gt = scratch[b] + scratch[b + 1u] + scratch[b + 2u] + scratch[b + 3u];
        let ut = scratch[32u + b] + scratch[32u + b + 1u] + scratch[32u + b + 2u] + scratch[32u + b + 3u];
        let silu = gt / (1.0 + exp(-gt));
        output[col] = silu * ut;
    }}
}}",
        in_dim=in_dim, out_dim=out_dim, gs=gs, pc=in_dim/4, ng=in_dim/gs, ppg=gs/4,
    )
}

fn build_embed_from_argmax_int8_const(dim: u32, gs: u32, chunk_boundary: u32) -> String {
    format!("\
const DIM: u32 = {dim}u;
const GROUP_SIZE: u32 = {gs}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const CHUNK_BOUNDARY: u32 = {cb}u;
struct ArgmaxResult {{ idx: u32, val: f32, }}
@group(0) @binding(0) var<storage, read> chunk0: array<u32>;
@group(0) @binding(1) var<storage, read> chunk1: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> argmax_result: ArgmaxResult;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= DIM) {{ return; }}
    let tok = argmax_result.idx;
    let local_tok = select(tok - CHUNK_BOUNDARY, tok, tok < CHUNK_BOUNDARY);
    let packed_idx = i / 4u;
    let byte_idx = i % 4u;
    let row_off = local_tok * PACKED_COLS + packed_idx;
    let packed = select(chunk1[row_off], chunk0[row_off], tok < CHUNK_BOUNDARY);
    let byte_val = (packed >> (byte_idx * 8u)) & 0xFFu;
    let group = i / GROUP_SIZE;
    let sb_idx = tok * N_GROUPS + group;
    let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
    let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
    output[i] = f32(byte_val) * scale + bias;
}}",
        dim=dim, gs=gs, pc=dim/4, ng=dim/gs, cb=chunk_boundary,
    )
}

// ── Batched prefill shader builders ─────────────────────────────────────
