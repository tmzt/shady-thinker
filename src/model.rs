use crate::gpu::{self, GpuContext};
#[cfg(feature = "jit-lora")]
use crate::lora::LoraState;
use crate::weights::{ModelConfig, ModelWeights, QuantConfig};

mod shaders {
    pub const GPTQ_MATVEC_4T: &str = include_str!("shaders/gptq_matvec_4t.wgsl");
    pub const FUSED_GATE_UP_SILU_4T: &str = include_str!("shaders/fused_gate_up_silu_4t.wgsl");
    pub const FUSED_SILU_GPTQ_4T: &str = include_str!("shaders/fused_silu_gptq_4t.wgsl");
    pub const ADD_RMSNORM: &str = include_str!("shaders/add_rmsnorm.wgsl");
    pub const RMSNORM: &str = include_str!("shaders/rmsnorm.wgsl");
    pub const EMBEDDING: &str = include_str!("shaders/embedding.wgsl");
    pub const ARGMAX: &str = include_str!("shaders/argmax.wgsl");
    pub const SIGMOID_MUL: &str = include_str!("shaders/sigmoid_mul.wgsl");
    pub const GQA_ATTENTION_HEAD: &str = include_str!("shaders/gqa_attention_head.wgsl");
    pub const GQA_REDUCE: &str = include_str!("shaders/gqa_reduce.wgsl");
    pub const GPTQ_MATVEC: &str = include_str!("shaders/gptq_matvec.wgsl");
    pub const FUSED_SILU_GPTQ: &str = include_str!("shaders/fused_silu_gptq.wgsl");
    pub const BF16_MATVEC: &str = include_str!("shaders/bf16_matvec.wgsl");
    pub const FUSED_CONV_DELTANET_NORM: &str = include_str!("shaders/fused_conv_deltanet_norm.wgsl");

    // LoRA shaders
    #[cfg(feature = "jit-lora")]
    pub const LORA_DOWN: &str = include_str!("shaders/lora_down.wgsl");
    #[cfg(feature = "jit-lora")]
    pub const LORA_UP_ADD: &str = include_str!("shaders/lora_up_add.wgsl");
    #[cfg(feature = "jit-lora")]
    pub const LORA_DOWN_SILU: &str = include_str!("shaders/lora_down_silu.wgsl");
}

/// Build the fused_split_qknorm_kvstore shader source with model-specific constants.
fn build_qknorm_shader(config: &ModelConfig) -> String {
    let partial_dim = (config.head_dim as f32 * config.partial_rotary_factor) as u32;
    let interleaved = config.mrope_interleaved();
    // mRoPE section limits for modulo-3 interleaved selection
    let s_limit = partial_dim / 2;
    format!(
        "const ROPE_THETA: f32 = {:.1};\n\
         const MROPE_S1_LIMIT: u32 = {}u;\n\
         const MROPE_S2_LIMIT: u32 = {}u;\n\
         const PARTIAL_DIM: u32 = {}u;\n\
         const MROPE_INTERLEAVED: bool = {};\n\n{}",
        config.rope_theta,
        s_limit,
        s_limit,
        partial_dim,
        interleaved,
        include_str!("shaders/fused_split_qknorm_kvstore.wgsl")
            .lines()
            .skip(5) // skip the hardcoded const lines (now 5 lines)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Runtime buffers for inference
pub struct InferenceState {
    pub hidden: wgpu::Buffer,
    pub residual: wgpu::Buffer,
    pub normed: wgpu::Buffer,
    pub q_out: wgpu::Buffer,
    pub q_proj: wgpu::Buffer,
    pub q_gate: wgpu::Buffer,
    pub k_out: wgpu::Buffer,
    pub v_out: wgpu::Buffer,
    pub attn_output: wgpu::Buffer,
    pub attn_partials: wgpu::Buffer,
    pub o_proj_out: wgpu::Buffer,
    pub gate_out: wgpu::Buffer,
    pub up_out: wgpu::Buffer,
    pub mlp_output: wgpu::Buffer,
    pub k_cache: Vec<wgpu::Buffer>,
    pub v_cache: Vec<wgpu::Buffer>,
    pub logits: wgpu::Buffer,
    pub argmax_result: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub qknorm_params: Vec<wgpu::Buffer>,

    // DeltaNet state (per linear-attn layer)
    pub deltanet_qkv: wgpu::Buffer,      // [total_channels] for conv input
    pub deltanet_hist: Vec<wgpu::Buffer>, // [3 * total_channels] conv history per layer
    pub deltanet_state: Vec<wgpu::Buffer>, // [num_heads * key_dim * value_dim] recurrent state per layer
    pub deltanet_output: wgpu::Buffer,    // [num_value_heads * value_dim]
    pub deltanet_ab: wgpu::Buffer,        // merged in_proj_a @ hidden for alpha/beta
    pub deltanet_z: wgpu::Buffer,         // Z-gate output [num_value_heads * value_dim]
}

/// Number of attention splits for GQA (trade off parallelism vs overhead)
const NUM_ATTN_SPLITS: u32 = 1; // 1 = no multi-split, single-pass online softmax

/// Swappable per-sequence state: KV caches + DeltaNet recurrent state.
/// Allocate multiple slots to run different prompts without destroying context.
pub struct CacheSlot {
    pub k_cache: Vec<wgpu::Buffer>,
    pub v_cache: Vec<wgpu::Buffer>,
    pub deltanet_hist: Vec<wgpu::Buffer>,
    pub deltanet_state: Vec<wgpu::Buffer>,
    pub seq_len: u32,
}

pub struct Model {
    pub config: ModelConfig,
    pub quant_config: QuantConfig,
    pub weights: ModelWeights,
    pub state: InferenceState,
    #[cfg(feature = "jit-lora")]
    pub lora: Option<LoraState>,
    pub seq_len: u32,
    /// Token history for repetition penalty
    pub generated_tokens: Vec<u32>,
    /// Probability of last sampled token (for confidence tracking)
    pub last_token_prob: f32,
    /// When true, skip logits readback + sampling (training mode)
    #[cfg(feature = "jit-lora")]
    pub training_mode: bool,
    /// Simple RNG state for sampling
    rng_state: u64,
    use_4t: bool,
    tied_embeddings: bool,
    qknorm_shader_src: String,
    linear_num_key_heads: u32,
    linear_key_dim: u32,
    linear_value_dim: u32,
    linear_num_value_heads: u32,
}

/// Build the QK norm uniform buffer data for fused_split_qknorm_kvstore.
/// Layout: 8 u32 scalars + array<vec4<u32>, 320> of packed BF16 norm weights.
fn build_qknorm_params(
    config: &ModelConfig,
    q_norm_bytes: &[u8],
    k_norm_bytes: &[u8],
) -> Vec<u8> {
    // The uniform struct:
    //   num_heads: u32, num_kv_heads: u32, head_dim: u32, eps: f32,
    //   cache_position: u32, position: u32, position_h: u32, position_w: u32,
    //   qk_norm_weight: array<vec4<u32>, 320>  (5120 bytes = 320 * 16)
    // Total = 32 + 5120 = 5152 bytes

    let header_size = 32usize;
    let weight_size = 320 * 16; // array<vec4<u32>, 320>
    let total = header_size + weight_size;
    let mut buf = vec![0u8; total];

    // Write header (will be updated per-token for cache_position/position)
    let header: [u32; 8] = [
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rms_norm_eps.to_bits(),
        0, // cache_position (updated per token)
        0, // position (updated per token)
        0, // position_h
        0, // position_w
    ];
    buf[..header_size].copy_from_slice(bytemuck::cast_slice(&header));

    // Pack Q and K norm weights into the weight array
    // The weights come as raw bytes from safetensors (BF16 format already)
    // Copy Q norm weights first, then K norm weights
    let q_bytes = q_norm_bytes.len();
    let k_bytes = k_norm_bytes.len();
    let weight_start = header_size;

    if q_bytes + k_bytes <= weight_size {
        buf[weight_start..weight_start + q_bytes].copy_from_slice(q_norm_bytes);
        buf[weight_start + q_bytes..weight_start + q_bytes + k_bytes]
            .copy_from_slice(k_norm_bytes);
    } else {
        log::warn!(
            "QK norm weights too large: {} + {} > {} bytes",
            q_bytes,
            k_bytes,
            weight_size
        );
    }

    buf
}

impl Model {
    pub fn new(
        gpu: &GpuContext,
        config: ModelConfig,
        quant_config: QuantConfig,
        weights: ModelWeights,
        max_seq_len: u32,
    ) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let nl = config.num_hidden_layers;
        let f = 4u64;

        let use_4t =
            (h % 32 == 0) && (inter % 32 == 0) && (quant_config.group_size % 32 == 0);
        if use_4t {
            log::info!("Using 4-thread GPTQ variants");
        }

        // Build per-layer QK norm uniform buffers
        // We need to read the raw norm weight bytes from the GPU buffers,
        // but they were uploaded from safetensors. We'll build these during
        // weight loading instead. For now, create placeholder buffers.
        let qknorm_buf_size = 32 + 320 * 16; // 5152 bytes, aligned
        let qknorm_params: Vec<wgpu::Buffer> = (0..nl)
            .map(|i| {
                gpu.create_buffer(
                    &format!("qknorm_params_{i}"),
                    qknorm_buf_size as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                )
            })
            .collect();

        // Attention partials for multi-split GQA
        let partials_size = if NUM_ATTN_SPLITS > 1 {
            nh as u64 * NUM_ATTN_SPLITS as u64 * (hd as u64 + 2) * f
        } else {
            // When num_splits=1, output goes directly to attn_output
            4 // minimum size
        };

        let state = InferenceState {
            hidden: gpu.create_storage_buffer("hidden", h as u64 * f),
            residual: gpu.create_storage_buffer("residual", h as u64 * f),
            normed: gpu.create_storage_buffer("normed", h as u64 * f),
            q_out: gpu.create_storage_buffer("q_out", (nh * hd * 2) as u64 * f),
            q_proj: gpu.create_storage_buffer("q_proj", (nh * hd) as u64 * f),
            q_gate: gpu.create_storage_buffer("q_gate", (nh * hd) as u64 * f),
            k_out: gpu.create_storage_buffer("k_out", (nkv * hd) as u64 * f),
            v_out: gpu.create_storage_buffer("v_out", (nkv * hd) as u64 * f),
            attn_output: gpu.create_storage_buffer("attn_output", (nh * hd) as u64 * f),
            attn_partials: gpu.create_storage_buffer("attn_partials", partials_size),
            o_proj_out: gpu.create_storage_buffer("o_proj_out", h as u64 * f),
            gate_out: gpu.create_storage_buffer("gate_out", inter as u64 * f),
            up_out: gpu.create_storage_buffer("up_out", inter as u64 * f),
            mlp_output: gpu.create_storage_buffer("mlp_output", h as u64 * f),
            k_cache: (0..nl)
                .map(|i| {
                    gpu.create_storage_buffer(
                        &format!("k_cache_{i}"),
                        max_seq_len as u64 * nkv as u64 * hd as u64 * f,
                    )
                })
                .collect(),
            v_cache: (0..nl)
                .map(|i| {
                    gpu.create_storage_buffer(
                        &format!("v_cache_{i}"),
                        max_seq_len as u64 * nkv as u64 * hd as u64 * f,
                    )
                })
                .collect(),
            logits: gpu.create_storage_buffer("logits", config.vocab_size as u64 * f),
            argmax_result: gpu.create_storage_buffer("argmax_result", 8), // {idx: u32, val: f32}
            params: gpu.create_buffer(
                "params",
                256,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            qknorm_params,

            // DeltaNet buffers
            deltanet_qkv: {
                let lnkh = config.linear_num_key_heads;
                let lkd = config.linear_key_head_dim;
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;
                gpu.create_storage_buffer("deltanet_qkv", total_ch as u64 * f)
            },
            deltanet_hist: {
                let lnkh = config.linear_num_key_heads;
                let lkd = config.linear_key_head_dim;
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;
                let num_linear = nl as usize - weights.self_attn_layers.len();
                (0..num_linear)
                    .map(|i| gpu.create_storage_buffer(&format!("dn_hist_{i}"), 3 * total_ch as u64 * f))
                    .collect()
            },
            deltanet_state: {
                let lnkh = config.linear_num_key_heads;
                let lkd = config.linear_key_head_dim;
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let num_linear = nl as usize - weights.self_attn_layers.len();
                (0..num_linear)
                    .map(|i| gpu.create_storage_buffer(&format!("dn_state_{i}"), (lnkh * lkd * (lnvh / lnkh) * lvd) as u64 * f))
                    .collect()
            },
            deltanet_output: {
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                gpu.create_storage_buffer("deltanet_output", (lnvh * lvd) as u64 * f)
            },
            deltanet_ab: gpu.create_storage_buffer("deltanet_ab", h as u64 * f),
            deltanet_z: {
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                gpu.create_storage_buffer("deltanet_z", (lnvh * lvd) as u64 * f)
            },
        };

        let tied_embeddings = config.tie_word_embeddings;
        let qknorm_shader_src = build_qknorm_shader(&config);

        Self {
            linear_num_key_heads: config.linear_num_key_heads,
            linear_key_dim: config.linear_key_head_dim,
            linear_value_dim: config.linear_value_head_dim,
            linear_num_value_heads: config.linear_num_value_heads,
            config,
            quant_config,
            weights,
            state,
            seq_len: 0,
            generated_tokens: Vec::new(),
            last_token_prob: 0.0,
            #[cfg(feature = "jit-lora")]
            training_mode: false,
            rng_state: 0x5DEECE66Du64,
            #[cfg(feature = "jit-lora")]
            lora: None,
            use_4t,
            tied_embeddings,
            qknorm_shader_src,
        }
    }

    /// Initialize per-layer QK norm uniform buffers from raw weight data.
    /// Call after construction, providing the raw BF16 bytes from safetensors.
    pub fn init_qknorm_params(
        &self,
        gpu: &GpuContext,
        layer_idx: usize,
        q_norm_bytes: &[u8],
        k_norm_bytes: &[u8],
    ) {
        let data = build_qknorm_params(&self.config, q_norm_bytes, k_norm_bytes);
        gpu.write_buffer(&self.state.qknorm_params[layer_idx], 0, &data);
    }

    fn write_params(&self, gpu: &mut GpuContext, data: &[u8]) {
        // Must flush pending dispatches before overwriting params uniform,
        // otherwise previous dispatches would read the new params value.
        gpu.flush();
        gpu.write_buffer(&self.state.params, 0, data);
    }

    // ── Dispatch helpers ──────────────────────────────────────────────

    fn gptq_matvec(
        &self, gpu: &mut GpuContext, name: &str,
        input: &wgpu::Buffer, qweight: &wgpu::Buffer, scales: &wgpu::Buffer,
        output: &wgpu::Buffer, k: u32, n: u32,
    ) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { k: u32, n: u32, group_size: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&P {
            k, n, group_size: self.quant_config.group_size,
        }));
        if self.use_4t {
            gpu.dispatch(name, shaders::GPTQ_MATVEC_4T, &[
                gpu::bind(0, input), gpu::bind(1, qweight),
                gpu::bind(2, scales), gpu::bind(3, output),
                gpu::bind(4, &self.state.params),
            ], (n.div_ceil(8), 1, 1));
        } else {
            gpu.dispatch(name, shaders::GPTQ_MATVEC, &[
                gpu::bind(0, input), gpu::bind(1, qweight),
                gpu::bind(2, scales), gpu::bind(3, output),
                gpu::bind(4, &self.state.params),
            ], (n.div_ceil(32), 1, 1));
        }
    }

    fn fused_silu_gptq_down(
        &self, gpu: &mut GpuContext,
        gate_out: &wgpu::Buffer, up_out: &wgpu::Buffer,
        down_qw: &wgpu::Buffer, down_sc: &wgpu::Buffer,
        output: &wgpu::Buffer, k: u32, n: u32,
    ) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { k: u32, n: u32, group_size: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&P {
            k, n, group_size: self.quant_config.group_size,
        }));
        if self.use_4t {
            gpu.dispatch("fused_silu_gptq_4t", shaders::FUSED_SILU_GPTQ_4T, &[
                gpu::bind(0, gate_out), gpu::bind(1, up_out),
                gpu::bind(2, down_qw), gpu::bind(3, down_sc),
                gpu::bind(4, output), gpu::bind(5, &self.state.params),
            ], (n.div_ceil(8), 1, 1));
        } else {
            gpu.dispatch("fused_silu_gptq", shaders::FUSED_SILU_GPTQ, &[
                gpu::bind(0, gate_out), gpu::bind(1, up_out),
                gpu::bind(2, down_qw), gpu::bind(3, down_sc),
                gpu::bind(4, output), gpu::bind(5, &self.state.params),
            ], (n.div_ceil(32), 1, 1));
        }
    }

    fn add_rmsnorm(
        &self, gpu: &mut GpuContext,
        hidden: &wgpu::Buffer, addend: &wgpu::Buffer,
        weight: &wgpu::Buffer, output: &wgpu::Buffer, n: u32,
    ) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { n: u32, eps: f32 }
        self.write_params(gpu, bytemuck::bytes_of(&P { n, eps: self.config.rms_norm_eps }));
        gpu.dispatch("add_rmsnorm", shaders::ADD_RMSNORM, &[
            gpu::bind(0, hidden), gpu::bind(1, addend),
            gpu::bind(2, weight), gpu::bind(3, output),
            gpu::bind(4, &self.state.params),
        ], (1, 1, 1));
    }

    fn rmsnorm(
        &self, gpu: &mut GpuContext,
        input: &wgpu::Buffer, weight: &wgpu::Buffer,
        output: &wgpu::Buffer, n: u32,
    ) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { n: u32, eps: f32 }
        self.write_params(gpu, bytemuck::bytes_of(&P { n, eps: self.config.rms_norm_eps }));
        gpu.dispatch("rmsnorm", shaders::RMSNORM, &[
            gpu::bind(0, input), gpu::bind(1, weight),
            gpu::bind(2, output), gpu::bind(3, &self.state.params),
        ], (1, 1, 1));
    }

    fn embedding(&self, gpu: &mut GpuContext, token_id: u32) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { token_id: u32, dim: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&P {
            token_id, dim: self.config.hidden_size,
        }));
        gpu.dispatch("embedding", shaders::EMBEDDING, &[
            gpu::bind(0, &self.weights.embed_tokens),
            gpu::bind(1, &self.state.hidden),
            gpu::bind(2, &self.state.params),
        ], (self.config.hidden_size.div_ceil(256), 1, 1));
    }

    fn bf16_lm_head(&self, gpu: &mut GpuContext, weight: &wgpu::Buffer, h: u32) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct LmP { hidden_size: u32, vocab_size: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&LmP {
            hidden_size: h, vocab_size: self.config.vocab_size,
        }));
        gpu.dispatch("lm_head_bf16", shaders::BF16_MATVEC, &[
            gpu::bind(0, &self.state.normed),
            gpu::bind(1, weight),
            gpu::bind(2, &self.state.logits),
            gpu::bind(3, &self.state.params),
        ], (self.config.vocab_size.div_ceil(32), 1, 1));
    }

    fn argmax(&self, gpu: &mut GpuContext) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { n: u32, _pad: [u32; 3] }
        self.write_params(gpu, bytemuck::bytes_of(&P {
            n: self.config.vocab_size, _pad: [0; 3],
        }));
        gpu.dispatch("argmax", shaders::ARGMAX, &[
            gpu::bind(0, &self.state.logits),
            gpu::bind(1, &self.state.argmax_result),
            gpu::bind(2, &self.state.params),
        ], (1, 1, 1));
    }

    /// Fused Q/K norm + mRoPE + KV cache store.
    /// Dispatches (num_heads + num_kv_heads) workgroups.
    fn fused_split_qknorm_kvstore(
        &self, gpu: &mut GpuContext, layer_idx: usize,
    ) {
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;

        // Update per-token fields in the qknorm uniform (cache_position, position)
        let position = self.seq_len;
        let updates: [u32; 4] = [position, position, position, position];
        // Offset 16 = cache_position(u32) + position(u32) + position_h(u32) + position_w(u32)
        gpu.write_buffer(&self.state.qknorm_params[layer_idx], 16, bytemuck::cast_slice(&updates));

        gpu.dispatch(
            &format!("qknorm_l{layer_idx}"),
            &self.qknorm_shader_src,
            &[
                gpu::bind(0, &self.state.q_out),   // q_proj_full (interleaved q+gate)
                gpu::bind(1, &self.state.k_out),    // k_proj (read+write)
                gpu::bind(2, &self.state.v_out),    // v_proj (read)
                gpu::bind(3, &self.state.q_proj),   // q_proj output (normed+RoPE'd)
                gpu::bind(4, &self.state.q_gate),   // q_gate output
                gpu::bind(5, &self.state.k_cache[layer_idx]),
                gpu::bind(6, &self.state.v_cache[layer_idx]),
                gpu::bind(7, &self.state.qknorm_params[layer_idx]),
            ],
            (nh + nkv, 1, 1), // one workgroup per Q head + one per KV head
        );
    }

    /// GQA attention: online softmax over KV cache.
    fn gqa_attention(&self, gpu: &mut GpuContext, layer_idx: usize) {
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let heads_per_kv = nh / nkv;
        let seq_len = self.seq_len + 1; // include current token
        let ns = NUM_ATTN_SPLITS;

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P {
            seq_len: u32, head_dim: u32, num_kv_heads: u32, num_q_heads: u32,
            heads_per_kv: u32, num_splits: u32, _pad0: u32, _pad1: u32,
        }
        self.write_params(gpu, bytemuck::bytes_of(&P {
            seq_len, head_dim: hd, num_kv_heads: nkv, num_q_heads: nh,
            heads_per_kv, num_splits: ns, _pad0: 0, _pad1: 0,
        }));

        let output_buf = if ns == 1 {
            &self.state.attn_output
        } else {
            &self.state.attn_partials
        };

        gpu.dispatch(
            &format!("gqa_l{layer_idx}"),
            shaders::GQA_ATTENTION_HEAD,
            &[
                gpu::bind(0, &self.state.q_proj),
                gpu::bind(1, &self.state.k_cache[layer_idx]),
                gpu::bind(2, &self.state.v_cache[layer_idx]),
                gpu::bind(3, output_buf),
                gpu::bind(4, &self.state.params),
            ],
            (nh, ns, 1), // one workgroup per Q head per split
        );

        // Multi-split reduction
        if ns > 1 {
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct RP { head_dim: u32, num_splits: u32, num_heads: u32, _pad: u32 }
            self.write_params(gpu, bytemuck::bytes_of(&RP {
                head_dim: hd, num_splits: ns, num_heads: nh, _pad: 0,
            }));
            gpu.dispatch(
                &format!("gqa_reduce_l{layer_idx}"),
                shaders::GQA_REDUCE,
                &[
                    gpu::bind(0, &self.state.attn_partials),
                    gpu::bind(1, &self.state.attn_output),
                    gpu::bind(2, &self.state.params),
                ],
                (nh, 1, 1),
            );
        }
    }

    /// Gated attention: attn_output[i] *= sigmoid(q_gate[i])
    fn sigmoid_mul_gate(&self, gpu: &mut GpuContext) {
        let n = self.config.num_attention_heads * self.config.head_dim;
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { n: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&P { n }));

        // sigmoid_mul: output[i] = x[i] / (1 + exp(-gate[i]))
        // Can't alias read+write on same buffer in wgpu, so write to o_proj_out as temp
        gpu.dispatch("sigmoid_mul", shaders::SIGMOID_MUL, &[
            gpu::bind(0, &self.state.attn_output),
            gpu::bind(1, &self.state.q_gate),
            gpu::bind(2, &self.state.o_proj_out), // temp output
            gpu::bind(3, &self.state.params),
        ], (n.div_ceil(256), 1, 1));
        // Copy back to attn_output
        gpu.copy_buffer(&self.state.o_proj_out, &self.state.attn_output, n as u64 * 4);
    }

    // ── LoRA dispatch helpers ────────────────────────────────────────

    #[cfg(feature = "jit-lora")]
    /// Apply LoRA: out += (x @ A @ B) * scale. Two GPU dispatches.
    pub fn lora_apply(
        &self, gpu: &mut GpuContext, name: &str,
        input: &wgpu::Buffer,
        lora_a: &wgpu::Buffer, lora_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        in_features: u32, out_features: u32,
    ) {
        let lora = self.lora.as_ref().unwrap();
        let rank = lora.config.rank;
        let scale = lora.config.scale;

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct DownP { in_features: u32, rank: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&DownP { in_features, rank }));
        gpu.dispatch(
            &format!("lora_down_{name}"), shaders::LORA_DOWN,
            &[gpu::bind(0, input), gpu::bind(1, lora_a),
              gpu::bind(2, &lora.lora_hidden), gpu::bind(3, &self.state.params)],
            (rank.div_ceil(32), 1, 1),
        );

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct UpP { rank: u32, out_features: u32, scale: f32 }
        self.write_params(gpu, bytemuck::bytes_of(&UpP { rank, out_features, scale }));
        gpu.dispatch(
            &format!("lora_up_{name}"), shaders::LORA_UP_ADD,
            &[gpu::bind(0, &lora.lora_hidden), gpu::bind(1, lora_b),
              gpu::bind(2, output), gpu::bind(3, &self.state.params)],
            (out_features.div_ceil(32), 1, 1),
        );
    }

    #[cfg(feature = "jit-lora")]
    /// Apply LoRA to down_proj with fused SiLU
    pub fn lora_apply_down_proj(&self, gpu: &mut GpuContext, layer_idx: usize) {
        let lora = self.lora.as_ref().unwrap();
        let rank = lora.config.rank;
        let scale = lora.config.scale;
        let inter = self.config.intermediate_size;
        let h = self.config.hidden_size;

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct DownP { in_features: u32, rank: u32 }
        self.write_params(gpu, bytemuck::bytes_of(&DownP { in_features: inter, rank }));
        gpu.dispatch(
            &format!("lora_down_silu_l{layer_idx}"), shaders::LORA_DOWN_SILU,
            &[gpu::bind(0, &self.state.gate_out), gpu::bind(1, &self.state.up_out),
              gpu::bind(2, &lora.layers[layer_idx].down_proj_a),
              gpu::bind(3, &lora.lora_hidden), gpu::bind(4, &self.state.params)],
            (rank.div_ceil(32), 1, 1),
        );

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct UpP { rank: u32, out_features: u32, scale: f32 }
        self.write_params(gpu, bytemuck::bytes_of(&UpP { rank, out_features: h, scale }));
        gpu.dispatch(
            &format!("lora_up_down_l{layer_idx}"), shaders::LORA_UP_ADD,
            &[gpu::bind(0, &lora.lora_hidden),
              gpu::bind(1, &lora.layers[layer_idx].down_proj_b),
              gpu::bind(2, &self.state.mlp_output), gpu::bind(3, &self.state.params)],
            (h.div_ceil(32), 1, 1),
        );
    }

    // ── Forward pass ──────────────────────────────────────────────────

    pub fn forward(&mut self, gpu: &mut GpuContext, token_id: u32) -> u32 {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;

        // 1. Embedding (flush after to ensure params aren't overwritten)
        self.embedding(gpu, token_id);
        gpu.flush();
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);

        // 2. Layer loop
        for i in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[i];

            // ── Pre-attention norm ──
            if i == 0 {
                self.rmsnorm(gpu, &self.state.hidden, &layer.input_layernorm, &self.state.normed, h);
            } else {
                self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
                    &layer.input_layernorm, &self.state.normed, h);
            }

            // ── Attention (self-attn or DeltaNet) ──
            if let Some(sa) = layer.self_attn() {
                // Standard self-attention path
                let q_dim = nh * hd * 2;
                let kv_dim = nkv * hd;
                self.gptq_matvec(gpu, &format!("qproj_l{i}"),
                    &self.state.normed, &sa.q_proj_qweight, &sa.q_proj_scales,
                    &self.state.q_out, h, q_dim);
                #[cfg(feature = "jit-lora")]
                if self.lora.as_ref().map_or(false, |l| l.config.targets[0]) {
                    let lw = &self.lora.as_ref().unwrap().layers[i];
                    self.lora_apply(gpu, &format!("qproj_l{i}"),
                        &self.state.normed, &lw.q_proj_a, &lw.q_proj_b,
                        &self.state.q_out, h, q_dim);
                }

                self.gptq_matvec(gpu, &format!("kproj_l{i}"),
                    &self.state.normed, &sa.k_proj_qweight, &sa.k_proj_scales,
                    &self.state.k_out, h, kv_dim);
                self.gptq_matvec(gpu, &format!("vproj_l{i}"),
                    &self.state.normed, &sa.v_proj_qweight, &sa.v_proj_scales,
                    &self.state.v_out, h, kv_dim);
                #[cfg(feature = "jit-lora")]
                if self.lora.as_ref().map_or(false, |l| l.config.targets[1]) {
                    let lw = &self.lora.as_ref().unwrap().layers[i];
                    self.lora_apply(gpu, &format!("vproj_l{i}"),
                        &self.state.normed, &lw.v_proj_a, &lw.v_proj_b,
                        &self.state.v_out, h, kv_dim);
                }

                self.fused_split_qknorm_kvstore(gpu, i);
                self.gqa_attention(gpu, i);
                self.sigmoid_mul_gate(gpu);

                self.gptq_matvec(gpu, &format!("oproj_l{i}"),
                    &self.state.attn_output, &sa.o_proj_qweight, &sa.o_proj_scales,
                    &self.state.o_proj_out, nh * hd, h);
                #[cfg(feature = "jit-lora")]
                if self.lora.as_ref().map_or(false, |l| l.config.targets[2]) {
                    let lw = &self.lora.as_ref().unwrap().layers[i];
                    self.lora_apply(gpu, &format!("oproj_l{i}"),
                        &self.state.attn_output, &lw.o_proj_a, &lw.o_proj_b,
                        &self.state.o_proj_out, nh * hd, h);
                }
            } else if let Some(la) = layer.linear_attn() {
                // DeltaNet linear attention
                let lnkh = self.linear_num_key_heads;
                let lkd = self.linear_key_dim;
                let lnvh = self.linear_num_value_heads;
                let lvd = self.linear_value_dim;
                let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;

                // Count which linear-attn layer index this is
                let lin_idx = (0..i).filter(|j| !self.weights.self_attn_layers.contains(j)).count();

                // QKV projection
                self.gptq_matvec(gpu, &format!("dn_qkv_l{i}"),
                    &self.state.normed, &la.in_proj_qkv_qweight, &la.in_proj_qkv_scales,
                    &self.state.deltanet_qkv, h, total_ch);

                // Z-gate projection: normed -> z_gate [num_value_heads * value_dim]
                self.gptq_matvec(gpu, &format!("dn_z_l{i}"),
                    &self.state.normed, &la.in_proj_z_qweight, &la.in_proj_z_scales,
                    &self.state.deltanet_z, h, lnvh * lvd);

                #[repr(C)]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                struct DnP {
                    num_heads: u32, key_dim: u32, value_dim: u32, total_channels: u32,
                    eps: f32, hidden_size: u32, num_value_heads: u32,
                }
                self.write_params(gpu, bytemuck::bytes_of(&DnP {
                    num_heads: lnkh, key_dim: lkd, value_dim: lvd, total_channels: total_ch,
                    eps: self.config.rms_norm_eps, hidden_size: h, num_value_heads: lnvh,
                }));

                gpu.dispatch(
                    &format!("deltanet_l{i}"),
                    shaders::FUSED_CONV_DELTANET_NORM,
                    &[
                        gpu::bind(0, &self.state.deltanet_qkv),
                        gpu::bind(1, &self.state.deltanet_hist[lin_idx]),
                        gpu::bind(2, &la.conv1d_weight),
                        gpu::bind(3, &self.state.deltanet_state[lin_idx]),
                        gpu::bind(4, &self.state.deltanet_output),
                        gpu::bind(5, &self.state.normed), // hidden_input for alpha/beta
                        gpu::bind(6, &la.ab_weight),
                        gpu::bind(7, &la.a_log),
                        gpu::bind(8, &la.dt_bias),
                        gpu::bind(9, &la.norm_weight),
                        gpu::bind(10, &self.state.params),
                    ],
                    (lnkh, 1, 1),
                );

                // Fused SiLU(z_gate) * deltanet_output @ out_proj → o_proj_out
                self.fused_silu_gptq_down(gpu,
                    &self.state.deltanet_z, &self.state.deltanet_output,
                    &la.out_proj_qweight, &la.out_proj_scales,
                    &self.state.o_proj_out, lnvh * lvd, h);
            } else {
                // Fallback: pass through
                gpu.copy_buffer(&self.state.normed, &self.state.o_proj_out, h as u64 * 4);
            }

            // ── Post-attention norm ──
            self.add_rmsnorm(gpu, &self.state.residual, &self.state.o_proj_out,
                &layer.post_attn_layernorm, &self.state.normed, h);

            // ── MLP: separate gate+up, then fused SiLU+down ──
            self.gptq_matvec(gpu, &format!("gate_l{i}"),
                &self.state.normed, &layer.gate_proj_qweight, &layer.gate_proj_scales,
                &self.state.gate_out, h, inter);
            self.gptq_matvec(gpu, &format!("up_l{i}"),
                &self.state.normed, &layer.up_proj_qweight, &layer.up_proj_scales,
                &self.state.up_out, h, inter);
            self.fused_silu_gptq_down(gpu,
                &self.state.gate_out, &self.state.up_out,
                &layer.down_proj_qweight, &layer.down_proj_scales,
                &self.state.mlp_output, inter, h);
            #[cfg(feature = "jit-lora")]
            if self.lora.as_ref().map_or(false, |l| l.config.targets[3]) {
                self.lora_apply_down_proj(gpu, i);
            }
        }

        // 3. Final norm
        self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
            &self.weights.final_norm, &self.state.normed, h);

        // 4. LM head
        if self.tied_embeddings {
            // Tied embeddings: logits = embed_tokens (BF16) @ normed
            self.bf16_lm_head(gpu, &self.weights.embed_tokens, h);
        } else if self.weights.lm_head_is_bf16 {
            // Separate unquantized BF16 lm_head weight
            self.bf16_lm_head(gpu, &self.weights.lm_head_qweight, h);
        } else {
            self.gptq_matvec(gpu, "lm_head",
                &self.state.normed, &self.weights.lm_head_qweight, &self.weights.lm_head_scales,
                &self.state.logits, h, self.config.vocab_size);
        }

        // 5. Read logits, apply frequency penalty + temperature, sample
        self.seq_len += 1;

        // Training mode: skip the expensive logits readback + sampling
        #[cfg(feature = "jit-lora")]
        if self.training_mode {
            return 0;
        }

        let logits_bytes = gpu.read_buffer(&self.state.logits, self.config.vocab_size as u64 * 4);
        let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let mut logits_vec = logits.to_vec();

        // Repetition penalty (multiplicative, matching browser version)
        let rep_penalty = 1.0f32;
        let presence_penalty = 1.5f32;
        {
            let mut seen = std::collections::HashSet::<u32>::new();
            for &tok in &self.generated_tokens {
                seen.insert(tok);
            }
            for &tok in &seen {
                let idx = tok as usize;
                if idx < logits_vec.len() {
                    // Multiplicative repetition penalty
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= rep_penalty.max(1.001);
                    } else {
                        logits_vec[idx] *= rep_penalty.max(1.001);
                    }
                    // Additive presence penalty
                    logits_vec[idx] -= presence_penalty;
                }
            }
        }

        // Hard ban: if last 2 tokens are the same, ban that token
        let n = self.generated_tokens.len();
        if n >= 2 && self.generated_tokens[n-1] == self.generated_tokens[n-2] {
            let banned = self.generated_tokens[n-1] as usize;
            if banned < logits_vec.len() {
                logits_vec[banned] = f32::NEG_INFINITY;
            }
        }
        // Hard ban: if last 2-token pair repeats (catches "What\nWhat\n" patterns)
        if n >= 4
            && self.generated_tokens[n-1] == self.generated_tokens[n-3]
            && self.generated_tokens[n-2] == self.generated_tokens[n-4]
        {
            // Ban both tokens in the pair
            for &banned_tok in &[self.generated_tokens[n-1], self.generated_tokens[n-2]] {
                let idx = banned_tok as usize;
                if idx < logits_vec.len() {
                    logits_vec[idx] = f32::NEG_INFINITY;
                }
            }
        }
        // Hard ban: if last 3-token pattern repeats (catches "X Y Z X Y Z" loops)
        if n >= 6
            && self.generated_tokens[n-1] == self.generated_tokens[n-4]
            && self.generated_tokens[n-2] == self.generated_tokens[n-5]
            && self.generated_tokens[n-3] == self.generated_tokens[n-6]
        {
            let idx = self.generated_tokens[n-1] as usize;
            if idx < logits_vec.len() {
                logits_vec[idx] = f32::NEG_INFINITY;
            }
        }

        // Temperature (0.7)
        let temperature = 0.7f32;
        let vocab_size = logits_vec.len();
        for v in logits_vec.iter_mut() {
            *v /= temperature;
        }

        // Top-k (k=20)
        let k = 20usize.min(vocab_size);
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        indices.select_nth_unstable_by(k, |&a, &b| logits_vec[b].partial_cmp(&logits_vec[a]).unwrap());
        let top_k_threshold = logits_vec[indices[k - 1]];

        // Top-p (p=0.80) — nucleus sampling within top-k
        // First softmax the top-k candidates
        let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = logits_vec.iter().enumerate()
            .filter(|(_, &v)| v >= top_k_threshold)
            .map(|(i, &v)| (i, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }

        // Sort by probability descending for top-p truncation
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_p = 0.80f32;
        let mut cumsum = 0.0f32;
        let mut nucleus: Vec<(usize, f32)> = Vec::new();
        for (i, p) in &probs {
            cumsum += p;
            nucleus.push((*i, *p));
            if cumsum >= top_p {
                break;
            }
        }

        // Renormalize nucleus
        let nuc_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
        for (_, p) in nucleus.iter_mut() {
            *p /= nuc_sum;
        }

        // Sample from nucleus
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let r = (self.rng_state & 0xFFFFFFFF) as f64 / u32::MAX as f64;
        let mut cumulative = 0.0f64;
        let mut sampled = nucleus[0].0 as u32;
        let mut sampled_prob = nucleus[0].1;
        for &(idx, p) in &nucleus {
            cumulative += p as f64;
            if cumulative >= r {
                sampled = idx as u32;
                sampled_prob = p;
                break;
            }
        }

        self.last_token_prob = sampled_prob;
        self.generated_tokens.push(sampled);
        sampled
    }

    /// Allocate a second cache slot on GPU for running a separate prompt
    /// (e.g., Q/A extraction) without destroying the main chat context.
    pub fn alloc_cache_slot(&self, gpu: &GpuContext, max_seq_len: u32) -> CacheSlot {
        let f = 4u64;
        let nl = self.config.num_hidden_layers;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;

        let k_cache: Vec<_> = (0..nl)
            .map(|i| gpu.create_storage_buffer(
                &format!("k_cache2_{i}"),
                max_seq_len as u64 * nkv as u64 * hd as u64 * f,
            ))
            .collect();
        let v_cache: Vec<_> = (0..nl)
            .map(|i| gpu.create_storage_buffer(
                &format!("v_cache2_{i}"),
                max_seq_len as u64 * nkv as u64 * hd as u64 * f,
            ))
            .collect();

        let lnkh = self.linear_num_key_heads;
        let lkd = self.linear_key_dim;
        let lnvh = self.linear_num_value_heads;
        let lvd = self.linear_value_dim;
        let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;
        let num_linear = nl as usize - self.weights.self_attn_layers.len();

        let deltanet_hist: Vec<_> = (0..num_linear)
            .map(|i| gpu.create_storage_buffer(
                &format!("dn_hist2_{i}"),
                3 * total_ch as u64 * f,
            ))
            .collect();
        let deltanet_state: Vec<_> = (0..num_linear)
            .map(|i| gpu.create_storage_buffer(
                &format!("dn_state2_{i}"),
                (lnkh * lkd * (lnvh / lnkh) * lvd) as u64 * f,
            ))
            .collect();

        CacheSlot {
            k_cache,
            v_cache,
            deltanet_hist,
            deltanet_state,
            seq_len: 0,
        }
    }

    /// Swap the active KV cache / DeltaNet state with a secondary slot.
    /// This is a pointer swap — no GPU copies, instant.
    pub fn swap_cache(&mut self, slot: &mut CacheSlot) {
        std::mem::swap(&mut self.state.k_cache, &mut slot.k_cache);
        std::mem::swap(&mut self.state.v_cache, &mut slot.v_cache);
        std::mem::swap(&mut self.state.deltanet_hist, &mut slot.deltanet_hist);
        std::mem::swap(&mut self.state.deltanet_state, &mut slot.deltanet_state);
        std::mem::swap(&mut self.seq_len, &mut slot.seq_len);
    }
}
