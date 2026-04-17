use crate::gpu::{self, GpuContext};
#[cfg(feature = "jit-lora")]
use crate::lora::LoraState;
use crate::weights::{ModelConfig, ModelWeights, QuantConfig};

pub(crate) mod shaders {
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
    pub const BF16_MATVEC_TILED: &str = include_str!("shaders/bf16_matvec_tiled.wgsl");
    pub const SILU_MUL: &str = include_str!("shaders/silu_mul.wgsl");
    pub const FUSED_CONV_DELTANET_NORM: &str = include_str!("shaders/fused_conv_deltanet_norm.wgsl");

    // Batched prefill shaders
    pub const BF16_GEMM: &str = include_str!("shaders/bf16_gemm.wgsl");
    pub const RMSNORM_DIRECT: &str = include_str!("shaders/rmsnorm_direct.wgsl");
    pub const ADD_RMSNORM_DIRECT: &str = include_str!("shaders/add_rmsnorm_direct.wgsl");
    pub const BATCHED_RMSNORM: &str = include_str!("shaders/batched_rmsnorm.wgsl");
    pub const BATCHED_ADD_RMSNORM: &str = include_str!("shaders/batched_add_rmsnorm.wgsl");
    pub const BATCHED_RMSNORM_1PW: &str = include_str!("shaders/batched_rmsnorm_1pw.wgsl");
    pub const BATCHED_ADD_RMSNORM_1PW: &str = include_str!("shaders/batched_add_rmsnorm_1pw.wgsl");
    pub const BATCHED_SILU_MUL: &str = include_str!("shaders/batched_silu_mul.wgsl");
    pub const CAUSAL_ATTENTION_PREFILL: &str = include_str!("shaders/causal_attention_prefill.wgsl");
    pub const INT4_MATVEC_MLX: &str = include_str!("shaders/int4_matvec_mlx.wgsl");
    pub const INT4_MATVEC_MLX_BF16: &str = include_str!("shaders/int4_matvec_mlx_bf16.wgsl");
    pub const FUSED_SILU_INT4_MLX: &str = include_str!("shaders/fused_silu_int4_mlx.wgsl");
    pub const INT4_EMBEDDING_MLX: &str = include_str!("shaders/int4_embedding_mlx.wgsl");

    // GPTQ GEMM shaders (batched prefill)
    pub const GPTQ_GEMM: &str = include_str!("shaders/gptq_gemm.wgsl");
    pub const GPTQ_GEMM_4T: &str = include_str!("shaders/gptq_gemm_4t.wgsl");
    pub const FUSED_SILU_GPTQ_GEMM: &str = include_str!("shaders/fused_silu_gptq_gemm.wgsl");
    pub const FUSED_SILU_GPTQ_GEMM_4T: &str = include_str!("shaders/fused_silu_gptq_gemm_4t.wgsl");
    pub const BATCHED_QKNORM_ROPE_GATED: &str = include_str!("shaders/batched_qknorm_rope_gated.wgsl");
    pub const BATCHED_DELTANET_PREFILL:  &str = include_str!("shaders/batched_deltanet_prefill.wgsl");

    // Phase 2: decode op fusion shaders
    pub const FUSED_GATE_UP_GPTQ: &str = include_str!("shaders/fused_gate_up_gptq.wgsl");
    pub const FUSED_GATE_UP_GPTQ_4T: &str = include_str!("shaders/fused_gate_up_gptq_4t.wgsl");

    // Sampling shader (combined penalty + gate + top-K in one pass)
    pub const SAMPLE_TOPK: &str = include_str!("shaders/sample_topk.wgsl");

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
    build_qknorm_shader_gated(config, true)
}

fn build_qknorm_shader_gated(config: &ModelConfig, q_gated: bool) -> String {
    build_qknorm_shader_full(config, q_gated, 1.0) // Qwen3.5: (1 + w) norm scaling
}

fn build_qknorm_shader_full(config: &ModelConfig, q_gated: bool, norm_offset: f32) -> String {
    let partial_dim = (config.head_dim as f32 * config.partial_rotary_factor) as u32;
    let interleaved = config.mrope_interleaved();
    let s_limit = partial_dim / 2;
    format!(
        "const ROPE_THETA: f32 = {:.1};\n\
         const MROPE_S1_LIMIT: u32 = {}u;\n\
         const MROPE_S2_LIMIT: u32 = {}u;\n\
         const PARTIAL_DIM: u32 = {}u;\n\
         const MROPE_INTERLEAVED: bool = {};\n\
         const Q_GATED: bool = {};\n\
         const NORM_OFFSET: f32 = {:.1};\n\n{}",
        config.rope_theta,
        s_limit,
        s_limit,
        partial_dim,
        interleaved,
        q_gated,
        norm_offset,
        include_str!("shaders/fused_split_qknorm_kvstore.wgsl")
            .lines()
            .skip(7) // skip the 7 hardcoded const lines
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Build GQA attention shader with Q_GATED sigmoid fusion injected as a const.
/// When Q_GATED=true, binding 5 (q_gate) is read and sigmoid gate is applied to output.
fn build_gqa_shader(q_gated: bool) -> String {
    format!(
        "const Q_GATED: bool = {};\n\n{}",
        q_gated,
        // Skip the comment lines at top that reference the injected const (lines 1-14)
        include_str!("shaders/gqa_attention_head.wgsl")
            .lines()
            .skip(14)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Build the batched Q_GATED qknorm+RoPE+KV-store shader for GPTQ prefill.
/// Injects ROPE_THETA, MROPE limits, PARTIAL_DIM, MROPE_INTERLEAVED, NORM_OFFSET.
fn build_batched_qknorm_shader_gated(config: &ModelConfig) -> String {
    let partial_dim = (config.head_dim as f32 * config.partial_rotary_factor) as u32;
    let interleaved = config.mrope_interleaved();
    let s_limit = partial_dim / 2;
    format!(
        "const ROPE_THETA: f32 = {:.1};\n\
         const MROPE_S1_LIMIT: u32 = {}u;\n\
         const MROPE_S2_LIMIT: u32 = {}u;\n\
         const PARTIAL_DIM: u32 = {}u;\n\
         const MROPE_INTERLEAVED: bool = {};\n\
         const NORM_OFFSET: f32 = {:.1};\n\n{}",
        config.rope_theta,
        s_limit,
        s_limit,
        partial_dim,
        interleaved,
        1.0f32, // Qwen3.5 always uses (1 + w) norm scaling
        // Skip the 17-line header comment + injected-constants comment block
        include_str!("shaders/batched_qknorm_rope_gated.wgsl")
            .lines()
            .skip(17)
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
    // Named static param buffers — pre-initialized at model load, no per-dispatch writes.
    // GPTQ/MLX matvec params {k, n, group_size}:
    pub p_gptq_q:     wgpu::Buffer,  // k=hidden, n=q_dim
    pub p_gptq_kv:    wgpu::Buffer,  // k=hidden, n=kv_dim
    pub p_gptq_o:     wgpu::Buffer,  // k=nh*hd, n=hidden
    pub p_gptq_gu:    wgpu::Buffer,  // k=hidden, n=inter (gate and up share same k/n)
    pub p_gptq_down:  wgpu::Buffer,  // k=inter, n=hidden
    pub p_gptq_lm:    wgpu::Buffer,  // k=hidden, n=vocab
    // BF16 matvec params {hidden_size, vocab_size}:
    pub p_bf16_q:     wgpu::Buffer,  // hidden=h, vocab=q_dim
    pub p_bf16_kv:    wgpu::Buffer,  // hidden=h, vocab=kv_dim
    pub p_bf16_o:     wgpu::Buffer,  // hidden=nh*hd, vocab=h
    pub p_bf16_gu:    wgpu::Buffer,  // hidden=h, vocab=inter
    pub p_bf16_lm:    wgpu::Buffer,  // hidden=h, vocab=vocab_size
    pub p_bf16_down:  wgpu::Buffer,  // hidden=inter, vocab=h (for down_proj in bf16 mode)
    pub p_bf16_silu:  wgpu::Buffer,  // {n: inter} for SILU_MUL in bf16 fused path
    // Other static params:
    pub p_norm:       wgpu::Buffer,  // {n: hidden, eps} for rmsnorm/add_rmsnorm
    pub p_argmax:     wgpu::Buffer,  // {n: vocab, 0, 0, 0}
    pub p_gqa_reduce: wgpu::Buffer,  // {head_dim, num_splits, num_heads, 0}
    pub p_dn:         wgpu::Buffer,  // deltanet params
    // Static DeltaNet projection params (replaces per-dispatch p_scratch writes on the hot path):
    pub p_dn_qkv:     wgpu::Buffer,  // {h, total_ch, gs, 0} for dn_qkv matvec
    pub p_dn_z:       wgpu::Buffer,  // {h, dn_z_n, gs, 0}   for dn_z matvec
    pub p_dn_down:    wgpu::Buffer,  // {dn_z_n, h, gs, 0}   for fused_silu_down
    pub p_scratch:    wgpu::Buffer,  // 64-byte scratch for rare per-call writes (chunked lm_head)
    // Per-token params (2 fields written at start of forward()):
    pub p_embed:      wgpu::Buffer,  // {token_id: u32, dim: u32}  <- token_id written per token
    pub p_attn:       wgpu::Buffer,  // {seq_len, head_dim, num_kv_heads, num_q_heads, heads_per_kv, num_splits, 0, 0}  <- seq_len written per token
    #[cfg(feature = "jit-lora")]
    pub p_lora_down:  wgpu::Buffer,  // {in_features, rank}
    #[cfg(feature = "jit-lora")]
    pub p_lora_up:    wgpu::Buffer,  // {rank, out_features, scale: f32}
    pub qknorm_params: Vec<wgpu::Buffer>,
    /// Per-layer qknorm params in the batched-prefill format (seq_len at offset 16).
    /// Written alongside qknorm_params in init_qknorm_params().
    pub batched_qknorm_params: Vec<wgpu::Buffer>,
    /// Bitmap of seen tokens for GPU repetition penalty (vocab_size/32 u32s)
    pub seen_bitmap: wgpu::Buffer,
    /// Top-K output from GPU sampler: array of {idx: u32, val: f32}, length TOPK_K
    pub topk_out: wgpu::Buffer,
    /// Uniform buffer for combined penalty+gate+topk shader (64 bytes)
    pub penalty_uniform: wgpu::Buffer,
    /// First byte of each token, indexed by token ID (0 = empty/special).
    /// Used by the gate: when gate_byte != 0, tokens whose first_bytes[i] != gate_byte are masked to -inf.
    /// Always allocated (vocab_size u32s); zeros = no gate effect when gate_byte = 0.
    pub first_bytes_buf: wgpu::Buffer,
    /// Per-token schema mask (packed bitfield, vocab_size/32 u32s).
    /// Bit N of word[N/32] set = token N allowed. All-ones = unconstrained.
    /// Uploaded each sampling step when JSON schema mode is active.
    pub token_mask_buf: wgpu::Buffer,
    /// Per-chunk temporary buffers for lm_head (one per embed_chunk, reused across decode steps).
    /// Avoids allocating new GPU buffers on every decode step, which fills the bind group cache.
    pub lm_chunk_tmps: Vec<wgpu::Buffer>,
    /// Per-chunk param buffers for lm_head (pre-initialized with {h, chunk_vocab, 0, 0}).
    /// Eliminates the flush()+write_buffer per chunk in bf16_lm_head, allowing all chunks
    /// to run in a single command buffer submission.
    pub lm_chunk_params: Vec<wgpu::Buffer>,

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
/// Number of top-K candidates extracted on GPU for sampling
const TOPK_K: u32 = 8; // must match const K in sample_topk.wgsl

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
    /// When true, all weights are bf16 (not GPTQ). Dispatch bf16_matvec instead of gptq_matvec.
    pub bf16_mode: bool,
    /// When true, Q projection outputs [nh * hd * 2] (Qwen3.5 SiGLU gated attention).
    /// When false, Q outputs [nh * hd] (standard attention, e.g. Qwen3-ASR decoder).
    pub q_gated: bool,
    /// When true, RMSNorm uses direct `w` scaling (ASR decoder).
    /// When false, uses `(1 + w)` scaling (Qwen3.5).
    pub norm_direct: bool,
    /// When true, weights are MLX INT4 (asymmetric minmax, row-major).
    /// Uses int4_matvec_mlx shader with separate biases buffer.
    pub mlx_int4_mode: bool,
    /// MLX scales/biases are BF16 (not F16). Use INT4_MATVEC_MLX_BF16 shader.
    pub mlx_bf16_scales: bool,
    /// Current layer index during forward pass (for accessing mlx_biases)
    mlx_current_layer: usize,
    /// Which projection within a layer: 0=q,1=k,2=v,3=o,4=gate,5=up,6=down
    mlx_current_proj: usize,
    tied_embeddings: bool,
    /// When true, skip lm_head + logit readback. Set during prefill to avoid
    /// 607KB GPU→CPU sync per token; only the final decode step needs the readback.
    prefill_kv_only: bool,
    /// CPU-side copy of seen_bitmap for incremental updates (one word written per token)
    pub seen_bitmap_cpu: Vec<u32>,
    qknorm_shader_src: String,
    /// GQA attention shader source with Q_GATED sigmoid fusion baked in.
    gqa_shader_src: String,
    /// Batched Q_GATED qknorm+RoPE shader for GPTQ prefill.
    batched_qknorm_gated_src: String,
    linear_num_key_heads: u32,
    linear_key_dim: u32,
    linear_value_dim: u32,
    linear_num_value_heads: u32,
    /// Optional JSON-constrained sampler. When Some, gates token candidates at structural positions.
    pub json_sampler: Option<crate::json_sampler::JsonSampler>,
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

/// Build batched qknorm params for the batched_qknorm_rope_gated shader.
/// Same layout as build_qknorm_params but with seq_len at header offset 16
/// instead of cache_position. seq_len is written as 0; caller writes the real
/// value (4 bytes at offset 16) before each prefill dispatch.
fn build_qknorm_params_batched(
    config: &ModelConfig,
    q_norm_bytes: &[u8],
    k_norm_bytes: &[u8],
) -> Vec<u8> {
    let header_size = 32usize;
    let weight_size = 320 * 16;
    let total = header_size + weight_size;
    let mut buf = vec![0u8; total];

    // Header: [num_heads, num_kv_heads, head_dim, eps_bits, seq_len=0, 0, 0, 0]
    let header: [u32; 8] = [
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rms_norm_eps.to_bits(),
        0, // seq_len (updated per prefill call at offset 16)
        0, 0, 0,
    ];
    buf[..header_size].copy_from_slice(bytemuck::cast_slice(&header));

    let q_bytes = q_norm_bytes.len();
    let k_bytes = k_norm_bytes.len();
    if q_bytes + k_bytes <= weight_size {
        buf[header_size..header_size + q_bytes].copy_from_slice(q_norm_bytes);
        buf[header_size + q_bytes..header_size + q_bytes + k_bytes]
            .copy_from_slice(k_norm_bytes);
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
        let batched_qknorm_params: Vec<wgpu::Buffer> = (0..nl)
            .map(|i| {
                gpu.create_buffer(
                    &format!("batched_qknorm_params_{i}"),
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

        let mut state = InferenceState {
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
            o_proj_out: gpu.create_storage_buffer("o_proj_out", h.max(nh * hd) as u64 * f),
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
            p_gptq_q: {
                let gs = quant_config.group_size;
                let q_dim = nh * hd * 2; // safe default; updated in rebuild_qknorm_shader if needed
                let buf = gpu.create_buffer("p_gptq_q", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, q_dim, gs, 0u32]));
                buf
            },
            p_gptq_kv: {
                let gs = quant_config.group_size;
                let kv_dim = nkv * hd;
                let buf = gpu.create_buffer("p_gptq_kv", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, kv_dim, gs, 0u32]));
                buf
            },
            p_gptq_o: {
                let gs = quant_config.group_size;
                let buf = gpu.create_buffer("p_gptq_o", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[nh * hd, h, gs, 0u32]));
                buf
            },
            p_gptq_gu: {
                let gs = quant_config.group_size;
                let buf = gpu.create_buffer("p_gptq_gu", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, inter, gs, 0u32]));
                buf
            },
            p_gptq_down: {
                let gs = quant_config.group_size;
                let buf = gpu.create_buffer("p_gptq_down", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[inter, h, gs, 0u32]));
                buf
            },
            p_gptq_lm: {
                let gs = quant_config.group_size;
                let vocab = config.vocab_size;
                let buf = gpu.create_buffer("p_gptq_lm", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, vocab, gs, 0u32]));
                buf
            },
            p_bf16_q: {
                let q_dim = nh * hd * 2;
                let buf = gpu.create_buffer("p_bf16_q", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, q_dim, 0u32, 0u32]));
                buf
            },
            p_bf16_kv: {
                let kv_dim = nkv * hd;
                let buf = gpu.create_buffer("p_bf16_kv", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, kv_dim, 0u32, 0u32]));
                buf
            },
            p_bf16_o: {
                let buf = gpu.create_buffer("p_bf16_o", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[nh * hd, h, 0u32, 0u32]));
                buf
            },
            p_bf16_gu: {
                let buf = gpu.create_buffer("p_bf16_gu", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, inter, 0u32, 0u32]));
                buf
            },
            p_bf16_lm: {
                let vocab = config.vocab_size;
                let buf = gpu.create_buffer("p_bf16_lm", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, vocab, 0u32, 0u32]));
                buf
            },
            p_bf16_down: {
                let buf = gpu.create_buffer("p_bf16_down", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[inter, h, 0u32, 0u32]));
                buf
            },
            p_bf16_silu: {
                let buf = gpu.create_buffer("p_bf16_silu", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[inter, 0u32, 0u32, 0u32]));
                buf
            },
            p_norm: {
                let eps_bits = config.rms_norm_eps.to_bits();
                let buf = gpu.create_buffer("p_norm", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, eps_bits, 0u32, 0u32]));
                buf
            },
            p_argmax: {
                let vocab = config.vocab_size;
                let buf = gpu.create_buffer("p_argmax", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[vocab, 0u32, 0u32, 0u32]));
                buf
            },
            p_gqa_reduce: {
                let buf = gpu.create_buffer("p_gqa_reduce", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[hd, NUM_ATTN_SPLITS, nh, 0u32]));
                buf
            },
            p_dn: {
                let lnkh = config.linear_num_key_heads;
                let lkd = config.linear_key_head_dim;
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;
                let eps_bits = config.rms_norm_eps.to_bits();
                let buf = gpu.create_buffer("p_dn", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[lnkh, lkd, lvd, total_ch, eps_bits, h, lnvh, 0u32]));
                buf
            },
            p_dn_qkv: {
                let lnkh = config.linear_num_key_heads;
                let lkd = config.linear_key_head_dim;
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;
                let buf = gpu.create_buffer("p_dn_qkv", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, total_ch, quant_config.group_size, 0u32]));
                buf
            },
            p_dn_z: {
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let dn_z_n = lnvh * lvd;
                let buf = gpu.create_buffer("p_dn_z", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, dn_z_n, quant_config.group_size, 0u32]));
                buf
            },
            p_dn_down: {
                let lnvh = config.linear_num_value_heads;
                let lvd = config.linear_value_head_dim;
                let dn_z_n = lnvh * lvd;
                let buf = gpu.create_buffer("p_dn_down", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[dn_z_n, h, quant_config.group_size, 0u32]));
                buf
            },
            p_scratch: gpu.create_buffer("p_scratch", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            p_embed: {
                let buf = gpu.create_buffer("p_embed", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[0u32, h]));
                buf
            },
            p_attn: {
                let kv_dim_for_attn = nkv * hd;
                let _ = kv_dim_for_attn; // suppress unused warning
                let buf = gpu.create_buffer("p_attn", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
                let heads_per_kv = if nkv > 0 { nh / nkv } else { 1 };
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[0u32, hd, nkv, nh, heads_per_kv, NUM_ATTN_SPLITS, 0u32, 0u32]));
                buf
            },
            #[cfg(feature = "jit-lora")]
            p_lora_down: gpu.create_buffer("p_lora_down", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            #[cfg(feature = "jit-lora")]
            p_lora_up: gpu.create_buffer("p_lora_up", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            qknorm_params,
            batched_qknorm_params,
            seen_bitmap: {
                let words = config.vocab_size.div_ceil(32);
                gpu.create_storage_buffer("seen_bitmap", words as u64 * 4)
            },
            topk_out: gpu.create_storage_buffer("topk_out", TOPK_K as u64 * 8),
            first_bytes_buf: gpu.create_storage_buffer("first_bytes", config.vocab_size as u64 * 4),
            token_mask_buf: gpu.create_storage_buffer("token_mask", config.vocab_size.div_ceil(32) as u64 * 4),
            penalty_uniform: gpu.create_buffer(
                "penalty_uniform",
                64, // PenaltyParams struct: 4*4 + 4*4 + 4 = 48 bytes, pad to 64
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            lm_chunk_tmps: Vec::new(),   // populated below if embed_chunks is used
            lm_chunk_params: Vec::new(), // populated below if embed_chunks is used

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
        let gqa_shader_src = build_gqa_shader(true); // q_gated=true by default (Qwen3.5)
        let batched_qknorm_gated_src = build_batched_qknorm_shader_gated(&config);
        let vocab_size_for_bitmap = config.vocab_size;

        // Pre-allocate persistent per-chunk buffers for lm_head.
        // - lm_chunk_tmps: one output buffer per chunk (avoids GPU alloc per decode step)
        // - lm_chunk_params: pre-initialized param buffers (avoids flush+write per chunk,
        //   letting all chunks run in a single command buffer submission)
        if !weights.embed_chunks.is_empty() {
            let h = config.hidden_size;
            let chunk_size = weights.embed_chunk_size;
            let vocab = config.vocab_size;
            state.lm_chunk_tmps = (0..weights.embed_chunks.len()).map(|i| {
                gpu.create_storage_buffer(&format!("lm_chunk_tmp_{i}"), chunk_size as u64 * 4)
            }).collect();
            state.lm_chunk_params = (0..weights.embed_chunks.len()).map(|i| {
                let chunk_start = i as u32 * chunk_size;
                let chunk_vocab = chunk_size.min(vocab - chunk_start);
                let buf = gpu.create_buffer(
                    &format!("lm_chunk_params_{i}"), 16,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                );
                gpu.write_buffer(&buf, 0, bytemuck::cast_slice(&[h, chunk_vocab, 0u32, 0u32]));
                buf
            }).collect();
        }

        let bf16_scales = weights.bf16_scales;
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
            bf16_mode: false,
            q_gated: true, // default: Qwen3.5 gated attention
            norm_direct: false,
            mlx_int4_mode: false,
            mlx_bf16_scales: bf16_scales,
            mlx_current_layer: 0,
            mlx_current_proj: 0,
            tied_embeddings,
            prefill_kv_only: false,
            seen_bitmap_cpu: vec![0u32; vocab_size_for_bitmap.div_ceil(32) as usize],
            qknorm_shader_src,
            gqa_shader_src,
            batched_qknorm_gated_src,
            json_sampler: None,
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
        // Also write batched_qknorm_params with the batched-format header
        // (seq_len at offset 16 instead of cache_position; will be overwritten per call)
        let batched_data = build_qknorm_params_batched(&self.config, q_norm_bytes, k_norm_bytes);
        gpu.write_buffer(&self.state.batched_qknorm_params[layer_idx], 0, &batched_data);
    }

    /// Rebuild the QK norm and GQA shaders with the current q_gated setting.
    /// Must be called after changing q_gated.
    pub fn rebuild_qknorm_shader(&mut self) {
        let norm_offset = if self.norm_direct { 0.0 } else { 1.0 };
        self.qknorm_shader_src = build_qknorm_shader_full(&self.config, self.q_gated, norm_offset);
        self.gqa_shader_src = build_gqa_shader(self.q_gated);
        self.batched_qknorm_gated_src = build_batched_qknorm_shader_gated(&self.config);
    }

    /// Re-write static param buffers whose values depend on runtime settings (q_gated).
    /// Call after changing q_gated (e.g. in load_bf16_model / load_mlx_model).
    pub fn rebuild_static_params(&self, gpu: &GpuContext) {
        let h  = self.config.hidden_size;
        let nh = self.config.num_attention_heads;
        let hd = self.config.head_dim;
        let gs = self.quant_config.group_size;
        let q_dim: u32 = if self.q_gated { nh * hd * 2 } else { nh * hd };
        // Re-write the Q-proj param buffers that depend on q_dim.
        gpu.write_buffer(&self.state.p_gptq_q, 0,
            bytemuck::cast_slice::<u32, u8>(&[h, q_dim, gs, 0]));
        gpu.write_buffer(&self.state.p_bf16_q, 0,
            bytemuck::cast_slice::<u32, u8>(&[h, q_dim, 0, 0]));
    }

    // ── Dispatch helpers ──────────────────────────────────────────────

    /// Dispatch matvec with mode-aware shader selection.
    /// For MLX INT4: pass biases from mlx_biases[layer][proj].
    /// proj: 0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down
    pub fn matvec_layer(
        &self, gpu: &mut GpuContext, name: &str,
        input: &wgpu::Buffer, qweight: &wgpu::Buffer, scales: &wgpu::Buffer,
        output: &wgpu::Buffer, k: u32, n: u32,
        layer: usize, proj: usize,
    ) {
        if self.mlx_int4_mode && !self.weights.mlx_biases.is_empty() {
            let biases = &self.weights.mlx_biases[layer][proj];
            self.mlx_matvec(gpu, name, input, qweight, scales, biases, output, k, n);
        } else {
            // Use p_scratch for dynamic k/n (this path is not on the inner hot loop)
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct P { k: u32, n: u32, group_size: u32, _pad: u32 }
            gpu.flush();
            gpu.write_buffer(&self.state.p_scratch, 0, bytemuck::bytes_of(&P {
                k, n, group_size: self.quant_config.group_size, _pad: 0,
            }));
            self.gptq_matvec(gpu, name, input, qweight, scales, output, n, &self.state.p_scratch);
        }
    }

    /// MLX INT4 matvec: asymmetric dequant with separate biases buffer.
    pub fn mlx_matvec(
        &self, gpu: &mut GpuContext, name: &str,
        input: &wgpu::Buffer, qweight: &wgpu::Buffer, scales: &wgpu::Buffer,
        biases: &wgpu::Buffer, output: &wgpu::Buffer, k: u32, n: u32,
    ) {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { in_dim: u32, out_dim: u32, group_size: u32, _pad: u32 }
        gpu.flush();
        gpu.write_buffer(&self.state.p_scratch, 0, bytemuck::bytes_of(&P {
            in_dim: k, out_dim: n, group_size: self.quant_config.group_size, _pad: 0,
        }));
        let shader = if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX };
        gpu.dispatch(name, shader, &[
            gpu::bind(0, input), gpu::bind(1, qweight),
            gpu::bind(2, scales), gpu::bind(3, biases),
            gpu::bind(4, output), gpu::bind(5, &self.state.p_scratch),
        ], (n.div_ceil(32), 1, 1));
    }

    pub fn gptq_matvec(
        &self, gpu: &mut GpuContext, name: &str,
        input: &wgpu::Buffer, qweight: &wgpu::Buffer, scales: &wgpu::Buffer,
        output: &wgpu::Buffer, n: u32, params_buf: &wgpu::Buffer,
    ) {
        if self.bf16_mode {
            // BF16 mode: qweight contains bf16 packed weights, scales is unused
            // params_buf is a bf16 params buffer with {hidden_size, vocab_size}
            gpu.dispatch(name, shaders::BF16_MATVEC, &[
                gpu::bind(0, input), gpu::bind(1, qweight),
                gpu::bind(2, output), gpu::bind(3, params_buf),
            ], (n.div_ceil(32), 1, 1));
            return;
        }
        gpu.dispatch(name, shaders::GPTQ_MATVEC_4T, &[
            gpu::bind(0, input), gpu::bind(1, qweight),
            gpu::bind(2, scales), gpu::bind(3, output),
            gpu::bind(4, params_buf),
        ], (n.div_ceil(8), 1, 1));
    }

    pub fn fused_silu_gptq_down(
        &self, gpu: &mut GpuContext,
        gate_out: &wgpu::Buffer, up_out: &wgpu::Buffer,
        down_qw: &wgpu::Buffer, down_sc: &wgpu::Buffer,
        output: &wgpu::Buffer, n: u32, params_buf: &wgpu::Buffer,
    ) {
        if self.bf16_mode {
            // BF16 mode: SiLU(gate) * up → logits (temp, large enough), then bf16_matvec → output.
            // Use logits buffer as temp — it's [vocab_size] which is always >= inter.
            gpu.dispatch("silu_mul_bf16", shaders::SILU_MUL, &[
                gpu::bind(0, gate_out), gpu::bind(1, up_out),
                gpu::bind(2, &self.state.logits), // temp: logits buffer is large enough
                gpu::bind(3, &self.state.p_bf16_silu),
            ], (self.config.intermediate_size.div_ceil(256), 1, 1));
            // bf16_matvec: output[row] = sum(down_proj[row,k] * silu_result[k])
            gpu.dispatch("down_bf16", shaders::BF16_MATVEC, &[
                gpu::bind(0, &self.state.logits), // silu result from temp
                gpu::bind(1, down_qw),
                gpu::bind(2, output),
                gpu::bind(3, params_buf),
            ], (n.div_ceil(32), 1, 1));
            return;
        }
        gpu.dispatch("fused_silu_gptq_4t", shaders::FUSED_SILU_GPTQ_4T, &[
            gpu::bind(0, gate_out), gpu::bind(1, up_out),
            gpu::bind(2, down_qw), gpu::bind(3, down_sc),
            gpu::bind(4, output), gpu::bind(5, params_buf),
        ], (n.div_ceil(8), 1, 1));
    }

    /// Fused gate+up GPTQ dispatch: both MLP projections in one pass over K input elements.
    /// Phase 2 decode optimization: saves 1 dispatch per layer vs two separate matvec calls.
    pub fn fused_gate_up_gptq(
        &self, gpu: &mut GpuContext,
        input: &wgpu::Buffer,
        gate_qw: &wgpu::Buffer, gate_sc: &wgpu::Buffer,
        up_qw: &wgpu::Buffer, up_sc: &wgpu::Buffer,
        gate_out: &wgpu::Buffer, up_out: &wgpu::Buffer,
        n: u32, params_buf: &wgpu::Buffer,
    ) {
        if self.bf16_mode {
            // BF16: two separate bf16_matvec dispatches for gate and up
            gpu.dispatch("gate_bf16", shaders::BF16_MATVEC, &[
                gpu::bind(0, input), gpu::bind(1, gate_qw),
                gpu::bind(2, gate_out), gpu::bind(3, params_buf),
            ], (n.div_ceil(32), 1, 1));
            gpu.dispatch("up_bf16", shaders::BF16_MATVEC, &[
                gpu::bind(0, input), gpu::bind(1, up_qw),
                gpu::bind(2, up_out), gpu::bind(3, params_buf),
            ], (n.div_ceil(32), 1, 1));
            return;
        }
        gpu.dispatch("gate_up_fused", shaders::FUSED_GATE_UP_GPTQ_4T, &[
            gpu::bind(0, input),
            gpu::bind(1, gate_qw), gpu::bind(2, gate_sc),
            gpu::bind(3, up_qw),   gpu::bind(4, up_sc),
            gpu::bind(5, gate_out), gpu::bind(6, up_out),
            gpu::bind(7, params_buf),
        ], (n.div_ceil(8), 1, 1));
    }

    pub fn add_rmsnorm(
        &self, gpu: &mut GpuContext,
        hidden: &wgpu::Buffer, addend: &wgpu::Buffer,
        weight: &wgpu::Buffer, output: &wgpu::Buffer,
    ) {
        let shader = if self.norm_direct { shaders::ADD_RMSNORM_DIRECT } else { shaders::ADD_RMSNORM };
        gpu.dispatch("add_rmsnorm", shader, &[
            gpu::bind(0, hidden), gpu::bind(1, addend),
            gpu::bind(2, weight), gpu::bind(3, output),
            gpu::bind(4, &self.state.p_norm),
        ], (1, 1, 1));
    }

    pub fn rmsnorm(
        &self, gpu: &mut GpuContext,
        input: &wgpu::Buffer, weight: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) {
        let shader = if self.norm_direct { shaders::RMSNORM_DIRECT } else { shaders::RMSNORM };
        gpu.dispatch("rmsnorm", shader, &[
            gpu::bind(0, input), gpu::bind(1, weight),
            gpu::bind(2, output), gpu::bind(3, &self.state.p_norm),
        ], (1, 1, 1));
    }

    pub fn embedding(&self, gpu: &mut GpuContext, token_id: u32) {
        // Chunked embedding lookup
        if !self.weights.embed_chunks.is_empty() {
            let chunk_size = self.weights.embed_chunk_size;
            let chunk_idx = token_id / chunk_size;
            let local_id = token_id % chunk_size;
            let chunk = &self.weights.embed_chunks[chunk_idx as usize];
            // Write local token id into p_embed[0] (dim at [4] is already correct)
            gpu.write_buffer(&self.state.p_embed, 0, &local_id.to_le_bytes());
            gpu.dispatch("embedding", shaders::EMBEDDING, &[
                gpu::bind(0, chunk),
                gpu::bind(1, &self.state.hidden),
                gpu::bind(2, &self.state.p_embed),
            ], (self.config.hidden_size.div_ceil(256), 1, 1));
            return;
        }
        // token_id was already written to p_embed[0] at start of forward()
        gpu.dispatch("embedding", shaders::EMBEDDING, &[
            gpu::bind(0, &self.weights.embed_tokens),
            gpu::bind(1, &self.state.hidden),
            gpu::bind(2, &self.state.p_embed),
        ], (self.config.hidden_size.div_ceil(256), 1, 1));
    }

    fn bf16_lm_head(&self, gpu: &mut GpuContext, weight: &wgpu::Buffer, h: u32) {
        // Chunked lm_head: dispatch bf16_matvec per chunk, write to logits at correct offset
        if !self.weights.embed_chunks.is_empty() {
            let chunk_size = self.weights.embed_chunk_size;
            let vocab = self.config.vocab_size;

            // All chunks run in a single encoder (no flush between chunks).
            // Pre-allocated per-chunk buffers (lm_chunk_tmps, lm_chunk_params) avoid
            // both the per-step GPU allocation and the flush()+write_buffer per chunk.
            for (ci, chunk) in self.weights.embed_chunks.iter().enumerate() {
                let chunk_start = ci as u32 * chunk_size;
                let chunk_vocab = chunk_size.min(vocab - chunk_start);
                let chunk_logits = &self.state.lm_chunk_tmps[ci];
                let chunk_params = &self.state.lm_chunk_params[ci];

                gpu.dispatch("lm_head_chunked", shaders::BF16_MATVEC, &[
                    gpu::bind(0, &self.state.normed),
                    gpu::bind(1, chunk),
                    gpu::bind(2, chunk_logits),
                    gpu::bind(3, chunk_params),
                ], (chunk_vocab.div_ceil(32), 1, 1));

                // GPU-side copy to the correct offset in the full logits buffer
                gpu.copy_buffer_offset(chunk_logits, 0,
                    &self.state.logits, chunk_start as u64 * 4,
                    chunk_vocab as u64 * 4);
            }
            return;
        }
        // Non-chunked: use static p_bf16_lm (h, vocab_size pre-initialized at construction)
        gpu.dispatch("lm_head_bf16", shaders::BF16_MATVEC, &[
            gpu::bind(0, &self.state.normed),
            gpu::bind(1, weight),
            gpu::bind(2, &self.state.logits),
            gpu::bind(3, &self.state.p_bf16_lm),
        ], (self.config.vocab_size.div_ceil(32), 1, 1));
    }

    /// Dispatch LM head logits computation — handles bf16, GPTQ, and MLX INT4 tied embeddings.
    pub fn dispatch_lm_head(&self, gpu: &mut GpuContext) {
        let h = self.config.hidden_size;
        log::debug!("[model] dispatch_lm_head: mlx_int4={} tied={} bf16={} has_scales={}",
            self.mlx_int4_mode, self.tied_embeddings, self.weights.lm_head_is_bf16,
            self.weights.embed_scales.is_some());
        if self.mlx_int4_mode && self.tied_embeddings {
            // MLX INT4 tied embeddings: use INT4_MATVEC_MLX with embed scales/biases
            if let (Some(ref scales), Some(ref biases)) = (&self.weights.embed_scales, &self.weights.embed_biases) {
                gpu.flush();
                let normed_bytes = gpu.read_buffer(&self.state.normed, h as u64 * 4);
                let nv: &[f32] = bytemuck::cast_slice(&normed_bytes);
                let nn: f32 = nv.iter().map(|x| x*x).sum::<f32>().sqrt();
                log::debug!("[model] MLX INT4 lm_head: vocab={}, hidden={}, normed_norm={nn:.4}", self.config.vocab_size, h);
                gpu.dispatch("lm_head_mlx", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                    gpu::bind(0, &self.state.normed),
                    gpu::bind(1, &self.weights.embed_tokens),
                    gpu::bind(2, scales),
                    gpu::bind(3, biases),
                    gpu::bind(4, &self.state.logits),
                    gpu::bind(5, &self.state.p_gptq_lm),
                ], (self.config.vocab_size.div_ceil(32), 1, 1));
            } else {
                log::error!("[model] mlx_int4 tied_embeddings but no embed_scales/biases!");
                self.bf16_lm_head(gpu, &self.weights.embed_tokens, h);
            }
        } else if self.tied_embeddings {
            self.bf16_lm_head(gpu, &self.weights.embed_tokens, h);
        } else if self.weights.lm_head_is_bf16 {
            self.bf16_lm_head(gpu, &self.weights.lm_head_qweight, h);
        } else {
            self.gptq_matvec(gpu, "lm_head",
                &self.state.normed, &self.weights.lm_head_qweight, &self.weights.lm_head_scales,
                &self.state.logits, self.config.vocab_size, &self.state.p_gptq_lm);
        }
    }

    fn argmax(&self, gpu: &mut GpuContext) {
        gpu.dispatch("argmax", shaders::ARGMAX, &[
            gpu::bind(0, &self.state.logits),
            gpu::bind(1, &self.state.argmax_result),
            gpu::bind(2, &self.state.p_argmax),
        ], (1, 1, 1));
    }

    /// Fused Q/K norm + mRoPE + KV cache store.
    /// Dispatches (num_heads + num_kv_heads) workgroups.
    pub fn fused_split_qknorm_kvstore(
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
            "qknorm",
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
    pub fn gqa_attention(&self, gpu: &mut GpuContext, layer_idx: usize) {
        let nh = self.config.num_attention_heads;
        let ns = NUM_ATTN_SPLITS;

        let output_buf = if ns == 1 {
            &self.state.attn_output
        } else {
            &self.state.attn_partials
        };

        // Write current seq_len to p_attn (tells shader how many KV positions to scan).
        // Done here so all forward paths (GPTQ, bf16, MLX) share the same write point.
        let seq_len_val = self.seq_len + 1;
        gpu.write_buffer(&self.state.p_attn, 0, &seq_len_val.to_le_bytes());

        gpu.dispatch(
            if self.q_gated { "gqa_gated" } else { "gqa" },
            &self.gqa_shader_src,
            &[
                gpu::bind(0, &self.state.q_proj),
                gpu::bind(1, &self.state.k_cache[layer_idx]),
                gpu::bind(2, &self.state.v_cache[layer_idx]),
                gpu::bind(3, output_buf),
                gpu::bind(4, &self.state.p_attn),
                gpu::bind(5, &self.state.q_gate),
            ],
            (nh, ns, 1), // one workgroup per Q head per split
        );

        // Multi-split reduction
        if ns > 1 {
            gpu.dispatch(
                "gqa_reduce",
                shaders::GQA_REDUCE,
                &[
                    gpu::bind(0, &self.state.attn_partials),
                    gpu::bind(1, &self.state.attn_output),
                    gpu::bind(2, &self.state.p_gqa_reduce),
                ],
                (nh, 1, 1),
            );
        }
    }

    /// Gated attention: attn_output[i] *= sigmoid(q_gate[i])
    fn sigmoid_mul_gate(&self, gpu: &mut GpuContext) {
        let n = self.config.num_attention_heads * self.config.head_dim;
        // sigmoid_mul: output[i] = x[i] / (1 + exp(-gate[i]))
        // Can't alias read+write on same buffer in wgpu, so write to o_proj_out as temp.
        // p_gqa_reduce[0] = head_dim (matches n = nh*hd layout) — but we need just {n: u32}.
        // Use p_scratch for this (flush + write n before dispatch).
        gpu.flush();
        gpu.write_buffer(&self.state.p_scratch, 0, &n.to_le_bytes());
        gpu.dispatch("sigmoid_mul", shaders::SIGMOID_MUL, &[
            gpu::bind(0, &self.state.attn_output),
            gpu::bind(1, &self.state.q_gate),
            gpu::bind(2, &self.state.o_proj_out), // temp output
            gpu::bind(3, &self.state.p_scratch),
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
        struct DownP { in_features: u32, rank: u32, _pad: [u32; 2] }
        gpu.flush();
        gpu.write_buffer(&self.state.p_scratch, 0, bytemuck::bytes_of(&DownP { in_features, rank, _pad: [0; 2] }));
        gpu.dispatch(
            &format!("lora_down_{name}"), shaders::LORA_DOWN,
            &[gpu::bind(0, input), gpu::bind(1, lora_a),
              gpu::bind(2, &lora.lora_hidden), gpu::bind(3, &self.state.p_scratch)],
            (rank.div_ceil(32), 1, 1),
        );

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct UpP { rank: u32, out_features: u32, scale: f32, _pad: u32 }
        gpu.flush();
        gpu.write_buffer(&self.state.p_scratch, 0, bytemuck::bytes_of(&UpP { rank, out_features, scale, _pad: 0 }));
        gpu.dispatch(
            &format!("lora_up_{name}"), shaders::LORA_UP_ADD,
            &[gpu::bind(0, &lora.lora_hidden), gpu::bind(1, lora_b),
              gpu::bind(2, output), gpu::bind(3, &self.state.p_scratch)],
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
        struct DownP { in_features: u32, rank: u32, _pad: [u32; 2] }
        gpu.flush();
        gpu.write_buffer(&self.state.p_scratch, 0, bytemuck::bytes_of(&DownP { in_features: inter, rank, _pad: [0; 2] }));
        gpu.dispatch(
            "lora_down_silu", shaders::LORA_DOWN_SILU,
            &[gpu::bind(0, &self.state.gate_out), gpu::bind(1, &self.state.up_out),
              gpu::bind(2, &lora.layers[layer_idx].down_proj_a),
              gpu::bind(3, &lora.lora_hidden), gpu::bind(4, &self.state.p_scratch)],
            (rank.div_ceil(32), 1, 1),
        );

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct UpP { rank: u32, out_features: u32, scale: f32, _pad: u32 }
        gpu.flush();
        gpu.write_buffer(&self.state.p_scratch, 0, bytemuck::bytes_of(&UpP { rank, out_features: h, scale, _pad: 0 }));
        gpu.dispatch(
            "lora_up_down", shaders::LORA_UP_ADD,
            &[gpu::bind(0, &lora.lora_hidden),
              gpu::bind(1, &lora.layers[layer_idx].down_proj_b),
              gpu::bind(2, &self.state.mlp_output), gpu::bind(3, &self.state.p_scratch)],
            (h.div_ceil(32), 1, 1),
        );
    }

    // ── Forward pass ──────────────────────────────────────────────────

    /// Returns true if any layer uses linear (non-self) attention.
    /// `prefill_gptq` only supports pure self-attn models.
    pub fn is_hybrid_attn(&self) -> bool {
        self.weights.layers.iter().any(|l| l.linear_attn().is_some())
    }

    pub fn forward(&mut self, gpu: &mut GpuContext, token_id: u32) -> u32 {
        let h = self.config.hidden_size;
        gpu.write_buffer(&self.state.p_embed, 0, &token_id.to_le_bytes());
        self.embedding(gpu, token_id);
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);
        self.forward_layers(gpu)
    }

    /// Prefill-only forward pass: runs embedding + all layers + KV cache update,
    /// but skips lm_head and the GPU→CPU logit readback.
    pub fn forward_kv_only(&mut self, gpu: &mut GpuContext, token_id: u32) {
        let h = self.config.hidden_size;
        gpu.write_buffer(&self.state.p_embed, 0, &token_id.to_le_bytes());
        self.embedding(gpu, token_id);
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);
        self.prefill_kv_only = true;
        self.forward_layers(gpu);
        self.prefill_kv_only = false;
    }

    /// Forward pass with raw f32 embedding instead of token ID lookup.
    /// Used for ASR decoder where encoder output embeddings are injected directly.
    pub fn forward_embed(&mut self, gpu: &mut GpuContext, embed: &[f32]) -> u32 {
        let h = self.config.hidden_size;
        gpu.write_buffer(&self.state.hidden, 0, bytemuck::cast_slice(embed));
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);
        self.forward_layers(gpu)
    }

    /// Run transformer layers + lm_head + sampling (shared by forward and forward_embed).
    fn forward_layers(&mut self, gpu: &mut GpuContext) -> u32 {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;

        // Layer loop
        for i in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[i];

            // ── Pre-attention norm ──
            if i == 0 {
                self.rmsnorm(gpu, &self.state.hidden, &layer.input_layernorm, &self.state.normed);
            } else {
                self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
                    &layer.input_layernorm, &self.state.normed);
            }

            // ── Attention (self-attn or DeltaNet) ──
            if let Some(sa) = layer.self_attn() {
                let q_dim = if self.q_gated { nh * hd * 2 } else { nh * hd };
                let kv_dim = nkv * hd;
                let p_q = if self.bf16_mode { &self.state.p_bf16_q } else { &self.state.p_gptq_q };
                let p_kv = if self.bf16_mode { &self.state.p_bf16_kv } else { &self.state.p_gptq_kv };
                let p_o = if self.bf16_mode { &self.state.p_bf16_o } else { &self.state.p_gptq_o };
                self.gptq_matvec(gpu, "qproj",
                    &self.state.normed, &sa.q_proj_qweight, &sa.q_proj_scales,
                    &self.state.q_out, q_dim, p_q);
                #[cfg(feature = "jit-lora")]
                if self.lora.as_ref().map_or(false, |l| l.config.targets[0]) {
                    let lw = &self.lora.as_ref().unwrap().layers[i];
                    self.lora_apply(gpu, "qproj",
                        &self.state.normed, &lw.q_proj_a, &lw.q_proj_b,
                        &self.state.q_out, h, q_dim);
                }

                self.gptq_matvec(gpu, "kproj",
                    &self.state.normed, &sa.k_proj_qweight, &sa.k_proj_scales,
                    &self.state.k_out, kv_dim, p_kv);
                self.gptq_matvec(gpu, "vproj",
                    &self.state.normed, &sa.v_proj_qweight, &sa.v_proj_scales,
                    &self.state.v_out, kv_dim, p_kv);
                #[cfg(feature = "jit-lora")]
                if self.lora.as_ref().map_or(false, |l| l.config.targets[1]) {
                    let lw = &self.lora.as_ref().unwrap().layers[i];
                    self.lora_apply(gpu, "vproj",
                        &self.state.normed, &lw.v_proj_a, &lw.v_proj_b,
                        &self.state.v_out, h, kv_dim);
                }

                self.fused_split_qknorm_kvstore(gpu, i);
                self.gqa_attention(gpu, i); // sigmoid gate fused in when q_gated

                self.gptq_matvec(gpu, "oproj",
                    &self.state.attn_output, &sa.o_proj_qweight, &sa.o_proj_scales,
                    &self.state.o_proj_out, h, p_o);


                #[cfg(feature = "jit-lora")]
                if self.lora.as_ref().map_or(false, |l| l.config.targets[2]) {
                    let lw = &self.lora.as_ref().unwrap().layers[i];
                    self.lora_apply(gpu, "oproj",
                        &self.state.attn_output, &lw.o_proj_a, &lw.o_proj_b,
                        &self.state.o_proj_out, nh * hd, h);
                }
            } else if let Some(la) = layer.linear_attn() {
                let lnkh = self.linear_num_key_heads;
                let lkd = self.linear_key_dim;
                let lnvh = self.linear_num_value_heads;
                let lvd = self.linear_value_dim;
                let total_ch = lnkh * lkd + lnkh * lkd + lnvh * lvd;

                let lin_idx = (0..i).filter(|j| !self.weights.self_attn_layers.contains(j)).count();

                // Static params buffers — no flush needed, values pre-computed at model init.
                self.gptq_matvec(gpu, "dn_qkv",
                    &self.state.normed, &la.in_proj_qkv_qweight, &la.in_proj_qkv_scales,
                    &self.state.deltanet_qkv, total_ch, &self.state.p_dn_qkv);

                let dn_z_n = lnvh * lvd;
                self.gptq_matvec(gpu, "dn_z",
                    &self.state.normed, &la.in_proj_z_qweight, &la.in_proj_z_scales,
                    &self.state.deltanet_z, dn_z_n, &self.state.p_dn_z);

                gpu.dispatch(
                    "deltanet",
                    shaders::FUSED_CONV_DELTANET_NORM,
                    &[
                        gpu::bind(0, &self.state.deltanet_qkv),
                        gpu::bind(1, &self.state.deltanet_hist[lin_idx]),
                        gpu::bind(2, &la.conv1d_weight),
                        gpu::bind(3, &self.state.deltanet_state[lin_idx]),
                        gpu::bind(4, &self.state.deltanet_output),
                        gpu::bind(5, &self.state.normed),
                        gpu::bind(6, &la.ab_weight),
                        gpu::bind(7, &la.a_log),
                        gpu::bind(8, &la.dt_bias),
                        gpu::bind(9, &la.norm_weight),
                        gpu::bind(10, &self.state.p_dn),
                    ],
                    (lnkh, 1, 1),
                );

                self.fused_silu_gptq_down(gpu,
                    &self.state.deltanet_z, &self.state.deltanet_output,
                    &la.out_proj_qweight, &la.out_proj_scales,
                    &self.state.o_proj_out, h, &self.state.p_dn_down);
            } else {
                gpu.copy_buffer(&self.state.normed, &self.state.o_proj_out, h as u64 * 4);
            }

            // ── Post-attention norm ──
            self.add_rmsnorm(gpu, &self.state.residual, &self.state.o_proj_out,
                &layer.post_attn_layernorm, &self.state.normed);

            // ── MLP ── (fused gate+up dispatch, then fused SiLU+down)
            let p_gu = if self.bf16_mode { &self.state.p_bf16_gu } else { &self.state.p_gptq_gu };
            let p_down = if self.bf16_mode { &self.state.p_bf16_down } else { &self.state.p_gptq_down };
            self.fused_gate_up_gptq(gpu,
                &self.state.normed,
                &layer.gate_proj_qweight, &layer.gate_proj_scales,
                &layer.up_proj_qweight, &layer.up_proj_scales,
                &self.state.gate_out, &self.state.up_out, inter, p_gu);
            self.fused_silu_gptq_down(gpu,
                &self.state.gate_out, &self.state.up_out,
                &layer.down_proj_qweight, &layer.down_proj_scales,
                &self.state.mlp_output, h, p_down);
            #[cfg(feature = "jit-lora")]
            if self.lora.as_ref().map_or(false, |l| l.config.targets[3]) {
                self.lora_apply_down_proj(gpu, i);
            }

        }

        // Final norm
        self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
            &self.weights.final_norm, &self.state.normed);

        self.seq_len += 1;

        // Prefill-only mode: skip lm_head + GPU→CPU readback.
        // Submit this token's layers as its own command buffer (non-blocking).
        // Vulkan in-order queue execution guarantees each token sees the previous one's KV writes.
        // The single device.poll() in the final read_buffer() syncs all prefill submissions.
        if self.prefill_kv_only {
            gpu.flush();
            return 0;
        }

        // Sync layers before lm_head to bound work per poll on PowerVR.
        // Poll #1: layers (32 layers × ~5 dispatches each)
        // Poll #2: lm_head chunks + penalty + topk (via read_buffer)
        log::info!("[forward] seq={} polling layers...", self.seq_len);
        gpu.flush_and_wait();
        log::info!("[forward] seq={} layers done, dispatching lm_head", self.seq_len);

        // LM head
        self.dispatch_lm_head(gpu);

        #[cfg(feature = "jit-lora")]
        if self.training_mode {
            return 0;
        }

        self.sample_token_gpu(gpu)
    }

    /// GPU-accelerated sampling:
    ///   1. Apply repetition penalty + presence penalty + temperature on GPU (1 dispatch, O(V))
    ///   2. Extract top-K candidates on GPU (1 dispatch, O(K·V) in one workgroup, no readback of full logits)
    ///   3. CPU reads TOPK_K * 8 = 64 bytes, does softmax + top-p nucleus sampling
    ///
    /// Reduces GPU→CPU transfer from 607KB (full logits) to 160 bytes.
    fn sample_token_gpu(&mut self, gpu: &mut GpuContext) -> u32 {
        let vocab = self.config.vocab_size;
        let rep_penalty = 1.0f32;
        let presence_penalty = 1.5f32;
        let temperature = 0.7f32;

        // ── Hard-ban: compute up to 6 banned token IDs on CPU ──────────────
        let n = self.generated_tokens.len();
        let mut bans = [u32::MAX; 6];
        let mut n_bans = 0u32;

        if n >= 2 && self.generated_tokens[n-1] == self.generated_tokens[n-2] {
            bans[n_bans as usize] = self.generated_tokens[n-1]; n_bans += 1;
        }
        if n >= 4
            && self.generated_tokens[n-1] == self.generated_tokens[n-3]
            && self.generated_tokens[n-2] == self.generated_tokens[n-4]
        {
            bans[n_bans as usize] = self.generated_tokens[n-1]; n_bans += 1;
            bans[n_bans as usize] = self.generated_tokens[n-2]; n_bans += 1;
        }
        if n >= 6
            && self.generated_tokens[n-1] == self.generated_tokens[n-4]
            && self.generated_tokens[n-2] == self.generated_tokens[n-5]
            && self.generated_tokens[n-3] == self.generated_tokens[n-6]
        {
            bans[n_bans as usize] = self.generated_tokens[n-1]; n_bans += 1;
        }

        // ── Upload penalty uniform ─────────────────────────────────────────
        // PenaltyParams layout (64 bytes):
        //   u32 vocab_size, f32 rep_penalty, f32 presence_penalty, f32 temperature  (16)
        //   u32 ban0..ban3  (16)
        //   u32 ban4, ban5, u32 n_bans, u32 _pad  (16)
        //   u32 gate_w0..gate_w3  (16) — 128-bit first-byte bitmap; all-ones = unconstrained
        let mut pu = [0u8; 64];
        pu[0..4].copy_from_slice(&vocab.to_le_bytes());
        pu[4..8].copy_from_slice(&rep_penalty.to_le_bytes());
        pu[8..12].copy_from_slice(&presence_penalty.to_le_bytes());
        pu[12..16].copy_from_slice(&temperature.to_le_bytes());
        pu[16..20].copy_from_slice(&bans[0].to_le_bytes());
        pu[20..24].copy_from_slice(&bans[1].to_le_bytes());
        pu[24..28].copy_from_slice(&bans[2].to_le_bytes());
        pu[28..32].copy_from_slice(&bans[3].to_le_bytes());
        pu[32..36].copy_from_slice(&bans[4].to_le_bytes());
        pu[36..40].copy_from_slice(&bans[5].to_le_bytes());
        pu[40..44].copy_from_slice(&n_bans.to_le_bytes());
        // gate_w0..w3: 128-bit bitmap over ASCII first bytes; all-ones = unconstrained
        let gate_words = self.json_sampler.as_ref()
            .map(|js| js.required_gate().as_gate_words())
            .unwrap_or([!0u32; 4]);
        // pu[44..48] = _pad (already zero)
        pu[48..52].copy_from_slice(&gate_words[0].to_le_bytes());
        pu[52..56].copy_from_slice(&gate_words[1].to_le_bytes());
        pu[56..60].copy_from_slice(&gate_words[2].to_le_bytes());
        pu[60..64].copy_from_slice(&gate_words[3].to_le_bytes());
        gpu.flush();
        gpu.write_buffer(&self.state.penalty_uniform, 0, &pu);

        // ── Upload per-token schema mask (packed bitfield) ────────────────
        // When schema is active, each bit says whether the token is valid
        // given the current FST state. When unconstrained, all bits set.
        let mask_words = self.json_sampler.as_ref()
            .and_then(|js| js.token_mask_words());
        if let Some(ref words) = mask_words {
            let set_bits: usize = words.iter().map(|w| w.count_ones() as usize).sum();
            log::debug!("[sample] token_mask: {}/{} tokens allowed", set_bits, vocab);
            let mask_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            gpu.write_buffer(&self.state.token_mask_buf, 0, &mask_bytes);
        } else {
            log::debug!("[sample] token_mask: unconstrained (all-ones)");
            // Unconstrained: all-ones (every token allowed)
            let num_words = vocab.div_ceil(32) as usize;
            let all_ones: Vec<u8> = vec![0xFFu8; num_words * 4];
            gpu.write_buffer(&self.state.token_mask_buf, 0, &all_ones);
        }

        // ── Combined penalty + gate + top-K dispatch (single vocab pass) ──
        gpu.dispatch(
            "sample_topk",
            shaders::SAMPLE_TOPK,
            &[
                gpu::bind(0, &self.state.logits),
                gpu::bind(1, &self.state.seen_bitmap),
                gpu::bind(2, &self.state.penalty_uniform),
                gpu::bind(3, &self.state.first_bytes_buf),
                gpu::bind(4, &self.state.topk_out),
                gpu::bind(5, &self.state.token_mask_buf),
            ],
            (1, 1, 1),
        );

        // ── Read back only the top-K candidates (24 bytes) ────────────────
        log::info!("[forward] seq={} polling lm_head+topk...", self.seq_len);
        let topk_bytes = gpu.read_buffer(&self.state.topk_out, TOPK_K as u64 * 8);
        log::info!("[forward] seq={} readback done", self.seq_len);
        // Each candidate is {idx: u32, val: f32} = 8 bytes
        let mut candidates: Vec<(u32, f32)> = (0..TOPK_K as usize)
            .map(|i| {
                let base = i * 8;
                let idx = u32::from_le_bytes(topk_bytes[base..base+4].try_into().unwrap());
                let val = f32::from_le_bytes(topk_bytes[base+4..base+8].try_into().unwrap());
                (idx, val)
            })
            .filter(|&(_, v)| v.is_finite())
            .collect();

        // Log top-K for diagnostics
        log::info!("[topk] seq={} gate={:?} candidates={:?}", self.seq_len,
            self.json_sampler.as_ref().map(|js| format!("{:?}", js.required_gate())),
            &candidates);

        // CPU-side schema validation: filter candidates by full token byte sequence
        if let Some(ref js) = self.json_sampler {
            // JSON complete → force EOS to stop generation
            if js.is_complete() {
                let eos = js.first_eos_id().unwrap_or(0);
                log::info!("[json-sampler] JSON complete at seq={}, forcing EOS token {}", self.seq_len, eos);
                self.generated_tokens.push(eos);
                return eos;
            }
            js.filter_by_schema(&mut candidates);
            js.suppress_eos_if_incomplete(&mut candidates);
        }

        if candidates.is_empty() {
            // Fallback: return first candidate unconditionally
            let idx = u32::from_le_bytes(topk_bytes[0..4].try_into().unwrap());
            self.last_token_prob = 1.0;
            self.generated_tokens.push(idx);
            self.mark_seen(gpu, idx);
            if let Some(ref mut js) = self.json_sampler {
                js.advance_token(idx);
            }
            return idx;
        }

        // ── At hard-gate positions, take top-1 (greedy) to avoid sampling valid-but-wrong tokens ──
        // e.g. at Root gate (requires '{'), greedy prevents selecting '{}_' over '{"'.
        let at_hard_gate = self.json_sampler.as_ref()
            .map(|js| !js.required_gate().is_any())
            .unwrap_or(false);

        // ── Top-p nucleus sampling on the small candidate set ─────────────
        let max_val = candidates.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(u32, f32)> = candidates.iter()
            .map(|&(idx, v)| (idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() { *p /= sum; }
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_p = 0.80f32;
        let mut cumsum = 0.0f32;
        let mut nucleus: Vec<(u32, f32)> = Vec::new();
        for (idx, p) in &probs {
            cumsum += p;
            nucleus.push((*idx, *p));
            if cumsum >= top_p { break; }
            if at_hard_gate { break; } // greedy: stop after top-1
        }
        let nuc_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
        for (_, p) in nucleus.iter_mut() { *p /= nuc_sum; }

        // ── XorShift64 sample ──────────────────────────────────────────────
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let r = (self.rng_state & 0xFFFFFFFF) as f64 / u32::MAX as f64;
        let mut cumulative = 0.0f64;
        let mut sampled = nucleus[0].0;
        let mut sampled_prob = nucleus[0].1;
        for &(idx, p) in &nucleus {
            cumulative += p as f64;
            if cumulative >= r {
                sampled = idx;
                sampled_prob = p;
                break;
            }
        }

        self.last_token_prob = sampled_prob;
        self.generated_tokens.push(sampled);
        self.mark_seen(gpu, sampled);
        if let Some(ref mut js) = self.json_sampler {
            js.advance_token(sampled);
        }
        sampled
    }

    /// Update the seen_bitmap (CPU + GPU) for a newly generated token.
    /// Writes a single 4-byte word to GPU — negligible overhead.
    fn mark_seen(&mut self, gpu: &GpuContext, token_id: u32) {
        let word_idx = (token_id / 32) as usize;
        if word_idx < self.seen_bitmap_cpu.len() {
            self.seen_bitmap_cpu[word_idx] |= 1u32 << (token_id % 32);
            let bytes = self.seen_bitmap_cpu[word_idx].to_le_bytes();
            gpu.write_buffer(&self.state.seen_bitmap, word_idx as u64 * 4, &bytes);
        }
    }

    /// Greedy argmax decode — no sampling, no penalties. For ASR.
    pub fn forward_argmax(&mut self, gpu: &mut GpuContext, token_id: u32) -> u32 {
        let h = self.config.hidden_size;
        gpu.write_buffer(&self.state.p_embed, 0, &token_id.to_le_bytes());
        self.embedding(gpu, token_id);
        gpu.flush();
        if self.seq_len == 0 {
            let dbg = gpu.read_buffer(&self.state.hidden, h as u64 * 4);
            let dv: &[f32] = bytemuck::cast_slice(&dbg);
            let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
            log::debug!("[gpu-decoder] embed tok={token_id}: norm={norm:.4} first4={:?}", &dv[..4]);
        }
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);
        self.forward_layers_argmax(gpu)
    }

    /// Greedy argmax with raw embedding injection. For ASR encoder output tokens.
    pub fn forward_embed_argmax(&mut self, gpu: &mut GpuContext, embed: &[f32]) -> u32 {
        let h = self.config.hidden_size;
        gpu.write_buffer(&self.state.hidden, 0, bytemuck::cast_slice(embed));
        gpu.flush();
        gpu.copy_buffer(&self.state.hidden, &self.state.residual, h as u64 * 4);
        self.forward_layers_argmax(gpu)
    }

    /// Run transformer layers + lm_head + argmax (no sampling).
    fn forward_layers_argmax(&mut self, gpu: &mut GpuContext) -> u32 {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;

        // Layer loop (same as forward_layers but skip sampling)
        for i in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[i];

            if i == 0 {
                self.rmsnorm(gpu, &self.state.hidden, &layer.input_layernorm, &self.state.normed);
            } else {
                self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
                    &layer.input_layernorm, &self.state.normed);
            }

            if let Some(sa) = layer.self_attn() {
                let nh = self.config.num_attention_heads;
                let nkv = self.config.num_key_value_heads;
                let hd = self.config.head_dim;
                let q_dim = if self.q_gated { nh * hd * 2 } else { nh * hd };
                let kv_dim = nkv * hd;
                let p_q = if self.bf16_mode { &self.state.p_bf16_q } else { &self.state.p_gptq_q };
                let p_kv = if self.bf16_mode { &self.state.p_bf16_kv } else { &self.state.p_gptq_kv };
                let p_o = if self.bf16_mode { &self.state.p_bf16_o } else { &self.state.p_gptq_o };
                self.gptq_matvec(gpu, "qproj",
                    &self.state.normed, &sa.q_proj_qweight, &sa.q_proj_scales,
                    &self.state.q_out, q_dim, p_q);
                self.gptq_matvec(gpu, "kproj",
                    &self.state.normed, &sa.k_proj_qweight, &sa.k_proj_scales,
                    &self.state.k_out, kv_dim, p_kv);
                self.gptq_matvec(gpu, "vproj",
                    &self.state.normed, &sa.v_proj_qweight, &sa.v_proj_scales,
                    &self.state.v_out, kv_dim, p_kv);
                self.fused_split_qknorm_kvstore(gpu, i);
                self.gqa_attention(gpu, i); // sigmoid gate fused in when q_gated
                self.gptq_matvec(gpu, "oproj",
                    &self.state.attn_output, &sa.o_proj_qweight, &sa.o_proj_scales,
                    &self.state.o_proj_out, h, p_o);
            } else {
                gpu.copy_buffer(&self.state.normed, &self.state.o_proj_out, h as u64 * 4);
            }

            self.add_rmsnorm(gpu, &self.state.residual, &self.state.o_proj_out,
                &layer.post_attn_layernorm, &self.state.normed);

            // Dump at seq_len=26 (last prefill position, matches C dump at pos 25)
            if self.seq_len == 0 && i < 3 {
                gpu.flush();
                let dbg = gpu.read_buffer(&self.state.residual, h as u64 * 4);
                let dv: &[f32] = bytemuck::cast_slice(&dbg);
                let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
                log::debug!("[gpu-decoder] L{i} after attn+res (seq={}): norm={norm:.4} first4={:?}", self.seq_len, &dv[..4]);
            }

            let p_gu = if self.bf16_mode { &self.state.p_bf16_gu } else { &self.state.p_gptq_gu };
            let p_down = if self.bf16_mode { &self.state.p_bf16_down } else { &self.state.p_gptq_down };
            self.fused_gate_up_gptq(gpu,
                &self.state.normed,
                &layer.gate_proj_qweight, &layer.gate_proj_scales,
                &layer.up_proj_qweight, &layer.up_proj_scales,
                &self.state.gate_out, &self.state.up_out, inter, p_gu);
            self.fused_silu_gptq_down(gpu,
                &self.state.gate_out, &self.state.up_out,
                &layer.down_proj_qweight, &layer.down_proj_scales,
                &self.state.mlp_output, h, p_down);

            if self.seq_len == 0 && i < 3 {
                gpu.flush();
                let dbg = gpu.read_buffer(&self.state.mlp_output, h as u64 * 4);
                let dv: &[f32] = bytemuck::cast_slice(&dbg);
                let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
                log::debug!("[gpu-decoder] L{i} mlp_out (seq={}): norm={norm:.4} first4={:?}", self.seq_len, &dv[..4]);
            }
        }

        // Final norm + LM head
        self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
            &self.weights.final_norm, &self.state.normed);

        self.dispatch_lm_head(gpu);

        self.seq_len += 1;

        // Greedy argmax
        let logits_bytes = gpu.read_buffer(&self.state.logits, self.config.vocab_size as u64 * 4);
        let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let (max_idx, _) = logits.iter().enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        let token = max_idx as u32;
        self.generated_tokens.push(token);
        token
    }

    /// MLX INT4 embedding lookup.
    pub fn embedding_mlx(&self, gpu: &mut GpuContext, token_id: u32) {
        // mlx_biases is empty for embed — use the dedicated embed buffers from weights
        // TODO: store embed scales/biases properly. For now, use the shader with
        // the correct buffers passed from the asr_decoder.
        let _ = (gpu, token_id);
        panic!("embedding_mlx needs dedicated embed scale/bias buffers — call from asr_decoder");
    }

    /// MLX INT4 forward: single token through all layers using int4_matvec_mlx.
    /// Dedicated forward path — no changes to existing gptq/bf16 methods.
    pub fn forward_mlx_argmax(&mut self, gpu: &mut GpuContext) -> u32 {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;

        // MLX INT4 uses named params buffers (same byte layout as GPTQ {k, n, gs})
        for i in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[i];
            let biases = &self.weights.mlx_biases[i];
            // biases: [q=0, k=1, v=2, o=3, gate=4, up=5, down=6]

            // Pre-attention norm
            if i == 0 {
                self.rmsnorm(gpu, &self.state.hidden, &layer.input_layernorm, &self.state.normed);
            } else {
                self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
                    &layer.input_layernorm, &self.state.normed);
            }

            if let Some(sa) = layer.self_attn() {
                let q_dim = if self.q_gated { nh * hd * 2 } else { nh * hd };
                let kv_dim = nkv * hd;

                if self.generated_tokens.is_empty() && i == 0 {
                    gpu.flush();
                    let dbg = gpu.read_buffer(&self.state.normed, h as u64 * 4);
                    let dv: &[f32] = bytemuck::cast_slice(&dbg);
                    let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let has_nan = dv.iter().any(|x| x.is_nan());
                    log::debug!("[model] L0 after rmsnorm: norm={norm:.4} nan={has_nan} first4={:?}", &dv[..4]);
                }

                // Q projection (MLX INT4) — p_gptq_q has {h, q_dim, gs} layout
                gpu.dispatch("qproj", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &sa.q_proj_qweight),
                    gpu::bind(2, &sa.q_proj_scales), gpu::bind(3, &biases[0]),
                    gpu::bind(4, &self.state.q_out), gpu::bind(5, &self.state.p_gptq_q),
                ], (q_dim.div_ceil(32), 1, 1));

                // K projection
                gpu.dispatch("kproj", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &sa.k_proj_qweight),
                    gpu::bind(2, &sa.k_proj_scales), gpu::bind(3, &biases[1]),
                    gpu::bind(4, &self.state.k_out), gpu::bind(5, &self.state.p_gptq_kv),
                ], (kv_dim.div_ceil(32), 1, 1));

                // V projection
                gpu.dispatch("vproj", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                    gpu::bind(0, &self.state.normed), gpu::bind(1, &sa.v_proj_qweight),
                    gpu::bind(2, &sa.v_proj_scales), gpu::bind(3, &biases[2]),
                    gpu::bind(4, &self.state.v_out), gpu::bind(5, &self.state.p_gptq_kv),
                ], (kv_dim.div_ceil(32), 1, 1));

                if self.generated_tokens.is_empty() && i == 0 {
                    gpu.flush();
                    let dbg = gpu.read_buffer(&self.state.q_out, q_dim as u64 * 4);
                    let dv: &[f32] = bytemuck::cast_slice(&dbg);
                    let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let has_nan = dv.iter().any(|x| x.is_nan());
                    log::debug!("[model] L0 after qproj: q_norm={norm:.4} nan={has_nan} q_dim={q_dim}");
                    let kdbg = gpu.read_buffer(&self.state.k_out, kv_dim as u64 * 4);
                    let kv: &[f32] = bytemuck::cast_slice(&kdbg);
                    let knorm: f32 = kv.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let knan = kv.iter().any(|x| x.is_nan());
                    log::debug!("[model] L0 after kproj: k_norm={knorm:.4} nan={knan} kv_dim={kv_dim}");
                }

                self.fused_split_qknorm_kvstore(gpu, i);

                if self.generated_tokens.is_empty() && i == 0 {
                    gpu.flush();
                    // Check K cache at current position
                    let kc_size = (nkv * hd) as u64 * 4;
                    let kc_off = self.seq_len as u64 * kc_size;
                    let kdbg = gpu.read_buffer_offset(&self.state.k_cache[0], kc_off, kc_size);
                    let kv: &[f32] = bytemuck::cast_slice(&kdbg);
                    let knorm: f32 = kv.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let knan = kv.iter().any(|x| x.is_nan());
                    log::debug!("[model] L0 after qknorm+kvstore: k_cache norm={knorm:.4} nan={knan} seq_len={}", self.seq_len);
                }

                self.gqa_attention(gpu, i); // sigmoid gate fused in when q_gated

                if self.generated_tokens.is_empty() && i == 0 {
                    gpu.flush();
                    let dbg = gpu.read_buffer(&self.state.attn_output, h as u64 * 4);
                    let dv: &[f32] = bytemuck::cast_slice(&dbg);
                    let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let has_nan = dv.iter().any(|x| x.is_nan());
                    let has_inf = dv.iter().any(|x| x.is_infinite());
                    let first_nan = dv.iter().position(|x| x.is_nan());
                    log::debug!("[model] L0 after attention: norm={norm:.4} nan={has_nan} inf={has_inf} first_nan={first_nan:?} first4={:?}", &dv[..4.min(dv.len())]);
                    // Also check q_proj (post-norm, post-RoPE)
                    let qdbg = gpu.read_buffer(&self.state.q_proj, (nh * hd) as u64 * 4);
                    let qv: &[f32] = bytemuck::cast_slice(&qdbg);
                    let qnorm: f32 = qv.iter().map(|x| x*x).sum::<f32>().sqrt();
                    let qnan = qv.iter().any(|x| x.is_nan());
                    log::debug!("[model] L0 q_proj (normed+RoPE): norm={qnorm:.4} nan={qnan}");
                }

                // O projection
                gpu.dispatch("oproj", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                    gpu::bind(0, &self.state.attn_output), gpu::bind(1, &sa.o_proj_qweight),
                    gpu::bind(2, &sa.o_proj_scales), gpu::bind(3, &biases[3]),
                    gpu::bind(4, &self.state.o_proj_out), gpu::bind(5, &self.state.p_gptq_o),
                ], (h.div_ceil(32), 1, 1));
            } else {
                gpu.copy_buffer(&self.state.normed, &self.state.o_proj_out, h as u64 * 4);
            }

            // Post-attention norm
            self.add_rmsnorm(gpu, &self.state.residual, &self.state.o_proj_out,
                &layer.post_attn_layernorm, &self.state.normed);

            // MLP: gate + up (INT4), then fused SiLU + down (INT4)
            gpu.dispatch("gate", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                gpu::bind(0, &self.state.normed), gpu::bind(1, &layer.gate_proj_qweight),
                gpu::bind(2, &layer.gate_proj_scales), gpu::bind(3, &biases[4]),
                gpu::bind(4, &self.state.gate_out), gpu::bind(5, &self.state.p_gptq_gu),
            ], (inter.div_ceil(32), 1, 1));

            gpu.dispatch("up", if self.mlx_bf16_scales { shaders::INT4_MATVEC_MLX_BF16 } else { shaders::INT4_MATVEC_MLX }, &[
                gpu::bind(0, &self.state.normed), gpu::bind(1, &layer.up_proj_qweight),
                gpu::bind(2, &layer.up_proj_scales), gpu::bind(3, &biases[5]),
                gpu::bind(4, &self.state.up_out), gpu::bind(5, &self.state.p_gptq_gu),
            ], (inter.div_ceil(32), 1, 1));

            // Fused SiLU(gate) * up → INT4 down_proj → mlp_output
            gpu.dispatch("silu_down", shaders::FUSED_SILU_INT4_MLX, &[
                gpu::bind(0, &self.state.gate_out), gpu::bind(1, &self.state.up_out),
                gpu::bind(2, &layer.down_proj_qweight),
                gpu::bind(3, &layer.down_proj_scales), gpu::bind(4, &biases[6]),
                gpu::bind(5, &self.state.mlp_output), gpu::bind(6, &self.state.p_gptq_down),
            ], (h.div_ceil(32), 1, 1));

            if self.generated_tokens.is_empty() && (i == 0 || i == self.config.num_hidden_layers as usize - 1) {
                gpu.flush();
                let dbg = gpu.read_buffer(&self.state.mlp_output, h as u64 * 4);
                let dv: &[f32] = bytemuck::cast_slice(&dbg);
                let has_nan = dv.iter().any(|x| x.is_nan());
                let norm: f32 = dv.iter().map(|x| x*x).sum::<f32>().sqrt();
                log::debug!("[model] mlx layer {i}: mlp_out norm={norm:.4} nan={has_nan}");
            }
        }

        // Final norm
        self.add_rmsnorm(gpu, &self.state.residual, &self.state.mlp_output,
            &self.weights.final_norm, &self.state.normed);

        // LM head
        self.dispatch_lm_head(gpu);

        self.seq_len += 1;

        // Greedy argmax
        let logits_bytes = gpu.read_buffer(&self.state.logits, self.config.vocab_size as u64 * 4);
        let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let (max_idx, max_val) = logits.iter().enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        let token = max_idx as u32;
        if self.generated_tokens.len() < 3 {
            let norm: f32 = logits.iter().map(|x| x*x).sum::<f32>().sqrt();
            let nonzero = logits.iter().filter(|&&x| x.abs() > 1e-10).count();
            log::debug!("[model] logits: norm={norm:.4} max={max_val:.4}@{max_idx} nonzero={nonzero}/{}", logits.len());
        }
        self.generated_tokens.push(token);
        token
    }

    /// Batched prefill: process input_embeds [seq_len, hidden_size] through all layers.
    /// Populates KV cache for subsequent autoregressive decode.
    /// Returns the argmax token predicted from the last position.
    /// bf16_mode must be true. q_gated must be false (ASR decoder).
    pub fn prefill(&mut self, gpu: &mut GpuContext, input_embeds: &[f32], seq_len: u32) -> u32 {
        assert!(self.bf16_mode, "prefill requires bf16_mode");
        assert!(!self.q_gated, "prefill only supports non-gated Q (ASR decoder)");

        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let f = 4u64; // bytes per f32
        let sl = seq_len as u64;

        // Upload input embeddings
        let x_buf = gpu.upload_buffer("prefill_x", bytemuck::cast_slice(input_embeds));

        // Allocate batched scratch buffers
        let residual = gpu.create_storage_buffer("pf_residual", sl * h as u64 * f);
        let normed = gpu.create_storage_buffer("pf_normed", sl * h as u64 * f);
        let q_buf = gpu.create_storage_buffer("pf_q", sl * (nh * hd) as u64 * f);
        let k_buf = gpu.create_storage_buffer("pf_k", sl * (nkv * hd) as u64 * f);
        let v_buf = gpu.create_storage_buffer("pf_v", sl * (nkv * hd) as u64 * f);
        let attn_out = gpu.create_storage_buffer("pf_attn", sl * (nh * hd) as u64 * f);
        let o_out = gpu.create_storage_buffer("pf_o", sl * h as u64 * f);
        let gate_buf = gpu.create_storage_buffer("pf_gate", sl * inter as u64 * f);
        let up_buf = gpu.create_storage_buffer("pf_up", sl * inter as u64 * f);
        let silu_buf = gpu.create_storage_buffer("pf_silu", sl * inter as u64 * f);
        let mlp_out = gpu.create_storage_buffer("pf_mlp", sl * h as u64 * f);

        // Params buffer for prefill dispatches
        let params = gpu.create_buffer("pf_params", 256,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        // Copy x → residual
        gpu.copy_buffer(&x_buf, &residual, sl * h as u64 * f);

        // Build batched qknorm shader with model constants
        let batched_qknorm_src = {
            let partial_dim = (self.config.head_dim as f32 * self.config.partial_rotary_factor) as u32;
            let s_limit = partial_dim / 2;
            format!(
                "const ROPE_THETA: f32 = {:.1};\n\
                 const PARTIAL_DIM: u32 = {}u;\n\
                 const MROPE_INTERLEAVED: bool = {};\n\n{}",
                self.config.rope_theta,
                partial_dim,
                self.config.mrope_interleaved(),
                include_str!("shaders/batched_qknorm_rope.wgsl")
                    .lines()
                    .skip(12) // skip comment block (9 lines) + 3 const lines
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        };

        for layer_idx in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[layer_idx];

            // ── RMSNorm ──
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct NormP { n: u32, eps: f32, seq_len: u32, _pad: u32 }

            if layer_idx == 0 {
                // First layer: rmsnorm(x_buf)
                gpu.flush();
                gpu.write_buffer(&params, 0, bytemuck::bytes_of(&NormP {
                    n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
                gpu.dispatch("pf_norm", shaders::BATCHED_RMSNORM, &[
                    gpu::bind(0, &x_buf),
                    gpu::bind(1, &layer.input_layernorm),
                    gpu::bind(2, &normed),
                    gpu::bind(3, &params),
                ], (seq_len, 1, 1));
            } else {
                // add_rmsnorm(residual += mlp_out, normed)
                gpu.flush();
                gpu.write_buffer(&params, 0, bytemuck::bytes_of(&NormP {
                    n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
                gpu.dispatch("pf_addnorm", shaders::BATCHED_ADD_RMSNORM, &[
                    gpu::bind(0, &residual),
                    gpu::bind(1, &mlp_out),
                    gpu::bind(2, &layer.input_layernorm),
                    gpu::bind(3, &normed),
                    gpu::bind(4, &params),
                ], (seq_len, 1, 1));
            }

            // ── Q/K/V projections (bf16 GEMM) ──
            let sa = layer.self_attn().expect("ASR decoder must be self-attn");

            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct GemmP { d_in: u32, d_out: u32, seq_len: u32, has_bias: u32 }

            // Q: [seq_len, h] × [nh*hd, h]^T → [seq_len, nh*hd]
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: h, d_out: nh * hd, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_q", shaders::BF16_GEMM, &[
                gpu::bind(0, &normed),
                gpu::bind(1, &sa.q_proj_qweight),
                gpu::bind(2, &normed), // bias unused (has_bias=0)
                gpu::bind(3, &q_buf),
                gpu::bind(4, &params),
            ], ((nh * hd).div_ceil(32), seq_len, 1));

            // K: [seq_len, h] → [seq_len, nkv*hd]
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: h, d_out: nkv * hd, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_k", shaders::BF16_GEMM, &[
                gpu::bind(0, &normed),
                gpu::bind(1, &sa.k_proj_qweight),
                gpu::bind(2, &normed),
                gpu::bind(3, &k_buf),
                gpu::bind(4, &params),
            ], ((nkv * hd).div_ceil(32), seq_len, 1));

            // V: [seq_len, h] → [seq_len, nkv*hd]
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: h, d_out: nkv * hd, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_v", shaders::BF16_GEMM, &[
                gpu::bind(0, &normed),
                gpu::bind(1, &sa.v_proj_qweight),
                gpu::bind(2, &normed),
                gpu::bind(3, &v_buf),
                gpu::bind(4, &params),
            ], ((nkv * hd).div_ceil(32), seq_len, 1));

            // ── Q/K Norm + RoPE + KV cache write ──
            // qknorm_params[layer_idx] was pre-filled by init_qknorm_params with
            // header (num_heads, kv_heads, head_dim, eps) + packed norm weights.
            // The batched shader reads position from workgroup_id.y, not the buffer.
            gpu.flush();
            gpu.dispatch("pf_qknorm", &batched_qknorm_src, &[
                gpu::bind(0, &q_buf),
                gpu::bind(1, &k_buf),
                gpu::bind(2, &v_buf),
                gpu::bind(3, &self.state.k_cache[layer_idx]),
                gpu::bind(4, &self.state.v_cache[layer_idx]),
                gpu::bind(5, &self.state.qknorm_params[layer_idx]),
            ], (nh + nkv, seq_len, 1));

            // ── Causal attention ──
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct AttnP { seq_len: u32, head_dim: u32, num_kv_heads: u32, num_q_heads: u32, heads_per_kv: u32 }
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&AttnP {
                seq_len, head_dim: hd, num_kv_heads: nkv, num_q_heads: nh,
                heads_per_kv: nh / nkv }));
            gpu.dispatch("pf_attn", shaders::CAUSAL_ATTENTION_PREFILL, &[
                gpu::bind(0, &q_buf),
                gpu::bind(1, &k_buf),
                gpu::bind(2, &v_buf),
                gpu::bind(3, &attn_out),
                gpu::bind(4, &params),
            ], (nh, seq_len, 1));

            // ── O projection: [seq_len, nh*hd] → [seq_len, h] ──
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: nh * hd, d_out: h, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_o", shaders::BF16_GEMM, &[
                gpu::bind(0, &attn_out),
                gpu::bind(1, &sa.o_proj_qweight),
                gpu::bind(2, &attn_out),
                gpu::bind(3, &o_out),
                gpu::bind(4, &params),
            ], (h.div_ceil(32), seq_len, 1));

            // ── Post-attention norm: residual += o_out ──
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&NormP {
                n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
            gpu.dispatch("pf_postnorm", shaders::BATCHED_ADD_RMSNORM, &[
                gpu::bind(0, &residual),
                gpu::bind(1, &o_out),
                gpu::bind(2, &layer.post_attn_layernorm),
                gpu::bind(3, &normed),
                gpu::bind(4, &params),
            ], (seq_len, 1, 1));

            // ── MLP: gate + up → SiLU → down ──
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: h, d_out: inter, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_gate", shaders::BF16_GEMM, &[
                gpu::bind(0, &normed),
                gpu::bind(1, &layer.gate_proj_qweight),
                gpu::bind(2, &normed),
                gpu::bind(3, &gate_buf),
                gpu::bind(4, &params),
            ], (inter.div_ceil(32), seq_len, 1));

            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: h, d_out: inter, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_up", shaders::BF16_GEMM, &[
                gpu::bind(0, &normed),
                gpu::bind(1, &layer.up_proj_qweight),
                gpu::bind(2, &normed),
                gpu::bind(3, &up_buf),
                gpu::bind(4, &params),
            ], (inter.div_ceil(32), seq_len, 1));

            // SiLU(gate) * up → silu_buf
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct SiluP { n: u32 }
            let total_silu = seq_len * inter;
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&SiluP { n: total_silu }));
            gpu.dispatch("pf_silu", shaders::BATCHED_SILU_MUL, &[
                gpu::bind(0, &gate_buf),
                gpu::bind(1, &up_buf),
                gpu::bind(2, &silu_buf),
                gpu::bind(3, &params),
            ], (total_silu.div_ceil(256), 1, 1));

            // down_proj: [seq_len, inter] → [seq_len, h]
            gpu.flush();
            gpu.write_buffer(&params, 0, bytemuck::bytes_of(&GemmP {
                d_in: inter, d_out: h, seq_len, has_bias: 0 }));
            gpu.dispatch("pf_down", shaders::BF16_GEMM, &[
                gpu::bind(0, &silu_buf),
                gpu::bind(1, &layer.down_proj_qweight),
                gpu::bind(2, &silu_buf),
                gpu::bind(3, &mlp_out),
                gpu::bind(4, &params),
            ], (h.div_ceil(32), seq_len, 1));

            if (layer_idx + 1) % 7 == 0 {
                log::info!("[prefill] layer {}/{}", layer_idx + 1, self.config.num_hidden_layers);
            }
        }

        // Final: residual += mlp_out from last layer
        gpu.flush();
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct NormP2 { n: u32, eps: f32, seq_len: u32, _pad: u32 }
        gpu.write_buffer(&params, 0, bytemuck::bytes_of(&NormP2 {
            n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
        gpu.dispatch("pf_final_addnorm", shaders::BATCHED_ADD_RMSNORM, &[
            gpu::bind(0, &residual),
            gpu::bind(1, &mlp_out),
            gpu::bind(2, &self.weights.final_norm),
            gpu::bind(3, &normed),
            gpu::bind(4, &params),
        ], (seq_len, 1, 1));

        // Extract last position's hidden state → state.normed for LM head
        gpu.flush();
        let last_offset = ((seq_len - 1) as u64) * h as u64 * f;
        {
            let last_bytes = gpu.read_buffer_offset(&normed, last_offset, h as u64 * f);
            gpu.write_buffer(&self.state.normed, 0, &last_bytes);
        }

        // LM head
        self.dispatch_lm_head(gpu);

        self.seq_len = seq_len;

        // Greedy argmax
        let logits_bytes = gpu.read_buffer(&self.state.logits, self.config.vocab_size as u64 * 4);
        let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let (max_idx, _) = logits.iter().enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        let token = max_idx as u32;
        self.generated_tokens.push(token);
        token
    }

    /// Zero out all DeltaNet conv history and recurrent state buffers.
    /// Call before serial decode to ensure a clean start matching the batch prefill path.
    pub fn clear_deltanet_state(&self, gpu: &mut GpuContext) {
        let lnkh = self.linear_num_key_heads;
        let lkd  = self.linear_key_dim;
        let lnvh = self.linear_num_value_heads;
        let lvd  = self.linear_value_dim;
        let dn_total_ch = lnkh * lkd * 2 + lnvh * lvd;
        let hist_zeros  = vec![0u8; 3 * dn_total_ch as usize * 4];
        let state_zeros = vec![0u8; (lnvh * lkd * lvd) as usize * 4];
        for lin_idx in 0..self.state.deltanet_hist.len() {
            gpu.write_buffer(&self.state.deltanet_hist[lin_idx],  0, &hist_zeros);
            gpu.write_buffer(&self.state.deltanet_state[lin_idx], 0, &state_zeros);
        }
        gpu.flush();
    }

    /// Batched GPTQ prefill: process all N input tokens in one GPU submission.
    /// Only supports GPTQ INT4 weights with Q_GATED=true (Qwen3.5).
    /// Returns the first sampled decode token.
    pub fn prefill_gptq(&mut self, gpu: &mut GpuContext, input_ids: &[u32]) {
        assert!(!self.bf16_mode, "prefill_gptq requires GPTQ mode (use prefill() for bf16)");
        let seq_len = input_ids.len() as u32;
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let f = 4u64;
        let sl = seq_len as u64;

        log::info!("[prefill_gptq] seq_len={} hidden={} inter={}", seq_len, h, inter);

        // ── Embed all input tokens into a flat [seq_len, hidden] f32 buffer ──
        let residual = gpu.create_storage_buffer("pg_residual", sl * h as u64 * f);
        // Embed token by token into residual (reuse state.hidden as scratch per token)
        for (i, &tok) in input_ids.iter().enumerate() {
            // Non-chunked embedding path reads p_embed[0] directly — write the token id first.
            // Must flush before each iteration: wgpu::Queue::write_buffer is staged and all
            // pending writes are applied at submission time, so without a flush the last
            // write would overwrite p_embed for all iterations.
            gpu.write_buffer(&self.state.p_embed, 0, &tok.to_le_bytes());
            self.embedding(gpu, tok);
            gpu.copy_buffer_offset(
                &self.state.hidden, 0,
                &residual, i as u64 * h as u64 * f,
                h as u64 * f,
            );
            gpu.flush();
        }
        gpu.flush();

        // ── Allocate batched scratch buffers ──
        let normed  = gpu.create_storage_buffer("pg_normed",  sl * h as u64 * f);
        let q_raw   = gpu.create_storage_buffer("pg_q_raw",  sl * (nh * hd * 2) as u64 * f);
        let q_proj  = gpu.create_storage_buffer("pg_q_proj", sl * (nh * hd) as u64 * f);
        let q_gate  = gpu.create_storage_buffer("pg_q_gate", sl * (nh * hd) as u64 * f);
        let k_buf   = gpu.create_storage_buffer("pg_k",      sl * (nkv * hd) as u64 * f);
        let v_buf   = gpu.create_storage_buffer("pg_v",      sl * (nkv * hd) as u64 * f);
        let attn_out = gpu.create_storage_buffer("pg_attn",  sl * (nh * hd) as u64 * f);
        let o_out   = gpu.create_storage_buffer("pg_o",      sl * (nh * hd) as u64 * f);
        let gate_buf = gpu.create_storage_buffer("pg_gate",  sl * inter as u64 * f);
        let up_buf  = gpu.create_storage_buffer("pg_up",     sl * inter as u64 * f);
        let mlp_out = gpu.create_storage_buffer("pg_mlp",    sl * h as u64 * f);

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct NormP { n: u32, eps: f32, seq_len: u32, _pad: u32 }

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct GemmP { k: u32, n: u32, group_size: u32, _pad: u32 }

        let gs = self.quant_config.group_size;
        // Local params buffer for prefill dispatches — reused per layer with flush+write
        let pg_params = gpu.create_buffer("pg_params", 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        // Clone the shader src to avoid borrow conflicts inside the layer loop
        let batched_qknorm_gated_src = self.batched_qknorm_gated_src.clone();

        // ── DeltaNet scratch buffers (used only for hybrid linear-attn layers) ──
        let lnkh = self.linear_num_key_heads;
        let lkd  = self.linear_key_dim;
        let lnvh = self.linear_num_value_heads;
        let lvd  = self.linear_value_dim;
        let dn_total_ch = lnkh * lkd * 2 + lnvh * lvd;  // total Q+K+V channels
        let dn_z_n = lnvh * lvd;

        let dn_qkv_batch = gpu.create_storage_buffer("pg_dn_qkv", sl * dn_total_ch as u64 * f);
        let dn_z_batch   = gpu.create_storage_buffer("pg_dn_z",   sl * dn_z_n as u64 * f);
        let dn_out_batch = gpu.create_storage_buffer("pg_dn_out", sl * dn_z_n as u64 * f);

        // Separate params buffer for DeltaNet uniform (8 fields, 32 bytes)
        let pg_dn_params = gpu.create_buffer("pg_dn_params", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        // Zero all DeltaNet hist and state buffers so prefill always starts from a clean
        // initial condition regardless of leftover state from previous generate() calls.
        let hist_zeros  = vec![0u8; 3 * dn_total_ch as usize * 4];
        let state_zeros = vec![0u8; (lnvh * lkd * lvd) as usize * 4];
        for lin_idx in 0..self.state.deltanet_hist.len() {
            gpu.write_buffer(&self.state.deltanet_hist[lin_idx],  0, &hist_zeros);
            gpu.write_buffer(&self.state.deltanet_state[lin_idx], 0, &state_zeros);
        }
        gpu.flush();

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct DnP {
            num_heads: u32, key_dim: u32, value_dim: u32, total_channels: u32,
            eps: f32, hidden_size: u32, num_value_heads: u32, seq_len: u32,
        }

        for layer_idx in 0..self.config.num_hidden_layers as usize {
            let layer = &self.weights.layers[layer_idx];

            // ── Pre-attention norm ──
            gpu.flush();
            if layer_idx == 0 {
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&NormP {
                    n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
                gpu.dispatch("pg_norm0", shaders::BATCHED_RMSNORM_1PW, &[
                    gpu::bind(0, &residual), gpu::bind(1, &layer.input_layernorm),
                    gpu::bind(2, &normed),   gpu::bind(3, &pg_params),
                ], (seq_len, 1, 1));
            } else {
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&NormP {
                    n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
                gpu.dispatch("pg_addnorm", shaders::BATCHED_ADD_RMSNORM_1PW, &[
                    gpu::bind(0, &residual), gpu::bind(1, &mlp_out),
                    gpu::bind(2, &layer.input_layernorm), gpu::bind(3, &normed),
                    gpu::bind(4, &pg_params),
                ], (seq_len, 1, 1));
            }

            let gemm_shader = shaders::GPTQ_GEMM_4T;
            let wg_h = h.div_ceil(8);
            let is_sa = self.weights.self_attn_layers.contains(&layer_idx);

            if is_sa {
                let sa = layer.self_attn().expect("prefill_gptq: layer is_sa but no self_attn weights");
                let q_dim = nh * hd * 2; // Q_GATED: output [nh, hd*2]
                let kv_dim = nkv * hd;

                // ── Q proj: [seq, h] → [seq, nh*hd*2] ──
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: q_dim, group_size: gs, _pad: 0 }));
                let wg_n = q_dim.div_ceil(8);
                gpu.dispatch("pg_q", gemm_shader, &[
                    gpu::bind(0, &normed), gpu::bind(1, &sa.q_proj_qweight),
                    gpu::bind(2, &sa.q_proj_scales), gpu::bind(3, &q_raw),
                    gpu::bind(4, &pg_params),
                ], (wg_n, seq_len, 1));

                // ── K proj ──
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: kv_dim, group_size: gs, _pad: 0 }));
                let wg_kv = kv_dim.div_ceil(8);
                gpu.dispatch("pg_k", gemm_shader, &[
                    gpu::bind(0, &normed), gpu::bind(1, &sa.k_proj_qweight),
                    gpu::bind(2, &sa.k_proj_scales), gpu::bind(3, &k_buf),
                    gpu::bind(4, &pg_params),
                ], (wg_kv, seq_len, 1));

                // ── V proj ──
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: kv_dim, group_size: gs, _pad: 0 }));
                gpu.dispatch("pg_v", gemm_shader, &[
                    gpu::bind(0, &normed), gpu::bind(1, &sa.v_proj_qweight),
                    gpu::bind(2, &sa.v_proj_scales), gpu::bind(3, &v_buf),
                    gpu::bind(4, &pg_params),
                ], (wg_kv, seq_len, 1));

                // ── Batched Q_GATED qknorm + RoPE + KV cache write ──
                // batched_qknorm_params[layer_idx] has header+weights pre-loaded.
                // Write seq_len (4 bytes) at offset 16 (seq_len field in header).
                gpu.flush();
                gpu.write_buffer(&self.state.batched_qknorm_params[layer_idx], 16, &seq_len.to_le_bytes());
                gpu.dispatch("pg_qknorm", &batched_qknorm_gated_src, &[
                    gpu::bind(0, &q_raw),   // [seq, nh, hd*2]
                    gpu::bind(1, &q_proj),  // [seq, nh, hd] out
                    gpu::bind(2, &q_gate),  // [seq, nh, hd] out
                    gpu::bind(3, &k_buf),   // [seq, nkv, hd] in/out
                    gpu::bind(4, &v_buf),   // [seq, nkv, hd] in
                    gpu::bind(5, &self.state.k_cache[layer_idx]),
                    gpu::bind(6, &self.state.v_cache[layer_idx]),
                    gpu::bind(7, &self.state.batched_qknorm_params[layer_idx]),
                ], (nh + nkv, seq_len, 1));

                // ── Causal attention (batched) ──
                {
                    #[repr(C)]
                    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                    struct AttnP { seq_len: u32, head_dim: u32, num_kv_heads: u32, num_q_heads: u32, heads_per_kv: u32 }
                    gpu.flush();
                    gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&AttnP {
                        seq_len, head_dim: hd, num_kv_heads: nkv,
                        num_q_heads: nh, heads_per_kv: nh / nkv }));
                    gpu.dispatch("pg_attn", shaders::CAUSAL_ATTENTION_PREFILL, &[
                        gpu::bind(0, &q_proj), gpu::bind(1, &k_buf),
                        gpu::bind(2, &v_buf),  gpu::bind(3, &attn_out),
                        gpu::bind(4, &pg_params),
                    ], (nh, seq_len, 1));
                }

                // ── Sigmoid gate on attention output ──
                {
                    let n = seq_len * nh * hd;
                    gpu.flush();
                    gpu.write_buffer(&pg_params, 0, &n.to_le_bytes());
                    gpu.dispatch("pg_siggate", shaders::SIGMOID_MUL, &[
                        gpu::bind(0, &attn_out), gpu::bind(1, &q_gate),
                        gpu::bind(2, &o_out),    gpu::bind(3, &pg_params),
                    ], (n.div_ceil(256), 1, 1));
                }

                // ── O projection: [seq, nh*hd] → [seq, h] → mlp_out ──
                let o_h_dim = nh * hd;
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: o_h_dim, n: h, group_size: gs, _pad: 0 }));
                gpu.dispatch("pg_o", gemm_shader, &[
                    gpu::bind(0, &o_out), gpu::bind(1, &sa.o_proj_qweight),
                    gpu::bind(2, &sa.o_proj_scales), gpu::bind(3, &mlp_out),
                    gpu::bind(4, &pg_params),
                ], (wg_h, seq_len, 1));

            } else if let Some(la) = layer.linear_attn() {
                // ── DeltaNet batched prefill ──
                let lin_idx = (0..layer_idx)
                    .filter(|j| !self.weights.self_attn_layers.contains(j))
                    .count();

                // QKV projection: normed[seq, h] → dn_qkv_batch[seq, total_ch]
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: dn_total_ch, group_size: gs, _pad: 0 }));
                gpu.dispatch("pg_dn_qkv", gemm_shader, &[
                    gpu::bind(0, &normed), gpu::bind(1, &la.in_proj_qkv_qweight),
                    gpu::bind(2, &la.in_proj_qkv_scales), gpu::bind(3, &dn_qkv_batch),
                    gpu::bind(4, &pg_params),
                ], (dn_total_ch.div_ceil(8), seq_len, 1));

                // Z gate projection: normed[seq, h] → dn_z_batch[seq, dn_z_n]
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: dn_z_n, group_size: gs, _pad: 0 }));
                gpu.dispatch("pg_dn_z", gemm_shader, &[
                    gpu::bind(0, &normed), gpu::bind(1, &la.in_proj_z_qweight),
                    gpu::bind(2, &la.in_proj_z_scales), gpu::bind(3, &dn_z_batch),
                    gpu::bind(4, &pg_params),
                ], (dn_z_n.div_ceil(8), seq_len, 1));

                // DeltaNet recurrence + RMSNorm over all positions (serial inside shader)
                gpu.flush();
                gpu.write_buffer(&pg_dn_params, 0, bytemuck::bytes_of(&DnP {
                    num_heads: lnkh, key_dim: lkd, value_dim: lvd,
                    total_channels: dn_total_ch, eps: self.config.rms_norm_eps,
                    hidden_size: h, num_value_heads: lnvh, seq_len,
                }));
                gpu.dispatch("pg_dn_recur", shaders::BATCHED_DELTANET_PREFILL, &[
                    gpu::bind(0, &dn_qkv_batch),
                    gpu::bind(1, &self.state.deltanet_hist[lin_idx]),
                    gpu::bind(2, &la.conv1d_weight),
                    gpu::bind(3, &self.state.deltanet_state[lin_idx]),
                    gpu::bind(4, &dn_out_batch),
                    gpu::bind(5, &normed),   // hidden_batch [seq, h]
                    gpu::bind(6, &la.ab_weight),
                    gpu::bind(7, &la.a_log),
                    gpu::bind(8, &la.dt_bias),
                    gpu::bind(9, &la.norm_weight),
                    gpu::bind(10, &pg_dn_params),
                ], (lnkh, 1, 1));

                // Out projection: silu(dn_z_batch) * dn_out_batch → mlp_out[seq, h]
                gpu.flush();
                gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: dn_z_n, n: h, group_size: gs, _pad: 0 }));
                gpu.dispatch("pg_dn_out", shaders::FUSED_SILU_GPTQ_GEMM_4T, &[
                    gpu::bind(0, &dn_z_batch), gpu::bind(1, &dn_out_batch),
                    gpu::bind(2, &la.out_proj_qweight), gpu::bind(3, &la.out_proj_scales),
                    gpu::bind(4, &mlp_out), gpu::bind(5, &pg_params),
                ], (wg_h, seq_len, 1));
            }

            // ── Post-attention add+norm: residual += o_proj_out ──
            gpu.flush();
            gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&NormP {
                n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
            gpu.dispatch("pg_postnorm", shaders::BATCHED_ADD_RMSNORM_1PW, &[
                gpu::bind(0, &residual), gpu::bind(1, &mlp_out),
                gpu::bind(2, &layer.post_attn_layernorm), gpu::bind(3, &normed),
                gpu::bind(4, &pg_params),
            ], (seq_len, 1, 1));

            // ── MLP: gate + up (fused) ──
            let gemm_gate_wg = inter.div_ceil(8);
            // The fused gate+up shader only supports seq=1 (no batch dim).
            // For batched prefill, fall back to two separate GPTQ GEMMs for gate and up.
            gpu.flush();
            gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: inter, group_size: gs, _pad: 0 }));
            gpu.dispatch("pg_gate", gemm_shader, &[
                gpu::bind(0, &normed), gpu::bind(1, &layer.gate_proj_qweight),
                gpu::bind(2, &layer.gate_proj_scales), gpu::bind(3, &gate_buf),
                gpu::bind(4, &pg_params),
            ], (gemm_gate_wg, seq_len, 1));

            gpu.flush();
            gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: h, n: inter, group_size: gs, _pad: 0 }));
            gpu.dispatch("pg_up", gemm_shader, &[
                gpu::bind(0, &normed), gpu::bind(1, &layer.up_proj_qweight),
                gpu::bind(2, &layer.up_proj_scales), gpu::bind(3, &up_buf),
                gpu::bind(4, &pg_params),
            ], (gemm_gate_wg, seq_len, 1));

            // ── Fused SiLU(gate) × up × down: [seq, inter] → [seq, h] ──
            let fused_down_shader = shaders::FUSED_SILU_GPTQ_GEMM_4T;
            gpu.flush();
            gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&GemmP { k: inter, n: h, group_size: gs, _pad: 0 }));
            gpu.dispatch("pg_down", fused_down_shader, &[
                gpu::bind(0, &gate_buf), gpu::bind(1, &up_buf),
                gpu::bind(2, &layer.down_proj_qweight), gpu::bind(3, &layer.down_proj_scales),
                gpu::bind(4, &mlp_out), gpu::bind(5, &pg_params),
            ], (wg_h, seq_len, 1));

            if (layer_idx + 1) % 7 == 0 {
                log::info!("[prefill_gptq] layer {}/{}", layer_idx + 1, self.config.num_hidden_layers);
            }

        }

        // ── Final norm ──
        gpu.flush();
        {
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct NormP2 { n: u32, eps: f32, seq_len: u32, _pad: u32 }
            gpu.write_buffer(&pg_params, 0, bytemuck::bytes_of(&NormP2 {
                n: h, eps: self.config.rms_norm_eps, seq_len, _pad: 0 }));
            gpu.dispatch("pg_final_norm", shaders::BATCHED_ADD_RMSNORM_1PW, &[
                gpu::bind(0, &residual), gpu::bind(1, &mlp_out),
                gpu::bind(2, &self.weights.final_norm), gpu::bind(3, &normed),
                gpu::bind(4, &pg_params),
            ], (seq_len, 1, 1));
        }

        // ── Extract last position hidden state → state.normed for LM head ──
        gpu.flush();
        {
            let last_off = (seq_len - 1) as u64 * h as u64 * f;
            let last_bytes = gpu.read_buffer_offset(&normed, last_off, h as u64 * f);
            gpu.write_buffer(&self.state.normed, 0, &last_bytes);
        }

        // ── LM head ──
        self.dispatch_lm_head(gpu);

        self.seq_len = seq_len;
        log::info!("[prefill_gptq] done, seq_len={}", seq_len);
        // Caller must call sample_first_decode_token() to obtain the first decode token.
        // This allows feature-gated think injection before sampling.
    }

    /// Sample the first decode token from the current lm_head logit state.
    /// Must be called after prefill_gptq() / dispatch_lm_head() and any optional think injection.
    /// sample_token_gpu() already pushes the token and marks it seen.
    pub fn sample_first_decode_token(&mut self, gpu: &mut GpuContext) -> u32 {
        let token = self.sample_token_gpu(gpu);
        log::info!("[prefill_gptq] first_decode_token={}", token);
        token
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
