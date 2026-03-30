//! AsrPipeline: merged encoder + decoder on a single shared GPU context.
//!
//! Replaces the two-context approach (AsrEncoder owns one GpuContext,
//! AsrModel/load_asr_model owns another) with a unified pipeline that:
//!   1. Shares a single GpuContext between encoder and decoder
//!   2. Keeps encoder output on GPU (no readback between encoder → decoder)
//!   3. Pre-embeds prefix/suffix tokens at load time
//!   4. Assembles the full prefill buffer via GPU copies
//!   5. Runs batched prefill through the decoder in one pass
//!   6. Falls back to the existing zero-write decode loop for autoregressive generation

use std::path::Path;

use crate::asr_decoder::DecodeResult;
use crate::asr_encoder::AsrEncoder;
use crate::asr_model::AsrModel;
use crate::gpu::GpuContext;
use crate::weights::{ModelConfig, QuantConfig};

// ── Constants ────────────────────────────────────────────────────────────

/// Maximum prefill sequence length (prefix + encoder tokens + suffix).
/// 9 prefix + up to ~100 encoder tokens + 7 suffix = ~116, round up.
const MAX_PREFILL: u32 = 128;

/// Maximum autoregressive decode tokens.
const MAX_DECODE_TOKENS: u32 = 448;

/// Qwen3-ASR prompt token IDs (mirrors asr_decoder.rs)
const TOKEN_IM_START: u32 = 151644;
const TOKEN_IM_END: u32 = 151645;
const TOKEN_ENDOFTEXT: u32 = 151643;
const TOKEN_AUDIO_START: u32 = 151669;
const TOKEN_AUDIO_END: u32 = 151670;
const TOKEN_ASR_TEXT: u32 = 151704;

/// <|im_start|>system\n
const PREFIX_HEAD: &[u32] = &[TOKEN_IM_START, 8948, 198];
/// <|im_end|>\n<|im_start|>user\n<|audio_start|>
const PREFIX_TAIL: &[u32] = &[TOKEN_IM_END, 198, TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START];
/// <|audio_end|><|im_end|>\n<|im_start|>assistant\n<|asr_text|>
const SUFFIX_TOKENS: &[u32] = &[TOKEN_AUDIO_END, TOKEN_IM_END, 198, TOKEN_IM_START, 77091, 198, TOKEN_ASR_TEXT];

/// Number of fixed prefix tokens (PREFIX_HEAD + PREFIX_TAIL).
const PREFIX_LEN: u32 = (PREFIX_HEAD.len() + PREFIX_TAIL.len()) as u32; // 9
/// Number of fixed suffix tokens.
const SUFFIX_LEN: u32 = SUFFIX_TOKENS.len() as u32; // 7

// ── Const-specialized batched prefill shader builders ──────────────────
// Model dimensions baked as `const` (one-time at load). Only `seq_len` is
// runtime — read from a storage buffer (same as seq_counter pattern).
// Bindings: all storage (no uniform), same as single-token path.

/// Batched INT8 GEMM: input[seq_len, IN_DIM] × qweight[OUT_DIM, IN_DIM/4] → output[seq_len, OUT_DIM]
/// Dispatch: (OUT_DIM/32, seq_len, 1)
fn build_batched_int8_gemm(in_dim: u32, out_dim: u32, gs: u32) -> String {
    format!("\
const OUT_DIM: u32 = {out_dim}u;
const IN_DIM: u32 = {in_dim}u;
const PACKED_COLS: u32 = {pc}u;
const N_GROUPS: u32 = {ng}u;
const PACKED_PER_GROUP: u32 = {ppg}u;
const GROUP_SIZE: u32 = {gs}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> seq_counter: array<u32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let col = wg_id.x * 32u + lid.x;
    let row = wg_id.y;
    let seq_len = seq_counter[0];
    if (col >= OUT_DIM || row >= seq_len) {{ return; }}
    let w_row_off = col * PACKED_COLS;
    let sg_row_off = col * N_GROUPS;
    let in_base = row * IN_DIM;
    var sum: f32 = 0.0;
    for (var g: u32 = 0u; g < N_GROUPS; g++) {{
        let sb_idx = sg_row_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let group_start = g * PACKED_PER_GROUP;
        let input_base = in_base + g * GROUP_SIZE;
        for (var p: u32 = 0u; p < PACKED_PER_GROUP; p++) {{
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 4u;
            sum += (f32((packed) & 0xFFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 8u) & 0xFFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 16u) & 0xFFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 24u) & 0xFFu) * scale + bias) * input[ib + 3u];
        }}
    }}
    output[row * OUT_DIM + col] = sum;
}}", out_dim=out_dim, in_dim=in_dim, gs=gs, pc=in_dim/4, ng=in_dim/gs, ppg=gs/4)
}

/// Batched RMSNorm: input[seq_len, N] → output[seq_len, N]
/// Weight is bf16 packed. Dispatch: (seq_len, 1, 1)
fn build_batched_rmsnorm(hidden: u32, eps: f32) -> String {
    format!("\
const N: u32 = {n}u;
const EPS: f32 = {eps};
const HALF_N: u32 = {hn}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> wg_scratch: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x;
    let tid = lid.x;
    let base = row * N;
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < N) {{ let v = input[base + i]; sum_sq += v * v; i += 256u; }}
    wg_scratch[tid] = sum_sq;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{ if (tid < stride) {{ wg_scratch[tid] += wg_scratch[tid + stride]; }} workgroupBarrier(); stride >>= 1u; }}
    let rms = 1.0 / sqrt(wg_scratch[0] / f32(N) + EPS);
    i = tid;
    while (i < N) {{
        let w_packed = weight[i / 2u];
        let w = bitcast<f32>(((w_packed >> ((i & 1u) * 16u)) & 0xFFFFu) << 16u);
        output[base + i] = input[base + i] * rms * w;
        i += 256u;
    }}
}}", n=hidden, eps=eps, hn=hidden/2)
}

/// Batched Add+RMSNorm: residual[seq_len, N] += addend; output = rmsnorm(residual)
/// Dispatch: (seq_len, 1, 1)
fn build_batched_add_rmsnorm(hidden: u32, eps: f32) -> String {
    format!("\
const N: u32 = {n}u;
const EPS: f32 = {eps};
@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> wg_scratch: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wg_id.x;
    let tid = lid.x;
    let base = row * N;
    // Add residual
    var i = tid;
    while (i < N) {{ residual[base + i] += addend[base + i]; i += 256u; }}
    workgroupBarrier();
    // RMSNorm
    var sum_sq: f32 = 0.0;
    i = tid;
    while (i < N) {{ let v = residual[base + i]; sum_sq += v * v; i += 256u; }}
    wg_scratch[tid] = sum_sq;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{ if (tid < stride) {{ wg_scratch[tid] += wg_scratch[tid + stride]; }} workgroupBarrier(); stride >>= 1u; }}
    let rms = 1.0 / sqrt(wg_scratch[0] / f32(N) + EPS);
    i = tid;
    while (i < N) {{
        let w_packed = weight[i / 2u];
        let w = bitcast<f32>(((w_packed >> ((i & 1u) * 16u)) & 0xFFFFu) << 16u);
        output[base + i] = residual[base + i] * rms * w;
        i += 256u;
    }}
}}", n=hidden, eps=eps)
}

/// Batched SiLU(gate) × up: gate[seq_len, INTER] = SiLU(gate) * up
/// Dispatch: (ceil(INTER/256), seq_len, 1)
fn build_batched_silu_mul(inter: u32) -> String {
    format!("\
const INTER: u32 = {inter}u;
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let col = wg_id.x * 256u + lid.x;
    let row = wg_id.y;
    if (col >= INTER) {{ return; }}
    let idx = row * INTER + col;
    let x = gate[idx];
    gate[idx] = (x / (1.0 + exp(-x))) * up[idx];
}}", inter=inter)
}

/// Batched causal GQA attention for prefill.
/// Q from dense prefill_q, K/V from KV cache (written by qknorm shader).
/// Dispatch: (num_q_heads, seq_len, 1)
fn build_batched_causal_attn(nh: u32, nkv: u32, hd: u32) -> String {
    format!("\
const HEAD_DIM: u32 = {hd}u;
const NUM_Q_HEADS: u32 = {nh}u;
const NUM_KV_HEADS: u32 = {nkv}u;
const HEADS_PER_KV: u32 = {hpk}u;
@group(0) @binding(0) var<storage, read> q_proj: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> wg_score: array<f32, 256>;
var<workgroup> wg_reduce: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.x;
    let q_head = wg_id.x;
    let q_pos = wg_id.y;
    let kv_head = q_head / HEADS_PER_KV;
    if (q_head >= NUM_Q_HEADS) {{ return; }}
    let q_base = q_pos * NUM_Q_HEADS * HEAD_DIM + q_head * HEAD_DIM;
    let scale = 1.0 / sqrt(f32(HEAD_DIM));
    let causal_len = q_pos + 1u;
    // Phase 1: scores + max
    var local_max: f32 = -1e30;
    var j = tid;
    while (j < causal_len) {{
        let k_base = j * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
        var dot: f32 = 0.0;
        for (var d = 0u; d < HEAD_DIM; d++) {{ dot += q_proj[q_base + d] * k_cache[k_base + d]; }}
        let s = dot * scale;
        wg_score[j] = s;
        local_max = max(local_max, s);
        j += 256u;
    }}
    wg_reduce[tid] = local_max;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {{ if (tid < stride) {{ wg_reduce[tid] = max(wg_reduce[tid], wg_reduce[tid + stride]); }} workgroupBarrier(); stride >>= 1u; }}
    let max_score = wg_reduce[0];
    workgroupBarrier();
    // Phase 2: exp + sum
    var local_sum: f32 = 0.0;
    j = tid;
    while (j < causal_len) {{ let e = exp(wg_score[j] - max_score); wg_score[j] = e; local_sum += e; j += 256u; }}
    wg_reduce[tid] = local_sum;
    workgroupBarrier();
    stride = 128u;
    while (stride > 0u) {{ if (tid < stride) {{ wg_reduce[tid] += wg_reduce[tid + stride]; }} workgroupBarrier(); stride >>= 1u; }}
    let sum_exp = wg_reduce[0];
    workgroupBarrier();
    // Phase 3: weighted V
    let out_base = q_pos * NUM_Q_HEADS * HEAD_DIM + q_head * HEAD_DIM;
    var d = tid;
    while (d < HEAD_DIM) {{
        var wsum: f32 = 0.0;
        for (var jj = 0u; jj < causal_len; jj++) {{
            let v_base = jj * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
            wsum += (wg_score[jj] / sum_exp) * v_cache[v_base + d];
        }}
        output[out_base + d] = wsum;
        d += 256u;
    }}
}}", hd=hd, nh=nh, nkv=nkv, hpk=nh/nkv)
}

// ── Pipeline struct ─────────────────────────────────────────────────────

/// Merged encoder + decoder pipeline sharing a single GPU context.
///
/// Owns all weights, state buffers, and pre-computed embeddings needed
/// to go from raw mel spectrogram to decoded text in one `forward` call.
pub struct AsrPipeline {
    // ── Encoder (owns the shared GpuContext via encoder.gpu) ──
    pub encoder: AsrEncoder,

    // ── Decoder ──
    pub decoder: AsrModel,
    pub decoder_config: ModelConfig,
    pub decoder_quant: QuantConfig,

    // ── Pre-embedded prefix/suffix tokens (f32, on GPU) ──
    /// PREFIX_HEAD + PREFIX_TAIL embeddings: [PREFIX_LEN, hidden] f32
    prefix_embed_buf: wgpu::Buffer,
    /// SUFFIX_TOKENS embeddings: [SUFFIX_LEN, hidden] f32
    suffix_embed_buf: wgpu::Buffer,

    // ── Prefill buffers (allocated at MAX_PREFILL × hidden) ──
    /// Assembled input embeddings for batched prefill: [MAX_PREFILL, hidden] f32
    prefill_input: wgpu::Buffer,
    /// Residual stream for prefill: [MAX_PREFILL, hidden] f32
    prefill_residual: wgpu::Buffer,
    /// Normed activations: [MAX_PREFILL, hidden] f32
    prefill_normed: wgpu::Buffer,
    /// Q projection output: [MAX_PREFILL, q_dim] f32
    prefill_q: wgpu::Buffer,
    /// K projection output: [MAX_PREFILL, kv_dim] f32
    prefill_k: wgpu::Buffer,
    /// V projection output: [MAX_PREFILL, kv_dim] f32
    prefill_v: wgpu::Buffer,
    /// Attention output: [MAX_PREFILL, q_dim] f32
    prefill_attn_out: wgpu::Buffer,
    /// O projection output: [MAX_PREFILL, hidden] f32
    prefill_o_out: wgpu::Buffer,
    /// Gate MLP output: [MAX_PREFILL, intermediate] f32
    prefill_gate: wgpu::Buffer,
    /// Up MLP output: [MAX_PREFILL, intermediate] f32
    prefill_up: wgpu::Buffer,
    /// MLP output (after down proj): [MAX_PREFILL, hidden] f32
    prefill_mlp_out: wgpu::Buffer,
    /// Logits for last-token LM head: [vocab_size] f32
    prefill_logits: wgpu::Buffer,

    // ── Const-specialized batched shader sources ──
    s_gemm_q: String,
    s_gemm_kv: String,
    s_gemm_o: String,
    s_gemm_gate: String,
    s_gemm_up: String,
    s_gemm_down: String,
    s_batched_rmsnorm: String,
    s_batched_add_rmsnorm: String,
    s_batched_qknorm: String,
    s_batched_causal_attn: String,
    s_batched_silu_mul: String,

    /// Prefix KV already resident on GPU (skip re-prefill of prefix tokens).
    prefix_kv_cached: bool,
}

impl AsrPipeline {
    /// Load the full encoder + decoder pipeline onto a single GPU.
    ///
    /// `encoder_dir` — directory with encoder safetensors + config (audio_config).
    /// `decoder_dir` — directory with decoder safetensors + config (text_config).
    ///   (These may be the same directory for a unified Qwen3-ASR checkpoint.)
    pub fn load(model_dir: &Path) -> Self {
        log::info!("[asr-pipeline] loading merged encoder+decoder from {:?}", model_dir);
        let t0 = std::time::Instant::now();

        // ── 1. Create shared GpuContext ──
        let gpu = GpuContext::new();

        // ── 2. Load encoder weights (reuses AsrEncoder::load with shared GPU) ──
        let mut encoder = AsrEncoder::load(gpu, model_dir);
        // Take the gpu back out — AsrEncoder::load consumed it, but we need to
        // pass it around. The encoder stores it as `self.gpu`.
        // NOTE: We split-borrow through the pipeline struct instead.

        // ── 3. Load decoder config + weights ──
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("config.json")).expect("config.json"),
        )
        .expect("parse json");

        let text_cfg = if raw["thinker_config"]["text_config"].is_object() {
            &raw["thinker_config"]["text_config"]
        } else if raw["text_decoder"].is_object() {
            &raw["text_decoder"]
        } else if raw["text_config"].is_object() {
            &raw["text_config"]
        } else {
            &raw
        };
        let mut decoder_config: ModelConfig =
            serde_json::from_value(text_cfg.clone()).expect("parse ModelConfig");
        if decoder_config.partial_rotary_factor < 1.0
            && text_cfg.get("partial_rotary_factor").is_none()
        {
            decoder_config.partial_rotary_factor = 1.0;
        }

        let qcfg = raw
            .get("quantization_config")
            .or_else(|| raw.get("quantization"));
        let bits = qcfg.and_then(|q| q["bits"].as_u64()).unwrap_or(8) as u32;
        let group_size = qcfg.and_then(|q| q["group_size"].as_u64()).unwrap_or(64) as u32;
        log::info!(
            "[asr-pipeline] decoder: {} layers, hidden={}, heads={}, kv_heads={}, vocab={}, {}bit gs={}",
            decoder_config.num_hidden_layers, decoder_config.hidden_size,
            decoder_config.num_attention_heads, decoder_config.num_key_value_heads,
            decoder_config.vocab_size, bits, group_size
        );

        // Load decoder weights using the encoder's GPU (shared context)
        let (dec_weights, raw_norms) =
            crate::weights::load_weights_mlx(&encoder.gpu, model_dir, &decoder_config, bits);
        let decoder_quant = QuantConfig {
            bits,
            group_size,
            quant_method: "mlx".to_string(),
            sym: false,
        };

        let max_seq_len = MAX_PREFILL + MAX_DECODE_TOKENS;
        let mut decoder = AsrModel::new(
            &encoder.gpu,
            decoder_config.clone(),
            decoder_quant.clone(),
            dec_weights,
            max_seq_len,
        );

        // Initialize QK norm params
        for (i, norm) in raw_norms.layers.iter().enumerate() {
            if let Some((q, k)) = norm {
                decoder.init_qknorm_params(&encoder.gpu, i, q, k);
            }
        }

        // ── 4. Pre-embed prefix and suffix tokens ──
        let h = decoder_config.hidden_size;
        let hidden_bytes = h as u64 * 4;

        let prefix_tokens: Vec<u32> = PREFIX_HEAD
            .iter()
            .chain(PREFIX_TAIL.iter())
            .copied()
            .collect();
        let prefix_embed_buf = Self::pre_embed_tokens(
            &encoder.gpu,
            &decoder,
            &prefix_tokens,
            "prefix_embed",
        );

        let suffix_embed_buf = Self::pre_embed_tokens(
            &encoder.gpu,
            &decoder,
            SUFFIX_TOKENS,
            "suffix_embed",
        );

        log::info!(
            "[asr-pipeline] pre-embedded {} prefix + {} suffix tokens",
            prefix_tokens.len(),
            SUFFIX_TOKENS.len()
        );

        // ── 5. Allocate prefill buffers ──
        let f = 4u64; // sizeof(f32)
        let q_dim = (decoder_config.num_attention_heads * decoder_config.head_dim) as u64;
        let kv_dim =
            (decoder_config.num_key_value_heads * decoder_config.head_dim) as u64;
        let inter = decoder_config.intermediate_size as u64;
        let max_pf = MAX_PREFILL as u64;

        let prefill_input =
            encoder.gpu.create_storage_buffer("pf_input", max_pf * h as u64 * f);
        let prefill_residual =
            encoder.gpu.create_storage_buffer("pf_residual", max_pf * h as u64 * f);
        let prefill_normed =
            encoder.gpu.create_storage_buffer("pf_normed", max_pf * h as u64 * f);
        let prefill_q =
            encoder.gpu.create_storage_buffer("pf_q", max_pf * q_dim * f);
        let prefill_k =
            encoder.gpu.create_storage_buffer("pf_k", max_pf * kv_dim * f);
        let prefill_v =
            encoder.gpu.create_storage_buffer("pf_v", max_pf * kv_dim * f);
        let prefill_attn_out =
            encoder.gpu.create_storage_buffer("pf_attn", max_pf * q_dim * f);
        let prefill_o_out =
            encoder.gpu.create_storage_buffer("pf_o", max_pf * h as u64 * f);
        let prefill_gate =
            encoder.gpu.create_storage_buffer("pf_gate", max_pf * inter * f);
        let prefill_up =
            encoder.gpu.create_storage_buffer("pf_up", max_pf * inter * f);
        let prefill_mlp_out =
            encoder.gpu.create_storage_buffer("pf_mlp", max_pf * h as u64 * f);
        let prefill_logits = encoder
            .gpu
            .create_storage_buffer("pf_logits", decoder_config.vocab_size as u64 * f);

        // ── 6. Build const-specialized batched shader sources ──
        // These mirror the per-token shaders in AsrModel but operate on [seq_len, dim]
        // matrices instead of single vectors.
        let nh = decoder_config.num_attention_heads;
        let nkv = decoder_config.num_key_value_heads;
        let hd = decoder_config.head_dim;
        let gs = group_size;

        // TODO: Replace these placeholders with actual const-specialized builders
        // once the batched shader .wgsl files are finalized. These will be
        // analogous to build_int8_gemm_batched() etc. from asr_model.rs but
        // with MAX_PREFILL baked in.
        let s_gemm_q = build_batched_int8_gemm(h, nh * hd, gs);
        let s_gemm_kv = build_batched_int8_gemm(h, nkv * hd, gs);
        let s_gemm_o = build_batched_int8_gemm(nh * hd, h, gs);
        let s_gemm_gate = build_batched_int8_gemm(h, inter as u32, gs);
        let s_gemm_up = build_batched_int8_gemm(h, inter as u32, gs);
        let s_gemm_down = build_batched_int8_gemm(inter as u32, h, gs);
        let s_batched_rmsnorm = build_batched_rmsnorm(h, decoder_config.rms_norm_eps);
        let s_batched_add_rmsnorm = build_batched_add_rmsnorm(h, decoder_config.rms_norm_eps);
        // QKNorm+RoPE uses existing static shader (handles variable seq via dispatch dims)
        let s_batched_qknorm = include_str!("shaders/batched_qknorm_rope.wgsl").to_string();
        let s_batched_causal_attn = build_batched_causal_attn(nh, nkv, hd);
        let s_batched_silu_mul = build_batched_silu_mul(inter as u32);

        // ── Compile all batched prefill shaders now (deterministic, no lazy compilation) ──
        let t_shaders = std::time::Instant::now();
        encoder.gpu.ensure_pipeline("pf_q", &s_gemm_q);
        encoder.gpu.ensure_pipeline("pf_k", &s_gemm_kv);
        encoder.gpu.ensure_pipeline("pf_v", &s_gemm_kv); // same source as K
        encoder.gpu.ensure_pipeline("pf_o", &s_gemm_o);
        encoder.gpu.ensure_pipeline("pf_gate", &s_gemm_gate);
        encoder.gpu.ensure_pipeline("pf_up", &s_gemm_up); // same source as gate
        encoder.gpu.ensure_pipeline("pf_down", &s_gemm_down);
        encoder.gpu.ensure_pipeline("pf_norm", &s_batched_rmsnorm);
        encoder.gpu.ensure_pipeline("pf_addnorm", &s_batched_add_rmsnorm);
        encoder.gpu.ensure_pipeline("pf_postnorm", &s_batched_add_rmsnorm); // same source
        encoder.gpu.ensure_pipeline("pf_final_norm", &s_batched_add_rmsnorm); // same source
        encoder.gpu.ensure_pipeline("pf_qknorm", &s_batched_qknorm);
        encoder.gpu.ensure_pipeline("pf_attn", &s_batched_causal_attn);
        encoder.gpu.ensure_pipeline("pf_silu", &s_batched_silu_mul);
        // Also compile the decoder's LM head shaders used after prefill
        for (ci, shader) in decoder.s_lm_head.iter().enumerate() {
            encoder.gpu.ensure_pipeline(&format!("pf_lmh_{ci}"), shader);
        }
        encoder.gpu.ensure_pipeline("asr_argmax_const", &decoder.s_argmax);
        // Also pre-compile encoder shaders (names must match dispatch calls in asr_encoder.rs)
        encoder.gpu.ensure_pipeline("bf16_gemm", include_str!("shaders/bf16_gemm.wgsl"));
        encoder.gpu.ensure_pipeline("layernorm", include_str!("shaders/layernorm.wgsl"));
        encoder.gpu.ensure_pipeline("gelu_mul", include_str!("shaders/gelu_mul.wgsl"));
        encoder.gpu.ensure_pipeline("qwen_asr_bidir_attn", include_str!("shaders/qwen_asr_bidir_attn.wgsl"));
        encoder.gpu.ensure_pipeline("add", include_str!("shaders/add.wgsl"));
        // Conv stem shaders (runtime-generated, need model dims)
        let conv1_shader = crate::asr_encoder::build_conv2d_gelu_shader(1, 480);
        let conv23_shader = crate::asr_encoder::build_conv2d_gelu_shader(480, 480);
        let proj_shader = crate::asr_encoder::build_reshape_proj_pe_shader();
        encoder.gpu.ensure_pipeline("conv_stem_1", &conv1_shader);
        encoder.gpu.ensure_pipeline("conv_stem_2", &conv23_shader);
        encoder.gpu.ensure_pipeline("conv_stem_3", &conv23_shader);
        encoder.gpu.ensure_pipeline("conv_proj_pe", &proj_shader);
        log::info!("[asr-pipeline] shaders compiled in {}ms", t_shaders.elapsed().as_millis());

        let load_ms = t0.elapsed().as_millis();
        log::info!("[asr-pipeline] loaded in {}ms (weights + shaders + embeds)", load_ms);

        let mut pipeline = Self {
            encoder,
            decoder,
            decoder_config: decoder_config.clone(),
            decoder_quant,
            prefix_embed_buf,
            suffix_embed_buf,
            prefill_input,
            prefill_residual,
            prefill_normed,
            prefill_q,
            prefill_k,
            prefill_v,
            prefill_attn_out,
            prefill_o_out,
            prefill_gate,
            prefill_up,
            prefill_mlp_out,
            prefill_logits,
            s_gemm_q,
            s_gemm_kv,
            s_gemm_o,
            s_gemm_gate,
            s_gemm_up,
            s_gemm_down,
            s_batched_rmsnorm,
            s_batched_add_rmsnorm,
            s_batched_qknorm,
            s_batched_causal_attn,
            s_batched_silu_mul,
            prefix_kv_cached: false,
        };
        pipeline.fill_prefix_suffix_embeds();
        pipeline
    }

    /// Pre-embed a list of token IDs into a single GPU buffer [len, hidden] f32.
    /// Uses the decoder's INT8/INT4 embedding lookup, reads back, uploads as contiguous f32.
    fn pre_embed_tokens(
        // Borrow gpu from encoder since we haven't constructed Self yet
        gpu: &GpuContext,
        decoder: &AsrModel,
        tokens: &[u32],
        label: &str,
    ) -> wgpu::Buffer {
        let h = decoder.config.hidden_size as usize;
        let mut all_embeds: Vec<u8> = Vec::with_capacity(tokens.len() * h * 4);

        // We need a mutable GpuContext for dispatch — but at load time we can
        // create a temporary command encoder. For now, use a simpler approach:
        // do a CPU-side dequant of the embedding table for the small number of
        // prefix/suffix tokens (~16 total).
        //
        // TODO: Once the pipeline is fully wired, use the GPU embedding shader
        // with a temporary mutable borrow. For now, read the quantized embedding
        // table, dequant the needed rows on CPU, and upload the result.
        //
        // This is a one-time cost at load (~16 tokens) so CPU dequant is fine.

        // Placeholder: allocate zero buffer (will be filled by GPU dequant at load)
        let buf = gpu.create_storage_buffer(
            label,
            (tokens.len() * h) as u64 * 4,
        );
        // NOTE: Actual embedding will be done in load() after we have &mut GpuContext.
        // The caller (load) will fill this buffer with proper embeddings.
        buf
    }

    /// CPU-side dequant of prefix/suffix embeddings. One-time cost at load (~16 tokens).
    fn fill_prefix_suffix_embeds(&mut self) {
        let h = self.decoder_config.hidden_size as usize;
        let gs = self.decoder_quant.group_size as usize;
        let bits = self.decoder_quant.bits;
        let gpu = &mut self.encoder.gpu;

        // Read quantized embedding table from GPU
        let embed_buf = if !self.decoder.weights.embed_chunks.is_empty() {
            &self.decoder.weights.embed_chunks[0]
        } else {
            &self.decoder.weights.embed_tokens
        };
        let sc_buf = self.decoder.weights.mlx_embed_scales.as_ref().unwrap();
        let bi_buf = self.decoder.weights.mlx_embed_biases.as_ref().unwrap();

        let vals_per_u32 = if bits == 8 { 4 } else { 8 };
        let packed_cols = h / vals_per_u32;
        let n_groups = h / gs;

        // Read buffers to CPU for dequant
        let vocab = self.decoder_config.vocab_size as usize;
        let qw_bytes = gpu.read_buffer(embed_buf, (vocab * packed_cols) as u64 * 4);
        let qw: &[u32] = bytemuck::cast_slice(&qw_bytes);
        let sc_bytes = gpu.read_buffer(sc_buf, (vocab * n_groups) as u64 * 2);
        let sc_u16: &[u16] = bytemuck::cast_slice(&sc_bytes);
        let bi_bytes = gpu.read_buffer(bi_buf, (vocab * n_groups) as u64 * 2);
        let bi_u16: &[u16] = bytemuck::cast_slice(&bi_bytes);

        let dequant_row = |tok: u32| -> Vec<f32> {
            let tok = tok as usize;
            let mut row = vec![0.0f32; h];
            let row_off = tok * packed_cols;
            let sg_off = tok * n_groups;
            for g in 0..n_groups {
                let scale = f32::from_bits((sc_u16[sg_off + g] as u32) << 16);
                let bias = f32::from_bits((bi_u16[sg_off + g] as u32) << 16);
                for p in 0..(gs / vals_per_u32) {
                    let packed = qw[row_off + g * (gs / vals_per_u32) + p];
                    for b in 0..vals_per_u32 {
                        let shift = if bits == 8 { b * 8 } else { b * 4 };
                        let mask = if bits == 8 { 0xFF } else { 0xF };
                        let val = ((packed >> shift) & mask) as f32;
                        row[g * gs + p * vals_per_u32 + b] = val * scale + bias;
                    }
                }
            }
            row
        };

        // Dequant prefix tokens
        let prefix_tokens: Vec<u32> = PREFIX_HEAD.iter().chain(PREFIX_TAIL.iter()).copied().collect();
        let mut prefix_data: Vec<f32> = Vec::with_capacity(prefix_tokens.len() * h);
        for &tok in &prefix_tokens {
            prefix_data.extend_from_slice(&dequant_row(tok));
        }
        gpu.write_buffer(&self.prefix_embed_buf, 0, bytemuck::cast_slice(&prefix_data));

        // Dequant suffix tokens
        let mut suffix_data: Vec<f32> = Vec::with_capacity(SUFFIX_TOKENS.len() * h);
        for &tok in SUFFIX_TOKENS {
            suffix_data.extend_from_slice(&dequant_row(tok));
        }
        gpu.write_buffer(&self.suffix_embed_buf, 0, bytemuck::cast_slice(&suffix_data));

        // Debug: log first few embedding values
        let fp: Vec<String> = prefix_data.iter().take(4).map(|v| format!("{:.4}", v)).collect();
        log::info!("[asr-pipeline] pre-embedded {} prefix + {} suffix tokens, embed[0..4]=[{}]",
            prefix_tokens.len(), SUFFIX_TOKENS.len(), fp.join(" "));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Forward: mel → text tokens
    // ═══════════════════════════════════════════════════════════════════════

    /// Full pipeline: mel spectrogram → decoded token IDs.
    ///
    /// 1. Conv stem (GPU)
    /// 2. Encoder transformer (GPU)
    /// 3. Assemble prefill buffer on GPU (prefix_embed + encoder_out + suffix_embed)
    /// 4. Batched decoder prefill (GPU)
    /// 5. Autoregressive zero-write decode loop (GPU)
    pub fn forward(&mut self, mel: &[f32], mel_frames: u32) -> DecodeResult {
        let t0 = std::time::Instant::now();
        let gpu = &mut self.encoder.gpu;
        gpu.invalidate_bind_groups();

        let h = self.decoder_config.hidden_size;

        // ── 1. Conv stem on GPU ──
        let (conv_out_buf, enc_tokens) = self.encoder_conv_stem(mel, mel_frames);
        let conv_ms = t0.elapsed().as_millis();

        // ── 2. Encoder transformer on GPU ──
        let t1 = std::time::Instant::now();
        let encoder_out_buf = self.encoder_transformer(&conv_out_buf, enc_tokens);
        let enc_ms = t1.elapsed().as_millis();
        log::info!(
            "[asr-pipeline] encoder: conv={}ms, transformer={}ms, {} tokens",
            conv_ms, enc_ms, enc_tokens
        );

        // ── 3. Assemble prefill buffer on GPU ──
        let actual_len = PREFIX_LEN + enc_tokens + SUFFIX_LEN;
        assert!(
            actual_len <= MAX_PREFILL,
            "prefill length {} exceeds MAX_PREFILL {}",
            actual_len,
            MAX_PREFILL
        );
        self.assemble_prefill_buffer(&encoder_out_buf, enc_tokens);

        // ── 4. Batched decoder prefill ──
        let t2 = std::time::Instant::now();
        let first_token = self.prefill_batched(actual_len);
        let prefill_ms = t2.elapsed().as_millis();
        log::info!(
            "[asr-pipeline] prefill: {}ms for {} tokens, first_decode=0x{:x}",
            prefill_ms, actual_len, first_token
        );

        // Early exit: first token is EOS → no speech
        if first_token == TOKEN_ENDOFTEXT || first_token == TOKEN_IM_END {
            let prefill_total_ms = t0.elapsed().as_millis();
            return DecodeResult::NoSpeech {
                top3: vec![(first_token, 0.0)],
                prefill_ms: prefill_total_ms,
            };
        }

        // ── 5. Autoregressive zero-write decode loop ──
        let t3 = std::time::Instant::now();
        let gpu = &mut self.encoder.gpu;

        self.decoder.init_decode(gpu, first_token, actual_len);

        let eos_check_interval = 4u32;
        let mut n_generated = 0u32;

        loop {
            let batch = eos_check_interval.min(MAX_DECODE_TOKENS - n_generated);
            for _ in 0..batch {
                self.decoder.forward_zero_write(gpu);
                gpu.flush();
            }
            n_generated += batch;

            gpu.flush();
            if self.decoder.check_eos(gpu) || n_generated >= MAX_DECODE_TOKENS {
                break;
            }
        }

        let decode_ms = t3.elapsed().as_millis();
        let ring_tokens = self.decoder.read_generated_tokens(gpu);

        // ── Build output ──
        const MIN_TOKENS_BEFORE_EOS: usize = 5;
        let mut token_ids = Vec::with_capacity(ring_tokens.len() + 1);
        if first_token != TOKEN_ASR_TEXT {
            token_ids.push(first_token);
        }
        for (i, &t) in ring_tokens.iter().enumerate() {
            let is_eos = t == TOKEN_ENDOFTEXT || t == TOKEN_IM_END;
            if is_eos && i >= MIN_TOKENS_BEFORE_EOS {
                break;
            }
            let is_special = t >= 151000 || t == TOKEN_ASR_TEXT;
            if !is_eos && !is_special {
                token_ids.push(t);
            }
        }

        // Read logit values
        let logit_bytes =
            gpu.read_buffer(&self.decoder.state.logit_ring, ring_tokens.len() as u64 * 4);
        let logit_vals: &[f32] = bytemuck::cast_slice(&logit_bytes);
        let raw_ring: Vec<(u32, f32)> = ring_tokens
            .iter()
            .zip(logit_vals.iter())
            .map(|(&t, &l)| (t, l))
            .collect();

        let total_ms = t0.elapsed().as_millis();
        log::info!(
            "[asr-pipeline] total={}ms (enc={}ms, prefill={}ms, decode={}ms) → {} tokens",
            total_ms,
            conv_ms + enc_ms,
            prefill_ms,
            decode_ms,
            token_ids.len()
        );

        DecodeResult::Speech {
            token_ids,
            raw_ring,
            first_token: (first_token, 0.0),
            prefill_ms: t0.elapsed().as_millis() - decode_ms,
            decode_ms,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Encoder stages (reuse AsrEncoder logic, keep output on GPU)
    // ═══════════════════════════════════════════════════════════════════════

    /// Run conv stem on GPU. Returns (output_buffer, n_tokens).
    /// The output buffer is [n_tokens, d_model] f32, kept on GPU.
    fn encoder_conv_stem(&mut self, mel: &[f32], mel_frames: u32) -> (wgpu::Buffer, u32) {
        // Delegate to the encoder's conv_stem_gpu but keep the result on GPU.
        // For now, use the existing forward_mel which reads back to CPU,
        // then re-upload. This will be optimized to stay on-GPU once the
        // encoder is refactored to expose GPU-resident buffers.
        //
        // TODO: Refactor AsrEncoder to expose a `conv_stem_gpu_buf` that
        // returns a wgpu::Buffer instead of Vec<f32>.
        let (conv_out_cpu, n_tokens, _conv_ms, _enc_ms) =
            self.encoder.forward_mel(mel, mel_frames);

        // For the pipeline, we only need the conv stem output (pre-transformer).
        // But forward_mel runs both conv+transformer. We'll need to split this.
        // For now, use the full encoder output as our "encoder_out" and skip
        // the separate transformer step in forward().
        let buf = self
            .encoder
            .gpu
            .upload_buffer("enc_out", bytemuck::cast_slice(&conv_out_cpu));
        (buf, n_tokens)
    }

    /// Run encoder transformer on GPU. Input is conv stem output on GPU.
    /// Returns encoder output buffer [n_tokens, output_dim] f32, on GPU.
    fn encoder_transformer(
        &mut self,
        _conv_out_buf: &wgpu::Buffer,
        _n_tokens: u32,
    ) -> wgpu::Buffer {
        // TODO: Refactor AsrEncoder::forward to accept a GPU buffer and return
        // a GPU buffer, avoiding the CPU round-trip. For now, encoder_conv_stem
        // already ran the full encoder (conv + transformer) via forward_mel,
        // so this is a no-op — the buffer already contains the final output.
        //
        // When we split conv stem from transformer in AsrEncoder, this method
        // will dispatch the transformer layers keeping data on GPU.
        _conv_out_buf.clone()
        // NOTE: wgpu::Buffer doesn't implement Clone. In the real implementation,
        // we'd return a reference or keep the buffer in a field. For the skeleton,
        // just re-upload or pass through. This will be restructured.
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Prefill buffer assembly (all GPU-to-GPU copies)
    // ═══════════════════════════════════════════════════════════════════════

    /// Assemble the prefill input buffer on GPU:
    /// ```text
    /// [0 .. PREFIX_LEN*h)           = prefix_embed_buf
    /// [PREFIX_LEN*h .. (PFX+N)*h)   = encoder_output_buf
    /// [(PFX+N)*h .. (PFX+N+SFX)*h)  = suffix_embed_buf
    /// [(PFX+N+SFX)*h .. MAX_PF*h)   = zeros
    /// ```
    fn assemble_prefill_buffer(
        &mut self,
        encoder_out_buf: &wgpu::Buffer,
        enc_tokens: u32,
    ) {
        let h = self.decoder_config.hidden_size as u64;
        let f: u64 = 4; // sizeof(f32)
        let gpu = &mut self.encoder.gpu;

        let prefix_bytes = PREFIX_LEN as u64 * h * f;
        let enc_bytes = enc_tokens as u64 * h * f;
        let suffix_bytes = SUFFIX_LEN as u64 * h * f;
        let actual_bytes = (PREFIX_LEN as u64 + enc_tokens as u64 + SUFFIX_LEN as u64) * h * f;

        // Copy prefix embeddings → prefill_input[0..]
        gpu.copy_buffer_offset(
            &self.prefix_embed_buf, 0,
            &self.prefill_input, 0,
            prefix_bytes,
        );

        // Copy encoder output → prefill_input[prefix_bytes..]
        gpu.copy_buffer_offset(
            encoder_out_buf, 0,
            &self.prefill_input, prefix_bytes,
            enc_bytes,
        );

        // Copy suffix embeddings → prefill_input[prefix_bytes + enc_bytes..]
        gpu.copy_buffer_offset(
            &self.suffix_embed_buf, 0,
            &self.prefill_input, prefix_bytes + enc_bytes,
            suffix_bytes,
        );

        // Zero the rest of the buffer (padding beyond actual sequence)
        gpu.clear_buffer_range(&self.prefill_input, actual_bytes);

        // Initial residual = copy of input
        gpu.copy_buffer_offset(
            &self.prefill_input, 0,
            &self.prefill_residual, 0,
            actual_bytes,
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Batched prefill through decoder layers
    // ═══════════════════════════════════════════════════════════════════════

    /// Batched prefill: process `actual_len` tokens through all decoder layers
    /// in parallel (GEMM, not matvec). Returns the first decode token (argmax
    /// of logits at position actual_len-1).
    fn prefill_batched(&mut self, actual_len: u32) -> u32 {
        use crate::gpu::bind;
        let h = self.decoder_config.hidden_size;
        let nh = self.decoder_config.num_attention_heads;
        let nkv = self.decoder_config.num_key_value_heads;
        let hd = self.decoder_config.head_dim;
        let inter = self.decoder_config.intermediate_size;
        let nl = self.decoder_config.num_hidden_layers as usize;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let gpu = &mut self.encoder.gpu;

        // Write seq_len to seq_counter — batched GEMM shaders read it
        gpu.write_buffer(&self.decoder.state.seq_counter, 0,
            bytemuck::cast_slice(&[actual_len]));

        // Helper: dispatch batched GEMM (dims baked as const, seq_len from seq_counter)
        macro_rules! gemm {
            ($name:expr, $shader:expr, $input:expr, $qw:expr, $sc:expr, $bi:expr, $output:expr, $out_d:expr) => {{
                gpu.dispatch($name, $shader, &[
                    bind(0, $input), bind(1, $qw), bind(2, $sc), bind(3, $bi),
                    bind(4, $output), bind(5, &self.decoder.state.seq_counter),
                ], ($out_d.div_ceil(32), actual_len, 1));
            }};
        }

        for layer_idx in 0..nl {
            let layer = &self.decoder.weights.layers[layer_idx];
            let biases = &self.decoder.weights.mlx_biases[layer_idx];

            // 1. Norm (dims baked as const, no params needed)
            if layer_idx == 0 {
                gpu.dispatch("pf_norm", &self.s_batched_rmsnorm, &[
                    bind(0, &self.prefill_residual),
                    bind(1, &layer.input_layernorm),
                    bind(2, &self.prefill_normed),
                ], (actual_len, 1, 1));
            } else {
                gpu.dispatch("pf_addnorm", &self.s_batched_add_rmsnorm, &[
                    bind(0, &self.prefill_residual),
                    bind(1, &self.prefill_mlp_out),
                    bind(2, &layer.input_layernorm),
                    bind(3, &self.prefill_normed),
                ], (actual_len, 1, 1));
            }

            // Debug: check norm output at layer 0
            if layer_idx == 0 {
                gpu.flush();
                let norm_bytes = gpu.read_buffer(&self.prefill_normed, 16);
                let norm_vals: &[f32] = bytemuck::cast_slice(&norm_bytes);
                let res_bytes = gpu.read_buffer(&self.prefill_residual, 16);
                let res_vals: &[f32] = bytemuck::cast_slice(&res_bytes);
                log::info!("[pf-debug] layer 0: residual[0..4]={:?} normed[0..4]={:?}",
                    &res_vals[..4], &norm_vals[..4]);
            }

            // 2. QKV projections (batched GEMM, dims const-specialized)
            if let Some(sa) = layer.self_attn() {
                gemm!("pf_q", &self.s_gemm_q, &self.prefill_normed,
                    &sa.q_proj_qweight, &sa.q_proj_scales, &biases[0],
                    &self.prefill_q, q_dim);
                gemm!("pf_k", &self.s_gemm_kv, &self.prefill_normed,
                    &sa.k_proj_qweight, &sa.k_proj_scales, &biases[1],
                    &self.prefill_k, kv_dim);
                gemm!("pf_v", &self.s_gemm_kv, &self.prefill_normed,
                    &sa.v_proj_qweight, &sa.v_proj_scales, &biases[2],
                    &self.prefill_v, kv_dim);

                // 3. QKNorm + RoPE + KV cache write
                gpu.dispatch("pf_qknorm", &self.s_batched_qknorm, &[
                    bind(0, &self.prefill_q),
                    bind(1, &self.prefill_k),
                    bind(2, &self.prefill_v),
                    bind(3, &self.decoder.state.k_cache[layer_idx]),
                    bind(4, &self.decoder.state.v_cache[layer_idx]),
                    bind(5, &self.decoder.state.qknorm_params[layer_idx]),
                ], ((nh + nkv), actual_len, 1));

                // 4. Causal attention (dims const-specialized)
                gpu.dispatch("pf_attn", &self.s_batched_causal_attn, &[
                    bind(0, &self.prefill_q),
                    bind(1, &self.decoder.state.k_cache[layer_idx]),
                    bind(2, &self.decoder.state.v_cache[layer_idx]),
                    bind(3, &self.prefill_attn_out),
                ], (nh, actual_len, 1));

                // 5. O projection
                gemm!("pf_o", &self.s_gemm_o, &self.prefill_attn_out,
                    &sa.o_proj_qweight, &sa.o_proj_scales, &biases[3],
                    &self.prefill_o_out, h);
            }

            // 6. Post-attention add+norm
            gpu.dispatch("pf_postnorm", &self.s_batched_add_rmsnorm, &[
                bind(0, &self.prefill_residual),
                bind(1, &self.prefill_o_out),
                bind(2, &layer.post_attn_layernorm),
                bind(3, &self.prefill_normed),
            ], (actual_len, 1, 1));

            // 7. MLP
            gemm!("pf_gate", &self.s_gemm_gate, &self.prefill_normed,
                &layer.gate_proj_qweight, &layer.gate_proj_scales, &biases[4],
                &self.prefill_gate, inter);
            gemm!("pf_up", &self.s_gemm_up, &self.prefill_normed,
                &layer.up_proj_qweight, &layer.up_proj_scales, &biases[5],
                &self.prefill_up, inter);

            // SiLU(gate) × up (dims const-specialized)
            gpu.dispatch("pf_silu", &self.s_batched_silu_mul, &[
                bind(0, &self.prefill_gate),
                bind(1, &self.prefill_up),
            ], (inter.div_ceil(256), actual_len, 1));

            // Down projection
            gemm!("pf_down", &self.s_gemm_down, &self.prefill_gate,
                &layer.down_proj_qweight, &layer.down_proj_scales, &biases[6],
                &self.prefill_mlp_out, h);
        }

        // 8. Final add+norm
        gpu.dispatch("pf_final_norm", &self.s_batched_add_rmsnorm, &[
            bind(0, &self.prefill_residual),
            bind(1, &self.prefill_mlp_out),
            bind(2, &self.decoder.weights.final_norm),
            bind(3, &self.prefill_normed),
        ], (actual_len, 1, 1));

        // 9. LM head on last token
        let last_offset = (actual_len - 1) as u64 * h as u64 * 4;
        gpu.copy_buffer_offset(
            &self.prefill_normed, last_offset,
            &self.decoder.state.normed, 0,
            h as u64 * 4,
        );

        let sc = self.decoder.weights.mlx_embed_scales.as_ref().unwrap();
        let bi = self.decoder.weights.mlx_embed_biases.as_ref().unwrap();
        let chunks = if !self.decoder.weights.embed_chunks.is_empty() {
            &self.decoder.weights.embed_chunks[..]
        } else {
            std::slice::from_ref(&self.decoder.weights.embed_tokens)
        };
        let cs = if !self.decoder.weights.embed_chunks.is_empty() {
            self.decoder.weights.embed_chunk_size
        } else {
            self.decoder_config.vocab_size
        };

        for (ci, (chunk, shader)) in chunks.iter().zip(self.decoder.s_lm_head.iter()).enumerate() {
            let n = ((ci as u32 + 1) * cs).min(self.decoder_config.vocab_size) - ci as u32 * cs;
            gpu.dispatch(&format!("pf_lmh_{ci}"), shader, &[
                bind(0, &self.decoder.state.normed),
                bind(1, chunk), bind(2, sc), bind(3, bi),
                bind(4, &self.decoder.state.logits),
            ], (n.div_ceil(32), 1, 1));
        }

        // ── 10. CPU argmax on logits to get first decode token ──
        gpu.flush();
        let logits_bytes =
            gpu.read_buffer(&self.decoder.state.logits, self.decoder_config.vocab_size as u64 * 4);
        let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });

        // Set decoder seq_len so the autoregressive loop starts at the right position
        self.decoder.seq_len = actual_len;
        self.decoder.generated_tokens.clear();
        self.decoder.generated_tokens.push(max_idx as u32);

        max_idx as u32
    }
}
