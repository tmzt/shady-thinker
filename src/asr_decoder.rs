//! GPU-accelerated Qwen3-ASR decoder (bf16 causal LLM).
//!
//! Takes encoder output embeddings + prompt tokens, runs prefill + autoregressive decode.
//! Weights are bf16, computation in f32 (same as encoder).

use std::path::{Path, PathBuf};
use safetensors::SafeTensors;
use crate::gpu::GpuContext;
use crate::model::Model;
use crate::weights::{ModelConfig, QuantConfig};

const INT4_EMBEDDING_MLX_SRC: &str = include_str!("shaders/int4_embedding_mlx.wgsl");

/// Load a bf16 ASR decoder model ready for inference.
/// Load ASR decoder model onto an existing GPU context.
/// Handles nested ASR config and bf16/MLX-INT4/INT4-runtime formats.
pub fn load_model_on_gpu(gpu: &GpuContext, model_dir: &Path, max_seq_len: u32) -> Model {
    log::info!("[asr-decoder] loading model from {:?}", model_dir);

    // Parse config — handle ASR nesting
    let raw: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("config.json")).expect("config.json")
    ).expect("parse json");

    let text_cfg = if raw["thinker_config"]["text_config"].is_object() {
        &raw["thinker_config"]["text_config"]
    } else if raw["text_decoder"].is_object() {
        &raw["text_decoder"]  // MLX format
    } else if raw["text_config"].is_object() {
        &raw["text_config"]
    } else {
        &raw
    };
    let mut config: ModelConfig = serde_json::from_value(text_cfg.clone()).expect("parse ModelConfig");
    if config.partial_rotary_factor < 1.0 && !text_cfg.get("partial_rotary_factor").is_some() {
        config.partial_rotary_factor = 1.0;
    }
    log::info!("[asr-decoder] config: {} layers, hidden={}, heads={}, kv_heads={}, vocab={}, rope_theta={}, head_dim={}, partial_rot={}",
        config.num_hidden_layers, config.hidden_size,
        config.num_attention_heads, config.num_key_value_heads, config.vocab_size,
        config.rope_theta, config.head_dim, config.partial_rotary_factor);

    let is_mlx = {
        let cfg_text = std::fs::read_to_string(model_dir.join("config.json")).unwrap_or_default();
        cfg_text.contains("\"quant_method\"") || cfg_text.contains("\"quantization_config\"")
    };
    let use_int4_runtime = std::env::var("USE_INT4").map(|v| v == "1").unwrap_or(false);

    let (weights, raw_norms, quant_config, mode) = if is_mlx {
        let (w, n) = crate::weights::load_weights_mlx_int4(gpu, model_dir, &config);
        let qc = QuantConfig { bits: 4, group_size: 64, quant_method: "mlx".to_string(), sym: false };
        (w, n, qc, "mlx-int4")
    } else if use_int4_runtime {
        let (w, n) = crate::weights::load_weights_int4(gpu, model_dir, &config, 128);
        let qc = QuantConfig { bits: 4, group_size: 128, quant_method: "gptq".to_string(), sym: true };
        (w, n, qc, "int4-runtime")
    } else {
        let (w, n) = crate::weights::load_weights_bf16(gpu, model_dir, &config);
        let qc = QuantConfig { bits: 16, group_size: 1, quant_method: "bf16".to_string(), sym: false };
        (w, n, qc, "bf16")
    };

    let chunked = !weights.embed_chunks.is_empty();
    let has_mlx_biases = !weights.mlx_biases.is_empty();
    let mut model = Model::new(gpu, config.clone(), quant_config, weights, max_seq_len);
    if mode == "bf16" {
        model.bf16_mode = true;
    }
    if has_mlx_biases {
        model.mlx_int4_mode = true;
    }
    model.q_gated = false;
    model.norm_direct = true;
    // ASR decoder uses NeoX split-half RoPE (not interleaved) regardless of config
    if let Some(ref mut rp) = model.config.rope_parameters {
        rp.mrope_interleaved = false;
    }
    model.rebuild_qknorm_shader();
    model.rebuild_static_params(gpu);
    log::info!("[asr-decoder] mode={}, chunked_embed={}", mode, chunked);

    for (i, norm) in raw_norms.layers.iter().enumerate() {
        if let Some((q, k)) = norm {
            model.init_qknorm_params(gpu, i, q, k);
        }
    }

    log::info!("[asr-decoder] model ready");
    model
}

/// Convenience: create a new GPU context and load the model onto it.
pub fn load_bf16_model(model_dir: &Path, max_seq_len: u32) -> (GpuContext, Model) {
    let gpu = GpuContext::new();
    let model = load_model_on_gpu(&gpu, model_dir, max_seq_len);
    (gpu, model)
}

// ── Qwen3-ASR prompt token IDs ──
const TOKEN_IM_START: u32 = 151644;
const TOKEN_IM_END: u32 = 151645;
const TOKEN_ENDOFTEXT: u32 = 151643;
const TOKEN_AUDIO_START: u32 = 151669;
const TOKEN_AUDIO_END: u32 = 151670;
const TOKEN_ASR_TEXT: u32 = 151704;

/// Prompt structure: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
const PREFIX_HEAD: &[u32] = &[TOKEN_IM_START, 8948, 198]; // <|im_start|>system\n
const PREFIX_TAIL: &[u32] = &[TOKEN_IM_END, 198, TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START];
/// <|audio_end|><|im_end|>\n<|im_start|>assistant\n
const SUFFIX_BASE: &[u32] = &[TOKEN_AUDIO_END, TOKEN_IM_END, 198, TOKEN_IM_START, 77091, 198];

/// Zero the KV cache buffers. Required before first use since GPU buffers contain undefined data.
fn clear_kv_cache(gpu: &mut GpuContext, model: &Model) {
    let nl = model.config.num_hidden_layers as usize;
    let nkv = model.config.num_key_value_heads;
    let hd = model.config.head_dim;
    let max_seq = 256u32; // matches load_model_on_gpu max_seq_len
    let cache_size = (max_seq * nkv * hd) as usize * 4;
    let zeros = vec![0u8; cache_size];
    for i in 0..nl {
        gpu.write_buffer(&model.state.k_cache[i], 0, &zeros);
        gpu.write_buffer(&model.state.v_cache[i], 0, &zeros);
    }
    gpu.flush();
}

/// Forward one token through the decoder. Handles MLX INT4 embedding + forward.
/// `embed_scales_biases`: needed for MLX INT4 embedding lookup (from prefix cache or model weights).
fn forward_token(
    gpu: &mut GpuContext, model: &mut Model, token_id: u32,
    embed_sb: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
) -> u32 {
    let h = model.config.hidden_size;
    if model.mlx_int4_mode {
        // MLX INT4: use INT4 embedding shader
        if let Some((sc, bi)) = embed_sb.or_else(|| {
            // Try model weights' embed_scales/biases
            match (&model.weights.embed_scales, &model.weights.embed_biases) {
                (Some(s), Some(b)) => Some((s, b)),
                _ => None,
            }
        }) {
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
            struct EP { token_id: u32, dim: u32, group_size: u32, _pad: u32 }
            gpu.flush();
            gpu.write_buffer(&model.state.p_scratch, 0, bytemuck::bytes_of(&EP {
                token_id, dim: h, group_size: model.quant_config.group_size, _pad: 0,
            }));
            gpu.dispatch("emb_mlx", INT4_EMBEDDING_MLX_SRC, &[
                crate::gpu::bind(0, &model.weights.embed_tokens),
                crate::gpu::bind(1, sc), crate::gpu::bind(2, bi),
                crate::gpu::bind(3, &model.state.hidden),
                crate::gpu::bind(4, &model.state.p_scratch),
            ], (h.div_ceil(256), 1, 1));
        } else {
            // Fallback: bf16 embedding (works if embed_tokens is bf16)
            model.embedding(gpu, token_id);
        }
        gpu.flush();
        gpu.copy_buffer(&model.state.hidden, &model.state.residual, h as u64 * 4);
        model.forward_mlx_argmax(gpu)
    } else {
        model.forward_argmax(gpu, token_id)
    }
}

/// GPU ASR decode: encoder output → text.
/// Takes encoder output embeddings [seq_len × hidden_size] and decodes to text tokens.
/// Returns decoded text string.
pub fn gpu_asr_decode(
    gpu: &mut GpuContext,
    model: &mut Model,
    encoder_output: &[f32],
    enc_seq_len: u32,
) -> String {
    let hidden = model.config.hidden_size as usize;
    assert_eq!(encoder_output.len(), enc_seq_len as usize * hidden);

    // Reset model state for fresh decode
    model.seq_len = 0;
    model.generated_tokens.clear();
    clear_kv_cache(gpu, model);

    let t0 = std::time::Instant::now();

    // ── Prefill: prefix tokens ──
    for &tok in PREFIX_HEAD {
        forward_token(gpu, model, tok, None);
    }
    for &tok in PREFIX_TAIL {
        forward_token(gpu, model, tok, None);
    }

    let prefix_ms = t0.elapsed().as_millis();
    log::info!("[asr-decode] prefix: {} tokens in {}ms",
        PREFIX_HEAD.len() + PREFIX_TAIL.len(), prefix_ms);

    // ── Prefill: encoder output embeddings ──
    let t1 = std::time::Instant::now();
    let is_mlx = model.mlx_int4_mode;
    for i in 0..enc_seq_len as usize {
        let embed = &encoder_output[i * hidden..(i + 1) * hidden];
        let h = model.config.hidden_size;
        gpu.write_buffer(&model.state.hidden, 0, bytemuck::cast_slice(embed));
        gpu.flush();
        gpu.copy_buffer(&model.state.hidden, &model.state.residual, h as u64 * 4);
        if is_mlx {
            model.forward_mlx_argmax(gpu);
        } else {
            model.forward_embed_argmax(gpu, embed);
        }
    }
    let enc_ms = t1.elapsed().as_millis();
    log::info!("[asr-decode] encoder prefill: {} tokens in {}ms ({:.1}ms/tok)",
        enc_seq_len, enc_ms, enc_ms as f64 / enc_seq_len as f64);

    // ── Prefill: suffix tokens ──
    // Don't include TOKEN_ASR_TEXT — model generates it naturally (matches C reference)
    for &tok in &SUFFIX_BASE[..SUFFIX_BASE.len() - 1] {
        forward_token(gpu, model, tok, None);
    }

    // ── Generate from last suffix token ──
    let t2 = std::time::Instant::now();
    let mut token = forward_token(gpu, model, SUFFIX_BASE[SUFFIX_BASE.len() - 1], None);

    // Debug: dump final hidden state
    {
        gpu.flush();
        let hid_bytes = gpu.read_buffer(&model.state.residual, hidden as u64 * 4);
        let hv: &[f32] = bytemuck::cast_slice(&hid_bytes);
        let hn: f32 = hv.iter().map(|x| x*x).sum::<f32>().sqrt();
        log::debug!("[asr-decode] slow path final hidden: norm={hn:.4} first4={:?} seq_len={}", &hv[..4], model.seq_len);
    }

    let mut text = String::new();
    let mut n_generated = 0u32;
    let max_tokens = 448u32; // ASR rarely needs more

    // We need a tokenizer to decode tokens to text.
    // For now, collect token IDs and return them as a format string
    // that the caller can decode with the C tokenizer.
    let mut token_ids: Vec<u32> = Vec::new();
    let mut past_asr_text = true; // We forced <|asr_text|> as last prefix token

    while n_generated < max_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        if token == TOKEN_ASR_TEXT {
            past_asr_text = true;
        } else if past_asr_text {
            token_ids.push(token);
        }

        token = forward_token(gpu, model, token, None);
    }

    let decode_ms = t2.elapsed().as_millis();
    let total_ms = t0.elapsed().as_millis();
    log::info!("[asr-decode] decode: {} tokens in {}ms ({:.1}ms/tok), total={}ms",
        n_generated, decode_ms,
        if n_generated > 0 { decode_ms as f64 / n_generated as f64 } else { 0.0 },
        total_ms);

    // Return token IDs as space-separated string for C tokenizer decode,
    // or if we add a Rust tokenizer later, decode directly here.
    // For now, encode as binary: prefix with "TOKS:" marker.
    token_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
}

/// Cached prefix KV state — computed once, restored before each decode.
pub struct PrefixCache {
    /// KV cache snapshot: Vec of (k_data, v_data) per layer
    kv_snapshots: Vec<(Vec<u8>, Vec<u8>)>,
    /// Number of prefix tokens cached
    pub prefix_len: u32,
    /// Pre-computed prefix embeddings [prefix_len, hidden]
    pub prefix_embeds: Vec<f32>,
    /// MLX INT4 embed scales buffer (for dequant embedding lookup)
    pub embed_scales: Option<wgpu::Buffer>,
    /// MLX INT4 embed biases buffer
    pub embed_biases: Option<wgpu::Buffer>,
}

/// Pre-compute KV cache for the fixed ASR prompt prefix.
/// Call once after model load. Returns cache to pass to gpu_asr_decode_tokens.
pub fn precompute_prefix_cache(
    gpu: &mut GpuContext,
    model: &mut Model,
    model_dir: &Path,
) -> PrefixCache {
    let prefix: Vec<u32> = PREFIX_HEAD.iter().chain(PREFIX_TAIL.iter()).copied().collect();
    let hidden = model.config.hidden_size as usize;
    let nkv = model.config.num_key_value_heads;
    let hd = model.config.head_dim;
    let nl = model.config.num_hidden_layers as usize;
    let prefix_len = prefix.len() as u32;
    let kv_entry_bytes = (nkv * hd) as u64 * 4; // bytes per position per layer

    let t0 = std::time::Instant::now();

    // Build prefix embeddings — MLX INT4 or bf16
    let (embed_scales, embed_biases) = if model.mlx_int4_mode {
        // For MLX, need to load embed scales/biases from the safetensor
        // The MLX loader stored them — we need to extract from the model dir
        // Actually, the load_weights_mlx_int4 uploaded embed scales/biases as
        // separate buffers. But they're not stored in ModelWeights yet.
        // For now, do the embedding dequant via the int4_embedding_mlx shader.
        // We need to pass the embed qweight (in embed_tokens), and load
        // the scales+biases buffers here.
        let model_dir_cfg = std::fs::read_to_string(model_dir.join("config.json")).ok();
        // The MLX safetensors has model.embed_tokens.scales and model.embed_tokens.biases
        // We need to load them. For now, open the safetensors directly.
        let mut sf: Vec<_> = std::fs::read_dir(model_dir).unwrap()
            .filter_map(|e| e.ok()).filter(|e| e.path().extension().map_or(false, |x| x=="safetensors"))
            .map(|e| e.path()).collect();
        sf.sort();
        let mm: Vec<memmap2::Mmap> = sf.iter()
            .map(|p| unsafe { memmap2::Mmap::map(&std::fs::File::open(p).unwrap()).unwrap() }).collect();
        let sts: Vec<safetensors::SafeTensors> = mm.iter()
            .map(|m| safetensors::SafeTensors::deserialize(m).unwrap()).collect();
        let get = |n: &str| -> &[u8] {
            for s in &sts { if let Ok(t)=s.tensor(n) { return t.data(); } }
            panic!("{n}");
        };
        let es = gpu.upload_buffer("emb_sc", get("model.embed_tokens.scales"));
        let eb = gpu.upload_buffer("emb_bi", get("model.embed_tokens.biases"));
        (Some(es), Some(eb))
    } else {
        (None, None)
    };

    let mut prefix_embeds = Vec::with_capacity(prefix.len() * hidden);
    for &tok in &prefix {
        if model.mlx_int4_mode {
            if let (Some(ref sc), Some(ref bi)) = (&embed_scales, &embed_biases) {
                #[repr(C)]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                struct EP { token_id: u32, dim: u32, group_size: u32, _pad: u32 }
                gpu.flush();
                gpu.write_buffer(&model.state.p_scratch, 0, bytemuck::bytes_of(&EP {
                    token_id: tok, dim: model.config.hidden_size,
                    group_size: model.quant_config.group_size, _pad: 0,
                }));
                gpu.dispatch("emb_mlx", INT4_EMBEDDING_MLX_SRC, &[
                    crate::gpu::bind(0, &model.weights.embed_tokens),
                    crate::gpu::bind(1, sc),
                    crate::gpu::bind(2, bi),
                    crate::gpu::bind(3, &model.state.hidden),
                    crate::gpu::bind(4, &model.state.p_scratch),
                ], (model.config.hidden_size.div_ceil(256), 1, 1));
            }
        } else {
            model.embedding(gpu, tok);
        }
        gpu.flush();
        let bytes = gpu.read_buffer(&model.state.hidden, hidden as u64 * 4);
        prefix_embeds.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&bytes));
    }

    // Run prefix through model to populate KV cache
    model.seq_len = 0;
    model.generated_tokens.clear();
    clear_kv_cache(gpu, model);
    if model.bf16_mode {
        model.prefill(gpu, &prefix_embeds, prefix_len);
    } else if model.mlx_int4_mode {
        for chunk in prefix_embeds.chunks_exact(hidden) {
            gpu.write_buffer(&model.state.hidden, 0, bytemuck::cast_slice(chunk));
            gpu.flush();
            gpu.copy_buffer(&model.state.hidden, &model.state.residual, hidden as u64 * 4);
            model.forward_mlx_argmax(gpu);
        }
    } else {
        for chunk in prefix_embeds.chunks_exact(hidden) {
            model.forward_embed_argmax(gpu, chunk);
        }
    }

    // Snapshot KV cache
    let snapshot_bytes = prefix_len as u64 * kv_entry_bytes;
    let mut kv_snapshots = Vec::with_capacity(nl);
    for i in 0..nl {
        let k = gpu.read_buffer(&model.state.k_cache[i], snapshot_bytes);
        let v = gpu.read_buffer(&model.state.v_cache[i], snapshot_bytes);
        kv_snapshots.push((k, v));
    }

    log::info!("[asr-decode] prefix cache: {} tokens, {}KB per layer, computed in {}ms",
        prefix_len, snapshot_bytes / 1024, t0.elapsed().as_millis());

    PrefixCache { kv_snapshots, prefix_len, prefix_embeds, embed_scales, embed_biases }
}

/// GPU ASR decode returning raw token IDs (for caller to decode with tokenizer).
/// Uses prefix KV cache + batched GEMM prefill for audio embeddings + suffix.
pub fn gpu_asr_decode_tokens(
    gpu: &mut GpuContext,
    model: &mut Model,
    prefix_cache: &PrefixCache,
    encoder_output: &[f32],
    enc_seq_len: u32,
) -> Vec<u32> {
    let hidden = model.config.hidden_size as usize;
    assert_eq!(encoder_output.len(), enc_seq_len as usize * hidden);

    let t0 = std::time::Instant::now();

    // Restore prefix KV cache
    let nl = model.config.num_hidden_layers as usize;
    for i in 0..nl {
        let (ref k, ref v) = prefix_cache.kv_snapshots[i];
        gpu.write_buffer(&model.state.k_cache[i], 0, k);
        gpu.write_buffer(&model.state.v_cache[i], 0, v);
    }
    model.seq_len = prefix_cache.prefix_len;
    model.generated_tokens.clear();

    // Build remaining embeddings: audio + suffix_base only (prefix is cached)
    // TOKEN_ASR_TEXT is NOT prefilled — model generates it naturally (matches C reference)
    let remain_seq = enc_seq_len as usize + SUFFIX_BASE.len();

    let mut input_embeds = Vec::with_capacity(remain_seq * hidden);
    input_embeds.extend_from_slice(encoder_output);
    for &tok in SUFFIX_BASE {
        if model.mlx_int4_mode {
            if let (Some(ref sc), Some(ref bi)) = (&prefix_cache.embed_scales, &prefix_cache.embed_biases) {
                #[repr(C)]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                struct EP { token_id: u32, dim: u32, group_size: u32, _pad: u32 }
                gpu.flush();
                gpu.write_buffer(&model.state.p_scratch, 0, bytemuck::bytes_of(&EP {
                    token_id: tok, dim: model.config.hidden_size,
                    group_size: model.quant_config.group_size, _pad: 0,
                }));
                gpu.dispatch("emb_mlx", INT4_EMBEDDING_MLX_SRC, &[
                    crate::gpu::bind(0, &model.weights.embed_tokens),
                    crate::gpu::bind(1, sc), crate::gpu::bind(2, bi),
                    crate::gpu::bind(3, &model.state.hidden),
                    crate::gpu::bind(4, &model.state.p_scratch),
                ], (model.config.hidden_size.div_ceil(256), 1, 1));
            }
        } else {
            model.embedding(gpu, tok);
        }
        gpu.flush();
        let bytes = gpu.read_buffer(&model.state.hidden, hidden as u64 * 4);
        input_embeds.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&bytes));
    }

    let embed_ms = t0.elapsed().as_millis();

    // Debug: check encoder output is non-zero
    let enc_norm: f32 = encoder_output.iter().map(|x| x * x).sum::<f32>().sqrt();
    let enc_first8: Vec<f32> = encoder_output.iter().take(8).copied().collect();
    log::debug!("[asr-decode] encoder output: norm={enc_norm:.4}, first8={enc_first8:?}");
    log::debug!("[asr-decode] input_embeds: {} floats, remain_seq={}", input_embeds.len(), remain_seq);
    log::info!("[asr-decode] restored prefix ({} tokens), built {} remaining embeds in {}ms",
        prefix_cache.prefix_len, remain_seq, embed_ms);

    // Prefill remaining tokens (audio + suffix) — token by token
    let t1 = std::time::Instant::now();
    let h = model.config.hidden_size as usize;
    let is_mlx = model.mlx_int4_mode;
    for (i, chunk) in input_embeds.chunks_exact(h).enumerate() {
        gpu.write_buffer(&model.state.hidden, 0, bytemuck::cast_slice(chunk));
        gpu.flush();
        gpu.copy_buffer(&model.state.hidden, &model.state.residual, h as u64 * 4);
        if is_mlx {
            model.forward_mlx_argmax(gpu);
        } else {
            model.forward_embed_argmax(gpu, chunk);
        }
        if i < 3 || i == remain_seq - 1 {
            gpu.flush();
            let hid_bytes = gpu.read_buffer(&model.state.hidden, h as u64 * 4);
            let hv: &[f32] = bytemuck::cast_slice(&hid_bytes);
            let has_nan = hv.iter().any(|x| x.is_nan());
            let norm: f32 = hv.iter().map(|x| x*x).sum::<f32>().sqrt();
            log::debug!("[asr-decode] prefill token {i}: hidden norm={norm:.4} nan={has_nan} tok={}",
                model.generated_tokens.last().unwrap_or(&0));
        }
    }
    let mut token = *model.generated_tokens.last().unwrap_or(&0);

    // Debug: check hidden + normed + logits at last prefill position
    {
        gpu.flush();
        // Read raw hidden (residual after all layers)
        let hid_bytes = gpu.read_buffer(&model.state.residual, h as u64 * 4);
        let hv: &[f32] = bytemuck::cast_slice(&hid_bytes);
        let hn: f32 = hv.iter().map(|x| x*x).sum::<f32>().sqrt();
        log::debug!("[asr-decode] final hidden: norm={hn:.4} first4={:?}", &hv[..4]);

        let normed_bytes = gpu.read_buffer(&model.state.normed, h as u64 * 4);
        let nv: &[f32] = bytemuck::cast_slice(&normed_bytes);
        let nn: f32 = nv.iter().map(|x| x*x).sum::<f32>().sqrt();
        let logits_bytes = gpu.read_buffer(&model.state.logits, model.config.vocab_size as u64 * 4);
        let lv: &[f32] = bytemuck::cast_slice(&logits_bytes);
        let (max_idx, max_val) = lv.iter().enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        let nonzero = lv.iter().filter(|&&x| x.abs() > 1e-10).count();
        log::debug!("[asr-decode] last prefill: normed_norm={nn:.4} logits max={max_val:.4}@{max_idx} nonzero={nonzero} first_token={token}");
    }

    let prefill_ms = t1.elapsed().as_millis();
    log::info!("[asr-decode] prefill: {} tokens in {}ms (prefix cached), first_token={}",
        remain_seq, prefill_ms, token);

    // Don't force TOKEN_ASR_TEXT — let the model generate it naturally (matches C reference)
    // The prefill ended at suffix_base's last token; model should predict ASR_TEXT next.

    // Autoregressive decode
    let t2 = std::time::Instant::now();
    let mut token_ids: Vec<u32> = Vec::new();
    let mut n_generated = 0u32;

    while n_generated < 448 {
        n_generated += 1;
        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END { break; }
        if token != TOKEN_ASR_TEXT {
            token_ids.push(token);
        }
        // MLX mode: use embedding_mlx + forward_mlx_argmax for decode tokens
        if is_mlx {
            if let (Some(ref sc), Some(ref bi)) = (&prefix_cache.embed_scales, &prefix_cache.embed_biases) {
                #[repr(C)]
                #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
                struct EP { token_id: u32, dim: u32, group_size: u32, _pad: u32 }
                gpu.flush();
                gpu.write_buffer(&model.state.p_scratch, 0, bytemuck::bytes_of(&EP {
                    token_id: token, dim: model.config.hidden_size,
                    group_size: model.quant_config.group_size, _pad: 0,
                }));
                gpu.dispatch("emb_mlx", INT4_EMBEDDING_MLX_SRC, &[
                    crate::gpu::bind(0, &model.weights.embed_tokens),
                    crate::gpu::bind(1, sc), crate::gpu::bind(2, bi),
                    crate::gpu::bind(3, &model.state.hidden),
                    crate::gpu::bind(4, &model.state.p_scratch),
                ], (model.config.hidden_size.div_ceil(256), 1, 1));
            }
            gpu.flush();
            gpu.copy_buffer(&model.state.hidden, &model.state.residual, h as u64 * 4);
            token = model.forward_mlx_argmax(gpu);
        } else {
            token = model.forward_argmax(gpu, token);
        }
    }

    let total_ms = t0.elapsed().as_millis();
    log::info!("[asr-decode] {} tokens generated in {}ms (prefill: {}ms, decode: {}ms)",
        token_ids.len(), total_ms, prefill_ms, t2.elapsed().as_millis());

    token_ids
}

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

    // Note: The actual decode logic uses the existing Model infrastructure.
    // Use load_bf16_model() to get a (GpuContext, Model) pair,
    // then call model.forward() for autoregressive decoding.

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
