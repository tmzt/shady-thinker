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

// ── Batched prefill shader sources ──────────────────────────────────────

mod shaders {
    // Batched GEMM for INT8 quantized matmul (prefill: [seq, in] × [in, out] → [seq, out])
    // TODO: include_str! once the .wgsl files are finalized by the shader agent
    // pub const BATCHED_INT8_GEMM: &str = include_str!("shaders/batched_int8_gemm.wgsl");
    // pub const BATCHED_RMSNORM: &str = include_str!("shaders/batched_rmsnorm.wgsl");
    // pub const BATCHED_ADD_RMSNORM: &str = include_str!("shaders/batched_add_rmsnorm.wgsl");
    // pub const BATCHED_ROPE_QKNORM: &str = include_str!("shaders/batched_rope_qknorm.wgsl");
    // pub const BATCHED_CAUSAL_ATTN: &str = include_str!("shaders/batched_causal_attn.wgsl");
    // pub const BATCHED_SILU_MUL: &str = include_str!("shaders/batched_silu_mul.wgsl");
}

// ── Pipeline struct ─────────────────────────────────────────────────────

/// Merged encoder + decoder pipeline sharing a single GPU context.
///
/// Owns all weights, state buffers, and pre-computed embeddings needed
/// to go from raw mel spectrogram to decoded text in one `forward` call.
pub struct AsrPipeline {
    // ── Shared GPU ──
    pub gpu: GpuContext,

    // ── Encoder ──
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
        let encoder = AsrEncoder::load(gpu, model_dir);
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
        let s_gemm_q = format!("// TODO: batched INT8 GEMM Q: [{h}, {}] seq={MAX_PREFILL}", nh * hd);
        let s_gemm_kv = format!("// TODO: batched INT8 GEMM KV: [{h}, {}] seq={MAX_PREFILL}", nkv * hd);
        let s_gemm_o = format!("// TODO: batched INT8 GEMM O: [{}, {h}] seq={MAX_PREFILL}", nh * hd);
        let s_gemm_gate = format!("// TODO: batched INT8 GEMM gate: [{h}, {inter}] seq={MAX_PREFILL}");
        let s_gemm_up = format!("// TODO: batched INT8 GEMM up: [{h}, {inter}] seq={MAX_PREFILL}");
        let s_gemm_down = format!("// TODO: batched INT8 GEMM down: [{inter}, {h}] seq={MAX_PREFILL}");
        let s_batched_rmsnorm = format!(
            "// TODO: batched RMSNorm h={h} eps={} seq={MAX_PREFILL}",
            decoder_config.rms_norm_eps
        );
        let s_batched_add_rmsnorm = format!(
            "// TODO: batched Add+RMSNorm h={h} eps={} seq={MAX_PREFILL}",
            decoder_config.rms_norm_eps
        );
        let s_batched_qknorm = format!(
            "// TODO: batched QKNorm+RoPE nh={nh} nkv={nkv} hd={hd} seq={MAX_PREFILL}"
        );
        let s_batched_causal_attn = format!(
            "// TODO: batched causal attention nh={nh} nkv={nkv} hd={hd} seq={MAX_PREFILL}"
        );
        let s_batched_silu_mul = format!(
            "// TODO: batched SiLU×mul inter={inter} seq={MAX_PREFILL}"
        );

        let load_ms = t0.elapsed().as_millis();
        log::info!("[asr-pipeline] loaded in {}ms (encoder + decoder + prefill bufs)", load_ms);

        Self {
            gpu: GpuContext::new(), // Placeholder — see note below about split borrow
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
        }
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

    /// Fill pre-embedded token buffers using the decoder's GPU embedding shader.
    /// Called once during load after the pipeline is fully constructed.
    fn fill_prefix_suffix_embeds(&mut self) {
        let h = self.decoder_config.hidden_size as usize;
        let gpu = &mut self.encoder.gpu;

        let prefix_tokens: Vec<u32> = PREFIX_HEAD
            .iter()
            .chain(PREFIX_TAIL.iter())
            .copied()
            .collect();

        // Embed each prefix token and copy into the pre-allocated buffer
        for (i, &tok) in prefix_tokens.iter().enumerate() {
            self.decoder.embedding(gpu, tok);
            gpu.flush();
            let offset = (i * h) as u64 * 4;
            gpu.copy_buffer_offset(
                &self.decoder.state.hidden, 0,
                &self.prefix_embed_buf, offset,
                h as u64 * 4,
            );
        }

        // Embed each suffix token
        for (i, &tok) in SUFFIX_TOKENS.iter().enumerate() {
            self.decoder.embedding(gpu, tok);
            gpu.flush();
            let offset = (i * h) as u64 * 4;
            gpu.copy_buffer_offset(
                &self.decoder.state.hidden, 0,
                &self.suffix_embed_buf, offset,
                h as u64 * 4,
            );
        }

        gpu.flush();
        log::info!(
            "[asr-pipeline] pre-embedded {} prefix + {} suffix tokens on GPU",
            prefix_tokens.len(),
            SUFFIX_TOKENS.len()
        );
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
        let h = self.decoder_config.hidden_size;
        let nh = self.decoder_config.num_attention_heads;
        let nkv = self.decoder_config.num_key_value_heads;
        let hd = self.decoder_config.head_dim;
        let inter = self.decoder_config.intermediate_size;
        let nl = self.decoder_config.num_hidden_layers as usize;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let gpu = &mut self.encoder.gpu;

        for layer_idx in 0..nl {
            let layer = &self.decoder.weights.layers[layer_idx];
            let biases = &self.decoder.weights.mlx_biases[layer_idx];

            // ── 1. RMSNorm (or Add+RMSNorm for layers 1+) ──
            if layer_idx == 0 {
                // RMSNorm(prefill_input) → prefill_normed
                // TODO: dispatch batched rmsnorm shader
                //   gpu.dispatch("pf_norm", &self.s_batched_rmsnorm, &[
                //       gpu::bind(0, &self.prefill_residual),
                //       gpu::bind(1, &layer.input_layernorm),
                //       gpu::bind(2, &self.prefill_normed),
                //   ], (actual_len, 1, 1));
            } else {
                // Add(residual, mlp_out) + RMSNorm → normed, update residual
                // TODO: dispatch batched add+rmsnorm shader
                //   gpu.dispatch("pf_addnorm", &self.s_batched_add_rmsnorm, &[
                //       gpu::bind(0, &self.prefill_residual),
                //       gpu::bind(1, &self.prefill_mlp_out),
                //       gpu::bind(2, &layer.input_layernorm),
                //       gpu::bind(3, &self.prefill_normed),
                //   ], (actual_len, 1, 1));
            }

            // ── 2. QKV projections (batched INT8 GEMM) ──
            if let Some(sa) = layer.self_attn() {
                // Q: [actual_len, h] × [h, q_dim] → [actual_len, q_dim]
                // TODO: dispatch batched GEMM for Q projection
                //   gpu.dispatch("pf_q", &self.s_gemm_q, &[
                //       gpu::bind(0, &self.prefill_normed),
                //       gpu::bind(1, &sa.q_proj_qweight),
                //       gpu::bind(2, &sa.q_proj_scales),
                //       gpu::bind(3, &biases[0]),
                //       gpu::bind(4, &self.prefill_q),
                //   ], (actual_len, q_dim.div_ceil(32), 1));

                // K: [actual_len, h] × [h, kv_dim] → [actual_len, kv_dim]
                // TODO: dispatch batched GEMM for K projection
                //   gpu.dispatch("pf_k", &self.s_gemm_kv, &[
                //       gpu::bind(0, &self.prefill_normed),
                //       gpu::bind(1, &sa.k_proj_qweight),
                //       gpu::bind(2, &sa.k_proj_scales),
                //       gpu::bind(3, &biases[1]),
                //       gpu::bind(4, &self.prefill_k),
                //   ], (actual_len, kv_dim.div_ceil(32), 1));

                // V: [actual_len, h] × [h, kv_dim] → [actual_len, kv_dim]
                // TODO: dispatch batched GEMM for V projection
                //   gpu.dispatch("pf_v", &self.s_gemm_kv, &[
                //       gpu::bind(0, &self.prefill_normed),
                //       gpu::bind(1, &sa.v_proj_qweight),
                //       gpu::bind(2, &sa.v_proj_scales),
                //       gpu::bind(3, &biases[2]),
                //       gpu::bind(4, &self.prefill_v),
                //   ], (actual_len, kv_dim.div_ceil(32), 1));

                // ── 3. QKNorm + RoPE + KV cache write (batched) ──
                // Applies per-head RMSNorm to Q and K, then rotary position embeddings,
                // then writes K and V into the KV cache at positions [0..actual_len).
                // TODO: dispatch batched qknorm+rope+kv_cache_write shader
                //   gpu.dispatch("pf_qknorm", &self.s_batched_qknorm, &[
                //       gpu::bind(0, &self.prefill_q),
                //       gpu::bind(1, &self.prefill_k),
                //       gpu::bind(2, &self.prefill_v),
                //       gpu::bind(3, &self.prefill_q),          // Q output (in-place)
                //       gpu::bind(4, &self.decoder.state.k_cache[layer_idx]),
                //       gpu::bind(5, &self.decoder.state.v_cache[layer_idx]),
                //       gpu::bind(6, &self.decoder.state.qknorm_params[layer_idx]),
                //   ], (actual_len * (nh + nkv), 1, 1));

                // ── 4. Batched causal self-attention ──
                // For each head: scores = Q @ K^T (causal mask), attn = softmax(scores) @ V
                // Reads from KV cache (just written).
                // TODO: dispatch batched causal attention shader
                //   gpu.dispatch("pf_attn", &self.s_batched_causal_attn, &[
                //       gpu::bind(0, &self.prefill_q),
                //       gpu::bind(1, &self.decoder.state.k_cache[layer_idx]),
                //       gpu::bind(2, &self.decoder.state.v_cache[layer_idx]),
                //       gpu::bind(3, &self.prefill_attn_out),
                //   ], (nh, actual_len, 1));

                // ── 5. O projection (batched GEMM) ──
                // [actual_len, q_dim] × [q_dim, h] → [actual_len, h]
                // TODO: dispatch batched GEMM for O projection
                //   gpu.dispatch("pf_o", &self.s_gemm_o, &[
                //       gpu::bind(0, &self.prefill_attn_out),
                //       gpu::bind(1, &sa.o_proj_qweight),
                //       gpu::bind(2, &sa.o_proj_scales),
                //       gpu::bind(3, &biases[3]),
                //       gpu::bind(4, &self.prefill_o_out),
                //   ], (actual_len, h.div_ceil(32), 1));
            }

            // ── 6. Post-attention Add+RMSNorm ──
            // residual += o_proj_out; normed = rmsnorm(residual)
            // TODO: dispatch batched add+rmsnorm
            //   gpu.dispatch("pf_postnorm", &self.s_batched_add_rmsnorm, &[
            //       gpu::bind(0, &self.prefill_residual),
            //       gpu::bind(1, &self.prefill_o_out),
            //       gpu::bind(2, &layer.post_attn_layernorm),
            //       gpu::bind(3, &self.prefill_normed),
            //   ], (actual_len, 1, 1));

            // ── 7. MLP: gate + up + SiLU×up + down ──
            // Gate: [actual_len, h] × [h, inter] → [actual_len, inter]
            // TODO: dispatch batched GEMM for gate
            //   gpu.dispatch("pf_gate", &self.s_gemm_gate, &[
            //       gpu::bind(0, &self.prefill_normed),
            //       gpu::bind(1, &layer.gate_proj_qweight),
            //       gpu::bind(2, &layer.gate_proj_scales),
            //       gpu::bind(3, &biases[4]),
            //       gpu::bind(4, &self.prefill_gate),
            //   ], (actual_len, inter.div_ceil(32), 1));

            // Up: [actual_len, h] × [h, inter] → [actual_len, inter]
            // TODO: dispatch batched GEMM for up
            //   gpu.dispatch("pf_up", &self.s_gemm_up, &[
            //       gpu::bind(0, &self.prefill_normed),
            //       gpu::bind(1, &layer.up_proj_qweight),
            //       gpu::bind(2, &layer.up_proj_scales),
            //       gpu::bind(3, &biases[5]),
            //       gpu::bind(4, &self.prefill_up),
            //   ], (actual_len, inter.div_ceil(32), 1));

            // SiLU(gate) × up → gate (in-place)
            // TODO: dispatch batched SiLU×mul shader
            //   gpu.dispatch("pf_silu", &self.s_batched_silu_mul, &[
            //       gpu::bind(0, &self.prefill_gate),
            //       gpu::bind(1, &self.prefill_up),
            //   ], ((actual_len as u64 * inter as u64).div_ceil(256) as u32, 1, 1));

            // Down: [actual_len, inter] × [inter, h] → [actual_len, h]
            // TODO: dispatch batched GEMM for down projection
            //   gpu.dispatch("pf_down", &self.s_gemm_down, &[
            //       gpu::bind(0, &self.prefill_gate),
            //       gpu::bind(1, &layer.down_proj_qweight),
            //       gpu::bind(2, &layer.down_proj_scales),
            //       gpu::bind(3, &biases[6]),
            //       gpu::bind(4, &self.prefill_mlp_out),
            //   ], (actual_len, h.div_ceil(32), 1));
        }

        // ── 8. Final Add+RMSNorm (last layer residual + MLP output) ──
        // TODO: dispatch batched add+rmsnorm for final norm
        //   gpu.dispatch("pf_final_norm", &self.s_batched_add_rmsnorm, &[
        //       gpu::bind(0, &self.prefill_residual),
        //       gpu::bind(1, &self.prefill_mlp_out),
        //       gpu::bind(2, &self.decoder.weights.final_norm),
        //       gpu::bind(3, &self.prefill_normed),
        //   ], (actual_len, 1, 1));

        // ── 9. LM head on last token only ──
        // Extract normed[actual_len-1] → decoder.state.normed (single vector)
        let last_offset = (actual_len - 1) as u64 * h as u64 * 4;
        gpu.copy_buffer_offset(
            &self.prefill_normed, last_offset,
            &self.decoder.state.normed, 0,
            h as u64 * 4,
        );

        // Reuse the decoder's LM head (per-token matvec, fast enough for 1 token)
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

        // TODO: uncomment once decoder shader strings are available
        // for (ci, (chunk, shader)) in chunks.iter().zip(self.decoder.s_lm_head.iter()).enumerate() {
        //     let n = ((ci as u32 + 1) * cs).min(self.decoder_config.vocab_size) - ci as u32 * cs;
        //     gpu.dispatch(&format!("pf_lmh_{ci}"), shader, &[
        //         gpu::bind(0, &self.decoder.state.normed),
        //         gpu::bind(1, chunk),
        //         gpu::bind(2, sc),
        //         gpu::bind(3, bi),
        //         gpu::bind(4, &self.decoder.state.logits),
        //     ], (n.div_ceil(32), 1, 1));
        // }

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
