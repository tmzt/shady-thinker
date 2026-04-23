//! Pure inference session with RLM epiphany support.
//!
//! Provides token-ID-in, token-ID-out generation with mid-generation
//! tool execution. Tools are dispatched via an EpiphanyDispatcher trait.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::GpuContext;
use crate::model::{shaders, Model};
use crate::weights;
use crate::weights::ModelConfig;

#[cfg(feature = "jit-lora")]
use crate::lora::LoraState;

/// Hash of all dispatched shader sources — used to key the Vulkan pipeline cache file.
/// Any shader change produces a new hash, a new cache filename, and a fresh compilation.
fn shader_cache_key() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    shaders::GPTQ_MATVEC_4T.hash(&mut h);
    shaders::FUSED_SILU_GPTQ_4T.hash(&mut h);
    shaders::FUSED_GATE_UP_GPTQ_4T.hash(&mut h);
    shaders::ADD_RMSNORM.hash(&mut h);
    shaders::ADD_RMSNORM_DIRECT.hash(&mut h);
    shaders::RMSNORM.hash(&mut h);
    shaders::RMSNORM_DIRECT.hash(&mut h);
    shaders::EMBEDDING.hash(&mut h);
    shaders::GQA_ATTENTION_HEAD.hash(&mut h);
    shaders::SIGMOID_MUL.hash(&mut h);
    shaders::FUSED_CONV_DELTANET_NORM.hash(&mut h);
    shaders::BATCHED_DELTANET_PREFILL.hash(&mut h);
    shaders::BF16_MATVEC.hash(&mut h);
    shaders::SAMPLE_TOPK.hash(&mut h);
    shaders::GPTQ_GEMM_4T.hash(&mut h);
    shaders::FUSED_SILU_GPTQ_GEMM_4T.hash(&mut h);
    shaders::BATCHED_ADD_RMSNORM.hash(&mut h);
    shaders::CAUSAL_ATTENTION_PREFILL.hash(&mut h);
    shaders::BATCHED_QKNORM_ROPE_GATED.hash(&mut h);
    h.finish()
}

/// Cost estimate for a tool call — determines execution strategy.
#[derive(Debug, Clone, Copy)]
pub enum ToolCost {
    /// Execute inline, block generation briefly (~<10ms). E.g. math, lookup.
    Instant,
    /// Fire async, inject result when ready without rewind. Model keeps generating.
    Async,
}

/// Result from executing a tool call.
pub struct ToolResult {
    /// The response text to inject (will be tokenized and fed into KV cache).
    pub response: String,
    /// Whether this was an async result arriving later.
    pub was_async: bool,
}

/// Trait for dispatching tool calls detected during generation.
/// Implementors handle tool execution and return results.
pub trait EpiphanyDispatcher: Send {
    /// Estimate the cost of executing a tool call.
    /// Called immediately when `</tool_call>` is detected.
    fn estimate_cost(&self, tool_body: &str) -> ToolCost;

    /// Execute a tool call synchronously. Only called for `ToolCost::Instant`.
    fn execute_sync(&self, tool_body: &str) -> ToolResult;

    /// Fire an async tool call. Only called for `ToolCost::Async`.
    /// The result will be collected later via `poll_async_results()`.
    fn fire_async(&self, call_id: u32, tool_body: &str);

    /// Poll for completed async tool results. Non-blocking.
    /// Returns vec of (call_id, result) for completed calls.
    fn poll_async_results(&self) -> Vec<(u32, ToolResult)>;

    /// Tokenize text into token IDs (needed to inject tool responses).
    fn tokenize(&self, text: &str) -> Vec<u32>;

    /// Decode token IDs to text (needed to detect tool call patterns).
    fn decode(&self, token_ids: &[u32]) -> String;

    /// Token IDs that form the `</tool_call>` marker.
    fn tool_call_end_marker(&self) -> &[u32];

    /// Observe a tool-response payload that was just injected into the
    /// generation stream. `body` is the raw text between the
    /// `<tool_response>` markers (no wrapping tags, no extra whitespace
    /// trimmed). Default no-op — implementations that record
    /// transcripts override this.
    fn record_injection(&self, _body: &str) {}
}

/// Generation outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerateState {
    Complete,
    Interrupted,
}

/// Result of a generation call.
pub struct GenerateResult {
    pub token_ids: Vec<u32>,
    pub token_count: usize,
    pub tokens_per_sec: f64,
    pub state: GenerateState,
    /// Number of epiphanies (tool calls executed) during this generation.
    pub epiphany_count: u32,
}

/// Configurable think token injection for reasoning models.
pub struct ThinkConfig {
    pub think_prefix_ids: Vec<u32>,
    pub think_end_ids: Vec<u32>,
}

/// Tool call detection config for Qwen3.5 native function calling.
/// Enables a state machine: free thinking → JSON-constrained tool call → stop.
pub struct ToolCallConfig {
    pub tool_call_start_ids: Vec<u32>, // <tool_call> token sequence
    pub tool_call_end_ids: Vec<u32>,   // </tool_call> token sequence
}

/// Generation state for think-then-tool models.
#[derive(Clone, Copy, PartialEq, Debug)]
enum GenState {
    /// Free generation inside <think> block (no constraints)
    Thinking,
    /// After </think>, waiting for <tool_call> or text
    AwaitTool,
    /// Inside <tool_call>, JSON sampler active
    JsonConstrained,
}

/// In-memory snapshot of GPU state after system prompt prefill.
/// Allows each generate() to restore the prefix state and run only
/// the query tokens, instead of re-running the full context.
pub struct PrefixSnapshot {
    /// Per self-attn layer (indexed by global layer index): (k_bytes, v_bytes).
    /// Only SA layer indices have non-empty vecs.
    pub(crate) kv: Vec<(Vec<u8>, Vec<u8>)>,
    /// Per linear-attn layer (0-based): (hist_bytes, state_bytes).
    pub(crate) dn: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Pure inference session: model + GPU context.
pub struct InferenceSession {
    pub model: Model,
    pub gpu: GpuContext,
    pub config: ModelConfig,
    pub think_config: Option<ThinkConfig>,
    /// Tool call detection for think-then-constrain models (Qwen3.5 Fast).
    pub tool_call_config: Option<ToolCallConfig>,
    /// KV cache position after prefix prefill. Each generate() call resets
    /// seq_len to this value so the cached system prompt is never re-run.
    pub prefix_len: u32,
    /// In-memory snapshot captured after set_prefix() completes.
    pub prefix_snapshot: Option<PrefixSnapshot>,
    /// Identifier of the prefix currently loaded in `prefix_snapshot`.
    /// `Some("static")` when the on-startup system-prompt cache is
    /// active; `Some("openai-<hash>")` when a per-request prefix is
    /// installed; `None` until any prefix is set. Used by the GPU
    /// loop to skip a re-load when the next request's prefix key
    /// matches what's already in memory.
    pub current_prefix_key: Option<String>,
    /// Optional ASR encoder for audio input (Qwen3-ASR models).
    pub asr_encoder: Option<crate::asr_encoder::AsrEncoder>,
    /// ASR-specific prefix cache (built by asr_decoder, not from system prompt text).
    pub asr_prefix_cache: Option<crate::asr_decoder::PrefixCache>,
}

impl InferenceSession {
    pub fn new(model_dir: PathBuf, max_seq_len: u32) -> Self {
        let t0 = std::time::Instant::now();
        log::info!("[shady-thinker] loading model from {:?} (max_seq={})", model_dir, max_seq_len);

        log::info!("[shady-thinker] step 1/6: reading config.json");
        let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
        log::info!("[shady-thinker] step 2/6: detecting weight format");
        let weight_format = weights::detect_weight_format(&model_dir);
        let quant_config = if weight_format == "bf16" {
            log::info!("[shady-thinker] detected BF16 weights");
            weights::QuantConfig { bits: 16, group_size: 0, quant_method: "bf16".to_string(), sym: false }
        } else {
            // Try both naming conventions: GPTQ uses quantize_config.json,
            // AutoRound uses quantization_config.json
            let path1 = model_dir.join("quantize_config.json");
            let path2 = model_dir.join("quantization_config.json");
            if path1.exists() {
                weights::QuantConfig::from_file(&path1)
            } else {
                weights::QuantConfig::from_file(&path2)
            }
        };
        log::info!("[shady-thinker] config: {} layers, {} heads, dim={}, format={}, bits={} ({:.1}s)",
            config.num_hidden_layers, config.num_attention_heads, config.hidden_size,
            weight_format, quant_config.bits, t0.elapsed().as_secs_f32());

        log::info!("[shady-thinker] step 3/6: creating GPU context");
        let mut gpu = GpuContext::new();
        log::info!("[shady-thinker] GPU context ready ({:.1}s)", t0.elapsed().as_secs_f32());

        let _ = shader_cache_key;

        log::info!("[shady-thinker] step 4/6: loading weights from safetensors");
        let (model_weights, raw_norms) = if weight_format == "bf16" {
            weights::load_weights_bf16(&gpu, &model_dir, &config)
        } else {
            weights::load_weights(&gpu, &model_dir, &config)
        };
        log::info!("[shady-thinker] weights loaded ({:.1}s)", t0.elapsed().as_secs_f32());

        log::info!("[shady-thinker] step 5/6: creating model pipeline");
        let mut model =
            Model::new(&gpu, config.clone(), quant_config, model_weights, max_seq_len);
        if weight_format == "bf16" {
            model.bf16_mode = true;
            log::info!("[shady-thinker] BF16 mode enabled");
        }
        log::info!("[shady-thinker] model pipeline created ({:.1}s), q_gated={}, bf16={}, vocab={}",
            t0.elapsed().as_secs_f32(), model.q_gated, model.bf16_mode, model.config.vocab_size);

        log::info!("[shady-thinker] step 6/6: initializing QK norm params");
        for (i, norm_data) in raw_norms.layers.iter().enumerate() {
            if let Some((q_bytes, k_bytes)) = norm_data {
                model.init_qknorm_params(&gpu, i, q_bytes, k_bytes);
            }
        }

        log::info!("[shady-thinker] model ready ({:.1}s total)", t0.elapsed().as_secs_f32());

        // Drain all pending weight-upload submissions before warmup.
        // upload_buffer() submits without polling; the GPU may lag behind the CPU.
        // One sync here ensures the GPU has all weights before the first forward pass.
        gpu.flush_and_wait();

        let mut session = Self { model, gpu, config, think_config: None, tool_call_config: None, prefix_len: 0, prefix_snapshot: None, current_prefix_key: None, asr_encoder: None, asr_prefix_cache: None };

        // Warm-up: run one token through the model to force Vulkan pipeline compilation.
        // This makes the first real inference fast (cache hit instead of JIT compile).
        // Then save the compiled pipeline cache to disk for subsequent app launches.
        log::info!("[shady-thinker] warming up shaders...");
        let warmup_start = std::time::Instant::now();
        session.model.forward(&mut session.gpu, 0);
        session.model.seq_len = 0; // reset KV cache after warmup
        session.model.generated_tokens.clear();
        session.model.seen_bitmap_cpu.iter_mut().for_each(|w| *w = 0);
        let zero_bitmap = vec![0u8; session.model.seen_bitmap_cpu.len() * 4];
        session.gpu.write_buffer(&session.model.state.seen_bitmap, 0, &zero_bitmap);
        log::info!("[shady-thinker] warmup done ({:.1}s)", warmup_start.elapsed().as_secs_f32());

        // Pipeline cache not saved (see comment above).

        session
    }

    pub fn set_think_config(&mut self, config: ThinkConfig) {
        self.think_config = Some(config);
    }

    pub fn set_tool_call_config(&mut self, config: ToolCallConfig) {
        self.tool_call_config = Some(config);
    }

    /// Load the ASR audio encoder from the model directory.
    /// Call after new() for ASR-capable models (Qwen3-ASR).
    pub fn load_asr_encoder(&mut self, model_dir: &std::path::Path) {
        let enc_gpu = GpuContext::from_device_queue(
            self.gpu.device.clone(), self.gpu.queue.clone(),
        );
        self.asr_encoder = Some(crate::asr_encoder::AsrEncoder::load(enc_gpu, model_dir));
        log::info!("[shady-thinker] ASR encoder loaded");
    }

    /// Run ASR inference: mel → encoder → asr_decoder (proven path).
    ///
    /// Uses the `asr_prefix_cache` (built by `asr_decoder::precompute_prefix_cache`)
    /// and delegates to `gpu_asr_decode_tokens` which uses `forward_embed_argmax`.
    pub fn infer_mel(&mut self, mel_data: &[f32], mel_frames: u32) -> Vec<u32> {
        let encoder = match self.asr_encoder.as_mut() {
            Some(e) => e,
            None => {
                log::error!("[shady-thinker:{}] infer_mel: no ASR encoder", self.gpu.role_tag);
                return Vec::new();
            }
        };

        let t0 = std::time::Instant::now();

        // Run encoder on real mel frames (no padding, no mask needed).
        let (encoder_output, enc_seq_len, _, _) = encoder.forward_mel(mel_data, mel_frames);
        log::info!("[shady-thinker:{}][asr] encoder: {} mel frames → {} tokens in {}ms",
            self.gpu.role_tag, mel_frames, enc_seq_len, t0.elapsed().as_millis());

        let prefix_cache = match self.asr_prefix_cache.as_ref() {
            Some(c) => c,
            None => {
                log::error!("[shady-thinker:{}][asr] no asr_prefix_cache — call precompute_prefix_cache first", self.gpu.role_tag);
                return Vec::new();
            }
        };

        crate::asr_decoder::gpu_asr_decode_tokens(
            &mut self.gpu, &mut self.model, prefix_cache,
            &encoder_output, enc_seq_len,
        )
    }

    /// Run Omni-style inference: mel → encoder → embed injection → decode.
    ///
    /// Uses forward_embed_kv_only for audio embeddings then standard
    /// forward() decode loop. For Qwen2.5-Omni multimodal models.
    pub fn omni_infer_mel(&mut self, mel_data: &[f32], mel_frames: u32) -> Vec<u32> {
        let encoder = match self.asr_encoder.as_mut() {
            Some(e) => e,
            None => {
                log::error!("[shady-thinker:{}] infer_mel called without ASR encoder loaded", self.gpu.role_tag);
                return Vec::new();
            }
        };

        let t0 = std::time::Instant::now();

        // Encoder: mel → hidden states
        let (encoder_output, enc_seq_len, _conv_ms, _transformer_ms) =
            encoder.forward_mel(mel_data, mel_frames);
        let enc_ms = t0.elapsed().as_millis();
        log::info!("[shady-thinker:{}][asr] encoder: {} frames → {} tokens in {}ms",
            self.gpu.role_tag, mel_frames, enc_seq_len, enc_ms);

        // Restore prefix (decoder system prompt KV cache)
        if self.prefix_snapshot.is_some() {
            self.restore_prefix_snapshot();
        } else {
            self.model.seq_len = self.prefix_len;
        }
        self.model.generated_tokens.clear();

        // Inject encoder output embeddings into decoder KV cache.
        // All except last via kv_only, last via full forward (produces first decode token).
        let h = self.config.hidden_size as usize;
        let n_enc = enc_seq_len as usize;
        for (i, chunk) in encoder_output.chunks_exact(h).enumerate() {
            if i < n_enc - 1 {
                self.model.forward_embed_kv_only(&mut self.gpu, chunk);
                if (i + 1) % 4 == 0 { self.gpu.flush_and_wait(); }
            }
        }
        self.gpu.flush_and_wait();

        // Inject <|audio_start|> token before audio embeddings
        self.model.forward_kv_only(&mut self.gpu, 151669); // <|audio_start|>

        // Inject last encoder embedding
        let last_embed = &encoder_output[(n_enc - 1) * h..n_enc * h];
        self.model.forward_embed_kv_only(&mut self.gpu, last_embed);
        self.gpu.flush_and_wait();

        // Suffix: <|audio_end|><|im_end|>\n<|im_start|>assistant\n<asr_text>
        let suffix_ids: &[u32] = &[
            151670,  // <|audio_end|>
            151645,  // <|im_end|>
            198,     // \n
            151644,  // <|im_start|>
            77091,   // assistant
            198,     // \n
            151704,  // <asr_text>
        ];
        for &tok in &suffix_ids[..suffix_ids.len() - 1] {
            self.model.forward_kv_only(&mut self.gpu, tok);
        }
        self.gpu.flush_and_wait();

        // Last suffix token through full forward to produce first decode token
        let t1 = std::time::Instant::now();
        let mut token = self.model.forward(&mut self.gpu, suffix_ids[suffix_ids.len() - 1]);
        let eos_ids = [151643u32, 151645]; // <|endoftext|>, <|im_end|>

        let mut token_ids = Vec::new();
        let max_tokens = 448u32;
        for _ in 0..max_tokens {
            if eos_ids.contains(&token) { break; }
            token_ids.push(token);
            token = self.model.forward(&mut self.gpu, token);
        }

        let dec_ms = t1.elapsed().as_millis();
        log::info!("[shady-thinker:{}][asr] decode: {} tokens in {}ms ({:.1} tok/s)",
            self.gpu.role_tag, token_ids.len(), dec_ms,
            token_ids.len() as f64 / (dec_ms as f64 / 1000.0).max(0.001));

        token_ids
    }

    /// Enable JSON-constrained sampling for subsequent generate calls.
    /// Uploads first-byte table to GPU; gate_byte in the penalty uniform selects the active gate.
    /// Pass stop/EOS token IDs so they are suppressed from sampling until JSON is complete.
    pub fn enable_json_mode(&mut self, token_bytes: Vec<Vec<u8>>, eos_ids: Vec<u32>) {
        // Build first_bytes as u32 array (one per token, 0 = empty/special).
        let vocab = self.model.config.vocab_size as usize;
        let mut fb = vec![0u32; vocab];
        for (i, bytes) in token_bytes.iter().enumerate() {
            if i < vocab {
                fb[i] = bytes.first().copied().unwrap_or(0) as u32;
            }
        }
        let fb_bytes: Vec<u8> = fb.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write_buffer(&self.model.state.first_bytes_buf, 0, &fb_bytes);
        let mut sampler = crate::json_sampler::JsonSampler::new(token_bytes, eos_ids);
        sampler.set_min_keys(2);
        sampler.enable_schema();
        self.model.json_sampler = Some(sampler);
    }

    /// Disable JSON-constrained sampling.
    pub fn disable_json_mode(&mut self) {
        self.model.json_sampler = None;
    }

    /// Prefill the KV cache with a fixed prefix (e.g., system prompt).
    /// After this call, every generate() starts from prefix_len — the prefix
    /// is never re-run. Call once after model load, before serving requests.
    /// Capture the current GPU state (KV caches + DeltaNet hist/state) into an
    /// in-memory snapshot.  Called after set_prefix() and after loading a prefix
    /// cache from disk so each generate() can restore from this snapshot cheaply.
    pub(crate) fn capture_prefix_snapshot(&mut self) {
        let nkv  = self.config.num_key_value_heads as u64;
        let hd   = self.config.head_dim as u64;
        let plen = self.prefix_len as u64;
        let sa_stride   = (plen * nkv * hd * 4) as usize;

        let dn_nhv = self.config.linear_num_value_heads as u64;
        let dn_kd  = self.config.linear_key_head_dim as u64;
        let dn_vd  = self.config.linear_value_head_dim as u64;
        let dn_nkh = self.config.linear_num_key_heads as u64;
        let dn_total_ch = dn_nkh * dn_kd * 2 + dn_nhv * dn_vd;
        let hist_bytes  = (3 * dn_total_ch * 4) as usize;
        let state_bytes = (dn_nhv * dn_kd * dn_vd * 4) as usize;

        let timeout = std::time::Duration::from_secs(30);
        let attn_layers = self.model.weights.self_attn_layers.clone();
        let num_layers  = self.model.weights.layers.len();

        let mut kv = vec![(Vec::new(), Vec::new()); num_layers];
        for &li in &attn_layers {
            let k = self.gpu.try_read_buffer_offset(
                &self.model.state.k_cache[li], 0, sa_stride as u64, timeout)
                .unwrap_or_default();
            let v = self.gpu.try_read_buffer_offset(
                &self.model.state.v_cache[li], 0, sa_stride as u64, timeout)
                .unwrap_or_default();
            kv[li] = (k, v);
        }

        let num_linear = self.model.state.deltanet_hist.len();
        let mut dn = Vec::with_capacity(num_linear);
        for lin_idx in 0..num_linear {
            let h = self.gpu.try_read_buffer_offset(
                &self.model.state.deltanet_hist[lin_idx], 0, hist_bytes as u64, timeout)
                .unwrap_or_default();
            let s = self.gpu.try_read_buffer_offset(
                &self.model.state.deltanet_state[lin_idx], 0, state_bytes as u64, timeout)
                .unwrap_or_default();
            dn.push((h, s));
        }

        self.prefix_snapshot = Some(PrefixSnapshot { kv, dn });
        log::info!("[shady-thinker:{}] prefix snapshot captured (sa={} la={})",
            self.gpu.role_tag, attn_layers.len(), num_linear);
    }

    /// Restore the prefix snapshot to GPU buffers and reset seq_len to prefix_len.
    /// After this, the model is ready to process query tokens incrementally.
    pub fn restore_prefix_snapshot(&mut self) {
        let snap = match self.prefix_snapshot.as_ref() {
            Some(s) => s,
            None => return,
        };
        let attn_layers = self.model.weights.self_attn_layers.clone();
        for &li in &attn_layers {
            let (k, v) = &snap.kv[li];
            if !k.is_empty() {
                self.gpu.write_buffer(&self.model.state.k_cache[li], 0, k);
                self.gpu.write_buffer(&self.model.state.v_cache[li], 0, v);
            }
        }
        for (lin_idx, (h, s)) in snap.dn.iter().enumerate() {
            self.gpu.write_buffer(&self.model.state.deltanet_hist[lin_idx],  0, h);
            self.gpu.write_buffer(&self.model.state.deltanet_state[lin_idx], 0, s);
        }
        self.gpu.flush();
        self.model.seq_len = self.prefix_len;
    }

    pub fn set_prefix(&mut self, ids: &[u32]) {
        if ids.is_empty() { return; }
        // Reset to clean state first.
        self.model.seq_len = 0;
        self.model.generated_tokens.clear();
        self.model.seen_bitmap_cpu.iter_mut().for_each(|w| *w = 0);
        let zero_bitmap = vec![0u8; self.model.seen_bitmap_cpu.len() * 4];
        self.gpu.write_buffer(&self.model.state.seen_bitmap, 0, &zero_bitmap);

        // Above this threshold the batched gptq prefill submits a
        // single huge command list that the wgpu queue can't drain
        // within its 30s flush_and_wait budget on integrated GPUs
        // (Apple Metal in particular). Falling through to per-token
        // forward_kv_only is much slower wall-clock but completes
        // reliably on any input length.
        const PREFILL_BATCH_LIMIT: usize = 1024;

        let t0 = std::time::Instant::now();
        let use_per_token = self.model.bf16_mode || ids.len() > PREFILL_BATCH_LIMIT;
        if use_per_token {
            log::info!("[shady-thinker:{}] set_prefix: {} tokens, per-token path (bf16={}, over_batch_limit={})",
                self.gpu.role_tag, ids.len(), self.model.bf16_mode,
                ids.len() > PREFILL_BATCH_LIMIT);
            for (i, &tok) in ids.iter().enumerate() {
                self.model.forward_kv_only(&mut self.gpu, tok);
                if (i + 1) % 4 == 0 {
                    self.gpu.flush_and_wait();
                }
            }
            self.gpu.flush_and_wait();
        } else {
            // Batched GPTQ prefill: processes all tokens in one GPU pass per layer.
            // Handles both self-attn and DeltaNet linear-attn layers.
            self.model.prefill_gptq(&mut self.gpu, ids);
        }

        self.prefix_len = self.model.seq_len;
        log::info!("[shady-thinker:{}] prefix cached: {} tokens in {:.1}s (seq_len={})",
            self.gpu.role_tag, ids.len(), t0.elapsed().as_secs_f32(), self.prefix_len);

        // Ensure all GPU writes are complete before reading back for snapshot.
        self.gpu.flush_and_wait();
        self.capture_prefix_snapshot();
    }

    #[cfg(feature = "jit-lora")]
    pub fn load_lora(&mut self, path: &std::path::Path) {
        let lora = LoraState::load_safetensors(&self.gpu, path, &self.config);
        self.model.lora = Some(lora);
    }

    /// Generate tokens (no think injection, no epiphanies).
    pub fn generate_tokens(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> GenerateResult {
        self.generate_inner(input_ids, max_tokens, eos_ids, cancel, false, None)
    }

    /// Generate with think token injection (reasoning mode, no epiphanies).
    /// Build a cache-key path for the prefix cache file.
    /// Encodes the model (via shader hash) and prompt content so a cache is
    /// invalidated if either changes.
    pub fn prefix_cache_path(model_dir: &std::path::Path, prompt: &str) -> std::path::PathBuf {
        Self::prefix_cache_path_with_extra(model_dir, prompt, 0)
    }

    /// Build a cache-key path with an extra hash component (e.g. external tool definitions).
    /// The extra_hash is mixed into the prompt hash so the cache invalidates when
    /// either the prompt text or external config changes.
    pub fn prefix_cache_path_with_extra(
        model_dir: &std::path::Path, prompt: &str, extra_hash: u64,
    ) -> std::path::PathBuf {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let model_hash = shader_cache_key();
        let mut h = DefaultHasher::new();
        prompt.hash(&mut h);
        extra_hash.hash(&mut h);
        let prompt_hash = h.finish();
        model_dir.join(format!("prefix_cache_{model_hash:016x}_{prompt_hash:016x}.bin"))
    }

    /// Save the prefix cache (KV cache + DeltaNet hist/state) to a binary file.
    /// Version 2 format includes both self-attn KV caches and DeltaNet recurrent state.
    /// Call after set_prefix() completes.
    pub fn save_prefix_cache(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let nkv  = self.config.num_key_value_heads as u64;
        let hd   = self.config.head_dim as u64;
        let plen = self.prefix_len as u64;

        // DeltaNet dimensions (from model config — same source used at init time)
        let dn_nhv = self.config.linear_num_value_heads as u64;
        let dn_kd  = self.config.linear_key_head_dim as u64;
        let dn_vd  = self.config.linear_value_head_dim as u64;
        let dn_nkh = self.config.linear_num_key_heads as u64;
        let dn_total_ch = dn_nkh * dn_kd * 2 + dn_nhv * dn_vd;

        let attn_layers = self.model.weights.self_attn_layers.clone();
        let num_linear  = self.model.state.deltanet_hist.len();
        let sa_stride   = plen * nkv * hd * 4;                   // bytes per K or V cache
        let hist_bytes  = 3 * dn_total_ch * 4;                   // bytes per hist buffer
        let state_bytes = dn_nhv * dn_kd * dn_vd * 4;            // bytes per state buffer

        let mut buf = Vec::with_capacity(
            44 + attn_layers.len() * (4 + 2 * sa_stride as usize)
               + num_linear * (4 + hist_bytes as usize + state_bytes as usize),
        );

        // ── Header (44 bytes) ──
        buf.write_all(&0xCA5E_CAFE_u32.to_le_bytes())?;          // [0]  magic
        buf.write_all(&2_u32.to_le_bytes())?;                     // [4]  version = 2
        buf.write_all(&(self.prefix_len).to_le_bytes())?;         // [8]  prefix_len
        buf.write_all(&(attn_layers.len() as u32).to_le_bytes())?;// [12] n_sa_layers
        buf.write_all(&(nkv as u32).to_le_bytes())?;              // [16] num_kv_heads
        buf.write_all(&(hd  as u32).to_le_bytes())?;              // [20] head_dim
        buf.write_all(&(num_linear as u32).to_le_bytes())?;       // [24] n_la_layers
        buf.write_all(&(dn_total_ch as u32).to_le_bytes())?;      // [28] dn_total_channels
        buf.write_all(&(dn_nhv as u32).to_le_bytes())?;           // [32] dn_num_value_heads
        buf.write_all(&(dn_kd  as u32).to_le_bytes())?;           // [36] dn_key_dim
        buf.write_all(&(dn_vd  as u32).to_le_bytes())?;           // [40] dn_value_dim

        let readback_timeout = std::time::Duration::from_secs(20);

        // ── Self-attn KV caches ──
        for &li in &attn_layers {
            buf.write_all(&(li as u32).to_le_bytes())?;
            let k = self.gpu.try_read_buffer_offset(&self.model.state.k_cache[li], 0, sa_stride, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            let v = self.gpu.try_read_buffer_offset(&self.model.state.v_cache[li], 0, sa_stride, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            buf.write_all(&k)?;
            buf.write_all(&v)?;
        }

        // ── DeltaNet hist + state ──
        // lin_idx is 0-based sequential index over linear-attn layers (same order as model).
        // We store the global layer index for integrity checking on load.
        let mut lin_idx = 0usize;
        for layer_idx in 0..self.model.weights.layers.len() {
            if attn_layers.contains(&layer_idx) { continue; }
            if lin_idx >= num_linear { break; }
            buf.write_all(&(layer_idx as u32).to_le_bytes())?;
            let hist = self.gpu.try_read_buffer_offset(
                &self.model.state.deltanet_hist[lin_idx], 0, hist_bytes, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            let state = self.gpu.try_read_buffer_offset(
                &self.model.state.deltanet_state[lin_idx], 0, state_bytes, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            buf.write_all(&hist)?;
            buf.write_all(&state)?;
            lin_idx += 1;
        }

        std::fs::write(path, &buf)?;
        log::info!("[shady-thinker:{}] saved prefix cache v2: {} ({} bytes, {} sa + {} la layers)",
            self.gpu.role_tag, path.display(), buf.len(), attn_layers.len(), num_linear);
        Ok(())
    }

    /// Save the full conversation KV state (up to `model.seq_len` tokens).
    /// Same binary v2 format as prefix cache — the loader doesn't distinguish.
    pub fn save_session_cache(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let nkv  = self.config.num_key_value_heads as u64;
        let hd   = self.config.head_dim as u64;
        let slen = self.model.seq_len as u64;

        let dn_nhv = self.config.linear_num_value_heads as u64;
        let dn_kd  = self.config.linear_key_head_dim as u64;
        let dn_vd  = self.config.linear_value_head_dim as u64;
        let dn_nkh = self.config.linear_num_key_heads as u64;
        let dn_total_ch = dn_nkh * dn_kd * 2 + dn_nhv * dn_vd;

        let attn_layers = self.model.weights.self_attn_layers.clone();
        let num_linear  = self.model.state.deltanet_hist.len();
        let sa_stride   = slen * nkv * hd * 4;
        let hist_bytes  = 3 * dn_total_ch * 4;
        let state_bytes = dn_nhv * dn_kd * dn_vd * 4;

        let mut buf = Vec::with_capacity(
            44 + attn_layers.len() * (4 + 2 * sa_stride as usize)
               + num_linear * (4 + hist_bytes as usize + state_bytes as usize),
        );

        buf.write_all(&0xCA5E_CAFE_u32.to_le_bytes())?;
        buf.write_all(&2_u32.to_le_bytes())?;
        buf.write_all(&(self.model.seq_len).to_le_bytes())?;
        buf.write_all(&(attn_layers.len() as u32).to_le_bytes())?;
        buf.write_all(&(nkv as u32).to_le_bytes())?;
        buf.write_all(&(hd  as u32).to_le_bytes())?;
        buf.write_all(&(num_linear as u32).to_le_bytes())?;
        buf.write_all(&(dn_total_ch as u32).to_le_bytes())?;
        buf.write_all(&(dn_nhv as u32).to_le_bytes())?;
        buf.write_all(&(dn_kd  as u32).to_le_bytes())?;
        buf.write_all(&(dn_vd  as u32).to_le_bytes())?;

        let readback_timeout = std::time::Duration::from_secs(20);

        for &li in &attn_layers {
            buf.write_all(&(li as u32).to_le_bytes())?;
            let k = self.gpu.try_read_buffer_offset(&self.model.state.k_cache[li], 0, sa_stride, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            let v = self.gpu.try_read_buffer_offset(&self.model.state.v_cache[li], 0, sa_stride, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            buf.write_all(&k)?;
            buf.write_all(&v)?;
        }

        let mut lin_idx = 0usize;
        for layer_idx in 0..self.model.weights.layers.len() {
            if attn_layers.contains(&layer_idx) { continue; }
            if lin_idx >= num_linear { break; }
            buf.write_all(&(layer_idx as u32).to_le_bytes())?;
            let hist = self.gpu.try_read_buffer_offset(
                &self.model.state.deltanet_hist[lin_idx], 0, hist_bytes, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            let state = self.gpu.try_read_buffer_offset(
                &self.model.state.deltanet_state[lin_idx], 0, state_bytes, readback_timeout)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "GPU readback timed out"))?;
            buf.write_all(&hist)?;
            buf.write_all(&state)?;
            lin_idx += 1;
        }

        std::fs::write(path, &buf)?;
        log::info!("[shady-thinker:{}] saved session cache: {} ({} bytes, seq_len={}, {} sa + {} la layers)",
            self.gpu.role_tag, path.display(), buf.len(), self.model.seq_len, attn_layers.len(), num_linear);
        Ok(())
    }

    /// Try to load a prefix cache built by save_prefix_cache().
    /// Returns true and sets prefix_len on success; returns false if the file is
    /// absent, invalid, or has mismatched dimensions.
    pub fn try_load_prefix_cache(&mut self, path: &std::path::Path) -> bool {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mmap = match unsafe { memmap2::Mmap::map(&file) } {
            Ok(m) => m,
            Err(e) => {
                log::warn!("[shady-thinker] mmap failed for {:?}: {e}", path);
                return false;
            }
        };
        log::info!("[shady-thinker] loaded prefix cache file: {} ({} bytes)", path.display(), mmap.len());
        self.try_load_prefix_cache_bytes(&mmap)
    }

    /// Load a prefix cache from raw bytes (e.g. embedded via `include_bytes!`).
    /// Returns true and sets prefix_len on success.
    pub fn try_load_prefix_cache_bytes(&mut self, data: &[u8]) -> bool {
        if data.len() < 24 { return false; }
        let magic   = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());

        if magic != 0xCA5E_CAFE { log::warn!("[shady-thinker] prefix cache: bad magic"); return false; }

        match version {
            2 => self.try_load_prefix_cache_v2(data),
            v => { log::warn!("[shady-thinker] prefix cache: unsupported version {v}"); false }
        }
    }

    fn try_load_prefix_cache_v2(&mut self, data: &[u8]) -> bool {
        if data.len() < 44 { return false; }
        let plen        = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let n_sa_layers = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let nkv         = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let hd          = u32::from_le_bytes(data[20..24].try_into().unwrap());
        let n_la_layers = u32::from_le_bytes(data[24..28].try_into().unwrap());
        let dn_total_ch = u32::from_le_bytes(data[28..32].try_into().unwrap());
        let dn_nhv      = u32::from_le_bytes(data[32..36].try_into().unwrap());
        let dn_kd       = u32::from_le_bytes(data[36..40].try_into().unwrap());
        let dn_vd       = u32::from_le_bytes(data[40..44].try_into().unwrap());

        // Validate self-attn dimensions
        if nkv != self.config.num_key_value_heads || hd != self.config.head_dim {
            log::warn!("[shady-thinker] prefix cache v2: KV dim mismatch ({nkv}×{hd} vs {}×{})",
                self.config.num_key_value_heads, self.config.head_dim);
            return false;
        }
        // Validate DeltaNet dimensions
        let exp_nhv = self.config.linear_num_value_heads;
        let exp_kd  = self.config.linear_key_head_dim;
        let exp_vd  = self.config.linear_value_head_dim;
        let exp_nkh = self.config.linear_num_key_heads;
        let exp_ch  = exp_nkh * exp_kd * 2 + exp_nhv * exp_vd;
        if dn_nhv != exp_nhv || dn_kd != exp_kd || dn_vd != exp_vd || dn_total_ch != exp_ch {
            log::warn!("[shady-thinker] prefix cache v2: DeltaNet dim mismatch");
            return false;
        }

        let sa_stride   = plen as usize * nkv as usize * hd as usize * 4;
        let hist_bytes  = 3 * dn_total_ch as usize * 4;
        let state_bytes = dn_nhv as usize * dn_kd as usize * dn_vd as usize * 4;

        let num_layers  = self.model.weights.layers.len();
        let mut snap_kv = vec![(Vec::new(), Vec::new()); num_layers];
        let mut snap_dn = Vec::with_capacity(n_la_layers as usize);

        let mut pos = 44usize;

        // ── Self-attn KV caches ──
        for _ in 0..n_sa_layers {
            if pos + 4 + 2 * sa_stride > data.len() {
                log::warn!("[shady-thinker] prefix cache v2: truncated (sa section)");
                return false;
            }
            let li = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if li >= self.model.state.k_cache.len() {
                log::warn!("[shady-thinker] prefix cache v2: sa layer {li} out of range");
                return false;
            }
            self.gpu.write_buffer(&self.model.state.k_cache[li], 0, &data[pos..pos+sa_stride]);
            snap_kv[li].0 = data[pos..pos+sa_stride].to_vec();
            pos += sa_stride;
            self.gpu.write_buffer(&self.model.state.v_cache[li], 0, &data[pos..pos+sa_stride]);
            snap_kv[li].1 = data[pos..pos+sa_stride].to_vec();
            pos += sa_stride;
        }

        // ── DeltaNet hist + state ──
        let num_linear = self.model.state.deltanet_hist.len();
        for lin_idx in 0..(n_la_layers as usize) {
            if pos + 4 + hist_bytes + state_bytes > data.len() {
                log::warn!("[shady-thinker] prefix cache v2: truncated (la section)");
                return false;
            }
            let _layer_idx = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            if lin_idx >= num_linear {
                log::warn!("[shady-thinker] prefix cache v2: lin_idx {lin_idx} out of range");
                return false;
            }
            self.gpu.write_buffer(&self.model.state.deltanet_hist[lin_idx],  0, &data[pos..pos+hist_bytes]);
            let h = data[pos..pos+hist_bytes].to_vec();
            pos += hist_bytes;
            self.gpu.write_buffer(&self.model.state.deltanet_state[lin_idx], 0, &data[pos..pos+state_bytes]);
            let s = data[pos..pos+state_bytes].to_vec();
            pos += state_bytes;
            snap_dn.push((h, s));
        }

        self.gpu.flush_and_wait();
        self.model.seq_len = plen;
        self.prefix_len = plen;
        self.prefix_snapshot = Some(PrefixSnapshot { kv: snap_kv, dn: snap_dn });
        log::info!("[shady-thinker] loaded prefix cache v2: prefix_len={plen} \
            ({n_sa_layers} sa + {n_la_layers} la layers, {} bytes)", data.len());
        true
    }

    pub fn generate_tokens_thinking(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> GenerateResult {
        self.generate_inner(input_ids, max_tokens, eos_ids, cancel, true, None)
    }

    /// Generate with think injection + RLM epiphanies.
    /// The dispatcher handles tool call detection, execution, and tokenization.
    pub fn generate_with_epiphanies(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
        dispatcher: &dyn EpiphanyDispatcher,
    ) -> GenerateResult {
        self.generate_inner(input_ids, max_tokens, eos_ids, cancel, true, Some(dispatcher))
    }

    fn generate_inner(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
        inject_think: bool,
        dispatcher: Option<&dyn EpiphanyDispatcher>,
    ) -> GenerateResult {
        if input_ids.is_empty() {
            return GenerateResult {
                token_ids: Vec::new(), token_count: 0,
                tokens_per_sec: 0.0, state: GenerateState::Complete, epiphany_count: 0,
            };
        }

        log::info!("[shady-thinker:{}] generate: {} input tokens, max_tokens={}, prefix_snap={}, prefix_len={}",
            self.gpu.role_tag, input_ids.len(), max_tokens, self.prefix_snapshot.is_some(), self.prefix_len);

        // Reset KV cache to prefix boundary (0 if no prefix is set) and clear sampling state.
        // Always restore prefix snapshot when available — ensures DeltaNet state is clean.
        if self.prefix_snapshot.is_some() {
            self.restore_prefix_snapshot();
            log::info!("[shady-thinker:{}] prefix restored, seq_len={}", self.gpu.role_tag, self.model.seq_len);
        } else {
            self.model.seq_len = self.prefix_len;
            log::info!("[shady-thinker:{}] no snapshot, seq_len reset to {}", self.gpu.role_tag, self.prefix_len);
        }
        self.model.generated_tokens.clear();
        if let Some(ref mut js) = self.model.json_sampler {
            js.reset();
        }
        // Clear seen-token bitmap (CPU side); GPU side cleared lazily via full zero-upload
        self.model.seen_bitmap_cpu.iter_mut().for_each(|w| *w = 0);
        let zero_bitmap = vec![0u8; self.model.seen_bitmap_cpu.len() * 4];
        self.gpu.write_buffer(&self.model.state.seen_bitmap, 0, &zero_bitmap);

        let prefill_start = std::time::Instant::now();

        // Think injection
        // Reset JSON sampler for new generation (schema FST + state machine)
        if let Some(ref mut js) = self.model.json_sampler {
            js.reset();
        }

        let mut generated = Vec::new();
        let bf16 = self.model.bf16_mode;
        let hybrid = self.model.is_hybrid_attn();
        // Batched GPTQ prefill: processes all input tokens in one GPU pass per layer.
        // Works with think-injection now that escalate() uses the full system prompt
        // (matching the prefix cache), so input_ids > prefix_len and DeltaNet gets
        // the full token stream from scratch.
        let use_gptq = !bf16;
        log::info!("[shady-thinker:{}] prefill-path: {} tokens, bf16={} inject_think={} hybrid={} → {}",
            self.gpu.role_tag, input_ids.len(), bf16, inject_think, hybrid,
            if use_gptq { "gptq-batch" } else { "kv-only-loop" });
        let first_decode_token = if use_gptq {
            // ── Phase 1: prefill ──
            let use_incremental = self.prefix_len > 0
                && self.prefix_snapshot.is_some()
                && input_ids.len() > self.prefix_len as usize;

            if use_incremental {
                // Restore prefix state, then run only the query tokens token-by-token.
                // Much faster than re-running the full context (e.g., 12 tokens vs 552).
                self.restore_prefix_snapshot();
                let query_ids = &input_ids[self.prefix_len as usize..];
                let n_query = query_ids.len();
                // Use forward_kv_only for ALL query tokens to avoid double-sampling:
                // forward() calls sample_token_gpu() which penalizes logits in-place;
                // calling sample_first_decode_token() after would apply penalty twice.
                for (i, &tok) in query_ids.iter().enumerate() {
                    self.model.forward_kv_only(&mut self.gpu, tok);
                    if (i + 1) % 4 == 0 { self.gpu.flush_and_wait(); }
                }
                self.gpu.flush_and_wait();
                // Dispatch lm_head onto the last token's hidden state (in state.normed).
                self.model.dispatch_lm_head(&mut self.gpu);
                let prefill_ms = prefill_start.elapsed().as_millis();
                log::info!("[shady-thinker:{}] incremental prefill: {} query tokens in {}ms ({:.1} tok/s)",
                    self.gpu.role_tag, n_query, prefill_ms,
                    n_query as f64 / (prefill_ms as f64 / 1000.0).max(0.001));
            } else {
                // Full prefill from scratch (no prefix snapshot or first run).
                // Same guard as set_prefix: huge batches stall the wgpu
                // queue past 30s, so for inputs over the limit we fall
                // back to the per-token path.
                const FULL_PREFILL_BATCH_LIMIT: usize = 1024;
                if input_ids.len() > FULL_PREFILL_BATCH_LIMIT {
                    log::info!("[shady-thinker:{}] full prefill: {} tokens > {} limit — using per-token path",
                        self.gpu.role_tag, input_ids.len(), FULL_PREFILL_BATCH_LIMIT);
                    for (i, &tok) in input_ids[..input_ids.len() - 1].iter().enumerate() {
                        self.model.forward_kv_only(&mut self.gpu, tok);
                        if (i + 1) % 4 == 0 { self.gpu.flush_and_wait(); }
                    }
                    self.gpu.flush_and_wait();
                    // Last token through full forward to populate lm_head logits.
                    self.model.forward(&mut self.gpu, input_ids[input_ids.len() - 1]);
                    self.gpu.flush_and_wait();
                } else {
                    self.model.prefill_gptq(&mut self.gpu, input_ids);
                }
                let prefill_ms = prefill_start.elapsed().as_millis();
                log::info!("[shady-thinker:{}] prefill_gptq: {} tokens in {}ms ({:.1} tok/s)",
                    self.gpu.role_tag, input_ids.len(), prefill_ms,
                    input_ids.len() as f64 / (prefill_ms as f64 / 1000.0).max(0.001));
            }

            // ── Phase 2 (optional): think token injection ──
            // Gated: only compiled and executed when think-injection feature is enabled.
            let first_decode;
            #[cfg(feature = "think-injection")]
            {
                first_decode = if inject_think {
                    if let Some(ref tc) = self.think_config {
                        let think_ids = tc.think_prefix_ids.clone();
                        for &tok in &think_ids[..think_ids.len().saturating_sub(1)] {
                            self.model.forward(&mut self.gpu, tok);
                        }
                        generated.extend_from_slice(&think_ids);
                        *think_ids.last().unwrap_or(&input_ids[input_ids.len() - 1])
                    } else {
                        self.model.sample_first_decode_token(&mut self.gpu)
                    }
                } else {
                    // ── Phase 3: sample first decode token ──
                    self.model.sample_first_decode_token(&mut self.gpu)
                };
            }
            #[cfg(not(feature = "think-injection"))]
            {
                // ── Phase 3: sample first decode token (fast path) ──
                first_decode = self.model.sample_first_decode_token(&mut self.gpu);
            }
            first_decode
        } else {
            // Fallback: bf16 mode or think-injection — token-by-token prefill.
            // Sync every 4 tokens to bound the GPU queue depth on PowerVR.
            // Each token submits its own command buffer (non-blocking flush in forward_kv_only);
            // flush_and_wait every 4 keeps at most 4 submissions pending at once.
            for (i, &tok) in input_ids[..input_ids.len() - 1].iter().enumerate() {
                self.model.forward_kv_only(&mut self.gpu, tok);
                if (i + 1) % 4 == 0 {
                    self.gpu.flush_and_wait();
                }
            }
            // Drain any remaining prefill submissions before starting decode.
            // The last batch may have fewer than 4 tokens with no flush_and_wait yet.
            self.gpu.flush_and_wait();
            let prefill_ms = prefill_start.elapsed().as_millis();
            log::info!("[shady-thinker:{}] prefill: {} tokens in {}ms", self.gpu.role_tag, input_ids.len() - 1, prefill_ms);

            if inject_think {
                if let Some(ref tc) = self.think_config {
                    let think_ids = tc.think_prefix_ids.clone();
                    self.model.forward(&mut self.gpu, input_ids[input_ids.len() - 1]);
                    for &tok in &think_ids[..think_ids.len().saturating_sub(1)] {
                        self.model.forward(&mut self.gpu, tok);
                    }
                    generated.extend_from_slice(&think_ids);
                    *think_ids.last().unwrap_or(&input_ids[input_ids.len() - 1])
                } else {
                    input_ids[input_ids.len() - 1]
                }
            } else {
                input_ids[input_ids.len() - 1]
            }
        };

        // For the batch-prefill path (use_gptq=true), first_decode_token is the *sampled first
        // output token* (from sample_first_decode_token after the LM head in prefill_gptq).
        // The kv-only path returns the last *input* token, so forward() below produces the first
        // output. Include first_decode_token in generated for the batch path only.
        if use_gptq && !eos_ids.contains(&first_decode_token) {
            generated.push(first_decode_token);
        }

        // Decode loop with epiphany support
        let decode_start = std::time::Instant::now();
        log::info!("[shady-thinker:{}] decode: starting first forward (seq={})", self.gpu.role_tag, self.model.seq_len);
        let mut token = self.model.forward(&mut self.gpu, first_decode_token);
        log::info!("[shady-thinker:{}] decode: first token={} ({:.0}ms)", self.gpu.role_tag, token, decode_start.elapsed().as_millis());
        let mut state = GenerateState::Complete;
        let mut epiphany_count = 0u32;
        let mut next_async_id = 1u32;
        // Track pending async tool calls: (call_id, injection_point)
        let mut _pending_async: Vec<(u32, usize)> = Vec::new();

        let marker = dispatcher.map(|d| d.tool_call_end_marker().to_vec());
        let marker_len = marker.as_ref().map(|m| m.len()).unwrap_or(0);

        // Tool call state machine (think-then-constrain for Fast thinker)
        let mut gen_state = if self.think_config.is_some() {
            GenState::Thinking
        } else {
            GenState::AwaitTool
        };
        let tc_config_start = self.tool_call_config.as_ref().map(|c| c.tool_call_start_ids.clone());
        let tc_config_end = self.tool_call_config.as_ref().map(|c| c.tool_call_end_ids.clone());
        let think_end = self.think_config.as_ref().map(|c| c.think_end_ids.clone());

        for _ in 0..max_tokens {
            if eos_ids.contains(&token) { break; }
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) { state = GenerateState::Interrupted; break; }
            }
            generated.push(token);

            // ── Tool call state machine ──
            if self.tool_call_config.is_some() {
                match gen_state {
                    GenState::Thinking => {
                        if let Some(ref end_ids) = think_end {
                            if ends_with_ids(&generated, end_ids) {
                                gen_state = GenState::AwaitTool;
                                log::info!("[gen-state] Thinking → AwaitTool (</think> detected)");
                            }
                        }
                    }
                    GenState::AwaitTool => {
                        if let Some(ref start_ids) = tc_config_start {
                            if ends_with_ids(&generated, start_ids) {
                                // Enable JSON constraint dynamically
                                if self.model.json_sampler.is_none() {
                                    // Re-use the pre-built first_bytes_buf (uploaded at enable_json_mode time)
                                    // For dynamic enable, we just need to set the sampler
                                    let sampler = crate::json_sampler::JsonSampler::new_unconstrained();
                                    self.model.json_sampler = Some(sampler);
                                }
                                gen_state = GenState::JsonConstrained;
                                log::info!("[gen-state] AwaitTool → JsonConstrained (<tool_call> detected)");
                            }
                        }
                    }
                    GenState::JsonConstrained => {
                        if let Some(ref end_ids) = tc_config_end {
                            if ends_with_ids(&generated, end_ids) {
                                // Tool call complete — stop immediately, free GPU
                                log::info!("[gen-state] JsonConstrained → Done (</tool_call>, stopping)");
                                break;
                            }
                        }
                    }
                }
            }

            // Check for tool call end marker
            if let (Some(ref marker), Some(dispatcher)) = (&marker, dispatcher) {
                if marker_len > 0 && generated.len() >= marker_len {
                    let tail = &generated[generated.len() - marker_len..];
                    if tail == &marker[..] {
                        // Decode to extract the tool call body
                        let check_start = generated.len().saturating_sub(500);
                        let text = dispatcher.decode(&generated[check_start..]);

                        if let Some(call_end) = text.rfind("</tool_call>") {
                            if let Some(call_start) = text[..call_end].rfind("<tool_call>") {
                                let call_body = text[call_start + 11..call_end].trim();
                                let cost = dispatcher.estimate_cost(call_body);

                                match cost {
                                    ToolCost::Instant => {
                                        // Execute inline, inject immediately
                                        let result = dispatcher.execute_sync(call_body);
                                        let response = format!("\n<tool_response>\n{}\n</tool_response>\n", result.response);
                                        let response_ids = dispatcher.tokenize(&response);

                                        // Truncate to avoid blowing context window — max 512 tokens per tool response.
                                        let max_inject = 512;
                                        let inject_ids = if response_ids.len() > max_inject {
                                            log::warn!("[epiphany] truncating tool response: {} → {} tokens", response_ids.len(), max_inject);
                                            &response_ids[..max_inject]
                                        } else {
                                            &response_ids[..]
                                        };
                                        log::info!("[epiphany] instant: injecting {} tokens", inject_ids.len());
                                        // Forward the injection payload to whatever transcript
                                        // sink the dispatcher carries (Conversation JSONL in
                                        // the chitin path; default no-op elsewhere). Pass the
                                        // raw response body — wrapping tags add no info.
                                        dispatcher.record_injection(&result.response);
                                        for &inj_tok in inject_ids {
                                            self.model.forward(&mut self.gpu, inj_tok);
                                        }
                                        generated.extend_from_slice(inject_ids);
                                        epiphany_count += 1;

                                        // Continue decode from last injected token
                                        token = self.model.forward(&mut self.gpu,
                                            *inject_ids.last().unwrap_or(&token));
                                        continue;
                                    }
                                    ToolCost::Async => {
                                        // Fire async, model keeps going
                                        let call_id = next_async_id;
                                        next_async_id += 1;
                                        dispatcher.fire_async(call_id, call_body);
                                        _pending_async.push((call_id, generated.len()));
                                        log::info!("[epiphany] async fired: call_id={}", call_id);
                                    }
                                }
                            }
                        }
                    }
                }

                // Poll for completed async results and inject
                if !_pending_async.is_empty() {
                    let results = dispatcher.poll_async_results();
                    let got_results = !results.is_empty();
                    for (call_id, result) in results {
                        _pending_async.retain(|&(id, _)| id != call_id);
                        let response = format!("\n<tool_response>\n{}\n</tool_response>\n", result.response);
                        let response_ids = dispatcher.tokenize(&response);

                        log::info!("[epiphany] async result: call_id={}, injecting {} tokens", call_id, response_ids.len());
                        dispatcher.record_injection(&result.response);
                        for &inj_tok in &response_ids {
                            self.model.forward(&mut self.gpu, inj_tok);
                        }
                        generated.extend_from_slice(&response_ids);
                        epiphany_count += 1;

                        token = self.model.forward(&mut self.gpu,
                            *response_ids.last().unwrap_or(&token));
                    }
                    if got_results { continue; }
                }
            }

            token = self.model.forward(&mut self.gpu, token);


            log::info!("[shady-thinker:{}] stream_tx: {:?}, stream_tokenizer: {:?}",
                self.gpu.role_tag,
                self.gpu.stream_tx.is_some(),
                self.gpu.stream_tokenizer.is_some());

            // Per-token streaming: if a sink + tokenizer were
            // installed on the GpuContext for this generation, decode
            // the new token and push its text. Fires every token (not
            // every 4) so SSE clients see motion at the model's
            // natural pace.
            if let (Some(ref tx), Some(ref tok)) = (&self.gpu.stream_tx, &self.gpu.stream_tokenizer) {
                let text = tok.decode(&[token], true).unwrap_or_default();
                if !text.is_empty() {
                    log::info!("[shady-thinker:{}] streaming token: {} → '{}'", self.gpu.role_tag, token, text);
                    let _ = tx.try_send(common::handles::StreamChunk {
                        delta_text: text,
                        finish_reason: None,
                    });
                }
            }
            if generated.len() % 4 == 0 {
                let ms = decode_start.elapsed().as_millis();
                log::info!("[shady-thinker:{}] decode: {} tokens, {:.1} tok/s, seq={}",
                    self.gpu.role_tag,
                    generated.len(),
                    generated.len() as f64 / (ms as f64 / 1000.0).max(0.001),
                    self.model.seq_len);
            }
        }

        let elapsed = decode_start.elapsed();
        let count = generated.len();
        let tps = if count > 0 { count as f64 / elapsed.as_secs_f64() } else { 0.0 };
        log::info!("[shady-thinker:{}] decode: {} tokens in {:.0}ms ({:.1} tok/s), {} epiphanies",
            self.gpu.role_tag, count, elapsed.as_millis(), tps, epiphany_count);

        GenerateResult {
            token_ids: generated,
            token_count: count,
            tokens_per_sec: tps,
            state,
            epiphany_count,
        }
    }
}

/// Check if the last N tokens of `generated` match `marker` exactly.
fn ends_with_ids(generated: &[u32], marker: &[u32]) -> bool {
    if marker.is_empty() || generated.len() < marker.len() {
        return false;
    }
    &generated[generated.len() - marker.len()..] == marker
}
