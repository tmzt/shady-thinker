//! Pure inference session with RLM epiphany support.
//!
//! Provides token-ID-in, token-ID-out generation with mid-generation
//! tool execution. Tools are dispatched via an EpiphanyDispatcher trait.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::GpuContext;
use crate::model::Model;
use crate::weights;
use crate::weights::ModelConfig;

#[cfg(feature = "jit-lora")]
use crate::lora::LoraState;

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

/// Pure inference session: model + GPU context.
pub struct InferenceSession {
    pub model: Model,
    pub gpu: GpuContext,
    pub config: ModelConfig,
    pub think_config: Option<ThinkConfig>,
}

impl InferenceSession {
    pub fn new(model_dir: PathBuf, max_seq_len: u32) -> Self {
        let t0 = std::time::Instant::now();
        log::info!("[shady-thinker] loading model from {:?} (max_seq={})", model_dir, max_seq_len);

        log::info!("[shady-thinker] step 1/6: reading config.json");
        let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
        log::info!("[shady-thinker] step 2/6: reading quantize_config.json");
        let quant_config =
            weights::QuantConfig::from_file(&model_dir.join("quantize_config.json"));
        log::info!("[shady-thinker] config: {} layers, {} heads, dim={}, bits={}, group_size={} ({:.1}s)",
            config.num_hidden_layers, config.num_attention_heads, config.hidden_size,
            quant_config.bits, quant_config.group_size, t0.elapsed().as_secs_f32());

        log::info!("[shady-thinker] step 3/6: creating GPU context");
        let mut gpu = GpuContext::new();
        log::info!("[shady-thinker] GPU context ready ({:.1}s)", t0.elapsed().as_secs_f32());

        // Load Vulkan pipeline cache from disk (critical for fast startup on Android).
        // First run compiles all shaders from scratch and saves the binary cache.
        // Subsequent runs load the cached binary — avoiding expensive JIT compilation.
        let cache_path = model_dir.join("pipeline_cache.bin");
        if gpu.supports_pipeline_cache() {
            if let Ok(cached) = std::fs::read(&cache_path) {
                gpu.load_pipeline_cache(&cached);
            } else {
                gpu.create_pipeline_cache();
                log::info!("[shady-thinker] no pipeline cache found — will compile shaders (first run)");
            }
        }

        log::info!("[shady-thinker] step 4/6: loading weights from safetensors");
        let (model_weights, raw_norms) = weights::load_weights(&gpu, &model_dir, &config);
        log::info!("[shady-thinker] weights loaded ({:.1}s)", t0.elapsed().as_secs_f32());

        log::info!("[shady-thinker] step 5/6: creating model pipeline");
        let mut model =
            Model::new(&gpu, config.clone(), quant_config, model_weights, max_seq_len);
        log::info!("[shady-thinker] model pipeline created ({:.1}s)", t0.elapsed().as_secs_f32());

        log::info!("[shady-thinker] step 6/6: initializing QK norm params");
        for (i, norm_data) in raw_norms.layers.iter().enumerate() {
            if let Some((q_bytes, k_bytes)) = norm_data {
                model.init_qknorm_params(&gpu, i, q_bytes, k_bytes);
            }
        }

        log::info!("[shady-thinker] model ready ({:.1}s total)", t0.elapsed().as_secs_f32());
        let mut session = Self { model, gpu, config, think_config: None };

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

        // Save pipeline cache to disk if it was freshly compiled.
        if !cache_path.exists() {
            if let Some(data) = session.gpu.get_pipeline_cache_data() {
                if !data.is_empty() {
                    if let Err(e) = std::fs::write(&cache_path, &data) {
                        log::warn!("[shady-thinker] failed to save pipeline cache: {e}");
                    } else {
                        log::info!("[shady-thinker] saved pipeline cache ({} bytes)", data.len());
                    }
                }
            }
        }

        session
    }

    pub fn set_think_config(&mut self, config: ThinkConfig) {
        self.think_config = Some(config);
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

        log::info!("[shady-thinker] generate: {} input tokens, max_tokens={}",
            input_ids.len(), max_tokens);

        // Reset KV cache and sampling state
        self.model.seq_len = 0;
        self.model.generated_tokens.clear();
        // Clear seen-token bitmap (CPU side); GPU side cleared lazily via full zero-upload
        self.model.seen_bitmap_cpu.iter_mut().for_each(|w| *w = 0);
        let zero_bitmap = vec![0u8; self.model.seen_bitmap_cpu.len() * 4];
        self.gpu.write_buffer(&self.model.state.seen_bitmap, 0, &zero_bitmap);

        // Prefill — skip lm_head + logit readback for all but the final token.
        // forward_kv_only sets prefill_kv_only=true so forward_layers returns early,
        // avoiding 607KB GPU→CPU sync per token.
        let prefill_start = std::time::Instant::now();
        for &tok in &input_ids[..input_ids.len() - 1] {
            self.model.forward_kv_only(&mut self.gpu, tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis();
        log::info!("[shady-thinker] prefill: {} tokens in {}ms", input_ids.len() - 1, prefill_ms);

        // Think injection
        let mut generated = Vec::new();
        let first_decode_token = if inject_think {
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
        };

        // Decode loop with epiphany support
        let decode_start = std::time::Instant::now();
        let mut token = self.model.forward(&mut self.gpu, first_decode_token);
        let mut state = GenerateState::Complete;
        let mut epiphany_count = 0u32;
        let mut next_async_id = 1u32;
        // Track pending async tool calls: (call_id, injection_point)
        let mut _pending_async: Vec<(u32, usize)> = Vec::new();

        let marker = dispatcher.map(|d| d.tool_call_end_marker().to_vec());
        let marker_len = marker.as_ref().map(|m| m.len()).unwrap_or(0);

        for _ in 0..max_tokens {
            if eos_ids.contains(&token) { break; }
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) { state = GenerateState::Interrupted; break; }
            }
            generated.push(token);

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

                                        log::info!("[epiphany] instant: injecting {} tokens", response_ids.len());
                                        for &inj_tok in &response_ids {
                                            self.model.forward(&mut self.gpu, inj_tok);
                                        }
                                        generated.extend_from_slice(&response_ids);
                                        epiphany_count += 1;

                                        // Continue decode from last injected token
                                        token = self.model.forward(&mut self.gpu,
                                            *response_ids.last().unwrap_or(&token));
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
        }

        let elapsed = decode_start.elapsed();
        let count = generated.len();
        let tps = if count > 0 { count as f64 / elapsed.as_secs_f64() } else { 0.0 };
        log::info!("[shady-thinker] decode: {} tokens in {:.0}ms ({:.1} tok/s), {} epiphanies",
            count, elapsed.as_millis(), tps, epiphany_count);

        GenerateResult {
            token_ids: generated,
            token_count: count,
            tokens_per_sec: tps,
            state,
            epiphany_count,
        }
    }
}
