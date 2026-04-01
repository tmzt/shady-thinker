//! Pure inference session — no tokenizer, no chat template.
//!
//! Loads model weights and provides token-ID-in, token-ID-out generation.
//! Tokenization and chat formatting are the caller's responsibility.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::GpuContext;
use crate::model::Model;
use crate::weights;
use crate::weights::ModelConfig;

#[cfg(feature = "jit-lora")]
use crate::lora::LoraState;

/// Result of a generation call.
pub struct GenerateResult {
    /// Generated token IDs (excludes prompt tokens).
    pub token_ids: Vec<u32>,
    /// Number of tokens generated.
    pub token_count: usize,
    /// Tokens per second.
    pub tokens_per_sec: f64,
    /// Whether generation was interrupted by cancel flag.
    pub interrupted: bool,
}

/// Configurable think token injection for reasoning models.
pub struct ThinkConfig {
    /// Token IDs to inject after prefill to enter thinking mode.
    /// For Qwen3.5: the tokenized `<think>\n` sequence.
    pub think_prefix_ids: Vec<u32>,
    /// Token ID(s) that mark end of thinking (e.g. `</think>`).
    /// Generation won't stop at these — they just mark the boundary.
    pub think_end_ids: Vec<u32>,
}

/// Pure inference session: model + GPU context.
///
/// No tokenizer, no chat template, no conversation log.
/// Caller provides token IDs and receives token IDs.
pub struct InferenceSession {
    pub model: Model,
    pub gpu: GpuContext,
    pub config: ModelConfig,
    /// Optional think token injection config.
    pub think_config: Option<ThinkConfig>,
}

impl InferenceSession {
    /// Load model from a directory containing safetensors + config.json + quantize_config.json.
    pub fn new(model_dir: PathBuf, max_seq_len: u32) -> Self {
        log::info!("[shady-thinker] loading model from {:?} (max_seq={})", model_dir, max_seq_len);
        let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
        let quant_config =
            weights::QuantConfig::from_file(&model_dir.join("quantize_config.json"));
        log::info!("[shady-thinker] config: {} layers, {} heads, dim={}",
            config.num_hidden_layers, config.num_attention_heads, config.hidden_size);

        let mut gpu = GpuContext::new();
        let (model_weights, raw_norms) = weights::load_weights(&gpu, &model_dir, &config);
        let mut model =
            Model::new(&gpu, config.clone(), quant_config, model_weights, max_seq_len);

        for (i, norm_data) in raw_norms.layers.iter().enumerate() {
            if let Some((q_bytes, k_bytes)) = norm_data {
                model.init_qknorm_params(&gpu, i, q_bytes, k_bytes);
            }
        }

        log::info!("[shady-thinker] model ready");
        Self { model, gpu, config, think_config: None }
    }

    /// Set think token injection config for reasoning models.
    pub fn set_think_config(&mut self, config: ThinkConfig) {
        self.think_config = Some(config);
    }

    /// Load a LoRA adapter from safetensors file.
    #[cfg(feature = "jit-lora")]
    pub fn load_lora(&mut self, path: &std::path::Path) {
        let lora = LoraState::load_safetensors(&self.gpu, path, &self.config);
        self.model.lora = Some(lora);
    }

    /// Generate from token IDs. Returns generated token IDs.
    ///
    /// Resets KV cache before generation. Prefills all input tokens,
    /// then generates up to `max_tokens` new tokens. Stops at any token
    /// in `eos_ids`.
    ///
    /// If `inject_think` is true and `think_config` is set, the think prefix
    /// tokens are injected after prefill to force the model into reasoning mode.
    pub fn generate_tokens(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> GenerateResult {
        self.generate_tokens_inner(input_ids, max_tokens, eos_ids, cancel, false)
    }

    /// Generate with think token injection (forces reasoning mode).
    pub fn generate_tokens_thinking(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> GenerateResult {
        self.generate_tokens_inner(input_ids, max_tokens, eos_ids, cancel, true)
    }

    fn generate_tokens_inner(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
        inject_think: bool,
    ) -> GenerateResult {
        if input_ids.is_empty() {
            return GenerateResult {
                token_ids: Vec::new(),
                token_count: 0,
                tokens_per_sec: 0.0,
                interrupted: false,
            };
        }

        log::info!("[shady-thinker] generate: {} input tokens, max_tokens={}", input_ids.len(), max_tokens);

        // Reset KV cache
        self.model.seq_len = 0;
        self.model.generated_tokens.clear();

        // Prefill: feed all but last token
        let prefill_start = std::time::Instant::now();
        for &tok in &input_ids[..input_ids.len() - 1] {
            self.model.forward(&mut self.gpu, tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis();
        log::info!("[shady-thinker] prefill: {} tokens in {}ms", input_ids.len() - 1, prefill_ms);

        // Inject think prefix tokens if requested
        if inject_think {
            if let Some(ref tc) = self.think_config {
                let think_ids = tc.think_prefix_ids.clone();
                log::info!("[shady-thinker] injecting {} think prefix tokens", think_ids.len());
                // Feed last input token + think prefix through prefill
                self.model.forward(&mut self.gpu, input_ids[input_ids.len() - 1]);
                for &tok in &think_ids[..think_ids.len().saturating_sub(1)] {
                    self.model.forward(&mut self.gpu, tok);
                }
                // Use last think token as the first decode input
                let decode_start = std::time::Instant::now();
                let last_think = *think_ids.last().unwrap_or(&input_ids[input_ids.len() - 1]);
                let mut token = self.model.forward(&mut self.gpu, last_think);
                let mut generated = Vec::new();
                // Include think prefix in output so caller can see <think>..
                for &t in &think_ids {
                    generated.push(t);
                }
                let mut interrupted = false;
                for _ in 0..max_tokens {
                    if eos_ids.contains(&token) { break; }
                    if let Some(flag) = cancel {
                        if flag.load(Ordering::Relaxed) { interrupted = true; break; }
                    }
                    generated.push(token);
                    token = self.model.forward(&mut self.gpu, token);
                }
                let elapsed = decode_start.elapsed();
                let count = generated.len();
                let tps = if count > 0 { count as f64 / elapsed.as_secs_f64() } else { 0.0 };
                log::info!("[shady-thinker] decode (think): {} tokens in {:.0}ms ({:.1} tok/s)",
                    count, elapsed.as_millis(), tps);
                return GenerateResult {
                    token_ids: generated, token_count: count,
                    tokens_per_sec: tps, interrupted,
                };
            }
        }

        // First decode step (normal path)
        let decode_start = std::time::Instant::now();
        let mut token = self.model.forward(&mut self.gpu, input_ids[input_ids.len() - 1]);
        let mut generated = Vec::new();
        let mut interrupted = false;

        for _ in 0..max_tokens {
            if eos_ids.contains(&token) {
                break;
            }
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    interrupted = true;
                    break;
                }
            }
            generated.push(token);
            token = self.model.forward(&mut self.gpu, token);
        }

        let elapsed = decode_start.elapsed();
        let count = generated.len();
        let tps = if count > 0 {
            count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        log::info!("[shady-thinker] decode: {} tokens in {:.0}ms ({:.1} tok/s){}",
            count, elapsed.as_millis(), tps,
            if interrupted { " [interrupted]" } else { "" });

        GenerateResult {
            token_ids: generated,
            token_count: count,
            tokens_per_sec: tps,
            interrupted,
        }
    }
}
