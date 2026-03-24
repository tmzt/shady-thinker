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

/// Pure inference session: model + GPU context.
///
/// No tokenizer, no chat template, no conversation log.
/// Caller provides token IDs and receives token IDs.
pub struct InferenceSession {
    pub model: Model,
    pub gpu: GpuContext,
    pub config: ModelConfig,
}

impl InferenceSession {
    /// Load model from a directory containing safetensors + config.json + quantize_config.json.
    pub fn new(model_dir: PathBuf, max_seq_len: u32) -> Self {
        let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
        let quant_config =
            weights::QuantConfig::from_file(&model_dir.join("quantize_config.json"));

        let mut gpu = GpuContext::new();
        let (model_weights, raw_norms) = weights::load_weights(&gpu, &model_dir, &config);
        let mut model =
            Model::new(&gpu, config.clone(), quant_config, model_weights, max_seq_len);

        for (i, norm_data) in raw_norms.layers.iter().enumerate() {
            if let Some((q_bytes, k_bytes)) = norm_data {
                model.init_qknorm_params(&gpu, i, q_bytes, k_bytes);
            }
        }

        Self { model, gpu, config }
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
    pub fn generate_tokens(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> GenerateResult {
        if input_ids.is_empty() {
            return GenerateResult {
                token_ids: Vec::new(),
                token_count: 0,
                tokens_per_sec: 0.0,
                interrupted: false,
            };
        }

        // Reset KV cache
        self.model.seq_len = 0;
        self.model.generated_tokens.clear();

        // Prefill: feed all but last token
        for &tok in &input_ids[..input_ids.len() - 1] {
            self.model.forward(&mut self.gpu, tok);
        }

        // First decode step
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

        GenerateResult {
            token_ids: generated,
            token_count: count,
            tokens_per_sec: tps,
            interrupted,
        }
    }
}
