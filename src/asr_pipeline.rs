//! Fused ASR pipeline: mel → conv_stem (encoder) → decoder → tokens.
//!
//! Single entry point for speech-to-text on GPU. Owns encoder + decoder + prefix cache.

use std::path::Path;
use crate::gpu::GpuContext;
use crate::asr_encoder::AsrEncoder;
use crate::asr_decoder::{self, PrefixCache};
use crate::model::Model;

/// Result of a decode pass.
pub enum DecodeResult {
    /// No speech detected (first token was non-speech).
    NoSpeech {
        top3: Vec<(u32, f32)>,
        prefill_ms: u128,
    },
    /// Speech decoded successfully.
    Speech {
        token_ids: Vec<u32>,
        raw_ring: Vec<(u32, f32)>,
        first_token: (u32, f32),
        prefill_ms: u128,
        decode_ms: u128,
    },
}

/// Fused ASR pipeline owning all GPU resources.
pub struct AsrPipeline {
    gpu: GpuContext,      // encoder GPU context
    dec_gpu: GpuContext,  // decoder GPU context (owns model's buffers)
    encoder: AsrEncoder,
    model: Model,
    prefix_cache: Option<PrefixCache>,
}

impl AsrPipeline {
    /// Load encoder + decoder from model directory.
    pub fn load(model_dir: &Path) -> Self {
        let gpu = GpuContext::new();
        let encoder = AsrEncoder::load(GpuContext::from_device_queue(
            gpu.device.clone(), gpu.queue.clone(),
        ), model_dir);
        let (dec_gpu, model) = asr_decoder::load_bf16_model(model_dir, 256);

        // Precompute prefix cache for fast decode
        let prefix_cache = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut gpu_tmp, mut model_tmp) = asr_decoder::load_bf16_model(model_dir, 256);
            asr_decoder::precompute_prefix_cache(&mut gpu_tmp, &mut model_tmp, model_dir)
        })).ok();

        if prefix_cache.is_some() {
            log::info!("[asr-pipeline] prefix cache ready");
        } else {
            log::warn!("[asr-pipeline] prefix cache failed, using slow path");
        }

        Self { gpu, dec_gpu, encoder, model, prefix_cache }
    }

    /// Run the full pipeline: mel spectrogram → encoder → decoder → token IDs.
    ///
    /// `mel_data` is mel-bin-major: `[128 × n_frames]`.
    /// Returns `DecodeResult` with token IDs or no-speech indication.
    pub fn forward(&mut self, mel_data: &[f32], mel_frames: u32) -> DecodeResult {
        let t0 = std::time::Instant::now();

        // Conv stem (CPU) + encoder transformer (GPU) → hidden states
        let encoder_output = self.encoder.encode_mel(mel_data, mel_frames);
        let enc_seq_len = encoder_output.len() as u32 / self.model.config.hidden_size;
        let enc_ms = t0.elapsed().as_millis();

        // Decoder: hidden states → token IDs
        let t1 = std::time::Instant::now();
        let token_ids = if let Some(ref cache) = self.prefix_cache {
            asr_decoder::gpu_asr_decode_tokens(
                &mut self.dec_gpu, &mut self.model, cache,
                &encoder_output, enc_seq_len,
            )
        } else {
            // Slow path without prefix cache
            let text = asr_decoder::gpu_asr_decode(
                &mut self.dec_gpu, &mut self.model,
                &encoder_output, enc_seq_len,
            );
            // Can't get token_ids from string path — return as single-token placeholder
            log::warn!("[asr-pipeline] slow decode path, text={:?}", text);
            Vec::new()
        };
        let dec_ms = t1.elapsed().as_millis();

        log::info!("[asr-pipeline] enc={enc_ms}ms dec={dec_ms}ms tokens={}", token_ids.len());

        // Check for no-speech (empty or starts with no-speech token)
        if token_ids.is_empty() {
            return DecodeResult::NoSpeech {
                top3: Vec::new(),
                prefill_ms: enc_ms + dec_ms,
            };
        }

        let first_token = (token_ids[0], 0.0);
        DecodeResult::Speech {
            raw_ring: token_ids.iter().map(|&t| (t, 0.0)).collect(),
            first_token,
            prefill_ms: enc_ms,
            decode_ms: dec_ms,
            token_ids,
        }
    }
}
