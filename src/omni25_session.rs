//! Unified audio+text inference session for Qwen2.5-Omni.
//!
//! Wraps InferenceSession + OmniAudioEncoder. Text-only mode uses the
//! standard generate path. Audio mode injects audio tower embeddings
//! inline as token embeddings before decoding.
//!
//! Token sequence for audio:
//!   [prefix KV (system prompt)] <|audio_bos|> [audio embeddings] <|audio_eos|> [suffix] → decode
//!
//! The audio embeddings bypass the token embedding table — each audio frame's
//! encoder output (dim=3584) is written directly to the hidden state buffer.

use std::path::{Path, PathBuf};
use crate::gpu::GpuContext;
use crate::inference::InferenceSession;
use crate::omni25_audio::{Omni25AudioEncoder, Omni25AudioConfig};

/// Special token IDs for Qwen2.5-Omni (from config.json).
pub struct OmniTokens {
    pub audio_bos: u32,    // 151647 <|audio_bos|>
    pub audio_eos: u32,    // 151648 <|audio_eos|>
    pub audio_placeholder: u32, // 151646 <|AUDIO|>
}

impl Default for OmniTokens {
    fn default() -> Self {
        Self {
            audio_bos: 151647,
            audio_eos: 151648,
            audio_placeholder: 151646,
        }
    }
}

/// Unified session holding both the text decoder and audio encoder.
pub struct Omni25Session {
    /// Text decoder (standard InferenceSession, handles generate_tokens etc.)
    pub session: InferenceSession,
    /// Audio tower encoder (loads separately, shares GPU device)
    pub audio: Option<Omni25AudioEncoder>,
    /// Special token IDs
    pub tokens: OmniTokens,
}

impl Omni25Session {
    /// Load the Omni model: text decoder + audio tower.
    /// The text decoder uses the existing InferenceSession loader.
    /// The audio tower is loaded separately from the same safetensors.
    pub fn new(model_dir: PathBuf, max_seq_len: u32) -> Self {
        log::info!("[omni25] loading model from {:?}", model_dir);

        // Load text decoder (reuses existing Qwen2.5 weight loader)
        let session = InferenceSession::new(model_dir.clone(), max_seq_len);

        // Parse special token IDs from config
        let tokens = Self::parse_tokens(&model_dir);

        // Load audio tower using the shared GPU context
        // Note: we can't borrow session.gpu mutably here because InferenceSession
        // owns it. The audio tower creates its own GPU context for now.
        // TODO: share GPU context between text decoder and audio tower
        log::info!("[omni25] loading audio tower...");
        let mut audio_gpu = GpuContext::from_device_queue(
            session.gpu.device.clone(),
            session.gpu.queue.clone(),
        );
        let audio = Omni25AudioEncoder::load(&mut audio_gpu, &model_dir);
        log::info!("[omni25] audio tower ready ({} layers)", audio.config.num_layers);

        Self {
            session,
            audio: Some(audio),
            tokens,
        }
    }

    /// Load text decoder only (no audio tower). For text-only testing.
    pub fn text_only(model_dir: PathBuf, max_seq_len: u32) -> Self {
        let session = InferenceSession::new(model_dir.clone(), max_seq_len);
        let tokens = Self::parse_tokens(&model_dir);
        Self { session, audio: None, tokens }
    }

    /// Parse special token IDs from config.json.
    fn parse_tokens(model_dir: &Path) -> OmniTokens {
        let config_path = model_dir.join("config.json");
        if let Ok(data) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&data) {
                let tc = config.get("thinker_config").unwrap_or(&config);
                return OmniTokens {
                    audio_bos: tc["audio_start_token_id"].as_u64().unwrap_or(151647) as u32,
                    audio_eos: tc["audio_end_token_id"].as_u64().unwrap_or(151648) as u32,
                    audio_placeholder: tc["audio_token_index"].as_u64().unwrap_or(151646) as u32,
                };
            }
        }
        OmniTokens::default()
    }

    /// Run audio transcription: mel → audio tower → inject into decoder → generate text.
    ///
    /// `mel_data`: [128 × n_frames] f32 (mel-bin-major)
    /// `suffix_ids`: token IDs to append after audio (e.g. "Please transcribe the audio above.")
    /// `max_tokens`: max tokens to generate
    /// `eos_ids`: stop token IDs
    ///
    /// Returns generated text (transcription).
    pub fn transcribe_audio(
        &mut self,
        mel_data: &[f32],
        mel_frames: u32,
        suffix_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
    ) -> crate::inference::GenerateResult {
        let audio = self.audio.as_ref().expect("audio tower not loaded");

        // 1. Run audio tower: mel → encoder output [seq_len, 3584]
        let t0 = std::time::Instant::now();
        let (audio_output, audio_seq_len) = audio.encode_mel(
            &mut self.session.gpu, mel_data, mel_frames,
        );
        log::info!("[omni25] audio encoded: {} frames → {} tokens ({:.0}ms)",
            mel_frames, audio_seq_len, t0.elapsed().as_secs_f64() * 1000.0);

        // 2. Restore prefix snapshot (system prompt KV cache)
        //    This sets seq_len back to prefix_len
        // self.session.restore_prefix_snapshot() is private, but generate_inner handles it

        // 3. Build the input sequence:
        //    [audio_bos] + [AUDIO]*N + [audio_eos] + suffix_ids
        //    The AUDIO placeholder tokens will have their embeddings replaced
        let mut input_ids = Vec::with_capacity(2 + audio_seq_len as usize + suffix_ids.len());
        input_ids.push(self.tokens.audio_bos);
        for _ in 0..audio_seq_len {
            input_ids.push(self.tokens.audio_placeholder);
        }
        input_ids.push(self.tokens.audio_eos);
        input_ids.extend_from_slice(suffix_ids);

        log::info!("[omni25] input: {} tokens (1 bos + {} audio + 1 eos + {} suffix)",
            input_ids.len(), audio_seq_len, suffix_ids.len());

        // 4. Generate — for now uses standard path (audio embeddings not injected yet)
        // TODO: inject audio_output embeddings at AUDIO placeholder positions
        //       instead of looking up the embedding table
        self.session.generate_tokens(&input_ids, max_tokens, eos_ids, None)
    }

    /// Text-only inference (same as standard InferenceSession).
    pub fn generate_text(
        &mut self,
        input_ids: &[u32],
        max_tokens: u32,
        eos_ids: &[u32],
    ) -> crate::inference::GenerateResult {
        self.session.generate_tokens(input_ids, max_tokens, eos_ids, None)
    }

    /// Access prefix cache methods via the inner session.
    pub fn try_load_prefix_cache(&mut self, path: &Path) -> bool {
        self.session.try_load_prefix_cache(path)
    }

    pub fn save_prefix_cache(&mut self, path: &Path) -> std::io::Result<()> {
        self.session.save_prefix_cache(path)
    }

    pub fn set_prefix(&mut self, ids: &[u32]) {
        self.session.set_prefix(ids)
    }

    pub fn prefix_cache_path(model_dir: &Path, prompt: &str) -> PathBuf {
        InferenceSession::prefix_cache_path(model_dir, prompt)
    }
}
