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
///
/// Design: audio tower and text decoder share the same GPU device and
/// wgpu::Queue. Audio tower output buffers live on the same device, so
/// `forward_embed_kv_only` reads them directly (zero-copy).
///
/// For multi-device: split at the boundary between audio tower output
/// and decoder embedding injection — copy `[seq_len, 3584]` f32 buffer
/// between devices once per utterance (~few MB, negligible).
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
        pre_audio_ids: &[u32],
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
        self.session.model.seq_len = self.session.prefix_len;
        self.session.model.generated_tokens.clear();
        if let Some(ref snap) = self.session.prefix_snapshot {
            // Restore KV cache from snapshot
            for (li, (k, v)) in snap.kv.iter().enumerate() {
                if !k.is_empty() {
                    self.session.gpu.write_buffer(&self.session.model.state.k_cache[li], 0, k);
                    self.session.gpu.write_buffer(&self.session.model.state.v_cache[li], 0, v);
                }
            }
        }

        // 3. Forward through decoder:
        //    pre_audio → audio_bos → audio embeddings → audio_eos → suffix

        // Pre-audio tokens (e.g. "<|im_start|>user\n")
        for &tok in pre_audio_ids {
            self.session.model.forward_kv_only(&mut self.session.gpu, tok);
        }

        // Read audio tower output from GPU → CPU
        let hidden = self.session.model.config.hidden_size as usize;
        let audio_bytes = audio_seq_len as usize * hidden * 4;
        let audio_cpu = self.session.gpu.read_buffer(&audio_output, audio_bytes as u64);
        let audio_f32: &[f32] = bytemuck::cast_slice(&audio_cpu);

        log::info!("[omni25] injecting {} audio embeddings (dim={})", audio_seq_len, hidden);

        // audio_bos token
        self.session.model.forward_kv_only(&mut self.session.gpu, self.tokens.audio_bos);

        // Audio embeddings — each is [hidden_size] f32
        for i in 0..audio_seq_len as usize {
            let embed = &audio_f32[i * hidden..(i + 1) * hidden];
            self.session.model.forward_embed_kv_only(&mut self.session.gpu, embed);
            if (i + 1) % 10 == 0 {
                self.session.gpu.flush_and_wait();
            }
        }

        // audio_eos token
        self.session.model.forward_kv_only(&mut self.session.gpu, self.tokens.audio_eos);

        // Suffix tokens — all but last via kv_only, last via forward() to get lm_head logits
        if suffix_ids.len() > 1 {
            for &tok in &suffix_ids[..suffix_ids.len() - 1] {
                self.session.model.forward_kv_only(&mut self.session.gpu, tok);
            }
        }
        self.session.gpu.flush_and_wait();

        log::info!("[omni25] prefill done (seq_len={}), decoding...",
            self.session.model.seq_len);

        // 4. Decode: forward the last suffix token through full forward() to get first logits,
        //    then continue autoregressive decoding.
        let last_suffix = *suffix_ids.last().unwrap_or(&0);
        let first_token = self.session.model.forward(&mut self.session.gpu, last_suffix);
        log::info!("[omni25] first decode token: {}", first_token);

        let mut generated = vec![first_token];
        let mut token = first_token;
        for _ in 1..max_tokens {
            if eos_ids.contains(&token) { break; }
            token = self.session.model.forward(&mut self.session.gpu, token);
            generated.push(token);
        }

        crate::inference::GenerateResult {
            token_ids: generated.iter().filter(|t| !eos_ids.contains(t)).copied().collect(),
            token_count: generated.len(),
            tokens_per_sec: 0.0, // TODO: compute
            state: crate::inference::GenerateState::Complete,
            epiphany_count: 0,
        }
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
