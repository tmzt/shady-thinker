// Embed the pre-built prefix KV cache if FAST_THINKER_MODEL_DIR was set at build time.
// Exposes `EMBEDDED_PREFIX_CACHE: Option<&[u8]>`.
include!(concat!(env!("OUT_DIR"), "/prefix_cache_embed.rs"));

pub mod asr_decoder;
pub mod asr_encoder;
pub mod asr_pipeline;
pub mod gpu;
pub mod inference;
pub mod json_sampler;
pub mod json_schema;
#[cfg(feature = "jit-lora")]
pub mod lora;
pub mod model;
pub mod nomic;
#[cfg(feature = "jit-lora")]
pub mod train;
pub mod omni25_audio;
pub mod omni25_session;
pub mod weights;
pub mod wordpiece;
#[cfg(feature = "chat")]
pub mod chat;
