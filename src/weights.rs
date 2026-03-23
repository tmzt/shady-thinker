use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use crate::gpu::GpuContext;

/// Standard self-attention layer weights
pub struct SelfAttnWeights {
    pub q_proj_qweight: wgpu::Buffer,
    pub q_proj_scales: wgpu::Buffer,
    pub k_proj_qweight: wgpu::Buffer,
    pub k_proj_scales: wgpu::Buffer,
    pub v_proj_qweight: wgpu::Buffer,
    pub v_proj_scales: wgpu::Buffer,
    pub o_proj_qweight: wgpu::Buffer,
    pub o_proj_scales: wgpu::Buffer,
    pub q_norm: wgpu::Buffer,
    pub k_norm: wgpu::Buffer,
}

/// DeltaNet linear attention layer weights
pub struct LinearAttnWeights {
    pub in_proj_qkv_qweight: wgpu::Buffer,
    pub in_proj_qkv_scales: wgpu::Buffer,
    pub in_proj_z_qweight: wgpu::Buffer,
    pub in_proj_z_scales: wgpu::Buffer,
    pub out_proj_qweight: wgpu::Buffer,
    pub out_proj_scales: wgpu::Buffer,
    pub conv1d_weight: wgpu::Buffer,
    pub a_log: wgpu::Buffer,
    pub dt_bias: wgpu::Buffer,
    pub norm_weight: wgpu::Buffer,
    /// Merged in_proj_a + in_proj_b as BF16 packed [2*num_value_heads, hidden_size/2] u32
    pub ab_weight: wgpu::Buffer,
}

/// Per-layer weights — either self-attention or DeltaNet
pub enum AttnWeights {
    SelfAttn(SelfAttnWeights),
    LinearAttn(LinearAttnWeights),
}

/// Per-layer weight buffers on GPU
pub struct LayerWeights {
    pub attn: AttnWeights,
    // MLP projections (shared by both layer types)
    pub gate_proj_qweight: wgpu::Buffer,
    pub gate_proj_scales: wgpu::Buffer,
    pub up_proj_qweight: wgpu::Buffer,
    pub up_proj_scales: wgpu::Buffer,
    pub down_proj_qweight: wgpu::Buffer,
    pub down_proj_scales: wgpu::Buffer,
    // Norm weights (BF16 packed as u32)
    pub input_layernorm: wgpu::Buffer,
    pub post_attn_layernorm: wgpu::Buffer,
}

impl LayerWeights {
    pub fn is_self_attn(&self) -> bool {
        matches!(self.attn, AttnWeights::SelfAttn(_))
    }

    pub fn self_attn(&self) -> Option<&SelfAttnWeights> {
        match &self.attn {
            AttnWeights::SelfAttn(w) => Some(w),
            _ => None,
        }
    }

    pub fn linear_attn(&self) -> Option<&LinearAttnWeights> {
        match &self.attn {
            AttnWeights::LinearAttn(w) => Some(w),
            _ => None,
        }
    }
}

/// Model-level weight buffers
pub struct ModelWeights {
    pub embed_tokens: wgpu::Buffer,
    pub final_norm: wgpu::Buffer,
    pub lm_head_qweight: wgpu::Buffer,
    pub lm_head_scales: wgpu::Buffer,
    /// true if lm_head is raw BF16 (not GPTQ quantized)
    pub lm_head_is_bf16: bool,
    pub layers: Vec<LayerWeights>,
    /// Which layers are self-attention (vs DeltaNet)
    pub self_attn_layers: Vec<usize>,
}

/// Model configuration parsed from config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub hidden_size: u32,
    #[serde(default)]
    pub intermediate_size: u32,
    #[serde(default)]
    pub num_attention_heads: u32,
    #[serde(default)]
    pub num_key_value_heads: u32,
    #[serde(default = "default_head_dim")]
    pub head_dim: u32,
    #[serde(default)]
    pub num_hidden_layers: u32,
    #[serde(default)]
    pub vocab_size: u32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // DeltaNet linear attention config
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: u32,
    #[serde(default = "default_linear_key_dim")]
    pub linear_key_head_dim: u32,
    #[serde(default = "default_linear_value_dim")]
    pub linear_value_head_dim: u32,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: u32,
    #[serde(default)]
    pub text_config: Option<Box<ModelConfig>>,
}

fn default_head_dim() -> u32 { 128 }
fn default_rms_norm_eps() -> f32 { 1e-6 }
fn default_linear_num_key_heads() -> u32 { 16 }
fn default_linear_key_dim() -> u32 { 128 }
fn default_linear_value_dim() -> u32 { 128 }
fn default_linear_num_value_heads() -> u32 { 16 }
fn default_rope_theta() -> f32 { 10_000_000.0 }

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantConfig {
    #[serde(default = "default_bits")]
    pub bits: u32,
    #[serde(default = "default_group_size")]
    pub group_size: u32,
    #[serde(default)]
    pub quant_method: String,
    #[serde(default)]
    pub sym: bool,
}

fn default_bits() -> u32 { 4 }
fn default_group_size() -> u32 { 128 }

impl ModelConfig {
    pub fn from_file(path: &Path) -> Self {
        let data = std::fs::read_to_string(path).expect("failed to read config.json");
        let mut config: Self = serde_json::from_str(&data).expect("failed to parse config.json");
        if config.hidden_size == 0 {
            if let Some(tc) = config.text_config.take() {
                let model_type = config.model_type.clone();
                config = *tc;
                if config.model_type.is_empty() {
                    config.model_type = model_type;
                }
            }
        }
        config
    }
}

impl QuantConfig {
    pub fn from_file(path: &Path) -> Self {
        let data = std::fs::read_to_string(path).expect("failed to read quantize_config.json");
        serde_json::from_str(&data).expect("failed to parse quantize_config.json")
    }
}

/// Raw norm weight bytes per layer for QK norm uniform initialization
pub struct RawNormWeights {
    pub layers: Vec<Option<(Vec<u8>, Vec<u8>)>>, // Some((q_norm, k_norm)) for self-attn layers
}

fn detect_layer_prefix(tensor_map: &HashMap<String, wgpu::Buffer>) -> String {
    for name in tensor_map.keys() {
        if name.starts_with("model.language_model.layers.0.") {
            return "model.language_model.layers".to_string();
        }
        if name.starts_with("model.layers.0.") {
            return "model.layers".to_string();
        }
    }
    "model.layers".to_string()
}

fn detect_model_prefix(tensor_map: &HashMap<String, wgpu::Buffer>) -> (&'static str, &'static str, &'static str) {
    if tensor_map.contains_key("model.language_model.embed_tokens.weight") {
        ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "model.language_model.lm_head")
    } else {
        ("model.embed_tokens.weight", "model.norm.weight", "lm_head")
    }
}

/// Detect whether a layer is self-attention or DeltaNet linear attention
fn is_self_attn_layer(tensor_map: &HashMap<String, wgpu::Buffer>, prefix: &str, layer_idx: u32) -> bool {
    let key = format!("{prefix}.{layer_idx}.self_attn.q_proj.qweight");
    tensor_map.contains_key(&key)
}

/// Dequantize GPTQ INT4 symmetric weights to BF16 bytes.
/// Matches the GPU shader convention: `f32(nibble) - 8.0` × scale.
/// qweight: [packed_rows, N] as u32 (8 int4 per u32, row-major)
/// scales: [num_groups, N] as f16 (packed 2 per u32)
/// n_cols: output dimension N (must be provided — cannot be inferred unambiguously)
fn dequant_gptq_to_bf16(qweight: &[u8], scales: &[u8], group_size: u32, n_cols: u32) -> Vec<u8> {
    let qw: &[u32] = bytemuck::cast_slice(qweight);
    let sc: &[u32] = bytemuck::cast_slice(scales);

    if qw.is_empty() || sc.is_empty() {
        return Vec::new();
    }

    let packed_rows = qw.len() as u32 / n_cols;
    let k = packed_rows * 8;

    let mut out = Vec::with_capacity((k * n_cols) as usize * 2);

    for pr in 0..packed_rows {
        let group = (pr * 8) / group_size;
        for col in 0..n_cols {
            let packed = qw[(pr * n_cols + col) as usize];

            // Get scale: same layout as GPU shader — unpack2x16float
            let sf = group * n_cols + col;
            let sc_packed = sc[(sf / 2) as usize];
            let scale = if sf % 2 == 0 {
                half::f16::from_bits((sc_packed & 0xFFFF) as u16).to_f32()
            } else {
                half::f16::from_bits((sc_packed >> 16) as u16).to_f32()
            };

            // Dequantize 8 int4 values from this packed u32
            for nib in 0..8u32 {
                let val = ((packed >> (nib * 4)) & 0xF) as f32 - 8.0;
                let dequant = val * scale;
                let bf16 = half::bf16::from_f32(dequant).to_bits();
                out.extend_from_slice(&bf16.to_le_bytes());
            }
        }
    }

    // The above produces data in [packed_rows, N, 8] order but we need [K, N] (row-major).
    // Currently: for each packed_row, for each col, 8 rows → [pr][col][nib]
    // Need: [row][col] where row = pr*8 + nib
    // Transpose from [packed_rows × N × 8] to [K × N]
    let bf16_size = 2usize;
    let mut transposed = vec![0u8; (k * n_cols) as usize * bf16_size];
    for pr in 0..packed_rows as usize {
        for col in 0..n_cols as usize {
            for nib in 0..8usize {
                let src_off = (pr * n_cols as usize * 8 + col * 8 + nib) * bf16_size;
                let row = pr * 8 + nib;
                let dst_off = (row * n_cols as usize + col) * bf16_size;
                transposed[dst_off..dst_off + bf16_size]
                    .copy_from_slice(&out[src_off..src_off + bf16_size]);
            }
        }
    }

    transposed
}

pub fn load_weights(
    gpu: &GpuContext,
    model_dir: &Path,
    config: &ModelConfig,
) -> (ModelWeights, RawNormWeights) {
    let mut shard_files: Vec<_> = std::fs::read_dir(model_dir)
        .expect("failed to read model directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().map_or(false, |ext| ext == "safetensors")
                && !p.to_string_lossy().contains(".index.")
        })
        .map(|e| e.path())
        .collect();
    shard_files.sort();

    log::info!("Loading {} safetensors shard(s)", shard_files.len());

    let mut tensor_map: HashMap<String, wgpu::Buffer> = HashMap::new();
    let mut raw_bytes_map: HashMap<String, Vec<u8>> = HashMap::new();

    for shard_path in &shard_files {
        let data = std::fs::read(shard_path).expect("failed to read shard");
        let tensors = SafeTensors::deserialize(&data).expect("failed to parse safetensors");

        // Handle in_proj_a/b: if quantized (.qweight/.scales), dequantize to BF16
        // Collect only the small dequant tensors; everything else uses zero-copy refs
        {
            let mut dequant_qw: HashMap<String, Vec<u8>> = HashMap::new();
            let mut dequant_sc: HashMap<String, Vec<u8>> = HashMap::new();
            for (name, view) in tensors.tensors() {
                for proj in &["in_proj_a", "in_proj_b"] {
                    if name.ends_with(&format!(".linear_attn.{proj}.qweight")) {
                        dequant_qw.insert(name.to_string(), view.data().to_vec());
                    }
                    if name.ends_with(&format!(".linear_attn.{proj}.scales")) {
                        dequant_sc.insert(name.to_string(), view.data().to_vec());
                    }
                }
            }
            for (qw_key, qw_bytes) in &dequant_qw {
                for proj in &["in_proj_a", "in_proj_b"] {
                    let qw_suffix = format!(".linear_attn.{proj}.qweight");
                    if qw_key.ends_with(&qw_suffix) {
                        let sc_key = qw_key.replace(".qweight", ".scales");
                        let wt_key = qw_key.replace(&qw_suffix, &format!(".linear_attn.{proj}.weight"));
                        if let Some(sc_bytes) = dequant_sc.get(&sc_key) {
                            let bf16 = dequant_gptq_to_bf16(qw_bytes, sc_bytes, 128, config.linear_num_value_heads);
                            raw_bytes_map.insert(wt_key, bf16);
                        }
                    }
                }
            }
        }

        for (name, view) in tensors.tensors() {
            if name.ends_with(".qzeros") || name.ends_with(".g_idx") {
                continue;
            }
            // Skip quantized in_proj_a/b — already dequantized above
            if (name.ends_with(".in_proj_a.qweight") || name.ends_with(".in_proj_a.scales")
                || name.ends_with(".in_proj_b.qweight") || name.ends_with(".in_proj_b.scales"))
            {
                continue;
            }

            let bytes = view.data();

            // Keep raw bytes for norm weights and DeltaNet in_proj_a/b (for merging)
            if name.ends_with(".q_norm.weight")
                || name.ends_with(".k_norm.weight")
                || name.ends_with(".in_proj_a.weight")
                || name.ends_with(".in_proj_b.weight")
            {
                raw_bytes_map.insert(name.to_string(), bytes.to_vec());
            }

            // Skip uploading in_proj_a/b separately — they'll be merged into ab_weight
            if name.ends_with(".in_proj_a.weight") || name.ends_with(".in_proj_b.weight") {
                continue;
            }

            let buffer = gpu.upload_buffer(&name, bytes);
            tensor_map.insert(name.to_string(), buffer);
        }
    }

    let layer_prefix = detect_layer_prefix(&tensor_map);
    let (embed_name, norm_name, lm_head_prefix) = detect_model_prefix(&tensor_map);
    // Try both prefixed and unprefixed lm_head names
    let (lm_head_qw_name, lm_head_sc_name, lm_head_w_name) = {
        let prefixed_qw = format!("{lm_head_prefix}.qweight");
        let prefixed_w = format!("{lm_head_prefix}.weight");
        if tensor_map.contains_key(&prefixed_qw) || tensor_map.contains_key(&prefixed_w) {
            (prefixed_qw, format!("{lm_head_prefix}.scales"), prefixed_w)
        } else {
            // Fallback: try unprefixed "lm_head.*"
            ("lm_head.qweight".to_string(), "lm_head.scales".to_string(), "lm_head.weight".to_string())
        }
    };
    let lm_head_is_quantized = tensor_map.contains_key(&lm_head_qw_name);
    let lm_head_is_unquantized = tensor_map.contains_key(&lm_head_w_name);

    // Detect self-attn vs linear-attn per layer
    let mut self_attn_indices = Vec::new();
    for i in 0..config.num_hidden_layers {
        if is_self_attn_layer(&tensor_map, &layer_prefix, i) {
            self_attn_indices.push(i as usize);
        }
    }

    log::info!(
        "Layer types: {} self-attn {:?}, {} linear-attn",
        self_attn_indices.len(),
        self_attn_indices,
        config.num_hidden_layers as usize - self_attn_indices.len(),
    );

    // Helper: remove tensor by name, panic if missing
    fn take(map: &mut HashMap<String, wgpu::Buffer>, name: &str) -> wgpu::Buffer {
        map.remove(name).unwrap_or_else(|| panic!("missing tensor: {name}"))
    }

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    let mut norm_weights = Vec::with_capacity(config.num_hidden_layers as usize);

    for i in 0..config.num_hidden_layers {
        let pfx = format!("{layer_prefix}.{i}");
        let is_sa = self_attn_indices.contains(&(i as usize));

        let attn = if is_sa {
            AttnWeights::SelfAttn(SelfAttnWeights {
                q_proj_qweight: take(&mut tensor_map, &format!("{pfx}.self_attn.q_proj.qweight")),
                q_proj_scales: take(&mut tensor_map, &format!("{pfx}.self_attn.q_proj.scales")),
                k_proj_qweight: take(&mut tensor_map, &format!("{pfx}.self_attn.k_proj.qweight")),
                k_proj_scales: take(&mut tensor_map, &format!("{pfx}.self_attn.k_proj.scales")),
                v_proj_qweight: take(&mut tensor_map, &format!("{pfx}.self_attn.v_proj.qweight")),
                v_proj_scales: take(&mut tensor_map, &format!("{pfx}.self_attn.v_proj.scales")),
                o_proj_qweight: take(&mut tensor_map, &format!("{pfx}.self_attn.o_proj.qweight")),
                o_proj_scales: take(&mut tensor_map, &format!("{pfx}.self_attn.o_proj.scales")),
                q_norm: take(&mut tensor_map, &format!("{pfx}.self_attn.q_norm.weight")),
                k_norm: take(&mut tensor_map, &format!("{pfx}.self_attn.k_norm.weight")),
            })
        } else {
            // Merge in_proj_a + in_proj_b into ab_weight (concat raw BF16 bytes)
            let a_key = format!("{pfx}.linear_attn.in_proj_a.weight");
            let b_key = format!("{pfx}.linear_attn.in_proj_b.weight");
            let a_bytes = raw_bytes_map.remove(&a_key).unwrap_or_default();
            let b_bytes = raw_bytes_map.remove(&b_key).unwrap_or_default();
            let mut ab_merged = Vec::with_capacity(a_bytes.len() + b_bytes.len());
            ab_merged.extend_from_slice(&a_bytes);
            ab_merged.extend_from_slice(&b_bytes);
            if ab_merged.is_empty() {
                // Fallback: create a minimum-sized buffer to avoid 0-byte bind errors
                let nhv = config.linear_num_value_heads;
                let h = config.hidden_size;
                ab_merged = vec![0u8; (2 * nhv * h) as usize]; // BF16 zeros
                log::warn!("Layer {i}: ab_weight empty, using zeros ({} bytes)", ab_merged.len());
            }
            let ab_buf = gpu.upload_buffer(&format!("{pfx}.linear_attn.ab_weight"), &ab_merged);

            AttnWeights::LinearAttn(LinearAttnWeights {
                in_proj_qkv_qweight: take(&mut tensor_map, &format!("{pfx}.linear_attn.in_proj_qkv.qweight")),
                in_proj_qkv_scales: take(&mut tensor_map, &format!("{pfx}.linear_attn.in_proj_qkv.scales")),
                in_proj_z_qweight: take(&mut tensor_map, &format!("{pfx}.linear_attn.in_proj_z.qweight")),
                in_proj_z_scales: take(&mut tensor_map, &format!("{pfx}.linear_attn.in_proj_z.scales")),
                out_proj_qweight: take(&mut tensor_map, &format!("{pfx}.linear_attn.out_proj.qweight")),
                out_proj_scales: take(&mut tensor_map, &format!("{pfx}.linear_attn.out_proj.scales")),
                conv1d_weight: take(&mut tensor_map, &format!("{pfx}.linear_attn.conv1d.weight")),
                a_log: take(&mut tensor_map, &format!("{pfx}.linear_attn.A_log")),
                dt_bias: take(&mut tensor_map, &format!("{pfx}.linear_attn.dt_bias")),
                norm_weight: take(&mut tensor_map, &format!("{pfx}.linear_attn.norm.weight")),
                ab_weight: ab_buf,
            })
        };

        layers.push(LayerWeights {
            attn,
            gate_proj_qweight: take(&mut tensor_map, &format!("{pfx}.mlp.gate_proj.qweight")),
            gate_proj_scales: take(&mut tensor_map, &format!("{pfx}.mlp.gate_proj.scales")),
            up_proj_qweight: take(&mut tensor_map, &format!("{pfx}.mlp.up_proj.qweight")),
            up_proj_scales: take(&mut tensor_map, &format!("{pfx}.mlp.up_proj.scales")),
            down_proj_qweight: take(&mut tensor_map, &format!("{pfx}.mlp.down_proj.qweight")),
            down_proj_scales: take(&mut tensor_map, &format!("{pfx}.mlp.down_proj.scales")),
            input_layernorm: take(&mut tensor_map, &format!("{pfx}.input_layernorm.weight")),
            post_attn_layernorm: take(&mut tensor_map, &format!("{pfx}.post_attention_layernorm.weight")),
        });

        // Raw norm bytes for self-attn layers
        if is_sa {
            let q_bytes = raw_bytes_map
                .remove(&format!("{pfx}.self_attn.q_norm.weight"))
                .unwrap_or_default();
            let k_bytes = raw_bytes_map
                .remove(&format!("{pfx}.self_attn.k_norm.weight"))
                .unwrap_or_default();
            norm_weights.push(Some((q_bytes, k_bytes)));
        } else {
            norm_weights.push(None);
        }
    }

    let (lm_head_qweight, lm_head_scales) = if lm_head_is_quantized {
        (take(&mut tensor_map, &lm_head_qw_name), take(&mut tensor_map, &lm_head_sc_name))
    } else if lm_head_is_unquantized {
        log::warn!("lm_head is unquantized — using weight directly");
        let w = take(&mut tensor_map, &lm_head_w_name);
        let dummy = gpu.upload_buffer("lm_head_scales_dummy", &[0u8; 4]);
        (w, dummy)
    } else {
        // Tied embeddings — lm_head shares embed_tokens
        log::info!("No lm_head found — assuming tied embeddings");
        let dummy_qw = gpu.upload_buffer("lm_head_qw_dummy", &[0u8; 4]);
        let dummy_sc = gpu.upload_buffer("lm_head_sc_dummy", &[0u8; 4]);
        (dummy_qw, dummy_sc)
    };

    (
        ModelWeights {
            embed_tokens: take(&mut tensor_map, embed_name),
            final_norm: take(&mut tensor_map, norm_name),
            lm_head_qweight,
            lm_head_scales,
            lm_head_is_bf16: lm_head_is_unquantized,
            self_attn_layers: self_attn_indices,
            layers,
        },
        RawNormWeights {
            layers: norm_weights,
        },
    )
}
