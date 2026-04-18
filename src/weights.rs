use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

use crate::gpu::GpuContext;

/// Convert a bf16 weight matrix [rows, cols] to GPTQ INT4 format.
/// Returns (qweight_buffer, scales_buffer) ready for GPU upload.
/// Symmetric quantization: val ≈ (nibble - 8) * scale.
/// Column-major qweight layout matches gptq_matvec.wgsl.
fn quantize_bf16_to_int4(
    gpu: &GpuContext,
    label: &str,
    bf16_data: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> (wgpu::Buffer, wgpu::Buffer) {
    assert_eq!(bf16_data.len(), rows * cols * 2, "{label}: bf16 size mismatch");
    assert!(rows % 8 == 0, "{label}: rows must be multiple of 8");

    let packed_rows = rows / 8;
    let n_groups = (rows + group_size - 1) / group_size;

    // Step 1: bf16 → f32 batch conversion (row-major → column-major transpose)
    // Process one column at a time to be cache-friendly for the output
    let mut qweight = vec![0u32; packed_rows * cols];
    let mut scales_f16 = vec![0u16; n_groups * cols];

    for c in 0..cols {
        // Extract column c from row-major bf16 data, convert to f32
        // Stride: every `cols` elements, offset by c
        let mut col_f32 = vec![0.0f32; rows];
        for r in 0..rows {
            let idx = (r * cols + c) * 2;
            let bits = (bf16_data[idx] as u32) | ((bf16_data[idx + 1] as u32) << 8);
            col_f32[r] = f32::from_bits(bits << 16);
        }

        // Step 2: Per-group quantization for this column
        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(rows);

            // Find max absolute value in group (vectorizable)
            let mut max_abs: f32 = 0.0;
            for r in start..end {
                let a = col_f32[r].abs();
                if a > max_abs { max_abs = a; }
            }

            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales_f16[g * cols + c] = half::f16::from_f32(scale).to_bits();

            // Quantize and pack 8 values per u32
            let group_pr_start = start / 8;
            let group_pr_end = (end + 7) / 8;
            for pr in group_pr_start..group_pr_end.min(packed_rows) {
                let mut packed: u32 = 0;
                for nibble in 0..8u32 {
                    let r = pr * 8 + nibble as usize;
                    if r < end && r >= start {
                        let q = ((col_f32[r] * inv_scale).round() as i32 + 8).clamp(0, 15) as u32;
                        packed |= q << (nibble * 4);
                    } else if r < rows {
                        // Row belongs to adjacent group — already handled or will be
                        // Read existing packed value and preserve this nibble
                        let existing = qweight[pr * cols + c];
                        packed |= existing & (0xF << (nibble * 4));
                    }
                }
                qweight[pr * cols + c] = packed;
            }
        }
    }

    // Step 3: Upload to GPU
    let qw_buf = gpu.upload_buffer(&format!("{label}.qw"), bytemuck::cast_slice(&qweight));

    // Pack scales: two f16 per u32
    let scales_u32_len = (n_groups * cols + 1) / 2;
    let mut scales_packed = vec![0u32; scales_u32_len];
    for i in (0..n_groups * cols).step_by(2) {
        let lo = scales_f16[i] as u32;
        let hi = if i + 1 < n_groups * cols { scales_f16[i + 1] as u32 } else { 0 };
        scales_packed[i / 2] = lo | (hi << 16);
    }
    let sc_buf = gpu.upload_buffer(&format!("{label}.sc"), bytemuck::cast_slice(&scales_packed));

    (qw_buf, sc_buf)
}

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
    /// Chunked embedding table for 128MB binding limit.
    pub embed_chunks: Vec<wgpu::Buffer>,
    /// Tokens per chunk (0 if not chunked)
    pub embed_chunk_size: u32,
    /// MLX INT4 biases per layer: [q, k, v, o, gate, up, down]
    /// Empty for GPTQ/bf16 modes.
    pub mlx_biases: Vec<[wgpu::Buffer; 7]>,
    /// Embedding scales/biases for tied lm_head in MLX INT4 mode.
    /// When tied_embeddings=true and mlx_int4_mode=true, logits use these
    /// with INT4_MATVEC_MLX instead of bf16_lm_head.
    pub embed_scales: Option<wgpu::Buffer>,
    pub embed_biases: Option<wgpu::Buffer>,
    /// true if MLX scales/biases are BF16 (not F16)
    pub bf16_scales: bool,
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
    // mRoPE config (rope_scaling or rope_parameters in config.json)
    #[serde(default, alias = "rope_scaling")]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub text_config: Option<Box<ModelConfig>>,
    /// Qwen2.5-Omni: thinker config wraps text_config
    #[serde(default)]
    pub thinker_config: Option<Box<ModelConfig>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<u32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

fn default_head_dim() -> u32 { 128 }
fn default_rms_norm_eps() -> f32 { 1e-6 }
fn default_partial_rotary_factor() -> f32 { 0.25 }
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
        // Qwen2.5-Omni: thinker_config.text_config
        if config.hidden_size == 0 {
            if let Some(mut tc) = config.thinker_config.take() {
                let model_type = config.model_type.clone();
                if let Some(ttc) = tc.text_config.take() {
                    config = *ttc;
                } else {
                    config = *tc;
                }
                if config.model_type.is_empty() {
                    config.model_type = model_type;
                }
            } else if let Some(tc) = config.text_config.take() {
                let model_type = config.model_type.clone();
                config = *tc;
                if config.model_type.is_empty() {
                    config.model_type = model_type;
                }
            }
        }
        // Resolve rope_theta from rope_parameters if present
        if let Some(ref rp) = config.rope_parameters {
            if let Some(theta) = rp.rope_theta {
                config.rope_theta = theta;
            }
        }
        config
    }

    /// Get mRoPE section boundaries as cumulative sums.
    /// Returns (s1_limit, s2_limit) for contiguous section selection.
    /// Default: sections [11, 11, 10] → limits (11, 22).
    pub fn mrope_sections(&self) -> (u32, u32) {
        if let Some(ref rp) = self.rope_parameters {
            if rp.mrope_section.len() >= 3 {
                let s1 = rp.mrope_section[0];
                let s2 = rp.mrope_section[0] + rp.mrope_section[1];
                return (s1, s2);
            }
        }
        // Default for Qwen3.5
        (11, 22)
    }

    /// Whether mRoPE uses interleaved rotation pairs (2d, 2d+1) vs (d, d+partial_half).
    pub fn mrope_interleaved(&self) -> bool {
        self.rope_parameters
            .as_ref()
            .map_or(true, |rp| rp.mrope_interleaved)
    }
}

impl QuantConfig {
    pub fn from_file(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(data) => {
                serde_json::from_str(&data).unwrap_or_else(|e| {
                    log::warn!("[shady-thinker] failed to parse quantize_config.json: {e}, using defaults");
                    Self::default()
                })
            }
            Err(_) => {
                log::info!("[shady-thinker] no quantize_config.json, using defaults (bits=4, group_size=128)");
                Self::default()
            }
        }
    }
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: default_bits(),
            group_size: default_group_size(),
            quant_method: "gptq".to_string(),
            sym: false,
        }
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

fn detect_model_prefix(tensor_map: &HashMap<String, wgpu::Buffer>, oversized_raw: &HashMap<String, Vec<u8>>) -> (&'static str, &'static str, &'static str) {
    let has_key = |k: &str| tensor_map.contains_key(k) || oversized_raw.contains_key(k);
    if has_key("model.language_model.embed_tokens.weight") {
        ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "model.language_model.lm_head")
    } else {
        ("model.embed_tokens.weight", "model.norm.weight", "lm_head")
    }
}

/// Detect whether a layer is self-attention or DeltaNet linear attention.
/// Checks both GPTQ-quantized (.qweight) and unquantized BF16 (.weight) variants.
fn is_self_attn_layer(tensor_map: &HashMap<String, wgpu::Buffer>, prefix: &str, layer_idx: u32) -> bool {
    let qkey = format!("{prefix}.{layer_idx}.self_attn.q_proj.qweight");
    let wkey = format!("{prefix}.{layer_idx}.self_attn.q_proj.weight");
    tensor_map.contains_key(&qkey) || tensor_map.contains_key(&wkey)
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
    // Raw bytes for tensors too large to bind whole — needed for chunked upload
    let mut oversized_raw: HashMap<String, Vec<u8>> = HashMap::new();
    let max_binding = gpu.max_storage_binding_size();

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

            // Defer oversized tensors — they need chunked upload after we know hidden_size
            if bytes.len() as u64 > max_binding && name.ends_with("embed_tokens.weight") {
                oversized_raw.insert(name.to_string(), bytes.to_vec());
                continue;
            }
            let buffer = gpu.upload_buffer(&name, bytes);
            tensor_map.insert(name.to_string(), buffer);
        }
    }

    let layer_prefix = detect_layer_prefix(&tensor_map);
    let (embed_name, norm_name, lm_head_prefix) = detect_model_prefix(&tensor_map, &oversized_raw);
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

    // Chunk the embedding table if it exceeds the GPU's max storage binding size.
    // The embedding shader already supports chunked lookup (embed_chunks / embed_chunk_size).
    let hidden = config.hidden_size as u64;
    let bytes_per_row = hidden * 2; // BF16

    let (embed_tokens, embed_chunks, embed_chunk_size) =
        if let Some(embed_bytes) = oversized_raw.remove(embed_name) {
            // Tensor was captured raw during shard scan (too large to bind whole).
            let chunk_rows = (max_binding / bytes_per_row) as u32;
            log::info!(
                "embed_tokens {}MB > max_binding {}MB — chunking, rows_per_chunk={}",
                embed_bytes.len() / (1 << 20), max_binding / (1 << 20), chunk_rows,
            );
            let total_rows = (embed_bytes.len() as u64 / bytes_per_row) as u32;
            let num_chunks = total_rows.div_ceil(chunk_rows);
            let mut chunks = Vec::with_capacity(num_chunks as usize);
            for c in 0..num_chunks {
                let start = c as usize * chunk_rows as usize * bytes_per_row as usize;
                let end = (start + chunk_rows as usize * bytes_per_row as usize).min(embed_bytes.len());
                chunks.push(gpu.upload_buffer(&format!("embed_chunk_{c}"), &embed_bytes[start..end]));
            }
            // embed_tokens must exist in the struct; embedding shader uses embed_chunks when non-empty.
            let dummy = gpu.upload_buffer("embed_tokens_dummy", &[0u8; 4]);
            (dummy, chunks, chunk_rows)
        } else {
            (take(&mut tensor_map, embed_name), Vec::new(), 0)
        };

    (
        ModelWeights {
            embed_tokens,
            final_norm: take(&mut tensor_map, norm_name),
            lm_head_qweight,
            lm_head_scales,
            lm_head_is_bf16: lm_head_is_unquantized,
            self_attn_layers: self_attn_indices,
            layers,
            embed_chunks,
            embed_chunk_size,
            mlx_biases: Vec::new(), embed_scales: None, embed_biases: None, bf16_scales: false,
        },
        RawNormWeights {
            layers: norm_weights,
        },
    )
}

/// Load bf16 (unquantized) weights for the ASR decoder.
/// Creates ModelWeights with bf16 data in qweight fields, dummy scales.
/// The model.rs forward uses bf16_matvec when bf16_mode=true.
pub fn load_weights_bf16(
    gpu: &GpuContext,
    model_dir: &Path,
    config: &ModelConfig,
) -> (ModelWeights, RawNormWeights) {
    let mut shard_files: Vec<_> = std::fs::read_dir(model_dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();
    shard_files.sort();

    log::info!("[bf16] loading {} shard(s) from {:?}", shard_files.len(), model_dir);

    // Memory-map shards to avoid loading entire files into RAM.
    // On Android this reduces peak RSS from ~6.9GB to ~2GB.
    let shard_mmaps: Vec<memmap2::Mmap> = shard_files.iter()
        .map(|p| {
            let file = std::fs::File::open(p).expect("open shard");
            unsafe { memmap2::Mmap::map(&file).expect("mmap shard") }
        })
        .collect();
    let shards: Vec<SafeTensors> = shard_mmaps.iter()
        .map(|m| SafeTensors::deserialize(m).expect("parse"))
        .collect();

    let get = |name: &str| -> &[u8] {
        for st in &shards {
            if let Ok(t) = st.tensor(name) { return t.data(); }
        }
        panic!("[bf16] tensor not found: {name}");
    };

    let upload = |label: &str, name: &str| -> wgpu::Buffer {
        gpu.upload_buffer(label, get(name))
    };

    // Dummy 4-byte buffer for scales (unused in bf16 mode)
    let dummy = gpu.create_storage_buffer("dummy", 4);

    // Embedding — chunk if needed for binding limit
    let embed_raw = get("thinker.model.embed_tokens.weight");
    let binding_limit = gpu.max_storage_binding_size() as usize;
    let embed_total_bytes = embed_raw.len();
    let bytes_per_token = config.hidden_size as usize * 2; // bf16
    let needs_chunking = embed_total_bytes > binding_limit;

    let (embed, embed_chunks, embed_chunk_size) = if needs_chunking {
        let max_chunk_bytes = (binding_limit - 1024) & !3; // stay under limit, aligned
        let tokens_per_chunk = max_chunk_bytes / bytes_per_token;
        let n_chunks = (config.vocab_size as usize + tokens_per_chunk - 1) / tokens_per_chunk;
        log::info!("[bf16] embedding {}MB > {}MB binding — splitting into {} chunks of {} tokens",
            embed_total_bytes / (1024*1024), binding_limit / (1024*1024), n_chunks, tokens_per_chunk);

        let mut chunks = Vec::new();
        for c in 0..n_chunks {
            let start = c * tokens_per_chunk * bytes_per_token;
            let end = ((c + 1) * tokens_per_chunk * bytes_per_token).min(embed_raw.len());
            chunks.push(gpu.upload_buffer(&format!("embed_chunk_{c}"), &embed_raw[start..end]));
        }
        // First chunk doubles as embed_tokens for compatibility
        let dummy_embed = gpu.create_storage_buffer("embed_dummy", 4);
        (dummy_embed, chunks, tokens_per_chunk as u32)
    } else {
        let embed_bytes = embed_raw.to_vec();
        let embed = gpu.upload_buffer("embed", &embed_bytes);
        (embed, Vec::new(), 0u32)
    };
    let final_norm = upload("final_norm", "thinker.model.norm.weight");

    // lm_head
    let lm_head_exists = shards.iter().any(|st| st.tensor("thinker.lm_head.weight").is_ok());
    let lm_head = if lm_head_exists {
        upload("lm_head", "thinker.lm_head.weight")
    } else {
        log::info!("[bf16] no lm_head — assuming tied embeddings");
        gpu.create_storage_buffer("lm_head_dummy", 4)
    };

    let mut layers = Vec::new();
    let mut norm_weights = Vec::new();

    for i in 0..config.num_hidden_layers as usize {
        let p = format!("thinker.model.layers.{i}");

        // Q/K norm raw bytes for qknorm_params
        let q_norm_bytes = get(&format!("{p}.self_attn.q_norm.weight")).to_vec();
        let k_norm_bytes = get(&format!("{p}.self_attn.k_norm.weight")).to_vec();
        norm_weights.push(Some((q_norm_bytes, k_norm_bytes)));

        let sa = SelfAttnWeights {
            q_proj_qweight: upload(&format!("{p}.q"), &format!("{p}.self_attn.q_proj.weight")),
            q_proj_scales: dummy.clone(),
            k_proj_qweight: upload(&format!("{p}.k"), &format!("{p}.self_attn.k_proj.weight")),
            k_proj_scales: dummy.clone(),
            v_proj_qweight: upload(&format!("{p}.v"), &format!("{p}.self_attn.v_proj.weight")),
            v_proj_scales: dummy.clone(),
            o_proj_qweight: upload(&format!("{p}.o"), &format!("{p}.self_attn.o_proj.weight")),
            o_proj_scales: dummy.clone(),
            q_norm: upload(&format!("{p}.qn"), &format!("{p}.self_attn.q_norm.weight")),
            k_norm: upload(&format!("{p}.kn"), &format!("{p}.self_attn.k_norm.weight")),
        };

        let layer = LayerWeights {
            attn: AttnWeights::SelfAttn(sa),
            gate_proj_qweight: upload(&format!("{p}.gate"), &format!("{p}.mlp.gate_proj.weight")),
            gate_proj_scales: dummy.clone(),
            up_proj_qweight: upload(&format!("{p}.up"), &format!("{p}.mlp.up_proj.weight")),
            up_proj_scales: dummy.clone(),
            down_proj_qweight: upload(&format!("{p}.down"), &format!("{p}.mlp.down_proj.weight")),
            down_proj_scales: dummy.clone(),
            input_layernorm: upload(&format!("{p}.in"), &format!("{p}.input_layernorm.weight")),
            post_attn_layernorm: upload(&format!("{p}.pa"), &format!("{p}.post_attention_layernorm.weight")),
        };
        layers.push(layer);

        if (i + 1) % 7 == 0 {
            log::info!("[bf16] loaded layer {}/{}", i + 1, config.num_hidden_layers);
        }
    }

    let self_attn_indices: Vec<usize> = (0..config.num_hidden_layers as usize).collect();

    log::info!("[bf16] loaded {} layers", layers.len());

    (
        ModelWeights {
            embed_tokens: embed,
            final_norm,
            lm_head_qweight: lm_head,
            lm_head_scales: dummy,
            lm_head_is_bf16: true,
            self_attn_layers: self_attn_indices,
            layers,
            embed_chunks,
            embed_chunk_size,
            mlx_biases: Vec::new(), embed_scales: None, embed_biases: None, bf16_scales: false,
        },
        RawNormWeights {
            layers: norm_weights,
        },
    )
}

/// Load weights with runtime bf16→INT4 quantization.
/// Linear layers → GPTQ INT4 (symmetric, group_size). Norms/embeddings → bf16.
/// ~4x less GPU memory + bandwidth, uses standard gptq_matvec shader (no bf16_mode).
pub fn load_weights_int4(
    gpu: &GpuContext, model_dir: &Path, config: &ModelConfig, group_size: u32,
) -> (ModelWeights, RawNormWeights) {
    let mut sf: Vec<_> = std::fs::read_dir(model_dir).expect("read dir")
        .filter_map(|e| e.ok()).filter(|e| e.path().extension().map_or(false, |x| x=="safetensors"))
        .map(|e| e.path()).collect();
    sf.sort();
    log::info!("[int4] {} shard(s), group_size={}", sf.len(), group_size);
    let mm: Vec<memmap2::Mmap> = sf.iter()
        .map(|p| unsafe { memmap2::Mmap::map(&std::fs::File::open(p).unwrap()).unwrap() }).collect();
    let st: Vec<SafeTensors> = mm.iter().map(|m| SafeTensors::deserialize(m).unwrap()).collect();

    let get = |n: &str| -> &[u8] { for s in &st { if let Ok(t)=s.tensor(n) { return t.data(); } } panic!("{n}"); };
    let shp = |n: &str| -> Vec<usize> { for s in &st { if let Ok(t)=s.tensor(n) { return t.shape().to_vec(); } } panic!("{n}"); };
    let bf = |l:&str,n:&str| -> wgpu::Buffer { gpu.upload_buffer(l, get(n)) };
    let q4 = |l:&str,n:&str| -> (wgpu::Buffer,wgpu::Buffer) {
        let s=shp(n); quantize_bf16_to_int4(gpu,l,get(n),s[0],s[1],group_size as usize)
    };

    let er = get("thinker.model.embed_tokens.weight");
    let bl = gpu.max_storage_binding_size() as usize;
    let bt = config.hidden_size as usize * 2;
    let (emb,ech,ecs) = if er.len()>bl {
        let mc=(bl-1024)&!3; let tp=mc/bt; let nc=(config.vocab_size as usize+tp-1)/tp;
        log::info!("[int4] embed: {} bf16 chunks", nc);
        let mut c=Vec::new();
        for i in 0..nc { let s=i*tp*bt; let e=((i+1)*tp*bt).min(er.len()); c.push(gpu.upload_buffer(&format!("ec{i}"),&er[s..e])); }
        (gpu.create_storage_buffer("ed",4),c,tp as u32)
    } else { (gpu.upload_buffer("e",er),Vec::new(),0u32) };

    let fnrm = bf("fn","thinker.model.norm.weight");
    let lmh = if st.iter().any(|s| s.tensor("thinker.lm_head.weight").is_ok()) { bf("lh","thinker.lm_head.weight") }
              else { gpu.create_storage_buffer("ld",4) };
    let ds = gpu.create_storage_buffer("ds",4);
    let mut ly=Vec::new(); let mut nw=Vec::new(); let t0=std::time::Instant::now();

    for i in 0..config.num_hidden_layers as usize {
        let p=format!("thinker.model.layers.{i}");
        nw.push(Some((get(&format!("{p}.self_attn.q_norm.weight")).to_vec(),
                       get(&format!("{p}.self_attn.k_norm.weight")).to_vec())));
        let (qq,qs)=q4("q",&format!("{p}.self_attn.q_proj.weight"));
        let (kq,ks)=q4("k",&format!("{p}.self_attn.k_proj.weight"));
        let (vq,vs)=q4("v",&format!("{p}.self_attn.v_proj.weight"));
        let (oq,os)=q4("o",&format!("{p}.self_attn.o_proj.weight"));
        let (gq,gs)=q4("g",&format!("{p}.mlp.gate_proj.weight"));
        let (uq,us)=q4("u",&format!("{p}.mlp.up_proj.weight"));
        let (dq,dss)=q4("d",&format!("{p}.mlp.down_proj.weight"));
        ly.push(LayerWeights {
            attn: AttnWeights::SelfAttn(SelfAttnWeights {
                q_proj_qweight:qq,q_proj_scales:qs, k_proj_qweight:kq,k_proj_scales:ks,
                v_proj_qweight:vq,v_proj_scales:vs, o_proj_qweight:oq,o_proj_scales:os,
                q_norm:bf("qn",&format!("{p}.self_attn.q_norm.weight")),
                k_norm:bf("kn",&format!("{p}.self_attn.k_norm.weight")),
            }),
            gate_proj_qweight:gq,gate_proj_scales:gs, up_proj_qweight:uq,up_proj_scales:us,
            down_proj_qweight:dq,down_proj_scales:dss,
            input_layernorm:bf("il",&format!("{p}.input_layernorm.weight")),
            post_attn_layernorm:bf("pl",&format!("{p}.post_attention_layernorm.weight")),
        });
        if (i+1)%7==0 { log::info!("[int4] layer {}/{} ({}ms)", i+1, config.num_hidden_layers, t0.elapsed().as_millis()); }
    }
    log::info!("[int4] done: {} layers in {}ms", ly.len(), t0.elapsed().as_millis());

    (ModelWeights {
        embed_tokens:emb, final_norm:fnrm, lm_head_qweight:lmh, lm_head_scales:ds,
        lm_head_is_bf16:true, self_attn_layers:(0..config.num_hidden_layers as usize).collect(),
        layers:ly, embed_chunks:ech, embed_chunk_size:ecs,
        mlx_biases: Vec::new(), embed_scales: None, embed_biases: None, bf16_scales: false,
    }, RawNormWeights { layers: nw })
}

/// Load pre-quantized MLX INT4 weights (minmax, group_size=64).
/// Tensor names: model.layers.N.* (MLX convention, no "thinker." prefix).
/// Returns weights ready for int4_matvec_mlx shader.
pub fn load_weights_mlx_int4(
    gpu: &GpuContext, model_dir: &Path, config: &ModelConfig,
) -> (ModelWeights, RawNormWeights) {
    let mut sf: Vec<_> = std::fs::read_dir(model_dir).expect("read dir")
        .filter_map(|e| e.ok()).filter(|e| e.path().extension().map_or(false, |x| x=="safetensors"))
        .map(|e| e.path()).collect();
    sf.sort();
    log::info!("[mlx-int4] {} shard(s)", sf.len());
    let mm: Vec<memmap2::Mmap> = sf.iter()
        .map(|p| unsafe { memmap2::Mmap::map(&std::fs::File::open(p).unwrap()).unwrap() }).collect();
    let st: Vec<SafeTensors> = mm.iter().map(|m| SafeTensors::deserialize(m).unwrap()).collect();

    // Detect BF16 scales — check dtype of first scale tensor
    let scales_are_bf16 = st.iter().find_map(|s| {
        s.tensor("model.layers.0.self_attn.q_proj.scales").ok()
    }).map_or(false, |t| format!("{:?}", t.dtype()) == "BF16");
    if scales_are_bf16 {
        log::info!("[mlx-int4] scales/biases are BF16");
    }

    let get = |n: &str| -> &[u8] {
        for s in &st { if let Ok(t) = s.tensor(n) { return t.data(); } }
        // Tied embeddings: lm_head.weight == model.embed_tokens.weight
        if n == "lm_head.weight" {
            for s in &st { if let Ok(t) = s.tensor("model.embed_tokens.weight") { return t.data(); } }
        }
        panic!("[mlx-int4] tensor not found: {n}");
    };
    let up = |l:&str,n:&str| -> wgpu::Buffer { gpu.upload_buffer(l, get(n)) };

    // Embedding — quantized in MLX format, needs chunking for 128MB binding
    // For embedding lookup we need a special dequant shader too.
    // For now, upload qweight + scales + biases and dequant on CPU during lookup.
    // TODO: GPU embedding dequant shader
    let embed_qw = up("emb_qw", "model.embed_tokens.weight");
    let embed_sc = up("emb_sc", "model.embed_tokens.scales");
    let embed_bi = up("emb_bi", "model.embed_tokens.biases");

    let fnorm = up("fn", "model.norm.weight");
    let lmh = up("lmh", "lm_head.weight");
    let dsc = gpu.create_storage_buffer("ds", 4);

    let mut layers = Vec::new();
    let mut nw = Vec::new();
    let mut biases = Vec::new();
    let t0 = std::time::Instant::now();

    for i in 0..config.num_hidden_layers as usize {
        let p = format!("model.layers.{i}");
        nw.push(Some((
            get(&format!("{p}.self_attn.q_norm.weight")).to_vec(),
            get(&format!("{p}.self_attn.k_norm.weight")).to_vec(),
        )));

        let qq=up("qw",&format!("{p}.self_attn.q_proj.weight"));
        let qs=up("qs",&format!("{p}.self_attn.q_proj.scales"));
        let qb=up("qb",&format!("{p}.self_attn.q_proj.biases"));
        let kq=up("kw",&format!("{p}.self_attn.k_proj.weight"));
        let ks=up("ks",&format!("{p}.self_attn.k_proj.scales"));
        let kb=up("kb",&format!("{p}.self_attn.k_proj.biases"));
        let vq=up("vw",&format!("{p}.self_attn.v_proj.weight"));
        let vs=up("vs",&format!("{p}.self_attn.v_proj.scales"));
        let vb=up("vb",&format!("{p}.self_attn.v_proj.biases"));
        let oq=up("ow",&format!("{p}.self_attn.o_proj.weight"));
        let os=up("os",&format!("{p}.self_attn.o_proj.scales"));
        let ob=up("ob",&format!("{p}.self_attn.o_proj.biases"));
        let gq=up("gw",&format!("{p}.mlp.gate_proj.weight"));
        let gs=up("gs",&format!("{p}.mlp.gate_proj.scales"));
        let gb=up("gb",&format!("{p}.mlp.gate_proj.biases"));
        let uq=up("uw",&format!("{p}.mlp.up_proj.weight"));
        let us=up("us",&format!("{p}.mlp.up_proj.scales"));
        let ub=up("ub",&format!("{p}.mlp.up_proj.biases"));
        let dq=up("dw",&format!("{p}.mlp.down_proj.weight"));
        let dss=up("dss",&format!("{p}.mlp.down_proj.scales"));
        let db=up("db",&format!("{p}.mlp.down_proj.biases"));

        layers.push(LayerWeights {
            attn: AttnWeights::SelfAttn(SelfAttnWeights {
                q_proj_qweight:qq, q_proj_scales:qs,
                k_proj_qweight:kq, k_proj_scales:ks,
                v_proj_qweight:vq, v_proj_scales:vs,
                o_proj_qweight:oq, o_proj_scales:os,
                q_norm:up("qn",&format!("{p}.self_attn.q_norm.weight")),
                k_norm:up("kn",&format!("{p}.self_attn.k_norm.weight")),
            }),
            gate_proj_qweight:gq, gate_proj_scales:gs,
            up_proj_qweight:uq, up_proj_scales:us,
            down_proj_qweight:dq, down_proj_scales:dss,
            input_layernorm:up("il",&format!("{p}.input_layernorm.weight")),
            post_attn_layernorm:up("pl",&format!("{p}.post_attention_layernorm.weight")),
        });
        biases.push([qb, kb, vb, ob, gb, ub, db]);

        if (i+1)%7==0 { log::info!("[mlx-int4] layer {}/{} ({}ms)", i+1, config.num_hidden_layers, t0.elapsed().as_millis()); }
    }
    log::info!("[mlx-int4] done: {} layers in {}ms", layers.len(), t0.elapsed().as_millis());

    (ModelWeights {
        embed_tokens: embed_qw, final_norm: fnorm,
        lm_head_qweight: lmh, lm_head_scales: dsc, lm_head_is_bf16: true,
        self_attn_layers: (0..config.num_hidden_layers as usize).collect(),
        layers, embed_chunks: Vec::new(), embed_chunk_size: 0,
        mlx_biases: biases,
        embed_scales: Some(embed_sc), embed_biases: Some(embed_bi), bf16_scales: scales_are_bf16,
    }, RawNormWeights { layers: nw })
}
