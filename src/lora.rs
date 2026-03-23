use std::path::Path;

use crate::gpu::GpuContext;
use crate::weights::ModelConfig;

/// LoRA configuration
pub struct LoraConfig {
    pub rank: u32,
    pub alpha: f32,
    pub scale: f32,         // alpha / rank
    pub targets: [bool; 4], // [q_proj, v_proj, o_proj, down_proj]
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 32,
            alpha: 32.0,
            scale: 1.0,
            targets: [true, true, true, true],
        }
    }
}

/// Per-layer LoRA weight buffers (all GPU-resident)
pub struct LoraLayerWeights {
    pub q_proj_a: wgpu::Buffer,
    pub q_proj_b: wgpu::Buffer,
    pub v_proj_a: wgpu::Buffer,
    pub v_proj_b: wgpu::Buffer,
    pub o_proj_a: wgpu::Buffer,
    pub o_proj_b: wgpu::Buffer,
    pub down_proj_a: wgpu::Buffer,
    pub down_proj_b: wgpu::Buffer,
}

/// Full LoRA state — all buffers live on GPU
pub struct LoraState {
    pub config: LoraConfig,
    pub layers: Vec<LoraLayerWeights>,
    pub lora_hidden: wgpu::Buffer, // [rank] scratch, reused across all projections
}

/// Per-target dimensions for a layer
struct TargetDims {
    q_in: u32,
    q_out: u32,
    v_in: u32,
    v_out: u32,
    o_in: u32,
    o_out: u32,
    down_in: u32,
    down_out: u32,
}

fn target_dims(config: &ModelConfig) -> TargetDims {
    let h = config.hidden_size;
    let nh = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    let inter = config.intermediate_size;

    TargetDims {
        q_in: h,
        q_out: nh * hd * 2, // interleaved q + q_gate
        v_in: h,
        v_out: nkv * hd,
        o_in: nh * hd,
        o_out: h,
        down_in: inter,
        down_out: h,
    }
}

/// Kaiming uniform initialization: values from N(0, sqrt(2/fan_in))
/// Returns CPU-side f32 buffer ready for upload.
fn kaiming_init(fan_in: u32, size: usize, seed: u64) -> Vec<f32> {
    let std_dev = (2.0 / fan_in as f64).sqrt();
    // Simple xorshift64 PRNG — good enough for init
    let mut state = seed ^ 0x5DEECE66D;
    let mut out = Vec::with_capacity(size);
    for _ in 0..size {
        // Box-Muller for normal distribution
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (state & 0xFFFFFFFF) as f64 / u32::MAX as f64;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state & 0xFFFFFFFF) as f64 / u32::MAX as f64;
        let z = (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        out.push((z * std_dev) as f32);
    }
    out
}

impl LoraState {
    /// Create a new LoRA state with Kaiming-init A matrices and zero B matrices.
    /// All buffers are allocated on GPU.
    pub fn new(gpu: &GpuContext, config: LoraConfig, model_config: &ModelConfig) -> Self {
        let rank = config.rank;
        let dims = target_dims(model_config);
        let nl = model_config.num_hidden_layers as usize;
        let f = 4u64; // sizeof f32

        let mut layers = Vec::with_capacity(nl);
        let mut seed = 42u64;

        for i in 0..nl {
            let make_a = |fan_in: u32, fan_out: u32, s: &mut u64| -> wgpu::Buffer {
                let size = (fan_in * rank) as usize;
                *s = s.wrapping_add(i as u64 * 7 + fan_in as u64);
                let data = kaiming_init(fan_in, size, *s);
                gpu.upload_buffer(
                    &format!("lora_a_l{i}_{fan_in}x{fan_out}"),
                    bytemuck::cast_slice(&data),
                )
            };

            let make_b = |out_features: u32| -> wgpu::Buffer {
                // Zero-initialized: LoRA starts as identity
                let size = (rank * out_features) as u64 * f;
                gpu.create_storage_buffer(&format!("lora_b_l{i}_{rank}x{out_features}"), size)
            };

            layers.push(LoraLayerWeights {
                q_proj_a: make_a(dims.q_in, dims.q_out, &mut seed),
                q_proj_b: make_b(dims.q_out),
                v_proj_a: make_a(dims.v_in, dims.v_out, &mut seed),
                v_proj_b: make_b(dims.v_out),
                o_proj_a: make_a(dims.o_in, dims.o_out, &mut seed),
                o_proj_b: make_b(dims.o_out),
                down_proj_a: make_a(dims.down_in, dims.down_out, &mut seed),
                down_proj_b: make_b(dims.down_out),
            });
        }

        let lora_hidden = gpu.create_storage_buffer("lora_hidden", rank as u64 * f);

        Self {
            config,
            layers,
            lora_hidden,
        }
    }

    /// Load LoRA adapter weights from safetensors file.
    /// Uploads to GPU and discards CPU copies.
    pub fn load_safetensors(
        gpu: &GpuContext,
        path: &Path,
        model_config: &ModelConfig,
    ) -> Self {
        use safetensors::SafeTensors;

        let data = std::fs::read(path).expect("failed to read LoRA safetensors");
        let tensors = SafeTensors::deserialize(&data).expect("failed to parse LoRA safetensors");

        let config = LoraConfig::default();
        let rank = config.rank;
        let dims = target_dims(model_config);
        let nl = model_config.num_hidden_layers as usize;
        let f = 4u64;

        let get_or_zero = |name: &str, size: u64| -> wgpu::Buffer {
            if let Ok(view) = tensors.tensor(name) {
                gpu.upload_buffer(name, view.data())
            } else {
                gpu.create_storage_buffer(name, size)
            }
        };

        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            // PEFT-compatible naming convention
            let pfx = format!("base_model.model.layers.{i}");

            layers.push(LoraLayerWeights {
                q_proj_a: get_or_zero(
                    &format!("{pfx}.self_attn.q_proj.lora_A.weight"),
                    dims.q_in as u64 * rank as u64 * f,
                ),
                q_proj_b: get_or_zero(
                    &format!("{pfx}.self_attn.q_proj.lora_B.weight"),
                    rank as u64 * dims.q_out as u64 * f,
                ),
                v_proj_a: get_or_zero(
                    &format!("{pfx}.self_attn.v_proj.lora_A.weight"),
                    dims.v_in as u64 * rank as u64 * f,
                ),
                v_proj_b: get_or_zero(
                    &format!("{pfx}.self_attn.v_proj.lora_B.weight"),
                    rank as u64 * dims.v_out as u64 * f,
                ),
                o_proj_a: get_or_zero(
                    &format!("{pfx}.self_attn.o_proj.lora_A.weight"),
                    dims.o_in as u64 * rank as u64 * f,
                ),
                o_proj_b: get_or_zero(
                    &format!("{pfx}.self_attn.o_proj.lora_B.weight"),
                    rank as u64 * dims.o_out as u64 * f,
                ),
                down_proj_a: get_or_zero(
                    &format!("{pfx}.mlp.down_proj.lora_A.weight"),
                    dims.down_in as u64 * rank as u64 * f,
                ),
                down_proj_b: get_or_zero(
                    &format!("{pfx}.mlp.down_proj.lora_B.weight"),
                    rank as u64 * dims.down_out as u64 * f,
                ),
            });
        }

        let lora_hidden = gpu.create_storage_buffer("lora_hidden", rank as u64 * f);

        Self {
            config,
            layers,
            lora_hidden,
        }
    }

    /// Save LoRA adapter weights to safetensors file.
    /// This is the one allowed GPU→CPU readback — periodic checkpoint to UFS.
    pub fn save_safetensors(&self, gpu: &mut GpuContext, path: &Path, model_config: &ModelConfig) {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let rank = self.config.rank;
        let dims = target_dims(model_config);

        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();
        let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let pfx = format!("base_model.model.layers.{i}");

            let pairs: &[(&str, &wgpu::Buffer, &[usize])] = &[
                ("self_attn.q_proj.lora_A.weight", &layer.q_proj_a, &[dims.q_in as usize, rank as usize]),
                ("self_attn.q_proj.lora_B.weight", &layer.q_proj_b, &[rank as usize, dims.q_out as usize]),
                ("self_attn.v_proj.lora_A.weight", &layer.v_proj_a, &[dims.v_in as usize, rank as usize]),
                ("self_attn.v_proj.lora_B.weight", &layer.v_proj_b, &[rank as usize, dims.v_out as usize]),
                ("self_attn.o_proj.lora_A.weight", &layer.o_proj_a, &[dims.o_in as usize, rank as usize]),
                ("self_attn.o_proj.lora_B.weight", &layer.o_proj_b, &[rank as usize, dims.o_out as usize]),
                ("mlp.down_proj.lora_A.weight", &layer.down_proj_a, &[dims.down_in as usize, rank as usize]),
                ("mlp.down_proj.lora_B.weight", &layer.down_proj_b, &[rank as usize, dims.down_out as usize]),
            ];

            for (suffix, buf, shape) in pairs {
                let name = format!("{pfx}.{suffix}");
                let size = shape.iter().product::<usize>() * 4;
                let data = gpu.read_buffer(buf, size as u64);
                tensors.insert(name.clone(), data);
                metadata.push((name, shape.to_vec()));
            }
        }

        // Build TensorView map and serialize
        let views: HashMap<String, TensorView<'_>> = metadata
            .iter()
            .map(|(name, shape)| {
                let data = tensors.get(name).unwrap();
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let serialized = safetensors::tensor::serialize(&views, &None).unwrap();
        std::fs::write(path, serialized).expect("failed to write LoRA safetensors");

        log::info!("Saved LoRA adapter to {:?} ({} tensors)", path, metadata.len());
    }
}
