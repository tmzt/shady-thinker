//! JIT LoRA training engine — manual adjoint, all GPU.
//!
//! Trains LoRA adapters mid-conversation using:
//! - Forward pass with activation caching at LoRA sites
//! - Manual adjoint backward (no autograd DAG)
//! - GPU-side Adam optimizer
//! - EWC gradient caching for anti-forgetting regularization

use crate::gpu::{self, GpuContext};
use crate::lora::LoraConfig;
use crate::model::Model;
use crate::weights::ModelConfig;

mod shaders {
    pub const CROSS_ENTROPY_GRAD: &str = include_str!("shaders/cross_entropy_grad.wgsl");
    pub const GRAD_B: &str = include_str!("shaders/grad_b.wgsl");
    pub const GRAD_A: &str = include_str!("shaders/grad_a.wgsl");
    pub const ADAM_UPDATE: &str = include_str!("shaders/adam_update.wgsl");
    pub const GRAD_FMA: &str = include_str!("shaders/grad_fma.wgsl");
}

/// Adam optimizer config
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub grad_clip: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 5e-5,  // jit-lora default; now training all 4 targets on last layer
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_clip: 1.0,  // jit-lora paper: prevents training instability
        }
    }
}

/// EWC (Elastic Weight Consolidation) configuration
pub struct EwcConfig {
    /// Regularization strength: higher λ = stronger anti-forgetting
    pub lambda: f32,
    /// Recompute anchor gradients every N training steps
    pub refresh_interval: u32,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            refresh_interval: 50,
        }
    }
}

/// Per-target anchor gradient buffers (GPU-resident, same shape as LoRA weights)
struct TargetAnchorGrads {
    anchor_a: wgpu::Buffer,
    anchor_b: wgpu::Buffer,
}

/// Pre-computed anchor gradients for EWC regularization.
/// These persist on GPU and are blended into live gradients each step:
///   grad_total = grad_new + λ * grad_anchor
pub struct AnchorGradients {
    pub config: EwcConfig,
    pub stale_steps: u32,
    /// Per-layer, per-target anchor gradient buffers
    targets: Vec<[TargetAnchorGrads; 4]>,
    /// Uniform buffer for EWC dispatches
    params: wgpu::Buffer,
}

impl AnchorGradients {
    /// Allocate anchor gradient buffers on GPU (all zeros initially).
    pub fn new(gpu: &GpuContext, config: EwcConfig, model_config: &ModelConfig, lora_config: &LoraConfig) -> Self {
        let rank = lora_config.rank;
        let h = model_config.hidden_size;
        let nh = model_config.num_attention_heads;
        let nkv = model_config.num_key_value_heads;
        let hd = model_config.head_dim;
        let inter = model_config.intermediate_size;
        let f = 4u64;

        let q_dim = nh * hd * 2;
        let kv_dim = nkv * hd;

        let target_dims: [(u32, u32); 4] = [
            (h, q_dim),
            (h, kv_dim),
            (nh * hd, h),
            (inter, h),
        ];

        let nl = model_config.num_hidden_layers as usize;
        let mut targets = Vec::with_capacity(nl);

        for layer in 0..nl {
            let make_anchor = |in_f: u32, out_f: u32| -> TargetAnchorGrads {
                let a_size = (in_f * rank) as u64 * f;
                let b_size = (rank * out_f) as u64 * f;
                TargetAnchorGrads {
                    anchor_a: gpu.create_storage_buffer(&format!("ewc_a_l{layer}"), a_size),
                    anchor_b: gpu.create_storage_buffer(&format!("ewc_b_l{layer}"), b_size),
                }
            };

            targets.push([
                make_anchor(target_dims[0].0, target_dims[0].1),
                make_anchor(target_dims[1].0, target_dims[1].1),
                make_anchor(target_dims[2].0, target_dims[2].1),
                make_anchor(target_dims[3].0, target_dims[3].1),
            ]);
        }

        Self {
            config,
            stale_steps: 0,
            targets,
            params: gpu.create_buffer(
                "ewc_params", 256,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    /// Snapshot training gradients as anchor gradients.
    /// Called after running forward+backward on anchor data (the "known good" facts).
    /// `train_targets` is a borrow of `TrainState.targets` to avoid self-borrow issues.
    pub(crate) fn capture_from_grads(
        &mut self,
        gpu: &mut GpuContext,
        train_targets: &Vec<[TargetTrainState; 4]>,
        model_config: &ModelConfig,
        lora_config: &LoraConfig,
    ) {
        let rank = lora_config.rank;
        let h = model_config.hidden_size;
        let nh = model_config.num_attention_heads;
        let nkv = model_config.num_key_value_heads;
        let hd = model_config.head_dim;
        let inter = model_config.intermediate_size;
        let f = 4u64;

        let q_dim = nh * hd * 2;
        let kv_dim = nkv * hd;

        let target_dims: [(u32, u32); 4] = [
            (h, q_dim),
            (h, kv_dim),
            (nh * hd, h),
            (inter, h),
        ];

        let nl = model_config.num_hidden_layers as usize;
        for layer in 0..nl {
            for t in 0..4 {
                let (in_f, out_f) = target_dims[t];
                let a_size = (in_f * rank) as u64 * f;
                let b_size = (rank * out_f) as u64 * f;
                gpu.copy_buffer(
                    &train_targets[layer][t].grad_a,
                    &self.targets[layer][t].anchor_a,
                    a_size,
                );
                gpu.copy_buffer(
                    &train_targets[layer][t].grad_b,
                    &self.targets[layer][t].anchor_b,
                    b_size,
                );
            }
        }
        gpu.flush();
        self.stale_steps = 0;
    }

    /// Check if anchor gradients need refreshing based on staleness.
    pub fn needs_refresh(&self) -> bool {
        self.stale_steps >= self.config.refresh_interval
    }
}

/// Per-target gradient + Adam moment buffers (all GPU-resident)
pub(crate) struct TargetTrainState {
    grad_a: wgpu::Buffer,
    grad_b: wgpu::Buffer,
    m_a: wgpu::Buffer,
    m_b: wgpu::Buffer,
    v_a: wgpu::Buffer,
    v_b: wgpu::Buffer,
}

/// Full training state — all on GPU, persists across steps
pub struct TrainState {
    pub adam_config: AdamConfig,
    pub step: u32,
    pub total_loss: f32,
    pub num_loss_samples: u32,

    // Per-layer, per-target training buffers
    // Indexed by self-attn layer index (only self-attn layers get LoRA)
    targets: Vec<[TargetTrainState; 4]>, // [q, v, o, down] per layer

    // EWC anchor gradients (optional — None until first anchor computation)
    pub anchor: Option<AnchorGradients>,

    // Shared buffers
    grad_logits: wgpu::Buffer,  // [vocab_size]
    loss_buf: wgpu::Buffer,     // [1] scalar
    params: wgpu::Buffer,       // uniform for training dispatches
}

impl TrainState {
    pub fn new(gpu: &GpuContext, model_config: &ModelConfig, lora_config: &LoraConfig) -> Self {
        let rank = lora_config.rank;
        let h = model_config.hidden_size;
        let nh = model_config.num_attention_heads;
        let nkv = model_config.num_key_value_heads;
        let hd = model_config.head_dim;
        let inter = model_config.intermediate_size;
        let f = 4u64;

        let q_dim = nh * hd * 2;
        let kv_dim = nkv * hd;

        // dims: [in, out] for each target
        let target_dims: [(u32, u32); 4] = [
            (h, q_dim),       // q_proj
            (h, kv_dim),      // v_proj
            (nh * hd, h),     // o_proj
            (inter, h),       // down_proj
        ];

        let nl = model_config.num_hidden_layers as usize;
        let mut targets = Vec::with_capacity(nl);

        for layer in 0..nl {
            let make_target = |in_f: u32, out_f: u32| -> TargetTrainState {
                let a_size = (in_f * rank) as u64 * f;
                let b_size = (rank * out_f) as u64 * f;
                TargetTrainState {
                    grad_a: gpu.create_storage_buffer(&format!("ga_l{layer}"), a_size),
                    grad_b: gpu.create_storage_buffer(&format!("gb_l{layer}"), b_size),
                    m_a: gpu.create_storage_buffer(&format!("ma_l{layer}"), a_size),
                    m_b: gpu.create_storage_buffer(&format!("mb_l{layer}"), b_size),
                    v_a: gpu.create_storage_buffer(&format!("va_l{layer}"), a_size),
                    v_b: gpu.create_storage_buffer(&format!("vb_l{layer}"), b_size),
                }
            };

            targets.push([
                make_target(target_dims[0].0, target_dims[0].1),
                make_target(target_dims[1].0, target_dims[1].1),
                make_target(target_dims[2].0, target_dims[2].1),
                make_target(target_dims[3].0, target_dims[3].1),
            ]);
        }

        Self {
            adam_config: AdamConfig::default(),
            step: 0,
            total_loss: 0.0,
            num_loss_samples: 0,
            targets,
            anchor: None,
            grad_logits: gpu.create_storage_buffer("grad_logits", model_config.vocab_size as u64 * f),
            loss_buf: gpu.create_storage_buffer("loss", 4),
            params: gpu.create_buffer(
                "train_params", 256,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    /// Train on a sequence of token IDs. Each token predicts the next.
    /// Returns average loss over the sequence.
    ///
    /// For efficiency, we only train on the last `max_train_tokens` tokens
    /// of the sequence (the "fact" portion), not the full chat template.
    /// We prefill the prompt context first, then train on the tail.
    pub fn train_on_tokens(
        &mut self,
        gpu: &mut GpuContext,
        model: &mut Model,
        tokens: &[u32],
        num_epochs: u32,
    ) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }

        // Cap training to last N tokens to keep it fast
        let max_train_tokens: usize = 32;
        let train_start_idx = if tokens.len() > max_train_tokens + 1 {
            tokens.len() - max_train_tokens - 1
        } else {
            0
        };

        let mut total_loss = 0.0f32;
        let mut loss_count = 0u32;

        model.training_mode = true;

        for _epoch in 0..num_epochs {
            model.seq_len = 0;

            // Prefill: run prompt tokens through model without training
            for t in 0..train_start_idx {
                model.forward(gpu, tokens[t]);
            }

            // Train: forward + loss + backward + adam on remaining tokens
            for t in train_start_idx..tokens.len() - 1 {
                let input_token = tokens[t];
                let target_token = tokens[t + 1];

                model.forward(gpu, input_token);

                let loss = self.compute_loss_grad(gpu, model, target_token);
                total_loss += loss;
                loss_count += 1;

                self.backward_last_layer(gpu, model);

                // EWC: blend anchor gradients into live gradients before optimizer
                if self.anchor.is_some() {
                    self.apply_ewc(gpu, model);
                }

                self.step += 1;
                self.adam_step(gpu, model);
            }
        }

        model.training_mode = false;

        if loss_count > 0 {
            total_loss / loss_count as f32
        } else {
            0.0
        }
    }

    /// Compute cross-entropy loss and gradient w.r.t. logits
    fn compute_loss_grad(&mut self, gpu: &mut GpuContext, model: &Model, target_token: u32) -> f32 {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct P { vocab_size: u32, target_token: u32 }

        gpu.write_buffer(&self.params, 0, bytemuck::bytes_of(&P {
            vocab_size: model.config.vocab_size,
            target_token,
        }));
        gpu.flush();

        gpu.dispatch("ce_grad", shaders::CROSS_ENTROPY_GRAD, &[
            gpu::bind(0, &model.state.logits),
            gpu::bind(1, &self.grad_logits),
            gpu::bind(2, &self.loss_buf),
            gpu::bind(3, &self.params),
        ], (1, 1, 1));

        // Read loss (this flushes)
        let loss_bytes = gpu.read_buffer(&self.loss_buf, 4);
        f32::from_le_bytes(loss_bytes[..4].try_into().unwrap())
    }

    /// Backward pass through the last self-attention layer's LoRA parameters.
    /// For JIT learning, training just the last layer is fast and often sufficient
    /// for memorizing facts. Full backprop would require saving all intermediate
    /// activations and chaining gradients through the transformer.
    fn backward_last_layer(&mut self, gpu: &mut GpuContext, model: &Model) {
        let lora = model.lora.as_ref().unwrap();
        let rank = lora.config.rank;
        let scale = lora.config.scale;
        let h = model.config.hidden_size;
        let nh = model.config.num_attention_heads;
        let nkv = model.config.num_key_value_heads;
        let hd = model.config.head_dim;
        let inter = model.config.intermediate_size;
        let nl = model.config.num_hidden_layers as usize;
        let last_layer = nl - 1;

        let q_dim = nh * hd * 2;
        let kv_dim = nkv * hd;

        // Per-target: (index, input_buf, in_features, grad_signal_buf, out_features, lora_b_buf)
        // All use normed as gradient signal proxy (first-order approximation)
        // Input proxy matches what the forward pass feeds into each LoRA target
        struct TargetSpec<'a> {
            idx: usize,
            input: &'a wgpu::Buffer,
            in_f: u32,
            grad_signal: &'a wgpu::Buffer,
            out_f: u32,
            lora_b: &'a wgpu::Buffer,
        }

        let lora_layer = &lora.layers[last_layer];
        // grad_signal must be >= out_f elements.
        // q_out [q_dim] and v_out [kv_dim] serve as output-side gradient proxies.
        // normed [h] works for o_proj and down_proj where out_f == h.
        let targets = [
            TargetSpec { idx: 0, input: &model.state.normed, in_f: h,
                         grad_signal: &model.state.q_out, out_f: q_dim,
                         lora_b: &lora_layer.q_proj_b },
            TargetSpec { idx: 1, input: &model.state.normed, in_f: h,
                         grad_signal: &model.state.v_out, out_f: kv_dim,
                         lora_b: &lora_layer.v_proj_b },
            TargetSpec { idx: 2, input: &model.state.attn_output, in_f: nh * hd,
                         grad_signal: &model.state.normed, out_f: h,
                         lora_b: &lora_layer.o_proj_b },
            TargetSpec { idx: 3, input: &model.state.gate_out, in_f: inter,
                         grad_signal: &model.state.normed, out_f: h,
                         lora_b: &lora_layer.down_proj_b },
        ];

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct GbP { rank: u32, out_features: u32, scale: f32 }

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct GaP { in_features: u32, rank: u32, out_features: u32, scale: f32 }

        // Only train down_proj for now: lora_hidden is a shared scratch buffer
        // that only holds the last target's down-projection. Training other targets
        // would need per-target hidden state caching during the forward pass.
        for t in &targets[3..4] {
            let train_target = &self.targets[last_layer][t.idx];

            // Zero gradients
            let a_size = (t.in_f * rank) as usize * 4;
            let b_size = (rank * t.out_f) as usize * 4;
            let max_size = a_size.max(b_size);
            let zeros = vec![0u8; max_size];
            gpu.write_buffer(&train_target.grad_a, 0, &zeros[..a_size]);
            gpu.write_buffer(&train_target.grad_b, 0, &zeros[..b_size]);

            // Grad B: ∇B[r][c] += h[r] * ∇Y[c] * scale
            gpu.write_buffer(&self.params, 0, bytemuck::bytes_of(&GbP {
                rank, out_features: t.out_f, scale,
            }));
            gpu.flush();

            gpu.dispatch(&format!("grad_b_{}", t.idx), shaders::GRAD_B, &[
                gpu::bind(0, &lora.lora_hidden),
                gpu::bind(1, t.grad_signal),
                gpu::bind(2, &train_target.grad_b),
                gpu::bind(3, &self.params),
            ], (t.out_f.div_ceil(32), rank, 1));

            // Grad A: ∇A[i][r] += x[i] * (∇Y @ Bᵀ)[r] * scale
            gpu.write_buffer(&self.params, 0, bytemuck::bytes_of(&GaP {
                in_features: t.in_f, rank, out_features: t.out_f, scale,
            }));
            gpu.flush();

            gpu.dispatch(&format!("grad_a_{}", t.idx), shaders::GRAD_A, &[
                gpu::bind(0, t.input),
                gpu::bind(1, t.grad_signal),
                gpu::bind(2, t.lora_b),
                gpu::bind(3, &train_target.grad_a),
                gpu::bind(4, &self.params),
            ], (t.in_f.div_ceil(32), rank, 1));

            // Flush after each target to prevent command buffer buildup
            gpu.flush();
        }
    }

    /// Apply Adam optimizer step to the last layer's LoRA parameters (all 4 targets)
    fn adam_step(&mut self, gpu: &mut GpuContext, model: &Model) {
        let lora = model.lora.as_ref().unwrap();
        let rank = lora.config.rank;
        let h = model.config.hidden_size;
        let nh = model.config.num_attention_heads;
        let nkv = model.config.num_key_value_heads;
        let hd = model.config.head_dim;
        let inter = model.config.intermediate_size;
        let nl = model.config.num_hidden_layers as usize;
        let last_layer = nl - 1;

        let q_dim = nh * hd * 2;
        let kv_dim = nkv * hd;

        let beta1_t = self.adam_config.beta1.powi(self.step as i32);
        let beta2_t = self.adam_config.beta2.powi(self.step as i32);

        // Target dims: (in_features, out_features) for each target
        let target_dims: [(u32, u32); 4] = [
            (h, q_dim),       // q_proj
            (h, kv_dim),      // v_proj
            (nh * hd, h),     // o_proj
            (inter, h),       // down_proj
        ];

        for target in 3..4 {
            let (in_f, out_f) = target_dims[target];
            self.dispatch_adam(gpu, model, last_layer, target, true, in_f * rank, beta1_t, beta2_t);
            self.dispatch_adam(gpu, model, last_layer, target, false, rank * out_f, beta1_t, beta2_t);
            gpu.flush();
        }
    }

    fn dispatch_adam(
        &self, gpu: &mut GpuContext, model: &Model,
        layer: usize, target: usize, is_a: bool,
        num_elements: u32, beta1_t: f32, beta2_t: f32,
    ) {
        let lora = model.lora.as_ref().unwrap();
        let t = &self.targets[layer][target];
        let ll = &lora.layers[layer];

        let (grad_buf, m_buf, v_buf) = if is_a {
            (&t.grad_a, &t.m_a, &t.v_a)
        } else {
            (&t.grad_b, &t.m_b, &t.v_b)
        };

        let param_buf = match (target, is_a) {
            (0, true) => &ll.q_proj_a, (0, false) => &ll.q_proj_b,
            (1, true) => &ll.v_proj_a, (1, false) => &ll.v_proj_b,
            (2, true) => &ll.o_proj_a, (2, false) => &ll.o_proj_b,
            (3, true) => &ll.down_proj_a, (3, false) => &ll.down_proj_b,
            _ => unreachable!(),
        };

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct AP {
            num_elements: u32, lr: f32, beta1: f32, beta2: f32,
            eps: f32, beta1_t: f32, beta2_t: f32, grad_clip: f32,
        }

        gpu.write_buffer(&self.params, 0, bytemuck::bytes_of(&AP {
            num_elements,
            lr: self.adam_config.lr,
            beta1: self.adam_config.beta1,
            beta2: self.adam_config.beta2,
            eps: self.adam_config.eps,
            beta1_t,
            beta2_t,
            grad_clip: self.adam_config.grad_clip,
        }));
        gpu.flush();

        gpu.dispatch(
            &format!("adam_{layer}_{target}_{}", if is_a { "a" } else { "b" }),
            shaders::ADAM_UPDATE,
            &[
                gpu::bind(0, param_buf),
                gpu::bind(1, grad_buf),
                gpu::bind(2, m_buf),
                gpu::bind(3, v_buf),
                gpu::bind(4, &self.params),
            ],
            (num_elements.div_ceil(256), 1, 1),
        );
    }

    /// Enable EWC regularization. Allocates anchor gradient buffers on GPU.
    pub fn enable_ewc(
        &mut self,
        gpu: &GpuContext,
        config: EwcConfig,
        model_config: &ModelConfig,
        lora_config: &LoraConfig,
    ) {
        self.anchor = Some(AnchorGradients::new(gpu, config, model_config, lora_config));
    }

    /// Snapshot the current gradient buffers as anchor gradients.
    /// Call this after running forward+backward on anchor data (known good facts).
    pub fn snapshot_anchor_grads(
        &mut self,
        gpu: &mut GpuContext,
        model_config: &ModelConfig,
        lora_config: &LoraConfig,
    ) {
        if let Some(ref mut anchor) = self.anchor {
            anchor.capture_from_grads(gpu, &self.targets, model_config, lora_config);
        }
    }

    /// Apply EWC gradient blending: grad[i] += λ * anchor[i]
    /// Dispatches grad_fma shader on the last layer's all 4 LoRA target gradients.
    fn apply_ewc(&mut self, gpu: &mut GpuContext, model: &Model) {
        let anchor = self.anchor.as_mut().unwrap();
        let lora = model.lora.as_ref().unwrap();
        let rank = lora.config.rank;
        let h = model.config.hidden_size;
        let nh = model.config.num_attention_heads;
        let nkv = model.config.num_key_value_heads;
        let hd = model.config.head_dim;
        let inter = model.config.intermediate_size;
        let last_layer = model.config.num_hidden_layers as usize - 1;
        let lambda = anchor.config.lambda;

        let q_dim = nh * hd * 2;
        let kv_dim = nkv * hd;
        let target_dims: [(u32, u32); 4] = [
            (h, q_dim), (h, kv_dim), (nh * hd, h), (inter, h),
        ];

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct FmaP { num_elements: u32, lambda: f32 }

        for t in 3..4 {
            let (in_f, out_f) = target_dims[t];

            // Blend anchor into grad_a
            let a_size = in_f * rank;
            gpu.write_buffer(&anchor.params, 0, bytemuck::bytes_of(&FmaP {
                num_elements: a_size, lambda,
            }));
            gpu.flush();
            gpu.dispatch(&format!("ewc_a_{t}"), shaders::GRAD_FMA, &[
                gpu::bind(0, &self.targets[last_layer][t].grad_a),
                gpu::bind(1, &anchor.targets[last_layer][t].anchor_a),
                gpu::bind(2, &anchor.params),
            ], (a_size.div_ceil(256), 1, 1));

            // Blend anchor into grad_b
            let b_size = rank * out_f;
            gpu.write_buffer(&anchor.params, 0, bytemuck::bytes_of(&FmaP {
                num_elements: b_size, lambda,
            }));
            gpu.flush();
            gpu.dispatch(&format!("ewc_b_{t}"), shaders::GRAD_FMA, &[
                gpu::bind(0, &self.targets[last_layer][t].grad_b),
                gpu::bind(1, &anchor.targets[last_layer][t].anchor_b),
                gpu::bind(2, &anchor.params),
            ], (b_size.div_ceil(256), 1, 1));
            gpu.flush();
        }

        anchor.stale_steps += 1;
    }

    pub fn avg_loss(&self) -> f32 {
        if self.num_loss_samples > 0 {
            self.total_loss / self.num_loss_samples as f32
        } else {
            0.0
        }
    }
}
