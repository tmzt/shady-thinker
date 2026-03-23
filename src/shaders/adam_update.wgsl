// Bias-corrected Adam optimizer with gradient clipping.
// Dispatch: (ceil(num_elements/256), 1, 1)

struct Params {
    num_elements: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    beta1_t: f32,
    beta2_t: f32,
    grad_clip: f32,
}

@group(0) @binding(0) var<storage, read_write>  param: array<f32>;
@group(0) @binding(1) var<storage, read>        grad:  array<f32>;
@group(0) @binding(2) var<storage, read_write>  m:     array<f32>;
@group(0) @binding(3) var<storage, read_write>  v:     array<f32>;
@group(0) @binding(4) var<uniform>              params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.num_elements {
        return;
    }

    let g = clamp(grad[i], -params.grad_clip, params.grad_clip);

    let beta1 = params.beta1;
    let beta2 = params.beta2;

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    let m_hat = m[i] / (1.0f - params.beta1_t);
    let v_hat = v[i] / (1.0f - params.beta2_t);

    param[i] -= params.lr * m_hat / (sqrt(v_hat) + params.eps);
}
