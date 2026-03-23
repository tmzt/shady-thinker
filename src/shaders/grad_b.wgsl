// LoRA B gradient: outer product grad_b[r, c] += hidden[r] * grad_y[c] * scale.
// Dispatch: (ceil(out_features/32), rank, 1)

struct Params {
    rank: u32,
    out_features: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read>       hidden: array<f32>;
@group(0) @binding(1) var<storage, read>       grad_y: array<f32>;
@group(0) @binding(2) var<storage, read_write>  grad_b: array<f32>;
@group(0) @binding(3) var<uniform>              params: Params;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let r = wid.y;
    let c = wid.x * 32u + lid.x;
    let out_features = params.out_features;

    if c >= out_features {
        return;
    }

    let idx = r * out_features + c;
    grad_b[idx] += hidden[r] * grad_y[c] * params.scale;
}
