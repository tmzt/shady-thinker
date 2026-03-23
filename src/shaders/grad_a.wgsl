// LoRA A gradient: backprop through B then outer product with input.
// grad_a[feat, r] += input_x[feat] * (sum_c(grad_y[c] * B[r, c]) * scale)
// Dispatch: (ceil(in_features/32), rank, 1)

struct Params {
    in_features: u32,
    rank: u32,
    out_features: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read>       input_x: array<f32>;
@group(0) @binding(1) var<storage, read>       grad_y:  array<f32>;
@group(0) @binding(2) var<storage, read>       lora_b:  array<f32>;
@group(0) @binding(3) var<storage, read_write>  grad_a:  array<f32>;
@group(0) @binding(4) var<uniform>              params:  Params;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let r = wid.y;
    let feat = wid.x * 32u + lid.x;
    let in_features = params.in_features;
    let out_features = params.out_features;
    let rank = params.rank;

    if feat >= in_features {
        return;
    }

    // dh[r] = sum_c(grad_y[c] * B[r * out_features + c]) * scale
    var dh = 0.0f;
    let row_offset = r * out_features;
    for (var c = 0u; c < out_features; c++) {
        dh += grad_y[c] * lora_b[row_offset + c];
    }
    dh *= params.scale;

    grad_a[feat * rank + r] += input_x[feat] * dh;
}
