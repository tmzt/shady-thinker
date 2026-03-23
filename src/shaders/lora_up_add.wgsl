// lora_up_add.wgsl
// Up-projection with residual add: output[c] += scale * sum_r(h[r] * B[r * out_features + c])
// B is [rank, out_features] row-major.
// Each thread handles one output column c; loops over rank.

struct Params {
    rank: u32,
    out_features: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> lora_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.out_features) {
        return;
    }

    let rank = params.rank;
    let out_features = params.out_features;
    let scale = params.scale;

    var acc: f32 = 0.0;

    for (var r: u32 = 0u; r < rank; r = r + 1u) {
        acc = acc + hidden[r] * lora_b[r * out_features + c];
    }

    output[c] = output[c] + scale * acc;
}
