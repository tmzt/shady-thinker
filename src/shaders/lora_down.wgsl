// lora_down.wgsl
// Down-projection to low rank: h[r] = sum_i(x[i] * A[i * rank + r])
// A is [in_features, rank] row-major.
// Each thread handles one rank element r; loops over in_features with unroll 4.

struct Params {
    in_features: u32,
    rank: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> lora_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rank) {
        return;
    }

    let in_features = params.in_features;
    let rank = params.rank;

    var acc: f32 = 0.0;

    // Number of full groups of 4
    let full_iters = in_features / 4u;
    let remainder = in_features % 4u;

    for (var g: u32 = 0u; g < full_iters; g = g + 1u) {
        let i = g * 4u;
        acc = acc + input[i]      * lora_a[i * rank + r]
                   + input[i + 1u] * lora_a[(i + 1u) * rank + r]
                   + input[i + 2u] * lora_a[(i + 2u) * rank + r]
                   + input[i + 3u] * lora_a[(i + 3u) * rank + r];
    }

    // Handle remaining elements
    let tail_start = full_iters * 4u;
    for (var i: u32 = tail_start; i < in_features; i = i + 1u) {
        acc = acc + input[i] * lora_a[i * rank + r];
    }

    output[r] = acc;
}
