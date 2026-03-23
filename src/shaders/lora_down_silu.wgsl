// lora_down_silu.wgsl
// Fused SiLU activation + down-projection:
//   h[r] = sum_i(SiLU(gate[i]) * up[i] * A[i * rank + r])
// SiLU(x) = x / (1.0 + exp(-x))
// A is [in_features, rank] row-major.
// Each thread handles one rank element r; loops over in_features with unroll 4.

struct Params {
    in_features: u32,
    rank: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> lora_a: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

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

        let s0 = silu(gate[i])      * up[i];
        let s1 = silu(gate[i + 1u]) * up[i + 1u];
        let s2 = silu(gate[i + 2u]) * up[i + 2u];
        let s3 = silu(gate[i + 3u]) * up[i + 3u];

        acc = acc + s0 * lora_a[i * rank + r]
                  + s1 * lora_a[(i + 1u) * rank + r]
                  + s2 * lora_a[(i + 2u) * rank + r]
                  + s3 * lora_a[(i + 3u) * rank + r];
    }

    // Handle remaining elements
    let tail_start = full_iters * 4u;
    for (var i: u32 = tail_start; i < in_features; i = i + 1u) {
        acc = acc + silu(gate[i]) * up[i] * lora_a[i * rank + r];
    }

    output[r] = acc;
}
