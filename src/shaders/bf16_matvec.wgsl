// BF16 Matrix-Vector Multiply: output[row] = sum_i(weight[row, i] * input[i])
// Weights are BF16 packed (two BF16 values per u32), row-major layout.
// Dispatch: (ceil(vocab_size / 32), 1, 1)

struct Params {
    hidden_size: u32,
    vocab_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x * 32u + lid.x;
    if (row >= params.vocab_size) {
        return;
    }

    let hidden_size = params.hidden_size;
    let half_hidden = hidden_size / 2u;
    let base = row * half_hidden;

    var sum: f32 = 0.0;

    // Each u32 packs 2 BF16 values. Iterate over packed u32s.
    // Unroll by 4 packed elements (8 BF16 values) at a time.
    let packed_count = half_hidden;
    let unroll_end = packed_count & ~3u;

    var i: u32 = 0u;
    while (i < unroll_end) {
        let p0 = weight[base + i];
        let p1 = weight[base + i + 1u];
        let p2 = weight[base + i + 2u];
        let p3 = weight[base + i + 3u];

        let inp_base = i * 2u;

        sum += unpack_bf16(p0, 0u) * input[inp_base];
        sum += unpack_bf16(p0, 1u) * input[inp_base + 1u];
        sum += unpack_bf16(p1, 0u) * input[inp_base + 2u];
        sum += unpack_bf16(p1, 1u) * input[inp_base + 3u];
        sum += unpack_bf16(p2, 0u) * input[inp_base + 4u];
        sum += unpack_bf16(p2, 1u) * input[inp_base + 5u];
        sum += unpack_bf16(p3, 0u) * input[inp_base + 6u];
        sum += unpack_bf16(p3, 1u) * input[inp_base + 7u];

        i += 4u;
    }

    // Handle remainder
    while (i < packed_count) {
        let p = weight[base + i];
        let inp_base = i * 2u;
        sum += unpack_bf16(p, 0u) * input[inp_base];
        sum += unpack_bf16(p, 1u) * input[inp_base + 1u];
        i += 1u;
    }

    output[row] = sum;
}
