// BF16 Batched GEMM: output[row, col] = sum_k(input[row, k] * weight[col, k]) + bias[col]
// Computes: [seq_len, d_in] × [d_out, d_in]^T + [d_out] → [seq_len, d_out]
// Weights are BF16 packed (two BF16 values per u32), row-major [d_out, d_in/2].
// Bias is f32 (optional — set has_bias=0 to skip).
// Input and output are f32.
//
// Dispatch: (ceil(d_out / 32), seq_len, 1)
//   workgroup_id.x * 32 + local_id.x = output column
//   workgroup_id.y = input row (token index)

struct Params {
    d_in: u32,       // input dimension (columns of input, columns of weight)
    d_out: u32,      // output dimension (rows of weight)
    seq_len: u32,    // number of input rows (tokens)
    has_bias: u32,   // 1 = add bias, 0 = no bias
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;  // output column (weight row)
    let row = wg_id.y;                  // input row (token index)

    if (col >= params.d_out || row >= params.seq_len) {
        return;
    }

    let d_in = params.d_in;
    let half_d_in = d_in / 2u;

    // Weight base: row-major [d_out, d_in/2] packed
    let w_base = col * half_d_in;
    // Input base: row-major [seq_len, d_in]
    let in_base = row * d_in;

    var sum: f32 = 0.0;

    // Unroll by 4 packed elements (8 bf16 values)
    let unroll_end = half_d_in & ~3u;
    var i: u32 = 0u;
    while (i < unroll_end) {
        let p0 = weight[w_base + i];
        let p1 = weight[w_base + i + 1u];
        let p2 = weight[w_base + i + 2u];
        let p3 = weight[w_base + i + 3u];
        let k = i * 2u;

        sum += unpack_bf16(p0, 0u) * input[in_base + k];
        sum += unpack_bf16(p0, 1u) * input[in_base + k + 1u];
        sum += unpack_bf16(p1, 0u) * input[in_base + k + 2u];
        sum += unpack_bf16(p1, 1u) * input[in_base + k + 3u];
        sum += unpack_bf16(p2, 0u) * input[in_base + k + 4u];
        sum += unpack_bf16(p2, 1u) * input[in_base + k + 5u];
        sum += unpack_bf16(p3, 0u) * input[in_base + k + 6u];
        sum += unpack_bf16(p3, 1u) * input[in_base + k + 7u];

        i += 4u;
    }

    // Remainder
    while (i < half_d_in) {
        let p = weight[w_base + i];
        let k = i * 2u;
        sum += unpack_bf16(p, 0u) * input[in_base + k];
        sum += unpack_bf16(p, 1u) * input[in_base + k + 1u];
        i += 1u;
    }

    // Add bias if present
    if (params.has_bias != 0u) {
        sum += bias[col];
    }

    output[row * params.d_out + col] = sum;
}
