// F32 Batched GEMM: output[row, col] = sum_k(input[row, k] * weight[col, k]) + bias[col]
// Computes: [seq_len, d_in] × [d_out, d_in]^T + [d_out] → [seq_len, d_out]
// All buffers are f32. Weight layout: row-major [d_out, d_in].
//
// Dispatch: (ceil(d_out / 32), seq_len, 1)

struct Params {
    d_in: u32,
    d_out: u32,
    seq_len: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;
    let row = wg_id.y;

    if (col >= params.d_out || row >= params.seq_len) {
        return;
    }

    let d_in = params.d_in;
    let w_base = col * d_in;
    let in_base = row * d_in;

    var sum: f32 = 0.0;

    // Unroll by 4
    let unroll_end = d_in & ~3u;
    var i: u32 = 0u;
    while (i < unroll_end) {
        sum += weight[w_base + i] * input[in_base + i];
        sum += weight[w_base + i + 1u] * input[in_base + i + 1u];
        sum += weight[w_base + i + 2u] * input[in_base + i + 2u];
        sum += weight[w_base + i + 3u] * input[in_base + i + 3u];
        i += 4u;
    }
    while (i < d_in) {
        sum += weight[w_base + i] * input[in_base + i];
        i += 1u;
    }

    if (params.has_bias != 0u) {
        sum += bias[col];
    }

    output[row * params.d_out + col] = sum;
}
