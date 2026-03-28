// F32 Batched GEMM with Kahan compensated summation for precision.
// output[row, col] = sum_k(input[row, k] * weight[col, k]) + bias[col]
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

    // Kahan compensated summation — tracks rounding error
    var sum: f32 = 0.0;
    var comp: f32 = 0.0; // compensation for lost low-order bits

    var i: u32 = 0u;
    while (i < d_in) {
        let product = weight[w_base + i] * input[in_base + i];
        let y = product - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        i += 1u;
    }

    if (params.has_bias != 0u) {
        let y = bias[col] - comp;
        let t = sum + y;
        sum = t;
    }

    output[row * params.d_out + col] = sum;
}
