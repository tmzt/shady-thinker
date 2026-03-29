// Batched INT8 quantized GEMM (MLX asymmetric: value = byte * scale + bias).
// Computes: input[seq_len, in_dim] × qweight[out_dim, in_dim/4] → output[seq_len, out_dim]
//
// Weights: INT8 packed (4 bytes per u32), row-major [out_dim, in_dim/4].
// Scales/biases: BF16 packed (2 per u32), [out_dim, n_groups].
// Input/output: f32.
//
// Dispatch: (ceil(out_dim / 32), seq_len, 1)
//   wg_id.x * 32 + lid.x = output column (weight row)
//   wg_id.y = token index (batch row)
//
// General-purpose — usable for any LLM projection (Q, K, V, O, gate, up, down).

struct Params {
    in_dim: u32,
    out_dim: u32,
    seq_len: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> biases: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;  // output column (weight row)
    let row = wg_id.y;                  // token index

    if (col >= params.out_dim || row >= params.seq_len) { return; }

    let packed_cols = params.in_dim / 4u;
    let n_groups = params.in_dim / params.group_size;
    let packed_per_group = params.group_size / 4u;

    let w_row_off = col * packed_cols;
    let sg_row_off = col * n_groups;
    let in_base = row * params.in_dim;

    var sum: f32 = 0.0;

    for (var g: u32 = 0u; g < n_groups; g++) {
        let sb_idx = sg_row_off + g;
        let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
        let group_start = g * packed_per_group;
        let input_base = in_base + g * params.group_size;

        for (var p: u32 = 0u; p < packed_per_group; p++) {
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 4u;
            sum += (f32((packed) & 0xFFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 8u) & 0xFFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 16u) & 0xFFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 24u) & 0xFFu) * scale + bias) * input[ib + 3u];
        }
    }

    output[row * params.out_dim + col] = sum;
}
