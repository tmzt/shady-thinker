// INT4 Asymmetric (MLX minmax) Matrix-Vector Multiply.
// output[row] = sum_k( dequant(qweight[row, k]) * input[k] )
// where dequant(nibble) = nibble * scales[row, group] + biases[row, group]
//
// Row-major layout: qweight[row, packed_col], 8 nibbles per u32.
// Scales/biases: [out_dim, n_groups] as packed f16 pairs.
//
// Dispatch: (ceil(out_dim / 32), 1, 1)

struct Params {
    in_dim: u32,       // K: input dimension
    out_dim: u32,      // N: output dimension (rows of weight)
    group_size: u32,   // typically 64
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;   // [out_dim, in_dim/8]
@group(0) @binding(2) var<storage, read> scales: array<u32>;    // [out_dim, n_groups/2] packed f16
@group(0) @binding(3) var<storage, read> biases: array<u32>;    // [out_dim, n_groups/2] packed f16
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x * 32u + lid.x;
    if (row >= params.out_dim) { return; }

    let in_dim = params.in_dim;
    let packed_cols = in_dim / 8u;
    let group_size = params.group_size;
    let n_groups = in_dim / group_size;
    let packed_per_group = group_size / 8u;  // 64/8 = 8

    // Row offsets
    let w_row_off = row * packed_cols;
    let sg_row_off = row * n_groups;  // scales/biases row offset (in f16 elements)

    var sum: f32 = 0.0;

    for (var g: u32 = 0u; g < n_groups; g++) {
        // Load scale and bias for this group
        let sb_idx = sg_row_off + g;
        let s_packed = scales[sb_idx / 2u];
        let scale = unpack2x16float(s_packed)[sb_idx & 1u];
        let b_packed = biases[sb_idx / 2u];
        let bias = unpack2x16float(b_packed)[sb_idx & 1u];

        let group_start = g * packed_per_group;
        let input_base = g * group_size;

        for (var p: u32 = 0u; p < packed_per_group; p++) {
            let packed = qweight[w_row_off + group_start + p];
            let ib = input_base + p * 8u;

            sum += (f32((packed) & 0xFu) * scale + bias) * input[ib];
            sum += (f32((packed >> 4u) & 0xFu) * scale + bias) * input[ib + 1u];
            sum += (f32((packed >> 8u) & 0xFu) * scale + bias) * input[ib + 2u];
            sum += (f32((packed >> 12u) & 0xFu) * scale + bias) * input[ib + 3u];
            sum += (f32((packed >> 16u) & 0xFu) * scale + bias) * input[ib + 4u];
            sum += (f32((packed >> 20u) & 0xFu) * scale + bias) * input[ib + 5u];
            sum += (f32((packed >> 24u) & 0xFu) * scale + bias) * input[ib + 6u];
            sum += (f32((packed >> 28u) & 0xFu) * scale + bias) * input[ib + 7u];
        }
    }

    output[row] = sum;
}
