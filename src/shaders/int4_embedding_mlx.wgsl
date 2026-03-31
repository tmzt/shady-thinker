// INT4 MLX embedding lookup: dequantize one token's embedding.
// qweight[token_id, packed_col] → output[i] = nibble * scale + bias
// Dispatch: (ceil(dim / 256), 1, 1)

struct Params {
    token_id: u32,
    dim: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> qweight: array<u32>;   // [vocab, dim/8]
@group(0) @binding(1) var<storage, read> scales: array<u32>;    // [vocab, n_groups/2] packed f16
@group(0) @binding(2) var<storage, read> biases: array<u32>;    // [vocab, n_groups/2] packed f16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) { return; }

    let dim = params.dim;
    let packed_cols = dim / 8u;
    let group_size = params.group_size;
    let n_groups = dim / group_size;
    let tok = params.token_id;

    // Which packed u32 and nibble within it
    let packed_idx = i / 8u;
    let nibble_idx = i % 8u;
    let packed = qweight[tok * packed_cols + packed_idx];
    let nibble = (packed >> (nibble_idx * 4u)) & 0xFu;

    // Scale and bias for this group
    let group = i / group_size;
    let sb_idx = tok * n_groups + group;
    let scale = unpack2x16float(scales[sb_idx / 2u])[sb_idx & 1u];
    let bias = unpack2x16float(biases[sb_idx / 2u])[sb_idx & 1u];

    output[i] = f32(nibble) * scale + bias;
}
