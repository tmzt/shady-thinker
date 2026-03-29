// INT8 MLX embedding lookup: dequantize one token's embedding.
// qweight[token_id, packed_col] → output[i] = byte * scale + bias
// Dispatch: (ceil(dim / 256), 1, 1)

struct Params {
    token_id: u32,          // local token ID within chunk
    dim: u32,
    group_size: u32,
    token_id_global: u32,   // global token ID (for scales/biases)
}

@group(0) @binding(0) var<storage, read> qweight: array<u32>;   // [vocab, dim/4]
@group(0) @binding(1) var<storage, read> scales: array<u32>;    // [vocab, n_groups/2] packed bf16
@group(0) @binding(2) var<storage, read> biases: array<u32>;    // [vocab, n_groups/2] packed bf16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) { return; }

    let dim = params.dim;
    let packed_cols = dim / 4u;
    let group_size = params.group_size;
    let n_groups = dim / group_size;
    let tok = params.token_id;           // local
    let tok_g = params.token_id_global;  // global

    let packed_idx = i / 4u;
    let byte_idx = i % 4u;
    let packed = qweight[tok * packed_cols + packed_idx];
    let byte_val = (packed >> (byte_idx * 8u)) & 0xFFu;

    let group = i / group_size;
    let sb_idx = tok_g * n_groups + group;
    let scale = bitcast<f32>(((scales[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);
    let bias = bitcast<f32>(((biases[sb_idx / 2u] >> ((sb_idx & 1u) * 16u)) & 0xFFFFu) << 16u);

    output[i] = f32(byte_val) * scale + bias;
}
