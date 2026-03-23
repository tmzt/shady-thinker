// BF16 embedding lookup: output[i] = unpack_bf16(embeddings, token_id * dim + i)
// BF16 values are packed two per u32.
// Dispatch: (ceil(dim / 256), 1, 1)

struct Params {
    token_id: u32,
    dim: u32,
}

@group(0) @binding(0) var<storage, read> embeddings: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dim) {
        return;
    }
    let flat_idx = params.token_id * params.dim + i;
    let packed_idx = flat_idx / 2u;
    let sub_idx = flat_idx % 2u;
    output[i] = unpack_bf16(embeddings[packed_idx], sub_idx);
}
