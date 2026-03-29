// Batched SiLU-gated multiplication with 2D dispatch over [seq_len, inter_dim].
// Computes: output[row, col] = SiLU(gate[row, col]) * up[row, col]
// where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//
// This variant uses a 2D dispatch for explicit control over the token and
// intermediate dimension axes, unlike batched_silu_mul.wgsl which flattens
// everything into a 1D dispatch.
//
// All arrays layout: [seq_len, inter_dim] contiguous f32.
//
// Params uniform:
//   inter_dim — intermediate/FFN dimension
//   seq_len   — number of tokens in the batch
//
// Bindings:
//   @binding(0) gate   — [seq_len, inter_dim] f32, read-only
//   @binding(1) up     — [seq_len, inter_dim] f32, read-only
//   @binding(2) output — [seq_len, inter_dim] f32, read-write
//   @binding(3) params — uniform Params
//
// Dispatch: (ceil(inter_dim / 256), seq_len, 1)
//   wg_id.x * 256 + lid.x = column index (inter_dim axis)
//   wg_id.y = row index (token/seq_len axis)

struct Params {
    inter_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x * 256u + lid.x;
    let row = wg_id.y;

    if (col >= params.inter_dim || row >= params.seq_len) { return; }

    let idx = row * params.inter_dim + col;
    let x = gate[idx];
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    output[idx] = x * sigmoid_x * up[idx];
}
