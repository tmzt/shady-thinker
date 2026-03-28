// Batched SiLU-gated multiplication: output[i] = SiLU(gate[i]) * up[i]
// Works on flat arrays — seq_len × intermediate_size elements.
// Dispatch: (ceil(N / 256), 1, 1) where N = seq_len * intermediate_size

struct Params {
    N: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.N) { return; }
    let x = gate[i];
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    output[i] = x * sigmoid_x * up[i];
}
