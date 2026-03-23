// Sigmoid-gated multiplication: output[i] = x[i] * sigmoid(gate[i])
// sigmoid(x) = 1 / (1 + exp(-x))
// Dispatch: (ceil(N / 256), 1, 1)

struct Params {
    N: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.N) {
        return;
    }
    let g = gate[i];
    let sig = 1.0 / (1.0 + exp(-g));
    output[i] = x[i] * sig;
}
