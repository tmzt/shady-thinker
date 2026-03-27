// GELU activation: output[i] = x[i] * sigmoid(1.702 * x[i])
// Approximation of Gaussian Error Linear Unit.
// Dispatch: (ceil(N / 256), 1, 1)

struct Params {
    N: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.N) {
        return;
    }
    let x = input[i];
    let sigmoid_x = 1.0 / (1.0 + exp(-1.702 * x));
    output[i] = x * sigmoid_x;
}
