// Standard GELU activation: output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Matches torch.nn.functional.gelu (tanh approximation).
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
    let k = 0.7978845608 * (x + 0.044715 * x * x * x);
    output[i] = 0.5 * x * (1.0 + tanh(k));
}
