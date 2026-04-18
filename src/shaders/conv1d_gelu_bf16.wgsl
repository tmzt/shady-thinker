// Conv1D + GELU activation for Whisper-style audio encoder conv stem.
//
// Weights are BF16 packed as u32 (2 values per u32).
// Input/output are f32.
//
// weight: [C_out, C_in, kernel_size] packed as bf16 u32
// bias:   [C_out] packed as bf16 u32
// input:  [C_in, seq_len] f32
// output: [C_out, out_len] f32
//
// params: { c_in, c_out, seq_len, kernel_size, stride, pad, out_len, _pad }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // bf16 packed
@group(0) @binding(2) var<storage, read> bias: array<u32>;    // bf16 packed
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: array<u32, 8>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let half = select(packed & 0xFFFFu, packed >> 16u, idx & 1u == 1u);
    return bf16_to_f32(half);
}

fn gelu(x: f32) -> f32 {
    // Fast GELU approximation
    let k = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(k));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c_in     = params[0];
    let c_out    = params[1];
    let seq_len  = params[2];
    let kernel_sz = params[3];
    let stride   = params[4];
    let pad      = params[5];
    let out_len  = params[6];

    let idx = gid.x;
    if idx >= c_out * out_len { return; }

    let oc = idx / out_len;  // output channel
    let ot = idx % out_len;  // output time position

    var sum: f32 = 0.0;

    // Conv1D: sum over input channels and kernel positions
    for (var ic: u32 = 0u; ic < c_in; ic++) {
        for (var k: u32 = 0u; k < kernel_sz; k++) {
            let it = i32(ot * stride + k) - i32(pad);
            if it >= 0 && u32(it) < seq_len {
                let in_val = input[ic * seq_len + u32(it)];
                // weight layout: [c_out, c_in, kernel_size] as bf16
                let w_idx = (oc * c_in + ic) * kernel_sz + k;
                let w_val = unpack_bf16(weight[w_idx / 2u], w_idx);
                sum += in_val * w_val;
            }
        }
    }

    // Add bias (bf16)
    let b_val = unpack_bf16(bias[oc / 2u], oc);
    sum += b_val;

    // GELU activation
    output[idx] = gelu(sum);
}
