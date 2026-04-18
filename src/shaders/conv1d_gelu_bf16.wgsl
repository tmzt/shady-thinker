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
struct ConvParams {
    c_in: u32, c_out: u32, seq_len: u32, kernel_size: u32,
    stride: u32, pad: u32, out_len: u32, _pad: u32,
}
@group(0) @binding(4) var<uniform> params: ConvParams;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let half = select(packed & 0xFFFFu, packed >> 16u, (idx & 1u) == 1u);
    return bf16_to_f32(half);
}

fn gelu(x: f32) -> f32 {
    // Fast GELU approximation
    let k = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(k));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c_in     = params.c_in;
    let c_out    = params.c_out;
    let seq_len  = params.seq_len;
    let kernel_sz = params.kernel_size;
    let stride   = params.stride;
    let pad      = params.pad;
    let out_len  = params.out_len;

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
