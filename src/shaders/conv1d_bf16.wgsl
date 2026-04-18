// Conv1D (no activation) for Whisper-style audio encoder conv stem.
//
// Same as conv1d_gelu_bf16 but without GELU activation.
// Used for conv2 which has no activation in Qwen2.5-Omni.
//
// weight: [C_out, C_in, kernel_size] packed as bf16 u32
// bias:   [C_out] packed as bf16 u32
// input:  [C_in, seq_len] f32
// output: [C_out, out_len] f32

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<u32>;
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

    let oc = idx / out_len;
    let ot = idx % out_len;

    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < c_in; ic++) {
        for (var k: u32 = 0u; k < kernel_sz; k++) {
            let it = i32(ot * stride + k) - i32(pad);
            if it >= 0 && u32(it) < seq_len {
                let in_val = input[ic * seq_len + u32(it)];
                let w_idx = (oc * c_in + ic) * kernel_sz + k;
                let w_val = unpack_bf16(weight[w_idx / 2u], w_idx);
                sum += in_val * w_val;
            }
        }
    }

    let b_val = unpack_bf16(bias[oc / 2u], oc);
    output[idx] = sum + b_val;
}
