// LayerNorm: output[i] = (input[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]
// Weight and bias are f32 (ASR encoder uses f32 norms, not bf16).
// Dispatch: (1, 1, 1) — single workgroup per token
// For batched: dispatch (seq_len, 1, 1) with token offset = workgroup_id.x * N

struct Params {
    N: u32,       // hidden dimension
    eps: f32,
    seq_len: u32, // number of tokens (rows)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> wg_temp: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row = wid.x;
    if (row >= params.seq_len) {
        return;
    }
    let N = params.N;
    let base = row * N;

    // Step 1: Compute mean via tree reduce.
    var sum: f32 = 0.0;
    var i = tid;
    while (i < N) {
        sum += input[base + i];
        i += 256u;
    }
    wg_temp[tid] = sum;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_temp[tid] = wg_temp[tid] + wg_temp[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mean = wg_temp[0] / f32(N);
    workgroupBarrier();

    // Step 2: Compute variance via tree reduce.
    var sum_sq: f32 = 0.0;
    i = tid;
    while (i < N) {
        let diff = input[base + i] - mean;
        sum_sq += diff * diff;
        i += 256u;
    }
    wg_temp[tid] = sum_sq;
    workgroupBarrier();

    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_temp[tid] = wg_temp[tid] + wg_temp[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let inv_std = 1.0 / sqrt(wg_temp[0] / f32(N) + params.eps);

    // Step 3: Normalize, scale, shift.
    i = tid;
    while (i < N) {
        let idx = base + i;
        output[idx] = (input[idx] - mean) * inv_std * weight[i] + bias[i];
        i += 256u;
    }
}
