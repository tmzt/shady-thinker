// Batched Add + RMSNorm: hidden[row,i] += addend[row,i], then
// output[row,i] = hidden[row,i] * rms * weight[i]
// where rms = 1 / sqrt(mean(x^2) + eps).
//
// The residual add is fused: hidden is updated in-place before normalization.
// Weight is BF16 packed (two BF16 values per u32).
// All arrays layout: [seq_len, N] contiguous f32.
//
// Params uniform:
//   N        — hidden dimension (e.g. 1024)
//   eps      — normalization epsilon (e.g. 1e-6)
//   seq_len  — number of tokens in the batch
//
// Bindings:
//   @binding(0) hidden  — [seq_len, N] f32, read-write (residual, updated in-place)
//   @binding(1) addend  — [seq_len, N] f32, read-only (value to add)
//   @binding(2) weight  — [N/2] u32 (BF16 packed), read-only
//   @binding(3) output  — [seq_len, N] f32, read-write (normalized result)
//   @binding(4) params  — uniform Params
//
// Dispatch: (seq_len, 1, 1) — one workgroup per token
// Workgroup size: 256 threads stride over N

struct Params {
    N: u32,
    eps: f32,
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> wg_temp: array<f32, 256>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row = wid.x;
    if (row >= params.seq_len) { return; }
    let N = params.N;
    let base = row * N;

    // Phase 1: hidden += addend, accumulate sum of squares
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < N) {
        let v = hidden[base + i] + addend[base + i];
        hidden[base + i] = v;
        sum_sq += v * v;
        i += 256u;
    }
    wg_temp[tid] = sum_sq;
    workgroupBarrier();

    // Tree reduce
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_temp[tid] = wg_temp[tid] + wg_temp[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let rms = 1.0 / sqrt(wg_temp[0] / f32(N) + params.eps);

    // Write normalized output
    i = tid;
    while (i < N) {
        let w = unpack_bf16(weight[i / 2u], i % 2u);
        output[base + i] = hidden[base + i] * rms * w;
        i += 256u;
    }
}
