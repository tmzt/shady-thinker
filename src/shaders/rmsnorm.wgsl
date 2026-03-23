// RMSNorm: output[i] = input[i] * (1/sqrt(mean(input^2) + eps)) * (1 + weight[i])
// Weight is BF16 packed (two BF16 values per u32), using (1 + w) scaling.
// Dispatch: (1, 1, 1) — single workgroup

struct Params {
    N: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wg_temp: array<f32, 256>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params.N;

    // Step 1: Each thread accumulates sum of input[i]^2 for strided indices.
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < N) {
        let v = input[i];
        sum_sq += v * v;
        i += 256u;
    }
    wg_temp[tid] = sum_sq;
    workgroupBarrier();

    // Step 2: Tree reduce for sum of squares.
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_temp[tid] = wg_temp[tid] + wg_temp[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Step 3: Compute rms scale factor and write output.
    let rms = 1.0 / sqrt(wg_temp[0] / f32(N) + params.eps);

    i = tid;
    while (i < N) {
        let w = unpack_bf16(weight[i / 2u], i % 2u);
        output[i] = input[i] * rms * (1.0 + w);
        i += 256u;
    }
}
