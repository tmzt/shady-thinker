// Add + RMSNorm: hidden[i] += addend[i], then output[i] = hidden[i] * rms * (1 + weight[i])
// Weight is BF16 packed (two BF16 values per u32), using (1 + w) scaling.
// Dispatch: (1, 1, 1) — single workgroup

struct Params {
    N: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared: array<f32, 256>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let N = params.N;

    // Phase 1: hidden[i] += addend[i], accumulate sum of squares.
    var sum_sq: f32 = 0.0;
    var i = tid;
    while (i < N) {
        let v = hidden[i] + addend[i];
        hidden[i] = v;
        sum_sq += v * v;
        i += 256u;
    }
    shared[tid] = sum_sq;
    workgroupBarrier();

    // Phase 2: Tree reduce sum of squares.
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared[tid] = shared[tid] + shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Phase 3: Compute rms and write normalized output.
    let rms = 1.0 / sqrt(shared[0] / f32(N) + params.eps);

    i = tid;
    while (i < N) {
        let w = unpack_bf16(weight[i / 2u], i % 2u);
        output[i] = hidden[i] * rms * (1.0 + w);
        i += 256u;
    }
}
