// GPTQ INT4 Matrix-Vector Multiply with 4-thread lane parallelism.
// Same math as gptq_matvec but each output column is processed by 4 cooperative threads.
// 8 columns per workgroup (32 threads / 4 lanes per column).
// Dispatch: (ceil(N / 8), 1, 1)

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid & 3u;       // 0..3: which lane within the 4-thread group
    let col_local = tid >> 2u; // 0..7: which column within this workgroup
    let col = wg_id.x * 8u + col_local;

    if (col >= params.N) {
        scratch[tid] = 0.0;
        workgroupBarrier();
        workgroupBarrier();
        return;
    }

    let K = params.K;
    let N = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;

    // Divide packed_rows across 4 lanes
    let chunk = packed_rows / 4u;
    let remainder = packed_rows % 4u;
    // Distribute remainder to lower lanes
    let lane_start = lane * chunk + min(lane, remainder);
    let lane_end = lane_start + chunk + select(0u, 1u, lane < remainder);

    var sum: f32 = 0.0;

    var pr: u32 = lane_start;
    while (pr < lane_end) {
        let group = (pr * 8u) / group_size;
        let sf = group * N + col;
        let s2 = unpack2x16float(scales[sf / 2u]);
        let scale = select(s2.x, s2.y, (sf & 1u) == 1u);
        let packed = qweight[pr * N + col];

        let row_base = pr * 8u;
        sum += (f32((packed) & 0xFu) - 8.0) * scale * input[row_base];
        sum += (f32((packed >> 4u) & 0xFu) - 8.0) * scale * input[row_base + 1u];
        sum += (f32((packed >> 8u) & 0xFu) - 8.0) * scale * input[row_base + 2u];
        sum += (f32((packed >> 12u) & 0xFu) - 8.0) * scale * input[row_base + 3u];
        sum += (f32((packed >> 16u) & 0xFu) - 8.0) * scale * input[row_base + 4u];
        sum += (f32((packed >> 20u) & 0xFu) - 8.0) * scale * input[row_base + 5u];
        sum += (f32((packed >> 24u) & 0xFu) - 8.0) * scale * input[row_base + 6u];
        sum += (f32((packed >> 28u) & 0xFu) - 8.0) * scale * input[row_base + 7u];

        pr += 1u;
    }

    // Store partial sum and reduce across 4 lanes
    scratch[tid] = sum;
    workgroupBarrier();

    // Step 1: lanes 0,1 accumulate from lanes 2,3
    if (lane < 2u) {
        scratch[tid] = scratch[tid] + scratch[tid + 2u];
    }
    workgroupBarrier();

    // Step 2: lane 0 accumulates from lane 1 and writes output
    if (lane == 0u) {
        output[col] = scratch[tid] + scratch[tid + 1u];
    }
}
