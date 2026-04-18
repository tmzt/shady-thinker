// GPTQ INT4 Matrix-Matrix Multiply with fused bias addition.
// output[pos, col] = sum(dequant(qweight) * input[pos]) + bias[col]
// For models without bias, pass a zero-filled buffer.
// Dispatch: (ceil(N / 8), S, 1)  where S = sequence length

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read>       input:   array<f32>;  // [S, K]
@group(0) @binding(1) var<storage, read>       qweight: array<u32>;  // [K/8, N]
@group(0) @binding(2) var<storage, read>       scales:  array<u32>;  // [K/group_size, N] packed f16
@group(0) @binding(3) var<storage, read_write> output:  array<f32>;  // [S, N]
@group(0) @binding(4) var<uniform>             params:  Params;
@group(0) @binding(5) var<storage, read>       bias:    array<f32>;  // [N]

var<workgroup> scratch: array<f32, 32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid       = lid.x;
    let lane      = tid & 3u;       // 0..3: which lane within the 4-thread group
    let col_local = tid >> 2u;      // 0..7: which column within this workgroup
    let col       = wg_id.x * 8u + col_local;
    let pos       = wg_id.y;        // sequence position

    if (col >= params.N) {
        scratch[tid] = 0.0;
        workgroupBarrier();
        workgroupBarrier();
        return;
    }

    let K          = params.K;
    let N          = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;
    let in_base    = pos * K;
    let out_base   = pos * N;

    // Divide packed_rows across 4 lanes
    let chunk     = packed_rows / 4u;
    let remainder = packed_rows % 4u;
    let lane_start = lane * chunk + min(lane, remainder);
    let lane_end   = lane_start + chunk + select(0u, 1u, lane < remainder);

    var sum: f32 = 0.0;

    var pr: u32 = lane_start;
    while (pr < lane_end) {
        let group = (pr * 8u) / group_size;
        let sf    = group * N + col;
        let s2    = unpack2x16float(scales[sf / 2u]);
        let scale = select(s2.x, s2.y, (sf & 1u) == 1u);
        let packed = qweight[pr * N + col];

        let row_base = in_base + pr * 8u;
        sum += (f32((packed)       & 0xFu) - 8.0) * scale * input[row_base];
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
        output[out_base + col] = scratch[tid] + scratch[tid + 1u] + bias[col];
    }
}
