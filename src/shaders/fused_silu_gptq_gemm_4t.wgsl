// Fused SiLU-gated GPTQ INT4 GEMM with 4-thread lane parallelism.
// Extends fused_silu_gptq_4t with a sequence (batch) dimension in workgroup_id.y.
// 8 columns per workgroup (32 threads / 4 lanes per column).
// Dispatch: (ceil(N / 8), S, 1)

const GPTQ_ZP: f32 = 8.0;

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read>       a:       array<f32>;  // gate [S, K]
@group(0) @binding(1) var<storage, read>       b:       array<f32>;  // up   [S, K]
@group(0) @binding(2) var<storage, read>       qweight: array<u32>;  // down [K/8, N]
@group(0) @binding(3) var<storage, read>       scales:  array<u32>;  // [K/group_size, N] packed f16
@group(0) @binding(4) var<storage, read_write> output:  array<f32>;  // [S, N]
@group(0) @binding(5) var<uniform>             params:  Params;

var<workgroup> scratch: array<f32, 32>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid       = lid.x;
    let lane      = tid & 3u;
    let col_local = tid >> 2u;
    let col       = wg_id.x * 8u + col_local;
    let pos       = wg_id.y;

    let K          = params.K;
    let N          = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;
    let in_base    = pos * K;
    let out_base   = pos * N;

    var sum: f32 = 0.0;

    if (col < N) {
        let rows_per_lane = packed_rows / 4u;
        let pr_start = lane * rows_per_lane;
        let pr_end   = select(pr_start + rows_per_lane, packed_rows, lane == 3u);

        var pr: u32 = pr_start;
        while (pr < pr_end) {
            let group = (pr * 8u) / group_size;
            let sf    = group * N + col;
            let s2    = unpack2x16float(scales[sf / 2u]);
            let scale = select(s2.x, s2.y, (sf & 1u) == 1u);
            let packed = qweight[pr * N + col];

            let row_base = in_base + pr * 8u;
            sum += (f32((packed)       & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base])     * b[row_base];
            sum += (f32((packed >> 4u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 1u]) * b[row_base + 1u];
            sum += (f32((packed >> 8u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 2u]) * b[row_base + 2u];
            sum += (f32((packed >> 12u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 3u]) * b[row_base + 3u];
            sum += (f32((packed >> 16u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 4u]) * b[row_base + 4u];
            sum += (f32((packed >> 20u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 5u]) * b[row_base + 5u];
            sum += (f32((packed >> 24u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 6u]) * b[row_base + 6u];
            sum += (f32((packed >> 28u) & 0xFu) - GPTQ_ZP) * scale * silu(a[row_base + 7u]) * b[row_base + 7u];

            pr += 1u;
        }
    }

    scratch[tid] = sum;
    workgroupBarrier();

    if (lane == 0u && col < N) {
        let base  = col_local * 4u;
        let total = scratch[base] + scratch[base + 1u] + scratch[base + 2u] + scratch[base + 3u];
        output[out_base + col] = total;
    }
}
