// Fused SiLU-gated GPTQ INT4 matvec with 4-thread lane parallelism.
// output[col] = sum(SiLU(a[k]) * b[k] * dequant(W[k, col]))
// 4 threads per column, 8 columns per workgroup. Each lane handles 1/4 of packed_rows.
// INT4 quantized weights: 8 nibbles per u32. Scales stored as f16 pairs via unpack2x16float.
// Dispatch: (ceil(N / 8), 1, 1)

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> qweight: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 32>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let lane = tid & 3u;
    let col_local = tid >> 2u;
    let col = wg_id.x * 8u + col_local;

    let K = params.K;
    let N = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;

    var sum: f32 = 0.0;

    if (col < N) {
        // Each lane processes 1/4 of the packed rows
        let rows_per_lane = packed_rows / 4u;
        let pr_start = lane * rows_per_lane;
        let pr_end = select(pr_start + rows_per_lane, packed_rows, lane == 3u);

        var pr: u32 = pr_start;
        while (pr < pr_end) {
            let group = (pr * 8u) / group_size;
            let sf = group * N + col;
            let s2 = unpack2x16float(scales[sf / 2u]);
            let scale = select(s2.x, s2.y, (sf & 1u) == 1u);
            let packed = qweight[pr * N + col];

            let row_base = pr * 8u;
            sum += (f32((packed) & 0xFu) - 8.0) * scale * silu(a[row_base]) * b[row_base];
            sum += (f32((packed >> 4u) & 0xFu) - 8.0) * scale * silu(a[row_base + 1u]) * b[row_base + 1u];
            sum += (f32((packed >> 8u) & 0xFu) - 8.0) * scale * silu(a[row_base + 2u]) * b[row_base + 2u];
            sum += (f32((packed >> 12u) & 0xFu) - 8.0) * scale * silu(a[row_base + 3u]) * b[row_base + 3u];
            sum += (f32((packed >> 16u) & 0xFu) - 8.0) * scale * silu(a[row_base + 4u]) * b[row_base + 4u];
            sum += (f32((packed >> 20u) & 0xFu) - 8.0) * scale * silu(a[row_base + 5u]) * b[row_base + 5u];
            sum += (f32((packed >> 24u) & 0xFu) - 8.0) * scale * silu(a[row_base + 6u]) * b[row_base + 6u];
            sum += (f32((packed >> 28u) & 0xFu) - 8.0) * scale * silu(a[row_base + 7u]) * b[row_base + 7u];

            pr += 1u;
        }
    }

    // Write partial sum to scratch for reduction
    scratch[tid] = sum;
    workgroupBarrier();

    // Reduce 4 lanes per column: lane 0 accumulates lanes 1, 2, 3
    if (lane == 0u && col < N) {
        let base = col_local * 4u;
        let total = scratch[base] + scratch[base + 1u] + scratch[base + 2u] + scratch[base + 3u];
        output[col] = total;
    }
}
