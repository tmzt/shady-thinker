// Fused gate + up GPTQ matvec with SiLU activation and 4-thread lane parallelism.
// output[col] = SiLU(gate_sum) * up_sum
// 4 threads per column, 8 columns per workgroup. Each lane handles 1/4 of packed_rows.
// scratch[64]: indices 0..31 for gate partial sums, 32..63 for up partial sums.
// INT4 quantized weights: 8 nibbles per u32. Scales stored as f16 pairs via unpack2x16float.
// Dispatch: (ceil(N / 8), 1, 1)

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight_gate: array<u32>;
@group(0) @binding(2) var<storage, read> scales_gate: array<u32>;
@group(0) @binding(3) var<storage, read> qweight_up: array<u32>;
@group(0) @binding(4) var<storage, read> scales_up: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 64>;

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

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    if (col < N) {
        // Each lane processes 1/4 of the packed rows
        let rows_per_lane = packed_rows / 4u;
        let pr_start = lane * rows_per_lane;
        let pr_end = select(pr_start + rows_per_lane, packed_rows, lane == 3u);

        var pr: u32 = pr_start;
        while (pr < pr_end) {
            let group = (pr * 8u) / group_size;
            let sf = group * N + col;

            let sg2 = unpack2x16float(scales_gate[sf / 2u]);
            let scale_g = select(sg2.x, sg2.y, (sf & 1u) == 1u);
            let packed_g = qweight_gate[pr * N + col];

            let su2 = unpack2x16float(scales_up[sf / 2u]);
            let scale_u = select(su2.x, su2.y, (sf & 1u) == 1u);
            let packed_u = qweight_up[pr * N + col];

            let row_base = pr * 8u;
            for (var n: u32 = 0u; n < 8u; n++) {
                let shift = n * 4u;
                let dq_g = (f32((packed_g >> shift) & 0xFu) - 8.0) * scale_g;
                let dq_u = (f32((packed_u >> shift) & 0xFu) - 8.0) * scale_u;
                let inp = input[row_base + n];
                gate_sum += dq_g * inp;
                up_sum += dq_u * inp;
            }

            pr += 1u;
        }
    }

    // Write partial sums to scratch: gate in [0..31], up in [32..63]
    scratch[tid] = gate_sum;
    scratch[32u + tid] = up_sum;
    workgroupBarrier();

    // Reduce 4 lanes per column: lane 0 accumulates lanes 1, 2, 3
    if (lane == 0u && col < N) {
        let base = col_local * 4u;
        let gate_total = scratch[base] + scratch[base + 1u] + scratch[base + 2u] + scratch[base + 3u];
        let up_total = scratch[32u + base] + scratch[32u + base + 1u] + scratch[32u + base + 2u] + scratch[32u + base + 3u];
        output[col] = silu(gate_total) * up_total;
    }
}
