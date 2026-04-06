// Dual-output GPTQ INT4 matvec with 4-thread lane parallelism.
// Computes gate and up projections in one pass; both share the same input.
// 8 columns per workgroup (32 threads / 4 lanes per column).
// Dispatch: (ceil(inter / 8), 1, 1)

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;  // [K]
@group(0) @binding(1) var<storage, read>       gate_qweight: array<u32>;
@group(0) @binding(2) var<storage, read>       gate_scales:  array<u32>;
@group(0) @binding(3) var<storage, read>       up_qweight:   array<u32>;
@group(0) @binding(4) var<storage, read>       up_scales:    array<u32>;
@group(0) @binding(5) var<storage, read_write> gate_out:     array<f32>;
@group(0) @binding(6) var<storage, read_write> up_out:       array<f32>;
@group(0) @binding(7) var<uniform>             params:       Params;

// Two scratch arrays: first 32 for gate partials, second 32 for up partials
var<workgroup> scratch_g: array<f32, 32>;
var<workgroup> scratch_u: array<f32, 32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid       = lid.x;
    let lane      = tid & 3u;
    let col_local = tid >> 2u;
    let col       = wg_id.x * 8u + col_local;

    if (col >= params.N) {
        scratch_g[tid] = 0.0;
        scratch_u[tid] = 0.0;
        workgroupBarrier();
        workgroupBarrier();
        return;
    }

    let K          = params.K;
    let N          = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;

    let chunk     = packed_rows / 4u;
    let remainder = packed_rows % 4u;
    let lane_start = lane * chunk + min(lane, remainder);
    let lane_end   = lane_start + chunk + select(0u, 1u, lane < remainder);

    var gate_sum: f32 = 0.0;
    var up_sum:   f32 = 0.0;

    var pr: u32 = lane_start;
    while (pr < lane_end) {
        let group    = (pr * 8u) / group_size;
        let sf       = group * N + col;
        let s2_g     = unpack2x16float(gate_scales[sf / 2u]);
        let s2_u     = unpack2x16float(up_scales[sf / 2u]);
        let gate_sc  = select(s2_g.x, s2_g.y, (sf & 1u) == 1u);
        let up_sc    = select(s2_u.x, s2_u.y, (sf & 1u) == 1u);
        let g_packed = gate_qweight[pr * N + col];
        let u_packed = up_qweight[pr * N + col];

        let row_base = pr * 8u;
        for (var k: u32 = 0u; k < 8u; k++) {
            let shift = k * 4u;
            let inp   = input[row_base + k];
            gate_sum += (f32((g_packed >> shift) & 0xFu) - 8.0) * gate_sc * inp;
            up_sum   += (f32((u_packed >> shift) & 0xFu) - 8.0) * up_sc   * inp;
        }
        pr += 1u;
    }

    scratch_g[tid] = gate_sum;
    scratch_u[tid] = up_sum;
    workgroupBarrier();

    if (lane < 2u) {
        scratch_g[tid] = scratch_g[tid] + scratch_g[tid + 2u];
        scratch_u[tid] = scratch_u[tid] + scratch_u[tid + 2u];
    }
    workgroupBarrier();

    if (lane == 0u) {
        gate_out[col] = scratch_g[tid] + scratch_g[tid + 1u];
        up_out[col]   = scratch_u[tid] + scratch_u[tid + 1u];
    }
}
