// Dual-output GPTQ INT4 matvec: computes gate and up projections in one pass.
// Both projections share the same input (post-attention normed hidden state).
// One pass over the K input elements per output column reduces memory bandwidth.
// Dispatch: (ceil(inter / 32), 1, 1)

struct Params {
    K: u32,
    N: u32,       // intermediate_size (output columns for both gate and up)
    group_size: u32,
}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;  // [K] normed hidden
@group(0) @binding(1) var<storage, read>       gate_qweight: array<u32>;  // [K/8, N]
@group(0) @binding(2) var<storage, read>       gate_scales:  array<u32>;  // [K/group_size, N] f16
@group(0) @binding(3) var<storage, read>       up_qweight:   array<u32>;  // [K/8, N]
@group(0) @binding(4) var<storage, read>       up_scales:    array<u32>;  // [K/group_size, N] f16
@group(0) @binding(5) var<storage, read_write> gate_out:     array<f32>;  // [N]
@group(0) @binding(6) var<storage, read_write> up_out:       array<f32>;  // [N]
@group(0) @binding(7) var<uniform>             params:       Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;
    if (col >= params.N) { return; }

    let K          = params.K;
    let N          = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;

    var gate_sum: f32 = 0.0;
    var up_sum:   f32 = 0.0;

    // Unroll by 4 packed rows — accumulate both gate and up in one loop
    let unroll_end = packed_rows & ~3u;

    var pr: u32 = 0u;
    while (pr < unroll_end) {
        for (var j: u32 = 0u; j < 4u; j++) {
            let cur_pr    = pr + j;
            let group     = (cur_pr * 8u) / group_size;
            let sf        = group * N + col;
            let s2_g      = unpack2x16float(gate_scales[sf / 2u]);
            let s2_u      = unpack2x16float(up_scales[sf / 2u]);
            let gate_sc   = select(s2_g.x, s2_g.y, (sf & 1u) == 1u);
            let up_sc     = select(s2_u.x, s2_u.y, (sf & 1u) == 1u);
            let g_packed  = gate_qweight[cur_pr * N + col];
            let u_packed  = up_qweight[cur_pr * N + col];

            let row_base = cur_pr * 8u;
            for (var k: u32 = 0u; k < 8u; k++) {
                let shift = k * 4u;
                let inp   = input[row_base + k];
                gate_sum += (f32((g_packed >> shift) & 0xFu) - 8.0) * gate_sc * inp;
                up_sum   += (f32((u_packed >> shift) & 0xFu) - 8.0) * up_sc   * inp;
            }
        }
        pr += 4u;
    }

    while (pr < packed_rows) {
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

    gate_out[col] = gate_sum;
    up_out[col]   = up_sum;
}
