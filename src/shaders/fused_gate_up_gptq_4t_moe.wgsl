// Dual-output GPTQ INT4 matvec for MoE — reads expert offset from GPU buffer.
// Same algorithm as fused_gate_up_gptq_4t but with packed expert buffers.
// The expert_id is read from moe_selected[expert_idx * 2] (no CPU readback).
//
// Dispatch: (ceil(inter / 8), 1, 1)

const GPTQ_ZP: f32 = 8.0;

struct Params {
    K: u32,            // input dimension (hidden_size)
    N: u32,            // output dimension (inter_size, before packing)
    group_size: u32,
    expert_idx: u32,   // which of the K selected experts (0..K-1)
    expert_stride_qw: u32, // u32 elements per expert in qweight buffer
    expert_stride_sc: u32, // u32 elements per expert in scales buffer
}

@group(0) @binding(0) var<storage, read>       input:        array<f32>;  // [K]
@group(0) @binding(1) var<storage, read>       gate_qweight: array<u32>;  // packed [num_experts * stride]
@group(0) @binding(2) var<storage, read>       gate_scales:  array<u32>;  // packed
@group(0) @binding(3) var<storage, read>       up_qweight:   array<u32>;  // packed
@group(0) @binding(4) var<storage, read>       up_scales:    array<u32>;  // packed
@group(0) @binding(5) var<storage, read_write> gate_out:     array<f32>;
@group(0) @binding(6) var<storage, read_write> up_out:       array<f32>;
@group(0) @binding(7) var<uniform>             params:       Params;
@group(0) @binding(8) var<storage, read>       moe_selected: array<u32>;  // [K*2]: expert_id, weight

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

    // Read expert_id from moe_selected buffer (GPU-side, no CPU readback)
    let expert_id = moe_selected[params.expert_idx * 2u];
    let gate_base = expert_id * params.expert_stride_qw;
    let gate_sc_base = expert_id * params.expert_stride_sc;
    let up_base = expert_id * params.expert_stride_qw;
    let up_sc_base = expert_id * params.expert_stride_sc;

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
    let r_start   = lane * chunk;
    let r_end     = select(r_start + chunk, packed_rows, lane == 3u);

    var sum_g: f32 = 0.0;
    var sum_u: f32 = 0.0;

    for (var pr = r_start; pr < r_end; pr++) {
        let packed_g = gate_qweight[gate_base + pr * N + col];
        let packed_u = up_qweight[up_base + pr * N + col];

        let group = pr * 8u / group_size;
        let sc_word_g = gate_scales[gate_sc_base + (group * N + col) / 2u];
        let sc_word_u = up_scales[up_sc_base + (group * N + col) / 2u];
        let sc_shift = select(0u, 16u, ((group * N + col) & 1u) != 0u);
        let sc_g = unpack2x16float((sc_word_g >> sc_shift) & 0xFFFFu).x;
        let sc_u = unpack2x16float((sc_word_u >> sc_shift) & 0xFFFFu).x;

        for (var nib = 0u; nib < 8u; nib++) {
            let r = pr * 8u + nib;
            let x = input[r];
            sum_g += dequant_val(packed_g, nib, sc_g) * x;
            sum_u += dequant_val(packed_u, nib, sc_u) * x;
        }
    }

    scratch_g[tid] = sum_g;
    scratch_u[tid] = sum_u;
    workgroupBarrier();

    // Reduce 4 lanes → 1 per column
    if lane == 0u {
        let base = col_local * 4u;
        gate_out[col] = scratch_g[base] + scratch_g[base + 1u] + scratch_g[base + 2u] + scratch_g[base + 3u];
        up_out[col]   = scratch_u[base] + scratch_u[base + 1u] + scratch_u[base + 2u] + scratch_u[base + 3u];
    }
    workgroupBarrier();
}

fn dequant_val(packed: u32, nibble: u32, scale: f32) -> f32 {
    let val = (packed >> (nibble * 4u)) & 0xFu;
    return (f32(val) - GPTQ_ZP) * scale;
}
