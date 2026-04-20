// Fused SiLU + GPTQ down projection for MoE — reads expert offset from GPU buffer.
// Same algorithm as fused_silu_gptq_4t but with packed expert buffers.
//
// Computes: output[row] = down_proj[expert_id][row, :] @ (SiLU(gate) * up)
// expert_id read from moe_selected[expert_idx * 2] (no CPU readback).
//
// Dispatch: (ceil(hidden_size / 8), 1, 1)

const GPTQ_ZP: f32 = 8.0;

struct Params {
    K: u32,            // input dimension (inter_size)
    N: u32,            // output dimension (hidden_size)
    group_size: u32,
    expert_idx: u32,   // which of the K selected experts (0..K-1)
    expert_stride_qw: u32,
    expert_stride_sc: u32,
}

@group(0) @binding(0) var<storage, read>       gate_out:     array<f32>;  // [inter]
@group(0) @binding(1) var<storage, read>       up_out:       array<f32>;  // [inter]
@group(0) @binding(2) var<storage, read>       down_qweight: array<u32>;  // packed [num_experts * stride]
@group(0) @binding(3) var<storage, read>       down_scales:  array<u32>;  // packed
@group(0) @binding(4) var<storage, read_write> output:       array<f32>;  // [hidden]
@group(0) @binding(5) var<uniform>             params:       Params;
@group(0) @binding(6) var<storage, read>       moe_selected: array<u32>;  // [K*2]

var<workgroup> scratch: array<f32, 32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid       = lid.x;
    let lane      = tid & 3u;
    let col_local = tid >> 2u;
    let col       = wg_id.x * 8u + col_local;

    let expert_id = moe_selected[params.expert_idx * 2u];
    let down_base = expert_id * params.expert_stride_qw;
    let down_sc_base = expert_id * params.expert_stride_sc;

    if col >= params.N {
        scratch[tid] = 0.0;
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

    var sum: f32 = 0.0;

    for (var pr = r_start; pr < r_end; pr++) {
        let packed = down_qweight[down_base + pr * N + col];

        let group = pr * 8u / group_size;
        let sc_word = down_scales[down_sc_base + (group * N + col) / 2u];
        let sc_shift = select(0u, 16u, ((group * N + col) & 1u) != 0u);
        let sc = unpack2x16float((sc_word >> sc_shift) & 0xFFFFu).x;

        for (var nib = 0u; nib < 8u; nib++) {
            let r = pr * 8u + nib;
            // SiLU(gate) * up
            let g = gate_out[r];
            let silu_val = g / (1.0 + exp(-g)) * up_out[r];
            sum += dequant_val(packed, nib, sc) * silu_val;
        }
    }

    scratch[tid] = sum;
    workgroupBarrier();

    if lane == 0u {
        let base = col_local * 4u;
        output[col] = scratch[base] + scratch[base + 1u] + scratch[base + 2u] + scratch[base + 3u];
    }
    workgroupBarrier();
}

fn dequant_val(packed: u32, nibble: u32, scale: f32) -> f32 {
    let val = (packed >> (nibble * 4u)) & 0xFu;
    return (f32(val) - GPTQ_ZP) * scale;
}
