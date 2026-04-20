// MoE weighted accumulation: accum[i] += weight * src[i]
// Reads weight from moe_selected[expert_idx * 2 + 1] (GPU-side, no CPU readback).
// Dispatch: (ceil(num_elements/256), 1, 1)

struct Params {
    num_elements: u32,
    expert_idx: u32,
}

@group(0) @binding(0) var<storage, read_write> accum:        array<f32>;
@group(0) @binding(1) var<storage, read>       src:          array<f32>;
@group(0) @binding(2) var<uniform>             params:       Params;
@group(0) @binding(3) var<storage, read>       moe_selected: array<u32>; // [K*2]

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.num_elements { return; }

    let weight = bitcast<f32>(moe_selected[params.expert_idx * 2u + 1u]);
    accum[i] = accum[i] + weight * src[i];
}
