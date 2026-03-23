// EWC gradient blending: grad[i] += lambda * anchor[i].
// Dispatch: (ceil(num_elements/256), 1, 1)

struct Params {
    num_elements: u32,
    lambda: f32,
}

@group(0) @binding(0) var<storage, read_write>  grad:   array<f32>;
@group(0) @binding(1) var<storage, read>        anchor: array<f32>;
@group(0) @binding(2) var<uniform>              params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.num_elements {
        return;
    }

    grad[i] = grad[i] + params.lambda * anchor[i];
}
