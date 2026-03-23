// Argmax via tree reduction in shared memory.
// Finds (idx, val) where val = max(logits[0..N]).
// Dispatch: (1, 1, 1)

struct Params {
    N: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Result {
    idx: u32,
    val: f32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: Result;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.N;

    // Each thread finds its local max over a strided range.
    var local_max = -3.402823e+38f;
    var local_idx = 0u;

    var pos = tid;
    while (pos < n) {
        let v = logits[pos];
        if (v > local_max) {
            local_max = v;
            local_idx = pos;
        }
        pos = pos + 256u;
    }

    shared_val[tid] = local_max;
    shared_idx[tid] = local_idx;
    workgroupBarrier();

    // Tree reduction.
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            let other = tid + stride;
            if (shared_val[other] > shared_val[tid]) {
                shared_val[tid] = shared_val[other];
                shared_idx[tid] = shared_idx[other];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        result.idx = shared_idx[0];
        result.val = shared_val[0];
    }
}
