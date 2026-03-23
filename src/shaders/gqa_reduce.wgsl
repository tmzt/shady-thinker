// Merge attention splits via log-sum-exp weighting.
// Each workgroup merges all splits for one head.
// Partials layout: [(head_dim + 2) * num_splits * num_heads]
//   per entry: [partial_out(head_dim floats), log_sum_exp, max]
// Dispatch: (num_heads, 1, 1)

struct Params {
    head_dim: u32,
    num_splits: u32,
    num_heads: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_global_max: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let head = wid.x;
    let head_dim = params.head_dim;
    let num_splits = params.num_splits;
    let stride = head_dim + 2u;

    // Step 1: Find global max log-sum-exp across all splits.
    // Thread 0 computes this since num_splits is typically small.
    if (tid == 0u) {
        var global_max_lse: f32 = -3.402823e+38;
        var s = 0u;
        while (s < num_splits) {
            let base = (head * num_splits + s) * stride;
            let lse = partials[base + head_dim];
            global_max_lse = max(global_max_lse, lse);
            s += 1u;
        }
        shared_global_max = global_max_lse;
    }
    workgroupBarrier();

    let global_max_lse = shared_global_max;

    // Step 2: Each thread handles strided dimensions, weighted sum across splits.
    var d = tid;
    while (d < head_dim) {
        var acc: f32 = 0.0;
        var weight_sum: f32 = 0.0;
        var s = 0u;
        while (s < num_splits) {
            let base = (head * num_splits + s) * stride;
            let lse = partials[base + head_dim];
            let w = exp(lse - global_max_lse);
            acc += w * partials[base + d];
            weight_sum += w;
            s += 1u;
        }
        output[head * head_dim + d] = acc / weight_sum;
        d += 256u;
    }
}
