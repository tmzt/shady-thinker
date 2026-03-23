// Grouped Query Attention: online softmax attention over KV cache.
// Each workgroup handles one Q head over one split of the sequence.
// workgroup_id.x = q_head_idx, workgroup_id.y = split_idx.
// For num_splits == 1: output directly to output buffer.
// For num_splits > 1: write (partial_out[head_dim], log_sum_exp, max) per split
//   at output[(q_head * num_splits + split) * (head_dim + 2) + d].
// Dispatch: (num_q_heads, num_splits, 1)

struct Params {
    seq_len: u32,
    head_dim: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    heads_per_kv: u32,
    num_splits: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory: first 256 for reductions, then 256 for accumulator (head_dim <= 256).
var<workgroup> shared_reduce: array<f32, 256>;
var<workgroup> shared_acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let q_head = wid.x;
    let split_idx = wid.y;
    let head_dim = params.head_dim;
    let seq_len = params.seq_len;
    let num_kv_heads = params.num_kv_heads;
    let num_splits = params.num_splits;
    let kv_head = q_head / params.heads_per_kv;
    let scale = 1.0 / sqrt(f32(head_dim));

    // Determine split range.
    let positions_per_split = (seq_len + num_splits - 1u) / num_splits;
    let split_start = split_idx * positions_per_split;
    let split_end = min(split_start + positions_per_split, seq_len);

    // Initialize accumulator in shared memory to zero.
    if (tid < head_dim) {
        shared_acc[tid] = 0.0;
    }
    workgroupBarrier();

    var running_max: f32 = -3.402823e+38;
    var running_sum: f32 = 0.0;

    let q_base = q_head * head_dim;

    // Iterate over positions in this split.
    var pos = split_start;
    while (pos < split_end) {
        let kv_base = pos * num_kv_heads * head_dim + kv_head * head_dim;

        // --- Compute dot(q, k[pos]) via tree reduction ---
        // Each thread accumulates partial dot product over strided dimensions.
        var local_dot: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            local_dot += q[q_base + d] * k_cache[kv_base + d];
            d += 256u;
        }

        shared_reduce[tid] = local_dot;
        workgroupBarrier();

        // Tree reduce.
        var stride = 128u;
        while (stride > 0u) {
            if (tid < stride) {
                shared_reduce[tid] = shared_reduce[tid] + shared_reduce[tid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        let score = shared_reduce[0] * scale;

        // --- Online softmax update ---
        let new_max = max(running_max, score);
        let correction = exp(running_max - new_max);
        let exp_score = exp(score - new_max);
        running_sum = running_sum * correction + exp_score;

        // Update accumulator: acc[d] = acc[d] * correction + exp_score * v[pos][d].
        if (tid < head_dim) {
            shared_acc[tid] = shared_acc[tid] * correction + exp_score * v_cache[kv_base + tid];
        }
        workgroupBarrier();

        running_max = new_max;
        pos += 1u;
    }

    // --- Write output ---
    if (num_splits == 1u) {
        // Direct output.
        if (tid < head_dim) {
            output[q_head * head_dim + tid] = shared_acc[tid] / running_sum;
        }
    } else {
        // Write partial results with log_sum_exp metadata.
        let out_stride = head_dim + 2u;
        let base = (q_head * num_splits + split_idx) * out_stride;
        if (tid < head_dim) {
            output[base + tid] = shared_acc[tid] / running_sum;
        }
        if (tid == 0u) {
            output[base + head_dim] = log(running_sum) + running_max;
            output[base + head_dim + 1u] = running_max;
        }
    }
}
