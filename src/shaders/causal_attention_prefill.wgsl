// Causal GQA attention for prefill — reads K/V from KV cache.
// Each workgroup handles one Q head at one query position.
//
// After batched_qknorm_rope writes K and V into the KV cache, this shader
// reads them back using the cache layout: [max_seq, num_kv_heads, head_dim].
// Q is still read from the dense q_proj array: [seq_len, num_q_heads, head_dim].
// Output is dense: [seq_len, num_q_heads, head_dim].
//
// Causal masking: each query at position q_pos only attends to positions [0, q_pos].
//
// Params uniform:
//   seq_len     — number of tokens being prefilled
//   head_dim    — dimension per head
//   num_kv_heads — number of key/value heads
//   num_q_heads  — number of query heads
//   heads_per_kv — GQA group size (num_q_heads / num_kv_heads)
//
// Bindings:
//   @binding(0) q_proj  — [seq_len, num_q_heads, head_dim] f32, read-only
//   @binding(1) k_cache — [max_seq, num_kv_heads, head_dim] f32, read-only
//   @binding(2) v_cache — [max_seq, num_kv_heads, head_dim] f32, read-only
//   @binding(3) output  — [seq_len, num_q_heads, head_dim] f32, read-write
//   @binding(4) params  — uniform Params
//
// Dispatch: (num_q_heads, seq_len, 1)
//   workgroup_id.x = Q head index
//   workgroup_id.y = query token position

struct Params {
    seq_len: u32,
    head_dim: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    heads_per_kv: u32,
}

@group(0) @binding(0) var<storage, read> q_proj: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> wg_score: array<f32, 256>;
var<workgroup> wg_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let q_head = wg_id.x;
    let q_pos = wg_id.y;
    let head_dim = params.head_dim;
    let seq_len = params.seq_len;
    let kv_head = q_head / params.heads_per_kv;

    if (q_head >= params.num_q_heads || q_pos >= seq_len) { return; }

    let q_base = q_pos * params.num_q_heads * head_dim + q_head * head_dim;
    let scale = 1.0 / sqrt(f32(head_dim));
    let causal_len = q_pos + 1u;

    // Phase 1: compute attention scores Q.K^T and find max (for numerical stability)
    var local_max: f32 = -1e30;
    var j = tid;
    while (j < causal_len) {
        // K is in cache layout: [pos, num_kv_heads, head_dim]
        let k_base = j * params.num_kv_heads * head_dim + kv_head * head_dim;
        var dot: f32 = 0.0;
        for (var d = 0u; d < head_dim; d += 1u) {
            dot += q_proj[q_base + d] * k_cache[k_base + d];
        }
        let s = dot * scale;
        wg_score[j] = s;
        local_max = max(local_max, s);
        j += 256u;
    }

    // Max reduction across workgroup
    wg_reduce[tid] = local_max;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_reduce[tid] = max(wg_reduce[tid], wg_reduce[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let max_score = wg_reduce[0];
    workgroupBarrier();

    // Phase 2: exp(score - max) and sum for softmax denominator
    var local_sum: f32 = 0.0;
    j = tid;
    while (j < causal_len) {
        let e = exp(wg_score[j] - max_score);
        wg_score[j] = e;
        local_sum += e;
        j += 256u;
    }

    wg_reduce[tid] = local_sum;
    workgroupBarrier();
    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_reduce[tid] = wg_reduce[tid] + wg_reduce[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let sum_exp = wg_reduce[0];
    workgroupBarrier();

    // Phase 3: weighted V sum — output = softmax(scores) @ V
    let out_base = q_pos * params.num_q_heads * head_dim + q_head * head_dim;
    var d = tid;
    while (d < head_dim) {
        var weighted_sum: f32 = 0.0;
        for (var jj = 0u; jj < causal_len; jj += 1u) {
            let prob = wg_score[jj] / sum_exp;
            // V is in cache layout: [pos, num_kv_heads, head_dim]
            let v_base = jj * params.num_kv_heads * head_dim + kv_head * head_dim;
            weighted_sum += prob * v_cache[v_base + d];
        }
        output[out_base + d] = weighted_sum;
        d += 256u;
    }
}
