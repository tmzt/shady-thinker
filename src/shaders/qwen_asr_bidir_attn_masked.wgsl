// Qwen-ASR Bidirectional Windowed Attention with attention mask.
// Same as qwen_asr_bidir_attn.wgsl but masks out padded positions.
// Masked positions get -inf score before softmax.
//
// Dispatch: (num_heads, seq_len, 1)

struct Params {
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
    window_start: u32,
    window_end: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> attn_mask: array<u32>;  // [seq_len] 1=real, 0=padded

var<workgroup> shared_reduce: array<f32, 256>;
var<workgroup> shared_acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let head = wid.x;
    let q_pos = wid.y;
    let head_dim = params.head_dim;
    let num_heads = params.num_heads;
    let scale = 1.0 / sqrt(f32(head_dim));

    if (head >= num_heads || q_pos >= params.seq_len) {
        return;
    }

    // If this query position is masked, output zeros
    if (attn_mask[q_pos] == 0u) {
        if (tid < head_dim) {
            let out_base = (q_pos * num_heads + head) * head_dim;
            output[out_base + tid] = 0.0;
        }
        return;
    }

    let win_start = params.window_start;
    let win_end = min(params.window_end, params.seq_len);
    let q_base = (q_pos * num_heads + head) * head_dim;

    if (tid < head_dim) {
        shared_acc[tid] = 0.0;
    }
    workgroupBarrier();

    var running_max: f32 = -3.402823e+38;
    var running_sum: f32 = 0.0;

    var kv_pos = win_start;
    while (kv_pos < win_end) {
        // Skip masked key positions
        if (attn_mask[kv_pos] == 0u) {
            kv_pos += 1u;
            continue;
        }

        let kv_base = (kv_pos * num_heads + head) * head_dim;

        var local_dot: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            local_dot += q[q_base + d] * k[kv_base + d];
            d += 256u;
        }
        shared_reduce[tid] = local_dot;
        workgroupBarrier();

        var stride = 128u;
        while (stride > 0u) {
            if (tid < stride) {
                shared_reduce[tid] = shared_reduce[tid] + shared_reduce[tid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        let score = shared_reduce[0] * scale;

        let new_max = max(running_max, score);
        let correction = exp(running_max - new_max);
        let exp_score = exp(score - new_max);
        running_sum = running_sum * correction + exp_score;

        if (tid < head_dim) {
            shared_acc[tid] = shared_acc[tid] * correction + exp_score * v[kv_base + tid];
        }
        workgroupBarrier();

        running_max = new_max;
        kv_pos += 1u;
    }

    if (tid < head_dim) {
        let out_base = (q_pos * num_heads + head) * head_dim;
        if (running_sum > 0.0) {
            output[out_base + tid] = shared_acc[tid] / running_sum;
        } else {
            output[out_base + tid] = 0.0;
        }
    }
}
