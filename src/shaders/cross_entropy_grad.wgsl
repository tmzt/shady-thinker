// Cross-entropy loss gradient with numerically stable softmax.
// Single workgroup (256 threads) handles arbitrary vocab_size via strided access.

struct Params {
    vocab_size: u32,
    target_token: u32,
}

@group(0) @binding(0) var<storage, read>       logits:   array<f32>;
@group(0) @binding(1) var<storage, read_write>  grad_out: array<f32>;
@group(0) @binding(2) var<storage, read_write>  loss_out: array<f32>;
@group(0) @binding(3) var<uniform>              params:   Params;

var<workgroup> shared: array<f32, 256u>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local_id = lid.x;
    let vocab_size = params.vocab_size;
    let target = params.target_token;

    // Phase 1: Find max(logits) via tree reduction.
    var local_max = -3.402823e+38f; // -FLT_MAX
    for (var i = local_id; i < vocab_size; i += 256u) {
        local_max = max(local_max, logits[i]);
    }
    shared[local_id] = local_max;
    workgroupBarrier();

    // Tree reduction for max.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_id < stride {
            shared[local_id] = max(shared[local_id], shared[local_id + stride]);
        }
        workgroupBarrier();
    }
    let max_val = shared[0];
    workgroupBarrier();

    // Phase 2: Compute sum(exp(logits[i] - max_val)) via tree reduction.
    var local_sum = 0.0f;
    for (var i = local_id; i < vocab_size; i += 256u) {
        local_sum += exp(logits[i] - max_val);
    }
    shared[local_id] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if local_id < stride {
            shared[local_id] = shared[local_id] + shared[local_id + stride];
        }
        workgroupBarrier();
    }
    let sum_exp = shared[0];
    workgroupBarrier();

    // Phase 3: Write softmax gradient and loss.
    for (var i = local_id; i < vocab_size; i += 256u) {
        var prob = exp(logits[i] - max_val) / sum_exp;
        if i == target {
            prob -= 1.0f;
        }
        grad_out[i] = prob;
    }

    // Thread 0 writes the cross-entropy loss.
    if local_id == 0u {
        let loss = -(logits[target] - max_val - log(sum_exp));
        loss_out[0] = loss;
    }
}
