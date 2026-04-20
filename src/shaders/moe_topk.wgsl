// MoE top-k router: softmax over logits, select top-K experts, normalize weights.
// Runs in a single workgroup on the GPU — avoids CPU readback entirely.
//
// Input:  logits[num_experts] (f32, from router matvec)
// Output: selected[K] = { expert_id: u32, weight: f32 } pairs
//
// Dispatch: (1, 1, 1) — single workgroup

struct Params {
    num_experts: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read>       logits:   array<f32>;
@group(0) @binding(1) var<storage, read_write> selected: array<u32>; // [K*2]: expert_id, weight_bits alternating
@group(0) @binding(2) var<uniform>             params:   Params;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let n = params.num_experts;
    let k = params.k;

    // Step 1: find max for numerical stability
    var max_val: f32 = -1e30;
    for (var i = 0u; i < n; i++) {
        max_val = max(max_val, logits[i]);
    }

    // Step 2: softmax
    var sum_exp: f32 = 0.0;
    for (var i = 0u; i < n; i++) {
        sum_exp += exp(logits[i] - max_val);
    }

    // Step 3: top-k selection (simple iterative — K is small, typically 8)
    // Use a bitmask to track already-selected experts
    var used: array<u32, 4>; // 128-bit bitmask (supports up to 128 experts)
    for (var b = 0u; b < 4u; b++) { used[b] = 0u; }

    var weight_sum: f32 = 0.0;
    for (var ki = 0u; ki < k; ki++) {
        var best_idx = 0u;
        var best_prob: f32 = -1.0;
        for (var i = 0u; i < n; i++) {
            let word = i / 32u;
            let bit = 1u << (i % 32u);
            if (used[word] & bit) != 0u { continue; }

            let prob = exp(logits[i] - max_val) / sum_exp;
            if prob > best_prob {
                best_prob = prob;
                best_idx = i;
            }
        }
        // Mark as used
        let word = best_idx / 32u;
        let bit = 1u << (best_idx % 32u);
        used[word] = used[word] | bit;

        weight_sum += best_prob;

        // Store expert_id and weight (will renormalize after)
        selected[ki * 2u] = best_idx;
        selected[ki * 2u + 1u] = bitcast<u32>(best_prob);
    }

    // Step 4: renormalize weights to sum to 1
    if weight_sum > 0.0 {
        for (var ki = 0u; ki < k; ki++) {
            let w = bitcast<f32>(selected[ki * 2u + 1u]);
            selected[ki * 2u + 1u] = bitcast<u32>(w / weight_sum);
        }
    }
}
