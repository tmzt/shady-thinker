// Batched (prefill) DeltaNet: fused Conv1d + SiLU + DeltaNet recurrence + RMSNorm.
// Processes seq_len tokens in a serial outer position loop (DeltaNet state is sequential).
// One workgroup per key head.  Dispatch: (num_key_heads, 1, 1)
//
// Supports evd = (num_value_heads / num_key_heads) * value_dim up to 256.
// For Qwen3.5-4B hybrid: vpk=2, vd=128 → evd=256 ✓

struct Params {
    num_heads:      u32,  // linear_num_key_heads
    key_dim:        u32,  // linear_key_head_dim
    value_dim:      u32,  // linear_value_head_dim
    total_channels: u32,  // 2*nh*kd + nhv*vd
    eps:            f32,
    hidden_size:    u32,
    num_value_heads: u32,
    seq_len:        u32,
}

// binding 0: QKV after in_proj (read-only; conv writes to workgroup, not this buffer)
@group(0) @binding(0)  var<storage, read>       qkv_batch:    array<f32>;  // [seq, total_channels]
// binding 1: conv history, updated in-place across positions and across calls
@group(0) @binding(1)  var<storage, read_write> hist:         array<f32>;  // [3 * total_channels]
@group(0) @binding(2)  var<storage, read>       conv_weight:  array<u32>;  // BF16 [total_channels * 4]
// binding 3: recurrent state matrix, updated in-place
@group(0) @binding(3)  var<storage, read_write> state:        array<f32>;  // [num_value_heads * key_dim * value_dim]
// binding 4: per-position DeltaNet output (post-norm), written here
@group(0) @binding(4)  var<storage, read_write> output_batch: array<f32>;  // [seq, num_value_heads * value_dim]
// binding 5: pre-attn-norm hidden states (input to alpha/beta matmul)
@group(0) @binding(5)  var<storage, read>       hidden_batch: array<f32>;  // [seq, hidden_size]
@group(0) @binding(6)  var<storage, read>       ab_weight:    array<u32>;  // BF16 [2*nhv, hidden_size/2]
@group(0) @binding(7)  var<storage, read>       A_log:        array<u32>;  // BF16 [nhv/2]
@group(0) @binding(8)  var<storage, read>       dt_bias:      array<u32>;  // BF16 [nhv/2]
@group(0) @binding(9)  var<storage, read>       norm_weight:  array<u32>;  // BF16 [value_dim]
@group(0) @binding(10) var<uniform>             params:       Params;

var<workgroup> wg_q:          array<f32, 128>;  // normalized Q for this head
var<workgroup> wg_k:          array<f32, 128>;  // normalized K for this head
var<workgroup> wg_v:          array<f32, 256>;  // post-conv V (evd <= 256)
var<workgroup> shared_reduce:  array<f32, 128>;
var<workgroup> shared_reduce2: array<f32, 128>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid = lid.x;
    let h   = wg_id.x;  // key head index

    let nh   = params.num_heads;
    let kd   = params.key_dim;
    let vd   = params.value_dim;
    let ch   = params.total_channels;
    let eps  = params.eps;
    let H    = params.hidden_size;
    let nhv  = params.num_value_heads;
    let vpk  = nhv / nh;   // value heads per key head
    let evd  = vpk * vd;   // total value elements for this key head
    let sl   = params.seq_len;

    for (var pos: u32 = 0u; pos < sl; pos++) {
        let qkv_off = pos * ch;
        let hid_off = pos * H;
        let out_off = pos * nhv * vd;

        // ==================== Phase 1: Conv1d + SiLU ====================
        // Q channels for this head: global channel indices [h*kd, (h+1)*kd)
        var c = tid;
        while (c < kd) {
            let gc = h * kd + c;
            let w0 = unpack_bf16(conv_weight[gc * 2u],      0u);
            let w1 = unpack_bf16(conv_weight[gc * 2u],      1u);
            let w2 = unpack_bf16(conv_weight[gc * 2u + 1u], 0u);
            let w3 = unpack_bf16(conv_weight[gc * 2u + 1u], 1u);

            let x = qkv_batch[qkv_off + gc];
            let y = w0 * hist[gc] + w1 * hist[ch + gc] + w2 * hist[2u * ch + gc] + w3 * x;
            wg_q[c] = y / (1.0 + exp(-y));  // SiLU; will be L2-normalized next

            // shift history: slot0 ← slot1, slot1 ← slot2, slot2 ← current input
            hist[gc]           = hist[ch + gc];
            hist[ch + gc]      = hist[2u * ch + gc];
            hist[2u * ch + gc] = x;
            c += 128u;
        }
        workgroupBarrier();

        // K channels for this head: [nh*kd + h*kd, nh*kd + (h+1)*kd)
        c = tid;
        while (c < kd) {
            let gc = nh * kd + h * kd + c;
            let w0 = unpack_bf16(conv_weight[gc * 2u],      0u);
            let w1 = unpack_bf16(conv_weight[gc * 2u],      1u);
            let w2 = unpack_bf16(conv_weight[gc * 2u + 1u], 0u);
            let w3 = unpack_bf16(conv_weight[gc * 2u + 1u], 1u);

            let x = qkv_batch[qkv_off + gc];
            let y = w0 * hist[gc] + w1 * hist[ch + gc] + w2 * hist[2u * ch + gc] + w3 * x;
            wg_k[c] = y / (1.0 + exp(-y));

            hist[gc]           = hist[ch + gc];
            hist[ch + gc]      = hist[2u * ch + gc];
            hist[2u * ch + gc] = x;
            c += 128u;
        }
        workgroupBarrier();

        // V channels for this head: [2*nh*kd + h*evd, 2*nh*kd + (h+1)*evd)
        c = tid;
        while (c < evd) {
            let gc = 2u * nh * kd + h * evd + c;
            let w0 = unpack_bf16(conv_weight[gc * 2u],      0u);
            let w1 = unpack_bf16(conv_weight[gc * 2u],      1u);
            let w2 = unpack_bf16(conv_weight[gc * 2u + 1u], 0u);
            let w3 = unpack_bf16(conv_weight[gc * 2u + 1u], 1u);

            let x = qkv_batch[qkv_off + gc];
            let y = w0 * hist[gc] + w1 * hist[ch + gc] + w2 * hist[2u * ch + gc] + w3 * x;
            wg_v[c] = y / (1.0 + exp(-y));

            hist[gc]           = hist[ch + gc];
            hist[ch + gc]      = hist[2u * ch + gc];
            hist[2u * ch + gc] = x;
            c += 128u;
        }
        workgroupBarrier();

        // ==================== Phase 2: Q/K L2 normalization ====================
        var q_ss: f32 = 0.0;
        var k_ss: f32 = 0.0;
        c = tid;
        while (c < kd) {
            let qv = wg_q[c];
            let kv = wg_k[c];
            q_ss += qv * qv;
            k_ss += kv * kv;
            c += 128u;
        }

        shared_reduce[tid]  = q_ss;
        shared_reduce2[tid] = k_ss;
        workgroupBarrier();

        var stride = 64u;
        while (stride > 0u) {
            if (tid < stride) {
                shared_reduce[tid]  += shared_reduce[tid  + stride];
                shared_reduce2[tid] += shared_reduce2[tid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        let q_inv = 1.0 / max(sqrt(shared_reduce[0]),  1e-6) / sqrt(f32(kd));
        let k_inv = 1.0 / max(sqrt(shared_reduce2[0]), 1e-6);
        workgroupBarrier();

        c = tid;
        while (c < kd) {
            wg_q[c] *= q_inv;
            wg_k[c] *= k_inv;
            c += 128u;
        }
        workgroupBarrier();

        // ==================== Phase 3: DeltaNet recurrence ====================
        let half_H = H / 2u;

        for (var vhi: u32 = 0u; vhi < vpk; vhi++) {
            let vh = h * vpk + vhi;

            // alpha/beta matmul with hidden_batch[pos]
            var alpha_local: f32 = 0.0;
            var beta_local:  f32 = 0.0;
            var j = tid;
            while (j < half_H) {
                let a_packed = ab_weight[vh * half_H + j];
                let b_packed = ab_weight[(nhv + vh) * half_H + j];

                alpha_local += unpack_bf16(a_packed, 0u) * hidden_batch[hid_off + j * 2u];
                alpha_local += unpack_bf16(a_packed, 1u) * hidden_batch[hid_off + j * 2u + 1u];
                beta_local  += unpack_bf16(b_packed, 0u) * hidden_batch[hid_off + j * 2u];
                beta_local  += unpack_bf16(b_packed, 1u) * hidden_batch[hid_off + j * 2u + 1u];
                j += 128u;
            }

            // tree-reduce alpha
            shared_reduce[tid] = alpha_local;
            workgroupBarrier();
            stride = 64u;
            while (stride > 0u) {
                if (tid < stride) { shared_reduce[tid] += shared_reduce[tid + stride]; }
                workgroupBarrier();
                stride = stride >> 1u;
            }
            let alpha_sum = shared_reduce[0];
            workgroupBarrier();

            // tree-reduce beta_raw
            shared_reduce[tid] = beta_local;
            workgroupBarrier();
            stride = 64u;
            while (stride > 0u) {
                if (tid < stride) { shared_reduce[tid] += shared_reduce[tid + stride]; }
                workgroupBarrier();
                stride = stride >> 1u;
            }
            let beta_raw = shared_reduce[0];
            workgroupBarrier();

            let a_log_val = unpack_bf16(A_log[vh / 2u],  vh % 2u);
            let dt_b      = unpack_bf16(dt_bias[vh / 2u], vh % 2u);

            let sp_input = alpha_sum + dt_b;
            var softplus_val: f32;
            if (sp_input > 20.0) {
                softplus_val = sp_input;
            } else if (sp_input < -20.0) {
                softplus_val = exp(sp_input);
            } else {
                softplus_val = log(1.0 + exp(sp_input));
            }

            let decay = exp(-exp(a_log_val) * softplus_val);
            let beta  = 1.0 / (1.0 + exp(-beta_raw));

            let v_wg_off = vhi * vd;   // offset into wg_v for this value head
            let s_base   = vh * kd * vd;

            var vi = tid;
            while (vi < vd) {
                // Pass 1: decay state rows, accumulate kv_mem = S[:,vi] · k
                var kv_mem: f32 = 0.0;
                for (var ki: u32 = 0u; ki < kd; ki++) {
                    let s_idx = s_base + ki * vd + vi;
                    state[s_idx] *= decay;
                    kv_mem += state[s_idx] * wg_k[ki];
                }

                let delta = (wg_v[v_wg_off + vi] - kv_mem) * beta;

                // Pass 2: update state with outer product, compute o = S[:,vi] · q
                var o_val: f32 = 0.0;
                for (var ki: u32 = 0u; ki < kd; ki++) {
                    let s_idx = s_base + ki * vd + vi;
                    state[s_idx] += wg_k[ki] * delta;
                    o_val += state[s_idx] * wg_q[ki];
                }

                output_batch[out_off + vh * vd + vi] = o_val;
                vi += 128u;
            }
            storageBarrier();
            workgroupBarrier();
        }

        // ==================== Phase 4: RMSNorm per value head ====================
        for (var vhi: u32 = 0u; vhi < vpk; vhi++) {
            let vh       = h * vpk + vhi;
            let out_base = out_off + vh * vd;

            var ss: f32 = 0.0;
            var i = tid;
            while (i < vd) {
                let v = output_batch[out_base + i];
                ss += v * v;
                i += 128u;
            }

            shared_reduce[tid] = ss;
            workgroupBarrier();
            var stride2 = 64u;
            while (stride2 > 0u) {
                if (tid < stride2) { shared_reduce[tid] += shared_reduce[tid + stride2]; }
                workgroupBarrier();
                stride2 = stride2 >> 1u;
            }

            let rms = 1.0 / sqrt(shared_reduce[0] / f32(vd) + eps);
            workgroupBarrier();

            i = tid;
            while (i < vd) {
                let w = unpack_bf16(norm_weight[i / 2u], i % 2u);
                output_batch[out_base + i] *= rms * w;
                i += 128u;
            }
            storageBarrier();
            workgroupBarrier();
        }

    } // end for pos in 0..seq_len
}
