// Batched Q/K RMSNorm + RoPE for prefill.
// Processes Q[seq_len, num_q_heads, head_dim] and K[seq_len, num_kv_heads, head_dim] in-place.
// Also writes K and V to the KV cache for subsequent autoregressive decoding.
//
// Dispatch: (num_q_heads + num_kv_heads, seq_len, 1)
//   workgroup_id.x < num_q_heads: Q head processing
//   workgroup_id.x >= num_q_heads: K head processing (+ V/K cache write)
//   workgroup_id.y = token position

const ROPE_THETA: f32 = 1000000.0;
const PARTIAL_DIM: u32 = 128u;
const MROPE_INTERLEAVED: bool = true;

struct Params {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    eps: f32,
    seq_len: u32,
    // Continuation prefill offset: absolute position of the *first* new
    // token in the global sequence. RoPE angles use `pos_offset + pos`,
    // and the KV cache write lands at `(pos_offset + pos)`. From-scratch
    // prefill passes 0, reproducing the original math.
    pos_offset: u32,
    _pad1: u32,
    _pad2: u32,
    qk_norm_weight: array<vec4<u32>, 320>,
}

@group(0) @binding(0) var<storage, read_write> q_proj: array<f32>;  // [seq_len, num_q_heads, head_dim]
@group(0) @binding(1) var<storage, read_write> k_proj: array<f32>;  // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(2) var<storage, read> v_proj: array<f32>;        // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> k_cache: array<f32>; // [max_seq, num_kv_heads, head_dim]
@group(0) @binding(4) var<storage, read_write> v_cache: array<f32>; // [max_seq, num_kv_heads, head_dim]
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> wg_reduce: array<f32, 256>;
var<workgroup> wg_vals: array<f32, 256>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

fn get_norm_weight(p: u32) -> f32 {
    let vec_idx = p / 8u;
    let u32_in_vec = (p / 2u) % 4u;
    let bf16_in_u32 = p % 2u;
    return unpack_bf16(params.qk_norm_weight[vec_idx][u32_in_vec], bf16_in_u32);
}

fn apply_rope_interleaved(tid: u32, abs_pos: u32) {
    let partial_half = PARTIAL_DIM / 2u;
    if (MROPE_INTERLEAVED) {
        var d = tid;
        while (d < partial_half) {
            let freq = 1.0 / pow(ROPE_THETA, 2.0 * f32(d) / f32(PARTIAL_DIM));
            // All positions use temporal (same pos for h/w in text-only decoder)
            let angle = f32(abs_pos) * freq;
            let cos_a = cos(angle);
            let sin_a = sin(angle);
            let a = wg_vals[2u * d];
            let b = wg_vals[2u * d + 1u];
            wg_vals[2u * d] = a * cos_a - b * sin_a;
            wg_vals[2u * d + 1u] = b * cos_a + a * sin_a;
            d += 256u;
        }
    } else {
        var d = tid;
        while (d < partial_half) {
            let freq = 1.0 / pow(ROPE_THETA, 2.0 * f32(d) / f32(PARTIAL_DIM));
            let angle = f32(abs_pos) * freq;
            let cos_a = cos(angle);
            let sin_a = sin(angle);
            let a = wg_vals[d];
            let b = wg_vals[d + partial_half];
            wg_vals[d] = a * cos_a - b * sin_a;
            wg_vals[d + partial_half] = b * cos_a + a * sin_a;
            d += 256u;
        }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let head_idx = wg_id.x;
    let pos = wg_id.y;
    let abs_pos = params.pos_offset + pos;  // absolute position in cache / RoPE
    let head_dim = params.head_dim;
    let num_q = params.num_q_heads;
    let num_kv = params.num_kv_heads;

    let is_q = head_idx < num_q;

    if (is_q) {
        // Q head processing
        let h = head_idx;
        let base = pos * num_q * head_dim + h * head_dim;

        // Load Q values and compute sum of squares for RMSNorm
        var sum_sq: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            let v = q_proj[base + d];
            wg_vals[d] = v;
            sum_sq += v * v;
            d += 256u;
        }
        wg_reduce[tid] = sum_sq;
        workgroupBarrier();

        var stride = 128u;
        while (stride > 0u) {
            if (tid < stride) {
                wg_reduce[tid] = wg_reduce[tid] + wg_reduce[tid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        let rms = 1.0 / sqrt(wg_reduce[0] / f32(head_dim) + params.eps);
        workgroupBarrier();

        // Apply RMSNorm with (1 + w) scaling
        d = tid;
        while (d < head_dim) {
            let w = get_norm_weight(d);
            wg_vals[d] = wg_vals[d] * rms * w;
            d += 256u;
        }
        workgroupBarrier();

        // Apply RoPE (uses absolute position for continuation prefill)
        apply_rope_interleaved(tid, abs_pos);
        workgroupBarrier();

        // Write back to q_proj
        d = tid;
        while (d < head_dim) {
            q_proj[base + d] = wg_vals[d];
            d += 256u;
        }

    } else {
        // K head processing
        let kh = head_idx - num_q;
        if (kh >= num_kv) { return; }

        let base = pos * num_kv * head_dim + kh * head_dim;

        // Load K values
        var sum_sq: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            let v = k_proj[base + d];
            wg_vals[d] = v;
            sum_sq += v * v;
            d += 256u;
        }
        wg_reduce[tid] = sum_sq;
        workgroupBarrier();

        var stride = 128u;
        while (stride > 0u) {
            if (tid < stride) {
                wg_reduce[tid] = wg_reduce[tid] + wg_reduce[tid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        let rms = 1.0 / sqrt(wg_reduce[0] / f32(head_dim) + params.eps);
        workgroupBarrier();

        // K norm weights start at head_dim offset
        d = tid;
        while (d < head_dim) {
            let w = get_norm_weight(head_dim + d);
            wg_vals[d] = wg_vals[d] * rms * w;
            d += 256u;
        }
        workgroupBarrier();

        // Apply RoPE (uses absolute position for continuation prefill)
        apply_rope_interleaved(tid, abs_pos);
        workgroupBarrier();

        // Write K back to k_proj (batch-local index `pos`) and to k_cache
        // (absolute index `abs_pos` so continuation lands past the prefix).
        let cache_base = abs_pos * num_kv * head_dim + kh * head_dim;
        d = tid;
        while (d < head_dim) {
            let k_val = wg_vals[d];
            k_proj[base + d] = k_val;
            k_cache[cache_base + d] = k_val;
            d += 256u;
        }

        // Write V to v_cache
        d = tid;
        while (d < head_dim) {
            v_cache[cache_base + d] = v_proj[base + d];
            d += 256u;
        }
    }
}
