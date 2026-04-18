// Batched RoPE + KV cache write for prefill.
// NO QK norm — for models without head-wise Q/K normalization (e.g. Qwen2.5).
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
    _pad0: u32,
    seq_len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> q_proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_proj: array<f32>;
@group(0) @binding(2) var<storage, read> v_proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> wg_vals: array<f32, 256>;

fn apply_rope_interleaved(tid: u32, pos: u32) {
    let partial_half = PARTIAL_DIM / 2u;
    if (MROPE_INTERLEAVED) {
        var d = tid;
        while (d < partial_half) {
            let freq = 1.0 / pow(ROPE_THETA, 2.0 * f32(d) / f32(PARTIAL_DIM));
            let angle = f32(pos) * freq;
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
            let angle = f32(pos) * freq;
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
    let head_dim = params.head_dim;
    let num_q = params.num_q_heads;
    let num_kv = params.num_kv_heads;

    let is_q = head_idx < num_q;

    if (is_q) {
        let h = head_idx;
        let base = pos * num_q * head_dim + h * head_dim;

        // Load Q
        var d = tid;
        while (d < head_dim) {
            wg_vals[d] = q_proj[base + d];
            d += 256u;
        }
        workgroupBarrier();

        // RoPE
        apply_rope_interleaved(tid, pos);
        workgroupBarrier();

        // Write back
        d = tid;
        while (d < head_dim) {
            q_proj[base + d] = wg_vals[d];
            d += 256u;
        }

    } else {
        let kh = head_idx - num_q;
        if (kh >= num_kv) { return; }

        let base = pos * num_kv * head_dim + kh * head_dim;

        // Load K
        var d = tid;
        while (d < head_dim) {
            wg_vals[d] = k_proj[base + d];
            d += 256u;
        }
        workgroupBarrier();

        // RoPE
        apply_rope_interleaved(tid, pos);
        workgroupBarrier();

        // Write K to k_proj and k_cache, write V to v_cache
        let cache_base = pos * num_kv * head_dim + kh * head_dim;
        d = tid;
        while (d < head_dim) {
            let k_val = wg_vals[d];
            k_proj[base + d] = k_val;
            k_cache[cache_base + d] = k_val;
            v_cache[cache_base + d] = v_proj[base + d];
            d += 256u;
        }
    }
}
