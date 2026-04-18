// Fused Q/gate split, mRoPE positional encoding, and KV cache write.
// NO QK norm — for models without head-wise Q/K normalization (e.g. Qwen2.5).
// Dispatch: (num_heads + num_kv_heads, 1, 1)
// workgroup_id.x < num_heads  => Q head processing
// workgroup_id.x >= num_heads => K head processing (+ V cache write)

const ROPE_THETA: f32 = 10000000.0;
const MROPE_S1_LIMIT: u32 = 11u;
const MROPE_S2_LIMIT: u32 = 22u;
const PARTIAL_DIM: u32 = 64u;
const MROPE_INTERLEAVED: bool = true;
const Q_GATED: bool = true;

struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    _pad0: u32,
    cache_position: u32,
    position: u32,
    position_h: u32,
    position_w: u32,
}

@group(0) @binding(0) var<storage, read> q_proj_full: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_proj: array<f32>;
@group(0) @binding(2) var<storage, read> v_proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> wg_vals: array<f32, 256>;
var<workgroup> wg_gate: array<f32, 256>;

// Apply mRoPE rotation to a pair of values.
fn apply_mrope(val_a: f32, val_b: f32, freq_idx: u32) -> vec2<f32> {
    let freq = 1.0 / pow(ROPE_THETA, 2.0 * f32(freq_idx) / f32(PARTIAL_DIM));

    var pos: u32 = params.position;
    if ((freq_idx % 3u) == 1u && freq_idx < MROPE_S1_LIMIT) {
        pos = params.position_h;
    } else if ((freq_idx % 3u) == 2u && freq_idx < MROPE_S2_LIMIT) {
        pos = params.position_w;
    }

    let angle = f32(pos) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    return vec2<f32>(val_a * cos_a - val_b * sin_a, val_b * cos_a + val_a * sin_a);
}

fn apply_mrope_to_wg(tid: u32) {
    let partial_half = PARTIAL_DIM / 2u;

    if (MROPE_INTERLEAVED) {
        var d = tid;
        while (d < partial_half) {
            let a = wg_vals[2u * d];
            let b = wg_vals[2u * d + 1u];
            let r = apply_mrope(a, b, d);
            wg_vals[2u * d] = r.x;
            wg_vals[2u * d + 1u] = r.y;
            d += 256u;
        }
    } else {
        var d = tid;
        while (d < partial_half) {
            let a = wg_vals[d];
            let b = wg_vals[d + partial_half];
            let r = apply_mrope(a, b, d);
            wg_vals[d] = r.x;
            wg_vals[d + partial_half] = r.y;
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
    let num_heads = params.num_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim = params.head_dim;

    let is_q_head = wg_id.x < num_heads;

    if (is_q_head) {
        let h = wg_id.x;
        var src_off: u32;
        if (Q_GATED) {
            src_off = h * head_dim * 2u;
        } else {
            src_off = h * head_dim;
        }

        // Load Q (and gate if gated)
        var d = tid;
        while (d < head_dim) {
            wg_vals[d] = q_proj_full[src_off + d];
            if (Q_GATED) {
                wg_gate[d] = q_proj_full[src_off + head_dim + d];
            }
            d += 256u;
        }
        workgroupBarrier();

        // mRoPE on Q
        apply_mrope_to_wg(tid);
        workgroupBarrier();

        // Write to q_proj and q_gate output buffers
        d = tid;
        while (d < head_dim) {
            q_proj[h * head_dim + d] = wg_vals[d];
            if (Q_GATED) {
                q_gate[h * head_dim + d] = wg_gate[d];
            }
            d += 256u;
        }

    } else {
        let kh = wg_id.x - num_heads;
        if (kh >= num_kv_heads) { return; }

        // Load K
        var d = tid;
        while (d < head_dim) {
            wg_vals[d] = k_proj[kh * head_dim + d];
            d += 256u;
        }
        workgroupBarrier();

        // mRoPE on K
        apply_mrope_to_wg(tid);
        workgroupBarrier();

        // Write K to k_proj and k_cache, write V to v_cache
        let cache_off = params.cache_position * num_kv_heads * head_dim + kh * head_dim;
        d = tid;
        while (d < head_dim) {
            let k_val = wg_vals[d];
            k_proj[kh * head_dim + d] = k_val;
            k_cache[cache_off + d] = k_val;
            v_cache[cache_off + d] = v_proj[kh * head_dim + d];
            d += 256u;
        }
    }
}
