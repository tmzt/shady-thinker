// Batched Q/K RMSNorm + mRoPE + KV cache write for GPTQ prefill with Q_GATED=true.
// Processes all sequence positions in one dispatch.
// Q projection input is [seq_len, num_q_heads, head_dim * 2] (Q values + gate values).
// Outputs split Q to q_proj and gate to q_gate (both [seq_len, num_q_heads, head_dim]).
//
// Dispatch: (num_q_heads + num_kv_heads, seq_len, 1)
//   workgroup_id.x < num_q_heads  => Q head
//   workgroup_id.x >= num_q_heads => K head (+ V cache write)
//   workgroup_id.y = sequence position

// ── injected by build_batched_qknorm_shader_gated() ──
// const ROPE_THETA: f32 = ...;
// const MROPE_S1_LIMIT: u32 = ...;
// const MROPE_S2_LIMIT: u32 = ...;
// const PARTIAL_DIM: u32 = ...;
// const MROPE_INTERLEAVED: bool = ...;
// const NORM_OFFSET: f32 = ...;

struct Params {
    num_heads:    u32,
    num_kv_heads: u32,
    head_dim:     u32,
    eps:          f32,
    seq_len:      u32,
    // Continuation prefill offset: absolute position of the *first* new
    // token in the global sequence. RoPE angles use `pos_offset + pos`,
    // and the KV cache write lands at `(pos_offset + pos)`. From-scratch
    // prefill passes 0, reproducing the original math.
    pos_offset:   u32,
    _pad1:        u32,
    _pad2:        u32,
    qk_norm_weight: array<vec4<u32>, 320>,
}

// Q projection full output from GPTQ: [seq_len, num_q_heads, head_dim * 2]
@group(0) @binding(0) var<storage, read>       q_proj_full: array<f32>;
// Normalized Q: [seq_len, num_q_heads, head_dim]
@group(0) @binding(1) var<storage, read_write> q_proj:      array<f32>;
// Gate values: [seq_len, num_q_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> q_gate:      array<f32>;
// K projection: [seq_len, num_kv_heads, head_dim] (in-place norm+RoPE)
@group(0) @binding(3) var<storage, read_write> k_proj:      array<f32>;
// V projection: [seq_len, num_kv_heads, head_dim] (read-only for cache write)
@group(0) @binding(4) var<storage, read>       v_proj:      array<f32>;
// KV cache: [max_seq, num_kv_heads, head_dim]
@group(0) @binding(5) var<storage, read_write> k_cache:     array<f32>;
@group(0) @binding(6) var<storage, read_write> v_cache:     array<f32>;
@group(0) @binding(7) var<uniform>             params:      Params;

var<workgroup> wg_reduce: array<f32, 256>;
var<workgroup> wg_vals:   array<f32, 256>;
var<workgroup> wg_gate:   array<f32, 256>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

fn get_norm_weight(p: u32) -> f32 {
    let vec_idx     = p / 8u;
    let u32_in_vec  = (p / 2u) % 4u;
    let bf16_in_u32 = p % 2u;
    return unpack_bf16(params.qk_norm_weight[vec_idx][u32_in_vec], bf16_in_u32);
}

fn apply_mrope(val_a: f32, val_b: f32, freq_idx: u32, abs_pos: u32) -> vec2<f32> {
    let freq  = 1.0 / pow(ROPE_THETA, 2.0 * f32(freq_idx) / f32(PARTIAL_DIM));
    // Text-only prefill: all three mRoPE coordinates equal the sequence position
    let angle = f32(abs_pos) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    return vec2<f32>(val_a * cos_a - val_b * sin_a, val_b * cos_a + val_a * sin_a);
}

fn apply_mrope_to_wg(tid: u32, abs_pos: u32) {
    let partial_half = PARTIAL_DIM / 2u;
    if (MROPE_INTERLEAVED) {
        var d = tid;
        while (d < partial_half) {
            let r = apply_mrope(wg_vals[2u * d], wg_vals[2u * d + 1u], d, abs_pos);
            wg_vals[2u * d]      = r.x;
            wg_vals[2u * d + 1u] = r.y;
            d += 256u;
        }
    } else {
        var d = tid;
        while (d < partial_half) {
            let r = apply_mrope(wg_vals[d], wg_vals[d + partial_half], d, abs_pos);
            wg_vals[d]              = r.x;
            wg_vals[d + partial_half] = r.y;
            d += 256u;
        }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let tid       = lid.x;
    let head_idx  = wg_id.x;
    let pos       = wg_id.y;   // sequence position in [0, seq_len)
    let abs_pos   = params.pos_offset + pos;  // absolute position in cache / RoPE

    let num_heads    = params.num_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim     = params.head_dim;
    let eps          = params.eps;

    if (pos >= params.seq_len) { return; }

    if (head_idx < num_heads) {
        // ── Q HEAD ──
        let h = head_idx;
        // Raw Q+gate input: [pos, h, head_dim*2]
        let src_base = pos * num_heads * head_dim * 2u + h * head_dim * 2u;
        // Output Q: [pos, h, head_dim]
        let out_q_base = pos * num_heads * head_dim + h * head_dim;

        // Load Q and gate values
        var sum_sq: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            let q_val = q_proj_full[src_base + d];
            wg_vals[d] = q_val;
            wg_gate[d] = q_proj_full[src_base + head_dim + d];
            sum_sq += q_val * q_val;
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

        let rms = 1.0 / sqrt(wg_reduce[0] / f32(head_dim) + eps);
        workgroupBarrier();

        d = tid;
        while (d < head_dim) {
            let w = get_norm_weight(d);
            wg_vals[d] = wg_vals[d] * rms * (NORM_OFFSET + w);
            d += 256u;
        }
        workgroupBarrier();

        apply_mrope_to_wg(tid, abs_pos);
        workgroupBarrier();

        d = tid;
        while (d < head_dim) {
            q_proj[out_q_base + d] = wg_vals[d];
            q_gate[out_q_base + d] = wg_gate[d];
            d += 256u;
        }

    } else {
        // ── K HEAD ──
        let kh = head_idx - num_heads;
        if (kh >= num_kv_heads) { return; }

        let kv_base   = pos * num_kv_heads * head_dim + kh * head_dim;
        // Cache write uses ABSOLUTE position so continuation prefill lands
        // past the prefix. From-scratch prefill has pos_offset=0 → identical.
        let cache_off = abs_pos * num_kv_heads * head_dim + kh * head_dim;

        var sum_sq: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            let k_val = k_proj[kv_base + d];
            wg_vals[d] = k_val;
            sum_sq += k_val * k_val;
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

        let rms = 1.0 / sqrt(wg_reduce[0] / f32(head_dim) + eps);
        workgroupBarrier();

        d = tid;
        while (d < head_dim) {
            let w = get_norm_weight(head_dim + d);
            wg_vals[d] = wg_vals[d] * rms * (NORM_OFFSET + w);
            d += 256u;
        }
        workgroupBarrier();

        apply_mrope_to_wg(tid, abs_pos);
        workgroupBarrier();

        d = tid;
        while (d < head_dim) {
            let k_val = wg_vals[d];
            k_proj[kv_base + d]   = k_val;
            k_cache[cache_off + d] = k_val;
            v_cache[cache_off + d] = v_proj[kv_base + d];
            d += 256u;
        }
    }
}
