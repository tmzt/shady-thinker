const ROPE_THETA: f32 = 10000000.0;
const MROPE_S1_LIMIT: u32 = 16u;
const MROPE_S2_LIMIT: u32 = 16u;
const PARTIAL_DIM: u32 = 32u;

// Fused Q/gate split, Q/K RMSNorm, mRoPE positional encoding, and KV cache write.
// Dispatch: (num_heads + num_kv_heads, 1, 1)
// workgroup_id.x < num_heads  => Q head processing
// workgroup_id.x >= num_heads => K head processing (+ V cache write)

struct Params {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    eps: f32,
    cache_position: u32,
    position: u32,
    position_h: u32,
    position_w: u32,
    qk_norm_weight: array<vec4<u32>, 320>,
}

@group(0) @binding(0) var<storage, read> q_proj_full: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_proj: array<f32>;
@group(0) @binding(2) var<storage, read> v_proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> q_proj: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> wg_reduce: array<f32, 256>;
var<workgroup> wg_vals: array<f32, 256>;
var<workgroup> wg_gate: array<f32, 256>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

fn get_norm_weight(p: u32) -> f32 {
    let vec_idx = p / 8u;
    let u32_in_vec = (p / 2u) % 4u;
    let bf16_in_u32 = p % 2u;
    let packed_u32 = params.qk_norm_weight[vec_idx][u32_in_vec];
    return unpack_bf16(packed_u32, bf16_in_u32);
}

fn apply_mrope(val_a: f32, val_b: f32, freq_idx: u32, partial_half: u32) -> vec2<f32> {
    let freq = 1.0 / pow(ROPE_THETA, 2.0 * f32(freq_idx) / f32(PARTIAL_DIM));

    // Select position based on mRoPE section
    var pos: u32 = params.position;
    let section = freq_idx % 3u;
    if (section == 1u && freq_idx < MROPE_S1_LIMIT) {
        pos = params.position_h;
    } else if (section == 2u && freq_idx < MROPE_S2_LIMIT) {
        pos = params.position_w;
    }

    let angle = f32(pos) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let out_a = val_a * cos_a - val_b * sin_a;
    let out_b = val_b * cos_a + val_a * sin_a;
    return vec2<f32>(out_a, out_b);
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
    let eps = params.eps;

    let is_q_head = wg_id.x < num_heads;

    if (is_q_head) {
        // ===================== Q HEAD PROCESSING =====================
        let h = wg_id.x;

        // Step 1: Split interleaved Q+gate and accumulate sum of squares
        var sum_sq: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            let base = h * head_dim * 2u;
            let q_val = q_proj_full[base + 2u * d];
            let gate_val = q_proj_full[base + 2u * d + 1u];
            wg_vals[d] = q_val;
            wg_gate[d] = gate_val;
            sum_sq += q_val * q_val;
            d += 256u;
        }

        // Step 2: RMSNorm tree reduction
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

        // Apply RMSNorm with (1 + weight) scaling
        // Q norm weights are at BF16 positions [0, head_dim)
        d = tid;
        while (d < head_dim) {
            let w = get_norm_weight(d);
            wg_vals[d] = wg_vals[d] * rms * (1.0 + w);
            d += 256u;
        }
        workgroupBarrier();

        // Step 3: mRoPE on Q
        let partial_half = PARTIAL_DIM / 2u;
        d = tid;
        while (d < partial_half) {
            let freq_idx = d;
            let a = wg_vals[d];
            let b = wg_vals[d + partial_half];
            let rotated = apply_mrope(a, b, freq_idx, partial_half);
            wg_vals[d] = rotated.x;
            wg_vals[d + partial_half] = rotated.y;
            d += 256u;
        }
        workgroupBarrier();

        // Step 4: Write to q_proj and q_gate output buffers
        d = tid;
        while (d < head_dim) {
            q_proj[h * head_dim + d] = wg_vals[d];
            q_gate[h * head_dim + d] = wg_gate[d];
            d += 256u;
        }

    } else {
        // ===================== K HEAD PROCESSING =====================
        let kh = wg_id.x - num_heads;

        // Step 1: RMSNorm on K — accumulate sum of squares
        var sum_sq: f32 = 0.0;
        var d = tid;
        while (d < head_dim) {
            let k_val = k_proj[kh * head_dim + d];
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

        // Apply RMSNorm with (1 + weight) scaling
        // K norm weights are at BF16 positions [head_dim, 2*head_dim)
        d = tid;
        while (d < head_dim) {
            let w = get_norm_weight(head_dim + d);
            wg_vals[d] = wg_vals[d] * rms * (1.0 + w);
            d += 256u;
        }
        workgroupBarrier();

        // Step 2: mRoPE on K
        let partial_half = PARTIAL_DIM / 2u;
        d = tid;
        while (d < partial_half) {
            let freq_idx = d;
            let a = wg_vals[d];
            let b = wg_vals[d + partial_half];
            let rotated = apply_mrope(a, b, freq_idx, partial_half);
            wg_vals[d] = rotated.x;
            wg_vals[d + partial_half] = rotated.y;
            d += 256u;
        }
        workgroupBarrier();

        // Step 3: Write K to k_proj (in-place) and k_cache
        let cache_off = params.cache_position * num_kv_heads * head_dim + kh * head_dim;
        d = tid;
        while (d < head_dim) {
            k_proj[kh * head_dim + d] = wg_vals[d];
            k_cache[cache_off + d] = wg_vals[d];
            d += 256u;
        }

        // Step 4: Write v_proj to v_cache (no norm/RoPE for V)
        d = tid;
        while (d < head_dim) {
            v_cache[cache_off + d] = v_proj[kh * head_dim + d];
            d += 256u;
        }
    }
}
