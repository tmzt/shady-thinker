// Causal GQA attention for prefill (batched Q/K/V, no KV cache).
// Uses flash-attention online softmax so it correctly handles any sequence length.
//
// Each workgroup handles one Q head at one query position.
// Each thread tid handles head_dim index tid (supports head_dim ≤ 256).
//
// Input: Q[seq_len, num_q_heads, head_dim], K[seq_len, num_kv_heads, head_dim],
//        V[seq_len, num_kv_heads, head_dim]
// Output: attn_output[seq_len, num_q_heads, head_dim]
//
// Dispatch: (num_q_heads, seq_len, 1)

struct Params {
    seq_len: u32,
    head_dim: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    heads_per_kv: u32,
}

@group(0) @binding(0) var<storage, read> q_proj: array<f32>;
@group(0) @binding(1) var<storage, read> k_proj: array<f32>;
@group(0) @binding(2) var<storage, read> v_proj: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Tile of K-position scores for the current tile (shared across threads).
// 256 scores = one tile of K positions processed per round.
var<workgroup> wg_tile_scores: array<f32, 256>;

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
    let causal_len = q_pos + 1u;  // attend to positions 0..q_pos (inclusive)
    let scale = 1.0 / sqrt(f32(head_dim));
    let out_base = q_pos * params.num_q_heads * head_dim + q_head * head_dim;

    // ── Online (flash-attention) softmax state ──
    // Each thread maintains its own running softmax state.
    // Thread tid handles head_dim index = tid (valid when tid < head_dim).
    var m: f32 = -1e30;  // running max
    var l: f32 = 0.0;    // running sum of exp(score - m)
    var o: f32 = 0.0;    // running weighted V sum for dimension tid

    let num_tiles = (causal_len + 255u) / 256u;

    for (var tile = 0u; tile < num_tiles; tile++) {
        // ── Step 1: each thread computes the Q·K score for K position (tile*256 + tid) ──
        let k_pos = tile * 256u + tid;
        var score: f32 = -1e30;  // sentinal for out-of-range positions
        if (k_pos < causal_len) {
            let k_base = k_pos * params.num_kv_heads * head_dim + kv_head * head_dim;
            var dot: f32 = 0.0;
            for (var d = 0u; d < head_dim; d += 1u) {
                dot += q_proj[q_base + d] * k_proj[k_base + d];
            }
            score = dot * scale;
        }
        wg_tile_scores[tid] = score;
        workgroupBarrier();  // all scores for this tile are ready

        // ── Step 2: update online softmax state using this tile's scores ──
        // Only threads with tid < head_dim update the V accumulator.
        // Threads with tid >= head_dim only contributed to score computation above.
        if (tid < head_dim) {
            let tile_kv_start = tile * 256u;
            let tile_end = min(256u, causal_len - tile_kv_start);
            for (var kk = 0u; kk < tile_end; kk += 1u) {
                let s = wg_tile_scores[kk];
                let m_new = max(m, s);
                let corr = exp(m - m_new);  // correction for previous accumulation
                let e_s  = exp(s - m_new);

                let kk_pos = tile_kv_start + kk;
                let v_base = kk_pos * params.num_kv_heads * head_dim + kv_head * head_dim;
                o = o * corr + e_s * v_proj[v_base + tid];
                l = l * corr + e_s;
                m = m_new;
            }
        }
        workgroupBarrier();  // ensure all threads finish consuming tile scores before next tile
    }

    // ── Write normalized output ──
    if (tid < head_dim) {
        output[out_base + tid] = select(0.0, o / l, l > 0.0);
    }
}
