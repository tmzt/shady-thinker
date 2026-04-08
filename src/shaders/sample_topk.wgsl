// Combined penalty+temperature+gate+top-K shader.
//
// Single workgroup of 256 threads:
//   1. Each thread scans its strided vocab slice, applies penalties/temperature/gate,
//      and maintains a local top-K ascending heap (index 0 = min).
//   2. Local top-K written to workgroup shared memory (256 * K * 8 bytes).
//   3. Log-2 tree reduction merges pairs into global top-K.
//   4. Thread 0 writes topk_out[0..K] descending.
//
// gate_byte = 0  → no constraint.
// gate_byte != 0 → tokens whose first byte ≠ gate_byte are set to -inf.
//
// Dispatch: (1, 1, 1)

const K:       u32 = 8u;    // must match TOPK_K in model.rs
const WGSIZE:  u32 = 256u;
const NEG_INF: f32 = -3.402823e+38;

struct Candidate { idx: u32, val: f32 }

struct Params {
    vocab_size:       u32,
    rep_penalty:      f32,
    presence_penalty: f32,
    temperature:      f32,
    ban0: u32, ban1: u32, ban2: u32, ban3: u32,
    ban4: u32, ban5: u32,
    n_bans: u32,
    _pad:   u32,
    // 128-bit bitmap: bit N set means byte N is a valid first byte.
    // All-ones (default) = unconstrained.
    gate_w0: u32, gate_w1: u32, gate_w2: u32, gate_w3: u32,
}

@group(0) @binding(0) var<storage, read_write> logits:      array<f32>;
@group(0) @binding(1) var<storage, read>       seen_bitmap: array<u32>;
@group(0) @binding(2) var<uniform>             p:           Params;
@group(0) @binding(3) var<storage, read>       first_bytes: array<u32>;
@group(0) @binding(4) var<storage, read_write> topk_out:    array<Candidate>;

// 256 * 8 * 8 bytes = 16 KB
var<workgroup> wg: array<Candidate, 2048u>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let base = tid * K;

    // ── Phase 1: scan strided slice, apply penalties+gate, build local top-K ──
    // Heap stored ascending (index 0 = min so we can cheaply evict minimum).
    var t0: Candidate = Candidate(0u, NEG_INF);
    var t1: Candidate = Candidate(0u, NEG_INF);
    var t2: Candidate = Candidate(0u, NEG_INF);
    var t3: Candidate = Candidate(0u, NEG_INF);
    var t4: Candidate = Candidate(0u, NEG_INF);
    var t5: Candidate = Candidate(0u, NEG_INF);
    var t6: Candidate = Candidate(0u, NEG_INF);
    var t7: Candidate = Candidate(0u, NEG_INF);

    var pos = tid;
    while pos < p.vocab_size {
        var v = logits[pos];

        let word = seen_bitmap[pos >> 5u];
        let seen = (word >> (pos & 31u)) & 1u;
        if seen != 0u {
            if v > 0.0 { v /= p.rep_penalty; } else { v *= p.rep_penalty; }
            v -= p.presence_penalty;
        }

        if p.n_bans > 0u && pos == p.ban0 { v = NEG_INF; }
        if p.n_bans > 1u && pos == p.ban1 { v = NEG_INF; }
        if p.n_bans > 2u && pos == p.ban2 { v = NEG_INF; }
        if p.n_bans > 3u && pos == p.ban3 { v = NEG_INF; }
        if p.n_bans > 4u && pos == p.ban4 { v = NEG_INF; }
        if p.n_bans > 5u && pos == p.ban5 { v = NEG_INF; }

        // Gate: bitmap covers bytes 0-127 only.
        // AnyCharacter = all-ones ([0xFFFFFFFF; 4]) — no filtering.
        // JsonBitmap gate — block any token whose first byte is not in the bitmap:
        //   - ASCII byte (< 128): check the appropriate word/bit.
        //   - Multi-byte UTF-8 token (first byte >= 128): always blocked at a hard gate
        //     position, since structural chars ({, ", :) are all ASCII.
        let is_constrained = p.gate_w0 != 0xFFFFFFFFu || p.gate_w1 != 0xFFFFFFFFu
                          || p.gate_w2 != 0xFFFFFFFFu || p.gate_w3 != 0xFFFFFFFFu;
        if is_constrained {
            let fb = first_bytes[pos];
            if fb >= 128u {
                v = NEG_INF;
            } else {
                let bit_mask = 1u << (fb & 31u);
                var gate_word: u32;
                if      fb < 32u  { gate_word = p.gate_w0; }
                else if fb < 64u  { gate_word = p.gate_w1; }
                else if fb < 96u  { gate_word = p.gate_w2; }
                else               { gate_word = p.gate_w3; }
                if (gate_word & bit_mask) == 0u { v = NEG_INF; }
            }
        }

        v /= p.temperature;
        logits[pos] = v;

        // Insert into ascending heap: replace min (t0) if better, bubble up.
        if v > t0.val {
            t0 = Candidate(pos, v);
            var tmp: Candidate;
            if t0.val > t1.val { tmp = t0; t0 = t1; t1 = tmp; }
            if t1.val > t2.val { tmp = t1; t1 = t2; t2 = tmp; }
            if t2.val > t3.val { tmp = t2; t2 = t3; t3 = tmp; }
            if t3.val > t4.val { tmp = t3; t3 = t4; t4 = tmp; }
            if t4.val > t5.val { tmp = t4; t4 = t5; t5 = tmp; }
            if t5.val > t6.val { tmp = t5; t5 = t6; t6 = tmp; }
            if t6.val > t7.val { tmp = t6; t6 = t7; t7 = tmp; }
        }

        pos += WGSIZE;
    }

    // Write local top-K to workgroup memory
    wg[base + 0u] = t0; wg[base + 1u] = t1; wg[base + 2u] = t2; wg[base + 3u] = t3;
    wg[base + 4u] = t4; wg[base + 5u] = t5; wg[base + 6u] = t6; wg[base + 7u] = t7;
    workgroupBarrier();

    // ── Phase 2: tree reduction — merge partner's K into our K ──
    var stride = WGSIZE / 2u;
    while stride > 0u {
        if tid < stride {
            let pb = (tid + stride) * K;
            // Insert each of partner's K candidates into our heap
            for (var k = 0u; k < K; k++) {
                let c = wg[pb + k];
                if c.val > wg[base].val {
                    wg[base] = c;
                    // Bubble up within wg[base..base+K]
                    var tmp2: Candidate;
                    if wg[base+0u].val > wg[base+1u].val { tmp2 = wg[base+0u]; wg[base+0u] = wg[base+1u]; wg[base+1u] = tmp2; }
                    if wg[base+1u].val > wg[base+2u].val { tmp2 = wg[base+1u]; wg[base+1u] = wg[base+2u]; wg[base+2u] = tmp2; }
                    if wg[base+2u].val > wg[base+3u].val { tmp2 = wg[base+2u]; wg[base+2u] = wg[base+3u]; wg[base+3u] = tmp2; }
                    if wg[base+3u].val > wg[base+4u].val { tmp2 = wg[base+3u]; wg[base+3u] = wg[base+4u]; wg[base+4u] = tmp2; }
                    if wg[base+4u].val > wg[base+5u].val { tmp2 = wg[base+4u]; wg[base+4u] = wg[base+5u]; wg[base+5u] = tmp2; }
                    if wg[base+5u].val > wg[base+6u].val { tmp2 = wg[base+5u]; wg[base+5u] = wg[base+6u]; wg[base+6u] = tmp2; }
                    if wg[base+6u].val > wg[base+7u].val { tmp2 = wg[base+6u]; wg[base+6u] = wg[base+7u]; wg[base+7u] = tmp2; }
                }
            }
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // Thread 0 writes result descending (highest first)
    if tid == 0u {
        topk_out[0u] = wg[7u]; topk_out[1u] = wg[6u]; topk_out[2u] = wg[5u]; topk_out[3u] = wg[4u];
        topk_out[4u] = wg[3u]; topk_out[5u] = wg[2u]; topk_out[6u] = wg[1u]; topk_out[7u] = wg[0u];
    }
}
