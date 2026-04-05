// Apply repetition penalty, presence penalty, hard token bans, and temperature
// to the logits buffer in-place.
//
// Dispatch: (ceil(vocab_size / 256), 1, 1)

struct Params {
    vocab_size:       u32,
    rep_penalty:      f32,  // multiplicative repetition penalty (e.g. 1.0 = no-op)
    presence_penalty: f32,  // additive presence penalty (subtracted from seen tokens)
    temperature:      f32,  // divide logits by this
    // Up to 6 hard-banned token IDs (0xFFFFFFFFu = unused slot)
    ban0: u32, ban1: u32, ban2: u32, ban3: u32,
    ban4: u32, ban5: u32,
    n_bans: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> logits:     array<f32>;
@group(0) @binding(1) var<storage, read>       seen_bitmap: array<u32>;  // bit i set → token i has appeared
@group(0) @binding(2) var<uniform>             p:          Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= p.vocab_size { return; }

    var v = logits[idx];

    // Repetition / presence penalty for previously-seen tokens
    let word = seen_bitmap[idx >> 5u];
    let seen = (word >> (idx & 31u)) & 1u;
    if seen != 0u {
        if v > 0.0 {
            v /= p.rep_penalty;
        } else {
            v *= p.rep_penalty;
        }
        v -= p.presence_penalty;
    }

    // Hard bans (force to -inf)
    let NEG_INF = -3.402823e+38;
    if p.n_bans > 0u && idx == p.ban0 { v = NEG_INF; }
    if p.n_bans > 1u && idx == p.ban1 { v = NEG_INF; }
    if p.n_bans > 2u && idx == p.ban2 { v = NEG_INF; }
    if p.n_bans > 3u && idx == p.ban3 { v = NEG_INF; }
    if p.n_bans > 4u && idx == p.ban4 { v = NEG_INF; }
    if p.n_bans > 5u && idx == p.ban5 { v = NEG_INF; }

    // Temperature scaling
    v /= p.temperature;

    logits[idx] = v;
}
