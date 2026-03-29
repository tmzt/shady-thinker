// Store generated token to ring buffer and check for EOS.
// Reads token from argmax result, writes to token_ring, checks stop tokens.
// Dispatch: (1, 1, 1)
//
// Constants baked by build_store_token_eos_const():
//   TOKEN_ENDOFTEXT, TOKEN_IM_END, MIN_TOKENS_BEFORE_EOS

// const TOKEN_ENDOFTEXT: u32 = 151643u;  // baked by builder
// const TOKEN_IM_END: u32 = 151645u;     // baked by builder
// const MIN_TOKENS_BEFORE_EOS: u32 = 5u; // baked by builder

struct ArgmaxResult {
    idx: u32,
    val: f32,
}

@group(0) @binding(0) var<storage, read> argmax_result: ArgmaxResult;
@group(0) @binding(1) var<storage, read_write> eos_flag: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> token_ring: array<u32>;
@group(0) @binding(3) var<storage, read_write> token_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> logit_ring: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let tok = argmax_result.idx;
    let idx = atomicAdd(&token_count[0], 1u);
    token_ring[idx] = tok;
    logit_ring[idx] = argmax_result.val;
    if ((tok == TOKEN_ENDOFTEXT || tok == TOKEN_IM_END) && idx >= MIN_TOKENS_BEFORE_EOS) {
        atomicStore(&eos_flag[0], 1u);
    }
}
