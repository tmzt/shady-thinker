// Tiled BF16 Matrix-Vector Multiply with shared memory input caching.
// output[row] = sum_i(weight[row, i] * input[i])
// Weights are BF16 packed (two per u32). Input is f32, loaded into shared memory.
//
// Each workgroup: 32 threads compute 32 output rows.
// Input vector is loaded in tiles into shared memory — read once, used by all 32 threads.
//
// Dispatch: (ceil(vocab_size / 32), 1, 1)

struct Params {
    hidden_size: u32,
    vocab_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_K: u32 = 64u;  // elements per tile (must be <= workgroup_size * 2)
var<workgroup> shared_input: array<f32, 64>;

fn unpack_bf16(packed: u32, idx: u32) -> f32 {
    let bits = (packed >> (idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x * 32u + lid.x;
    let hidden_size = params.hidden_size;
    let half_hidden = hidden_size / 2u;

    var sum: f32 = 0.0;

    // Process input in tiles of TILE_K elements
    var tile_start: u32 = 0u;
    while (tile_start < hidden_size) {
        let tile_end = min(tile_start + TILE_K, hidden_size);
        let tile_len = tile_end - tile_start;

        // Cooperatively load input tile into shared memory
        // Each of 32 threads loads 2 elements (covers 64 elements)
        let load_idx = lid.x * 2u;
        if (load_idx < tile_len) {
            shared_input[load_idx] = input[tile_start + load_idx];
        }
        if (load_idx + 1u < tile_len) {
            shared_input[load_idx + 1u] = input[tile_start + load_idx + 1u];
        }
        workgroupBarrier();

        // Each thread accumulates its row's dot product with the shared tile
        if (row < params.vocab_size) {
            let weight_base = row * half_hidden + tile_start / 2u;
            let packed_tile = tile_len / 2u;

            var i: u32 = 0u;
            // Unroll by 4 packed values (8 elements)
            let unroll_end = packed_tile & ~3u;
            while (i < unroll_end) {
                let p0 = weight[weight_base + i];
                let p1 = weight[weight_base + i + 1u];
                let p2 = weight[weight_base + i + 2u];
                let p3 = weight[weight_base + i + 3u];
                let si = i * 2u;

                sum += unpack_bf16(p0, 0u) * shared_input[si];
                sum += unpack_bf16(p0, 1u) * shared_input[si + 1u];
                sum += unpack_bf16(p1, 0u) * shared_input[si + 2u];
                sum += unpack_bf16(p1, 1u) * shared_input[si + 3u];
                sum += unpack_bf16(p2, 0u) * shared_input[si + 4u];
                sum += unpack_bf16(p2, 1u) * shared_input[si + 5u];
                sum += unpack_bf16(p3, 0u) * shared_input[si + 6u];
                sum += unpack_bf16(p3, 1u) * shared_input[si + 7u];

                i += 4u;
            }
            while (i < packed_tile) {
                let p = weight[weight_base + i];
                let si = i * 2u;
                sum += unpack_bf16(p, 0u) * shared_input[si];
                sum += unpack_bf16(p, 1u) * shared_input[si + 1u];
                i += 1u;
            }
        }
        workgroupBarrier();

        tile_start += TILE_K;
    }

    if (row < params.vocab_size) {
        output[row] = sum;
    }
}
