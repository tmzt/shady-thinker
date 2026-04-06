// GPTQ INT4 Matrix-Matrix Multiply: output[pos, col] = sum over groups of (scale * sum(dequant(qweight) * input[pos,:]))
// Extends gptq_matvec with a sequence (batch) dimension in workgroup_id.y.
// Weights have no sequence dimension — same quantized matrix applied to every input position.
// INT4 quantized weights: 8 nibbles per u32. Scales stored as f16 pairs via unpack2x16float.
// Dispatch: (ceil(N / 32), S, 1)  where S = sequence length

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;  // [S, K]
@group(0) @binding(1) var<storage, read>       qweight: array<u32>; // [K/8, N]
@group(0) @binding(2) var<storage, read>       scales:  array<u32>; // [K/group_size, N] packed f16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;  // [S, N]
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:   vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;
    let pos = wg_id.y;   // sequence position
    if (col >= params.N) { return; }

    let K          = params.K;
    let N          = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;
    let in_base    = pos * K;
    let out_base   = pos * N;

    var sum: f32 = 0.0;

    // Unroll by 4 packed rows at a time
    let unroll_end = packed_rows & ~3u;

    var pr: u32 = 0u;
    while (pr < unroll_end) {
        for (var j: u32 = 0u; j < 4u; j++) {
            let cur_pr = pr + j;
            let group = (cur_pr * 8u) / group_size;
            let sf    = group * N + col;
            let s2    = unpack2x16float(scales[sf / 2u]);
            let scale = select(s2.x, s2.y, (sf & 1u) == 1u);
            let packed = qweight[cur_pr * N + col];

            let row_base = in_base + cur_pr * 8u;
            sum += (f32((packed)       & 0xFu) - 8.0) * scale * input[row_base];
            sum += (f32((packed >> 4u) & 0xFu) - 8.0) * scale * input[row_base + 1u];
            sum += (f32((packed >> 8u) & 0xFu) - 8.0) * scale * input[row_base + 2u];
            sum += (f32((packed >> 12u) & 0xFu) - 8.0) * scale * input[row_base + 3u];
            sum += (f32((packed >> 16u) & 0xFu) - 8.0) * scale * input[row_base + 4u];
            sum += (f32((packed >> 20u) & 0xFu) - 8.0) * scale * input[row_base + 5u];
            sum += (f32((packed >> 24u) & 0xFu) - 8.0) * scale * input[row_base + 6u];
            sum += (f32((packed >> 28u) & 0xFu) - 8.0) * scale * input[row_base + 7u];
        }
        pr += 4u;
    }

    // Handle remaining packed rows
    while (pr < packed_rows) {
        let group = (pr * 8u) / group_size;
        let sf    = group * N + col;
        let s2    = unpack2x16float(scales[sf / 2u]);
        let scale = select(s2.x, s2.y, (sf & 1u) == 1u);
        let packed = qweight[pr * N + col];

        let row_base = in_base + pr * 8u;
        sum += (f32((packed)       & 0xFu) - 8.0) * scale * input[row_base];
        sum += (f32((packed >> 4u) & 0xFu) - 8.0) * scale * input[row_base + 1u];
        sum += (f32((packed >> 8u) & 0xFu) - 8.0) * scale * input[row_base + 2u];
        sum += (f32((packed >> 12u) & 0xFu) - 8.0) * scale * input[row_base + 3u];
        sum += (f32((packed >> 16u) & 0xFu) - 8.0) * scale * input[row_base + 4u];
        sum += (f32((packed >> 20u) & 0xFu) - 8.0) * scale * input[row_base + 5u];
        sum += (f32((packed >> 24u) & 0xFu) - 8.0) * scale * input[row_base + 6u];
        sum += (f32((packed >> 28u) & 0xFu) - 8.0) * scale * input[row_base + 7u];

        pr += 1u;
    }

    output[out_base + col] = sum;
}
