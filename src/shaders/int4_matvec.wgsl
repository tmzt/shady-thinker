// INT4 Asymmetric (minmax) Matrix-Vector Multiply.
// output[col] = sum over groups of (scale * int4_val + bias) * input[row]
// INT4 packed: 8 nibbles per u32, column-major layout.
// Scales and biases stored as f16 pairs (two per u32).
//
// Dequantization: val = nibble * scale + bias
// where scale = (max - min) / 15, bias = min
//
// Dispatch: (ceil(N / 32), 1, 1)

struct Params {
    K: u32,       // input dimension (rows)
    N: u32,       // output dimension (columns)
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;   // f16 packed: scale
@group(0) @binding(3) var<storage, read> biases: array<u32>;   // f16 packed: bias
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;
    if (col >= params.N) { return; }

    let K = params.K;
    let N = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;

    var sum: f32 = 0.0;

    var pr: u32 = 0u;
    while (pr < packed_rows) {
        let group = (pr * 8u) / group_size;
        let sf_idx = group * N + col;
        let s2 = unpack2x16float(scales[sf_idx / 2u]);
        let scale = select(s2.x, s2.y, (sf_idx & 1u) == 1u);
        let b2 = unpack2x16float(biases[sf_idx / 2u]);
        let bias = select(b2.x, b2.y, (sf_idx & 1u) == 1u);

        let packed = qweight[pr * N + col];
        let row_base = pr * 8u;

        sum += (f32((packed) & 0xFu) * scale + bias) * input[row_base];
        sum += (f32((packed >> 4u) & 0xFu) * scale + bias) * input[row_base + 1u];
        sum += (f32((packed >> 8u) & 0xFu) * scale + bias) * input[row_base + 2u];
        sum += (f32((packed >> 12u) & 0xFu) * scale + bias) * input[row_base + 3u];
        sum += (f32((packed >> 16u) & 0xFu) * scale + bias) * input[row_base + 4u];
        sum += (f32((packed >> 20u) & 0xFu) * scale + bias) * input[row_base + 5u];
        sum += (f32((packed >> 24u) & 0xFu) * scale + bias) * input[row_base + 6u];
        sum += (f32((packed >> 28u) & 0xFu) * scale + bias) * input[row_base + 7u];

        pr += 1u;
    }

    output[col] = sum;
}
