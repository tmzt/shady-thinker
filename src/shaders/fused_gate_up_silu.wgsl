// Fused gate + up GPTQ matvec with SiLU activation.
// output[col] = SiLU(gate_sum) * up_sum
// where gate_sum = dot(input, dequant(W_gate[:, col])), up_sum = dot(input, dequant(W_up[:, col]))
// Computes both projections in one kernel, avoids two separate matvec + SiLU + multiply.
// INT4 quantized weights: 8 nibbles per u32. Scales stored as f16 pairs via unpack2x16float.
// Dispatch: (ceil(N / 32), 1, 1)

struct Params {
    K: u32,
    N: u32,
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight_gate: array<u32>;
@group(0) @binding(2) var<storage, read> scales_gate: array<u32>;
@group(0) @binding(3) var<storage, read> qweight_up: array<u32>;
@group(0) @binding(4) var<storage, read> scales_up: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wg_id.x * 32u + lid.x;
    if (col >= params.N) {
        return;
    }

    let K = params.K;
    let N = params.N;
    let group_size = params.group_size;
    let packed_rows = K / 8u;

    var gate_sum: f32 = 0.0;
    var up_sum: f32 = 0.0;

    // Unroll by 4 packed rows at a time
    let unroll_end = packed_rows & ~3u;

    var pr: u32 = 0u;
    while (pr < unroll_end) {
        for (var j: u32 = 0u; j < 4u; j++) {
            let cur_pr = pr + j;
            let group = (cur_pr * 8u) / group_size;
            let sf = group * N + col;

            let sg2 = unpack2x16float(scales_gate[sf / 2u]);
            let scale_g = select(sg2.x, sg2.y, (sf & 1u) == 1u);
            let packed_g = qweight_gate[cur_pr * N + col];

            let su2 = unpack2x16float(scales_up[sf / 2u]);
            let scale_u = select(su2.x, su2.y, (sf & 1u) == 1u);
            let packed_u = qweight_up[cur_pr * N + col];

            let row_base = cur_pr * 8u;
            for (var n: u32 = 0u; n < 8u; n++) {
                let shift = n * 4u;
                let dq_g = (f32((packed_g >> shift) & 0xFu) - 8.0) * scale_g;
                let dq_u = (f32((packed_u >> shift) & 0xFu) - 8.0) * scale_u;
                let inp = input[row_base + n];
                gate_sum += dq_g * inp;
                up_sum += dq_u * inp;
            }
        }
        pr += 4u;
    }

    // Handle remaining packed rows
    while (pr < packed_rows) {
        let group = (pr * 8u) / group_size;
        let sf = group * N + col;

        let sg2 = unpack2x16float(scales_gate[sf / 2u]);
        let scale_g = select(sg2.x, sg2.y, (sf & 1u) == 1u);
        let packed_g = qweight_gate[pr * N + col];

        let su2 = unpack2x16float(scales_up[sf / 2u]);
        let scale_u = select(su2.x, su2.y, (sf & 1u) == 1u);
        let packed_u = qweight_up[pr * N + col];

        let row_base = pr * 8u;
        for (var n: u32 = 0u; n < 8u; n++) {
            let shift = n * 4u;
            let dq_g = (f32((packed_g >> shift) & 0xFu) - 8.0) * scale_g;
            let dq_u = (f32((packed_u >> shift) & 0xFu) - 8.0) * scale_u;
            let inp = input[row_base + n];
            gate_sum += dq_g * inp;
            up_sum += dq_u * inp;
        }

        pr += 1u;
    }

    output[col] = silu(gate_sum) * up_sum;
}
