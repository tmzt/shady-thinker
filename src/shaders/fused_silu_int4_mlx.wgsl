// Fused SiLU(gate) * up → INT4 MLX down_proj matvec.
// output[row] = sum_k( dequant(down[row,k]) * SiLU(gate[k]) * up[k] )
// Avoids temp buffer for SiLU intermediate.
//
// Dispatch: (ceil(out_dim / 32), 1, 1)

struct Params {
    in_dim: u32,       // intermediate_size (K)
    out_dim: u32,      // hidden_size (N)
    group_size: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> qweight: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read> biases: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg_id.x * 32u + lid.x;
    if (row >= params.out_dim) { return; }

    let in_dim = params.in_dim;
    let packed_cols = in_dim / 8u;
    let group_size = params.group_size;
    let n_groups = in_dim / group_size;
    let packed_per_group = group_size / 8u;

    let w_off = row * packed_cols;
    let sg_off = row * n_groups;

    var sum: f32 = 0.0;

    for (var g: u32 = 0u; g < n_groups; g++) {
        let sb_idx = sg_off + g;
        let scale = unpack2x16float(scales[sb_idx / 2u])[sb_idx & 1u];
        let bias = unpack2x16float(biases[sb_idx / 2u])[sb_idx & 1u];
        let g_start = g * packed_per_group;
        let ib = g * group_size;

        for (var p: u32 = 0u; p < packed_per_group; p++) {
            let packed = qweight[w_off + g_start + p];
            let base = ib + p * 8u;

            // Inline SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            for (var n: u32 = 0u; n < 8u; n++) {
                let nibble = (packed >> (n * 4u)) & 0xFu;
                let w = f32(nibble) * scale + bias;
                let gv = gate[base + n];
                let silu = gv / (1.0 + exp(-gv));
                sum += w * silu * up[base + n];
            }
        }
    }

    output[row] = sum;
}
