// GPU top-K extraction via K sequential argmax passes.
//
// Single workgroup of 256 threads scans the full logits array in a strided loop.
// For each of K iterations:
//   1. Each thread finds its local max over its strided elements.
//   2. Tree reduction finds the global max index.
//   3. Thread 0 writes (idx, val) to topk_out[k] and masks that slot to -inf.
//
// After this dispatch, topk_out[0..K] holds the top-K candidates in descending order.
// Logits are destroyed (modified in-place); caller must reset before next use.
//
// Dispatch: (1, 1, 1)

const K: u32 = 20u;

struct Candidate {
    idx: u32,
    val: f32,
}

struct Params {
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> logits:   array<f32>;
@group(0) @binding(1) var<storage, read_write> topk_out: array<Candidate>;  // [K]
@group(0) @binding(2) var<uniform>             params:   Params;

var<workgroup> smem_val:  array<f32, 256>;
var<workgroup> smem_gidx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n   = params.vocab_size;

    for (var k = 0u; k < K; k++) {
        // ── Step 1: each thread finds its local max over a strided range ──
        var local_max: f32 = -3.402823e+38;
        var local_idx: u32 = 0u;
        var pos = tid;
        while (pos < n) {
            let v = logits[pos];
            if v > local_max {
                local_max = v;
                local_idx = pos;
            }
            pos += 256u;
        }
        smem_val[tid]  = local_max;
        smem_gidx[tid] = local_idx;
        workgroupBarrier();

        // ── Step 2: tree reduction to find global max ──
        var stride = 128u;
        while stride > 0u {
            if tid < stride {
                let other = tid + stride;
                if smem_val[other] > smem_val[tid] {
                    smem_val[tid]  = smem_val[other];
                    smem_gidx[tid] = smem_gidx[other];
                }
            }
            workgroupBarrier();
            stride >>= 1u;
        }

        // ── Step 3: thread 0 records winner and masks it out ──
        if tid == 0u {
            topk_out[k] = Candidate(smem_gidx[0], smem_val[0]);
            logits[smem_gidx[0]] = -3.402823e+38;
        }
        // Ensure the storage write is visible to all threads before the next iteration.
        storageBarrier();
        workgroupBarrier();
    }
}
