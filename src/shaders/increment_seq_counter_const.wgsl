// Increment the GPU-side sequence counter by 1.
// Dispatched once per token before the layer loop.
// Dispatch: (1, 1, 1)

@group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    atomicAdd(&counter[0], 1u);
}
