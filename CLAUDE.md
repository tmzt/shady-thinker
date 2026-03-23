

1. Implement the shaders described in SHADER_RESEARCH.md to target Rust+WGPU+naga. Each one should be a separate wgsl file with the final component of the base filename being the data size or dtype if relevant. (such as _f32 or _hf16). Assume they may be included to construct a larger shader package.

2. Implement a Qwen3.5 LLM architecture using the shaders and the reference code. Include LoRA support and support the following algorithm for online training. Make this optional and gated by a feature. When the feature is enabled there should be zero cost from the code.

[JIT-LORA]
https://huggingface.co/Ex0bit/jit-lora/blob/main/paper.pdf
https://github.com/eelbaz/jit-lora
./JIT_LORA.pdf
