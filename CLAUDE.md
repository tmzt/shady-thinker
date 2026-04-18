
## Shader rules

1. **No conditionals in shaders.** All branching is done on the CPU side by selecting the correct shader variant. If a model needs different behavior (e.g. with vs without QK norm), write a separate shader file, not a branch in the existing one.

2. **Bake constants via `_build` functions.** Each shader file has hardcoded `const` lines at the top (e.g. `const ROPE_THETA: f32 = 1000000.0;`). The corresponding `build_*_shader()` function in `model.rs` skips those lines and prepends the real values. This is the only acceptable form of shader parameterization.

3. **Replace full const lines, never patch values.** When building shader source, always skip the entire const declaration line and inject a new one. Never do string replacement of a value within a line (e.g. replacing `8.0` with `7.0` inside shader source).

4. **Verify against the reference implementation before debugging intermediate values.** When a model produces wrong output, read the HuggingFace transformers `modeling_*.py` for that architecture first. Compare the reference forward pass (norm convention, RoPE, attention, residual connections) with ours before dumping GPU buffers. Most bugs are architectural mismatches, not numerical issues.

5. **Use explicit metadata from config.json, not heuristics.** Model-specific behavior (norm convention, gated attention, QK norm presence) must be driven by fields in `config.json`, not by guessing from weight shapes or tensor names.

## Architecture

1. Implement the shaders described in SHADER_RESEARCH.md to target Rust+WGPU+naga. Each one should be a separate wgsl file with the final component of the base filename being the data size or dtype if relevant. (such as _f32 or _hf16). Assume they may be included to construct a larger shader package.

2. Implement a Qwen3.5 LLM architecture using the shaders and the reference code. Include LoRA support and support the following algorithm for online training. Make this optional and gated by a feature. When the feature is enabled there should be zero cost from the code.

[JIT-LORA]
https://huggingface.co/Ex0bit/jit-lora/blob/main/paper.pdf
https://github.com/eelbaz/jit-lora
./JIT_LORA.pdf
