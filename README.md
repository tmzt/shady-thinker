# Shady Thinker

GPU-accelerated LLM inference and online learning in pure Rust, powered by WGPU shaders.

## What is this?

Shady Thinker is a from-scratch implementation of LLM inference and fine-tuning using Rust + WGPU. No Python, no CUDA — just shaders doing the thinking.

Unlike the reference projects, this is a single-process Rust binary with no Python or Node backend.

## Inspiration

- [TensorBend](https://huggingface.co/spaces/Ex0bit/tensorbend) — a Hugging Face demo showcasing GPU shader-based tensor operations. The original is obfuscated, but it demonstrated that full LLM inference through compute shaders is viable.
- [JIT-LoRA](https://huggingface.co/Ex0bit/jit-lora/blob/main/paper.pdf) ([code](https://github.com/eelbaz/jit-lora)) — a paper and reference implementation for runtime LoRA adaptation, enabling online learning without a separate training pipeline.

## Methodology

The shader kernels were developed from descriptions of the required operations rather than ported line-by-line from the reference code. Claude was used to document the necessary kernels (matmul, softmax, RMSNorm, RoPE, etc.) at a functional level, and each was then implemented as a standalone WGSL compute shader targeting the WGPU/naga toolchain.

This approach — rewriting from a spec rather than translating source — avoids inheriting obfuscation or framework-specific patterns from the originals, and produces shaders that compose cleanly as building blocks for the full inference pipeline.

Engineering choices around kernel fusion are informed by the SOTA techniques used in TensorBend.

## Features

- WGSL compute shaders for matmul, softmax, RMSNorm, RoPE, and more
- Feature-gated modules: `chat` for tokenizer support
- Designed for portability across any GPU backend WGPU supports (Vulkan, Metal, DX12)

## Building

```sh
# Core library only
cargo build

# With chat/tokenizer support
cargo build --features chat
```

## Future Work

- **JIT-LoRA online learning** — runtime LoRA adaptation based on the [JIT-LoRA paper](https://huggingface.co/Ex0bit/jit-lora/blob/main/paper.pdf), enabling on-the-fly fine-tuning during inference. This is gated behind the `jit-lora` feature flag and is currently a work in progress.
  ```sh
  cargo build --features jit-lora
  ```

## License

[MIT](LICENSE)
