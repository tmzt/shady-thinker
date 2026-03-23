
# Crayonz - With a Z

    Crayonz is what happens when you send an LLM to kindergarten, and it uses shaders or something. So it make sense.

This is an implementation of two projects on Hugging Face:

1. https://huggingface.co/spaces/Ex0bit/tensorbend
2. https://huggingface.co/Ex0bit/jit-lora

The second one has a paper:

https://huggingface.co/Ex0bit/jit-lora/blob/main/paper.pdf

And a Git repository:

https://github.com/eelbaz/jit-lora

This is a project to combine them using Rust+WGPU and the shaders. Since the original Tensorbend is a demo and obfuscated I am sticking with a standard implementation of the shaders with some inspiration from the SOTA techniques used in the project. Mostly the engineering choices around fusion.

Claude was able to document the required kernels sufficiently to implement them in WGSL.

The second project (jit-lora) is the other cool part. It's goal is to enable online learning using LoRA. This implementation is based on the paper and the slight differences in the actual code. Unlike the Github repository, this is a single process using just Rust+WGPU with some crates. There's no backend Express or Python component.
