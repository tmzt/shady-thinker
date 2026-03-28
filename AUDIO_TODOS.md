# Audio Pipeline TODOs

## GPU Decoder Early EOS
The GPU bf16 decoder stops generating too early (e.g. "Hello." instead of "Hello. This is a test."). The C NEON decoder produces full transcriptions from the same encoder output. Root cause: precision divergence between GPU f32 accumulation and C NEON bf16 kernels causes the model to predict EOS token prematurely.

Possible fixes:
- Compare softmax attention scores between GPU and C at the divergence point
- Try f32 weights for the final norm + lm_head (higher precision where it matters most)
- Investigate if C uses fused gate+up matmul (single GEMM vs two separate) — different rounding
- Try temperature > 0 or nucleus sampling instead of greedy argmax for decode
- Check if the GPU encoder output (max_diff=0.019 vs C) contributes — test with C encoder output only

## GPU Encoder → Decoder Shared Buffer
Currently encoder output goes GPU→CPU→GPU (readback f32 then re-upload as embeddings). Could keep the encoder output buffer on GPU and pass directly to decoder prefill. Zero-copy VRAM handoff.

## ~~Chunked Embedding for 128MB Binding Limit~~ DONE
Embedding split into 5 × ~120MB chunks. Lookup dispatches correct chunk per token ID. LM head iterates over chunks.

## INT4 Quantization for Android GPU Decoder
The 1.7B bf16 decoder needs ~2.7GB GPU memory + ~3.5GB CPU during safetensor loading = 6.9GB total. Android OOM kills at this level. INT4 GPTQ quantization would cut weights to ~400MB, making it feasible. The existing GPTQ matvec shaders could handle this — need quantized weights.

## Fused Gate+Up GEMM
C decoder uses a single fused `gate_up_fused_bf16` weight `[2*intermediate, hidden]` for one GEMM. GPU does two separate GEMMs (gate_proj + up_proj). Fusing would halve the GEMM dispatches in MLP and may improve precision alignment with C.

## Prefill KV Cache → Decode Handoff
Verify that the KV cache populated during batched prefill is correctly read by the single-token GQA attention during autoregressive decode. The batched_qknorm_rope shader writes to k_cache/v_cache with layout `[pos, num_kv_heads, head_dim]` — confirm this matches what gqa_attention_head.wgsl expects.
