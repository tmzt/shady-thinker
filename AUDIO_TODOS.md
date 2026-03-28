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

## Chunked Embedding for 128MB Binding Limit (PowerVR)
The 1.7B embedding table is 584MB (151936 × 2048 × bf16). PowerVR DXT-48 has 128MB max_storage_buffer_binding. Need to split into 5 chunks (~120MB each) for both embedding lookup and lm_head dispatch. CPU fallback exists but is too slow for mobile.

## Fused Gate+Up GEMM
C decoder uses a single fused `gate_up_fused_bf16` weight `[2*intermediate, hidden]` for one GEMM. GPU does two separate GEMMs (gate_proj + up_proj). Fusing would halve the GEMM dispatches in MLP and may improve precision alignment with C.

## Prefill KV Cache → Decode Handoff
Verify that the KV cache populated during batched prefill is correctly read by the single-token GQA attention during autoregressive decode. The batched_qknorm_rope shader writes to k_cache/v_cache with layout `[pos, num_kv_heads, head_dim]` — confirm this matches what gqa_attention_head.wgsl expects.
