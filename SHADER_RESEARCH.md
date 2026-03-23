# Shader Specifications — Standard & Fused Operations

Complete mathematical specifications sufficient to reimplement from scratch.
Only covers standard (textbook) and fused (standard ops combined) operations.
ParoQuant and DeltaNet shaders excluded — see end of document for notes.

## Data Format Conventions

### BF16 Packing
Two BF16 values stored per u32. Unpack:
```
bits = (packed >> (idx * 16)) & 0xFFFF
value = bitcast<f32>(bits << 16)
```

### INT4 Quantization (GPTQ)
Eight 4-bit values packed per u32 (little-endian nibble order).
Dequantize: `value = f32((packed >> (nibble * 4)) & 0xF) - 8.0`
Range: 0-15 mapped to -8..+7. Per-group scales stored as f16 pairs (2 per u32).

### Tree Reduction
Workgroup-local parallel reduction: stride starts at workgroup_size/2, halves each step with barrier between steps.

---

## Standard Operations

### 1. add
**Math**: `a[i] += b[i]`

| Field | Value |
|-------|-------|
| Params | `N: u32` |
| Bindings | 0: `a` f32 read_write, 1: `b` f32 read, 2: params |
| Workgroup | 256 |
| Dispatch | (ceil(N/256), 1, 1) |

---

### 2. rmsnorm
**Math**: `output[i] = input[i] * (1/sqrt(mean(input^2) + eps)) * (1 + weight[i])`

| Field | Value |
|-------|-------|
| Params | `N: u32, eps: f32` |
| Bindings | 0: `input` f32 read, 1: `weight` BF16-packed read, 2: `output` f32 read_write, 3: params |
| Workgroup | 256 |
| Dispatch | (1, 1, 1) — single workgroup |
| Reduction | Tree reduce for sum of squares |

Weight uses `(1 + w)` scaling (not bare `w`).

---

### 3. embedding
**Math**: `output[i] = embeddings[token_id * dim + i]` (BF16 unpacked)

| Field | Value |
|-------|-------|
| Params | `token_id: u32, dim: u32` |
| Bindings | 0: `embeddings` BF16-packed read, 1: `output` f32 read_write, 2: params |
| Workgroup | 256 |
| Dispatch | (ceil(dim/256), 1, 1) |

---

### 4. argmax
**Math**: Find `(idx, val)` where `val = max(logits[0..N])`

| Field | Value |
|-------|-------|
| Params | `N: u32` |
| Bindings | 0: `logits` f32 read, 1: `result {idx: u32, val: f32}` read_write, 2: params |
| Workgroup | 256 |
| Dispatch | (1, 1, 1) — single workgroup |
| Reduction | Tree reduce tracking both max value and index |

---

### 5. silu_mul
**Math**: `output[i] = SiLU(gate[i]) * up[i]` where `SiLU(x) = x / (1 + exp(-x))`

| Field | Value |
|-------|-------|
| Params | `N: u32` |
| Bindings | 0: `gate` f32 read, 1: `up` f32 read, 2: `output` f32 read_write, 3: params |
| Workgroup | 256 |
| Dispatch | (ceil(N/256), 1, 1) |

---

### 6. sigmoid_mul
**Math**: `output[i] = x[i] * sigmoid(gate[i])` where `sigmoid(x) = 1 / (1 + exp(-x))`

| Field | Value |
|-------|-------|
| Params | `N: u32` |
| Bindings | 0: `x` f32 read, 1: `gate` f32 read, 2: `output` f32 read_write, 3: params |
| Workgroup | 256 |
| Dispatch | (ceil(N/256), 1, 1) |

---

### 7. bf16_matvec
**Math**: `output[row] = sum_i(weight[row, i] * input[i])` with BF16 weights

| Field | Value |
|-------|-------|
| Params | `hidden_size: u32, vocab_size: u32` |
| Bindings | 0: `input` f32 read [hidden_size], 1: `weight` BF16-packed read [vocab_size * hidden_size / 2], 2: `output` f32 read_write [vocab_size], 3: params |
| Workgroup | 32 |
| Dispatch | (ceil(vocab_size/32), 1, 1) |
| Loop unroll | 4 elements per iteration |

Used for tied-embedding LM head projection.

---

### 8. gptq_matvec
**Math**: `output[col] = sum_groups(scale_g * sum_rows(dequant(qweight) * input))`

| Field | Value |
|-------|-------|
| Params | `K: u32, N: u32, group_size: u32` |
| Bindings | 0: `input` f32 read [K], 1: `qweight` u32 read [K/8 * N], 2: `scales` u32 read (f16 pairs), 3: `output` f32 read_write [N], 4: params |
| Workgroup | 32 |
| Dispatch | (ceil(N/32), 1, 1) |

Layout: `qweight[packed_row * N + col]`, 8 INT4 values per u32. Scales indexed as `scales[(group * N + col) / 2]`, unpacked via `unpack2x16float`. Inner loop processes 4 packed rows at a time (32 input elements).

---

### 9. gptq_matvec_4t
**Math**: Same as gptq_matvec with 4-thread lane parallelism.

| Field | Value |
|-------|-------|
| Params | `K: u32, N: u32, group_size: u32` |
| Bindings | Same as gptq_matvec |
| Workgroup | 32 |
| Dispatch | (ceil(N/8), 1, 1) — 8 columns per workgroup |

Each 4-thread lane processes one column. Lane `tid & 3` handles 1/4 of the groups. Final reduction via workgroup scratch.

---

### 10. gqa_attention_head
**Math**: Online softmax attention over KV cache.

| Field | Value |
|-------|-------|
| Params | `seq_len: u32, head_dim: u32, num_kv_heads: u32, num_q_heads: u32, heads_per_kv: u32, num_splits: u32` |
| Bindings | 0: `q` f32 read, 1: `k_cache` f32 read, 2: `v_cache` f32 read, 3: `output` f32 read_write, 4: params |
| Workgroup | 256 |
| Dispatch | (num_q_heads, num_splits, 1) |

```
scale = 1 / sqrt(head_dim)
running_max = -inf, running_sum = 0, acc = 0
for each position in cache:
    score = dot(q, k[pos]) * scale           // tree-reduced
    new_max = max(running_max, score)
    correction = exp(running_max - new_max)
    exp_score = exp(score - new_max)
    running_sum = running_sum * correction + exp_score
    acc = acc * correction + exp_score * v[pos]
    running_max = new_max
output = acc / running_sum
```

For multi-split: writes `(partial_out, log_sum_exp, max)` per split.

---

### 11. gqa_reduce
**Math**: Merge attention splits via log-sum-exp weighting.

| Field | Value |
|-------|-------|
| Params | `head_dim: u32, num_splits: u32, num_heads: u32` |
| Bindings | 0: `partials` f32 read [(head_dim+2) * num_splits * num_heads], 1: `output` f32 read_write, 2: params |
| Workgroup | 256 |
| Dispatch | (num_heads, 1, 1) |

```
global_max_lse = max over splits of lse[s]
for each split:
    w = exp(lse[s] - global_max_lse)
    acc += w * partial_out[s]
    weight_sum += w
output = acc / weight_sum
```

---

### 12. lora_down
**Math**: `h[r] = sum_i(x[i] * A[i * rank + r])` — down-projection to low rank.

| Field | Value |
|-------|-------|
| Params | `in_features: u32, rank: u32` |
| Bindings | 0: `input` f32 read [in_features], 1: `lora_a` f32 read [in_features * rank], 2: `output` f32 read_write [rank], 3: params |
| Workgroup | 32 |
| Dispatch | (ceil(rank/32), 1, 1) |
| Loop unroll | 4 elements per iteration |

A matrix layout: `[in_features, rank]` row-major.

---

### 13. grad_b
**Math**: `grad_b[r * out_f + c] += h[r] * grad_y[c] * scale` — outer product.

| Field | Value |
|-------|-------|
| Params | `rank: u32, out_features: u32, scale: f32` |
| Bindings | 0: `hidden` f32 read [rank], 1: `grad_y` f32 read [out_features], 2: `grad_b` f32 read_write [rank * out_features], 3: params |
| Workgroup | 32 |
| Dispatch | (ceil(out_features/32), rank, 1) — X=columns, Y=rank |

---

### 14. grad_a
**Math**: Backprop through B then outer product with input.

| Field | Value |
|-------|-------|
| Params | `in_features: u32, rank: u32, out_features: u32, scale: f32` |
| Bindings | 0: `input_x` f32 read [in_features], 1: `grad_y` f32 read [out_features], 2: `lora_b` f32 read [rank * out_features], 3: `grad_a` f32 read_write [in_features * rank], 4: params |
| Workgroup | 32 |
| Dispatch | (ceil(in_features/32), rank, 1) — X=input features, Y=rank |

```
dh[r] = sum_c(grad_y[c] * B[r * out_f + c]) * scale
grad_a[feat * rank + r] += x[feat] * dh[r]
```

---

### 15. adam_update
**Math**: Bias-corrected Adam with gradient clipping.

| Field | Value |
|-------|-------|
| Params | `num_elements: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, beta1_t: f32, beta2_t: f32, grad_clip: f32` |
| Bindings | 0: `param` f32 read_write, 1: `grad` f32 read, 2: `m` f32 read_write (1st moment), 3: `v` f32 read_write (2nd moment), 4: params |
| Workgroup | 256 |
| Dispatch | (ceil(num_elements/256), 1, 1) |

```
g = clamp(grad[i], -grad_clip, grad_clip)
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

---

### 16. cross_entropy_grad
**Math**: Numerically stable softmax gradient + scalar loss.

| Field | Value |
|-------|-------|
| Params | `vocab_size: u32, target_token: u32` |
| Bindings | 0: `logits` f32 read [vocab_size], 1: `grad_out` f32 read_write [vocab_size], 2: `loss_out` f32 read_write [1], 3: params |
| Workgroup | 256 |
| Dispatch | (1, 1, 1) — single workgroup |

```
max_val = max(logits)                              // tree reduce
sum_exp = sum(exp(logits[i] - max_val))             // tree reduce
grad_out[i] = exp(logits[i] - max_val) / sum_exp    // softmax
grad_out[target] -= 1.0                             // - one_hot
loss = -(logits[target] - max_val - log(sum_exp))    // cross-entropy
```

---

### 17. grad_fma
**Math**: `grad[i] += lambda * anchor[i]` — EWC gradient blending.

| Field | Value |
|-------|-------|
| Params | `num_elements: u32, lambda: f32` |
| Bindings | 0: `grad` f32 read_write, 1: `anchor` f32 read, 2: params |
| Workgroup | 256 |
| Dispatch | (ceil(num_elements/256), 1, 1) |

---

## Fused Operations

### 18. add_rmsnorm
**Math**: `hidden[i] += addend[i]` then RMSNorm on the result.

| Field | Value |
|-------|-------|
| Params | `N: u32, eps: f32` |
| Bindings | 0: `hidden` f32 read_write [N], 1: `addend` f32 read [N], 2: `weight` BF16-packed read, 3: `output` f32 read_write [N], 4: params |
| Workgroup | 256 |
| Dispatch | (1, 1, 1) — single workgroup |

```
hidden[i] = hidden[i] + addend[i]     // in-place add
rms = 1 / sqrt(mean(hidden^2) + eps)   // tree reduce
output[i] = hidden[i] * rms * (1 + w[i])
```

Saves one pass over the data vs separate add + rmsnorm.

---

### 19. three_way_add_rmsnorm
**Math**: Sum three inputs, write raw sum and normalized result.

| Field | Value |
|-------|-------|
| Params | `N: u32, eps: f32` |
| Bindings | 0: `a` f32 read, 1: `b` f32 read, 2: `c` f32 read, 3: `weight` BF16-packed read, 4: `hidden_out` f32 read_write (raw sum), 5: `normed` f32 read_write (normalized), 6: params |
| Workgroup | 256, workgroup memory 7680 bytes |
| Dispatch | (1, 1, 1) — single workgroup |

```
val = a[i] + b[i] + c[i]
hidden_out[i] = val
normed[i] = val * rms * (1 + w[i])
```

---

### 20. fused_silu_gptq
**Math**: `output[col] = sum(SiLU(a) * b * dequant(W))` — SiLU activation fused into GPTQ matvec.

| Field | Value |
|-------|-------|
| Params | `K: u32, N: u32, group_size: u32` |
| Bindings | 0: `a` f32 read [K], 1: `b` f32 read [K], 2: `qweight` u32 read, 3: `scales` u32 read, 4: `output` f32 read_write [N], 5: params |
| Workgroup | 32 |
| Dispatch | (ceil(N/32), 1, 1) |

Identical to gptq_matvec but the dequant inner product applies `SiLU(a[k]) * b[k]` instead of plain `input[k]`. Avoids materializing the `SiLU(a)*b` intermediate buffer.

---

### 21. fused_silu_gptq_4t
Same as fused_silu_gptq with 4-thread lane parallelism.
Dispatch: (ceil(N/8), 1, 1).

---

### 22. fused_gate_up_silu
**Math**: Dual GPTQ matvec for gate and up projections, then `SiLU(gate_sum) * up_sum`.

| Field | Value |
|-------|-------|
| Params | `K: u32, N: u32, group_size: u32` |
| Bindings | 0: `input` f32 read [K], 1: `qweight_gate` u32 read, 2: `scales_gate` u32 read, 3: `qweight_up` u32 read, 4: `scales_up` u32 read, 5: `output` f32 read_write [N], 6: params |
| Workgroup | 32 |
| Dispatch | (ceil(N/32), 1, 1) |

```
for each col:
    gate_sum = input @ dequant(W_gate[:, col])
    up_sum   = input @ dequant(W_up[:, col])
    output[col] = SiLU(gate_sum) * up_sum
```

Computes both projections in one kernel, avoids two separate matvec + SiLU + multiply.

---

### 23. fused_gate_up_silu_4t
Same as fused_gate_up_silu with 4-thread lane parallelism.
Dispatch: (ceil(N/8), 1, 1). Uses workgroup scratch [64] for gate/up partial sums.

---

### 24. fused_split_qknorm_kvstore
**Math**: Q/K RMSNorm + Q/gate split + mRoPE positional encoding + KV cache write.

| Field | Value |
|-------|-------|
| Params | `num_heads: u32, num_kv_heads: u32, head_dim: u32, eps: f32, cache_position: u32, position: u32, position_h: u32, position_w: u32, qk_norm_weight: array<vec4<u32>, 320>` |
| Bindings | 0: `q_proj_full` f32 read (interleaved Q+gate), 1: `k_proj` f32 read_write, 2: `v_proj` f32 read, 3: `q_proj` f32 read_write (output Q), 4: `q_gate` f32 read_write, 5: `k_cache` f32 read_write, 6: `v_cache` f32 read_write, 7: params |
| Workgroup | 256 |
| Dispatch | (num_heads + num_kv_heads, 1, 1) |

Fuses 5 operations:
1. **Split** interleaved Q+gate into separate Q and gate buffers
2. **RMSNorm** on Q (per-head) and K (per-head)
3. **mRoPE** positional encoding (multi-dimensional RoPE with theta=10^7)
4. **KV cache write** at `cache_position`

mRoPE frequency selection:
```
freq_idx = i % (partial_dim / 2)
freq = 1.0 / pow(theta, 2 * freq_idx / partial_dim)
position = select based on freq_idx:
  - default: position
  - freq_idx % 3 == 1 && < S1_LIMIT: position_h
  - freq_idx % 3 == 2 && < S2_LIMIT: position_w
angle = position * freq
[x, y] -> [x*cos - y*sin, y*cos + x*sin]  // rotation
```

---

### 25. lora_up_add
**Math**: `output[c] += scale * sum_r(h[r] * B[r * out_f + c])` — up-projection with residual add.

| Field | Value |
|-------|-------|
| Params | `rank: u32, out_features: u32, scale: f32` |
| Bindings | 0: `hidden` f32 read [rank], 1: `lora_b` f32 read [rank * out_features], 2: `output` f32 read_write [out_features], 3: params |
| Workgroup | 32 |
| Dispatch | (ceil(out_features/32), 1, 1) |

Fuses h @ B multiplication with `+=` accumulation into existing output.

---

### 26. lora_down_silu
**Math**: `h[r] = sum_i(SiLU(gate[i]) * up[i] * A[i * rank + r])` — fused SiLU activation + down-projection.

| Field | Value |
|-------|-------|
| Params | `in_features: u32, rank: u32` |
| Bindings | 0: `gate` f32 read [in_features], 1: `up` f32 read [in_features], 2: `lora_a` f32 read [in_features * rank], 3: `output` f32 read_write [rank], 4: params |
| Workgroup | 32 |
| Dispatch | (ceil(rank/32), 1, 1) |
| Loop unroll | 4 elements per iteration |

Avoids materializing the `SiLU(gate) * up` intermediate vector before projection.

---

### 27. embed_from_argmax
**Math**: Look up embedding row using argmax result.

| Field | Value |
|-------|-------|
| Params | `dim: u32` |
| Bindings | 0: `embeddings` BF16-packed read, 1: `output` f32 read_write [dim], 2: `argmax_result {idx: u32, val: f32}` read, 3: params |
| Workgroup | 256 |
| Dispatch | (ceil(dim/256), 1, 1) |

```
token_id = argmax_result.idx
output[i] = unpack_bf16(embeddings[(token_id * dim + i) / 2], ...)
```

---

## LoRA Weight Conventions

- **A**: `[in_features, rank]` row-major. Kaiming init.
- **B**: `[rank, out_features]` row-major. Zero init (identity at start).
- **Scale**: `alpha / rank` (default: 32/32 = 1.0)
- **Forward**: `output += (x @ A @ B) * scale`
- **4 targets**: q_proj, v_proj, o_proj, down_proj

## Fused Architecture-Specific Operations

### 28. fused_conv_deltanet_norm
**Math**: Conv1d + SiLU + DeltaNet linear recurrence + RMSNorm.
Required for Qwen 3.5 (24 of 32 layers in 9B).
Algorithm published: [Schlag et al. 2021 — "Linear Transformers Are Secretly Fast Weight Programmers"](https://arxiv.org/abs/2102.11174).
All component operations are standard; the fusion is an engineering optimization.

| Field | Value |
|-------|-------|
| Params | `num_heads: u32, key_dim: u32, value_dim: u32, total_channels: u32, eps: f32, hidden_size: u32, num_value_heads: u32` |
| Bindings | 0: `qkv` f32 read_write [total_channels], 1: `hist` f32 read_write [3 * total_channels], 2: `conv_weight` BF16-packed read, 3: `state` f32 read_write [num_value_heads * key_dim * value_dim], 4: `output` f32 read_write [num_value_heads * value_dim], 5: `hidden_input` f32 read [hidden_size], 6: `ab_weight` BF16-packed read [2 * num_value_heads * hidden_size / 2], 7: `A_log` BF16-packed read [num_value_heads], 8: `dt_bias` BF16-packed read [num_value_heads], 9: `norm_weight` BF16-packed read [value_dim], 10: params |
| Workgroup | 128 |
| Dispatch | (num_heads, 1, 1) — one workgroup per key head |

**Derived values:**
```
vpk = num_value_heads / num_heads    // value heads per key head (1 for 2B, 2 for 9B)
evd = vpk * value_dim                // effective value dim per key head
cpb = key_dim + key_dim + evd        // channels per block (Q + K + V)
```

**Phase 1: Conv1d + SiLU** (per-channel, 4-tap causal convolution)
```
QKV channel layout: Q[0..nh*kd), K[nh*kd..2*nh*kd), V[2*nh*kd..total_ch)
Per key head h, channels = [h*kd..(h+1)*kd) for Q,
                            [nh*kd + h*kd .. nh*kd + (h+1)*kd) for K,
                            [2*nh*kd + h*evd .. 2*nh*kd + (h+1)*evd) for V

For each channel c in this head's range:
    conv_out = w0*hist[c] + w1*hist[ch+c] + w2*hist[2*ch+c] + w3*qkv[c]
    qkv[c] = SiLU(conv_out)     // x / (1 + exp(-x))

    // Shift history
    hist[c] = hist[ch+c]        // t-2 <- t-1
    hist[ch+c] = hist[2*ch+c]   // t-1 <- t
    hist[2*ch+c] = qkv_before   // t <- current (pre-conv value)
```

**Phase 2: Q/K L2 normalization** (loaded into workgroup shared memory)
```
q_ss = sum(qkv[qh_off + k]^2)   for k in [0, kd)
k_ss = sum(qkv[kh_off + k]^2)
q_inv = 1 / max(sqrt(q_ss), 1e-6) / sqrt(kd)
k_inv = 1 / max(sqrt(k_ss), 1e-6)
wg_q[k] = qkv[qh_off + k] * q_inv
wg_k[k] = qkv[kh_off + k] * k_inv
```

**Phase 3: DeltaNet recurrence** (per value head, vpk iterations per key head)
```
For vhi in [0, vpk):
    vh = h * vpk + vhi     // global value head index

    // Compute alpha (decay) and beta (gate) from hidden_input @ ab_weight
    alpha = dot(hidden_input, ab_weight[vh * H/2 : vh * H/2 + H/2])     // BF16 matvec
    beta_raw = dot(hidden_input, ab_weight[(nhv + vh) * H/2 : ...])

    decay = exp(-exp(A_log[vh]) * softplus(alpha + dt_bias[vh]))
    beta = sigmoid(beta_raw)

    // Recurrent state update (state S is [key_dim, value_dim] per value head)
    For each vi in [0, value_dim):
        // Pass 1: decay state + compute S @ k
        kv_mem = 0
        for ki in [0, key_dim):
            S[ki, vi] *= decay
            kv_mem += S[ki, vi] * wg_k[ki]

        delta = (V[vi] - kv_mem) * beta

        // Pass 2: update state + compute S @ q
        o_val = 0
        for ki in [0, key_dim):
            S[ki, vi] += wg_k[ki] * delta
            o_val += S[ki, vi] * wg_q[ki]

        output[vh * vd + vi] = o_val
```

**Phase 4: RMSNorm** (per value head, shared norm_weight)
```
ss = sum(output[vh*vd + i]^2) for i in [0, vd)
rms = 1 / sqrt(ss / vd + eps)
output[vh*vd + i] *= rms * unpack_bf16(norm_weight[i/2], i%2)
```

**Key optimization**: Merged state read/write loops — 2 passes over state instead of 4 (decay+S@k fused, update+S@q fused), halving memory traffic.

---

## Excluded from this document

- **ParoQuant shaders** (JS-only) — learned Givens rotations for quantization
- **gptq_matmul_b2.wgsl** — batch-2 GPTQ variant (optimization, same math)
- **Vision Transformer shaders** (JS-only) — ViT-specific ops
