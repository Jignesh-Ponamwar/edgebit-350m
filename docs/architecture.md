# EdgeBit-350M Architecture

## Overview

EdgeBit-350M is a decoder-only causal language model built from scratch for edge deployment. The implemented stack currently provides ternary-aware BitLinear training, INT8 KV cache, and verified packed checkpoint storage. Active NF4 embeddings and packed ternary matmul kernels are roadmap items.

The model is not a compressed version of a larger model. It is designed, from the first line of code, to train and run in low-bit precision.

```
┌──────────────────────────────────────────────────────────┐
│                   EdgeBit-350M                           │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Embedding Layer (NF4 quantized, 151936 x 1024)    │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │  Decoder Block x 24                                 │ │
│  │  ┌───────────────────────────────────────────────┐  │ │
│  │  │  RMSNorm                                      │  │ │
│  │  │  ┌─────────────────────────────────────────┐  │  │ │
│  │  │  │  GQA Attention (16Q / 4KV heads)        │  │  │ │
│  │  │  │  BitLinear Q,K,V,O projections          │  │  │ │
│  │  │  │  QK RMSNorm stabilization               │  │  │ │
│  │  │  │  INT8 KV Cache                          │  │  │ │
│  │  │  │  SubLN before output projection         │  │  │ │
│  │  │  │  RoPE positional encoding               │  │  │ │
│  │  │  └─────────────────────────────────────────┘  │  │ │
│  │  │  Residual + RMSNorm                           │  │ │
│  │  │  ┌─────────────────────────────────────────┐  │  │ │
│  │  │  │  SwiGLU FFN (1024 → 2816 → 1024)       │  │  │ │
│  │  │  │  BitLinear gate, up, down projections   │  │  │ │
│  │  │  │  SubLN before down projection           │  │  │ │
│  │  │  └─────────────────────────────────────────┘  │  │ │
│  │  │  Residual                                     │  │ │
│  │  └───────────────────────────────────────────────┘  │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │  RMSNorm → LM Head (tied with embedding)           │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Parameter Budget

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Embedding (151936 x 1024) | 155.6M | 46.6% |
| Attention (Q+K+V+O) x 24 | 75.5M | 22.6% |
| FFN (gate+up+down) x 24 | 202.8M | 60.7% |
| Norms + SubLN | ~0.2M | <0.1% |
| LM Head (tied) | 0 | 0% |
| **Total** | **~334M** | **100%** |

Tying the LM head to the embedding saves 155M parameters. The embedding itself uses NF4 quantization (4-bit), so it occupies ~75MB in fp16 but only ~19MB packed.

---

## Core Components

### 1. BitLinear: Ternary Weight Layers

BitLinear replaces `nn.Linear` throughout the transformer (except embeddings and the final norm). During the forward pass:

1. **Weight quantization**: Weights are quantized to {-1, 0, +1} using per-group absmean scaling.
2. **Activation quantization**: Inputs are scaled to INT8 range using per-token absmax.
3. **Matmul**: The quantized weight-activation product is computed.
4. **Rescaling**: Output is rescaled by the product of weight and activation scales.

```
Forward pass:

  x_norm = x / max(|x|) * 127          # per-token INT8 scaling
  w_ternary = round(w / mean(|w|))      # per-group ternary quantization
  w_ternary = clamp(w_ternary, -1, +1)

  y = x_norm @ w_ternary.T
  y = y * (scale_x * scale_w)           # rescale to original magnitude
```

**Why ternary?** Ternary weights ({-1, 0, +1}) replace multiply-accumulate (MAC) operations with additions and subtractions. On CPU, this means:
- No floating-point multiplies in the core matmul
- 16x weight compression (2 bits per weight vs 32 bits)
- Cache-friendly memory access patterns

**Gradient flow**: Quantization is non-differentiable. We use the Straight-Through Estimator (STE): during backprop, gradients pass through the quantization function as if it were the identity function. The float32 "shadow weights" accumulate gradients and are re-quantized each forward pass.

```
Forward:  w_q = quantize(w)     # discrete ternary values
Backward: ∂L/∂w = ∂L/∂w_q      # STE: gradient passes through
```

### 2. Grouped Quantization

Quantizing an entire weight matrix with a single scale factor loses too much information. Instead, we divide each weight row into groups of 128 elements and compute an independent scale per group.

```
For weight matrix W of shape [out_features, in_features]:
  groups = W.reshape(-1, 128)                    # [n_groups, 128]
  scale = mean(|groups|, dim=-1)                 # [n_groups]
  W_ternary = round(groups / scale).clamp(-1,1)  # {-1, 0, +1}
```

The scale factors are stored in fp16, adding minimal overhead: for a 1024x1024 matrix, there are 8192 groups, requiring 16KB of scale storage vs. 256KB for the ternary-packed weights.

### 3. Grouped Query Attention (GQA)

Standard multi-head attention uses separate K and V projections for each head. GQA shares K/V heads across multiple query heads, reducing KV cache memory and compute.

```
Configuration:
  Query heads:  16 (head_dim = 64, total = 1024)
  KV heads:      4 (head_dim = 64, total = 256)
  Expansion:     4 query heads share each KV head

Memory savings:
  MHA KV cache:  2 * 16 * seq_len * 64 = 2048 * seq_len bytes (fp16)
  GQA KV cache:  2 *  4 * seq_len * 64 =  512 * seq_len bytes (fp16)
  Savings: 4x reduction
```

At inference time, each KV head is repeated 4 times to match the 16 query heads:

```python
k = k[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1)  # [B, 4, 4, S, 64]
k = k.reshape(B, 16, S, 64)                              # [B, 16, S, 64]
```

### 4. QK Normalization

In low-bit networks, attention logits can grow unboundedly because quantized Q and K values lack the implicit normalization of full-precision training. This causes:
- Softmax saturation (attention collapses to single tokens)
- Gradient explosion through the softmax
- Training instability in deep networks

We apply independent RMSNorm to Q and K before computing attention scores:

```python
q = rms_norm(q) * learnable_scale_q
k = rms_norm(k) * learnable_scale_k
attn = (q @ k.T) / sqrt(head_dim)
```

The learnable scales start at 1.0 and allow the model to control attention magnitude during training.

### 5. SubLN (Sub-Layer Normalization)

Standard pre-norm transformers apply LayerNorm before each sub-layer. SubLN adds an additional normalization *inside* the sub-layer, specifically before the output projection.

```
Standard:     x + Attn(LN(x))
SubLN:        x + O_proj(SubLN(Attn_core(LN(x))))
```

The SubLN is initialized as identity (gamma = 1, no bias) so it has no effect at initialization. During training, it learns to normalize the internal representations, preventing activation magnitude drift in deep ternary networks.

This is particularly important for BitLinear because:
- Ternary weights produce outputs with unpredictable scale
- The STE introduces gradient noise that compounds across layers
- Without internal normalization, deep ternary networks diverge

### 6. NF4 Embeddings

The embedding table is the single largest parameter group (155M params, 46% of total). NF4 helper functions are implemented in `modeling.quant_utils`, but the active model still uses a standard PyTorch embedding so tied output weights and training remain straightforward. Treat active NF4 embedding storage as roadmap work.

```
NF4 quantization:
  1. Divide embedding into blocks of 64 elements
  2. Compute absmax per block
  3. Normalize values to [-1, 1]
  4. Map each value to nearest of 16 NF4 quantization levels
  5. Store as 4-bit index + fp16 absmax per block

NF4 quantization levels (optimized for normal distributions):
  [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0]

Storage: 4 bits/param + 2 bytes per 64-element block
  = 155M * 0.5 bytes + 155M/64 * 2 bytes ≈ 82MB
  vs 300MB in fp16 → 3.6x compression
```

NF4 levels are non-uniform, concentrated near zero where the normal distribution has highest density. This gives better reconstruction than uniform INT4 for normally-distributed weights.

### 7. INT8 KV Cache

During autoregressive generation, the KV cache grows linearly with sequence length. INT8 quantization halves its memory footprint:

```
Per-layer KV cache at seq_len=2048:
  FP16: 2 * 4 heads * 2048 * 64 * 2 bytes = 2.0 MB
  INT8: 2 * 4 heads * 2048 * 64 * 1 byte  = 1.0 MB

Full model (24 layers):
  FP16: 48 MB
  INT8: 24 MB
```

We use symmetric per-token quantization:

```python
scale = max(|x|) / 127
x_int8 = round(x / scale).clamp(-127, 127).to(int8)
x_reconstructed = x_int8.float() * scale
```

The reconstruction error is typically < 0.5% of the signal, which has negligible impact on generation quality.

### 8. RoPE (Rotary Position Embeddings)

RoPE encodes position by rotating query and key vectors in 2D subspaces:

```
For position m, dimension pair (2i, 2i+1):
  θ_i = 10000^(-2i/d)
  q'[2i]   = q[2i] * cos(m*θ_i) - q[2i+1] * sin(m*θ_i)
  q'[2i+1] = q[2i] * sin(m*θ_i) + q[2i+1] * cos(m*θ_i)
```

RoPE frequencies are precomputed as a fixed buffer (not learned), so they add no parameters. The relative position encoding naturally handles variable-length sequences without padding position embeddings.

### 9. SwiGLU FFN

The feed-forward network uses a gated architecture:

```python
def swiglu_ffn(x):
    gate = silu(gate_proj(x))    # [B, S, ffn_dim]
    up   = up_proj(x)            # [B, S, ffn_dim]
    return down_proj(gate * up)  # [B, S, hidden_dim]
```

SwiGLU consistently outperforms ReLU and GELU FFNs at equivalent parameter counts. The cost is 3 projections (gate, up, down) instead of 2, which we account for in the parameter budget.

---

## Progressive Quantization Curriculum

Training a ternary model from random initialization is unstable. The progressive curriculum gradually reduces precision:

```
Step 0              Step 0.1T           Step 0.3T           Step 0.55T          Step T
│                   │                   │                   │                   │
▼                   ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  BF16       │────▶│  INT8       │────▶│  INT4       │────▶│  Ternary    │
│  Warmup     │     │  Adapt      │     │  Adapt      │     │  Final      │
│  (10%)      │     │  (20%)      │     │  (25%)      │     │  (45%)      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
  Full precision      Mild quant          Aggressive          Target precision
  Learn features      Adapt to noise      Prepare for         Majority of
                                          ternary             training
```

**Why this works**: Each phase boundary causes a temporary loss spike as the model adjusts to lower precision. By the time ternary training begins, the model has already learned robust representations that survive quantization. The ternary phase gets 45% of total training time, enough to fully converge.

---

## Hidden-State Distillation

When a teacher model is available, we align intermediate representations:

```
Teacher (e.g., Qwen3-1.5B):
  Layer 4  ──────────────────────┐
  Layer 8  ──────────────────────┤  MSE loss
  Layer 12 ──────────────────────┤  (after projection)
                                 │
Student (EdgeBit-350M):          │
  Layer 2  ──projector──────────►┤
  Layer 5  ──projector──────────►┤
  Layer 8  ──projector──────────►┘

Total loss = α_kd * KL(student, teacher) + α_ce * CE(student, labels) + α_hidden * MSE(hidden)
           = 0.7  * KL_loss             + 0.3  * CE_loss              + 0.1    * hidden_loss
```

The projectors are small linear layers (student_dim → teacher_dim) trained alongside the student. They are discarded after training.

---

## Memory Budget at Inference

```
Component                    FP16        Packed Ternary
─────────────────────────────────────────────────────────
Embedding (NF4)              300 MB      82 MB
BitLinear weights            557 MB      35 MB*
Norms + scales               0.4 MB      0.4 MB
KV cache (2048 tokens)       48 MB       24 MB (INT8)
Activations (batch=1)        ~8 MB       ~8 MB
─────────────────────────────────────────────────────────
Total                        913 MB      149 MB

* 2-bit packed ternary: 4 weights per byte + group scales
```

The packed-storage target is plausible, but full device-fit claims should be
based on measured artifacts from `runtime.memory_profile` and
`export.export_packed`. The active Python runtime does not yet include NF4
embedding storage or packed ternary matmul execution.

---

## Design Tradeoffs

| Decision | Benefit | Cost |
|----------|---------|------|
| Ternary weights | 16x compression, no-multiply inference | Lower model quality per param |
| GQA (4 KV heads) | 4x KV cache savings | Slight attention capacity loss |
| SubLN | Stabilizes deep ternary training | 2 extra norms per layer |
| QK Norm | Prevents attention collapse | Small compute overhead |
| NF4 embeddings | Potential 3.6x embedding compression | Roadmap; helper functions exist but active model still uses standard embeddings |
| INT8 KV cache | 2x cache savings | < 0.5% reconstruction error |
| Tied embeddings | Save 155M params | Constrains output space |
| RoPE | No learned position params | Fixed position encoding |
| Progressive curriculum | Stable ternary convergence | Longer training time |

Every tradeoff favors the edge deployment goal. We accept slightly lower quality-per-parameter in exchange for dramatically lower memory and compute requirements.
