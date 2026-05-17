# Scaling Roadmap

## Where We Are

EdgeBit-350M is a proof-of-concept: a complete training, quantization, and deployment stack for ternary transformers. It proves that the architecture converges, the curriculum works, and the runtime engineering delivers real compression.

This document is honest about what comes next, what is hard, and what we do not yet know.

---

## Phase 1: Validate (Current — EdgeBit-350M)

**Status**: In progress

**Goal**: Prove convergence, build the stack, establish baselines.

| Milestone | Status |
|-----------|--------|
| Architecture implementation | Complete |
| Progressive quantization curriculum | Complete |
| Ternary weight packing | Complete |
| INT8 KV cache | Complete |
| Training pipeline | Complete |
| Export (HF, packed storage) | Implemented |
| GGUF ternary runtime compatibility | Experimental |
| Benchmarking infrastructure | Complete |
| Documentation | Complete |
| Training run (smoke test) | Ready |
| Training run (full 350M) | Pending compute |
| Benchmark results | Pending training |

**Open questions**:
- What is the actual ternary vs. FP16 quality gap at 350M?
- How does the progressive curriculum compare to direct ternary training?
- What is the minimum data volume for useful convergence?

---

## Phase 2: Scale to 1B (Next)

**Goal**: Demonstrate that the architecture and training pipeline scale.

**Model spec**:

| Component | EdgeBit-350M | EdgeBit-1B |
|-----------|-------------|------------|
| Hidden dim | 1024 | 2048 |
| FFN dim | 2816 | 5632 |
| Layers | 24 | 24 |
| Heads | 16 | 32 |
| KV heads | 4 | 8 |
| Params | ~334M | ~1.3B |
| Packed size | ~35 MB | ~140 MB |

**Estimated compute**: 200-400 A100-hours (~$300-600)

**What must change**:
- Gradient checkpointing for memory efficiency
- DeepSpeed ZeRO-3 or FSDP for multi-GPU training
- Larger pretraining corpus (20-50B tokens)
- More sophisticated learning rate scheduling
- Potentially distillation from a 7B+ teacher

**Known risks**:
- Ternary quantization loss may scale worse than linear with model size
- Attention instability may require additional stabilization at depth
- KV cache memory becomes significant at longer contexts

---

## Phase 3: Scale to 2-3B (Future)

**Goal**: Reach the quality threshold where ternary models are practically useful for real tasks.

**Why 2-3B matters**: At this scale, full-precision models can follow complex instructions, write code, and reason about multi-step problems. If ternary models can achieve 60-70% of this capability at 10x less memory, they become genuinely useful on edge devices.

**Model spec (tentative)**:

| Component | EdgeBit-2B |
|-----------|-----------|
| Hidden dim | 2560 |
| FFN dim | 7168 |
| Layers | 32 |
| Heads | 32 |
| KV heads | 8 |
| Params | ~2.5B |
| Packed size | ~350 MB |

**Estimated compute**: 1000-2000 A100-hours (~$1500-3000)

**What must change**:
- Sliding window or sparse attention for longer contexts
- Mixture of quantization: some layers may need higher precision
- Sophisticated distillation curriculum (progressive layer unfreezing)
- Custom CUDA kernels for ternary matmul
- Distributed training across multiple nodes

---

## Infrastructure Bottlenecks

### Training Bottlenecks

| Bottleneck | Severity | Mitigation |
|------------|----------|------------|
| Compute cost | High | Cloud spot instances, progressive scaling |
| Data pipeline | Medium | Streaming datasets, data preprocessing |
| Memory per GPU | Medium | ZeRO-3, gradient checkpointing |
| Training stability | Medium | Curriculum tuning, more SubLN layers |
| Hyperparameter search | Low | Stage 1/2 validation before full runs |

### Runtime Bottlenecks

| Bottleneck | Severity | Mitigation |
|------------|----------|------------|
| Embedding memory | High | Better NF4 or binary embeddings |
| KV cache at long contexts | Medium | Sliding window, GQA, cache eviction |
| CPU matmul speed | Medium | Custom ternary kernels (C/Rust) |
| Model loading time | Low | Memory-mapped weights |
| Tokenizer overhead | Low | Compiled tokenizers |

### Research Bottlenecks

| Question | Impact | How to Answer |
|----------|--------|---------------|
| Ternary quality ceiling | Critical | Train larger models, compare scaling curves |
| Optimal curriculum shape | High | Ablation studies across phase ratios |
| SubLN vs. other stabilization | Medium | Controlled experiments |
| Embedding quantization limit | Medium | Compare NF4, INT4, INT2, binary |
| Distillation effectiveness | High | Compare with/without teacher at each scale |

---

## Future Research Directions

### 1. Custom Ternary Kernels

The current implementation uses PyTorch's standard matmul with quantized weights. A custom kernel could exploit the ternary structure:

```
Ternary matmul: y = W_ternary @ x
  = (W_plus @ x) - (W_minus @ x)

where W_plus[i,j] = 1 if W[i,j] == +1, else 0
      W_minus[i,j] = 1 if W[i,j] == -1, else 0
```

This decomposes the matmul into two sparse additions, which could be implemented with bitwise operations on packed weights. Potential speedup: 2-5x on CPU.

### 2. Mixed-Precision Architecture

Not all layers need the same precision. Potential strategies:
- First and last layers at INT8 (most sensitive to quality)
- Middle layers at ternary (most redundant)
- Attention projections at INT4, FFN at ternary

This requires architecture search or sensitivity analysis.

### 3. KV Cache Eviction

For long-context deployment, the KV cache eventually exceeds memory. Strategies:
- Sliding window (drop oldest tokens)
- Attention sink (keep first + recent tokens)
- Heavy hitter cache (keep high-attention tokens)

### 4. Speculative Decoding

Use a tiny draft model (50M ternary) to propose tokens, verified by the full 350M model. This can increase effective throughput by 2-3x with minimal quality loss.

### 5. Structured Pruning + Ternary

Many ternary weights are zero. If entire rows or columns are zero, they can be pruned structurally, further reducing compute and memory.

---

## Honest Assessment

**What EdgeBit-350M is designed to prove**: The architecture, training pipeline, and deployment stack can support ternary-aware models. The current code verifies smoke training, checkpoint reload, generation, and packed checkpoint storage; full packed-runtime edge deployment remains roadmap work.

**What it does not prove (yet)**: That ternary models at this scale are useful for real tasks. That the quality gap with full-precision models is acceptable. That the approach scales to 2B+.

**The bet**: That the engineering stack — not the model quality — is the hard part. Once the stack works at 350M, scaling is an investment decision, not a research problem.

**The risk**: That ternary quantization has a quality ceiling that makes larger models not worth the engineering effort. If a 2B ternary model only matches a 200M FP16 model, the compression advantage may not justify the complexity.

**How we will know**: Train the 350M model, measure quality, compare with FP16 baselines. If the quality ratio is > 0.5x at matched parameter count, scaling is justified.
