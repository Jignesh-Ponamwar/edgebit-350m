# Model Card: EdgeBit-350M

## Model Details

- **Model name**: EdgeBit-350M
- **Model type**: Causal language model (decoder-only transformer)
- **Architecture**: Custom ternary-aware transformer with BitLinear layers
- **Parameters**: ~334M
- **Quantization**: Ternary-aware BitLinear training, INT8 KV cache, verified packed BitLinear checkpoint storage. NF4 embedding utilities exist but are not active in the model yet.
- **Context length**: 2048 tokens
- **Tokenizer**: Qwen/Qwen3-0.6B (151,936 vocab)
- **License**: TBD
- **Developer**: Jignesh Ponamwar

## Architecture Details

| Hyperparameter | Value |
|----------------|-------|
| Hidden dimension | 1024 |
| FFN dimension | 2816 |
| Number of layers | 24 |
| Attention heads | 16 (query), 4 (KV) |
| Head dimension | 64 |
| Activation | SwiGLU |
| Normalization | RMSNorm + SubLN |
| Position encoding | RoPE |
| Weight quantization | Ternary ({-1, 0, +1}) per group-128 |
| Embedding quantization | Roadmap; active model uses a standard PyTorch embedding |
| KV cache quantization | INT8 symmetric |

## Training

### Data
- Pretraining corpus: 5-8B tokens (target)
- Instruction tuning: 200-500K samples (target)

### Training Procedure
- **Optimizer**: AdamW (lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
- **Schedule**: Cosine annealing with warm restarts
- **Progressive quantization curriculum**:
  - Phase 1 (10%): BF16 warmup
  - Phase 2 (20%): INT8 quantization
  - Phase 3 (25%): INT4 quantization
  - Phase 4 (45%): Ternary (1.58-bit) final
- **Gradient clipping**: 1.0
- **Hardware**: NVIDIA A100 80GB
- **Budget**: 50-60 A100 hours across 3 stages

### Three-Stage Training
1. **Tiny validation** (~50M params): 3-5 A100 hours, validates convergence
2. **125M proof**: 10-15 A100 hours, validates scaling
3. **350M final**: 25-35 A100 hours, full training

## Intended Use

### Primary Use Cases
- Edge device inference (Raspberry Pi 5, mobile, IoT)
- Low-memory environments (< 512MB RAM)
- Research into extreme quantization architectures
- Proof of concept for ternary transformer training

### Out of Scope
- Production safety-critical applications without additional fine-tuning and evaluation
- Applications requiring state-of-the-art benchmark performance
- Long-context tasks (> 2048 tokens)

## Limitations

- Small model size limits reasoning and factual knowledge compared to larger models
- Ternary quantization introduces quality tradeoffs vs. full-precision models
- Not instruction-tuned or RLHF-aligned in base version
- Limited context window (2048 tokens)
- Benchmark scores will be lower than comparable FP16 models at same param count
- Packed checkpoint storage is implemented, but custom packed ternary matmul kernels are not yet included
- GGUF export is a minimal experimental container, not verified llama.cpp ternary execution

## Evaluation

Target benchmarks (scores TBD after training):
- MMLU (5-shot)
- HellaSwag (10-shot)
- GSM8K (5-shot)
- Winogrande (5-shot)
- ARC-Easy/Challenge (25-shot)

### Runtime Metrics
- Tokens/second on CPU
- Peak memory usage
- Model file size for verified packed checkpoint storage

## Ethical Considerations

This model is a research prototype for extreme quantization techniques. It has not undergone safety alignment (RLHF, constitutional AI, etc.) and should not be deployed in user-facing applications without appropriate guardrails.

## Citation

```
@misc{edgebit350m,
  title={EdgeBit-350M: A 1.58-bit Edge-Native Transformer},
  author={Ponamwar, Jignesh},
  year={2025},
}
```
