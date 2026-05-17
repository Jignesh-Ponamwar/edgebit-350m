# Benchmark Reports

## How to Run

```bash
# Model quality (requires lm-eval)
python -m eval.run_lm_eval --checkpoint /path/to/ckpt --tasks mmlu hellaswag gsm8k winogrande

# Runtime performance
python -m runtime.bench_runtime --preset base --quant_modes none int8 ternary --output_json benchmarks/results/runtime.json

# Token throughput sweep
python -m eval.bench_tokens --preset base --output_json benchmarks/results/throughput.json

# Memory profiling
python -m runtime.memory_profile --preset base --output_json benchmarks/results/memory.json
```

## Expected Results (350M Model, Untrained)

These are structural benchmarks — they measure runtime characteristics, not model quality.
Quality benchmarks require a trained checkpoint.

### Memory Budget Status

The current benchmark scripts measure the active PyTorch model and verified
packed checkpoint storage. The full ~156MB target requires an active NF4
embedding storage path plus packed ternary matmul/runtime integration.

| Component | Current Status |
|-----------|----------------|
| BitLinear 2-bit checkpoint storage | Implemented and tested |
| INT8 KV cache | Implemented |
| NF4 embedding in active model | Roadmap |
| Packed ternary CPU matmul | Roadmap |

### Compression Ratios

| Technique | Ratio | Description |
|-----------|-------|-------------|
| Ternary packing (2-bit) | 16x vs FP32 | 4 weights per byte |
| NF4 embedding | Roadmap | Helper functions exist; not active in model |
| INT8 KV cache | 2x vs FP16 | Symmetric quantization |
| **Total model** | **5.9x vs FP16** | Combined compression |

### Latency Targets (x86 CPU)

| Metric | Target | Notes |
|--------|--------|-------|
| TTFT (64 tokens) | < 200 ms | Time to first token |
| Decode throughput | > 30 tok/s | Autoregressive generation |
| Model load time | < 3 sec | Cold start |

### Quality Targets (After Training)

| Benchmark | Random (baseline) | Target | Notes |
|-----------|------------------|--------|-------|
| MMLU (5-shot) | ~25% | > 30% | Above random for 350M |
| HellaSwag (10-shot) | ~25% | > 35% | Basic common sense |
| Perplexity (wikitext) | Very high | < 30 | Meaningful language model |

These targets are realistic for a 350M ternary model. The goal is to demonstrate
convergence and useful learning, not to compete with larger models.

## Interpreting Results

### Model Quality
At 350M parameters with ternary weights, expect performance roughly equivalent to
a 60-90M FP16 model. The value is in the compression ratio and deployment capability,
not absolute benchmark scores.

### Runtime Performance
The key metrics are memory and latency. For now, report PyTorch fake-quant
runtime separately from packed checkpoint storage size. Do not report the
projected full packed-runtime size as a measured result until NF4 embeddings and
packed matmul are active.

### Tradeoffs
Every compression technique trades quality for efficiency. The benchmark results
should be read as: "how much quality did we give up, and was it worth the compression?"
