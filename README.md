# EdgeBit-350M

**A ternary-aware edge-native transformer stack built from scratch.**

Target architecture: ~334M parameters with BitLinear ternary-aware training, grouped-query attention, SubLN/QK normalization, INT8 KV-cache support, and verified 2-bit packed checkpoint storage. The current Python runtime reloads packed storage into normal PyTorch tensors; custom packed ternary matmul kernels and llama.cpp-compatible ternary GGUF execution are marked experimental until implemented and benchmarked.

---

## Why This Project Exists

The AI industry builds increasingly large models that require increasingly expensive hardware. EdgeBit takes the opposite approach: **build a model designed for the hardware that already exists** — phones, laptops, Raspberry Pis, IoT devices.

This is not a compressed version of a larger model. Every architectural decision — ternary quantization, grouped query attention, progressive training curriculum — is designed from the ground up for edge deployment.

**The thesis**: Useful local AI systems can be built with architecture-aware low-bit training and runtime-first engineering.

**The value**: Not model quality (a 350M ternary model will not beat GPT-4), but the complete, reproducible stack — training, quantization, packing, deployment, benchmarking — as a cohesive, deployable system.

---

## Architecture

| Component | Spec |
|-----------|------|
| Parameters | ~334M |
| Hidden dim | 1024 |
| FFN dim | 2816 (SwiGLU) |
| Layers | 24 |
| Attention | GQA: 16 query heads, 4 KV heads, head_dim=64 |
| Context | 2048 tokens |
| Position | RoPE (theta=10000) |
| Weights | Ternary (1.58-bit) via BitLinear with STE |
| Embeddings | FP32/FP16 in active model; NF4 utilities are available but not active |
| KV Cache | INT8 symmetric quantization |
| Normalization | RMSNorm + SubLN (identity init) |
| Attention stability | QK RMSNorm |
| Packed storage | 2-bit BitLinear checkpoint artifact implemented; full packed-runtime size depends on embedding storage |

### Memory Budget Status

```
Implemented today:
  - BitLinear weights can be exported to verified 2-bit packed storage.
  - INT8 KV cache is used during generation when enabled.
  - The active embedding layer is still a normal PyTorch embedding.

Projected target:
  - A ~156 MB full 350M artifact requires active NF4 embedding storage plus
    packed ternary compute/runtime integration. Those parts are roadmap items,
    not current measured runtime claims.
```

---

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Fast offline smoke train: creates a checkpoint without downloading a tokenizer
python -m training.train --smoke_test --config configs/model_smoke.yaml \
  --output_dir smoke_output/pipeline_check --max_steps 1 --tokenizer simple \
  --batch_size 2 --gradient_accumulation_steps 1

# Reload that checkpoint and generate
python demos/edge_deploy.py --checkpoint smoke_output/pipeline_check/checkpoint-1 \
  --config configs/model_smoke.yaml --tokenizer simple --max_tokens 8

# Run tests
pytest tests/ -v
```

> **New to this project?** See [GUIDE.md](GUIDE.md) for a complete step-by-step walkthrough from setup to deployment.

---

## Training Pipeline

Three-stage progressive training. Total budget: **50-60 A100-hours** (~$75-120 on cloud GPUs).

| Stage | Model | Params | Steps | A100 Hours | Purpose |
|-------|-------|--------|-------|------------|---------|
| 1 | Tiny | ~50M | 5,000 | 3-5 | Validate convergence |
| 2 | Small | ~125M | 15,000 | 10-15 | Prove scaling |
| 3 | Base | ~334M | 50,000 | 25-35 | Full training |

```bash
bash scripts/train_tiny.sh      # Stage 1
bash scripts/train_125m.sh      # Stage 2
bash scripts/train_350m.sh      # Stage 3
```

### Progressive Quantization Curriculum

Training transitions through precision stages, giving the model time to adapt:

```
  BF16 warmup (10%)  →  INT8 (20%)  →  INT4 (25%)  →  Ternary (45%)
       │                    │                │               │
   Learn basic          Adapt to         Prepare for      Target
   features             mild quant       low-bit          precision
```

See [Training Pipeline Documentation](docs/training_pipeline.md) for details.

---

## Cloud Training

```bash
# Quick setup on RunPod / Vast.ai / Lambda:
bash scripts/runpod_bootstrap.sh --stage tiny

# With your data:
bash scripts/runpod_bootstrap.sh --stage 350m --data-url https://your-data/train.jsonl

# Monitor training:
bash scripts/monitor.sh --output_dir /workspace/ckpts/edgebit-350m
```

---

## Demos

| Demo | Command | Description |
|------|---------|-------------|
| CLI Assistant | `python demos/cli_assistant.py` | Interactive text generation |
| Ticket Classifier | `python demos/ticket_classifier.py` | Support ticket routing |
| Summarizer | `python demos/summarizer.py` | Short text summarization |
| RAG | `python demos/rag_demo.py` | Retrieval-augmented Q&A |
| Edge Deploy | `python demos/edge_deploy.py` | Raspberry Pi deployment |

---

## Export

```bash
# HuggingFace format (safetensors + tokenizer)
python -m export.export_hf --checkpoint /path/to/ckpt --output_dir ./hf_export

# Minimal experimental GGUF-like container
python -m export.export_gguf --checkpoint /path/to/ckpt --output ./edgebit-350m.gguf

# Verified packed EdgeBit storage (reloads into the PyTorch runtime)
python -m export.export_packed --checkpoint /path/to/ckpt --output ./edgebit-packed.pt
```

---

## Benchmarking

```bash
# CPU inference benchmark
python -m runtime.bench_runtime --preset tiny --quant_modes none int8 ternary

# Token throughput sweep
python -m eval.bench_tokens --preset tiny --batch_sizes 1 2 4

# Memory profiling
python -m runtime.memory_profile --preset base --show_layers

# Full benchmark suite
bash benchmarks/run_all_benchmarks.sh
```

### Runtime Status

| Metric | Status |
|--------|--------|
| PyTorch fake-quant inference | Implemented and benchmarkable |
| INT8 KV-cache generation | Implemented |
| Packed checkpoint storage | Implemented and tested |
| Packed ternary CPU matmul | Experimental / roadmap |
| llama.cpp-compatible ternary GGUF | Experimental / roadmap |

---

## Docker

```bash
# CPU inference
docker build -t edgebit-cpu -f docker/Dockerfile.cpu .
docker run -it edgebit-cpu

# CUDA training
docker build -t edgebit-cuda -f docker/Dockerfile.cuda .
docker run --gpus all edgebit-cuda
```

---

## Project Structure

```
edgebit-350m/
├── modeling/           Core model architecture
│   ├── bitlinear.py      BitLinear (ternary/INT8/INT4 modes)
│   ├── quant_utils.py    Quantization primitives (STE, ternary, NF4, INT8)
│   ├── attention_utils.py QK norm, INT8 KV cache
│   ├── subln.py          Sub-Layer Normalization
│   ├── config.py         Model configs (tiny/125m/350m presets)
│   └── model.py          Full transformer implementation
├── training/           Training infrastructure
│   ├── train.py          Training loop with curriculum
│   ├── quant_scheduler.py Progressive quantization scheduler
│   ├── distill_losses.py KL + hidden-state distillation
│   └── data.py           JSONL data loading (pretraining, instruction, streaming)
├── runtime/            Inference engineering
│   ├── pack_ternary.py   2-bit packing (16x compression)
│   ├── bench_runtime.py  CPU inference benchmarks
│   └── memory_profile.py Memory profiling
├── export/             Model export (HuggingFace, GGUF)
├── eval/               Evaluation (lm-eval, throughput sweeps)
├── demos/              Deployable demo applications
├── configs/            Model + training configs
├── scripts/            Setup, training, and cloud scripts
├── docker/             CPU + CUDA Dockerfiles
├── tests/              Unit test suite
├── docs/               Architecture, training, runtime, scaling docs
├── blog/               Technical writeups
├── benchmarks/         Benchmark scripts and results
├── notebooks/          Jupyter walkthrough notebooks
└── assets/             Diagrams and visual materials
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[GUIDE.md](GUIDE.md)** | **Complete step-by-step walkthrough from setup to deployment** |
| [RUNBOOK.md](RUNBOOK.md) | Hour-by-hour training guide with expected loss values |
| [Architecture](docs/architecture.md) | Ternary quantization, GQA, SubLN, QK norm, memory budget |
| [Training Pipeline](docs/training_pipeline.md) | Curriculum, optimizer, stability, cloud setup |
| [Runtime](docs/runtime.md) | Packing, CPU optimization, deployment, latency |
| [Scaling Roadmap](docs/scaling_roadmap.md) | Path to 1B+, bottlenecks, research directions |
| [Showcase](docs/showcase.md) | Presentation guide, demo ideas, portfolio strategy |
| [Model Card](MODEL_CARD.md) | HuggingFace-style model card |

## Blog Posts

| Post | Topic |
|------|-------|
| [Why Edge-Native AI Matters](blog/01_why_edge_native_ai_matters.md) | Motivation and deployment philosophy |
| [Building a Ternary Transformer](blog/02_building_a_ternary_transformer.md) | BitLinear, STE, curriculum design |
| [What We Learned Training](blog/03_what_we_learned_training.md) | Practical lessons and debugging |
| [Runtime Engineering](blog/04_runtime_engineering_for_low_bit_llms.md) | Packing, CPU optimization, memory |
| [Why Small Models Matter](blog/05_why_small_models_matter.md) | Cost, privacy, latency arguments |
| [QAT Challenges](blog/06_challenges_of_quantization_aware_training.md) | Gradients, stability, reproducibility |

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Ternary from scratch** | Weights trained ternary via STE, not post-hoc quantized |
| **Progressive curriculum** | Gradual precision reduction prevents catastrophic loss |
| **GQA (4 KV heads)** | 4x KV cache savings at minimal quality cost |
| **SubLN** | Stabilizes deep ternary networks (identity init = no cost at start) |
| **QK normalization** | Prevents attention logit explosion in low-bit training |
| **NF4 helpers** | Quantization utilities exist; active embedding module/export integration is roadmap |
| **INT8 KV cache** | Implemented symmetric cache quantization for generation |
| **Tied embeddings** | Save 155M parameters (46% of model) |

## Implementation Status

| Capability | Status |
|------------|--------|
| Decoder-only model, RoPE, GQA, SwiGLU, SubLN, QK norm | Implemented |
| Progressive quantization curriculum | Implemented |
| Padding-aware causal training masks | Implemented |
| Correct RoPE offsets during cached generation | Implemented |
| Offline smoke tokenizer and smoke config | Implemented |
| Optional Accelerate launch path | Implemented |
| Optional logit distillation with matching vocab teacher | Implemented |
| NF4 embedding layer active in model | Roadmap |
| Packed ternary storage artifact | Implemented |
| Packed ternary compute kernel | Roadmap |
| GGUF ternary runtime compatibility | Experimental |

---

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 12.1+ (for GPU training)
- Training: NVIDIA A100 80GB recommended
- Inference: Raspberry Pi 5+ / any CPU with 512MB+ RAM

---

## License

MIT

---

## Citation

```bibtex
@misc{edgebit350m,
  title={EdgeBit-350M: A 1.58-bit Edge-Native Transformer},
  author={Ponamwar, Jignesh},
  year={2025},
  url={https://github.com/jigneshponamwar/edgebit-350m}
}
```
