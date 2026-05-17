# EdgeBit-350M: Complete Project Guide

This document walks you through the entire EdgeBit-350M project, from understanding what it does to training and deploying a model. It assumes you have basic familiarity with Python and machine learning concepts, but does not assume prior experience with quantization or edge deployment.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Understanding the Architecture](#2-understanding-the-architecture)
3. [Setting Up Your Environment](#3-setting-up-your-environment)
4. [Verifying Everything Works](#4-verifying-everything-works)
5. [Exploring the Code](#5-exploring-the-code)
6. [Preparing Training Data](#6-preparing-training-data)
7. [Training Stage 1: Tiny Model](#7-training-stage-1-tiny-model)
8. [Training Stage 2: 125M Model](#8-training-stage-2-125m-model)
9. [Training Stage 3: Full 350M Model](#9-training-stage-3-full-350m-model)
10. [Evaluating Your Model](#10-evaluating-your-model)
11. [Exporting for Deployment](#11-exporting-for-deployment)
12. [Deploying to Edge Devices](#12-deploying-to-edge-devices)
13. [Running Benchmarks](#13-running-benchmarks)
14. [Common Issues and Fixes](#14-common-issues-and-fixes)

---

## 1. What This Project Does

Most AI models are enormous -- they need expensive GPUs with tens of gigabytes of memory. EdgeBit takes the opposite approach: build a model that runs on the hardware people already have.

**The core idea**: Instead of relying only on 32-bit or 16-bit weights, the core linear layers train through low-bit modes and finish in ternary mode: -1, 0, and +1 with per-group scales. The repo also includes verified 2-bit packed checkpoint storage for BitLinear weights.

**The tradeoff**: A 350M-parameter ternary model will not match the quality of GPT-4 or even much smaller full-precision models. The value is in the complete system -- training pipeline, quantization techniques, packing, deployment -- as a working, reproducible project.

**Numbers that matter**:

| Metric | Value |
|--------|-------|
| Packed runtime target | ~156 MB after NF4 embeddings and packed matmul are implemented |
| Verified today | Smoke training, checkpoint reload, INT8 KV cache, packed BitLinear storage |
| Active Python runtime | PyTorch fake-quant inference |
| Packed ternary matmul | Roadmap |
| Training cost | ~$60-75 (cloud GPUs) |
| Training time | 50-60 A100-hours |
| Inference speed (CPU) | 30-50 tokens/sec |

---

## 2. Understanding the Architecture

Before you run anything, here is how the key pieces fit together.

### The Model

EdgeBit-350M is a standard decoder-only transformer (like GPT or Llama), but every linear layer uses **BitLinear** instead of `nn.Linear`. BitLinear quantizes weights to {-1, 0, +1} during the forward pass, while keeping full-precision "shadow weights" for gradient updates.

```
Input tokens
    |
    v
Embedding (standard PyTorch today; NF4 storage is roadmap)
    |
    v
[Decoder Block] x 24
    |-- RMSNorm
    |-- Grouped Query Attention (16 query heads, 4 KV heads)
    |   |-- BitLinear Q, K, V, O projections
    |   |-- QK normalization (prevents attention explosion)
    |   |-- INT8 KV cache (halves cache memory)
    |-- RMSNorm
    |-- SwiGLU FFN (gate + up + down, all BitLinear)
    |
    v
RMSNorm -> LM Head (tied with embedding) -> Output logits
```

### The Training Curriculum

You cannot train a ternary model from scratch -- random ternary weights produce meaningless gradients. Instead, training starts in full precision and gradually reduces:

```
Phase 1: BF16 (full precision)    10% of training
Phase 2: INT8 (256 levels)        20% of training
Phase 3: INT4 (16 levels)         25% of training
Phase 4: Ternary (3 levels)       45% of training
```

Each phase transition causes a temporary loss spike. The model recovers because it already learned useful representations at higher precision. The ternary phase gets the most time (45%) because adapting to only 3 weight values takes the longest.

### The Three Training Stages

Training is split into 3 stages to avoid wasting compute:

| Stage | Model Size | Time | Purpose |
|-------|-----------|------|---------|
| 1. Tiny | ~50M params | 3-5 hours | Does the architecture work at all? |
| 2. Small | ~125M params | 10-15 hours | Does it improve with more parameters? |
| 3. Full | ~334M params | 25-35 hours | Production training |

If Stage 1 fails, you fix the problem before spending 35 hours on Stage 3.

---

## 3. Setting Up Your Environment

### What You Need

**For exploring, testing, and smoke tests** (no GPU):
- Python 3.10+
- 4 GB RAM
- 2 GB disk space

**For real training**:
- NVIDIA GPU (A100 80GB recommended, RTX 3090/4090 works for Stage 1)
- CUDA 12.1+
- 50+ GB disk space for checkpoints
- Training data in JSONL format

### Installation (Step by Step)

**Step 1**: Clone the repository.

```bash
git clone https://github.com/jigneshponamwar/edgebit-350m.git
cd edgebit-350m
```

**Step 2**: Create and activate a virtual environment.

```bash
# Create
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

**Step 3**: Install PyTorch.

```bash
# For GPU training (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only (testing/inference):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 4**: Install project dependencies.

```bash
pip install -r requirements.txt
pip install pytest  # for running tests
```

**Step 5**: Verify the installation.

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
model = EdgeBitForCausalLM(EdgeBitConfig.tiny())
print(f'Model created: {model.count_parameters()[\"total\"]:,} params')
print('All good.')
"
```

If you see "All good." at the end, the setup is complete.

---

## 4. Verifying Everything Works

### Run the Test Suite

```bash
pytest tests/ -v
```

This runs the test suite covering:
- Quantization primitives (STE, ternary, INT8, NF4)
- BitLinear layer (forward, backward, mode switching)
- Full model (forward pass, generation, parameter counting)
- Training scheduler (curriculum phases, state persistence)
- Distillation losses (KL divergence, hidden state alignment)
- Weight packing (compression, roundtrip accuracy)

**Expected result**: all tests pass.

### Run the Smoke Test

The smoke test runs a miniature training loop to validate the entire pipeline:

```bash
python -m training.train --smoke_test --config configs/model_smoke.yaml \
  --output_dir smoke_output/pipeline_check --max_steps 1 --tokenizer simple \
  --batch_size 2 --gradient_accumulation_steps 1
```

**What happens**:
1. Creates 200 synthetic training samples
2. Builds a tiny smoke model with an offline tokenizer
3. Runs the requested number of training steps
4. Saves a checkpoint
5. Validates data loading, masking, loss, backward pass, and checkpoint writing

**What to look for in the output**:
- `loss=X.XXXX` should decrease over time
- `phase=bf16_warmup` transitions to `int8_qat`, then `int4_qat`, then `ternary`
- No errors or NaN values

This takes 2-10 minutes depending on your hardware. It works on CPU.

---

## 5. Exploring the Code

If you want to understand how the model works before training, here is a guided tour of the key files:

### Start Here: The Config

Open [modeling/config.py](modeling/config.py). This defines all model hyperparameters. The `tiny()`, `small_125m()`, and `base_350m()` classmethods create the three training stage configs.

### The Core: BitLinear

Open [modeling/bitlinear.py](modeling/bitlinear.py). This is the quantized linear layer that replaces `nn.Linear`. Read the `forward()` method to see how weights are quantized to ternary during the forward pass and how gradients flow through the STE.

### The Quantization Math

Open [modeling/quant_utils.py](modeling/quant_utils.py). This has the actual quantization functions: `ternary_quantize_absmean()`, `int8_symmetric_quantize()`, `nf4_quantize()`. These are the mathematical primitives that everything else builds on.

### The Full Model

Open [modeling/model.py](modeling/model.py). This assembles everything into a complete transformer: attention (with GQA), FFN (with SwiGLU), decoder layers, and the causal LM wrapper.

### The Training Loop

Open [training/train.py](training/train.py). This is the training script. It loads config from YAML, builds the model, sets up the optimizer and curriculum scheduler, and runs the training loop.

### Interactive Exploration

The Jupyter notebooks provide interactive walkthroughs:

```bash
pip install jupyter matplotlib
jupyter notebook notebooks/
```

- [01_architecture_walkthrough.ipynb](notebooks/01_architecture_walkthrough.ipynb): Build a model, run forward passes, visualize weight distributions, test packing compression.
- [02_training_curriculum_demo.ipynb](notebooks/02_training_curriculum_demo.ipynb): Visualize the quantization curriculum phases, see how weight distributions change.

---

## 6. Preparing Training Data

### Data Format

Training data is JSONL (one JSON object per line) with a `text` field:

```json
{"text": "The capital of France is Paris. It is known for the Eiffel Tower."}
{"text": "Python is a programming language used for web development and data science."}
{"text": "The Raspberry Pi is a small single-board computer developed in the UK."}
```

Each line is one document. Documents longer than 2048 tokens are truncated. Shorter documents are padded.

### How Much Data?

| Training stage | Minimum data | Recommended |
|---|---|---|
| Smoke test | Auto-generated (200 samples) | N/A |
| Stage 1 (Tiny) | 100K samples | 1-2B tokens |
| Stage 2 (125M) | 500K samples | 3-5B tokens |
| Stage 3 (350M) | 1M+ samples | 5-8B tokens |

### Where to Get Data

Good open-source pretraining datasets:
- **SlimPajama** -- a cleaned version of RedPajama, available on HuggingFace
- **The Pile** -- a diverse English text dataset
- **Dolma** -- Allen AI's open pretraining corpus

To convert a HuggingFace dataset to JSONL:

```python
from datasets import load_dataset
import json

ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
with open("train.jsonl", "w") as f:
    for i, row in enumerate(ds):
        if i >= 1_000_000:  # 1M samples
            break
        f.write(json.dumps({"text": row["text"]}) + "\n")
```

### Instruction Tuning Data (Optional)

For instruction-following after pretraining:

```json
{"instruction": "Summarize this article.", "response": "The article discusses..."}
```

or

```json
{"text": "### Instruction:\nSummarize this article.\n\n### Response:\nThe article discusses..."}
```

---

## 7. Training Stage 1: Tiny Model

**Goal**: Confirm the architecture converges through all 4 quantization phases.
**Time**: 3-5 A100-hours. **Cost**: ~$5-8 on cloud GPUs.

### Running It

```bash
# Set your data path and output directory
export DATA_PATH=/path/to/your/train.jsonl
export OUTPUT_DIR=./checkpoints/edgebit-tiny

# Run training
bash scripts/train_tiny.sh
```

Or run the Python command directly:

```bash
python -m training.train \
    --config configs/model_tiny.yaml \
    --data_path /path/to/your/train.jsonl \
    --output_dir ./checkpoints/edgebit-tiny \
    --max_steps 5000 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --logging_steps 25
```

### What to Watch

The logs print a line every 25 steps:

```
step=100 epoch=0 loss=9.2145 lr=4.95e-04 tok/s=12500 phase=bf16_warmup
step=500 epoch=0 loss=7.1234 lr=4.50e-04 tok/s=12800 phase=int8_qat
```

**Key things to check**:

| Checkpoint | Expected |
|---|---|
| Step 100 | Loss < 10.0, phase = bf16_warmup |
| Step 500 | Loss < 8.0, phase transitions to int8_qat |
| Step 1500 | Loss < 7.0, int8 -> int4 transition |
| Step 2750 | Loss < 6.5, int4 -> ternary transition (biggest spike) |
| Step 5000 | Loss < 6.0, ternary phase stable |

### Is It Working?

**Yes, if**: Loss decreases overall, phase transitions cause spikes that recover, final ternary loss is below 6.0.

**No, if**: Loss stays flat or increases, NaN appears, loss never recovers after a phase transition.

### If It Fails

1. **Loss is NaN**: Reduce learning rate to `1e-4`. Check that QK normalization is enabled in config.
2. **Loss never decreases**: Verify data is loading correctly. Check that the data file exists and has proper JSON format.
3. **Out of memory**: Reduce `BATCH_SIZE` to 4 and increase `GRAD_ACCUM` to 8.
4. **Phase transition spike never recovers**: Extend the previous phase by editing `configs/quant_curriculum.yaml`.

---

## 8. Training Stage 2: 125M Model

Only start this after Stage 1 succeeds.

**Goal**: Verify the architecture scales -- does a bigger model do better?
**Time**: 10-15 A100-hours. **Cost**: ~$15-22.

```bash
export DATA_PATH=/path/to/your/train.jsonl
export OUTPUT_DIR=./checkpoints/edgebit-125m
bash scripts/train_125m.sh
```

**Go/No-Go**: The ternary loss at the end should be within 1.5x of the BF16 warmup minimum, and lower than the Stage 1 tiny model's final loss. If 125M doesn't beat 50M, something is wrong -- debug before committing to Stage 3.

---

## 9. Training Stage 3: Full 350M Model

Only start this after Stage 2 succeeds.

**Goal**: Production-quality training of the full model.
**Time**: 25-35 A100-hours. **Cost**: ~$40-50.

```bash
export DATA_PATH=/path/to/your/train.jsonl
export OUTPUT_DIR=./checkpoints/edgebit-350m
bash scripts/train_350m.sh
```

See [RUNBOOK.md](RUNBOOK.md) for the detailed hour-by-hour training guide with:
- Expected loss values at each checkpoint
- Go/no-go criteria at each stage
- What to do if training stalls
- How to resume after interruption

### Monitoring (Optional)

In a separate terminal:

```bash
bash scripts/monitor.sh --output_dir ./checkpoints/edgebit-350m
```

This watches the checkpoint directory and reports training progress.

---

## 10. Evaluating Your Model

After training, evaluate on standard benchmarks:

```bash
python -m eval.run_lm_eval \
    --checkpoint ./checkpoints/edgebit-350m/checkpoint-50000 \
    --tasks mmlu hellaswag winogrande \
    --output results.json
```

This requires the `lm-eval` package: `pip install lm-eval`.

**Realistic expectations for a 350M ternary model**:
- MMLU: ~25-30% (5-shot) -- limited by model size
- HellaSwag: ~35-45% (10-shot)
- Winogrande: ~50-55% (5-shot)

These scores are lower than comparable fp16 models. The value is in the deployment efficiency, not benchmark numbers.

---

## 11. Exporting for Deployment

### HuggingFace Format

Creates `config.json`, `model.safetensors`, and tokenizer files:

```bash
python -m export.export_hf \
    --checkpoint ./checkpoints/edgebit-350m/checkpoint-50000 \
    --output_dir ./hf_export
```

### GGUF Format

Creates a minimal experimental GGUF-like file. This is not yet verified
llama.cpp-compatible ternary execution:

```bash
python -m export.export_gguf \
    --checkpoint ./checkpoints/edgebit-350m/checkpoint-50000 \
    --output ./edgebit-350m.gguf
```

---

## 12. Deploying to Edge Devices

### Raspberry Pi 5

```bash
# On the Pi:
sudo apt install python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pyyaml

# Copy the edgebit-350m directory to the Pi, then:
cd edgebit-350m
OMP_NUM_THREADS=4 python demos/edge_deploy.py --checkpoint /path/to/checkpoint
```

**Expected Pi 5 performance**:
- Model load: ~5 seconds
- Inference: 10-15 tokens/second
- Memory: ~200 MB

### Docker (Any Platform)

```bash
docker build -t edgebit-cpu -f docker/Dockerfile.cpu .
docker run -it edgebit-cpu python demos/edge_deploy.py --preset tiny
```

### Quick Test (No Trained Model Needed)

You can run the edge deployment demo with random weights to test the pipeline:

```bash
python demos/edge_deploy.py --preset tiny
```

This creates a tiny model with random weights and runs generation. The output will be gibberish, but it validates that the inference pipeline works on your device.

---

## 13. Running Benchmarks

### Inference Speed

```bash
# Compare quantization modes on CPU
python -m runtime.bench_runtime \
    --preset tiny \
    --quant_modes none int8 ternary \
    --prompt_tokens 64 \
    --gen_tokens 32
```

### Memory Usage

```bash
python -m runtime.memory_profile --preset tiny --show_layers
```

### Throughput Sweep

```bash
python -m eval.bench_tokens \
    --preset tiny \
    --batch_sizes 1 2 4 \
    --seq_lengths 128 256 512 1024
```

### Full Benchmark Suite

```bash
bash benchmarks/run_all_benchmarks.sh
```

---

## 14. Common Issues and Fixes

### Setup Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'modeling'` | Run commands from the `edgebit-350m/` directory |
| `ImportError: No module named 'transformers'` | `pip install transformers` |
| `ImportError: No module named 'yaml'` | `pip install pyyaml` |
| Tests fail | Verify Python 3.10+ and PyTorch 2.1+. Run `pip install -r requirements.txt` |

### Training Issues

| Problem | Fix |
|---------|-----|
| Loss is NaN | Reduce LR to 1e-4. Verify QK norm is enabled. Enable gradient clipping at 0.5 |
| Loss never decreases | Check data loading. Print a few samples to verify format |
| Loss spike after phase transition never recovers | Give it 500+ steps. If still broken, reduce LR by 2x |
| Out of GPU memory | Reduce batch_size, increase gradient_accumulation_steps |
| Training is very slow | Check GPU utilization with `nvidia-smi`. If low, increase num_workers |
| Pod disconnected | Training auto-resumes. Just run the same script again |

### Inference Issues

| Problem | Fix |
|---------|-----|
| Model output is gibberish | Expected with random weights. Train the model first |
| Very slow inference | Set `OMP_NUM_THREADS` to your CPU core count |
| High memory usage | Use the packed ternary model, not the fp32 checkpoint |

---

## Summary: The Complete Workflow

```
1. Setup          pip install -r requirements.txt
                  pytest tests/ -v                     (verify code works)
                        |
2. Smoke test     python -m training.train --smoke_test  (verify pipeline)
                        |
3. Prepare data   Convert your corpus to JSONL format
                        |
4. Train tiny     bash scripts/train_tiny.sh           (3-5 hours, ~$5)
                        |
5. Train 125M     bash scripts/train_125m.sh           (10-15 hours, ~$15)
                        |
6. Train 350M     bash scripts/train_350m.sh           (25-35 hours, ~$40)
                        |
7. Evaluate       python -m eval.run_lm_eval           (benchmark scores)
                        |
8. Export          python -m export.export_hf           (HuggingFace format)
                  python -m export.export_packed        (verified packed storage)
                  python -m export.export_gguf          (experimental container)
                        |
9. Deploy         python demos/edge_deploy.py           (run on device)
```

Each step builds on the previous one. You can stop at any point -- even Stage 1 alone is a complete exercise in training a quantized transformer.
