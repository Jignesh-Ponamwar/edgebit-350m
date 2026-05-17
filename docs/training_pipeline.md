# Training Pipeline

## Philosophy

EdgeBit-350M is trained with a single principle: **the model must converge to useful ternary weights, not merely survive quantization.**

This means we do not train a full-precision model and then quantize. We train *through* quantization, using a progressive curriculum that teaches the model to represent knowledge in ternary weights from the start.

---

## Three-Stage Training Strategy

The full training pipeline is designed for **50-60 A100-hours**. We split it into three stages, each validating a hypothesis before committing more compute.

### Stage 1: Tiny Validation (~50M params, 3-5 A100-hours)

**Purpose**: Validate that the architecture converges, the curriculum transitions work, and the training pipeline is correct.

```
Config:  512 hidden, 8 layers, 8 heads, 2 KV heads
Steps:   5,000
Batch:   32 effective (8 x 4 gradient accumulation)
LR:      5e-4
Data:    Subset of pretraining corpus
```

**What to verify**:
- Loss decreases from random (~10-12) to < 6.0
- All four curriculum phases execute without crashes
- Phase transitions cause recoverable loss spikes (< 2x)
- Checkpoint save/load works correctly (kill and resume)
- Gradient norms stay below 10.0

**Go/No-Go**: If the tiny model cannot converge through ternary training, there is an architecture bug. Fix it before scaling up.

### Stage 2: 125M Proof (10-15 A100-hours)

**Purpose**: Prove that the architecture scales, validate distillation mechanics, tune hyperparameters.

```
Config:  768 hidden, 12 layers, 12 heads, 4 KV heads
Steps:   15,000
Batch:   32 effective (4 x 8 gradient accumulation)
LR:      3e-4
Data:    Full pretraining corpus
```

**What to verify**:
- Loss scales predictably from tiny model
- Ternary loss is within 1.5x of BF16 warmup minimum
- Token throughput >= 5000 tok/s on A100
- Perplexity evaluation shows meaningful learning (not random)
- Distillation loss components are balanced

**Go/No-Go**: If 125M does not show clear scaling improvement over tiny, investigate before committing to 350M.

### Stage 3: 350M Final (25-35 A100-hours)

**Purpose**: Full-scale training. This is the production run.

```
Config:  1024 hidden, 24 layers, 16 heads, 4 KV heads
Steps:   50,000
Batch:   32 effective (4 x 8 gradient accumulation)
LR:      3e-4
Warmup:  500 steps
Data:    5-8B tokens pretraining + instruction tuning
```

---

## Progressive Quantization Curriculum

The curriculum is the core innovation in the training pipeline. It solves the fundamental problem of ternary training: **random ternary weights produce meaningless gradients**.

### Phase 1: BF16 Warmup (0-10% of training)

```
quant_mode: "none"
```

All BitLinear layers operate in full BF16 precision. The model learns basic token embeddings, attention patterns, and feature representations without quantization noise.

**What happens internally**: Weights are unconstrained floats. The model builds initial feature detectors and attention heads. Loss drops rapidly from random initialization.

**Why this phase matters**: Without it, the model must simultaneously learn features AND adapt to quantization noise. This is too many degrees of freedom, and training often diverges.

### Phase 2: INT8 Adaptation (10-30% of training)

```
quant_mode: "int8"
```

Weights are quantized to 8-bit integers per group. This is a mild perturbation: INT8 preserves most information but introduces quantization noise.

**What happens internally**: The STE gradient now carries quantization noise. The model adjusts its weight distribution to values that are robust to 8-bit rounding. Outlier weights that were critical in BF16 get redistributed.

**Expected behavior**: Small loss spike (10-20%) at transition, recovering within ~200 steps. If the spike exceeds 2x, reduce learning rate.

### Phase 3: INT4 Adaptation (30-55% of training)

```
quant_mode: "int4"
```

Weights are quantized to 4-bit integers. This is a significant precision reduction (16 distinct values per weight).

**What happens internally**: The model learns to cluster weights into 16 levels. Information that required fine-grained weight distinctions gets re-encoded into patterns across multiple weights. This is where the model begins developing the sparse, clustered weight distributions that ternary training needs.

**Expected behavior**: Moderate loss spike (30-50%), recovering within ~500 steps. The model's weight histograms visibly sharpen around a few dominant values.

### Phase 4: Ternary Final (55-100% of training)

```
quant_mode: "ternary"
```

Weights are quantized to {-1, 0, +1} with per-group absmean scaling. This is the target precision.

**What happens internally**: The model fully commits to ternary representations. Weights cluster into three groups: strongly negative, near-zero (pruned), and strongly positive. The absmean scale per group carries the magnitude information.

**Expected behavior**: Largest loss spike (50-100%), recovering over ~1000 steps. This phase gets 45% of total training time because the model needs extended training to fully adapt. By the end, loss should approach within 1.5x of the BF16 warmup minimum.

---

## Optimizer Configuration

### AdamW

```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),  # lower β2 for ternary stability
    weight_decay=0.01,
)
```

**Why β2=0.95 instead of 0.999**: The STE gradient is noisy because quantization introduces discontinuities. A lower β2 gives the optimizer a shorter memory for second-moment estimates, helping it adapt faster to the changing gradient statistics at phase transitions.

### Learning Rate Schedule

```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1)
```

Cosine annealing with a single cycle. The warmup is handled by the BF16 phase rather than a separate LR warmup, though we include 500 steps of linear LR warmup for the 350M model.

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Essential for ternary training. Without it, the STE can produce arbitrarily large gradients when the quantization boundary shifts many weights simultaneously.

---

## Stability Considerations

### Attention Logit Explosion

**Problem**: In low-bit networks, Q and K values can grow unboundedly. When attention logits exceed ~88 (the float32 exp overflow threshold), softmax produces NaN.

**Solution**: QK RMSNorm normalizes Q and K independently before the dot product:

```python
q = rms_norm(q) * scale_q  # scale_q is learnable, initialized to 1.0
k = rms_norm(k) * scale_k
attn_weights = (q @ k.T) / sqrt(head_dim)
```

**Monitoring**: Log `max(|attn_logits|)` during training. If it exceeds 50, the QK norm may need a lower initial scale.

### Gradient Norm Spikes at Phase Transitions

**Problem**: When the quantization mode changes, gradients spike because all weights are simultaneously perturbed.

**Solution**: The progressive curriculum provides soft transitions. Additionally:
- Gradient clipping caps spike magnitude
- SubLN normalizes internal activations
- The optimizer's momentum provides inertia against sudden changes

**Monitoring**: Log gradient norm per step. Normal range: 0.1-5.0. Warning zone: 5.0-10.0. Danger zone: > 10.0.

### Loss NaN/Inf

**Causes**:
1. Attention logit overflow (fix: verify QK norm is active)
2. Zero-scale quantization groups (fix: clamp scales to min=1e-8)
3. Extreme learning rate (fix: reduce to 1e-4)
4. Data corruption (fix: verify data loading)

**Recovery**: Load the last checkpoint and reduce learning rate by 2x.

---

## Data Pipeline

### Pretraining Data

```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Machine learning models can be deployed on edge devices."}
```

Format: JSONL with a `text` field. Each line is an independent document.

**Tokenization**: We use the Qwen3-0.6B tokenizer (151,936 vocab). Sequences are truncated or padded to `max_seq_length` (2048). Padding tokens get label=-100 to exclude them from the loss.

**Target data volume**: 5-8B tokens for pretraining. This is achievable with:
- RedPajama-v2 (sample)
- SlimPajama
- The Pile (subset)
- Dolma (subset)

### Instruction Data

```json
{"instruction": "Summarize this text.", "response": "The text describes..."}
```

or

```json
{"text": "### Instruction:\nSummarize this text.\n\n### Response:\nThe text describes..."}
```

**Target volume**: 200-500K instruction samples for basic instruction following.

### Streaming for Large Datasets

For datasets that exceed RAM, use `StreamingPretrainingDataset`:

```python
dataset = StreamingPretrainingDataset(
    path="/mnt/data/pretrain/",  # directory of .jsonl files
    tokenizer=tokenizer,
    max_length=2048,
)
```

This reads files lazily and supports multi-worker data loading with automatic file sharding.

---

## Cloud Training Setup

### Single GPU (Recommended for prototyping)

```bash
python -m training.train \
    --config configs/model_350m.yaml \
    --data_path /mnt/data/train.jsonl \
    --output_dir /mnt/ckpts/edgebit-350m
```

### Offline Pipeline Smoke Test

Use this before any expensive run. It trains a tiny smoke config for one step,
saves a checkpoint, and does not download a tokenizer:

```bash
python -m training.train \
    --smoke_test \
    --config configs/model_smoke.yaml \
    --output_dir smoke_output/pipeline_check \
    --max_steps 1 \
    --tokenizer simple \
    --batch_size 2 \
    --gradient_accumulation_steps 1
```

### Multi-GPU with torchrun

```bash
torchrun --nproc_per_node=4 \
    -m training.train \
    --config configs/model_350m.yaml \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```

### DeepSpeed ZeRO-2

```bash
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    -m training.train \
    --use_accelerate \
    --config configs/model_350m.yaml
```

### Cloud Providers

| Provider | GPU | Hourly Cost | Total (50h) |
|----------|-----|-------------|-------------|
| RunPod | A100 80GB | ~$1.50/hr | ~$75 |
| Vast.ai | A100 80GB | ~$1.20/hr | ~$60 |
| Lambda | A100 80GB | ~$1.50/hr | ~$75 |

Use `scripts/runpod_bootstrap.sh` for quick setup on any provider.

---

## Debugging Advice

### Training does not converge

1. Run the smoke test first: `python -m training.train --smoke_test --config configs/model_smoke.yaml --tokenizer simple --max_steps 1`
2. Check that the curriculum YAML is correct
3. Start with the tiny model before scaling up
4. Verify data is loading correctly (check a few samples manually)

### Loss is too high after ternary transition

1. Extend the INT4 phase (give the model more time at 4-bit before ternary)
2. Reduce learning rate to 1e-4 for the ternary phase
3. Check that SubLN and QK norm are enabled
4. Verify gradient clipping is active

### Training is slow

1. Check GPU utilization with `nvidia-smi` — if < 80%, the bottleneck is data loading
2. Increase `num_workers` in the DataLoader
3. Use larger batch size with gradient accumulation
4. Verify data is on fast storage (NVMe, not network mount)

### Out of memory

1. Reduce batch size to 2, increase gradient accumulation to 16
2. Use `accelerate launch --config_file configs/accelerate_config.yaml -m training.train --use_accelerate ...` to shard optimizer states when configured
3. Reduce sequence length to 1024 for initial experiments
4. Check for memory leaks (ensure no tensor accumulation in logging)

---

## What Is Realistically Achievable

At 350M parameters with ternary weights, this model will not compete with frontier models. That is not the goal.

**Realistic expectations**:
- Coherent text generation at short contexts (< 512 tokens)
- Basic instruction following after SFT
- Useful for constrained tasks: classification, extraction, summarization
- Perplexity competitive with full-precision models of similar *effective* capacity (~50-100M fp16 equivalent)

**Why 350M and not larger**: The architecture needs to prove convergence and deployment viability before scaling. A 350M model can be trained in a weekend on a single A100, making it accessible for iteration. The same architecture and training pipeline can scale to 1B+ with more compute.

**The value proposition**: Not model quality, but the complete stack — training, quantization, packing, deployment, benchmarking — as a cohesive, reproducible system.
