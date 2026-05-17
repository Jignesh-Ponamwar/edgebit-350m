# EdgeBit-350M Training Runbook

This is the hour-by-hour guide for training EdgeBit-350M. It tells you what to run, what to expect at each checkpoint, and what to do when things go wrong.

**Total budget**: 50-60 A100 hours across 3 stages (~$60-75 on cloud GPUs).

---

## Before You Start

### Checklist

- [ ] GPU available: A100 80GB (RunPod, Vast.ai, or Lambda)
- [ ] Environment set up: `pip install -r requirements.txt`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Offline smoke test passes: `python -m training.train --smoke_test --config configs/model_smoke.yaml --output_dir smoke_output/pipeline_check --max_steps 1 --tokenizer simple --batch_size 2 --gradient_accumulation_steps 1`
- [ ] Training data ready: JSONL file(s) with `{"text": "..."}` format, 5-8B tokens
- [ ] Disk space: 50+ GB free for checkpoints
- [ ] Running inside `tmux` or `screen` (so training survives disconnects)

### Cloud Setup (RunPod/Vast.ai/Lambda)

```bash
# SSH into your GPU instance, then:
git clone https://github.com/jigneshponamwar/edgebit-350m.git
cd edgebit-350m
bash scripts/setup_env.sh

# Or use the bootstrap script:
bash scripts/runpod_bootstrap.sh --stage tiny
```

---

## Stage 1: Tiny Model (~50M params)

**Goal**: Validate that the architecture converges and the curriculum works.
**Time**: 3-5 A100-hours.
**Cost**: ~$5-8.

### Run It

```bash
# Start a tmux session so training survives disconnects
tmux new -s train

# Run training
DATA_PATH=/mnt/data/train.jsonl \
OUTPUT_DIR=/mnt/ckpts/edgebit-tiny \
bash scripts/train_tiny.sh
```

### Configuration

| Setting | Value |
|---------|-------|
| Model | ~50M params (512 hidden, 8 layers, 8 heads, 2 KV heads) |
| Steps | 5,000 |
| Batch | 8 x 4 gradient accumulation = 32 effective |
| Learning rate | 5e-4 |
| Sequence length | 2048 |

### What to Expect (Hour by Hour)

| Time | Step | Phase | Expected Loss | What to Check |
|------|------|-------|---------------|---------------|
| 0:00 | 0 | bf16_warmup | ~10-12 | Training starts, loss is high (random init) |
| 0:30 | ~500 | bf16 -> int8 | ~8.0 | **Phase transition**. Small loss spike (< 2x). Should recover in ~100 steps |
| 1:00 | ~1000 | int8_qat | ~7.0 | Loss decreasing steadily |
| 1:30 | ~1500 | int8 -> int4 | ~6.5 | **Phase transition**. Moderate spike. Recovers in ~200 steps |
| 2:00 | ~2500 | int4_qat | ~6.0 | Loss decreasing |
| 2:30 | ~2750 | int4 -> ternary | ~7.0-8.0 | **Biggest spike**. This is the hardest transition. |
| 3:00 | ~3500 | ternary | ~6.0-6.5 | Loss recovering |
| 4:00 | ~4500 | ternary | ~5.5-6.0 | Loss stabilizing |
| 5:00 | 5000 | ternary | < 6.0 | **Done.** |

### Go/No-Go

**Proceed to Stage 2 if**:
- Final ternary loss is below 6.0
- All 4 curriculum phases executed without NaN
- Loss recovered after each phase transition (even if slowly)
- Gradient norms stayed below 10.0

**Stop and debug if**:
- Loss is NaN at any point
- Loss never decreases in the bf16_warmup phase
- Ternary loss is above 8.0 after 2000+ steps in ternary phase
- Gradient norms consistently above 10.0

### If Stage 1 Fails

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss flat from step 0 | Data not loading | Check data path. Print a sample: `head -1 /mnt/data/train.jsonl` |
| NaN after int8 transition | Learning rate too high | Set `LR=1e-4` |
| Ternary spike never recovers | Transition too abrupt | Edit `configs/quant_curriculum.yaml` to give INT4 more steps |
| OOM | Batch too large | Set `BATCH_SIZE=4 GRAD_ACCUM=8` |

---

## Stage 2: 125M Model

**Goal**: Prove that the architecture scales -- the 125M model should do better than the 50M model.
**Time**: 10-15 A100-hours.
**Cost**: ~$15-22.

### Run It

```bash
DATA_PATH=/mnt/data/train.jsonl \
OUTPUT_DIR=/mnt/ckpts/edgebit-125m \
bash scripts/train_125m.sh
```

### Configuration

| Setting | Value |
|---------|-------|
| Model | ~125M params (768 hidden, 12 layers, 12 heads, 4 KV heads) |
| Steps | 15,000 |
| Batch | 4 x 8 gradient accumulation = 32 effective |
| Learning rate | 3e-4 |

### What to Expect

| Time | Step | Phase | Expected Loss |
|------|------|-------|---------------|
| 0:00 | 0 | bf16_warmup | ~10-11 |
| 1:00 | ~1500 | int8_qat | < 8.0 |
| 3:00 | ~4500 | int8 -> int4 | < 7.0 |
| 6:00 | ~8000 | int4_qat | < 6.5 |
| 8:00 | ~8250 | int4 -> ternary | Spike to ~7.5 |
| 10:00 | ~11000 | ternary | < 6.0 |
| 13:00 | ~14000 | ternary | < 5.5 |
| 15:00 | 15000 | ternary | < 5.5 |

### Go/No-Go

**Proceed to Stage 3 if**:
- Final loss is lower than Stage 1 final loss
- Ternary loss is within 1.5x of BF16 warmup minimum
- Token throughput is >= 5000 tok/s on A100

**Stop and debug if**:
- 125M model does not beat 50M model
- Loss diverges during ternary phase

---

## Stage 3: Full 350M Model

**Goal**: Production training. This is the real run.
**Time**: 25-35 A100-hours.
**Cost**: ~$40-50.

### Run It

```bash
# Start in tmux
tmux new -s train350

DATA_PATH=/mnt/data/train.jsonl \
OUTPUT_DIR=/mnt/ckpts/edgebit-350m \
bash scripts/train_350m.sh
```

### Configuration

| Setting | Value |
|---------|-------|
| Model | ~334M params (1024 hidden, 24 layers, 16 heads, 4 KV heads) |
| Steps | 50,000 |
| Batch | 4 x 8 gradient accumulation = 32 effective |
| Learning rate | 3e-4 |
| Warmup | 500 steps |

### Monitoring (Run in a Separate Terminal)

```bash
tmux new -s monitor
bash scripts/monitor.sh --output_dir /mnt/ckpts/edgebit-350m
```

### What to Expect (Hour by Hour)

| Hour | Step | Phase | Expected Loss | Notes |
|------|------|-------|---------------|-------|
| 0 | 0 | bf16_warmup | ~10-11 | Training starts |
| 2 | ~2,000 | bf16_warmup | < 7.5 | Loss dropping fast |
| 4 | ~5,000 | bf16 -> int8 | < 6.5 | First transition. Small spike |
| 8 | ~10,000 | int8 -> int4 | < 6.0 | Second transition. Moderate spike |
| 12 | ~15,000 | int4_qat | < 5.5 | INT4 phase, steady improvement |
| 18 | ~25,000 | int4 -> ternary | Spike to ~7.0 | **Biggest transition** |
| 22 | ~30,000 | ternary | < 5.5 | Recovery in progress |
| 26 | ~35,000 | ternary | < 5.0 | Getting close to final quality |
| 30 | ~45,000 | ternary | < 4.8 | Fine-tuning in ternary |
| 33 | 50,000 | ternary | < 4.8 | **Done** |

### Checkpoints

Checkpoints are saved every 2,000 steps to `OUTPUT_DIR/checkpoint-NNNNN/`. Each contains:
- `training_state.pt` -- model weights, optimizer state, scheduler state
- `config.json` -- model configuration

### Resume After Interruption

Training auto-detects the latest checkpoint. Just run the same command:

```bash
# This automatically resumes from the latest checkpoint
bash scripts/train_350m.sh
```

Or specify a checkpoint explicitly:

```bash
python -m training.train \
    --config configs/model_350m.yaml \
    --data_path /mnt/data/train.jsonl \
    --output_dir /mnt/ckpts/edgebit-350m \
    --resume_from /mnt/ckpts/edgebit-350m/checkpoint-25000
```

---

## After Training

### 1. Evaluate

```bash
python -m eval.run_lm_eval \
    --checkpoint /mnt/ckpts/edgebit-350m/checkpoint-50000 \
    --tasks mmlu hellaswag gsm8k winogrande \
    --output results.json
```

### 2. Export

```bash
# HuggingFace format
python -m export.export_hf \
    --checkpoint /mnt/ckpts/edgebit-350m/checkpoint-50000 \
    --output_dir ./hf_export

# Minimal experimental GGUF-like container
python -m export.export_gguf \
    --checkpoint /mnt/ckpts/edgebit-350m/checkpoint-50000 \
    --output ./edgebit-350m.gguf

# Verified packed EdgeBit storage artifact
python -m export.export_packed \
    --checkpoint /mnt/ckpts/edgebit-350m/checkpoint-50000 \
    --output ./edgebit-packed.pt
```

### 3. Benchmark

```bash
python -m runtime.bench_runtime \
    --checkpoint /mnt/ckpts/edgebit-350m/checkpoint-50000 \
    --preset base \
    --device cpu \
    --quant_modes none ternary
```

### 4. Deploy

```bash
python demos/edge_deploy.py \
    --checkpoint /mnt/ckpts/edgebit-350m/checkpoint-50000 \
    --preset base
```

---

## Troubleshooting Reference

### Loss Explodes After Phase Transition

This is the most common issue. Phase transitions always cause loss spikes. They should recover within 200-500 steps.

**If it does not recover**:
1. Reduce learning rate by 2x for the remaining training
2. Edit `configs/quant_curriculum.yaml` to extend the previous phase
3. Check gradient norms -- if consistently > 10.0, reduce `--max_grad_norm` to 0.5

### NaN Loss

| Possible cause | How to check | Fix |
|---|---|---|
| Attention logit overflow | Log shows huge attention values | Verify `use_qk_norm: true` in config |
| Zero-scale quantization group | Check for 0.0 in scale values | Update quant_utils.py clamp min to 1e-8 |
| Learning rate too high | Happens early in training | Reduce to 1e-4 |
| Data corruption | Bad JSON in training data | Validate: `python -c "import json; [json.loads(l) for l in open('data.jsonl')]"` |

**Recovery**: Load the last good checkpoint and reduce learning rate by 2x:
```bash
python -m training.train \
    --config configs/model_350m.yaml \
    --resume_from /mnt/ckpts/edgebit-350m/checkpoint-LAST_GOOD \
    --learning_rate 1.5e-4
```

### Out of Memory on A100

| Fix | How |
|---|---|
| Reduce batch size | `BATCH_SIZE=2` |
| Increase gradient accumulation | `GRAD_ACCUM=16` (keeps effective batch the same) |
| Reduce sequence length | `SEQ_LEN=1024` (for experiments only) |
| Use Accelerate/DeepSpeed config | `accelerate launch --config_file configs/accelerate_config.yaml -m training.train --use_accelerate ...` |

### Training Stalls (Loss Stops Decreasing)

1. Check that the learning rate is not at 0: look at the `lr=` value in logs
2. Check the training phase: the model may need more time in the current phase
3. Verify data diversity: are batches varied, or is the same data repeating?
4. Check gradient norms: if very small (< 0.01), the model may have converged for this phase

### Pod Disconnection

1. Always use `tmux` or `screen`: `tmux new -s train`
2. Training auto-resumes from the latest checkpoint
3. The monitor script sends alerts if training stalls (if configured with webhook)

### Checkpoint Corruption

Keep at least the last 3 checkpoints. Delete older ones to save disk space.

Verify a checkpoint:
```bash
python -c "
import torch
state = torch.load('checkpoint-XXXXX/training_state.pt', map_location='cpu', weights_only=False)
print(f'Step: {state[\"step\"]}')
print(f'Loss: {state[\"loss\"]:.4f}')
print(f'Keys: {list(state.keys())}')
print('Checkpoint OK')
"
```
