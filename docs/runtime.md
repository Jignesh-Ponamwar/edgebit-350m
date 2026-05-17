# Runtime Engineering

## Design Goal

The EdgeBit runtime is designed for one scenario: **autoregressive text generation on a CPU with limited RAM**. The verified path today is PyTorch fake-quant inference, INT8 KV cache, and packed checkpoint storage. Custom packed ternary matmul kernels are a roadmap item.

Target device profile:
- Raspberry Pi 5 (4GB RAM, ARM Cortex-A76, no GPU)
- Commodity x86 laptop (8GB RAM, no discrete GPU)
- Cloud CPU instances (for cost-sensitive inference)

---

## Ternary Weight Packing

### The Problem

Ternary weights have only 3 possible values: {-1, 0, +1}. Storing each as a float32 wastes 30 of 32 bits. Even float16 wastes 14 of 16 bits.

### The Solution: 2-Bit Packing for Storage

Each ternary value needs only 2 bits:

```
Encoding:
  00 = 0
  01 = +1
  10 = -1
  11 = unused (reserved)

Packing: 4 ternary values per byte (uint8)

  Byte layout:
  ┌──────┬──────┬──────┬──────┐
  │ val3 │ val2 │ val1 │ val0 │
  │ [7:6]│ [5:4]│ [3:2]│ [1:0]│
  └──────┴──────┴──────┴──────┘
```

### Compression

```
                    Storage per weight
  ─────────────────────────────────────
  float32:          4.0 bytes
  float16:          2.0 bytes
  2-bit packed:     0.25 bytes
  + group scales:   ~0.27 bytes total

  Compression vs fp16:  7.4x
  Compression vs fp32: 14.8x
```

### Pack/Unpack Implementation

```python
# Pack: 4 ternary values → 1 byte
packed = (val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)).to(uint8)

# Unpack: 1 byte → 4 ternary values
val0 = (packed >> 0) & 0x03  # then map: 0→0, 1→+1, 2→-1
val1 = (packed >> 2) & 0x03
val2 = (packed >> 4) & 0x03
val3 = (packed >> 6) & 0x03
```

### Group Scales

Each group of 128 ternary weights has an associated float16 scale factor. The scale encodes the magnitude that the ternary values represent:

```
reconstructed_weight = ternary_value * group_scale

Storage overhead: 2 bytes per 128 weights = 0.016 bytes/weight
Total: 0.25 + 0.016 = 0.266 bytes/weight
```

---

## Memory Budget Status

### Full Model (350M params)

```
┌────────────────────────────────────────────────────────────┐
│                  Memory Layout (Packed)                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Embedding (NF4)         ███████████████████  82 MB        │
│  BitLinear (2-bit)       ████████  35 MB                   │
│  Scales + Norms          █  2 MB                           │
│  KV Cache (INT8, 2048)   █████  24 MB                      │
│  Activations             ██  8 MB                          │
│  Runtime overhead         █  5 MB                          │
│                                                            │
│  Total:                  ~156 MB                           │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  vs. FP16 model:         ~913 MB                          │
│  Compression:             5.9x                            │
└────────────────────────────────────────────────────────────┘
```

### KV Cache Scaling

```
KV Cache Memory (INT8, batch=1):

  Seq Length    FP16 Cache    INT8 Cache    Savings
  ──────────────────────────────────────────────────
  128           1.5 MB        0.75 MB       2x
  256           3.0 MB        1.5 MB        2x
  512           6.0 MB        3.0 MB        2x
  1024         12.0 MB        6.0 MB        2x
  2048         24.0 MB       12.0 MB        2x
```

### Device Fit Analysis

Use `python -m runtime.memory_profile` for active PyTorch runtime memory and
`python -m export.export_packed` for packed checkpoint storage size. The
projected ~156MB full-runtime target should not be reported as measured until
active NF4 embedding storage and packed ternary matmul are implemented.

```
Device              RAM       Model     KV Cache   Available   Fits?
                              (packed)  (2048)     for OS
──────────────────────────────────────────────────────────────────────
Raspberry Pi 5      4 GB      156 MB    24 MB      3.8 GB      Yes
Pi 5 (2GB model)    2 GB      156 MB    24 MB      1.8 GB      Yes
Old laptop          4 GB      156 MB    24 MB      3.8 GB      Yes
Modern laptop       8 GB      156 MB    24 MB      7.8 GB      Yes
Cloud CPU           2 GB      156 MB    24 MB      1.8 GB      Yes
```

---

## CPU Inference Optimization

### Why CPU?

Edge devices rarely have GPUs. Even when they do (e.g., mobile GPUs), the overhead of GPU memory management can exceed the compute savings at this model size.

For a 350M ternary model:
- Weight storage is tiny when BitLinear layers are exported as packed tensors
- Target packed matmuls can be additions/subtractions (no FP multiply)
- Batch size is always 1 (interactive inference)
- The bottleneck is memory bandwidth, not compute

### Ternary Matmul on CPU

Standard matmul: `y[i] = Σ w[i,j] * x[j]` requires multiply-accumulate.

Ternary matmul: `y[i] = Σ sign[i,j] * x[j]` reduces to:

```python
for each output element y[i]:
    y[i] = 0
    for each weight w[i,j]:
        if w[i,j] == +1: y[i] += x[j]
        if w[i,j] == -1: y[i] -= x[j]
        # if w[i,j] == 0:  skip (no operation)
```

This is ~2x faster than float multiply-accumulate on most CPUs because:
1. Integer add/subtract is cheaper than FP multiply
2. Zero weights are skipped entirely
3. The packed format is cache-friendly (4 values per byte)

The current Python runtime reloads packed storage into regular tensors for
PyTorch inference; it does not yet execute packed ternary matmul directly.

### Thread Configuration

```bash
# Optimal for Raspberry Pi 5 (4 cores)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Optimal for laptop (8 cores, but shared with OS)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
```

---

## Inference Pipeline

### Autoregressive Generation

```
Input: "The capital of France is"

Step 1: Prefill
  ┌──────────────────────┐
  │ Tokenize input       │  "The capital of France is" → [464, 6864, 315, 9822, 374]
  │ Forward pass (full)  │  Process all 5 tokens at once
  │ Populate KV cache    │  Cache K,V for all 5 positions
  │ Get logits[4]        │  Next-token prediction from last position
  └──────────────────────┘

Step 2..N: Decode (one token at a time)
  ┌──────────────────────┐
  │ Forward pass (1 tok) │  Process only the new token
  │ Append to KV cache   │  Cache grows by 1 position
  │ Get logits           │  Next-token prediction
  │ Sample next token    │  Apply temperature, top-k, etc.
  └──────────────────────┘
  Repeat until EOS or max_length

Output: "The capital of France is Paris."
```

### Latency Breakdown

For a typical generation (64 prompt tokens, 32 generated tokens):

```
Operation            Time (CPU)    Notes
─────────────────────────────────────────────
Tokenization         ~1 ms         Negligible
Prefill (64 tokens)  ~200 ms       Processes prompt in parallel
Decode (32 tokens)   ~800 ms       ~25 ms per token
Detokenization       ~0.1 ms       Negligible
─────────────────────────────────────────────
Total                ~1000 ms
Throughput           ~32 tok/s     Decode phase
```

These are estimates for an x86 CPU. ARM (Raspberry Pi) will be approximately 2-3x slower.

---

## Deployment Strategy

### Option 1: Python Runtime (Simplest)

```python
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
from transformers import AutoTokenizer

config = EdgeBitConfig()
model = EdgeBitForCausalLM(config)
model.load_state_dict(torch.load("model.pt"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### Option 2: Docker Container (Portable)

```bash
docker build -t edgebit-cpu -f docker/Dockerfile.cpu .
docker run -it edgebit-cpu python -c "
from modeling.model import EdgeBitForCausalLM
# ... inference code
"
```

### Option 3: GGUF Export (Experimental Container)

```bash
python -m export.export_gguf \
    --checkpoint /path/to/checkpoint \
    --output edgebit-350m.gguf
```

The current GGUF writer is a minimal experimental container. It does not yet
provide verified llama.cpp-compatible execution for EdgeBit's custom ternary
layers.

### Raspberry Pi 5 Deployment

```bash
# On the Pi:
sudo apt install python3-pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pyyaml

# Copy model files
scp -r edgebit-350m/ pi@raspberrypi:/home/pi/

# Run
cd /home/pi/edgebit-350m
OMP_NUM_THREADS=4 python demos/cli_assistant.py --checkpoint /path/to/model
```

**Expected Pi 5 performance**:
- Model load: ~5 seconds
- Prefill (64 tokens): ~500 ms
- Decode: ~10-15 tok/s
- Memory: ~200 MB RSS

---

## Benchmarking

### Running Benchmarks

```bash
# Quick benchmark
python -m runtime.bench_runtime --preset tiny --quant_modes none ternary

# Full benchmark suite
python -m runtime.bench_runtime \
    --preset base \
    --prompt_tokens 64 \
    --gen_tokens 32 \
    --device cpu \
    --quant_modes none int8 ternary \
    --output_json benchmarks/results/runtime_bench.json

# Memory profiling
python -m runtime.memory_profile --preset base --show_layers

# Throughput sweep
python -m eval.bench_tokens \
    --preset base \
    --batch_sizes 1 2 4 \
    --seq_lengths 128 256 512 1024
```

### Interpreting Results

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Decode tok/s (CPU) | > 20 | 10-20 | < 10 |
| TTFT (64 tokens) | < 200 ms | 200-500 ms | > 500 ms |
| Peak memory | < 200 MB | 200-500 MB | > 500 MB |
| Model file size | < 100 MB | 100-200 MB | > 200 MB |
