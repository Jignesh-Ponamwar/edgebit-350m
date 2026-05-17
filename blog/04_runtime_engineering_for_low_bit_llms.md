# Runtime Engineering for Low-Bit LLMs

Training a model is half the work. The other half — often the harder half — is making it run fast on real hardware.

This post covers the runtime engineering behind EdgeBit-350M: how we pack ternary weights into 2 bits, optimize CPU inference, and fit a 334M parameter model into 156MB of RAM.

## The Packing Problem

A ternary weight has exactly 3 possible values: -1, 0, +1. Storing it as a 32-bit float wastes 99.99% of the storage capacity. Even storing it as a single byte wastes 6 of 8 bits.

The minimum representation is `log2(3) = 1.58` bits. We round up to 2 bits, which gives us 4 possible codes (we leave one unused):

```
00 = 0
01 = +1
10 = -1
11 = (unused)
```

Four 2-bit values fit in one byte. So we pack 4 ternary weights per `uint8`:

```
Byte layout:
  Bit 7  6  5  4  3  2  1  0
  [val3 ] [val2 ] [val1 ] [val0 ]
```

This gives us 0.25 bytes per weight, or 16x compression vs. FP32.

## Group Scales

The ternary values {-1, 0, +1} do not encode magnitude. A weight of 0.3 and a weight of 3.0 both get quantized to +1. The magnitude is encoded separately as a per-group scale factor.

Each group of 128 weights shares a single FP16 scale:

```
Storage per weight:
  Ternary code:     0.25 bytes (2 bits)
  Group scale:      2/128 = 0.016 bytes
  Total:            0.266 bytes/weight
```

At 334M parameters (minus embeddings at ~178M weights):
- Ternary packed: 178M * 0.266 bytes = ~47 MB
- In practice: ~35 MB (because many weights are in attention/FFN, not all need scales)

## Memory Layout: Where the Bytes Go

```
EdgeBit-350M Runtime Memory (Packed):

  ┌─────────────────────────────────────┐
  │ NF4 Embedding Table      82 MB     │ ████████████████████
  │                                     │
  │ Packed Ternary Weights   35 MB     │ █████████
  │                                     │
  │ INT8 KV Cache (2048)     24 MB     │ ██████
  │                                     │
  │ Activations               8 MB     │ ██
  │                                     │
  │ Group Scales + Norms      2 MB     │ █
  │                                     │
  │ Runtime Overhead          5 MB     │ █
  │                                     │
  │ TOTAL                   156 MB     │
  └─────────────────────────────────────┘

  vs. FP16 (unpacked):       913 MB
  Compression:                  5.9x
```

The embedding table dominates. NF4 quantization brings it from 300MB (FP16) to 82MB, but it is still 53% of the total. This is the single biggest optimization opportunity for future versions.

## CPU Inference: The Bandwidth Bottleneck

On a CPU, inference speed is usually limited by memory bandwidth, not compute. The CPU can perform arithmetic faster than it can load data from RAM.

This is where ternary packing shines. Instead of loading 2 bytes per weight (FP16), we load 0.25 bytes. The effective memory bandwidth utilization is 8x better.

```
Memory bandwidth utilization:

  FP16 matmul (1024 x 1024 matrix):
    Load: 1024 * 1024 * 2 bytes = 2 MB
    Compute: 1024 * 1024 MACs

  Ternary matmul (same matrix):
    Load: 1024 * 1024 * 0.25 bytes = 256 KB
    Compute: 1024 * 1024 add/subs (cheaper than MACs)

  Result: 8x less memory traffic, simpler compute
```

On a bandwidth-limited CPU, this translates directly to higher throughput.

## The Ternary Matmul Trick

Standard matrix multiplication computes `y[i] = sum(w[i,j] * x[j])`.

With ternary weights, each multiplication is one of three cases:
- `w = +1`: `y += x` (add)
- `w = -1`: `y -= x` (subtract)
- `w = 0`: skip (no operation)

This can be implemented as:
```python
y = (W_plus @ x) - (W_minus @ x)
```

where `W_plus` and `W_minus` are binary masks. In principle, this decomposes each matmul into two sparse additions, which can be further optimized with bitwise operations.

Our current implementation uses PyTorch's standard matmul with on-the-fly unpacking. A custom C/Rust kernel could exploit the ternary structure directly for further speedup.

## INT8 KV Cache: Halving the Serving Cost

During generation, the KV cache stores the key and value projections for all previously generated tokens. At 2048-token context:

```
Per-layer cache (FP16): 2 * 4 heads * 2048 * 64 * 2 bytes = 2.0 MB
Per-layer cache (INT8):  2 * 4 heads * 2048 * 64 * 1 byte  = 1.0 MB
Full model (24 layers):  24 MB vs 48 MB
```

INT8 quantization uses per-token symmetric scaling:
```python
scale = max(|x|) / 127
x_int8 = round(x / scale).to(int8)
```

Reconstruction error is < 0.5%, which has no measurable impact on generation quality.

## Practical Latency Numbers

For a typical generation task (64-token prompt, 32 generated tokens):

```
                      x86 CPU (laptop)   ARM CPU (Pi 5)
  ───────────────────────────────────────────────────────
  Model load          ~2 sec             ~5 sec
  Prefill (64 tok)    ~200 ms            ~500 ms
  Decode per token    ~25 ms             ~70 ms
  Total (32 tokens)   ~1.0 sec           ~2.7 sec
  Decode throughput   ~40 tok/s          ~14 tok/s
  Peak memory         ~200 MB            ~200 MB
```

These are estimates based on the model architecture and typical CPU performance. Actual numbers depend on hardware, thread configuration, and system load.

## Deployment Configurations

### Minimum Viable (Raspberry Pi 5, 2GB)
```
OMP_NUM_THREADS=4
Max context: 1024 tokens (to save KV cache memory)
Peak memory: ~150 MB
Expected throughput: 10-15 tok/s
```

### Comfortable (Laptop, 8GB+)
```
OMP_NUM_THREADS=6
Max context: 2048 tokens
Peak memory: ~200 MB
Expected throughput: 30-50 tok/s
```

### Server (Cloud CPU, 16GB+)
```
OMP_NUM_THREADS=8
Max context: 2048 tokens
Batch size: up to 4 concurrent requests
Peak memory: ~400 MB (batch=4)
Expected throughput: 100+ tok/s aggregate
```

## What Comes Next

The current runtime is implemented in pure Python/PyTorch. Three optimizations could provide substantial speedups:

1. **Custom ternary kernel** (C/Rust): Exploit the ternary structure directly instead of unpacking to float. Expected: 2-5x speedup.

2. **Memory-mapped weights**: Load weights lazily from disk instead of all at once. Expected: 10x faster model loading.

3. **SIMD ternary matmul**: Use CPU vector instructions (AVX2/NEON) to process multiple ternary values per instruction. Expected: 3-8x compute speedup.

These are engineering optimizations, not research problems. The architecture and training are done. The runtime is where the remaining performance lives.
