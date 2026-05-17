#!/bin/bash
set -euo pipefail

# Run all benchmarks and save results.
# Usage: bash benchmarks/run_all_benchmarks.sh [--checkpoint /path/to/ckpt]

CHECKPOINT="${1:-}"
RESULTS_DIR="benchmarks/results"
PRESET="${PRESET:-tiny}"

mkdir -p "${RESULTS_DIR}"

echo "=== EdgeBit Benchmark Suite ==="
echo "Preset: ${PRESET}"
echo "Results: ${RESULTS_DIR}"
if [ -n "${CHECKPOINT}" ]; then
    echo "Checkpoint: ${CHECKPOINT}"
fi
echo ""

echo "[1/4] Memory Profile..."
python -m runtime.memory_profile \
    --preset "${PRESET}" \
    --output_json "${RESULTS_DIR}/memory_profile.json" \
    --show_layers \
    2>&1 | tee "${RESULTS_DIR}/memory_profile.txt"

echo ""
echo "[2/4] Runtime Benchmark..."
BENCH_ARGS="--preset ${PRESET} --quant_modes none int8 ternary --output_json ${RESULTS_DIR}/runtime_bench.json"
if [ -n "${CHECKPOINT}" ]; then
    BENCH_ARGS="--checkpoint ${CHECKPOINT} ${BENCH_ARGS}"
fi
python -m runtime.bench_runtime ${BENCH_ARGS} \
    2>&1 | tee "${RESULTS_DIR}/runtime_bench.txt"

echo ""
echo "[3/4] Throughput Sweep..."
SWEEP_ARGS="--preset ${PRESET} --batch_sizes 1 2 4 --seq_lengths 128 256 512 --output_json ${RESULTS_DIR}/throughput_sweep.json"
if [ -n "${CHECKPOINT}" ]; then
    SWEEP_ARGS="--checkpoint ${CHECKPOINT} ${SWEEP_ARGS}"
fi
python -m eval.bench_tokens ${SWEEP_ARGS} \
    2>&1 | tee "${RESULTS_DIR}/throughput_sweep.txt"

echo ""
echo "[4/4] Packing Stats..."
python -c "
import torch
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
from runtime.pack_ternary import compute_packing_stats
import json

config = EdgeBitConfig.${PRESET}() if hasattr(EdgeBitConfig, '${PRESET}') else EdgeBitConfig.tiny()
model = EdgeBitForCausalLM(config)
stats = {}
for name, p in model.named_parameters():
    if p.ndim == 2 and 'weight' in name:
        s = compute_packing_stats(p)
        stats[name] = s
total_fp16 = sum(s['fp16_mb'] for s in stats.values())
total_packed = sum(s['packed_mb'] for s in stats.values())
print(f'Weight layers: {len(stats)}')
print(f'FP16 total: {total_fp16:.2f} MB')
print(f'Packed total: {total_packed:.2f} MB')
print(f'Compression: {total_fp16/max(total_packed,0.001):.1f}x')
with open('${RESULTS_DIR}/packing_stats.json', 'w') as f:
    json.dump({'total_fp16_mb': total_fp16, 'total_packed_mb': total_packed, 'layers': len(stats)}, f, indent=2)
" 2>&1 | tee "${RESULTS_DIR}/packing_stats.txt"

echo ""
echo "=== All benchmarks complete ==="
echo "Results saved to ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
