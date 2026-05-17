#!/bin/bash
set -euo pipefail

# Stage 3: 350M model — 25-35 A100 hours
# Purpose: Full-scale training with progressive quantization curriculum

CONFIG="configs/model_350m.yaml"
CURRICULUM="configs/quant_curriculum.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ckpts/edgebit-350m}"
DATA_PATH="${DATA_PATH:-/mnt/data/pretrain/train.jsonl}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-0.6B}"

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-3e-4}"
MAX_STEPS="${MAX_STEPS:-50000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LOG_STEPS="${LOG_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

echo "=== EdgeBit 350M Training (Stage 3 — Final) ==="
echo "Config:    ${CONFIG}"
echo "Output:    ${OUTPUT_DIR}"
echo "Data:      ${DATA_PATH}"
echo "GPUs:      ${NUM_GPUS}"
echo "Steps:     ${MAX_STEPS}"
echo "Batch:     ${BATCH_SIZE} x ${GRAD_ACCUM} accum x ${NUM_GPUS} GPUs"
echo "Effective: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS * SEQ_LEN)) tokens/step"

RESUME_FLAG=""
LATEST_CKPT=$(find "${OUTPUT_DIR}" -name "training_state.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
if [ -n "${LATEST_CKPT}" ]; then
    RESUME_DIR=$(dirname "${LATEST_CKPT}")
    echo "Resuming from: ${RESUME_DIR}"
    RESUME_FLAG="--resume_from ${RESUME_DIR}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" \
        -m training.train \
        --config "${CONFIG}" \
        --curriculum "${CURRICULUM}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tokenizer "${TOKENIZER}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${LR}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --max_steps "${MAX_STEPS}" \
        --max_seq_length "${SEQ_LEN}" \
        --save_steps "${SAVE_STEPS}" \
        --logging_steps "${LOG_STEPS}" \
        ${RESUME_FLAG}
else
    python -m training.train \
        --config "${CONFIG}" \
        --curriculum "${CURRICULUM}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tokenizer "${TOKENIZER}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${LR}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --max_steps "${MAX_STEPS}" \
        --max_seq_length "${SEQ_LEN}" \
        --save_steps "${SAVE_STEPS}" \
        --logging_steps "${LOG_STEPS}" \
        ${RESUME_FLAG}
fi

echo "=== 350M training complete ==="
echo "Final checkpoint: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run eval:   python -m eval.run_lm_eval --checkpoint ${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
echo "  2. Export HF:  python -m export.export_hf --checkpoint ${OUTPUT_DIR}/checkpoint-${MAX_STEPS} --output_dir ./hf_export"
echo "  3. Export GGUF: python -m export.export_gguf --checkpoint ${OUTPUT_DIR}/checkpoint-${MAX_STEPS} --output ./edgebit-350m.gguf"
