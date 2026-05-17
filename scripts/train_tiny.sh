#!/bin/bash
set -euo pipefail

# Stage 1: Tiny model (~50M params) — 3-5 A100 hours
# Purpose: Validate architecture convergence, curriculum, and pipeline

CONFIG="configs/model_tiny.yaml"
CURRICULUM="configs/quant_curriculum.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ckpts/edgebit-tiny}"
DATA_PATH="${DATA_PATH:-/mnt/data/pretrain/train.jsonl}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-0.6B}"

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-4}"
MAX_STEPS="${MAX_STEPS:-5000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_STEPS="${LOG_STEPS:-25}"

echo "=== EdgeBit Tiny Training (Stage 1) ==="
echo "Config:    ${CONFIG}"
echo "Output:    ${OUTPUT_DIR}"
echo "Data:      ${DATA_PATH}"
echo "GPUs:      ${NUM_GPUS}"
echo "Steps:     ${MAX_STEPS}"
echo "Batch:     ${BATCH_SIZE} x ${GRAD_ACCUM} accum"

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
        --max_steps "${MAX_STEPS}" \
        --max_seq_length "${SEQ_LEN}" \
        --save_steps "${SAVE_STEPS}" \
        --logging_steps "${LOG_STEPS}" \
        ${RESUME_FLAG}
fi

echo "=== Tiny training complete ==="
echo "Checkpoints at: ${OUTPUT_DIR}"
