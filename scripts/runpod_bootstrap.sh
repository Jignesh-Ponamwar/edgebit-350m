#!/bin/bash
set -euo pipefail

# Bootstrap script for RunPod / Vast.ai / Lambda cloud instances.
# Run this on a fresh A100 pod to set up the training environment.
#
# Usage:
#   curl -sSL <raw-url>/scripts/runpod_bootstrap.sh | bash
#   OR
#   bash scripts/runpod_bootstrap.sh [--stage tiny|125m|350m] [--data-url URL]

STAGE="${STAGE:-tiny}"
REPO_URL="${REPO_URL:-}"
DATA_URL="${DATA_URL:-}"
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/edgebit-350m"
DATA_DIR="${WORKSPACE}/data"
CKPT_DIR="${WORKSPACE}/ckpts"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        --data-url) DATA_URL="$2"; shift 2 ;;
        --repo-url) REPO_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== RunPod Bootstrap for EdgeBit-350M ==="
echo "Stage:     ${STAGE}"
echo "Workspace: ${WORKSPACE}"

# System setup
echo "[1/6] System packages..."
apt-get update -qq && apt-get install -yqq \
    git wget curl htop tmux nvtop tree jq \
    2>/dev/null || true

# Check GPU
echo "[2/6] GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARN: No GPU detected"

# Clone or verify project
echo "[3/6] Project setup..."
if [ -n "${REPO_URL}" ] && [ ! -d "${PROJECT_DIR}" ]; then
    git clone "${REPO_URL}" "${PROJECT_DIR}"
elif [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: Project not found at ${PROJECT_DIR} and no REPO_URL set."
    echo "Copy your project to ${PROJECT_DIR} or set REPO_URL."
    exit 1
fi
cd "${PROJECT_DIR}"

# Python environment
echo "[4/6] Python environment..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q
pip install deepspeed -q 2>/dev/null || echo "WARN: deepspeed install failed"

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.1f}GB)')
"

# Data
echo "[5/6] Data setup..."
mkdir -p "${DATA_DIR}" "${CKPT_DIR}"
if [ -n "${DATA_URL}" ]; then
    echo "Downloading data from ${DATA_URL}..."
    wget -q -O "${DATA_DIR}/train.jsonl" "${DATA_URL}" || \
    curl -sSL "${DATA_URL}" -o "${DATA_DIR}/train.jsonl"
    wc -l "${DATA_DIR}/train.jsonl"
fi

if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "No training data found. Generating synthetic data for smoke test..."
    python -c "
from training.data import create_synthetic_data
create_synthetic_data('${DATA_DIR}/train.jsonl', n_samples=5000)
"
fi

# Smoke test
echo "[6/6] Smoke test..."
python -m training.train \
    --smoke_test \
    --config configs/model_tiny.yaml \
    --output_dir "${CKPT_DIR}/smoke" \
    --data_path "${DATA_DIR}/train.jsonl"

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "To start training:"
case "${STAGE}" in
    tiny)
        echo "  DATA_PATH=${DATA_DIR}/train.jsonl OUTPUT_DIR=${CKPT_DIR}/edgebit-tiny bash scripts/train_tiny.sh"
        ;;
    125m)
        echo "  DATA_PATH=${DATA_DIR}/train.jsonl OUTPUT_DIR=${CKPT_DIR}/edgebit-125m bash scripts/train_125m.sh"
        ;;
    350m)
        echo "  DATA_PATH=${DATA_DIR}/train.jsonl OUTPUT_DIR=${CKPT_DIR}/edgebit-350m bash scripts/train_350m.sh"
        ;;
esac
echo ""
echo "Monitor with:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f ${CKPT_DIR}/edgebit-${STAGE}/training.log"
echo ""
echo "Use tmux to persist training across SSH disconnects:"
echo "  tmux new -s train"
