#!/bin/bash
set -euo pipefail

echo "=== EdgeBit-350M Environment Setup ==="

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python ${PYTHON_VERSION}+"
    exit 1
fi

echo "Creating virtual environment in ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing PyTorch (CUDA ${CUDA_VERSION})..."
if [ "${CUDA_VERSION}" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"
fi

echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Installing development dependencies..."
pip install pytest pytest-cov ruff mypy

echo "Installing optional dependencies..."
pip install psutil gputil || true
pip install lm-eval || echo "WARN: lm-eval install failed (optional)"

echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

python3 -c "
from modeling.config import EdgeBitConfig
from modeling.model import EdgeBitForCausalLM
config = EdgeBitConfig.tiny()
model = EdgeBitForCausalLM(config)
params = model.count_parameters()
print(f'EdgeBit tiny model: {params[\"total\"]:,} params')
print('Setup verified OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Run smoke test: python -m training.train --smoke_test --config configs/model_tiny.yaml"
