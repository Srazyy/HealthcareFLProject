#!/usr/bin/env bash
set -e

echo "== FL-Healthcare setup (Linux) =="

PY=python3
if ! command -v $PY &> /dev/null; then
  echo "python3 not found. Install Python 3.10/3.11 first."
  exit 1
fi

$PY -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected — installing CUDA-enabled torch (cu121)."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "No NVIDIA GPU detected — installing CPU-only torch."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements/base.txt

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo "Verify GPU (CUDA) availability with:"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
