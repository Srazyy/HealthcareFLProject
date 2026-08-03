#!/usr/bin/env bash
set -e

echo "== FL-Healthcare setup (macOS) =="

PY=python3
if ! command -v $PY &> /dev/null; then
  echo "python3 not found. Install Python 3.10/3.11 first (e.g. 'brew install python@3.11')."
  exit 1
fi

$PY -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  echo "Apple Silicon detected — installing torch with MPS support."
else
  echo "Intel Mac detected — installing CPU torch build."
fi
# Default PyPI torch build works for both cases on macOS.
pip install torch torchvision torchaudio

pip install -r requirements/base.txt

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo "Verify GPU (MPS) availability with:"
echo "  python -c \"import torch; print(torch.backends.mps.is_available())\""
