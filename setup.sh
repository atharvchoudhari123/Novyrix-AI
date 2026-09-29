#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "========================================"
echo " LUMACORE SETUP"
echo "========================================"
echo

PYTHON_BIN="\${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 was not found."
  echo "Install Python 3.10+ and run this script again."
  exit 1
fi

echo "Using: $($PYTHON_BIN --version 2>&1)"
echo

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating .venv..."
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
else
  echo ".venv already exists."
fi

PYTHON="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: .venv was not created correctly."
  exit 1
fi

echo
echo "Upgrading pip/setuptools/wheel..."
"$PYTHON" -m pip install --upgrade pip setuptools wheel

echo
echo "Installing PyTorch..."
"$PIP" install --upgrade torch torchvision torchaudio

echo
echo "Installing LumaCore dependencies..."
"$PIP" install -r "$ROOT/requirements.txt"

echo
echo "Creating training directories..."
mkdir -p "$ROOT/training/data" "$ROOT/training/output"

echo
echo "Checking PyTorch..."
"$PYTHON" - <<'PY'
import platform
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("MPS available:", bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
print("CUDA available:", bool(torch.cuda.is_available()))
PY

echo
echo "Checking training dependencies..."
"$PYTHON" - <<'PY'
import accelerate
import datasets
import peft
import transformers
import trl

print("accelerate:", accelerate.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
PY

echo
echo "========================================"
echo " SETUP COMPLETE"
echo "========================================"
echo
echo "Activate:"
echo "  source .venv/bin/activate"
echo
echo "Train all tiers:"
echo "  ./train_all.sh"
echo
echo "This script does not start the API."
echo