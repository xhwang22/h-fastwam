#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install \
  "torch==2.7.1" \
  "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu128

"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
  "transformers==5.12.1" \
  "safetensors>=0.5.3" \
  "huggingface-hub>=0.34.0"

"${PYTHON_BIN}" -m pip uninstall -y torchaudio

"${PYTHON_BIN}" - <<'PY'
import huggingface_hub
import safetensors
import torch
import torchvision
import transformers

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"transformers={transformers.__version__}")
print(f"safetensors={safetensors.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
PY
