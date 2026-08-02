#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"

"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade --no-deps \
  "deepspeed==0.19.3"

"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
  "transformers==5.12.1" \
  "safetensors>=0.5.3" \
  "huggingface-hub>=0.34.0" \
  "timm>=1.0.19"

"${PYTHON_BIN}" -m pip uninstall -y torchaudio

"${PYTHON_BIN}" - <<'PY'
import huggingface_hub
import safetensors
import torch
import torchvision
import transformers
import timm
import accelerate
import deepspeed
import fastwam

if deepspeed.__version__ != "0.19.3":
    raise RuntimeError(
        f"Expected deepspeed==0.19.3, found {deepspeed.__version__} at {deepspeed.__file__}"
    )

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"transformers={transformers.__version__}")
print(f"safetensors={safetensors.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"timm={timm.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"deepspeed={deepspeed.__version__}")
print(f"fastwam={fastwam.__file__}")
PY
