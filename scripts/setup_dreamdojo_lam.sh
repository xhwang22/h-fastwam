#!/usr/bin/env bash
set -euo pipefail

readonly DREAMDOJO_GIT_URL="https://github.com/NVIDIA/DreamDojo.git"
readonly DREAMDOJO_COMMIT="02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"
readonly DREAMDOJO_HF_REPO="nvidia/DreamDojo"
readonly DREAMDOJO_HF_REVISION="89d029e10816d2995d700cb8ba06f171e0504203"
readonly DREAMDOJO_LAM_FILENAME="LAM_400k.ckpt"
readonly DREAMDOJO_LAM_SIZE="8518404954"
readonly DREAMDOJO_LAM_SHA256="d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f"
readonly DREAMDOJO_SOURCE_LICENSE="Apache-2.0"
readonly DREAMDOJO_CHECKPOINT_LICENSE="NVIDIA Open Model License"

usage() {
  cat <<EOF
Usage: ACCEPT_NVIDIA_OPEN_MODEL_LICENSE=1 $0

Pins and validates DreamDojo source/checkpoint metadata. Environment overrides:
  DREAMDOJO_ROOT       checkout path
  DREAMDOJO_CHECKPOINT LAM_400k.ckpt path
  UV_BIN               uv executable

This script can obtain missing assets. It never runs unless invoked explicitly.
EOF
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected arguments: $*" >&2
  usage >&2
  exit 2
fi

if [[ "${ACCEPT_NVIDIA_OPEN_MODEL_LICENSE:-0}" != "1" ]]; then
  echo "ERROR: set ACCEPT_NVIDIA_OPEN_MODEL_LICENSE=1 only after reviewing and accepting the NVIDIA Open Model License." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DREAMDOJO_ROOT="${DREAMDOJO_ROOT:-${REPO_ROOT}/external/DreamDojo}"
DREAMDOJO_CHECKPOINT="${DREAMDOJO_CHECKPOINT:-${DREAMDOJO_ROOT}/checkpoints/DreamDojo/${DREAMDOJO_LAM_FILENAME}}"
UV_BIN="${UV_BIN:-uv}"

if [[ -e "${DREAMDOJO_ROOT}" && ! -d "${DREAMDOJO_ROOT}/.git" ]]; then
  echo "ERROR: DREAMDOJO_ROOT exists but is not a git checkout: ${DREAMDOJO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DREAMDOJO_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${DREAMDOJO_ROOT}")"
  git clone --no-checkout "${DREAMDOJO_GIT_URL}" "${DREAMDOJO_ROOT}"
  git -C "${DREAMDOJO_ROOT}" fetch --depth=1 origin "${DREAMDOJO_COMMIT}"
  git -C "${DREAMDOJO_ROOT}" checkout --detach "${DREAMDOJO_COMMIT}"
fi

actual_commit="$(git -C "${DREAMDOJO_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${DREAMDOJO_COMMIT}" ]]; then
  echo "ERROR: existing DreamDojo checkout is ${actual_commit}; expected ${DREAMDOJO_COMMIT}." >&2
  echo "Use a new DREAMDOJO_ROOT; this script will not alter an existing checkout." >&2
  exit 1
fi
if [[ ! -f "${DREAMDOJO_ROOT}/external/lam/model.py" ]]; then
  echo "ERROR: pinned DreamDojo LAM source is missing: ${DREAMDOJO_ROOT}/external/lam/model.py" >&2
  exit 1
fi

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "ERROR: uv is required to create DreamDojo's isolated environment." >&2
  exit 1
fi
if [[ ! -x "${DREAMDOJO_ROOT}/.venv/bin/python" ]]; then
  "${UV_BIN}" --directory "${DREAMDOJO_ROOT}" sync --extra=cu128 --python 3.10
fi
DREAMDOJO_PYTHON="${DREAMDOJO_ROOT}/.venv/bin/python"
"${UV_BIN}" pip install --python "${DREAMDOJO_PYTHON}" \
  "lightning==2.5.5" "huggingface_hub==0.34.4"
PYTHONPATH="${DREAMDOJO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${DREAMDOJO_PYTHON}" -c 'from external.lam.model import LAM; assert LAM is not None'

mkdir -p "$(dirname "${DREAMDOJO_CHECKPOINT}")"
if [[ ! -f "${DREAMDOJO_CHECKPOINT}" ]]; then
  HF_CLI="${DREAMDOJO_ROOT}/.venv/bin/hf"
  if [[ ! -x "${HF_CLI}" ]]; then
    echo "ERROR: Hugging Face CLI is missing from the isolated DreamDojo environment." >&2
    exit 1
  fi
  "${HF_CLI}" download \
    "${DREAMDOJO_HF_REPO}" "${DREAMDOJO_LAM_FILENAME}" \
    --revision "${DREAMDOJO_HF_REVISION}" \
    --local-dir "$(dirname "${DREAMDOJO_CHECKPOINT}")"
fi

actual_size="$(stat -c '%s' "${DREAMDOJO_CHECKPOINT}")"
if [[ "${actual_size}" != "${DREAMDOJO_LAM_SIZE}" ]]; then
  echo "ERROR: ${DREAMDOJO_LAM_FILENAME} size ${actual_size}; expected ${DREAMDOJO_LAM_SIZE}." >&2
  exit 1
fi
actual_sha256="$(sha256sum "${DREAMDOJO_CHECKPOINT}" | cut -d' ' -f1)"
if [[ "${actual_sha256}" != "${DREAMDOJO_LAM_SHA256}" ]]; then
  echo "ERROR: ${DREAMDOJO_LAM_FILENAME} SHA256 ${actual_sha256}; expected ${DREAMDOJO_LAM_SHA256}." >&2
  exit 1
fi

cat <<EOF
DreamDojo LAM setup complete.
DREAMDOJO_ROOT=$(realpath "${DREAMDOJO_ROOT}")
DREAMDOJO_COMMIT=${DREAMDOJO_COMMIT}
DREAMDOJO_CHECKPOINT=$(realpath "${DREAMDOJO_CHECKPOINT}")
DREAMDOJO_CHECKPOINT_SHA256=${DREAMDOJO_LAM_SHA256}
DREAMDOJO_HF_REVISION=${DREAMDOJO_HF_REVISION}
DREAMDOJO_SOURCE_LICENSE=${DREAMDOJO_SOURCE_LICENSE}
DREAMDOJO_CHECKPOINT_LICENSE=${DREAMDOJO_CHECKPOINT_LICENSE}
PYTHON_BIN=${DREAMDOJO_ROOT}/.venv/bin/python
EOF
