#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ACCEPT_NVIDIA_OPEN_MODEL_LICENSE=1 \
  DREAMDOJO_ROOT=/path/to/DreamDojo \
  DREAMDOJO_CHECKPOINT=/path/to/LAM_400k.ckpt \
  ROBOTWIN_WEBDATASET_ROOT=/path/to/robotwin_webdataset \
  ROBOTWIN_SOURCE_ROOT=/path/to/original/robotwin2.0 \
  ROBOTWIN_LATENT_ACTION_CACHE_ROOT=/path/to/cache \
  scripts/run_precompute_robotwin_dreamdojo_latents.sh [precompute options]

Runs train first, then val with train-derived normalization. Outputs train/ and val/
under ROBOTWIN_LATENT_ACTION_CACHE_ROOT with one shared cache-family signature.
Unset paths default to external/DreamDojo and /efs/shaunxhwang RoboTwin source,
WebDataset, and latent-action cache locations.
EOF
  exit 0
fi
DREAMDOJO_ROOT="${DREAMDOJO_ROOT:-${REPO_ROOT}/external/DreamDojo}"
DREAMDOJO_CHECKPOINT="${DREAMDOJO_CHECKPOINT:-${DREAMDOJO_ROOT}/checkpoints/DreamDojo/LAM_400k.ckpt}"
ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
ROBOTWIN_SOURCE_ROOT="${ROBOTWIN_SOURCE_ROOT:-/efs/shaunxhwang/robotwin2.0/robotwin2.0}"
ROBOTWIN_LATENT_ACTION_CACHE_ROOT="${ROBOTWIN_LATENT_ACTION_CACHE_ROOT:-/efs/shaunxhwang/robotwin2.0_latent_action_cache}"
[[ "${ACCEPT_NVIDIA_OPEN_MODEL_LICENSE:-0}" == "1" ]] || {
  echo "ERROR: set ACCEPT_NVIDIA_OPEN_MODEL_LICENSE=1 only after reviewing and accepting the checkpoint license." >&2
  exit 2
}

PYTHON_BIN="${PYTHON_BIN:-${DREAMDOJO_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: DreamDojo Python is missing: ${PYTHON_BIN}. Run scripts/setup_dreamdojo_lam.sh first." >&2
  exit 1
}
SPLITS="${SPLITS:-train val}"
for split in ${SPLITS}; do
  normalization_args=()
  if [[ "${split}" == "val" ]]; then
    normalization_args=(--normalization-manifest "${ROBOTWIN_LATENT_ACTION_CACHE_ROOT}/train/manifest.json")
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_robotwin_dreamdojo_latents.py" \
    --dreamdojo-root "${DREAMDOJO_ROOT}" \
    --checkpoint "${DREAMDOJO_CHECKPOINT}" \
    --accept-nvidia-open-model-license \
    --preprocessed-root "${ROBOTWIN_WEBDATASET_ROOT}" \
    --source-root "${ROBOTWIN_SOURCE_ROOT}" \
    --split "${split}" \
    --output "${ROBOTWIN_LATENT_ACTION_CACHE_ROOT}/${split}" \
    "${normalization_args[@]}" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DREAMDOJO_DTYPE:-bfloat16}" \
    --cache-dtype "${LATENT_CACHE_DTYPE:-float32}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --shard-size "${SHARD_SIZE:-256}" \
    "$@"
done
