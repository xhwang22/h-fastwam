#!/usr/bin/env bash
# 48 GPUs: 32 samples/GPU, grad accumulation 1, global batch 1536.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DREAMDOJO_ROOT="${DREAMDOJO_ROOT:-${REPO_ROOT}/external/DreamDojo}"
export DREAMDOJO_CHECKPOINT="${DREAMDOJO_CHECKPOINT:-${DREAMDOJO_ROOT}/checkpoints/DreamDojo/LAM_400k.ckpt}"
if [[ ! -f "${DREAMDOJO_ROOT}/external/lam/modules/lam.py" ]]; then
  echo "ERROR: DreamDojo LAM source not found under ${DREAMDOJO_ROOT}." >&2
  exit 1
fi
if [[ ! -f "${DREAMDOJO_CHECKPOINT}" ]]; then
  echo "ERROR: DreamDojo checkpoint not found: ${DREAMDOJO_CHECKPOINT}" >&2
  exit 1
fi

# This launcher always computes targets online, even if the shell previously
# exported variables used by the legacy cache route.
unset LATENT_ACTION_CACHE_ROOT LATENT_ACTION_CACHE_SIGNATURE

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FASTWAM_EXPECTED_WORLD_SIZE=48
export PER_GPU_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=$(( \
  PER_GPU_BATCH_SIZE \
  * FASTWAM_EXPECTED_WORLD_SIZE \
  * GRADIENT_ACCUMULATION_STEPS \
))
export MODEL_CONFIG=hfastwam_latent_action_vjepa21
export DATA_CONFIG=robotwin_latent_action_interleaved_webdataset
export STANDARDISE_OUTPUT=true
export VIDEO_LATENT_CACHE_ENABLED=0
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_latent_action_vjepa21_48gpu_b32_cudnn_overlap_efa}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" \
  "$@"
