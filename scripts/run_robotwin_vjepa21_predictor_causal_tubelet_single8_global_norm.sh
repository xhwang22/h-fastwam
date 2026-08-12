#!/usr/bin/env bash
# RoboTwin V-JEPA predictor ablation using fixed offline global statistics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset PET_NNODES PET_NODE_RANK PET_MASTER_ADDR PET_MASTER_PORT PET_NPROC_PER_NODE
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK
unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT NODE_IP_LIST
unset FASTWAM_MANAGED_DISTRIBUTED _MULTINODE_LAUNCHED

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[robotwin-vjepa21-global-norm] ERROR: stats file not found: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Run scripts/precompute_robotwin_vjepa21_global_stats_single8.sh first." >&2
  exit 1
fi

export NO_CKPT="${NO_CKPT:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export FASTWAM_USE_EFA=0
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export STANDARDISE_OUTPUT=true
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_causal_tubelet_global_norm_single8}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" \
  "$@"
