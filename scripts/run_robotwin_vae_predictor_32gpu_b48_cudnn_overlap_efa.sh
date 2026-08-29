#!/usr/bin/env bash
# 32-GPU Wan VAE + deterministic video predictor on AWS HyperPod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export FASTWAM_EXPECTED_WORLD_SIZE=32
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-48}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( PER_GPU_BATCH_SIZE * FASTWAM_EXPECTED_WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))}"
export NO_CKPT=0
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export FASTWAM_USE_EFA=1
export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "[aws-vae-predictor] ERROR: completed WebDataset not found: ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
  echo "Set ROBOTWIN_WEBDATASET_ROOT to the completed indexed RoboTwin dataset." >&2
  exit 2
fi
export LOG_ROOT="${LOG_ROOT:-/efs/shaunxhwang}"
export RUN_NAME="${RUN_NAME:-robotwin_vae_predictor_32gpu_b48_cudnn_overlap_efa}"
export LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-vae-predictor}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vae_predictor_aws.sh" "$@"
