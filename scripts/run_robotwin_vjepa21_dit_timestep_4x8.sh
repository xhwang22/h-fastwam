#!/usr/bin/env bash
# Four-node V-JEPA 2.1 Flow-DiT timestep ablation with fixed global norm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=_timestep_sampling_preset.sh
source "${SCRIPT_DIR}/_timestep_sampling_preset.sh"
fastwam_video_timestep_sampling_preset "${TIMESTEP_SAMPLING_PRESET:-baseline}"

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
export ROBOTWIN_DATA_ROOT
export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "[vjepa21-dit-4x8] ERROR: completed WebDataset not found: ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
  echo "Set ROBOTWIN_WEBDATASET_ROOT to the completed indexed RoboTwin dataset." >&2
  exit 2
fi
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[vjepa21-dit-4x8] ERROR: global norm stats not found: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Set VJEPA21_NORMALISE_STATS_PATH to a shared path visible on all nodes." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=32
export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=1
export MODEL_CONFIG=hfastwam_small_vjepa21
export NO_CKPT=0
export STANDARDISE_OUTPUT=true
export CAUSAL_TUBELET_ENCODING=true
export TEMPORAL_DOWNSAMPLE=4
export TIMESTEP_SAMPLING_TARGET=video
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export FASTWAM_USE_EFA="${FASTWAM_USE_EFA:-1}"
export SAVE_EVERY=2000
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-vjepa21-dit}"
export WANDB_GROUP="${WANDB_GROUP:-video-scheduler-globalnorm-aws-4x8}"
export LOG_ROOT="${LOG_ROOT:-/efs/shaunxhwang}"
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_video_scheduler_${TIMESTEP_PRESET_SUFFIX}_globalnorm_aws_4x8_b48_acc1_gb1536}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_causal_tubelet_aws.sh" \
  "${TIMESTEP_SAMPLING_OVERRIDES[@]}" \
  "$@"
