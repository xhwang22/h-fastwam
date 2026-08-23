#!/usr/bin/env bash
# Four-node fixed-teacher training: 32 H20 GPUs, batch 48/GPU, global batch 1536.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "ERROR: set NODE_IP_LIST to exactly four nodes." >&2
  echo 'Example: NODE_IP_LIST="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4" bash <script>' >&2
  exit 2
fi
IFS=',' read -ra ROBOTWIN_FULL_ENCODER_NODES <<< "${NODE_IP_LIST}"
if [[ "${#ROBOTWIN_FULL_ENCODER_NODES[@]}" -ne 4 ]]; then
  echo "ERROR: expected 4 nodes, got ${#ROBOTWIN_FULL_ENCODER_NODES[@]}: ${NODE_IP_LIST}" >&2
  exit 2
fi

export PER_GPU_BATCH_SIZE=48
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=1536
export VIDEO_LATENT_CACHE_ENABLED=0
export NO_CKPT="${NO_CKPT:-1}"
export FASTWAM_ADAM_FUSED=1
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-0}"
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_fixed_teacher_full_encoder_4x8_b48_nockpt_cudnn_overlap_efa}"
export FIXED_TARGET_ENCODER=true
export VISUAL_ENCODER_FREEZE_BACKBONE=false
export VISUAL_ENCODER_ACTIVATION_CHECKPOINTING=true
export TRAINABLE_COMPONENTS='[dit,visual_encoder]'
export VISUAL_ENCODER_LR_MULTIPLIER="${VISUAL_ENCODER_LR_MULTIPLIER:-0.1}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa.sh" \
  "$@"
