#!/usr/bin/env bash
# Full online V-JEPA 2.1 training against a fixed original-checkpoint teacher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"

export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-48}"
export VIDEO_LATENT_CACHE_ENABLED=0
export FASTWAM_ADAM_FUSED=1
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-0}"
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_fixed_teacher_full_encoder_32gpu_b1_acc48_cudnn_overlap_efa}"
export FIXED_TARGET_ENCODER=true
export VISUAL_ENCODER_FREEZE_BACKBONE=false
export VISUAL_ENCODER_ACTIVATION_CHECKPOINTING=true
export TRAINABLE_COMPONENTS='[dit,visual_encoder]'
export VISUAL_ENCODER_LR_MULTIPLIER="${VISUAL_ENCODER_LR_MULTIPLIER:-0.1}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa.sh" \
  "$@"
