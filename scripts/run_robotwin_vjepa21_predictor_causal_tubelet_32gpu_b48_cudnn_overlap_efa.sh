#!/usr/bin/env bash
# 32-GPU V-JEPA21 JEPAPredictor: per-GPU batch 48, cuDNN SDPA, ZeRO-2 overlap, EFA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FASTWAM_EXPECTED_WORLD_SIZE=32
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-48}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( PER_GPU_BATCH_SIZE * FASTWAM_EXPECTED_WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))}"
export NO_CKPT=0
export FIXED_TARGET_ENCODER=false
export VISUAL_ENCODER_FREEZE_BACKBONE=true
export TRAINABLE_COMPONENTS='[dit]'
unset VISUAL_ENCODER_ACTIVATION_CHECKPOINTING VISUAL_ENCODER_LR_MULTIPLIER
export SAVE_EVERY=2000
export LOG_EVERY=10
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" "$@"
