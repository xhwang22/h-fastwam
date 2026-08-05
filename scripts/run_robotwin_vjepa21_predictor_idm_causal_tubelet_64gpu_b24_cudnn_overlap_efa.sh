#!/usr/bin/env bash
# 64-GPU V-JEPA21 IDM: per-GPU batch 24 = global batch 1536.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FASTWAM_EXPECTED_WORLD_SIZE=64
export GLOBAL_BATCH_SIZE=$(( 24 * FASTWAM_EXPECTED_WORLD_SIZE ))
export GRADIENT_ACCUMULATION_STEPS=1
export SAVE_EVERY=2000
export LOG_EVERY=10
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_idm_causal_tubelet_64gpu_b24_cudnn_overlap_efa}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_idm_vjepa21_predictor_causal_tubelet_aws.sh" "$@"
