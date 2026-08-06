#!/usr/bin/env bash
# 64-GPU V-JEPA21 full-condition IDM: per-GPU 12 x accumulation 2 = global 1536.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FASTWAM_EXPECTED_WORLD_SIZE=64
export GLOBAL_BATCH_SIZE=$(( 12 * FASTWAM_EXPECTED_WORLD_SIZE * 2 ))
export GRADIENT_ACCUMULATION_STEPS=2
export SAVE_EVERY=2000
export LOG_EVERY=10
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export MODEL_CONFIG=hfastwam_idm_vjepa21_predictor_full_condition
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_idm_full_condition_causal_tubelet_64gpu_b12_acc2_cudnn_overlap_efa}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_idm_vjepa21_predictor_causal_tubelet_aws.sh" "$@"
