#!/usr/bin/env bash
# 32-GPU V-JEPA21 JEPAPredictor: per-GPU batch 48, cuDNN SDPA, ZeRO-2 overlap, EFA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FASTWAM_EXPECTED_WORLD_SIZE=32
export GLOBAL_BATCH_SIZE=$(( 48 * FASTWAM_EXPECTED_WORLD_SIZE ))
export GRADIENT_ACCUMULATION_STEPS=1
export SAVE_EVERY=2000
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" "$@"
