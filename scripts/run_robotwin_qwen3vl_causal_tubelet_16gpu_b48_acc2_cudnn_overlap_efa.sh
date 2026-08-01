#!/usr/bin/env bash
# 16-GPU Qwen3-VL Flow DiT: per-GPU batch 48, grad accumulation 2, cuDNN SDPA, ZeRO-2 overlap, EFA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FASTWAM_EXPECTED_WORLD_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=2
export GLOBAL_BATCH_SIZE=$(( 48 * FASTWAM_EXPECTED_WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))
export SAVE_EVERY=2000
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_qwen3vl_causal_tubelet_16gpu_b48_acc2_cudnn_overlap_efa}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_qwen3vl_causal_tubelet_aws.sh" "$@"
