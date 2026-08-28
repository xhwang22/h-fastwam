#!/usr/bin/env bash
# 32-GPU Wan VAE + deterministic video predictor on AWS HyperPod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
export RUN_NAME="${RUN_NAME:-robotwin_vae_predictor_32gpu_b48_cudnn_overlap_efa}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-vae-predictor}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vae_predictor_aws.sh" "$@"
