#!/usr/bin/env bash
# Run one of five manual AWS shards for the Wan VAE predictor evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_KIND=vae_predictor
export EVAL_LABEL=vae_predictor
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export TASK_SHARD_COUNT=5
export RUN_DIR="${RUN_DIR:-/efs/shaunxhwang/robotwin_vae_predictor_32gpu_b48_cudnn_overlap_efa}"
export EVAL_ENTRYPOINT="${SCRIPT_DIR}/run_robotwin_eval_vae_predictor_h100_8gpu.sh"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval_manual_shard.sh" "$@"
