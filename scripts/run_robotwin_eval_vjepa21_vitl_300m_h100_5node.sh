#!/usr/bin/env bash
# Run one of five manual AWS shards for V-JEPA 2.1 ViT-L 300M evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_KIND=vjepa21_flow
export EVAL_LABEL=vjepa21_vitl_300m
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export TASK_SHARD_COUNT=5
export RUN_DIR="${RUN_DIR:-/efs/shaunxhwang/robotwin_vjepa21_vitl_300m_causal_tubelet_16gpu_b48_acc2_cudnn_overlap_efa}"
export CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_019570.pt}"
export OUTPUT_TAG="${OUTPUT_TAG:-vjepa21_vitl_300m_step019570_h100_${MODE}_5node}"
export EVAL_ENTRYPOINT="${SCRIPT_DIR}/run_robotwin_eval_vjepa21_vitl_300m_h100_8gpu.sh"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval_manual_shard.sh" "$@"
