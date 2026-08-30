#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_KIND=dinov3_flow
export EVAL_LABEL=dinov3_vithplus_840m
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export RUN_DIR="${RUN_DIR:-/efs/shaunxhwang/robotwin_dinov3_vithplus_840m_32gpu_b48_cudnn_overlap_efa}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${REPO_ROOT}/checkpoints/dinov3-vith16plus-pretrain-lvd1689m}"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
