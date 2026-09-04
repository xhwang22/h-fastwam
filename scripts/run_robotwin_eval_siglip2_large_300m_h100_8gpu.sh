#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_KIND=siglip2_flow
export EVAL_LABEL=siglip2_large_300m
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"

if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_32GPU="/efs/shaunxhwang/robotwin_native_siglip2_large_300m_causal_tubelet_32gpu_b48_cudnn_overlap_efa"
  RUN_16GPU="/efs/shaunxhwang/robotwin_native_siglip2_large_300m_causal_tubelet_16gpu_b48_acc2_cudnn_overlap_efa"
  if [[ -d "${RUN_32GPU}" ]]; then
    RUN_DIR="${RUN_32GPU}"
  elif [[ -d "${RUN_16GPU}" ]]; then
    RUN_DIR="${RUN_16GPU}"
  else
    RUN_DIR="${RUN_32GPU}"
  fi
fi
export RUN_DIR
export SIGLIP2_MODEL_PATH="${SIGLIP2_MODEL_PATH:-${REPO_ROOT}/checkpoints/siglip2-large-patch16-384}"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
