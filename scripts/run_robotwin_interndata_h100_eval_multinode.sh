#!/usr/bin/env bash
# Resume an InternData V-JEPA RoboTwin evaluation across H100 nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${RUN_DIR:?Set RUN_DIR to the completed InternData fine-tune run directory}"

export MODEL_KIND=vjepa
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export EXPECTED_EVAL_NNODES="${EXPECTED_EVAL_NNODES:-2}"
export FASTWAM_EVAL_USE_CURRENT_ENV="${FASTWAM_EVAL_USE_CURRENT_ENV:-0}"
export RENDER_BACKEND="${RENDER_BACKEND:-gpu}"
export CHECK_ALIGNMENT="${CHECK_ALIGNMENT:-1}"
export CHECK_ENV="${CHECK_ENV:-1}"
export QWEN_REVISION="${QWEN_REVISION:-89644892e4d85e24eaac8bacfd4f463576704203}"

if [[ -z "${STATS:-}" ]]; then
  if [[ -f "/efs/shaunxhwang/robotwin2.0_webdataset/dataset_stats.json" ]]; then
    STATS="/efs/shaunxhwang/robotwin2.0_webdataset/dataset_stats.json"
  elif [[ -f "${REPO_ROOT}/data/robotwin2.0_webdataset/dataset_stats.json" ]]; then
    STATS="${REPO_ROOT}/data/robotwin2.0_webdataset/dataset_stats.json"
  else
    STATS="${REPO_ROOT}/data/robotwin2.0/dataset_stats.json"
  fi
fi
export STATS

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval_multinode.sh" "$@"
