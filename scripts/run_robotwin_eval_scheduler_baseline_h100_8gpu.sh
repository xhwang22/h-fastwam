#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_KIND=vjepa21_flow
export EVAL_LABEL=vjepa21_scheduler_baseline
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export RUN_DIR="${RUN_DIR:-/efs/shaunxhwang/robotwin_vjepa21_video_scheduler_tbaseline_shift5_globalnorm_aws_4x8_b48_acc1_gb1536}"
export VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${REPO_ROOT}/checkpoints/torch_hub/hub/checkpoints/vjepa2_1_vitG_384.pt}"
export VJEPA21_REPO="${VJEPA21_REPO:-${REPO_ROOT}/checkpoints/torch_hub/hub/facebookresearch_vjepa2_main}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-/efs/shaunxhwang/robotwin2.0_webdataset/vjepa21_vitG_causal_tubelet_global_stats.pt}"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
