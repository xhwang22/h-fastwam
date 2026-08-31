#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_KIND=latent_action
export EVAL_LABEL=latent_action_vjepa21
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/robotwin_hfastwam/robotwin_latent_action_vjepa21_48gpu_b32_cudnn_overlap_efa}"
export VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${REPO_ROOT}/checkpoints/torch_hub/hub/checkpoints/vjepa2_1_vitG_384.pt}"
export VJEPA21_REPO="${VJEPA21_REPO:-${REPO_ROOT}/checkpoints/torch_hub/hub/facebookresearch_vjepa2_main}"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
