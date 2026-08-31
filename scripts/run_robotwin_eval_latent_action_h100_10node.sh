#!/usr/bin/env bash
# Run one of ten manual AWS shards for V-JEPA DreamDojo latent-action eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_KIND=latent_action
export EVAL_LABEL=latent_action_vjepa21
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
export TASK_SHARD_COUNT=10
export RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/robotwin_hfastwam/robotwin_latent_action_vjepa21_48gpu_b32_cudnn_overlap_efa}"
export EVAL_ENTRYPOINT="${SCRIPT_DIR}/run_robotwin_eval_latent_action_h100_8gpu.sh"

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval_manual_shard.sh" "$@"
