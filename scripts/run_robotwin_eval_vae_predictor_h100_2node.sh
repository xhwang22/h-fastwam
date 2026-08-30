#!/usr/bin/env bash
# Two-node task-sharded RoboTwin evaluation for the Wan VAE predictor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_KIND=vae_predictor
export EVAL_LABEL=vae_predictor
export MODE="${MODE:-full}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
export RUN_DIR="${RUN_DIR:-/efs/shaunxhwang/robotwin_vae_predictor_32gpu_b48_cudnn_overlap_efa}"

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "ERROR: set NODE_IP_LIST=chief_ip,worker_ip for two-node evaluation." >&2
  exit 2
fi
IFS=',' read -r -a _VAE_EVAL_NODES <<< "${NODE_IP_LIST}"
if [[ "${#_VAE_EVAL_NODES[@]}" -ne 2 ]]; then
  echo "ERROR: expected exactly two nodes, got ${#_VAE_EVAL_NODES[@]}: ${NODE_IP_LIST}" >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval_multinode.sh" "$@"
