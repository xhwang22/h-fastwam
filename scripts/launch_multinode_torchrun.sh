#!/usr/bin/env bash
# Multi-node launcher using torchrun (not DeepSpeed runner).
# Generates a per-rank accelerate config and launches training.
#
# Required env:
#   NNODES          total number of nodes
#   NODE_RANK       this node's rank (0-based)
#   MASTER_ADDR     chief node IP
#   MASTER_PORT     rendezvous port (default 29500)
#   NPROC_PER_NODE  GPUs per node (default 8)
#   MODEL / TASK / DATA / WANDB_NAME / LOG_ROOT / RUN_NAME
#
# This script is called by run_libero_vjepa2ac_predictor_multinode_fixed.sh
# on each node via SSH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NNODES="${NNODES:?}"
NODE_RANK="${NODE_RANK:?}"
MASTER_ADDR="${MASTER_ADDR:?}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TOTAL_PROCESSES=$(( NNODES * NPROC_PER_NODE ))

MODEL="${MODEL:?}"
TASK="${TASK:?}"
DATA="${DATA:?}"
WANDB_NAME="${WANDB_NAME:-${TASK}_mn}"
RUN_NAME="${RUN_NAME:?}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/${TASK}}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log.rank${NODE_RANK}"
FOREGROUND="${FOREGROUND:-0}"
EXTRA="${EXTRA:-}"

mkdir -p "${LOG_DIR}"

# Per-rank accelerate config (avoids YAML hardcoded num_machines=1)
ACCEL_CFG="${LOG_DIR}/accelerate_rank${NODE_RANK}.yaml"
cat > "${ACCEL_CFG}" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ${REPO_ROOT}/scripts/ds_configs/ds_zero1_config.json
  zero3_init_flag: false
mixed_precision: null
machine_rank: ${NODE_RANK}
main_training_function: main
num_machines: ${NNODES}
num_processes: ${TOTAL_PROCESSES}
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}
rdzv_backend: static
same_network: true
use_cpu: false
YAML

echo "[launch_mn] rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} gpus_per_node=${NPROC_PER_NODE} total=${TOTAL_PROCESSES}"
echo "[launch_mn] run_name=${RUN_NAME} log=${LOG_FILE}"

HYDRA_OVERRIDES=(
  "task=${TASK}"
  "data=${DATA}"
  "model=${MODEL}"
  "output_dir=${LOG_DIR}"
  "wandb.name=${WANDB_NAME}"
)
[[ -n "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]] && HYDRA_OVERRIDES+=( "++diffsynth_model_base_path=${DIFFSYNTH_MODEL_BASE_PATH}" )
if [[ -n "${EXTRA}" ]]; then
  # shellcheck disable=SC2206
  HYDRA_OVERRIDES+=( ${EXTRA} )
fi

CMD=(
  accelerate launch
    --config_file "${ACCEL_CFG}"
    scripts/train.py
    "${HYDRA_OVERRIDES[@]}"
)

if [[ "${FOREGROUND}" == "1" ]]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  nohup setsid "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
  PID=$!
  disown || true
  echo "${PID}" > "${LOG_DIR}/.launcher.pid.rank${NODE_RANK}"
  echo "[launch_mn] background PID=${PID}, tail: tail -f ${LOG_FILE}"
fi
