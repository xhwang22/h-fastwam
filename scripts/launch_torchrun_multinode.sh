#!/usr/bin/env bash
# Multi-node launcher using torchrun directly (bypasses deepspeed.launcher.runner).
# Accelerate picks up DeepSpeed via env vars instead of its own DeepSpeed runner.
#
# Required env: NNODES, NODE_RANK, MASTER_ADDR, NPROC_PER_NODE
#               MODEL, TASK, DATA, WANDB_NAME, RUN_NAME, LOG_ROOT

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
DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/scripts/ds_configs/ds_zero1_config.json}"

mkdir -p "${LOG_DIR}"

# Tell accelerate to use DeepSpeed via env vars (works with torchrun)
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${DS_CONFIG}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Disable ALL DeepSpeed JIT/CUDA compilation — it causes d_alloc_parallel
# deadlock on CephFS-backed shared filesystems when multiple processes race
# to create the same temp directories simultaneously.
export DS_BUILD_AIO=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_TRANSFORMER_INFERENCE=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_UTILS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_QUANTIZER=0
export DS_BUILD_RANDOM_LTD=0
export DS_BUILD_EVOFORMER_ATTN=0
export DS_SKIP_CUDA_CHECK=1
# Force DeepSpeed to use a per-rank unique tmp dir on local /tmp (not CephFS)
# to avoid racing directory creation across ranks.
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_rank${NODE_RANK:-0}"

# Prevent Python from writing __pycache__ bytecode to the FUSE/CephFS-backed
# conda environment, which causes d_alloc_parallel kernel deadlocks when
# multiple worker processes race to create the same cache directories.
export PYTHONDONTWRITEBYTECODE=1

# NCCL network interface: use bond1 which carries the node IPs (28.216.x.x).
# Without this NCCL may pick a wrong interface and fail with
# "socketPollConnect poll() returned 1, no POLLOUT events".
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"   # disable IB, use socket
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

HYDRA_OVERRIDES=(
  "task=${TASK}"
  "data=${DATA}"
  "model=${MODEL}"
  "output_dir=${LOG_DIR}"
  "wandb.name=${WANDB_NAME}"
)
if [[ -n "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
  export DIFFSYNTH_MODEL_BASE_PATH
fi
if [[ -n "${EXTRA}" ]]; then
  # shellcheck disable=SC2206
  HYDRA_OVERRIDES+=( ${EXTRA} )
fi

echo "[torchrun_mn] rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} gpus=${NPROC_PER_NODE} total=${TOTAL_PROCESSES}"
echo "[torchrun_mn] log=${LOG_FILE}"

CMD=(
  torchrun
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    scripts/train.py
    "${HYDRA_OVERRIDES[@]}"
)

if [[ "${FOREGROUND}" == "1" ]]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
else
  "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
  PID=$!
  disown "$PID" 2>/dev/null || disown || true
  echo "${PID}" > "${LOG_DIR}/.torchrun.pid.rank${NODE_RANK}"
  echo "[torchrun_mn] background PID=${PID}"
  echo "[torchrun_mn] tail: tail -f ${LOG_FILE}"
fi
