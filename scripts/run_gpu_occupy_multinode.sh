#!/usr/bin/env bash
# Launch /mnt/private_xh2/gpu.py via torchrun on the current node.
# Topology resolved from NODE_IP_LIST (ip:ngpu,ip:ngpu,...).
# Run this on every node (use launch_all_nodes.sh as the broadcaster).

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

: "${NODE_IP_LIST:?NODE_IP_LIST must be set}"

IFS=',' read -ra NODE_LIST <<< "${NODE_IP_LIST}"
NNODES="${#NODE_LIST[@]}"

declare -a NODE_IPS=()
declare -a NODE_GPUS=()
for entry in "${NODE_LIST[@]}"; do
  NODE_IPS+=("${entry%%:*}")
  NODE_GPUS+=("${entry##*:}")
done

MASTER_ADDR="${MASTER_ADDR:-${NODE_IPS[0]}}"
MASTER_PORT="${MASTER_PORT:-29503}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NODE_GPUS[0]}}"

LOCAL_IPS=$(hostname -I 2>/dev/null || true)
NODE_RANK="${NODE_RANK:-}"
if [[ -z "${NODE_RANK}" ]]; then
  for i in "${!NODE_IPS[@]}"; do
    for lip in ${LOCAL_IPS}; do
      if [[ "${lip}" == "${NODE_IPS[$i]}" ]]; then
        NODE_RANK="${i}"
        break 2
      fi
    done
  done
fi
NODE_RANK="${NODE_RANK:-0}"

LOG_DIR="/tmp/gpu_occupy"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/gpu_occupy.rank${NODE_RANK}.log"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export PYTHONDONTWRITEBYTECODE=1

echo "[gpu_occupy] rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} gpus=${NPROC_PER_NODE}"
echo "[gpu_occupy] log=${LOG_FILE}"

CMD=(
  torchrun
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    /mnt/private_xh2/gpu.py
)

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || disown || true
echo "${PID}" > "${LOG_DIR}/pid.rank${NODE_RANK}"
echo "[gpu_occupy] background PID=${PID}"
