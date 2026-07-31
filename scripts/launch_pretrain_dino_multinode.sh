#!/usr/bin/env bash
# One-shot launcher: start pretrain DINO on all 4 nodes.
# Run this on the master node (28.216.17.132).
# Usage: bash scripts/launch_pretrain_dino_multinode.sh [foreground]
#   foreground: if set, rank0 runs in foreground (shows live output)

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
[[ -f "${CONDA_ACTIVATE}" ]] && source "${CONDA_ACTIVATE}" fastwam

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NIP="28.216.17.132:8,28.216.19.22:8,28.216.19.71:8,28.216.19.146:8"
MASTER_ADDR="28.216.17.132"
MASTER_PORT="${MASTER_PORT:-29510}"
NNODES=4
NPROC=8
INNER="${SCRIPT_DIR}/run_pretrain_dino_ditproj_multinode.sh"

FOREGROUND_MODE="${1:-background}"

# Shared run name so all nodes write to same log dir
RUN_NAME="pretrain_dino_mn_$(date +%Y-%m-%d_%H-%M-%S)"
echo "[launcher] RUN_NAME=${RUN_NAME}"
echo "[launcher] Master: ${MASTER_ADDR}:${MASTER_PORT}"

# Export env for rank0 (used if running inline below)
export NODE_IP_LIST="${NIP}"
export RUN_NAME
export NNODES
export NODE_RANK=0
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE="${NPROC}"

# Launch workers (rank 1,2,3) via SSH in background
for rank in 1 2 3; do
  ip=$(echo "${NIP}" | tr ',' '\n' | sed -n "$((rank+1))p" | cut -d: -f1)
  echo "[launcher] Sending rank ${rank} -> ${ip}"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${ip}" \
    "export NODE_IP_LIST='${NIP}'; export RUN_NAME='${RUN_NAME}'; export NNODES=${NNODES}; export NODE_RANK=${rank}; export MASTER_ADDR=${MASTER_ADDR}; export MASTER_PORT=${MASTER_PORT}; export NPROC_PER_NODE=${NPROC}; bash ${INNER}" \
    </dev/null &
done

sleep 3

LOG_DIR="${REPO_ROOT}/runs/pretrain_dino_ditproj/${RUN_NAME}"
echo "[launcher] Log dir: ${LOG_DIR}"
echo "[launcher] Monitor: tail -f ${LOG_DIR}/train.log.rank0"

# Launch rank0
if [[ "${FOREGROUND_MODE}" == "foreground" ]]; then
  echo "[launcher] Running rank0 in FOREGROUND..."
  FOREGROUND=1 bash "${INNER}" 2>&1 | tee "${LOG_DIR}/train.log.rank0.tee"
else
  echo "[launcher] Running rank0 in background..."
  FOREGROUND=0 bash "${INNER}"
  wait
  echo "[launcher] All nodes launched."
  echo "[launcher] Tail logs:"
  echo "  tail -f ${LOG_DIR}/train.log.rank0"
  echo "  tail -f ${LOG_DIR}/train.log.rank1"
  echo "Stop all: for ip in 28.216.17.132 28.216.19.22 28.216.19.71 28.216.19.146; do"
  echo "    ssh \$ip 'pkill -TERM -f train.py; pkill -TERM -f train_zero1'; done"
fi
