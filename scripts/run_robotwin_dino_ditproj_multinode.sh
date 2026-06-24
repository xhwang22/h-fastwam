#!/usr/bin/env bash
# Multi-node finetune: DINO DiT-side projection on RoboTwin
#
# Reads NODE_IP_LIST (format: "ip1:gpus,ip2:gpus,...") to resolve topology.
# Run this script on EVERY node, or use launch_all_nodes.sh from the chief.
#
# Usage:
#   bash scripts/run_robotwin_dino_ditproj_multinode.sh
#   FOREGROUND=1 bash scripts/run_robotwin_dino_ditproj_multinode.sh
#   PRETRAIN_CKPT=/path/to/step_50000.pt bash scripts/run_robotwin_dino_ditproj_multinode.sh
#   EXTRA="max_steps=80000" bash scripts/run_robotwin_dino_ditproj_multinode.sh

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_tj5/share_302528826/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[robotwin-dino-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[robotwin-dino-mn] $*"; }

# ---------------------------------------------------------------------------
# Resolve topology from NODE_IP_LIST env var (format: ip1:ngpu,ip2:ngpu,...)
# ---------------------------------------------------------------------------
: "${NODE_IP_LIST:?NODE_IP_LIST must be set, e.g. 1.2.3.4:8,1.2.3.5:8}"

IFS=',' read -ra NODE_LIST <<< "${NODE_IP_LIST}"
NNODES="${#NODE_LIST[@]}"

declare -a NODE_IPS=()
declare -a NODE_GPUS=()
for entry in "${NODE_LIST[@]}"; do
  ip="${entry%%:*}"
  gpus="${entry##*:}"
  NODE_IPS+=("$ip")
  NODE_GPUS+=("$gpus")
done

MASTER_ADDR="${MASTER_ADDR:-${NODE_IPS[0]}}"
MASTER_PORT="${MASTER_PORT:-29502}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NODE_GPUS[0]}}"

# Determine current node rank by matching local IP
LOCAL_IPS=$(hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
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
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"

info "Detected topology:"
info "  NNODES         = ${NNODES}"
info "  NODE_RANK      = ${NODE_RANK}"
info "  MASTER_ADDR    = ${MASTER_ADDR}"
info "  MASTER_PORT    = ${MASTER_PORT}"
info "  NPROC_PER_NODE = ${NPROC_PER_NODE}"
info "  TOTAL_GPUS     = $(( NNODES * NPROC_PER_NODE ))"

TJ5_BASE="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints"

export DIFFSYNTH_MODEL_BASE_PATH="${TJ5_BASE}/"
export MODEL="${MODEL:-fastwam_dino_ditproj}"
export TASK="${TASK:-robotwin_uncond_3cam_384_1e-4}"
export DATA="${DATA:-robotwin}"
export RUN_PREFIX="${RUN_PREFIX:-robotwin_dino_mn}"
export WANDB_NAME="${WANDB_NAME:-robotwin_dino_ditproj_mn}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/robotwin_dino_ditproj}"
export FOREGROUND="${FOREGROUND:-0}"

# Build EXTRA: pass pretrain checkpoint if set
EXTRA_BASE="${EXTRA:-}"
if [[ -n "${PRETRAIN_CKPT:-}" ]]; then
  EXTRA_BASE="model.pretrain_checkpoint=${PRETRAIN_CKPT}${EXTRA_BASE:+ ${EXTRA_BASE}}"
fi
export EXTRA="${EXTRA_BASE}"

export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export NPROC_PER_NODE

exec bash "${SCRIPT_DIR}/run_multinode.sh"
