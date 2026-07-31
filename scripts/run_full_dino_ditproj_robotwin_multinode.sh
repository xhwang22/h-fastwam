#!/usr/bin/env bash
# =============================================================================
# Multi-node end-to-end: pretrain DINO on OpenVid → finetune on RoboTwin
#
# Reads NODE_IP_LIST (format: "ip1:gpus,ip2:gpus,...").
# Run from the CHIEF node — SSHes into all other nodes automatically.
#
# Usage:
#   bash scripts/run_full_dino_ditproj_robotwin_multinode.sh
#
#   # Skip pretrain (already have ckpt):
#   PRETRAIN_CKPT=/path/to/step_50000.pt \
#       bash scripts/run_full_dino_ditproj_robotwin_multinode.sh
#
#   # Extra hydra overrides:
#   PRETRAIN_EXTRA="max_steps=80000" \
#   FINETUNE_EXTRA="learning_rate=5e-5" \
#       bash scripts/run_full_dino_ditproj_robotwin_multinode.sh
# =============================================================================

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[full-pipeline-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[full-pipeline-mn] $*"; }

# ---------------------------------------------------------------------------
# Resolve topology
# ---------------------------------------------------------------------------
: "${NODE_IP_LIST:?NODE_IP_LIST must be set, e.g. 1.2.3.4:8,1.2.3.5:8}"

IFS=',' read -ra NODE_LIST <<< "${NODE_IP_LIST}"
NNODES="${#NODE_LIST[@]}"

declare -a NODE_IPS=()
declare -a NODE_GPUS=()
for entry in "${NODE_LIST[@]}"; do
  NODE_IPS+=("${entry%%:*}")
  NODE_GPUS+=("${entry##*:}")
done

MASTER_ADDR="${NODE_IPS[0]}"
PRETRAIN_EXTRA="${PRETRAIN_EXTRA:-}"
FINETUNE_EXTRA="${FINETUNE_EXTRA:-}"

echo "============================================================="
echo " DINO DiT-side Projection — Full Pipeline (Multi-Node)"
echo " Step 1: Pretrain on OpenVid-1M"
echo " Step 2: Finetune on RoboTwin"
echo "============================================================="
echo " NNODES:        ${NNODES}"
echo " MASTER:        ${MASTER_ADDR}"
echo " NODE_IP_LIST:  ${NODE_IP_LIST}"
echo " Pretrain ckpt: ${PRETRAIN_CKPT:-<will be produced by Step 1>}"
echo "============================================================="

# ---------------------------------------------------------------------------
# Helper: launch a per-node script on all nodes in parallel, then wait.
# Writes per-node env to a temp file on the shared fs that each node sources.
# Usage: launch_all <inner_script> <run_id> <log_root> [KEY=VALUE ...]
# ---------------------------------------------------------------------------
launch_all() {
  local inner_script="$1"
  local run_id="$2"
  local log_root="$3"
  shift 3
  local extra_kvs=("$@")   # extra KEY=VALUE pairs

  local pids=()

  for i in "${!NODE_IPS[@]}"; do
    local host="${NODE_IPS[$i]}"
    local node_log="${log_root}/${run_id}/launch.log.rank${i}"
    mkdir -p "${log_root}/${run_id}"

    # Write per-node env file to shared filesystem
    local env_file="${log_root}/${run_id}/.env.rank${i}"
    {
      echo "export NODE_IP_LIST='${NODE_IP_LIST}'"
      echo "export NODE_RANK=${i}"
      echo "export MASTER_ADDR=${MASTER_ADDR}"
      echo "export FOREGROUND=1"
      echo "export RUN_NAME='${run_id}'"
      echo "export LOG_ROOT='${log_root}'"
      for kv in "${extra_kvs[@]:-}"; do
        [[ -n "${kv}" ]] && echo "export ${kv}"
      done
    } > "${env_file}"

    info "  -> rank ${i} on ${host} (log: ${node_log})"

    if [[ "${i}" -eq 0 ]]; then
      # Chief node: run locally
      bash -c "source '${env_file}' && bash '${inner_script}'" > "${node_log}" 2>&1 &
      pids+=($!)
    else
      ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${host}" \
          "bash -c \"source '${env_file}' && bash '${inner_script}'\"" \
          > "${node_log}" 2>&1 &
      pids+=($!)
    fi
  done

  # Stream rank-0 log to console while waiting
  local rank0_log="${log_root}/${run_id}/launch.log.rank0"
  info "Streaming rank-0 log (${rank0_log})..."
  tail -f "${rank0_log}" &
  local tail_pid=$!

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || { info "WARNING: a node exited with error"; failed=1; }
  done
  kill "${tail_pid}" 2>/dev/null || true

  [[ "${failed}" -eq 0 ]] || err "One or more nodes failed. Check logs in ${log_root}/${run_id}/"
  info "All nodes finished for run: ${run_id}"
}

# ---------------------------------------------------------------------------
# Step 1: Pretrain on OpenVid-1M
# ---------------------------------------------------------------------------
if [[ -n "${PRETRAIN_CKPT:-}" ]] && [[ -f "${PRETRAIN_CKPT}" ]]; then
  info "[Step 1] SKIP — using existing checkpoint: ${PRETRAIN_CKPT}"
else
  info "[Step 1] Launching pretrain on OpenVid-1M across ${NNODES} nodes..."

  PRETRAIN_RUN_ID="pretrain_dino_mn_$(date +%Y-%m-%d_%H-%M-%S)"
  PRETRAIN_LOG_ROOT="${REPO_ROOT}/runs/pretrain_dino_ditproj"

  PRETRAIN_ENVS=("RUN_PREFIX=pretrain_dino_mn")
  [[ -n "${PRETRAIN_EXTRA}" ]] && PRETRAIN_ENVS+=("EXTRA=${PRETRAIN_EXTRA}")

  launch_all \
    "${SCRIPT_DIR}/run_pretrain_dino_ditproj_multinode.sh" \
    "${PRETRAIN_RUN_ID}" \
    "${PRETRAIN_LOG_ROOT}" \
    "${PRETRAIN_ENVS[@]}"

  # Find the latest checkpoint produced by rank-0
  PRETRAIN_CKPT="$(ls -t "${PRETRAIN_LOG_ROOT}/${PRETRAIN_RUN_ID}/checkpoints/weights/"*.pt 2>/dev/null | head -1 || true)"
  if [[ -z "${PRETRAIN_CKPT}" ]]; then
    err "No checkpoint found in ${PRETRAIN_LOG_ROOT}/${PRETRAIN_RUN_ID}/checkpoints/weights/"
  fi
  info "[Step 1] Done → ${PRETRAIN_CKPT}"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: Finetune on RoboTwin
# ---------------------------------------------------------------------------
info "[Step 2] Launching finetune on RoboTwin across ${NNODES} nodes..."

FINETUNE_RUN_ID="robotwin_dino_mn_$(date +%Y-%m-%d_%H-%M-%S)"
FINETUNE_LOG_ROOT="${REPO_ROOT}/runs/robotwin_dino_ditproj"

FINETUNE_ENVS=(
  "RUN_PREFIX=robotwin_dino_mn"
  "PRETRAIN_CKPT=${PRETRAIN_CKPT}"
)
[[ -n "${FINETUNE_EXTRA}" ]] && FINETUNE_ENVS+=("EXTRA=${FINETUNE_EXTRA}")

launch_all \
  "${SCRIPT_DIR}/run_robotwin_dino_ditproj_multinode.sh" \
  "${FINETUNE_RUN_ID}" \
  "${FINETUNE_LOG_ROOT}" \
  "${FINETUNE_ENVS[@]}"

echo ""
echo "============================================================="
echo " Pipeline complete!"
echo " Pretrain:  ${PRETRAIN_CKPT}"
echo " Finetune:  ${FINETUNE_LOG_ROOT}/${FINETUNE_RUN_ID}"
echo "============================================================="
