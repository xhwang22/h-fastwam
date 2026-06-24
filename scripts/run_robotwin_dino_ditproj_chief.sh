#!/usr/bin/env bash
# =============================================================================
# Chief-side launcher: DINO DiT-projection finetune on RoboTwin, multi-node.
#
# Mirrors scripts/run_libero_dino_ditproj.sh (single-node 8 GPUs, batch=16,
# global batch = 128) but spreads the work across the nodes listed in
# NODE_IP_LIST. To keep the global batch unchanged we lower per-GPU batch:
#
#     global_batch = batch_size * NPROC_PER_NODE * NNODES
#     128         = 4          * 8              * 4
#
# Usage (from chief node only):
#   bash scripts/run_robotwin_dino_ditproj_chief.sh
#
#   # With a pretrain checkpoint:
#   PRETRAIN_CKPT=/path/to/step_50000.pt \
#       bash scripts/run_robotwin_dino_ditproj_chief.sh
#
#   # Extra hydra overrides (forwarded to inner launcher's EXTRA):
#   EXTRA_OVERRIDES="learning_rate=5e-5" \
#       bash scripts/run_robotwin_dino_ditproj_chief.sh
# =============================================================================

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_tj5/share_302528826/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[robotwin-dino-chief] ERROR: $*" >&2; exit 1; }
info() { echo "[robotwin-dino-chief] $*"; }

: "${NODE_IP_LIST:?NODE_IP_LIST must be set, e.g. ip1:8,ip2:8,...}"

IFS=',' read -ra NODE_LIST <<< "${NODE_IP_LIST}"
NNODES="${#NODE_LIST[@]}"

declare -a NODE_IPS=()
declare -a NODE_GPUS=()
for entry in "${NODE_LIST[@]}"; do
  NODE_IPS+=("${entry%%:*}")
  NODE_GPUS+=("${entry##*:}")
done

NPROC_PER_NODE="${NODE_GPUS[0]}"
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))

# Original libero_dino_ditproj defaults: batch_size=16 on 8 GPUs => global=128.
# Preserve global batch by dividing per-GPU batch by (NNODES * NPROC_PER_NODE / 8).
ORIG_GLOBAL_BATCH="${ORIG_GLOBAL_BATCH:-128}"
if (( ORIG_GLOBAL_BATCH % TOTAL_GPUS != 0 )); then
  err "Cannot evenly split global batch ${ORIG_GLOBAL_BATCH} across ${TOTAL_GPUS} GPUs."
fi
PER_GPU_BATCH=$(( ORIG_GLOBAL_BATCH / TOTAL_GPUS ))

INNER_SCRIPT="${SCRIPT_DIR}/run_robotwin_dino_ditproj_multinode.sh"
[[ -f "${INNER_SCRIPT}" ]] || err "Inner launcher missing: ${INNER_SCRIPT}"

RUN_ID="${RUN_ID:-robotwin_dino_mn_$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/robotwin_dino_ditproj}"
LAUNCH_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${LAUNCH_DIR}"

# Build the EXTRA hydra-override string forwarded to every node.
EXTRA_BASE="batch_size=${PER_GPU_BATCH}"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  EXTRA_BASE="${EXTRA_BASE} ${EXTRA_OVERRIDES}"
fi

echo "============================================================="
echo " RoboTwin + DINO DiT-projection — Multi-Node Finetune"
echo "============================================================="
echo " NNODES         = ${NNODES}"
echo " NPROC_PER_NODE = ${NPROC_PER_NODE}"
echo " TOTAL_GPUS     = ${TOTAL_GPUS}"
echo " PER_GPU_BATCH  = ${PER_GPU_BATCH}  (global = ${ORIG_GLOBAL_BATCH})"
echo " RUN_ID         = ${RUN_ID}"
echo " LOG_ROOT       = ${LOG_ROOT}"
echo " PRETRAIN_CKPT  = ${PRETRAIN_CKPT:-<none>}"
echo " EXTRA          = ${EXTRA_BASE}"
echo " NODE_IP_LIST   = ${NODE_IP_LIST}"
echo "============================================================="

CHIEF_IP_GUESS="${NODE_IPS[0]}"
CHIEF_LOCAL_IPS=$(hostname -I 2>/dev/null || true)
if ! echo "${CHIEF_LOCAL_IPS}" | tr ' ' '\n' | grep -qx "${CHIEF_IP_GUESS}"; then
  info "WARNING: chief IP ${CHIEF_IP_GUESS} not on local interfaces (${CHIEF_LOCAL_IPS}). Continuing — first NODE_IP_LIST entry is treated as chief."
fi

declare -a PIDS=()
for i in "${!NODE_IPS[@]}"; do
  host="${NODE_IPS[$i]}"
  node_log="${LAUNCH_DIR}/launch.log.rank${i}"
  env_file="${LAUNCH_DIR}/.env.rank${i}"

  {
    echo "export NODE_IP_LIST='${NODE_IP_LIST}'"
    echo "export NODE_RANK=${i}"
    echo "export MASTER_ADDR=${CHIEF_IP_GUESS}"
    # Detached background torchrun so SSH returns quickly; the inner
    # launcher uses scripts/launch_detached.py and writes per-rank logs.
    echo "export FOREGROUND=0"
    echo "export RUN_NAME='${RUN_ID}'"
    echo "export LOG_ROOT='${LOG_ROOT}'"
    echo "export EXTRA='${EXTRA_BASE}'"
    if [[ -n "${PRETRAIN_CKPT:-}" ]]; then
      echo "export PRETRAIN_CKPT='${PRETRAIN_CKPT}'"
    fi
  } > "${env_file}"

  info "  -> rank ${i} on ${host}  (log: ${node_log})"

  if [[ "${i}" -eq 0 ]]; then
    bash -c "source '${env_file}' && bash '${INNER_SCRIPT}'" > "${node_log}" 2>&1 &
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${host}" \
        "bash -c \"source '${env_file}' && bash '${INNER_SCRIPT}'\"" \
        > "${node_log}" 2>&1 &
  fi
  PIDS+=("$!")
done

info "All node launchers kicked off. PIDs: ${PIDS[*]}"
info "Per-rank launch logs:  ${LAUNCH_DIR}/launch.log.rank{0..$((NNODES-1))}"
info "Per-rank train logs:   ${LOG_ROOT}/${RUN_ID}/train.log.rank{0..$((NNODES-1))}"
info "Tail rank-0 train log: tail -f ${LOG_ROOT}/${RUN_ID}/train.log.rank0"

# Wait for all launchers to return (they should return quickly because
# the inner launcher backgrounds the actual torchrun via launch_detached.py
# unless FOREGROUND=1 was set).
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || { info "WARNING: launcher PID=${pid} exited with non-zero status"; FAILED=1; }
done

if (( FAILED )); then
  info "One or more node launchers reported failure — check launch.log.rank* files above."
  exit 1
fi

info "Done — torchrun started on every node. Use scripts/monitor_training.sh or tail rank-0 to watch progress."
