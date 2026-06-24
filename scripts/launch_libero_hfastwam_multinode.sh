#!/usr/bin/env bash
# Chief-only one-click launcher for hfastwam LIBERO multi-node training.
#
# Run this once on the chief node inside tmux. It starts non-chief ranks in the
# background via SSH, then runs rank0 in the foreground in the current shell.
#
# Usage:
#   bash scripts/launch_libero_hfastwam_multinode.sh
#   RUN_NAME=myrun bash scripts/launch_libero_hfastwam_multinode.sh
#   EXTRA="max_steps=80000" bash scripts/launch_libero_hfastwam_multinode.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[launch-libero-hfastwam-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[launch-libero-hfastwam-mn] $*"; }

append_remote_env() {
  local key="$1"
  local value="$2"
  local assignment
  printf -v assignment '%s=%q' "${key}" "${value}"
  REMOTE_ENV_PARTS+=( "${assignment}" )
}

HOSTFILE="${HOSTFILE:-/etc/taiji/hostfile}"
[[ -f "${HOSTFILE}" ]] || err "Hostfile not found: ${HOSTFILE}"
mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
[[ "${#HOSTS[@]}" -gt 0 ]] || err "No hosts parsed from ${HOSTFILE}"
CHIEF_HOST="${CHIEF_HOST:-${HOSTS[0]}}"

RUN_PREFIX="${RUN_PREFIX:-libero_hfastwam_mn}"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
AUTO_KILL_EXISTING="${AUTO_KILL_EXISTING:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/libero_hfastwam}"

REMOTE_ENV_PARTS=()
append_remote_env RUN_NAME "${RUN_NAME}"
append_remote_env RUN_PREFIX "${RUN_PREFIX}"
append_remote_env AUTO_KILL_EXISTING "${AUTO_KILL_EXISTING}"
append_remote_env MASTER_PORT "${MASTER_PORT}"
append_remote_env LOG_ROOT "${LOG_ROOT}"

for key in \
  ACTION_DIT_PRETRAINED_PATH \
  BATCH_SIZE \
  GRADIENT_ACCUMULATION_STEPS \
  DS_CONFIG \
  PYTORCH_CUDA_ALLOC_CONF \
  MODEL \
  TASK \
  DATA \
  LIBERO_DATA_ROOT \
  LIBERO_SOURCE_ROOT \
  CACHE_LIBERO_LOCAL \
  LOCAL_LIBERO_DATA_ROOT \
  LOCAL_CACHE_PARALLEL \
  WANDB_NAME \
  LOG_EVERY \
  NUM_WORKERS \
  DATALOADER_TIMEOUT \
  NUM_SEGMENTS \
  SEGMENT_STRIDE \
  EXTRA; do
  if [[ -n "${!key:-}" ]]; then
    append_remote_env "${key}" "${!key}"
  fi
done

printf -v REMOTE_ENV '%s ' "${REMOTE_ENV_PARTS[@]}"
export REMOTE_ENV="${REMOTE_ENV% }"
export HOSTFILE

info "RUN_NAME=${RUN_NAME}"
info "HOSTFILE=${HOSTFILE}"
info "CHIEF_HOST=${CHIEF_HOST}"
info "REMOTE_ENV=${REMOTE_ENV}"

for host in "${HOSTS[@]}"; do
  if [[ "${host}" == "${CHIEF_HOST}" ]]; then
    continue
  fi
  info "Starting non-chief node in background: ${host}"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
      "${REMOTE_ENV} FOREGROUND=0 bash ${SCRIPT_DIR}/run_libero_hfastwam_multinode.sh" \
    || err "Launch failed on ${host}"
done

info "Starting rank0 in foreground. Ctrl-C stops rank0; stop other nodes separately if needed."
export RUN_NAME RUN_PREFIX AUTO_KILL_EXISTING MASTER_PORT LOG_ROOT
export FOREGROUND=1
exec bash "${SCRIPT_DIR}/run_libero_hfastwam_multinode.sh"
