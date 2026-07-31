#!/usr/bin/env bash
# =============================================================================
# JEPA (V-JEPA2-AC predictor) + RoboTwin — chief-orchestrated multi-node
#
# Run this ONCE on the chief node. It:
#   1. Resolves topology from NODE_IP_LIST.
#   2. Points the data config at the SHARED tj5 copy of the RoboTwin
#      dataset + text-embedding cache (tj5 is ~1GB/s read, gy2 FUSE is 21MB/s).
#      Pre-stage with scripts/sync_robotwin_to_tj5.sh if not already done.
#   3. SSH-launches run_robotwin_vjepa2ac_predictor_multinode.sh on every node
#      with NODE_RANK / NODE_IP_LIST / MASTER_ADDR / hydra data overrides.
#   4. Tails rank-0 log on the chief.
#
# Usage:
#   NODE_IP_LIST="$NODE_IP_LIST" bash scripts/run_robotwin_jepa_chief.sh
#   EXTRA="max_steps=80000" bash scripts/run_robotwin_jepa_chief.sh
#
#   # Use a different dataset path (e.g. point back to gy2 FUSE):
#   DATA_ROOT=/path/to/robotwin2.0/robotwin2.0 \
#   DATA_STATS=/path/to/robotwin2.0/dataset_stats.json \
#   TEXT_CACHE=/path/to/text_embeds_cache/robotwin \
#       bash scripts/run_robotwin_jepa_chief.sh
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

err()  { echo "[jepa-chief] ERROR: $*" >&2; exit 1; }
info() { echo "[jepa-chief] $*"; }

# ---------------------------------------------------------------------------
# 1. Resolve topology
# ---------------------------------------------------------------------------
: "${NODE_IP_LIST:?NODE_IP_LIST must be set, e.g. ip1:8,ip2:8,ip3:8,ip4:8}"

IFS=',' read -ra NODE_LIST <<< "${NODE_IP_LIST}"
NNODES="${#NODE_LIST[@]}"

declare -a NODE_IPS=()
declare -a NODE_GPUS=()
for entry in "${NODE_LIST[@]}"; do
  NODE_IPS+=("${entry%%:*}")
  NODE_GPUS+=("${entry##*:}")
done

MASTER_ADDR="${NODE_IPS[0]}"
NPROC_PER_NODE="${NODE_GPUS[0]}"
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))

# ---------------------------------------------------------------------------
# 2. Data paths — default to the shared tj5 copy (one mount, all nodes see it).
# ---------------------------------------------------------------------------
TJ5_DATA_BASE="${REPO_ROOT}/data"
DATA_ROOT="${DATA_ROOT:-${TJ5_DATA_BASE}/robotwin2.0/robotwin2.0}"
DATA_STATS="${DATA_STATS:-${TJ5_DATA_BASE}/robotwin2.0/dataset_stats.json}"
# Text embeddings are tiny (1MB each, read once per episode) and the rsync of
# the full cache from gy2 to tj5 is impractically slow due to small-file
# overhead. Default to reading text_embeds from gy2 FUSE — it does not
# bottleneck training. Override via TEXT_CACHE=... if you've pre-synced.
TEXT_CACHE="${TEXT_CACHE:-${REPO_ROOT}/data/text_embeds_cache/robotwin}"

[[ -d "${DATA_ROOT}" ]]   || err "DATA_ROOT does not exist: ${DATA_ROOT}"
[[ -f "${DATA_STATS}" ]]  || err "DATA_STATS not found: ${DATA_STATS}"
[[ -d "${TEXT_CACHE}" ]]  || err "TEXT_CACHE not found: ${TEXT_CACHE}"

RUN_PREFIX="${RUN_PREFIX:-robotwin_jepa_mn}"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/robotwin_vjepa2ac_predictor}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

EXTRA_USER="${EXTRA:-}"
MASTER_PORT="${MASTER_PORT:-29501}"

info "============================================="
info " JEPA + RoboTwin — multi-node training"
info " NNODES         = ${NNODES}"
info " NPROC_PER_NODE = ${NPROC_PER_NODE}"
info " TOTAL_GPUS     = ${TOTAL_GPUS}"
info " MASTER_ADDR    = ${MASTER_ADDR}"
info " MASTER_PORT    = ${MASTER_PORT}"
info " NODE_IPS       = ${NODE_IPS[*]}"
info " RUN_NAME       = ${RUN_NAME}"
info " LOG_DIR        = ${LOG_DIR}"
info " DATA_ROOT      = ${DATA_ROOT}"
info " DATA_STATS     = ${DATA_STATS}"
info " TEXT_CACHE     = ${TEXT_CACHE}"
info "============================================="

# ---------------------------------------------------------------------------
# 3. Build EXTRA hydra-overrides string forwarded to every node.
# ---------------------------------------------------------------------------
DATA_OVERRIDES=(
  "data.train.dataset_dirs=[${DATA_ROOT}]"
  "data.val.dataset_dirs=[${DATA_ROOT}]"
  "data.train.pretrained_norm_stats=${DATA_STATS}"
  "data.val.pretrained_norm_stats=${DATA_STATS}"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}"
  "data.val.text_embedding_cache_dir=${TEXT_CACHE}"
)
EXTRA_FULL="${DATA_OVERRIDES[*]}"
if [[ -n "${EXTRA_USER}" ]]; then
  EXTRA_FULL="${EXTRA_FULL} ${EXTRA_USER}"
fi
info "EXTRA hydra overrides:"
info "  ${EXTRA_FULL}"

# ---------------------------------------------------------------------------
# 4. Launch the per-node inner script on every node (parallel, via SSH).
# ---------------------------------------------------------------------------
INNER_SCRIPT="${SCRIPT_DIR}/run_robotwin_vjepa2ac_predictor_multinode.sh"
[[ -f "${INNER_SCRIPT}" ]] || err "Inner script not found: ${INNER_SCRIPT}"

launch_node() {
  local host="$1"
  local rank="$2"
  local launch_log="${LOG_DIR}/launch.log.rank${rank}"

  # Note: env vars need to survive shell quoting through ssh.
  local cmd="\
    export NODE_IP_LIST='${NODE_IP_LIST}'; \
    export NODE_RANK=${rank}; \
    export MASTER_ADDR='${MASTER_ADDR}'; \
    export MASTER_PORT='${MASTER_PORT}'; \
    export NPROC_PER_NODE=${NPROC_PER_NODE}; \
    export RUN_NAME='${RUN_NAME}'; \
    export LOG_ROOT='${LOG_ROOT}'; \
    export FOREGROUND=0; \
    export EXTRA='${EXTRA_FULL}'; \
    bash '${INNER_SCRIPT}'"

  if [[ "${rank}" -eq 0 ]]; then
    bash -c "${cmd}" > "${launch_log}" 2>&1
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${host}" \
        "${cmd}" > "${launch_log}" 2>&1
  fi
}

info "Launching training on all ${NNODES} nodes..."
launch_pids=()
for i in "${!NODE_IPS[@]}"; do
  info "  -> rank ${i} on ${NODE_IPS[$i]}  (launch log: ${LOG_DIR}/launch.log.rank${i})"
  launch_node "${NODE_IPS[$i]}" "${i}" &
  launch_pids+=($!)
done

for pid in "${launch_pids[@]}"; do
  wait "${pid}" || info "WARNING: launch wrapper exited non-zero (training may still be running)."
done

info "All per-node launchers returned. Training is detached on each node."
info ""
info "Rank-0 train log: ${LOG_DIR}/train.log.rank0"
info "Tail it with:     tail -f ${LOG_DIR}/train.log.rank0"
info ""
info "To stop all nodes:"
for h in "${NODE_IPS[@]}"; do
  info "  ssh ${h} 'pkill -TERM -f scripts/train.py; pkill -TERM -f torchrun'"
done

# ---------------------------------------------------------------------------
# 5. (Optional) Stream rank-0 log so the user sees startup progress.
# ---------------------------------------------------------------------------
TAIL_RANK0="${TAIL_RANK0:-1}"
if [[ "${TAIL_RANK0}" == "1" ]]; then
  RANK0_LOG="${LOG_DIR}/train.log.rank0"
  info "Waiting for rank-0 train log to appear..."
  for _ in $(seq 1 60); do
    [[ -f "${RANK0_LOG}" ]] && break
    sleep 2
  done
  if [[ -f "${RANK0_LOG}" ]]; then
    info "Streaming rank-0 log (Ctrl-C to stop tailing — training keeps running):"
    exec tail -F "${RANK0_LOG}"
  else
    info "Rank-0 log did not appear in 120s. Check ${LOG_DIR}/launch.log.rank0"
  fi
fi
