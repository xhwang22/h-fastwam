#!/usr/bin/env bash
# =============================================================================
# ONE-CLICK chief launcher: H-FastWAM SMALL (VAE) on RoboTwin, multi-node.
#
# Run this ONCE on the chief node (the first IP in NODE_IP_LIST). It will:
#   1. On every node (chief + workers via SSH): populate a LOCAL-DISK cache at
#      /tmp/fwam with the 75GB robotwin2.0 dataset
#      (scripts/_local_cache_robotwin_tmp.sh, with a live progress bar),
#      because CephFS/FUSE is too slow for the many small mp4 files. Using /tmp
#      (local overlay disk, 12T) instead of /dev/shm avoids eating ~86GB of RAM.
#   2. Launch detached torchrun on every node, training model=hfastwam_small
#      (VAE-48d, 2048/16/28 random-init, frozen Qwen language, language loss off),
#      reading the dataset from /tmp/fwam.
#
# The 4 nodes you gave are baked in as the default NODE_IP_LIST (8 GPUs each).
# Global batch is preserved at 128 by lowering per-GPU batch (128 / total_gpus).
#
# Usage (from the chief node = 28.216.16.72):
#   bash scripts/run_robotwin_hfastwam_vae_chief.sh
#
#   # Skip the cache step if /tmp/fwam is already populated on every node:
#   SKIP_CACHE=1 bash scripts/run_robotwin_hfastwam_vae_chief.sh
#
#   # Extra hydra overrides forwarded to every node:
#   EXTRA_OVERRIDES="num_epochs=8" bash scripts/run_robotwin_hfastwam_vae_chief.sh
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

err()  { echo "[robotwin-hfastwam-vae-chief] ERROR: $*" >&2; exit 1; }
info() { echo "[robotwin-hfastwam-vae-chief] $*"; }

# --- topology (your 4 nodes, 8 GPUs each, baked in as default) ---
export NODE_IP_LIST="${NODE_IP_LIST:-28.216.16.72:8,28.216.19.217:8,28.216.18.223:8,28.216.19.13:8}"

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

# Preserve original single-node global batch (16 * 8 = 128).
ORIG_GLOBAL_BATCH="${ORIG_GLOBAL_BATCH:-128}"
if (( ORIG_GLOBAL_BATCH % TOTAL_GPUS != 0 )); then
  err "Cannot evenly split global batch ${ORIG_GLOBAL_BATCH} across ${TOTAL_GPUS} GPUs."
fi
PER_GPU_BATCH=$(( ORIG_GLOBAL_BATCH / TOTAL_GPUS ))

# --- local-disk cache locations (populated by _local_cache_robotwin_tmp.sh) ---
# Uses /tmp (local overlay disk, 12T) instead of /dev/shm so it does NOT eat RAM.
LOCAL_BASE="${LOCAL_CACHE_BASE:-/tmp/fwam}"
LOCAL_DATA_DIR="${LOCAL_BASE}/data/robotwin2.0/robotwin2.0"
LOCAL_STATS="${LOCAL_BASE}/data/robotwin2.0/dataset_stats.json"
CACHE_SCRIPT="${SCRIPT_DIR}/_local_cache_robotwin_tmp.sh"
[[ -f "${CACHE_SCRIPT}" ]] || err "Cache script missing: ${CACHE_SCRIPT}"

INNER_SCRIPT="${SCRIPT_DIR}/run_robotwin_dino_ditproj_multinode.sh"
[[ -f "${INNER_SCRIPT}" ]] || err "Inner launcher missing: ${INNER_SCRIPT}"

RUN_ID="${RUN_ID:-robotwin_hfastwam_vae_mn_$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/robotwin_hfastwam_vae}"
LAUNCH_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${LAUNCH_DIR}"

# --- hydra overrides forwarded to every node ---
# Data → local /tmp disk; H-FastWAM VAE recipe (matches the libero small-vae run):
#   random-init both experts, frozen language, language loss off, KI off.
SHM_OVERRIDES="data.train.dataset_dirs=[${LOCAL_DATA_DIR}] data.val.dataset_dirs=[${LOCAL_DATA_DIR}] data.train.pretrained_norm_stats=${LOCAL_STATS} data.val.pretrained_norm_stats=${LOCAL_STATS}"
HFASTWAM_OVERRIDES="model.skip_dit_load_from_pretrain=true model.skip_video_dit_load_from_pretrain=true model.action_dit_pretrained_path=null model.knowledge_insulation=false model.freeze_language_expert=true model.freeze_video_expert=false model.freeze_action_expert=false model.loss_config.lambda_language=0.0"

EXTRA_BASE="batch_size=${PER_GPU_BATCH} ${SHM_OVERRIDES} ${HFASTWAM_OVERRIDES}"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  EXTRA_BASE="${EXTRA_BASE} ${EXTRA_OVERRIDES}"
fi

echo "============================================================="
echo " RoboTwin + H-FastWAM SMALL (VAE) — One-Click Multi-Node"
echo "============================================================="
echo " NODE_IP_LIST   = ${NODE_IP_LIST}"
echo " NNODES         = ${NNODES}   NPROC_PER_NODE = ${NPROC_PER_NODE}   TOTAL_GPUS = ${TOTAL_GPUS}"
echo " PER_GPU_BATCH  = ${PER_GPU_BATCH}  (global = ${ORIG_GLOBAL_BATCH})"
echo " MODEL=hfastwam_small  TASK=robotwin_uncond_3cam_384_1e-4  DATA=robotwin"
echo " SKIP_CACHE     = ${SKIP_CACHE:-0}   (cache → ${LOCAL_BASE} on local disk)"
echo " RUN_ID         = ${RUN_ID}"
echo " LOG_ROOT       = ${LOG_ROOT}"
echo " EXTRA          = ${EXTRA_BASE}"
echo "============================================================="

CHIEF_IP="${NODE_IPS[0]}"

# --- Step 1: populate the local-disk cache on every node (idempotent) ---
if [[ "${SKIP_CACHE:-0}" != "1" ]]; then
  info "Populating ${LOCAL_BASE} cache on every node (idempotent; live progress for rank 0)..."
  declare -a CACHE_PIDS=()
  for i in "${!NODE_IPS[@]}"; do
    host="${NODE_IPS[$i]}"
    clog="${LAUNCH_DIR}/cache.log.rank${i}"
    if [[ "${i}" -eq 0 ]]; then
      # rank 0 runs locally; tee so the chief terminal shows the live bar.
      LOCAL_CACHE_BASE="${LOCAL_BASE}" bash "${CACHE_SCRIPT}" 2>&1 | tee "${clog}" &
    else
      ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${host}" \
          "cd '${REPO_ROOT}' && LOCAL_CACHE_BASE='${LOCAL_BASE}' bash '${CACHE_SCRIPT}'" > "${clog}" 2>&1 &
    fi
    CACHE_PIDS+=("$!")
    info "  -> cache on rank ${i} (${host}), log: ${clog}"
  done
  CACHE_FAIL=0
  for pid in "${CACHE_PIDS[@]}"; do
    wait "${pid}" || { info "WARNING: cache PID=${pid} failed"; CACHE_FAIL=1; }
  done
  (( CACHE_FAIL )) && err "One or more nodes failed to populate ${LOCAL_BASE} cache — see cache.log.rank* above."
  info "All nodes cached to ${LOCAL_BASE}."
else
  info "SKIP_CACHE=1 — assuming ${LOCAL_BASE} already populated on every node."
fi

# --- Step 2: launch detached torchrun on every node ---
declare -a PIDS=()
for i in "${!NODE_IPS[@]}"; do
  host="${NODE_IPS[$i]}"
  node_log="${LAUNCH_DIR}/launch.log.rank${i}"
  env_file="${LAUNCH_DIR}/.env.rank${i}"

  {
    echo "export NODE_IP_LIST='${NODE_IP_LIST}'"
    echo "export NODE_RANK=${i}"
    echo "export MASTER_ADDR=${CHIEF_IP}"
    echo "export FOREGROUND=0"
    echo "export MODEL='hfastwam_small'"
    echo "export TASK='robotwin_uncond_3cam_384_1e-4'"
    echo "export DATA='robotwin'"
    echo "export RUN_PREFIX='robotwin_hfastwam_vae_mn'"
    echo "export WANDB_NAME='${RUN_ID}'"
    echo "export RUN_NAME='${RUN_ID}'"
    echo "export LOG_ROOT='${LOG_ROOT}'"
    echo "export EXTRA='${EXTRA_BASE}'"
  } > "${env_file}"

  info "  -> rank ${i} on ${host}  (log: ${node_log})"

  if [[ "${i}" -eq 0 ]]; then
    bash -c "source '${env_file}' && bash '${INNER_SCRIPT}'" > "${node_log}" 2>&1 &
  else
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${host}" \
        "bash -c \"cd '${REPO_ROOT}' && source '${env_file}' && bash '${INNER_SCRIPT}'\"" \
        > "${node_log}" 2>&1 &
  fi
  PIDS+=("$!")
done

info "All node launchers kicked off. PIDs: ${PIDS[*]}"
info "Per-rank launch logs: ${LAUNCH_DIR}/launch.log.rank{0..$((NNODES-1))}"
info "Tail rank-0 train log: tail -f ${LOG_ROOT}/${RUN_ID}/train.log.rank0"

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || { info "WARNING: launcher PID=${pid} exited non-zero"; FAILED=1; }
done

(( FAILED )) && err "One or more node launchers reported failure — check launch.log.rank* files."
info "Done — torchrun started on every node. Tail rank-0 to watch progress."
