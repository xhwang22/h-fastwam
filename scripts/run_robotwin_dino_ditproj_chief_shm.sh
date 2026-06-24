#!/usr/bin/env bash
# Wrapper around run_robotwin_dino_ditproj_chief.sh that redirects the
# robotwin dataset reads to /dev/shm/fwam (per-node RAM cache populated
# by scripts/_local_cache_robotwin_dino.sh).
#
# Why: CephFS/FUSE was too slow for 75GB of small mp4 files; 8 dataloader
# workers per node deadlocked on FUSE metadata calls (d_alloc_parallel /
# request_wait_answer kernel waits) before reaching the first training
# step.
#
# Usage (from chief node only):
#   NODE_IP_LIST=ip1:8,ip2:8,ip3:8,ip4:8 \
#       bash scripts/run_robotwin_dino_ditproj_chief_shm.sh
#
#   # With pretrain checkpoint:
#   PRETRAIN_CKPT=/path/to/step_50000.pt \
#       bash scripts/run_robotwin_dino_ditproj_chief_shm.sh

set -euo pipefail

SHM_BASE="/dev/shm/fwam"
SHM_DATA_DIR="${SHM_BASE}/data/robotwin2.0/robotwin2.0"
SHM_STATS="${SHM_BASE}/data/robotwin2.0/dataset_stats.json"

# Sanity check on chief
[[ -d "${SHM_DATA_DIR}" ]] || { echo "[shm-launch] ERROR: ${SHM_DATA_DIR} missing — run scripts/_local_cache_robotwin_dino.sh on every node first" >&2; exit 1; }
[[ -s "${SHM_STATS}" ]]   || { echo "[shm-launch] ERROR: ${SHM_STATS} missing — run scripts/_local_cache_robotwin_dino.sh on every node first" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hydra overrides forwarded to every node via EXTRA_OVERRIDES.
SHM_OVERRIDES="data.train.dataset_dirs=[${SHM_DATA_DIR}] data.val.dataset_dirs=[${SHM_DATA_DIR}] data.train.pretrained_norm_stats=${SHM_STATS} data.val.pretrained_norm_stats=${SHM_STATS}"

# Append to any caller-provided EXTRA_OVERRIDES
export EXTRA_OVERRIDES="${SHM_OVERRIDES} ${EXTRA_OVERRIDES:-}"

echo "[shm-launch] Using SHM cache at ${SHM_BASE}"
echo "[shm-launch] EXTRA_OVERRIDES=${EXTRA_OVERRIDES}"

exec bash "${SCRIPT_DIR}/run_robotwin_dino_ditproj_chief.sh"
