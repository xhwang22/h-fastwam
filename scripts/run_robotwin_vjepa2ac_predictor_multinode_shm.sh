#!/usr/bin/env bash
# Wrapper around run_robotwin_vjepa2ac_predictor_multinode.sh that
# redirects dataset + checkpoint reads to /dev/shm/fwam (local RAM cache
# populated by scripts/_local_cache_robotwin.sh on every node).
#
# Why: CephFS/FUSE was too slow for both the 11GB V-JEPA2-AC checkpoint
# and the 75GB robotwin2.0 dataset (lots of small mp4 files), causing
# 8 dataloader workers per node to deadlock on FUSE metadata calls.

set -euo pipefail

SHM_BASE="/dev/shm/fwam"
SHM_CKPT="${SHM_BASE}/ckpt/vjepa2/vjepa2-ac-vitg.pt"
SHM_DATA_DIR="${SHM_BASE}/data/robotwin2.0/robotwin2.0"
SHM_STATS="${SHM_BASE}/data/robotwin2.0/dataset_stats.json"

# Sanity check on rank 0 only
NODE_RANK_RESOLVED="${NODE_RANK:-${INDEX:-}}"
if [[ -z "${NODE_RANK_RESOLVED}" ]]; then
  # Try to detect from NODE_IP_LIST + hostname -I (same logic as inner script)
  if [[ -n "${NODE_IP_LIST:-}" ]]; then
    IFS=',' read -ra _NL <<< "${NODE_IP_LIST}"
    LIPS=$(hostname -I 2>/dev/null || true)
    for i in "${!_NL[@]}"; do
      ip="${_NL[$i]%%:*}"
      for lip in ${LIPS}; do
        if [[ "${lip}" == "${ip}" ]]; then NODE_RANK_RESOLVED="${i}"; break 2; fi
      done
    done
  fi
fi
NODE_RANK_RESOLVED="${NODE_RANK_RESOLVED:-0}"

if [[ "${NODE_RANK_RESOLVED}" == "0" ]]; then
  [[ -s "${SHM_CKPT}" ]] || { echo "[shm-launch] ERROR: ${SHM_CKPT} missing — run _local_cache_robotwin.sh first" >&2; exit 1; }
  [[ -d "${SHM_DATA_DIR}" ]] || { echo "[shm-launch] ERROR: ${SHM_DATA_DIR} missing — run _local_cache_robotwin.sh first" >&2; exit 1; }
  [[ -s "${SHM_STATS}" ]] || { echo "[shm-launch] ERROR: ${SHM_STATS} missing — run _local_cache_robotwin.sh first" >&2; exit 1; }
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build Hydra overrides:
#   * data.train.dataset_dirs / val.dataset_dirs   -> /dev/shm
#   * data.train.pretrained_norm_stats / val       -> /dev/shm
#   * model.visual_encoder.checkpoint_path         -> /dev/shm
#   * model.video_dit_config.predictor_ckpt_path   -> /dev/shm
SHM_OVERRIDES="\
data.train.dataset_dirs=[${SHM_DATA_DIR}] \
data.val.dataset_dirs=[${SHM_DATA_DIR}] \
data.train.pretrained_norm_stats=${SHM_STATS} \
data.val.pretrained_norm_stats=${SHM_STATS} \
model.visual_encoder.checkpoint_path=${SHM_CKPT} \
model.video_dit_config.predictor_ckpt_path=${SHM_CKPT}"

# Append to any caller-provided EXTRA
export EXTRA="${SHM_OVERRIDES} ${EXTRA:-}"

echo "[shm-launch] Using SHM cache at ${SHM_BASE}"
echo "[shm-launch] EXTRA=${EXTRA}"

exec bash "${SCRIPT_DIR}/run_robotwin_vjepa2ac_predictor_multinode.sh"
