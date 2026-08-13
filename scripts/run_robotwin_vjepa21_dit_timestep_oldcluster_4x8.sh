#!/usr/bin/env bash
# Old non-AWS cluster: four-node V-JEPA 2.1 DiT timestep ablation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "[vjepa21-dit-oldcluster-4x8] ERROR: set NODE_IP_LIST to exactly four nodes." >&2
  echo 'Example: NODE_IP_LIST="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4" bash <script>' >&2
  exit 1
fi
IFS=',' read -ra VJEPA21_OLDCLUSTER_NODES <<< "${NODE_IP_LIST}"
if [[ "${#VJEPA21_OLDCLUSTER_NODES[@]}" -ne 4 ]]; then
  echo "[vjepa21-dit-oldcluster-4x8] ERROR: expected 4 nodes, got ${#VJEPA21_OLDCLUSTER_NODES[@]}: ${NODE_IP_LIST}" >&2
  exit 1
fi

# shellcheck source=_timestep_sampling_preset.sh
source "${SCRIPT_DIR}/_timestep_sampling_preset.sh"
fastwam_timestep_sampling_preset "${TIMESTEP_SAMPLING_PRESET:-baseline}" both

ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[vjepa21-dit-oldcluster-4x8] ERROR: global norm stats not found: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Set VJEPA21_NORMALISE_STATS_PATH to a shared path visible on all four nodes." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=32
export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=1
export MODEL_CONFIG=hfastwam_small_vjepa21
export STANDARDISE_OUTPUT=true
export CAUSAL_TUBELET_ENCODING=true
export TEMPORAL_DOWNSAMPLE=4
export TIMESTEP_SAMPLING_TARGET=both
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-vjepa21-dit}"
export WANDB_GROUP="${WANDB_GROUP:-timestep-globalnorm-oldcluster-4x8}"
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_dit_${TIMESTEP_PRESET_SUFFIX}_globalnorm_oldcluster_4x8_b48_acc1_gb1536}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21.sh" "$@"
