#!/usr/bin/env bash
# Old non-AWS cluster: two nodes x 8 GPUs, batch 48, global batch 768.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${NODE_IP_LIST:-}" ]]; then
  echo "[multisource-oldcluster] ERROR: set NODE_IP_LIST to exactly two nodes." >&2
  exit 1
fi
IFS=',' read -ra MULTISOURCE_NODES <<< "${NODE_IP_LIST}"
if [[ "${#MULTISOURCE_NODES[@]}" -ne 2 ]]; then
  echo "[multisource-oldcluster] ERROR: expected 2 nodes, got ${#MULTISOURCE_NODES[@]}." >&2
  exit 1
fi

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

if [[ -z "${INTERN_A1_ROOT:-}" ]]; then
  echo "[multisource-oldcluster] ERROR: set INTERN_A1_ROOT on the old cluster." >&2
  exit 1
fi
export AGIBOT_WORLD_ROOT="${AGIBOT_WORLD_ROOT:-/apdcephfs/cs/jp_csgl/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/AgiBot-World-Beta_lerobotv3}"
export DROID_ROOT="${DROID_ROOT:-/apdcephfs/cs/jp_csgl/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/DROID_LeRobot-v3.0}"
export OPEN_X_ROOT="${OPEN_X_ROOT:-/apdcephfs/cs/jp_csgl/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/Open-X-Embodiment}"
export ROBOCOIN_ROOT="${ROBOCOIN_ROOT:-/apdcephfs/cs/jp_csgl/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/RoboCOIN_v3.0_official}"
export GALAXEA_ROOT="${GALAXEA_ROOT:-/apdcephfs/cs/jp_csgl/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain/Galaxea-Open-World-Dataset-LeRobot-v3.0}"
for variable in \
  INTERN_A1_ROOT \
  AGIBOT_WORLD_ROOT \
  DROID_ROOT \
  OPEN_X_ROOT \
  ROBOCOIN_ROOT \
  GALAXEA_ROOT; do
  path="${!variable}"
  if [[ ! -d "${path}" ]]; then
    echo "[multisource-oldcluster] ERROR: ${variable} does not exist: ${path}" >&2
    exit 1
  fi
done

export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v3}"
export MULTISOURCE_VIDEO_MANIFEST_DIR="${MULTISOURCE_VIDEO_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_multisource/canonical_manifest_v4}"
python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}"
python scripts/build_multisource_video_manifest.py \
  --registry configs/data/multisource_robot_v3_registry.yaml \
  --output "${MULTISOURCE_VIDEO_MANIFEST_DIR}"

export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${MULTISOURCE_VIDEO_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_declared_weights_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[multisource-oldcluster] ERROR: global stats not found: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=16
export GLOBAL_BATCH_SIZE=768
export GRADIENT_ACCUMULATION_STEPS=1
export MULTISOURCE_SAMPLES_PER_EPOCH="${MULTISOURCE_SAMPLES_PER_EPOCH:-768000}"
export TASK_CONFIG=multisource_robot_v3_pilot_3cam_384
export DATA_CONFIG=multisource_robot_v3
export MODEL_CONFIG=hfastwam_small_vjepa21_predictor
export USE_ROBOTWIN_DATA_OVERRIDES=0
export SET_NUM_SEGMENTS=0
export NUM_EPOCHS=1
export MAX_STEPS="${MAX_STEPS:-1000}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATALOADER_PERSISTENT_WORKERS=true
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-multisource-robot}"
export WANDB_GROUP="${WANDB_GROUP:-vjepa21-predictor-pilot-oldcluster}"
export RUN_NAME="${RUN_NAME:-multisource_robot_v3_pilot_oldcluster_2x8_b48_gb768}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" \
  "++model.language_pad_to_max_length=true" \
  "$@"
