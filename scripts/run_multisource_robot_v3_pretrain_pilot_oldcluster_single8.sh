#!/usr/bin/env bash
# Old non-AWS cluster: one node x 8 GPUs multisource V-JEPA Predictor pilot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

unset PET_NNODES PET_NODE_RANK PET_MASTER_ADDR PET_MASTER_PORT PET_NPROC_PER_NODE
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK
unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT NODE_IP_LIST
unset FASTWAM_MANAGED_DISTRIBUTED _MULTINODE_LAUNCHED

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

export INTERN_A1_ROOT="${INTERN_A1_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/InternData-A1_LeRobotv3}"
export AGIBOT_WORLD_ROOT="${AGIBOT_WORLD_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/AgiBot-World-Beta_lerobotv3}"
export DROID_ROOT="${DROID_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/DROID_LeRobot-v3.0}"
export OPEN_X_ROOT="${OPEN_X_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/Open-X-Embodiment}"
export ROBOCOIN_ROOT="${ROBOCOIN_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain_LeRobot/RoboCOIN_v3.0_official}"
export GALAXEA_ROOT="${GALAXEA_ROOT:-/apdcephfs_csgl/share_306089109/cheerchuang/Data_Pretrain/Galaxea-Open-World-Dataset-LeRobot-v3.0}"
for variable in \
  INTERN_A1_ROOT \
  AGIBOT_WORLD_ROOT \
  DROID_ROOT \
  OPEN_X_ROOT \
  ROBOCOIN_ROOT \
  GALAXEA_ROOT; do
  path="${!variable}"
  if [[ ! -d "${path}" ]]; then
    echo "[multisource-single8] ERROR: ${variable} does not exist: ${path}" >&2
    exit 1
  fi
done

MANIFEST_ROOT="${MULTISOURCE_MANIFEST_ROOT:-/apdcephfs_csgl/share_306089109/shaunxhwang/fastwam_manifests}"
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${MANIFEST_ROOT}/interndata_manifest_v5_10hz}"
export MULTISOURCE_VIDEO_MANIFEST_DIR="${MULTISOURCE_VIDEO_MANIFEST_DIR:-${MANIFEST_ROOT}/multisource_canonical_v5}"
python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}"
python scripts/build_multisource_video_manifest.py \
  --registry configs/data/multisource_robot_v3_registry.yaml \
  --output "${MULTISOURCE_VIDEO_MANIFEST_DIR}"

export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${MULTISOURCE_VIDEO_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_declared_weights_10hz_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[multisource-single8] ERROR: global stats not found: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Run scripts/precompute_multisource_vjepa21_global_stats_single8.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=8
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MULTISOURCE_SAMPLES_PER_EPOCH="${MULTISOURCE_SAMPLES_PER_EPOCH:-768000}"
export TASK_CONFIG=multisource_robot_v3_pilot_3cam_384
export DATA_CONFIG=multisource_robot_v3
export MODEL_CONFIG=hfastwam_small_vjepa21_predictor
export USE_ROBOTWIN_DATA_OVERRIDES=0
export SET_NUM_SEGMENTS=0
export NUM_EPOCHS=1
export MAX_STEPS="${MAX_STEPS:-1000}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATALOADER_PERSISTENT_WORKERS=true
export SAVE_EVERY="${SAVE_EVERY:-0}"
export LOG_EVERY=1
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export WANDB="${WANDB:-0}"
export RUN_NAME="${RUN_NAME:-multisource_robot_v3_oldcluster_single8_smoke}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" \
  "++model.language_pad_to_max_length=true" \
  "$@"
