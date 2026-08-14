#!/usr/bin/env bash
# Route-safe multisource pilot with per-source EEF20 adapters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-/fsx/pretrain_data}"
export INTERN_A1_ROOT="${INTERN_A1_ROOT:-${PRETRAIN_DATA_ROOT}/InternData-A1}"
export AGIBOT_WORLD_ROOT="${AGIBOT_WORLD_ROOT:-${PRETRAIN_DATA_ROOT}/AgiBot-World-Beta_lerobotv3}"
export DROID_ROOT="${DROID_ROOT:-${PRETRAIN_DATA_ROOT}/DROID_LeRobot-v3.0}"
export OPEN_X_ROOT="${OPEN_X_ROOT:-${PRETRAIN_DATA_ROOT}/Open-X-Embodiment}"
export ROBOCOIN_ROOT="${ROBOCOIN_ROOT:-${PRETRAIN_DATA_ROOT}/RoboCOIN_v3.0_official}"
export GALAXEA_ROOT="${GALAXEA_ROOT:-${PRETRAIN_DATA_ROOT}/Galaxea-Open-World-Dataset-LeRobot-v3.0}"

for variable in \
  INTERN_A1_ROOT \
  AGIBOT_WORLD_ROOT \
  DROID_ROOT \
  OPEN_X_ROOT \
  ROBOCOIN_ROOT \
  GALAXEA_ROOT; do
  path="${!variable}"
  if [[ ! -d "${path}" ]]; then
    echo "[multisource-v3] ERROR: ${variable} does not exist: ${path}" >&2
    exit 1
  fi
done

export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v5_10hz}"
export MULTISOURCE_VIDEO_MANIFEST_DIR="${MULTISOURCE_VIDEO_MANIFEST_DIR:-${PRETRAIN_DATA_ROOT}/.fastwam_multisource/canonical_manifest_v5}"
MULTISOURCE_REGISTRY="${MULTISOURCE_REGISTRY:-configs/data/multisource_robot_v3_registry.yaml}"

python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}"
python scripts/build_multisource_video_manifest.py \
  --registry "${MULTISOURCE_REGISTRY}" \
  --output "${MULTISOURCE_VIDEO_MANIFEST_DIR}"

export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${MULTISOURCE_VIDEO_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_declared_weights_10hz_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[multisource-v3] ERROR: V-JEPA global stats do not exist: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Run scripts/precompute_multisource_vjepa21_global_stats_single8.sh first." >&2
  exit 1
fi

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
export MULTISOURCE_SAMPLES_PER_EPOCH="${MULTISOURCE_SAMPLES_PER_EPOCH:-768000}"
export TASK_CONFIG=multisource_robot_v3_pilot_3cam_384
export DATA_CONFIG=multisource_robot_v3
export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_vjepa21_predictor}"
export USE_ROBOTWIN_DATA_OVERRIDES=0
export SET_NUM_SEGMENTS=0
export NUM_EPOCHS=1
export MAX_STEPS="${MAX_STEPS:-1000}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATALOADER_PERSISTENT_WORKERS=true
export SAVE_EVERY="${SAVE_EVERY:-500}"
export LOG_EVERY=1
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export RUN_NAME="${RUN_NAME:-multisource_robot_v3_pilot_declared_weights}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-multisource-robot}"
export WANDB_GROUP="${WANDB_GROUP:-vjepa21-predictor-pilot}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" \
  "++model.language_pad_to_max_length=true" \
  "$@"
