#!/usr/bin/env bash
# InternData-A1 native LeRobot v3 pretraining: 2 nodes x 8 GPUs, global batch 768.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

export INTERN_A1_ROOT="${INTERN_A1_ROOT:-/fsx/pretrain_data/InternData-A1}"
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v5_10hz}"
if [[ ! -d "${INTERN_A1_ROOT}" ]]; then
  echo "[interndata-a1] ERROR: dataset root does not exist: ${INTERN_A1_ROOT}" >&2
  exit 1
fi

python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}"

export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${INTERN_A1_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_10hz_global_stats.pt}"
if [[ ! -f "${VJEPA21_NORMALISE_STATS_PATH}" ]]; then
  echo "[interndata-a1] ERROR: V-JEPA global stats do not exist: ${VJEPA21_NORMALISE_STATS_PATH}" >&2
  echo "Run scripts/precompute_interndata_vjepa21_global_stats_single8.sh first." >&2
  exit 1
fi
export STANDARDISE_OUTPUT=true

DATA_GATE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
if [[ "${FASTWAM_RUN_DATA_GATE:-1}" == "1" && "${DATA_GATE_RANK}" == "0" ]]; then
  python scripts/check_interndata_a1_10hz_gate.py \
    --root "${INTERN_A1_ROOT}" \
    --manifest-dir "${INTERN_A1_MANIFEST_DIR}" \
    --samples-per-family "${INTERN_A1_GATE_SAMPLES_PER_FAMILY:-2}"
fi

export FASTWAM_EXPECTED_WORLD_SIZE="${FASTWAM_EXPECTED_WORLD_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( 48 * FASTWAM_EXPECTED_WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))}"
export TASK_CONFIG=interndata_a1_pretrain_3cam_384_1e-4
export DATA_CONFIG=interndata_a1_v3
export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_vjepa21_predictor}"
export USE_ROBOTWIN_DATA_OVERRIDES=0
export SET_NUM_SEGMENTS=0
export NUM_EPOCHS=1
export MAX_STEPS=null
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATALOADER_PERSISTENT_WORKERS=true
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY=1
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA="${FASTWAM_USE_EFA:-1}"
export RUN_NAME="${RUN_NAME:-interndata_a1_vjepa21_predictor_pretrain_10hz_globalnorm_16gpu_b48}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-interndata-a1}"
export WANDB_GROUP="${WANDB_GROUP:-vjepa21-predictor-pretrain-globalnorm}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" \
  "++model.language_pad_to_max_length=true" \
  "$@"
