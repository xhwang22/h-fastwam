#!/usr/bin/env bash
# InternData-A1 30Hz VAE+DiT pretraining baseline: 2 nodes x 8 GPUs.
# Preserve an effective batch contribution of 48 samples/GPU while using a
# smaller micro-batch for the higher-token-count VAE representation.
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
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v5_30hz}"
if [[ ! -d "${INTERN_A1_ROOT}" ]]; then
  echo "[interndata-a1-vae-dit] ERROR: dataset root does not exist: ${INTERN_A1_ROOT}" >&2
  exit 1
fi

python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}" \
  --target-control-hz 30

DATA_GATE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
if [[ "${FASTWAM_RUN_DATA_GATE:-1}" == "1" && "${DATA_GATE_RANK}" == "0" ]]; then
  python scripts/check_interndata_a1_30hz_gate.py \
    --root "${INTERN_A1_ROOT}" \
    --manifest-dir "${INTERN_A1_MANIFEST_DIR}" \
    --samples-per-family "${INTERN_A1_GATE_SAMPLES_PER_FAMILY:-2}"
fi

export FASTWAM_EXPECTED_WORLD_SIZE="${FASTWAM_EXPECTED_WORLD_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( 48 * FASTWAM_EXPECTED_WORLD_SIZE ))}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-24}"
if [[ ! "${PER_GPU_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[interndata-a1-vae-dit] ERROR: PER_GPU_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
batch_denominator=$(( FASTWAM_EXPECTED_WORLD_SIZE * PER_GPU_BATCH_SIZE ))
if (( GLOBAL_BATCH_SIZE % batch_denominator != 0 )); then
  echo "[interndata-a1-vae-dit] ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by world_size*micro_batch=${batch_denominator}." >&2
  exit 2
fi
derived_gradient_accumulation=$(( GLOBAL_BATCH_SIZE / batch_denominator ))
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]] && \
   (( GRADIENT_ACCUMULATION_STEPS != derived_gradient_accumulation )); then
  echo "[interndata-a1-vae-dit] ERROR: GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} conflicts with global batch ${GLOBAL_BATCH_SIZE}, world size ${FASTWAM_EXPECTED_WORLD_SIZE}, and micro-batch ${PER_GPU_BATCH_SIZE}; expected ${derived_gradient_accumulation}." >&2
  exit 2
fi
export GRADIENT_ACCUMULATION_STEPS="${derived_gradient_accumulation}"
export TASK_CONFIG=interndata_a1_pretrain_3cam_384_1e-4
export DATA_CONFIG=interndata_a1_v3_30hz
export MODEL_CONFIG=hfastwam_small
export USE_VJEPA21_VISUAL_ENCODER=0
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
export VIDEO_LATENT_CACHE_ENABLED=0
export RUN_NAME="${RUN_NAME:-interndata_a1_vae_dit_pretrain_30hz_16gpu_b${PER_GPU_BATCH_SIZE}_acc${GRADIENT_ACCUMULATION_STEPS}_gb${GLOBAL_BATCH_SIZE}}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-interndata-a1}"
export WANDB_GROUP="${WANDB_GROUP:-vae-dit-pretrain}"
export WANDB_MODE="${WANDB_MODE:-online}"

# VAE mode has no external visual encoder or representation-normalization stats.
unset VJEPA21_NORMALISE_STATS_PATH STANDARDISE_OUTPUT TEMPORAL_DOWNSAMPLE
unset CAUSAL_TUBELET_ENCODING FRAME_GAP FIXED_TARGET_ENCODER
unset VISUAL_ENCODER_FREEZE_BACKBONE VISUAL_ENCODER_ACTIVATION_CHECKPOINTING
unset TRAINABLE_COMPONENTS VISUAL_ENCODER_LR_MULTIPLIER

# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

echo "[interndata-a1-vae-dit] global_batch=${GLOBAL_BATCH_SIZE} world_size=${FASTWAM_EXPECTED_WORLD_SIZE} micro_batch=${PER_GPU_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" \
  "++model.language_pad_to_max_length=true" \
  "$@"
