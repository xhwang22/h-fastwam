#!/usr/bin/env bash
# Fine-tune transferred InternData VAE+DiT weights on RoboTwin with 64 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${FASTWAM_CHECKPOINT:?Set FASTWAM_CHECKPOINT to a converted RobotTwin-transfer checkpoint.}"
if [[ ! -f "${FASTWAM_CHECKPOINT}" ]]; then
  echo "ERROR: converted checkpoint not found: ${FASTWAM_CHECKPOINT}" >&2
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FASTWAM_EXPECTED_WORLD_SIZE="${FASTWAM_EXPECTED_WORLD_SIZE:-64}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-24}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( \
  FASTWAM_EXPECTED_WORLD_SIZE \
  * PER_GPU_BATCH_SIZE \
  * GRADIENT_ACCUMULATION_STEPS \
))}"
expected_global_batch=$(( \
  FASTWAM_EXPECTED_WORLD_SIZE \
  * PER_GPU_BATCH_SIZE \
  * GRADIENT_ACCUMULATION_STEPS \
))
if (( GLOBAL_BATCH_SIZE != expected_global_batch )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, expected ${expected_global_batch}=" >&2
  echo "  ${FASTWAM_EXPECTED_WORLD_SIZE} GPUs x batch ${PER_GPU_BATCH_SIZE} x accumulation ${GRADIENT_ACCUMULATION_STEPS}." >&2
  exit 2
fi

export TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
export MODEL_CONFIG=hfastwam_small
export USE_VJEPA21_VISUAL_ENCODER=0
export VIDEO_LATENT_CACHE_ENABLED=0
export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export MAX_STEPS="${MAX_STEPS:-null}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1

export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "ERROR: completed RoboTwin WebDataset not found:" >&2
  echo "  ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
  exit 2
fi

export LOG_ROOT="${LOG_ROOT:-/efs/shaunxhwang}"
export RUN_NAME="${RUN_NAME:-robotwin_vae_dit_finetune_interndata_64gpu_b${PER_GPU_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}}"
export LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-pretrain-transfer}"
export WANDB_GROUP="${WANDB_GROUP:-interndata-vae-dit-to-robotwin}"
export WANDB_MODE="${WANDB_MODE:-online}"

# VAE mode has no external visual-encoder normalization.
unset VJEPA21_NORMALISE_STATS_PATH STANDARDISE_OUTPUT TEMPORAL_DOWNSAMPLE
unset CAUSAL_TUBELET_ENCODING FRAME_GAP FIXED_TARGET_ENCODER
unset VISUAL_ENCODER_FREEZE_BACKBONE VISUAL_ENCODER_ACTIVATION_CHECKPOINTING
unset TRAINABLE_COMPONENTS VISUAL_ENCODER_LR_MULTIPLIER

echo "[robotwin-vae-dit-transfer] checkpoint=${FASTWAM_CHECKPOINT}"
echo "[robotwin-vae-dit-transfer] global_batch=${GLOBAL_BATCH_SIZE} world_size=${FASTWAM_EXPECTED_WORLD_SIZE} micro_batch=${PER_GPU_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[robotwin-vae-dit-transfer] output=${LOG_DIR}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" \
  "++model.language_pad_to_max_length=true" \
  "$@"
