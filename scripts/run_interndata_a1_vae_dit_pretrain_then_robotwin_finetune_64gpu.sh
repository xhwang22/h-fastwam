#!/usr/bin/env bash
# Run InternData VAE+DiT pretraining, convert its checkpoint, then fine-tune on RoboTwin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FASTWAM_EXPECTED_WORLD_SIZE="${FASTWAM_EXPECTED_WORLD_SIZE:-64}"
if (( FASTWAM_EXPECTED_WORLD_SIZE != 64 )); then
  echo "ERROR: this pipeline requires 8 nodes x 8 GPUs, got world size ${FASTWAM_EXPECTED_WORLD_SIZE}." >&2
  exit 2
fi

PIPELINE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
PIPELINE_LOG_ROOT="${LOG_ROOT:-/efs/shaunxhwang}"
PRETRAIN_PER_GPU_BATCH_SIZE="${PRETRAIN_PER_GPU_BATCH_SIZE:-24}"
PRETRAIN_GRADIENT_ACCUMULATION_STEPS="${PRETRAIN_GRADIENT_ACCUMULATION_STEPS:-2}"
PRETRAIN_GLOBAL_BATCH_SIZE=$(( \
  FASTWAM_EXPECTED_WORLD_SIZE \
  * PRETRAIN_PER_GPU_BATCH_SIZE \
  * PRETRAIN_GRADIENT_ACCUMULATION_STEPS \
))
PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-interndata_a1_vae_dit_pretrain_30hz_8x8_b${PRETRAIN_PER_GPU_BATCH_SIZE}_gb${PRETRAIN_GLOBAL_BATCH_SIZE}}"
PRETRAIN_LOG_DIR="${PRETRAIN_LOG_DIR:-${PIPELINE_LOG_ROOT}/${PRETRAIN_RUN_NAME}}"
PRETRAIN_DONE_MARKER="${PRETRAIN_LOG_DIR}/pipeline_pretrain.done"

FINETUNE_PER_GPU_BATCH_SIZE="${FINETUNE_PER_GPU_BATCH_SIZE:-24}"
FINETUNE_GRADIENT_ACCUMULATION_STEPS="${FINETUNE_GRADIENT_ACCUMULATION_STEPS:-1}"
FINETUNE_GLOBAL_BATCH_SIZE=$(( \
  FASTWAM_EXPECTED_WORLD_SIZE \
  * FINETUNE_PER_GPU_BATCH_SIZE \
  * FINETUNE_GRADIENT_ACCUMULATION_STEPS \
))
FINETUNE_RUN_NAME="${FINETUNE_RUN_NAME:-robotwin_vae_dit_finetune_interndata_64gpu_b${FINETUNE_PER_GPU_BATCH_SIZE}_gb${FINETUNE_GLOBAL_BATCH_SIZE}}"
FINETUNE_LOG_DIR="${FINETUNE_LOG_DIR:-${PIPELINE_LOG_ROOT}/${FINETUNE_RUN_NAME}}"

echo "[vae-transfer-pipeline] rank=${PIPELINE_RANK} pretrain=${PRETRAIN_LOG_DIR}"
echo "[vae-transfer-pipeline] rank=${PIPELINE_RANK} finetune=${FINETUNE_LOG_DIR}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
if [[ -z "${PRETRAIN_CKPT}" && -f "${PRETRAIN_DONE_MARKER}" ]]; then
  marker_checkpoint="$(cat "${PRETRAIN_DONE_MARKER}")"
  if [[ -f "${marker_checkpoint}" ]]; then
    PRETRAIN_CKPT="${marker_checkpoint}"
    echo "[vae-transfer-pipeline] reusing completed pretrain checkpoint: ${PRETRAIN_CKPT}"
  fi
fi

if [[ -z "${PRETRAIN_CKPT}" ]]; then
  echo "[vae-transfer-pipeline] starting InternData pretraining"
  RUN_NAME="${PRETRAIN_RUN_NAME}" \
  LOG_DIR="${PRETRAIN_LOG_DIR}" \
  PER_GPU_BATCH_SIZE="${PRETRAIN_PER_GPU_BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${PRETRAIN_GRADIENT_ACCUMULATION_STEPS}" \
  GLOBAL_BATCH_SIZE="${PRETRAIN_GLOBAL_BATCH_SIZE}" \
    bash "${SCRIPT_DIR}/run_interndata_a1_vae_dit_pretrain_64gpu_b48_cudnn_overlap_efa.sh"

  PRETRAIN_CKPT="$(
    find "${PRETRAIN_LOG_DIR}/checkpoints/weights" \
      -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -1
  )"
  if [[ -z "${PRETRAIN_CKPT}" || ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "ERROR: pretraining finished without a checkpoint under ${PRETRAIN_LOG_DIR}." >&2
    exit 1
  fi
  printf '%s\n' "${PRETRAIN_CKPT}" > "${PRETRAIN_DONE_MARKER}.rank${PIPELINE_RANK}"
  if (( PIPELINE_RANK == 0 )); then
    mv "${PRETRAIN_DONE_MARKER}.rank${PIPELINE_RANK}" "${PRETRAIN_DONE_MARKER}"
  fi
elif [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "ERROR: PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2
  exit 1
fi
PRETRAIN_CKPT="$(realpath "${PRETRAIN_CKPT}")"

TRANSFER_DIR="${TRANSFER_DIR:-${PRETRAIN_LOG_DIR}/robotwin_transfer}"
TRANSFER_CKPT="${TRANSFER_CKPT:-${TRANSFER_DIR}/$(basename "${PRETRAIN_CKPT}" .pt)_robotwin.pt}"
mkdir -p "${TRANSFER_DIR}"
(
  flock -x 9
  regenerate=0
  if [[ ! -f "${TRANSFER_CKPT}" || "${TRANSFER_CKPT}" -ot "${PRETRAIN_CKPT}" ]]; then
    regenerate=1
  fi
  if (( regenerate )); then
    temporary="${TRANSFER_CKPT}.tmp.$$"
    rm -f "${temporary}"
    echo "[vae-transfer-pipeline] converting ${PRETRAIN_CKPT}"
    python scripts/prepare_interndata_checkpoint_for_robotwin.py \
      --input "${PRETRAIN_CKPT}" \
      --output "${temporary}"
    mv "${temporary}" "${TRANSFER_CKPT}"
  fi
) 9>"${TRANSFER_CKPT}.lock"

if [[ ! -f "${TRANSFER_CKPT}" ]]; then
  echo "ERROR: converted transfer checkpoint not found: ${TRANSFER_CKPT}" >&2
  exit 1
fi

echo "[vae-transfer-pipeline] starting RoboTwin fine-tuning"
FASTWAM_CHECKPOINT="${TRANSFER_CKPT}" \
RUN_NAME="${FINETUNE_RUN_NAME}" \
LOG_DIR="${FINETUNE_LOG_DIR}" \
PER_GPU_BATCH_SIZE="${FINETUNE_PER_GPU_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${FINETUNE_GRADIENT_ACCUMULATION_STEPS}" \
GLOBAL_BATCH_SIZE="${FINETUNE_GLOBAL_BATCH_SIZE}" \
NUM_EPOCHS="${FINETUNE_NUM_EPOCHS:-5}" \
MAX_STEPS="${FINETUNE_MAX_STEPS:-null}" \
  exec bash "${SCRIPT_DIR}/run_robotwin_vae_dit_finetune_interndata_64gpu_b24_cudnn_overlap_efa.sh" \
    "$@"
