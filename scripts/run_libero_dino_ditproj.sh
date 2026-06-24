#!/usr/bin/env bash
# =============================================================================
# FastWAM training on LIBERO with DINO visual encoder (DiT-side projection)
# Uses DeepSpeed ZeRO-1
# =============================================================================
#
# Usage:
#   # 8 GPUs, with pretrain checkpoint
#   NUM_GPUS=8 PRETRAIN_CKPT=/path/to/step_50000.pt bash scripts/run_libero_dino_ditproj.sh
#
#   # 8 GPUs, no pretrain (train from scratch — patchify/head random init)
#   NUM_GPUS=8 bash scripts/run_libero_dino_ditproj.sh
#
#   # Extra hydra overrides
#   NUM_GPUS=8 bash scripts/run_libero_dino_ditproj.sh learning_rate=5e-5 batch_size=8

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-null}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="./runs/libero_dino_ditproj/${RUN_ID}"

echo "============================================="
echo " FastWAM + DINO DiT-side projection on LIBERO"
echo " GPUs:           ${NUM_GPUS}"
echo " ZeRO stage:     1"
echo " Pretrain ckpt:  ${PRETRAIN_CKPT}"
echo " Output:         ${OUTPUT_DIR}"
echo "============================================="

bash "${PROJECT_ROOT}/scripts/train_zero1.sh" "${NUM_GPUS}" \
    task=libero_uncond_2cam224_1e-4 \
    data=libero_2cam \
    model=fastwam_dino_ditproj \
    model.pretrain_checkpoint="${PRETRAIN_CKPT}" \
    output_dir="${OUTPUT_DIR}" \
    wandb.name="libero_dino_ditproj" \
    "$@"
