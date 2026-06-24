#!/usr/bin/env bash
# =============================================================================
# End-to-end: pretrain on OpenVid → finetune on LIBERO
# =============================================================================
#
# NOTE: DINO stats computation is no longer needed. The model config
# (configs/model/pretrain_dino_ditproj.yaml) now uses
#   standardise_output=false  (LDA-1B style)
# which means the encoder relies on DINO's internal final LayerNorm and does
# NOT apply any extra per-channel standardisation.  `normalise_stats_path`
# is therefore unused and is no longer passed to the trainer.
#
# Usage:
#   NUM_GPUS=8 bash scripts/run_pretrain_dino_ditproj_full.sh
#
#   # Skip pretrain (already have pretrain ckpt)
#   NUM_GPUS=8 PRETRAIN_CKPT=/path/to/step_50000.pt bash scripts/run_pretrain_dino_ditproj_full.sh
#
#   # Extra hydra overrides for pretrain/finetune
#   NUM_GPUS=8 bash scripts/run_pretrain_dino_ditproj_full.sh --pretrain-args "max_steps=80000" --finetune-args "learning_rate=5e-5"

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29501}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Data paths
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/OpenVid_Data}"
CSV_NAME="${CSV_NAME:-data/train/OpenVid-1M.csv}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"  # empty = run pretrain; set to skip

# Parse --pretrain-args and --finetune-args
PRETRAIN_EXTRA_ARGS=()
FINETUNE_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrain-args)
            shift; IFS=' ' read -ra PRETRAIN_EXTRA_ARGS <<< "$1"; shift ;;
        --finetune-args)
            shift; IFS=' ' read -ra FINETUNE_EXTRA_ARGS <<< "$1"; shift ;;
        *)
            # Default extra args go to both
            PRETRAIN_EXTRA_ARGS+=("$1")
            FINETUNE_EXTRA_ARGS+=("$1")
            shift ;;
    esac
done

echo "============================================="
echo " DINO DiT-side Projection — Full Pipeline"
echo " Step 1: Pretrain on OpenVid-1M"
echo " Step 2: Finetune on LIBERO"
echo "============================================="
echo " GPUs:            ${NUM_GPUS}"
echo " Pretrain ckpt:   ${PRETRAIN_CKPT:-<will be produced by Step 1>}"
echo "============================================="

# DeepSpeed ZeRO-1 (used for both pretrain and finetune)
_launch() {
    # Usage: _launch <script.py> [hydra args...]
    local script="$1"; shift
    bash "${PROJECT_ROOT}/scripts/train_zero1.sh" "${NUM_GPUS}" "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Pretrain on OpenVid-1M
# ─────────────────────────────────────────────────────────────────────────────
if [ -n "${PRETRAIN_CKPT}" ] && [ -f "${PRETRAIN_CKPT}" ]; then
    echo "[Step 1] SKIP — Using existing pretrain checkpoint: ${PRETRAIN_CKPT}"
else
    echo "[Step 1] Pretraining on OpenVid-1M with DINO DiT-side projection..."

    PRETRAIN_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
    PRETRAIN_OUTPUT_DIR="./runs/pretrain_dino_ditproj/${PRETRAIN_RUN_ID}"

    _launch "${PROJECT_ROOT}/scripts/pretrain.py" \
        task=pretrain_dino_openvid \
        data=openvid \
        model=pretrain_dino_ditproj \
        "output_dir=${PRETRAIN_OUTPUT_DIR}" \
        "wandb.name=pretrain_dino_ditproj" \
        "${PRETRAIN_EXTRA_ARGS[@]}"

    # Find the latest checkpoint
    PRETRAIN_CKPT="$(ls -t "${PRETRAIN_OUTPUT_DIR}/checkpoints/weights/"*.pt 2>/dev/null | head -1 || true)"
    if [ -z "${PRETRAIN_CKPT}" ]; then
        echo "[ERROR] No checkpoint found in ${PRETRAIN_OUTPUT_DIR}/checkpoints/weights/"
        exit 1
    fi
    echo "[Step 1] Done → ${PRETRAIN_CKPT}"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Finetune on LIBERO
# ─────────────────────────────────────────────────────────────────────────────
echo "[Step 2] Finetuning on LIBERO with pretrain checkpoint..."

FINETUNE_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
FINETUNE_OUTPUT_DIR="./runs/libero_dino_ditproj/${FINETUNE_RUN_ID}"

_launch "${PROJECT_ROOT}/scripts/train.py" \
    task=libero_uncond_2cam224_1e-4 \
    data=libero_2cam \
    model=fastwam_dino_ditproj \
    "model.pretrain_checkpoint=${PRETRAIN_CKPT}" \
    "output_dir=${FINETUNE_OUTPUT_DIR}" \
    "wandb.name=libero_dino_ditproj" \
    "${FINETUNE_EXTRA_ARGS[@]}"

echo ""
echo "============================================="
echo " Pipeline complete!"
echo " Pretrain:  ${PRETRAIN_CKPT}"
echo " Finetune:  ${FINETUNE_OUTPUT_DIR}"
echo "============================================="
