#!/usr/bin/env bash
# 8-GPU DDP (NON-DeepSpeed) run for H-FastWAM — SMALL variant.
#
# Identical launch/data/training setup to run_libero_hfastwam_8card.sh, but uses
# model=hfastwam_small: the video & action experts are shrunk and aligned to the
# language expert's attention geometry (16 heads * 128 head_dim = 2048 hidden,
# 28 layers), random-initialised (no Wan / ActionDiT pretrained load). This:
#   * removes all MoT q/k/v/o projection adapters (they collapse to nn.Identity
#     once every expert shares the same num_heads*attn_head_dim), and
#   * cuts trainable params from ~6.7B to ~3.8B,
# bringing per-GPU memory from ~49G to well under 40G WITHOUT touching the
# optimizer / LR / global batch / grad-accum / ZeRO settings.
#
# Memory levers preserved from the 8-card run:
#   FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 (ZeRO-1 optimizer-state sharding),
#   gradient checkpointing ON, fused AdamW, bf16 grad comm hook.
#
# Usage:
#   FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=small_test \
#     bash scripts/run_libero_hfastwam_8card_small.sh
set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_tj5/share_302528826/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- 8 GPUs / torchrun topology ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# --- make sure DeepSpeed is OFF (accelerate -> DDP under torchrun) ---
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset DS_CONFIG

# --- offline / fast init ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_8card_small_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- NCCL (match the working multinode/ddp settings) ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

# --- lightweight per-step timing on by default ---
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

# --- data / model paths (match the singlecard script) ---
export DIFFSYNTH_MODEL_BASE_PATH="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/apdcephfs_tj5/share_302528826/shaunxhwang/data}"

# Global batch = nproc * batch_size * grad_accum = 8 * 1 * 16 = 128.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

RUN_NAME="${RUN_NAME:-libero_hfastwam_8card_small}"
LOG_DIR="${REPO_ROOT}/runs/libero_hfastwam/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log.rank0"

# --- wandb (default ON) ---
# API key is read from ~/.wandb_key (chmod 600, NOT in the repo / git / logs).
# Disable logging entirely with WANDB=0. Override project/group/mode via env.
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_key")"
fi
WANDB_OVERRIDES=("wandb.enabled=false")
if [[ "${WANDB:-1}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[8card-small] ERROR: wandb enabled but no API key (set ~/.wandb_key or export WANDB_API_KEY, or pass WANDB=0)." >&2
    exit 1
  fi
  export WANDB_API_KEY
  WANDB_OVERRIDES=(
    "wandb.enabled=true"
    "wandb.project=${WANDB_PROJECT:-fast-wam}"
    "wandb.name=${RUN_NAME}"
    "wandb.mode=${WANDB_MODE:-online}"
  )
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_OVERRIDES+=("wandb.workspace=${WANDB_ENTITY}")
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_OVERRIDES+=("wandb.group=${WANDB_GROUP}")
fi

# Optional: also disable gradient checkpointing.
CKPT_OVERRIDES=()
if [[ "${NO_CKPT:-0}" == "1" ]]; then
  CKPT_OVERRIDES=(
    "model.video_dit_config.use_gradient_checkpointing=false"
    "model.action_dit_config.use_gradient_checkpointing=false"
  )
fi

CMD=(
  torchrun
    --nnodes=1
    --node_rank=0
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    scripts/train.py
      task=libero_uncond_2cam224_1e-4
      data=libero_2cam_interleaved
      model=hfastwam_small
      output_dir="${LOG_DIR}"
      "${WANDB_OVERRIDES[@]}"
      batch_size=1
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every=1
      num_workers=0
      dataloader_timeout=0
      data.train.num_segments=1
      data.val.num_segments=1
      "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
      "data.val.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
      num_epochs=3
      max_steps=null
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
      "model.action_loss_detach_video_expert=${DETACH_VIDEO:-false}"
      "${CKPT_OVERRIDES[@]}"
)

echo "[8card-small] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} NO_CKPT=${NO_CKPT:-0} PROFILE_STEPS=${FASTWAM_PROFILE_STEPS}"
echo "[8card-small] model=hfastwam_small (video+action aligned to language: 16x128=2048, 28 layers, random-init)"
echo "[8card-small] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[8card-small] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
