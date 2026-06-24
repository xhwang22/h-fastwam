#!/usr/bin/env bash
# 8-GPU DDP (NON-DeepSpeed) run for FastWAM — CONTROL for hfastwam_small (VAE).
#
# Same launch / data / training plumbing as run_libero_hfastwam_8card_small.sh
# (single-node 8-GPU DDP, ZeRO-1 optimizer-state sharding, gradient checkpointing,
# fused AdamW, bf16 grad comm hook), so the DDP backend MATCHES the three
# H-FastWAM SMALL runs and the comparison is apples-to-apples.
#
# model=fastwam (2-expert: video + action) is overridden to align EXACTLY with
# hfastwam_small so "both from scratch" is a fair comparison:
#   * action & video DiT geometry -> 2048 hidden / 16 heads / 128 head_dim / 28
#     layers (FastWAM default is 1024/24/30 action, 3072/24/30 video),
#   * RANDOM-init both experts: skip_dit_load_from_pretrain=true AND
#     action_dit_pretrained_path=null (FastWAM default loads a pretrained
#     ActionDiT — disabled here),
#   * VAE-48d latent world model (no visual encoder),
#   * action_loss_detach_video_expert=true (matches the H-FastWAM detach fix),
#   * SAME task (lr=1e-4 cosine, wd=1e-2, AdamW), per-GPU batch=1, grad_accum=16,
#     SAME LIBERO datasets, 3 epochs.
#
# Data note: FastWAM uses data=libero_2cam (RobotVideoDataset). FastWAM's
# training_loss() does not parse interleaved `segments`; with num_segments=1 the
# data content is equivalent and uses the same FastWAMProcessor / proprio / action
# dims as the H-FastWAM runs.
#
# Usage:
#   WANDB=1 FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=small_vae_fastwam \
#     bash scripts/run_libero_fastwam_8card_small.sh
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
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_8card_small_fastwam_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- NCCL (match the working multinode/ddp settings) ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

# --- lightweight per-step timing on by default ---
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

# --- data / model paths (match the H-FastWAM small script) ---
export DIFFSYNTH_MODEL_BASE_PATH="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/apdcephfs_tj5/share_302528826/shaunxhwang/data}"

# Global batch = nproc * batch_size * grad_accum = 8 * 1 * 16 = 128.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

RUN_NAME="${RUN_NAME:-libero_fastwam_8card_small}"
LOG_DIR="${REPO_ROOT}/runs/libero_fastwam/${RUN_NAME}"
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
    echo "[8card-small-fastwam] ERROR: wandb enabled but no API key (set ~/.wandb_key or export WANDB_API_KEY, or pass WANDB=0)." >&2
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

# Gradient checkpointing: hfastwam_small hardcodes use_gradient_checkpointing=true.
# FastWAM wires it to ${model.mot_checkpoint_mixed_attn}, which the task config
# sets to false — so to MATCH H-FastWAM memory behaviour we force it true here.
# Set NO_CKPT=1 to disable on both experts.
CKPT_OVERRIDES=( "model.mot_checkpoint_mixed_attn=true" )
if [[ "${NO_CKPT:-0}" == "1" ]]; then
  CKPT_OVERRIDES=(
    "model.mot_checkpoint_mixed_attn=false"
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
      data=libero_2cam
      model=fastwam
      output_dir="${LOG_DIR}"
      "${WANDB_OVERRIDES[@]}"
      batch_size=1
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every=1
      num_workers=0
      dataloader_timeout=0
      "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
      num_epochs=3
      max_steps=null
      # --- random-init both experts (true "from scratch", matches hfastwam_small) ---
      model.skip_dit_load_from_pretrain=true
      model.skip_video_dit_load_from_pretrain=true
      model.action_dit_pretrained_path=null
      # --- VAE-48d latent world model (no visual encoder) ---
      model.visual_encoder=null
      # --- align action DiT geometry to hfastwam_small (2048 / 16 / 28) ---
      model.action_dit_config.hidden_dim=2048
      model.action_dit_config.ffn_dim=8192
      model.action_dit_config.num_heads=16
      model.action_dit_config.attn_head_dim=128
      model.action_dit_config.num_layers=28
      # --- align video DiT geometry to hfastwam_small (2048 / 16 / 28) ---
      model.video_dit_config.hidden_dim=2048
      model.video_dit_config.ffn_dim=8192
      model.video_dit_config.num_heads=16
      model.video_dit_config.attn_head_dim=128
      model.video_dit_config.num_layers=28
      # --- detach video expert from action loss (matches the H-FastWAM fix) ---
      # Controlled by DETACH_VIDEO env (default true). Set DETACH_VIDEO=false to
      # let action-loss gradients shape the video expert's visual features.
      "model.loss.action_loss_detach_video_expert=${DETACH_VIDEO:-false}"
      "${CKPT_OVERRIDES[@]}"
)

echo "[8card-small-fastwam] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} NO_CKPT=${NO_CKPT:-0} PROFILE_STEPS=${FASTWAM_PROFILE_STEPS}"
echo "[8card-small-fastwam] model=fastwam aligned to hfastwam_small (2048/16/28 random-init, VAE-48d, detach_video_for_action=${DETACH_VIDEO:-false})"
echo "[8card-small-fastwam] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[8card-small-fastwam] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
