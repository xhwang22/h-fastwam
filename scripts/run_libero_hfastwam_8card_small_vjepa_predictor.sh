#!/usr/bin/env bash
# 8-GPU DDP (NON-DeepSpeed) run for H-FastWAM — SMALL + JEPAPredictor (Exp 3).
#
# Experiment 3: replaces the Wan-style flow-matching video DiT with a
# JEPAPredictor — a deterministic next-frame latent predictor trained with an
# L1 regression loss in V-JEPA 2-AC encoder space (1408-dim). The action expert
# remains a flow-matching ActionDiT. The language expert (Qwen3-VL-2B) is frozen.
#
# Differences vs hfastwam_8card_small_vjepa.sh (Experiment 2):
#   model=hfastwam_small_vjepa_predictor
#     → video_expert_type=jepa_predictor: the factory builds JEPAPredictor
#       instead of a random-init WAN DiT.
#     → No flow-matching scheduler for the video expert (deterministic).
#     → Video loss: L1 (not MSE + weight).
#   All other settings (topology, batch, grad-accum, ZeRO, LR) are identical.
#
# Memory profile: identical to hfastwam_small_vjepa (~3.8B trainable).
# No scheduler overhead for video → slightly faster per-step.
#
# Usage:
#   FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=exp3_test \
#     bash scripts/run_libero_hfastwam_8card_small_vjepa_predictor.sh
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
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_8card_small_vjepa_predictor_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- NCCL (match the working multinode/ddp settings) ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

# --- lightweight per-step timing on by default ---
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

# --- data / model paths ---
export DIFFSYNTH_MODEL_BASE_PATH="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/apdcephfs_tj5/share_302528826/shaunxhwang/data}"

# Global batch = nproc * batch_size * grad_accum = 8 * 1 * 16 = 128.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

RUN_NAME="${RUN_NAME:-libero_hfastwam_8card_small_vjepa_predictor}"
LOG_DIR="${REPO_ROOT}/runs/libero_hfastwam/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log.rank0"

# --- wandb (default ON) ---
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_key")"
fi
WANDB_OVERRIDES=("wandb.enabled=false")
if [[ "${WANDB:-1}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[8card-small-vjepa-predictor] ERROR: wandb enabled but no API key (set ~/.wandb_key or export WANDB_API_KEY, or pass WANDB=0)." >&2
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

# Optional: disable gradient checkpointing.
CKPT_OVERRIDES=()
if [[ "${NO_CKPT:-0}" == "1" ]]; then
  CKPT_OVERRIDES=(
    "model.video_dit_config.use_gradient_checkpointing=false"
    "model.action_dit_config.use_gradient_checkpointing=false"
  )
fi

# V-JEPA latent standardisation toggle.
# Default `standardise_output=true` does PER-BATCH channel standardisation over
# (B,T,H,W) on the frozen V-JEPA latents. Set STANDARDISE_OUTPUT=false to feed
# RAW V-JEPA features to the JEPAPredictor (no batch standardise), removing the
# zero-mean unit-variance target that lets an L1 regressor collapse to the
# channel mean.
STANDARDISE_OVERRIDES=()
if [[ -n "${STANDARDISE_OUTPUT:-}" ]]; then
  STANDARDISE_OVERRIDES=(
    "model.visual_encoder_config.standardise_output=${STANDARDISE_OUTPUT}"
  )
fi

# Predict-horizon (frame gap) control for the JEPA next-frame target.
# Default (action_video_freq_ratio=4, temporal_downsample=4) makes adjacent
# latent frames ~16 real control steps apart — far too large for an L1
# next-frame regressor (target becomes multi-modal → collapses to channel mean).
#
# IMPORTANT constraint: V-JEPA2-AC has temporal_patch=2, so it can only produce
# floor(T_video/2) REAL latent frames. Asking for more latent frames than that
# triggers trilinear UP-sampling (fake interpolated frames) in encode(), which
# pollutes the target. So tds must satisfy: T_lat = (T_video-1)//tds + 1 <= T_video//2.
#
# FRAME_GAP=3 → action_video_freq_ratio=1 (video keeps all 33 frames →
# V-JEPA real T_p=16) + temporal_downsample=3 → T_lat=11 (<=16, NO interpolation).
# Adjacent latent frames are ~3 real steps apart → near-deterministic next-frame
# target so L1 regression is well-posed. 11 real supervised latent frames/clip.
# Leave unset to keep the original (gap ~16) config.
GAP_OVERRIDES=()
if [[ "${FRAME_GAP:-}" == "3" ]]; then
  GAP_OVERRIDES=(
    "data.train.action_video_freq_ratio=1"
    "data.val.action_video_freq_ratio=1"
    "model.visual_encoder_config.temporal_downsample=3"
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
      model=hfastwam_small_vjepa_predictor
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
      "${STANDARDISE_OVERRIDES[@]}"
      "${GAP_OVERRIDES[@]}"
)

echo "[8card-small-vjepa-predictor] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} NO_CKPT=${NO_CKPT:-0} PROFILE_STEPS=${FASTWAM_PROFILE_STEPS}"
echo "[8card-small-vjepa-predictor] model=hfastwam_small_vjepa_predictor (JEPAPredictor 2048/16/28 random-init + frozen V-JEPA2-AC ViT-g, in/out_dim=1408, L1 loss)"
echo "[8card-small-vjepa-predictor] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[8card-small-vjepa-predictor] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
