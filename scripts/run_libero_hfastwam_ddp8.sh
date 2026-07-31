#!/usr/bin/env bash
# 8-GPU DDP (NO DeepSpeed) run for H-FastWAM.
#
# Root cause confirmed: DeepSpeed ZeRO's per-parameter gradient reduction is
# O(N_trainable_tensors^2) on CPU, which explodes once video+action experts
# (~6B params, ~1763 tensors) are unfrozen -> 17-22s backward. Plain PyTorch
# DDP (flat bucketed all-reduce, overlapped with backward) avoids that path.
#
# NOTE: DDP does NOT shard optimizer state. Each GPU holds the full AdamW
# fp32 state (~48G for 6B trainable params) + model (~16G) + activations.
# Gradient checkpointing is kept ON (costs ~20%, saves a lot of activation
# memory). If this OOMs, switch to FSDP (shards optimizer state).
#
# Usage:
#   bash scripts/run_libero_hfastwam_ddp8.sh
#   FASTWAM_PROFILE_STEPS=5 bash scripts/run_libero_hfastwam_ddp8.sh
set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# --- DeepSpeed OFF: accelerate auto-selects DDP (MULTI_GPU) under torchrun ---
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset DS_CONFIG

# --- offline / fast init ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_ddp_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NCCL (match the multinode script's working settings)
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${REPO_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

# global batch via grad accum: keep per-step work small, accumulate to match
# the previous effective batch (8 gpus * 16 accum = 128). Here 8 gpus * 2 = 16
# micro-batches; raise GRADIENT_ACCUMULATION_STEPS if you want a bigger global batch.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

RUN_NAME="${RUN_NAME:-libero_hfastwam_ddp8}"
LOG_DIR="${REPO_ROOT}/runs/libero_hfastwam/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log.rank0"

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
      model=hfastwam
      output_dir="${LOG_DIR}"
      wandb.enabled=false
      batch_size=1
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every=1
      num_workers=0
      dataloader_timeout=0
      data.train.num_segments=1
      data.val.num_segments=1
      "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
      "data.val.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
      model.visual_encoder_config=null
      model.visual_encoder=null
      model.video_dit_config.in_dim=48
      model.video_dit_config.out_dim=48
      model.skip_dit_load_from_pretrain=false
      model.skip_video_dit_load_from_pretrain=false
      model.action_dit_pretrained_path="${ACTION_DIT_PRETRAINED_PATH}"
      num_epochs=10
      max_steps=null
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
)

echo "[ddp8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[ddp8] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
