#!/usr/bin/env bash
# 8-GPU DeepSpeed ZeRO-2 run for H-FastWAM (SMALL model).
#
# Same task/data/freeze/global-batch as run_libero_hfastwam_8card_small.sh, but
# launches via `accelerate launch` with DeepSpeed ZeRO-2 instead of plain DDP +
# torch ZeroRedundancyOptimizer.
#
# Why DeepSpeed ZeRO-2 here:
#   * Stage-2 shards BOTH optimizer state AND gradients across the 8 ranks
#     (torch ZeRO-1 only shards optimizer state), so per-GPU memory is lower.
#   * In bf16 mode DeepSpeed maintains fp32 MASTER weights + fp32 optimizer
#     state internally (the bf16 params are just the compute copy). This is the
#     standard mixed-precision recipe and is numerically more robust than the
#     pure-bf16 torch-ZeRO path (no small-update swamping at lr=1e-4).
#
# Model config is UNCHANGED: model=hfastwam_small (video & action experts
# aligned to the language expert: 16 heads * 128 head_dim = 2048, 28 layers,
# random-init; VAE-48d latent world model, no DINO).
#
# Requirements:
#   pip install deepspeed   # not installed in the base env yet
#
# Usage:
#   RUN_NAME=ds_zero2_small bash scripts/run_libero_hfastwam_8card_ds.sh
#   GRADIENT_ACCUMULATION_STEPS=8 bash scripts/run_libero_hfastwam_8card_ds.sh
#   NO_CKPT=1 bash scripts/run_libero_hfastwam_8card_ds.sh
set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- 8 GPUs topology ---
# Single-node: leave NODE_IP_LIST unset.
# Multi-node:  NODE_IP_LIST="ip0,ip1,..."  (first IP = rank-0/master)
#              NODE_RANK=<this node's index>  (set per node: 0, 1, 2, ...)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -n "${NODE_IP_LIST:-}" ]]; then
  IFS=',' read -ra _NODES <<< "${NODE_IP_LIST}"
  NNODES="${#_NODES[@]}"
  MASTER_ADDR="${MASTER_ADDR:-${_NODES[0]%%:*}}"
else
  NNODES=1
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
NODE_RANK="${NODE_RANK:-0}"
# shellcheck source=_multinode_ssh_dispatch.sh
source "${SCRIPT_DIR}/_multinode_ssh_dispatch.sh"
_multinode_dispatch "${BASH_SOURCE[0]}"

# --- DeepSpeed is ON via the accelerate config; clear any stale overrides. ---
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset DS_CONFIG
# torch ZeroRedundancyOptimizer must stay OFF on the DeepSpeed path (the trainer
# ignores it under DEEPSPEED, but make the intent explicit).
unset FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER

# --- offline / fast init ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_8card_ds_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- NCCL (match the working multinode/ddp settings) ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

# --- lightweight per-step timing on by default ---
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

# --- data / model paths (match the singlecard script) ---
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data}"

# Global batch = nproc * batch_size * grad_accum = 8 * 1 * 16 = 128.
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

RUN_NAME="${RUN_NAME:-libero_hfastwam_8card_ds}"
LOG_DIR="${REPO_ROOT}/runs/libero_hfastwam/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log.rank${NODE_RANK}"

ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"

# Optional: also disable gradient checkpointing.
CKPT_OVERRIDES=()
if [[ "${NO_CKPT:-0}" == "1" ]]; then
  CKPT_OVERRIDES=(
    "model.video_dit_config.use_gradient_checkpointing=false"
    "model.action_dit_config.use_gradient_checkpointing=false"
  )
fi

CMD=(
  accelerate launch
    --config_file "${ACCEL_CONFIG}"
    --num_machines "${NNODES}"
    --machine_rank "${NODE_RANK}"
    --main_process_ip "${MASTER_ADDR}"
    --main_process_port "${MASTER_PORT}"
    --num_processes "$(( NPROC_PER_NODE * NNODES ))"
    --deepspeed_multinode_launcher standard
    scripts/train.py
      task=libero_uncond_2cam224_1e-4
      data=libero_2cam_interleaved
      model=hfastwam_small
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
      num_epochs=10
      max_steps=null
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
      "${CKPT_OVERRIDES[@]}"
)

echo "[8card-ds] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} NO_CKPT=${NO_CKPT:-0} PROFILE_STEPS=${FASTWAM_PROFILE_STEPS}"
echo "[8card-ds] DeepSpeed ZeRO-2 (bf16 + fp32 master) via ${ACCEL_CONFIG}"
echo "[8card-ds] model=hfastwam_small (video+action aligned to language: 16x128=2048, 28 layers, random-init)"
echo "[8card-ds] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[8card-ds] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
