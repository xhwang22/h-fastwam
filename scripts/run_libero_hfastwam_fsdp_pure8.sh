#!/usr/bin/env bash
set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_tj5/share_302528826/shaunxhwang/miniconda3/bin/activate"
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
MASTER_PORT="${MASTER_PORT:-29512}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
RUN_NAME="${RUN_NAME:-libero_hfastwam_fsdp_pure8}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_fsdp_pure_$$"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

export DIFFSYNTH_MODEL_BASE_PATH="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/apdcephfs_tj5/share_302528826/shaunxhwang/data}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${REPO_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
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
    scripts/train_fsdp_pure.py
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
      num_epochs=3
      max_steps=null
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
)

if [[ -n "${EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[fsdp-pure8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[fsdp-pure8] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
