#!/usr/bin/env bash
# 8-GPU FSDP (NO DeepSpeed) run for H-FastWAM.
#
# Why FSDP: root cause of the 17-22s backward is DeepSpeed ZeRO's per-parameter
# gradient reduction (O(N_tensors^2) CPU bookkeeping), which explodes once
# video+action experts (~6B params) are unfrozen. Plain DDP fixes the speed but
# OOMs (no optimizer-state sharding: ~48G AdamW fp32 per GPU). FSDP shards the
# optimizer state across GPUs (48/8 ~= 6G each) AND uses flat-bucket
# reduce-scatter (no per-param hook), so it is both memory-safe and fast.
#
# All three experts use the same DiTBlock class, so we transformer-wrap DiTBlock
# -> every transformer layer (language/video/action) becomes an FSDP unit.
#
# Usage:
#   bash scripts/run_libero_hfastwam_fsdp8.sh
#   FASTWAM_PROFILE_STEPS=5 bash scripts/run_libero_hfastwam_fsdp8.sh
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
MASTER_PORT="${MASTER_PORT:-29502}"

# --- DeepSpeed OFF; turn FSDP ON via accelerate env vars ---
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset DS_CONFIG

export ACCELERATE_USE_FSDP=true
# Default to FSDP1. trainer.py configures the plugin with FSDP1-style
# lambda_auto_wrap_policy + use_orig_params + ignored_modules, where
# ignored_modules reliably keeps the frozen language expert and the
# non-DiT pre/post Linear modules (e.g. video_expert.time_embedding) as
# plain replicated tensors. Under FSDP2 use_orig_params is obsolete and
# ignored_modules behaves differently, which previously caused the
# "mixed torch.Tensor and DTensor" addmm error. Override with FSDP_VERSION=2
# only if you have explicitly ported the wrap/ignore logic to FSDP2.
export FSDP_VERSION="${FSDP_VERSION:-1}"
# Full shard (== ZeRO-3 sharding of params+grads+optimizer). For fsdp v2 this is a bool.
export FSDP_RESHARD_AFTER_FORWARD="${FSDP_RESHARD_AFTER_FORWARD:-true}"
export FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT
export FSDP_OFFLOAD_PARAMS=false
# NOTE: auto_wrap_policy / use_orig_params are set programmatically in
# trainer.py (custom policy that wraps ONLY trainable DiTBlocks, leaving the
# frozen language expert unsharded to avoid the embedding DTensor error).
# Do NOT also set FSDP_AUTO_WRAP_POLICY / FSDP_TRANSFORMER_CLS_TO_WRAP here.
# DDP find_unused is irrelevant under FSDP; disable that code path.
export FASTWAM_DDP_FIND_UNUSED=0

# --- offline / fast init ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_fsdp_$$"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"

export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${REPO_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"

RUN_NAME="${RUN_NAME:-libero_hfastwam_fsdp8}"
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

echo "[fsdp8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} fsdp_v=${FSDP_VERSION}"
echo "[fsdp8] wrap=DiTBlock reshard_after_fwd=${FSDP_RESHARD_AFTER_FORWARD}"
echo "[fsdp8] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
