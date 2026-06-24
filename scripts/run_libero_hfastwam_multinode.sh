#!/usr/bin/env bash
# Multi-node launcher: hfastwam on LIBERO 2-camera.
#
# Matches the single-node command:
#   bash scripts/train_zero1.sh 8 data=libero_2cam model=hfastwam \
#     task=libero_uncond_2cam224_1e-4 \
#     model.visual_encoder_config=null \
#     model.visual_encoder=null \
#     model.video_dit_config.in_dim=48 \
#     model.video_dit_config.out_dim=48 \
#     model.freeze_language_expert=true \
#     model.action_dit_pretrained_path=/path/to/ActionDiT.pt \
#     model.freeze_video_expert=false
#
# Run this on every node of the Taiji job. Each node already has its own INDEX
# env var injected by the platform.
#
# Usage:
#   # Recommended: run once on chief to launch all nodes.
#   bash scripts/launch_libero_hfastwam_multinode.sh
#
#   # Or run this inner script manually on every node.
#   bash scripts/run_libero_hfastwam_multinode.sh
#   FOREGROUND=1 bash scripts/run_libero_hfastwam_multinode.sh
#   RUN_NAME=myrun bash scripts/run_libero_hfastwam_multinode.sh
#
#   # Extra hydra overrides:
#   EXTRA="trainer.max_steps=80000" bash scripts/run_libero_hfastwam_multinode.sh

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_tj5/share_302528826/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
  echo "activate fastwam conda environment"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[libero-hfastwam-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[libero-hfastwam-mn] $*"; }

is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

kill_existing_training() {
  local pattern pid cwd
  for pattern in "scripts/train.py" "torchrun"; do
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      [[ "${pid}" != "$$" ]] || continue
      [[ -d "/proc/${pid}" ]] || continue
      cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
      [[ "${cwd}" == "${REPO_ROOT}" ]] || continue
      info "Stopping stale process: pid=${pid} pattern=${pattern}"
      kill -TERM "${pid}" 2>/dev/null || true
    done < <(pgrep -f "${pattern}" 2>/dev/null || true)
  done
}

# Resolve multi-node topology from Taiji platform env vars, matching
# run_libero_vjepa2ac_predictor_multinode.sh.
export NNODES="${NNODES:-${HOST_NUM:-1}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${HOST_GPU_NUM:-8}}"
export NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

for v in NNODES NPROC_PER_NODE NODE_RANK MASTER_PORT; do
  is_integer "${!v}" || err "${v}=${!v} is not an integer."
done

info "Detected topology:"
info "  NNODES         = ${NNODES}"
info "  NODE_RANK      = ${NODE_RANK}"
info "  MASTER_ADDR    = ${MASTER_ADDR}"
info "  MASTER_PORT    = ${MASTER_PORT}"
info "  NPROC_PER_NODE = ${NPROC_PER_NODE}"
info "  TOTAL_GPUS     = $(( NNODES * NPROC_PER_NODE ))"

TJ5_BASE="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints"
ACTION_DIT_FILENAME="ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
LOCAL_ACTION_DIT_PRETRAINED_PATH="${REPO_ROOT}/checkpoints/${ACTION_DIT_FILENAME}"
TJ5_ACTION_DIT_PRETRAINED_PATH="${TJ5_BASE}/${ACTION_DIT_FILENAME}"
if [[ -z "${ACTION_DIT_PRETRAINED_PATH:-}" ]]; then
  if [[ -f "${LOCAL_ACTION_DIT_PRETRAINED_PATH}" ]]; then
    ACTION_DIT_PRETRAINED_PATH="${LOCAL_ACTION_DIT_PRETRAINED_PATH}"
  else
    ACTION_DIT_PRETRAINED_PATH="${TJ5_ACTION_DIT_PRETRAINED_PATH}"
  fi
fi

if [[ "${NODE_RANK}" == "0" && ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
  info "WARNING: ACTION_DIT_PRETRAINED_PATH not found on rank 0: ${ACTION_DIT_PRETRAINED_PATH}"
  info "         Override it with ACTION_DIT_PRETRAINED_PATH=/path/to/checkpoint.pt if needed."
fi

export DIFFSYNTH_MODEL_BASE_PATH="${TJ5_BASE}/"
export MODEL="${MODEL:-hfastwam}"
export TASK="${TASK:-libero_uncond_2cam224_1e-4}"
export DATA="${DATA:-libero_2cam_interleaved}"
export LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT:-/apdcephfs_tj5/share_302528826/shaunxhwang/data}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${LIBERO_SOURCE_ROOT}}"
export CACHE_LIBERO_LOCAL="${CACHE_LIBERO_LOCAL:-0}"
export LOCAL_LIBERO_DATA_ROOT="${LOCAL_LIBERO_DATA_ROOT:-/tmp/fastwam_data/libero}"
export LOCAL_CACHE_PARALLEL="${LOCAL_CACHE_PARALLEL:-4}"
export RUN_PREFIX="${RUN_PREFIX:-libero_hfastwam_mn}"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
export WANDB_NAME="${WANDB_NAME:-libero_hfastwam_mn}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/libero_hfastwam}"
export FOREGROUND="${FOREGROUND:-0}"
export AUTO_KILL_EXISTING="${AUTO_KILL_EXISTING:-1}"
export DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/scripts/ds_configs/ds_zero1_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export LOG_EVERY="${LOG_EVERY:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-0}"
export NUM_SEGMENTS="${NUM_SEGMENTS:-1}"

if [[ "${NUM_WORKERS}" == "0" && "${DATALOADER_TIMEOUT}" != "0" ]]; then
  info "Forcing DATALOADER_TIMEOUT=0 because NUM_WORKERS=0."
  export DATALOADER_TIMEOUT=0
fi

if [[ "${CACHE_LIBERO_LOCAL}" == "1" ]]; then
  info "Caching LIBERO data locally before training."
  info "  source=${LIBERO_SOURCE_ROOT}"
  info "  local=${LOCAL_LIBERO_DATA_ROOT}"
  LIBERO_SOURCE_ROOT="${LIBERO_SOURCE_ROOT}" \
  LOCAL_LIBERO_DATA_ROOT="${LOCAL_LIBERO_DATA_ROOT}" \
  LOCAL_CACHE_PARALLEL="${LOCAL_CACHE_PARALLEL}" \
    bash "${SCRIPT_DIR}/_local_cache_libero.sh"
  export LIBERO_DATA_ROOT="${LOCAL_LIBERO_DATA_ROOT}"
  info "Using local LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}"
fi

if [[ "${DATA}" == "libero_2cam_interleaved" ]]; then
  info "LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}"
  df -h "${LIBERO_DATA_ROOT}" 2>/dev/null || info "WARNING: df failed for LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}"
  for subset in \
    libero_spatial_no_noops_lerobot \
    libero_object_no_noops_lerobot \
    libero_goal_no_noops_lerobot \
    libero_10_no_noops_lerobot; do
    ds_dir="${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/${subset}"
    if [[ -d "${ds_dir}" ]]; then
      info "Found dataset dir: ${ds_dir}"
      find "${ds_dir}/meta" -maxdepth 1 -type f 2>/dev/null | sed "s#^#[libero-hfastwam-mn]   meta #"
    else
      info "WARNING: missing dataset dir: ${ds_dir}"
    fi
  done
fi

if [[ "${AUTO_KILL_EXISTING}" == "1" ]]; then
  kill_existing_training
fi

EXTRA_BASE=(
  "batch_size=${BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "log_every=${LOG_EVERY}"
  "num_workers=${NUM_WORKERS}"
  "dataloader_timeout=${DATALOADER_TIMEOUT}"
  "data.train.num_segments=${NUM_SEGMENTS}"
  "data.val.num_segments=${NUM_SEGMENTS}"
  "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
  "data.val.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
  "model.visual_encoder_config=null"
  "model.visual_encoder=null"
  "model.video_dit_config.in_dim=48"
  "model.video_dit_config.out_dim=48"
  "model.skip_dit_load_from_pretrain=false"
  "model.skip_video_dit_load_from_pretrain=false"
  "model.freeze_language_expert=true"
  "model.action_dit_pretrained_path=${ACTION_DIT_PRETRAINED_PATH}"
  "model.freeze_video_expert=false"
)

if [[ -n "${SEGMENT_STRIDE:-}" ]]; then
  EXTRA_BASE+=( "data.train.segment_stride=${SEGMENT_STRIDE}" )
fi

if [[ -n "${EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_USER=( ${EXTRA} )
  EXTRA_BASE+=( "${EXTRA_USER[@]}" )
fi

printf -v EXTRA_JOINED '%s ' "${EXTRA_BASE[@]}"
export EXTRA="${EXTRA_JOINED% }"

exec bash "${SCRIPT_DIR}/launch_torchrun_multinode.sh"
