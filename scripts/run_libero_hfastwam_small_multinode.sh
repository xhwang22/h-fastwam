#!/usr/bin/env bash
# Multi-node launcher: H-FastWAM SMALL (VAE-48d, from scratch) on LIBERO 2-camera.
#
# This is the multi-node version of scripts/run_libero_hfastwam_8card_small.sh.
# Topology is resolved from the Taiji platform env vars (HOST_NUM / HOST_GPU_NUM /
# INDEX / CHIEF_IP) exactly like run_libero_hfastwam_multinode.sh, then it
# delegates the actual torchrun launch to launch_torchrun_multinode.sh.
#
# Model: model=hfastwam_small
#   * video & action experts aligned to the language expert geometry
#     (16 heads * 128 head_dim = 2048 hidden, 28 layers), RANDOM-init
#     (no Wan / ActionDiT pretrained load),
#   * VAE-48d latent world model (no DINO / V-JEPA encoder),
#   * language expert FROZEN, language loss disabled (action+video only).
#
# This run uses the CODE FIX in hfastwam.py `_run_mot_two_experts_va` /
# interleaved no-language branch: the action stream now attends to *detached*
# video K/V (detach_kv_experts={"video"}), matching FastWAM's
# `action_loss_detach_video_expert=True`. This is the H-FastWAM side of the
# controlled comparison against run_libero_fastwam_small_multinode.sh.
#
# Usage (run on EVERY node, or via launch_libero_hfastwam_multinode.sh on chief):
#   WANDB=1 FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=small_vae_mn \
#     bash scripts/run_libero_hfastwam_small_multinode.sh
#
#   # Extra hydra overrides:
#   EXTRA="num_epochs=5" bash scripts/run_libero_hfastwam_small_multinode.sh

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

err()  { echo "[hfastwam-small-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[hfastwam-small-mn] $*"; }
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

# --- Resolve multi-node topology from Taiji platform env vars ---
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

# --- data / model base paths ---
TJ5_BASE="/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints"
export DIFFSYNTH_MODEL_BASE_PATH="${TJ5_BASE}/"
export MODEL="${MODEL:-hfastwam_small}"
export TASK="${TASK:-libero_uncond_2cam224_1e-4}"
export DATA="${DATA:-libero_2cam_interleaved}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/apdcephfs_tj5/share_302528826/shaunxhwang/data}"

export RUN_PREFIX="${RUN_PREFIX:-libero_hfastwam_small_mn}"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/libero_hfastwam}"
export FOREGROUND="${FOREGROUND:-0}"
export AUTO_KILL_EXISTING="${AUTO_KILL_EXISTING:-1}"

# DeepSpeed ZeRO-1 config (consumed by launch_torchrun_multinode.sh).
export DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/scripts/ds_configs/ds_zero1_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Match the 8-card SMALL run's per-GPU batch / grad-accum so the GLOBAL batch
# scales only with the number of nodes (8card single-node global = 8*1*16 = 128).
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export LOG_EVERY="${LOG_EVERY:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-0}"
export NUM_SEGMENTS="${NUM_SEGMENTS:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"

if [[ "${NUM_WORKERS}" == "0" && "${DATALOADER_TIMEOUT}" != "0" ]]; then
  info "Forcing DATALOADER_TIMEOUT=0 because NUM_WORKERS=0."
  export DATALOADER_TIMEOUT=0
fi

# --- wandb (default ON; key read from ~/.wandb_key, never logged) ---
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_key")"
fi
WANDB_OVERRIDES=( "wandb.enabled=false" )
if [[ "${WANDB:-1}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    err "wandb enabled but no API key (set ~/.wandb_key or export WANDB_API_KEY, or pass WANDB=0)."
  fi
  export WANDB_API_KEY
  WANDB_OVERRIDES=(
    "wandb.enabled=true"
    "wandb.project=${WANDB_PROJECT:-fast-wam}"
    "wandb.mode=${WANDB_MODE:-online}"
  )
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_OVERRIDES+=( "wandb.workspace=${WANDB_ENTITY}" )
  [[ -n "${WANDB_GROUP:-}" ]] && WANDB_OVERRIDES+=( "wandb.group=${WANDB_GROUP}" )
fi

if [[ "${AUTO_KILL_EXISTING}" == "1" ]]; then
  kill_existing_training
fi

# --- Hydra overrides: identical to run_libero_hfastwam_8card_small.sh ---
EXTRA_BASE=(
  "batch_size=${BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "log_every=${LOG_EVERY}"
  "num_workers=${NUM_WORKERS}"
  "dataloader_timeout=${DATALOADER_TIMEOUT}"
  "num_epochs=${NUM_EPOCHS}"
  "max_steps=null"
  "data.train.num_segments=${NUM_SEGMENTS}"
  "data.val.num_segments=${NUM_SEGMENTS}"
  "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
  "data.val.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
  "model.knowledge_insulation=false"
  "model.freeze_language_expert=true"
  "model.freeze_video_expert=false"
  "model.freeze_action_expert=false"
  "model.loss_config.lambda_language=0.0"
  "${WANDB_OVERRIDES[@]}"
)

if [[ -n "${EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_USER=( ${EXTRA} )
  EXTRA_BASE+=( "${EXTRA_USER[@]}" )
fi

printf -v EXTRA_JOINED '%s ' "${EXTRA_BASE[@]}"
export EXTRA="${EXTRA_JOINED% }"

info "model=${MODEL} data=${DATA} task=${TASK}"
info "global batch = NNODES * NPROC * batch_size * grad_accum = ${NNODES} * ${NPROC_PER_NODE} * ${BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS}"
info "RUN_NAME=${RUN_NAME}"

exec bash "${SCRIPT_DIR}/launch_torchrun_multinode.sh"
