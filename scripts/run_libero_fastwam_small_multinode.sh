#!/usr/bin/env bash
# Multi-node launcher: FastWAM CONTROL run, aligned to H-FastWAM SMALL.
#
# This is the FastWAM (2-expert: video + action) baseline for the controlled
# comparison against scripts/run_libero_hfastwam_small_multinode.sh. Topology is
# resolved from the Taiji platform env vars (HOST_NUM / HOST_GPU_NUM / INDEX /
# CHIEF_IP) and the actual torchrun launch is delegated to
# launch_torchrun_multinode.sh — identical plumbing to the H-FastWAM script so
# the only differences are the model architecture itself.
#
# ALIGNMENT to hfastwam_small (so "both from scratch" is truly apples-to-apples):
#   * action & video DiT geometry overridden to 2048 hidden / 16 heads / 28
#     layers (FastWAM default is 1024/24/30 action, 3072/24/30 video),
#   * RANDOM-init both experts: skip_dit_load_from_pretrain=true AND
#     action_dit_pretrained_path=null (FastWAM default loads a pretrained
#     ActionDiT — disabled here),
#   * VAE-48d latent world model (FastWAM default; no visual_encoder),
#   * action_loss_detach_video_expert=true (FastWAM default; stated explicitly).
#     This matches the H-FastWAM code fix where the action stream attends to
#     detached video K/V.
#   * SAME task config (lr=1e-4 cosine, wd=1e-2, AdamW), SAME per-GPU batch and
#     grad-accum, SAME LIBERO datasets.
#
# NOTE on data: FastWAM uses data=libero_2cam (RobotVideoDataset), NOT the
# interleaved variant H-FastWAM uses — FastWAM's training_loss() does not parse
# `segments`. With num_segments=1 the interleaved dataset is equivalent, and both
# use the same FastWAMProcessor / proprio / action dims, so the data content is
# aligned; only the sample packaging differs.
#
# Usage (run on EVERY node, or via a chief launcher):
#   WANDB=1 FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=fastwam_small_vae_mn \
#     bash scripts/run_libero_fastwam_small_multinode.sh
#
#   # Extra hydra overrides:
#   EXTRA="num_epochs=5" bash scripts/run_libero_fastwam_small_multinode.sh

set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
  echo "activate fastwam conda environment"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[fastwam-small-mn] ERROR: $*" >&2; exit 1; }
info() { echo "[fastwam-small-mn] $*"; }
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
CKPT_BASE="${REPO_ROOT}/checkpoints"
export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_BASE}/"
export MODEL="${MODEL:-fastwam}"
export TASK="${TASK:-libero_uncond_2cam224_1e-4}"
# FastWAM uses the non-interleaved RobotVideoDataset (training_loss has no
# segment handling). Same underlying LIBERO data / processor as H-FastWAM.
export DATA="${DATA:-libero_2cam}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data}"

export RUN_PREFIX="${RUN_PREFIX:-libero_fastwam_small_mn}"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/libero_fastwam}"
export FOREGROUND="${FOREGROUND:-0}"
export AUTO_KILL_EXISTING="${AUTO_KILL_EXISTING:-1}"

export DS_CONFIG="${DS_CONFIG:-${REPO_ROOT}/scripts/ds_configs/ds_zero1_config.json}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Same per-GPU batch / grad-accum as the H-FastWAM SMALL run.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export LOG_EVERY="${LOG_EVERY:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-0}"
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

# --- Hydra overrides: align FastWAM to hfastwam_small ---
EXTRA_BASE=(
  "batch_size=${BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "log_every=${LOG_EVERY}"
  "num_workers=${NUM_WORKERS}"
  "dataloader_timeout=${DATALOADER_TIMEOUT}"
  "num_epochs=${NUM_EPOCHS}"
  "max_steps=null"
  "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
  "data.val.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
  # --- random-init both experts (true "from scratch") ---
  "model.skip_dit_load_from_pretrain=true"
  "model.skip_video_dit_load_from_pretrain=true"
  "model.action_dit_pretrained_path=null"
  # --- VAE-48d latent world model (no visual encoder) ---
  "model.visual_encoder=null"
  # --- align action DiT geometry to hfastwam_small (2048 / 16 / 28) ---
  "model.action_dit_config.hidden_dim=2048"
  "model.action_dit_config.ffn_dim=8192"
  "model.action_dit_config.num_heads=16"
  "model.action_dit_config.attn_head_dim=128"
  "model.action_dit_config.num_layers=28"
  # --- align video DiT geometry to hfastwam_small (2048 / 16 / 28) ---
  "model.video_dit_config.hidden_dim=2048"
  "model.video_dit_config.ffn_dim=8192"
  "model.video_dit_config.num_heads=16"
  "model.video_dit_config.attn_head_dim=128"
  "model.video_dit_config.num_layers=28"
  # --- detach video expert from action loss (matches H-FastWAM code fix) ---
  "model.loss.action_loss_detach_video_expert=true"
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
info "action/video DiT aligned to 2048/16/28, random-init, detach_video_for_action=true"
info "global batch = NNODES * NPROC * batch_size * grad_accum = ${NNODES} * ${NPROC_PER_NODE} * ${BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS}"
info "RUN_NAME=${RUN_NAME}"

exec bash "${SCRIPT_DIR}/launch_torchrun_multinode.sh"
