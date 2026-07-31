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

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- 8 GPUs / torchrun topology ---
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
# shellcheck source=_auto_resume.sh
source "${SCRIPT_DIR}/_auto_resume.sh"
_multinode_dispatch "${BASH_SOURCE[0]}"

# --- DeepSpeed ZeRO-2 is ON via the accelerate config; clear stale overrides. ---
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset DS_CONFIG
# torch ZeroRedundancyOptimizer must stay OFF on the DeepSpeed path.
unset FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER
ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"

# --- corporate HTTP proxy (fallback; weights are pre-downloaded to shared FS) ---
# Only affects HTTP(S); NCCL/GLOO/torchrun use raw TCP (cluster IPs in no_proxy).
export no_proxy="${no_proxy:-.woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com,.myqcloud.com,.tencentcos.cn}"
export NO_PROXY="${no_proxy}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export ftp_proxy="${ftp_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"

# --- HuggingFace token (read from ~/.hf_token; override via HF_TOKEN env) ---
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# --- offline / fast init ---
# HF_HOME points at the in-repo (shared-FS) cache holding pre-downloaded model
# weights; datasets cache stays on node-local disk to avoid cephfs filelock
# races across ranks.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
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
# Pin the bootstrap/rendezvous NIC too: this node has 9 bond* interfaces, and
# NCCL's bootstrap (separate from data-plane SOCKET_IFNAME) can otherwise pick
# the wrong one and fail the first cross-node handshake on multi-node runs.
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-bond1}"
# Debug knobs (default quiet). For multi-node handshake issues run with
# NCCL_DEBUG=INFO to see which rank/interface fails.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# --- lightweight per-step timing on by default ---
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-5}"

# --- data / model paths (match the H-FastWAM small script) ---
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-data}"

# Effective (global) batch stays FIXED regardless of GPU count:
#   global_batch = world_size * batch_size * grad_accum,  world_size = nproc * nnodes
# We pin GLOBAL_BATCH_SIZE and derive the per-GPU batch_size from the ACTUAL world
# size, so scaling from 8 → 32 GPUs keeps the same effective batch (e.g. 128/32/1=4).
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
_BATCH_DENOM=$(( WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))
if (( GLOBAL_BATCH_SIZE % _BATCH_DENOM != 0 )); then
  echo "[8card-small-fastwam] ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by world_size*grad_accum=${_BATCH_DENOM} (nproc=${NPROC_PER_NODE} nnodes=${NNODES} grad_accum=${GRADIENT_ACCUMULATION_STEPS})." >&2
  exit 1
fi
BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / _BATCH_DENOM ))

RUN_NAME="${RUN_NAME:-libero_fastwam_8card_small_ds_0702}"
LOG_DIR="${REPO_ROOT}/runs/libero_fastwam/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log.rank${NODE_RANK}"

# --- auto-resume (preemptible / tidal resource) ---
# Frequent checkpointing + keep only the last few; on restart, auto-resume from
# the newest valid checkpoint. Override save freq via SAVE_EVERY, retention via
# FASTWAM_KEEP_LAST_CKPT, force scratch via FRESH=1, explicit via RESUME=<path>.
export FASTWAM_KEEP_LAST_CKPT="${FASTWAM_KEEP_LAST_CKPT:-3}"
SAVE_EVERY="${SAVE_EVERY:-200}"
_compute_resume_override "${LOG_DIR}"

# --- wandb (default ON) ---
# API key is read from ~/.wandb_key (chmod 600, NOT in the repo / git / logs).
# Disable logging entirely with WANDB=0. Override project/group/mode via env.
WANDB_API_KEY="${WANDB_API_KEY:-}"
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
      data=libero_2cam
      model=fastwam
      output_dir="${LOG_DIR}"
      "${WANDB_OVERRIDES[@]}"
      batch_size="${BATCH_SIZE}"
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every=1
      num_workers="${NUM_WORKERS:-8}"
      dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR:-4}"
      dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS:-true}"
      dataloader_timeout=0
      "data.train.dataset_dirs=[${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_object_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,${LIBERO_DATA_ROOT}/libero_mujoco3.3.2/libero_10_no_noops_lerobot]"
      num_epochs=10
      max_steps=null
      save_every="${SAVE_EVERY}"
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
      "${RESUME_OVERRIDES[@]}"
)

echo "[8card-small-fastwam] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} NO_CKPT=${NO_CKPT:-0} PROFILE_STEPS=${FASTWAM_PROFILE_STEPS}"
echo "[8card-small-fastwam] DeepSpeed ZeRO-2 via ${ACCEL_CONFIG} (global_batch=${GLOBAL_BATCH_SIZE} = world_size ${WORLD_SIZE} * batch_size ${BATCH_SIZE} * grad_accum ${GRADIENT_ACCUMULATION_STEPS})"
echo "[8card-small-fastwam] model=fastwam aligned to hfastwam_small (2048/16/28 random-init, VAE-48d, detach_video_for_action=${DETACH_VIDEO:-false})"
echo "[8card-small-fastwam] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[8card-small-fastwam] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
