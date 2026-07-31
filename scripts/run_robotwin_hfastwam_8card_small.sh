#!/usr/bin/env bash
# 8-GPU DDP (NON-DeepSpeed) run for H-FastWAM — SMALL variant (RoboTwin 2.0).
#
# Identical launch/data/training setup to run_libero_hfastwam_8card.sh, but uses
# model=hfastwam_small: the video & action experts are shrunk and aligned to the
# language expert's attention geometry (16 heads * 128 head_dim = 2048 hidden,
# 28 layers), random-initialised (no Wan / ActionDiT pretrained load). This:
#   * removes all MoT q/k/v/o projection adapters (they collapse to nn.Identity
#     once every expert shares the same num_heads*attn_head_dim), and
#   * cuts trainable params from ~6.7B to ~3.8B,
# bringing per-GPU memory from ~49G to well under 40G WITHOUT touching the
# optimizer / LR / global batch / grad-accum / ZeRO settings.
#
# Memory levers preserved from the 8-card run:
#   FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 (ZeRO-1 optimizer-state sharding),
#   gradient checkpointing ON, fused AdamW, bf16 grad comm hook.
#
# Usage:
#   FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=small_test \
#     bash scripts/run_libero_hfastwam_8card_small.sh
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
# HF_HOME points at the in-repo (shared-FS) cache holding pre-downloaded
# Qwen3-VL-2B weights, so language_local_files_only resolves offline per node.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
# Keep the datasets cache on NODE-LOCAL disk: the parquet builder uses filelock +
# .incomplete temp dirs that race/corrupt when 16 ranks share one cephfs dir.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_robotwin_8card_small_$$"
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

# --- data / model paths (match the singlecard script) ---
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"

# Effective (global) batch stays FIXED regardless of GPU count:
#   global_batch = world_size * batch_size * grad_accum,  world_size = nproc * nnodes
# We pin GLOBAL_BATCH_SIZE and derive the per-GPU batch_size from the ACTUAL world
# size, so scaling from 8 → 32 GPUs keeps the same effective batch (e.g. 128/32/1=4).
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
_BATCH_DENOM=$(( WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS ))
if (( GLOBAL_BATCH_SIZE % _BATCH_DENOM != 0 )); then
  echo "[robotwin-small] ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by world_size*grad_accum=${_BATCH_DENOM} (nproc=${NPROC_PER_NODE} nnodes=${NNODES} grad_accum=${GRADIENT_ACCUMULATION_STEPS})." >&2
  exit 1
fi
BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / _BATCH_DENOM ))

RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_ds}"
LOG_DIR="${REPO_ROOT}/runs/robotwin_hfastwam/${RUN_NAME}"
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
    echo "[robotwin-small] ERROR: wandb enabled but no API key (set ~/.wandb_key or export WANDB_API_KEY, or pass WANDB=0)." >&2
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
      task=robotwin_uncond_3cam_384_1e-4
      data=robotwin_interleaved
      model=hfastwam_small
      output_dir="${LOG_DIR}"
      "${WANDB_OVERRIDES[@]}"
      batch_size="${BATCH_SIZE}"
      gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
      log_every=1
      num_workers="${NUM_WORKERS:-8}"
      dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR:-4}"
      dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS:-true}"
      dataloader_timeout=0
      data.train.num_segments=1
      data.val.num_segments=1
      "data.train.dataset_dirs=[${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0]"
      "data.train.pretrained_norm_stats=${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json"
      "data.val.dataset_dirs=[${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0]"
      "data.val.pretrained_norm_stats=${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json"
      num_epochs=5
      max_steps=null
      save_every="${SAVE_EVERY}"
      model.knowledge_insulation=false
      model.freeze_language_expert=true
      model.freeze_video_expert=false
      model.freeze_action_expert=false
      model.loss_config.lambda_language=0.0
      "model.action_loss_detach_video_expert=${DETACH_VIDEO:-false}"
      "${CKPT_OVERRIDES[@]}"
      "${RESUME_OVERRIDES[@]}"
)

echo "[robotwin-small] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} NO_CKPT=${NO_CKPT:-0} PROFILE_STEPS=${FASTWAM_PROFILE_STEPS}"
echo "[robotwin-small] DeepSpeed ZeRO-2 via ${ACCEL_CONFIG} (global_batch=${GLOBAL_BATCH_SIZE} = world_size ${WORLD_SIZE} * batch_size ${BATCH_SIZE} * grad_accum ${GRADIENT_ACCUMULATION_STEPS})"
echo "[robotwin-small] model=hfastwam_small (video+action aligned to language: 16x128=2048, 28 layers, random-init)"
echo "[robotwin-small] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[robotwin-small] log=${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
