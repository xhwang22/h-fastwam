#!/usr/bin/env bash
# LIBERO simulator evaluation launcher for the four SMALL training runs.
#
# Runs the multi-GPU parallel eval driver (experiments/libero/run_libero_parallel_test.sh)
# with the EXACT model/data/subtask overrides needed to align each eval with how
# that model was trained. Handles conda env, offline HF cache, proxy, and mesa.
#
# Usage:
#   bash scripts/run_libero_eval.sh <model>
#     <model> ∈ { hfastwam_small | dino | vjepa_predictor | fastwam | smoke }
#
#   smoke = single-GPU, 1 task, 1 trial sanity check (uses hfastwam_small).
#
# Env overrides (optional):
#   NUM_TRIALS (default 50), MAX_TASKS_PER_GPU (default 2),
#   CUDA_VISIBLE_DEVICES (default 0..7), TASK_LIST (default all_suites_10each.txt),
#   RUN_ID (default eval_<model>), SKIP_YUM=1 (skip mesa install).
#
# Multi-machine: give each node a distinct RUN_ID and its own TASK_LIST shard, e.g.
#   split -d -n l/4 --additional-suffix=.txt experiments/libero/task_lists/all_suites_10each.txt \
#     experiments/libero/task_lists/shard_
#   RUN_ID=eval_dino_node0 TASK_LIST=experiments/libero/task_lists/shard_00.txt bash scripts/run_libero_eval.sh dino
# ...or simply run one model per machine with the full task list.
set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: bash scripts/run_libero_eval.sh <hfastwam_small|dino|vjepa_predictor|fastwam|smoke>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- conda env: fastwam (holds transformers 5.12.1 for Qwen3-VL + libero via .pth) ---
CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
# shellcheck disable=SC1090
source "${CONDA_ACTIVATE}" fastwam

# --- system mesa for robosuite (libGL.so.1 / OSMesa); per-machine, not on shared FS ---
if [[ "${SKIP_YUM:-0}" != "1" ]]; then
  if ! python -c 'import ctypes; ctypes.CDLL("libGL.so.1")' >/dev/null 2>&1; then
    echo "[eval] installing mesa (libGL/OSMesa) via yum ..."
    yum install -y mesa-libGL mesa-libGL-devel mesa-libOSMesa mesa-libOSMesa-devel mesa-dri-drivers \
      || echo "[eval] WARN: yum mesa install failed; continuing (may already be present)."
  fi
fi

# --- offline HF cache + torch hub (shared FS), rendering, proxy ---
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints/"
export HF_HOME="${REPO_ROOT}/checkpoints/hf_cache"
export TORCH_HOME="${REPO_ROOT}/checkpoints/torch_hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export no_proxy="${no_proxy:-.woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.mqq.com,.qqinternal.com,.sc.oa.com,.teg.local,mirrors.tencent.com,csighub.tencentyun.com,.myqcloud.com,.tencentcos.cn}"
export NO_PROXY="${no_proxy}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export ftp_proxy="${ftp_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${http_proxy}" HTTPS_PROXY="${https_proxy}"

# --- per-model config: run dir, model name, aligned EXTRA_ARGS ---
CONFIG=libero_uncond_2cam224_1e-4
case "${MODEL}" in
  hfastwam_small|small)
    CKPT_DIR="runs/libero_hfastwam/libero_hfastwam_8card_small_ds"
    EXTRA_ARGS="model=hfastwam_small data=libero_2cam_interleaved model.visual_encoder=null model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null EVALUATION.disable_subtask_generation=true"
    ;;
  dino|hfastwam_small_dino)
    CKPT_DIR="runs/libero_hfastwam/libero_hfastwam_8card_small_dino_ds"
    EXTRA_ARGS="model=hfastwam_small_dino data=libero_2cam_interleaved model.visual_encoder=null model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null EVALUATION.disable_subtask_generation=true"
    ;;
  vjepa_predictor|vjepa|hfastwam_small_vjepa_predictor)
    CKPT_DIR="runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor_ds_0702"
    EXTRA_ARGS="model=hfastwam_small_vjepa_predictor data=libero_2cam_interleaved model.visual_encoder=null model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null EVALUATION.disable_subtask_generation=true"
    ;;
  fastwam)
    CKPT_DIR="runs/libero_fastwam/libero_fastwam_8card_small_ds_0702"
    # fastwam.infer_action has NO subtask param -> do NOT set disable_subtask_generation.
    # load_text_encoder=true: training used PRECOMPUTED context/context_mask (trained
    # with load_text_encoder=false), but eval must encode prompts live. The eval-time
    # encoder is the SAME frozen umt5-xxl used by precompute_text_embeds.py, so language
    # conditioning is identical — does NOT break alignment.
    # Re-apply the 2048/16/28 geometry the run trained with (config default is 3072/24/30).
    EXTRA_ARGS="model=fastwam data=libero_2cam model.visual_encoder=null model.load_text_encoder=true model.skip_dit_load_from_pretrain=true model.skip_video_dit_load_from_pretrain=true model.action_dit_pretrained_path=null model.mot_checkpoint_mixed_attn=true model.action_dit_config.hidden_dim=2048 model.action_dit_config.ffn_dim=8192 model.action_dit_config.num_heads=16 model.action_dit_config.attn_head_dim=128 model.action_dit_config.num_layers=28 model.video_dit_config.hidden_dim=2048 model.video_dit_config.ffn_dim=8192 model.video_dit_config.num_heads=16 model.video_dit_config.attn_head_dim=128 model.video_dit_config.num_layers=28"
    ;;
  smoke)
    CKPT_DIR="runs/libero_hfastwam/libero_hfastwam_8card_small_ds"
    CKPT="${REPO_ROOT}/${CKPT_DIR}/checkpoints/weights/step_021700.pt"
    echo "[eval] SMOKE test: hfastwam_small, libero_spatial task 0, 1 trial, GPU ${CUDA_VISIBLE_DEVICES:-0}"
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python experiments/libero/eval_libero_single.py \
      task="${CONFIG}" ckpt="${CKPT}" gpu_id="${CUDA_VISIBLE_DEVICES:-0}" \
      model=hfastwam_small data=libero_2cam_interleaved \
      model.visual_encoder=null model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null \
      EVALUATION.disable_subtask_generation=true \
      EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 \
      EVALUATION.num_trials=1 EVALUATION.output_dir="${REPO_ROOT}/evaluate_results/_smoketest"
    ;;
  *)
    echo "Unknown model '${MODEL}'. Use: hfastwam_small | dino | vjepa_predictor | fastwam | smoke" >&2
    exit 1
    ;;
esac

# --- pick latest checkpoint in the run dir ---
CKPT="$(ls -t "${REPO_ROOT}/${CKPT_DIR}"/checkpoints/weights/step_*.pt 2>/dev/null | head -1)"
if [[ -z "${CKPT}" ]]; then
  echo "[eval] ERROR: no checkpoint found under ${CKPT_DIR}/checkpoints/weights/" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-eval_${MODEL}}"
TASK_LIST="${TASK_LIST:-experiments/libero/task_lists/all_suites_10each.txt}"

echo "[eval] model=${MODEL}  ckpt=${CKPT}"
echo "[eval] RUN_ID=${RUN_ID}  TASK_LIST=${TASK_LIST}  GPUs=${CUDA_VISIBLE_DEVICES:-0-7}"
echo "[eval] output -> ${REPO_ROOT}/evaluate_results/${RUN_ID}/"

CKPT="${CKPT}" \
CONFIG="${CONFIG}" \
NUM_TRIALS="${NUM_TRIALS:-50}" \
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
RUN_ID="${RUN_ID}" \
EXTRA_ARGS="${EXTRA_ARGS}" \
bash experiments/libero/run_libero_parallel_test.sh "${TASK_LIST}"
