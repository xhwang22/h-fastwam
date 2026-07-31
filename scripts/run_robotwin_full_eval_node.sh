#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${TRAIN_CONFIG:?Set TRAIN_CONFIG to the saved training config.yaml}"
: "${CKPT:?Set CKPT to the evaluation checkpoint}"
: "${OUTPUT_TAG:?Set OUTPUT_TAG to a unique evaluation run name}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/checkpoints/RoboTwin}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
SWIFTSHADER_ICD="${SWIFTSHADER_ICD:-${REPO_ROOT}/checkpoints/swiftshader/vk_swiftshader_icd.json}"
NUM_GPUS="${NUM_GPUS:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-5}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
TASK_SHARD_COUNT="${TASK_SHARD_COUNT:-1}"
TASK_SHARD_INDEX="${TASK_SHARD_INDEX:-0}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

export no_proxy="${no_proxy:-.woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com,.myqcloud.com,.tencentcos.cn}"
export NO_PROXY="${no_proxy}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export ftp_proxy="${ftp_proxy:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt="${CKPT}" \
  EVALUATION.train_config_path="${TRAIN_CONFIG}" \
  EVALUATION.dataset_stats_path="${REPO_ROOT}/data/robotwin2.0/dataset_stats.json" \
  EVALUATION.robotwin_root="${ROBOTWIN_ROOT}" \
  EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES}" \
  EVALUATION.render_backend=cpu \
  EVALUATION.swiftshader_icd_path="${SWIFTSHADER_ICD}" \
  EVALUATION.output_dir="evaluate_results/${OUTPUT_TAG}" \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.replan_steps=24 \
  EVALUATION.timing_enabled=false \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.eval_video_log=false \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${WORKERS_PER_GPU}" \
  MULTIRUN.max_retries=2 \
  MULTIRUN.task_shard_count="${TASK_SHARD_COUNT}" \
  MULTIRUN.task_shard_index="${TASK_SHARD_INDEX}"
