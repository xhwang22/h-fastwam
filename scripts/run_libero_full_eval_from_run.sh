#!/usr/bin/env bash
set -euo pipefail

: "${RUN_DIR:?Set RUN_DIR to the completed training directory}"
: "${OUTPUT_TAG:?Set OUTPUT_TAG to the LIBERO evaluation output name}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
# shellcheck disable=SC1090
source "${CONDA_ACTIVATE}" fastwam

TRAIN_CONFIG="${RUN_DIR}/config.yaml"
DATASET_STATS="${RUN_DIR}/dataset_stats.json"
CKPT="${CKPT:-$(find "${RUN_DIR}/checkpoints/weights" -maxdepth 1 -name 'step_*.pt' -printf '%p\n' | sort -V | tail -1)}"

for required in "${TRAIN_CONFIG}" "${DATASET_STATS}" "${CKPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[libero-eval] ERROR: missing ${required}" >&2
    exit 1
  fi
done

if ! python -c 'import ctypes; ctypes.CDLL("libGL.so.1")' >/dev/null 2>&1; then
  yum install -y mesa-libGL mesa-libGL-devel mesa-libOSMesa mesa-libOSMesa-devel mesa-dri-drivers
fi

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints"
export HF_HOME="${REPO_ROOT}/checkpoints/hf_cache"
export TORCH_HOME="${REPO_ROOT}/checkpoints/torch_hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=(
  "EVALUATION.train_config_path=${TRAIN_CONFIG}"
  "EVALUATION.dataset_stats_path=${DATASET_STATS}"
  "EVALUATION.disable_subtask_generation=true"
  "EVALUATION.visualize_future_video=false"
)

CKPT="${CKPT}" \
CONFIG=libero_uncond_2cam224_1e-4 \
NUM_TRIALS="${NUM_TRIALS:-50}" \
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
RUN_ID="${OUTPUT_TAG}" \
OUTPUT_DIR="${REPO_ROOT}/evaluate_results/${OUTPUT_TAG}" \
EXTRA_ARGS="${EXTRA_ARGS[*]}" \
bash experiments/libero/run_libero_parallel_test.sh \
  experiments/libero/task_lists/all_suites_10each.txt
