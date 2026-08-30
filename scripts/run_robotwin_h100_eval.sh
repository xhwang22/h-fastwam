#!/usr/bin/env bash
# Generic single-node H100 RoboTwin evaluation launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
CURRENT_USER="${USER:-$(id -un)}"

FASTWAM_EVAL_USE_CURRENT_ENV="${FASTWAM_EVAL_USE_CURRENT_ENV:-0}"
if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV}" == "1" ]]; then
  if [[ -z "${PYTHON_BIN:-}" && -x "/opt/venv/bin/python" ]]; then
    PYTHON_BIN=/opt/venv/bin/python
  fi
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
  if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "[h100-eval] ERROR: current Python not found; set PYTHON_BIN." >&2
    exit 1
  fi
  PYTHON_ENV_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
else
  if [[ -z "${FASTWAM_EVAL_ENV:-}" ]]; then
    if [[ -d "/fsx/conda-envs/fastwam-eval" ]]; then
      FASTWAM_EVAL_ENV="/fsx/conda-envs/fastwam-eval"
    else
      FASTWAM_EVAL_ENV="/fsx/${CURRENT_USER}/conda-envs/fastwam-eval"
    fi
  fi
  if [[ -z "${CONDA_SH:-}" ]]; then
    if [[ -f "/fsx/miniforge3/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="/fsx/miniforge3/etc/profile.d/conda.sh"
    else
      CONDA_SH="/fsx/${CURRENT_USER}/miniforge3/etc/profile.d/conda.sh"
    fi
  fi
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[h100-eval] ERROR: conda activation script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # Always reactivate so a stale nested virtualenv cannot take precedence.
  # shellcheck disable=SC1090
  set +u
  source "${CONDA_SH}"
  conda activate "${FASTWAM_EVAL_ENV}"
  unset VIRTUAL_ENV
  set -u
  hash -r
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  PYTHON_ENV_PREFIX="${CONDA_PREFIX}"
fi
export FASTWAM_PYTHON_ENV_PREFIX="${PYTHON_ENV_PREFIX}"

if [[ -z "${NVIDIA_GRAPHICS_ENV:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_VERSION="$(
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
  )"
  CANDIDATE_GRAPHICS_ENV="/fsx/nvidia-userspace/${DRIVER_VERSION}/activate.sh"
  if [[ -f "${CANDIDATE_GRAPHICS_ENV}" ]]; then
    NVIDIA_GRAPHICS_ENV="${CANDIDATE_GRAPHICS_ENV}"
  fi
fi
if [[ -n "${NVIDIA_GRAPHICS_ENV:-}" ]]; then
  if [[ ! -f "${NVIDIA_GRAPHICS_ENV}" ]]; then
    echo "[h100-eval] ERROR: NVIDIA graphics activation not found: ${NVIDIA_GRAPHICS_ENV}" >&2
    exit 1
  else
    # shellcheck disable=SC1090
    source "${NVIDIA_GRAPHICS_ENV}"
  fi
fi

MODEL_KIND="${MODEL_KIND:?Set MODEL_KIND to a supported evaluation profile}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed training run directory}"
CAMERA_TYPE="${CAMERA_TYPE:-Large_D435}"
for override in "$@"; do
  override_key="${override%%=*}"
  case "${override_key}" in
    EVALUATION.task_name|EVALUATION.eval_num_episodes|EVALUATION.eval_video_log|\
    EVALUATION.render_backend|EVALUATION.num_inference_steps|\
    EVALUATION.replan_steps|EVALUATION.skip_get_obs_within_replan|\
    EVALUATION.timing_enabled|MULTIRUN.*)
      ;;
    *)
      echo "[h100-eval] ERROR: unsupported/unsafe override: ${override}" >&2
      exit 1
      ;;
  esac
done
RUN_DIR="$(realpath "${RUN_DIR}")"
TRAIN_CONFIG="${TRAIN_CONFIG:-${RUN_DIR}/config.yaml}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/checkpoints/RoboTwin}"
if [[ -z "${STATS:-}" ]]; then
  if [[ -f "/efs/shaunxhwang/robotwin2.0_webdataset/dataset_stats.json" ]]; then
    STATS="/efs/shaunxhwang/robotwin2.0_webdataset/dataset_stats.json"
  elif [[ -f "${REPO_ROOT}/data/robotwin2.0_webdataset/dataset_stats.json" ]]; then
    STATS="${REPO_ROOT}/data/robotwin2.0_webdataset/dataset_stats.json"
  else
    STATS="${REPO_ROOT}/data/robotwin2.0/dataset_stats.json"
  fi
fi

if [[ -n "${CKPT:-}" ]]; then
  CKPT="$(realpath "${CKPT}")"
else
  CKPT="$(find "${RUN_DIR}/checkpoints/weights" -maxdepth 1 -type f -name 'step_*.pt' \
    | sort -V | tail -1)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[h100-eval] ERROR: checkpoint not found under ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "[h100-eval] ERROR: training config not found: ${TRAIN_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${STATS}" ]]; then
  echo "[h100-eval] ERROR: dataset stats not found: ${STATS}" >&2
  exit 1
fi
if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "[h100-eval] ERROR: RoboTwin root not found: ${ROBOTWIN_ROOT}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${REPO_ROOT}/checkpoints/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
# The experimental IDM KV cache changed predictions and must stay disabled.
export FASTWAM_IDM_INFERENCE_KV_CACHE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/fastwam_torch_extensions_${CURRENT_USER}}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/fastwam_warp_cache_${CURRENT_USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/fastwam_xdg_cache_${CURRENT_USER}}"
export LD_LIBRARY_PATH="${PYTHON_ENV_PREFIX}/lib:${PYTHON_ENV_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${WARP_CACHE_PATH}" "${XDG_CACHE_HOME}"

if [[ -z "${QWEN_DIR:-}" ]]; then
  DIRECT_QWEN_DIR="${REPO_ROOT}/checkpoints/Qwen/Qwen3-VL-2B-Instruct"
  SNAPSHOT_ROOT="${HF_HUB_CACHE}/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
  if [[ -f "${DIRECT_QWEN_DIR}/config.json" ]]; then
    QWEN_DIR="${DIRECT_QWEN_DIR}"
  elif [[ -d "${SNAPSHOT_ROOT}" ]]; then
    QWEN_DIR="$(find -L "${SNAPSHOT_ROOT}" \
      -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
  fi
fi
QWEN_DIR="${QWEN_DIR:-}"
if [[ ! -f "${QWEN_DIR}/config.json" ]]; then
  echo "[h100-eval] ERROR: complete Qwen3-VL-2B snapshot not found: ${QWEN_DIR}" >&2
  exit 1
fi
if [[ "${SKIP_MODEL_PREFLIGHT:-0}" != "1" ]]; then
"${PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import transformers
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

if Version(transformers.__version__) < Version("4.57.0"):
    raise RuntimeError(
        f"transformers>=4.57.0 is required, found {transformers.__version__}"
    )
print(f"[h100-eval] transformers={transformers.__version__} Qwen3-VL=OK")
PY
fi

EVAL_CONFIG_DIR="${EVAL_CONFIG_DIR:-${REPO_ROOT}/checkpoints/h100_eval_configs}"
mkdir -p "${EVAL_CONFIG_DIR}"
EVAL_CONFIG="${EVAL_CONFIG:-${EVAL_CONFIG_DIR}/${MODEL_KIND}_$(basename "${RUN_DIR}").yaml}"

PREPARE_ARGS=(
  --model "${MODEL_KIND}"
  --input "${TRAIN_CONFIG}"
  --output "${EVAL_CONFIG}"
  --qwen-dir "${QWEN_DIR}"
)
case "${MODEL_KIND}" in
  xr1)
    XR1_CHECKPOINT="${XR1_CHECKPOINT:-${REPO_ROOT}/checkpoints/XiaomiRobotics/Xiaomi-Robotics-1-5B}"
    PREPARE_ARGS+=(--xr1-checkpoint "${XR1_CHECKPOINT}")
    TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
    ;;
  idm)
    VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
    VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
    PREPARE_ARGS+=(
      --vjepa-checkpoint "${VJEPA21_CHECKPOINT}"
      --vjepa-repo "${VJEPA21_REPO}"
    )
    TASK_CONFIG=robotwin_idm_3cam_384_1e-4
    ;;
  vjepa21_flow)
    VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
    VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
    PREPARE_ARGS+=(
      --vjepa-checkpoint "${VJEPA21_CHECKPOINT}"
      --vjepa-repo "${VJEPA21_REPO}"
    )
    if [[ -n "${VJEPA21_NORMALISE_STATS_PATH:-}" ]]; then
      PREPARE_ARGS+=(
        --vjepa-normalise-stats "${VJEPA21_NORMALISE_STATS_PATH}"
      )
    fi
    TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
    ;;
  dinov3_flow)
    export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-${REPO_ROOT}/checkpoints/dinov3-vith16plus-pretrain-lvd1689m}"
    PREPARE_ARGS+=(--dinov3-model "${DINOV3_MODEL_PATH}")
    TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
    ;;
  siglip2_flow)
    export SIGLIP2_MODEL_PATH="${SIGLIP2_MODEL_PATH:-${REPO_ROOT}/checkpoints/siglip2-so400m-patch16-384}"
    PREPARE_ARGS+=(--siglip2-model "${SIGLIP2_MODEL_PATH}")
    TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
    ;;
  vae_predictor)
    TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
    ;;
  *)
    echo "[h100-eval] ERROR: unsupported MODEL_KIND=${MODEL_KIND}." >&2
    echo "Expected xr1, idm, vjepa21_flow, dinov3_flow, siglip2_flow, or vae_predictor." >&2
    exit 1
    ;;
esac

if [[ "${SKIP_EVAL_PREPARE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/prepare_robotwin_h100_eval_config.py "${PREPARE_ARGS[@]}"

  EXPECTED_POLICY="${REPO_ROOT}/experiments/robotwin/fastwam_policy"
  POLICY_LINK="${ROBOTWIN_ROOT}/policy/fastwam_policy"
  if [[ -L "${POLICY_LINK}" ]]; then
    CURRENT_POLICY="$(realpath "${POLICY_LINK}")"
    if [[ "${CURRENT_POLICY}" != "$(realpath "${EXPECTED_POLICY}")" ]]; then
      echo "[h100-eval] ERROR: policy symlink conflict:" >&2
      echo "  current:  ${CURRENT_POLICY}" >&2
      echo "  expected: ${EXPECTED_POLICY}" >&2
      exit 1
    fi
  elif [[ -e "${POLICY_LINK}" ]]; then
    echo "[h100-eval] ERROR: policy path exists and is not a symlink: ${POLICY_LINK}" >&2
    exit 1
  else
    ln -s "${EXPECTED_POLICY}" "${POLICY_LINK}"
  fi
  "${PYTHON_BIN}" scripts/patch_robotwin_eval_compat.py --robotwin-root "${ROBOTWIN_ROOT}"
else
  if [[ ! -f "${EVAL_CONFIG}" ]]; then
    echo "[h100-eval] ERROR: prepared eval config not found: ${EVAL_CONFIG}" >&2
    exit 1
  fi
fi

if [[ "${CHECK_ALIGNMENT:-1}" == "1" ]]; then
  ALIGN_ARGS=(
    --model "${MODEL_KIND}"
    --train-config "${TRAIN_CONFIG}"
    --eval-config "${EVAL_CONFIG}"
    --checkpoint "${CKPT}"
    --dataset-stats "${STATS}"
    --repo-root "${REPO_ROOT}"
    --camera-type "${CAMERA_TYPE}"
  )
  if [[ -n "${TRAINING_STATS:-}" ]]; then
    ALIGN_ARGS+=(--training-stats "${TRAINING_STATS}")
  fi
  "${PYTHON_BIN}" scripts/check_robotwin_eval_alignment.py "${ALIGN_ARGS[@]}"
fi

if [[ "${CHECK_ENV:-1}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/check_robotwin_h100_eval_env.py --robotwin-root "${ROBOTWIN_ROOT}"
fi

MODE="${MODE:-smoke}"
if [[ "${MODE}" == "smoke" ]]; then
  EVAL_EPISODES="${EVAL_EPISODES:-2}"
  TASK_NAME="${TASK_NAME:-click_alarmclock}"
  EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-true}"
elif [[ "${MODE}" == "full" ]]; then
  EVAL_EPISODES="${EVAL_EPISODES:-100}"
  TASK_NAME="${TASK_NAME:-}"
  EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-false}"
else
  echo "[h100-eval] ERROR: MODE must be smoke or full." >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
RENDER_BACKEND="${RENDER_BACKEND:-gpu}"
CKPT_NAME="$(basename "${CKPT}" .pt)"
EVAL_LABEL="${EVAL_LABEL:-${MODEL_KIND}}"
OUTPUT_TAG="${OUTPUT_TAG:-${EVAL_LABEL}_${CKPT_NAME}_h100_${MODE}_aligned_v2}"

CMD=(
  "${PYTHON_BIN}" experiments/robotwin/run_robotwin_manager.py
  task="${TASK_CONFIG}"
  ckpt="${CKPT}"
  EVALUATION.train_config_path="${EVAL_CONFIG}"
  EVALUATION.dataset_stats_path="${STATS}"
  EVALUATION.robotwin_root="${ROBOTWIN_ROOT}"
  EVALUATION.eval_num_episodes="${EVAL_EPISODES}"
  EVALUATION.eval_video_log="${EVAL_VIDEO_LOG}"
  EVALUATION.render_backend="${RENDER_BACKEND}"
  EVALUATION.camera_type="${CAMERA_TYPE}"
  EVALUATION.num_inference_steps="${NUM_INFERENCE_STEPS}"
  EVALUATION.replan_steps="${REPLAN_STEPS}"
  EVALUATION.skip_get_obs_within_replan=true
  EVALUATION.timing_enabled="${TIMING_ENABLED:-false}"
  EVALUATION.output_dir="evaluate_results/${OUTPUT_TAG}"
  MULTIRUN.num_gpus="${NUM_GPUS}"
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}"
  MULTIRUN.max_retries="${MAX_RETRIES}"
)
if [[ -n "${TASK_NAME}" ]]; then
  CMD+=(EVALUATION.task_name="${TASK_NAME}")
fi
CMD+=("$@")

echo "[h100-eval] model=${MODEL_KIND} mode=${MODE}"
echo "[h100-eval] ckpt=${CKPT}"
echo "[h100-eval] config=${EVAL_CONFIG}"
echo "[h100-eval] gpus=${NUM_GPUS} workers_per_gpu=${MAX_TASKS_PER_GPU}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[h100-eval] command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}"
