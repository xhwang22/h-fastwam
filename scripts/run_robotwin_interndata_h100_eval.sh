#!/usr/bin/env bash
# One-node H100 evaluation for run_robotwin_finetune_interndata_4x8.sh outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${RUN_DIR:-}" ]]; then
  shopt -s nullglob
  candidate_runs=(
    "${REPO_ROOT}"/runs/robotwin_hfastwam/robotwin_ft_interndata_*_4x8_b48
  )
  shopt -u nullglob
  valid_runs=()
  for candidate in "${candidate_runs[@]}"; do
    if [[ -f "${candidate}/config.yaml" ]] && compgen -G \
        "${candidate}/checkpoints/weights/step_*.pt" >/dev/null; then
      valid_runs+=("${candidate}")
    fi
  done
  if (( ${#valid_runs[@]} == 1 )); then
    RUN_DIR="${valid_runs[0]}"
  elif (( ${#valid_runs[@]} > 1 )); then
    echo "[interndata-h100-eval] ERROR: multiple completed runs found:" >&2
    printf '  %s\n' "${valid_runs[@]}" >&2
    echo "Set RUN_DIR to the exact run to evaluate." >&2
    exit 1
  fi
fi
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "[interndata-h100-eval] ERROR: no completed interndata fine-tune run found." >&2
  echo "Set RUN_DIR=/fsx/h-fastwam/runs/robotwin_hfastwam/<run-name>." >&2
  exit 1
fi

SETUP_EVAL_ENV="${SETUP_EVAL_ENV:-1}"
if [[ -z "${FASTWAM_EVAL_USE_CURRENT_ENV:-}" ]]; then
  if [[ "${SETUP_EVAL_ENV}" == "0" || -n "${PYTHON_BIN:-}" ]]; then
    export FASTWAM_EVAL_USE_CURRENT_ENV=1
  else
    export FASTWAM_EVAL_USE_CURRENT_ENV=0
  fi
fi
if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV}" == "1" && \
      -z "${PYTHON_BIN:-}" && -x "/opt/venv/bin/python" ]]; then
  export PYTHON_BIN=/opt/venv/bin/python
fi
if [[ "${SETUP_EVAL_ENV}" == "1" ]]; then
  bash "${SCRIPT_DIR}/setup_robotwin_h100_eval_env.sh"
fi
export USE_SYSTEM_NVIDIA_GRAPHICS="${USE_SYSTEM_NVIDIA_GRAPHICS:-${FASTWAM_EVAL_USE_CURRENT_ENV}}"

DRIVER_VERSION="$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader \
    | head -1
)"
NVIDIA_GRAPHICS_ROOT="${NVIDIA_GRAPHICS_ROOT:-/fsx/nvidia-userspace/${DRIVER_VERSION}}"
if [[ "${USE_SYSTEM_NVIDIA_GRAPHICS}" != "1" && \
      -z "${NVIDIA_GRAPHICS_ENV:-}" && \
      -f "${NVIDIA_GRAPHICS_ROOT}/activate.sh" ]]; then
  export NVIDIA_GRAPHICS_ENV="${NVIDIA_GRAPHICS_ROOT}/activate.sh"
fi
export MODEL_KIND=vjepa
export RUN_DIR
export MODE="${MODE:-smoke}"
export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
export RENDER_BACKEND=gpu
export CHECK_ALIGNMENT=1
export CHECK_ENV=1
export QWEN_REVISION="${QWEN_REVISION:-89644892e4d85e24eaac8bacfd4f463576704203}"
if [[ -z "${STATS:-}" ]]; then
  if [[ -f "/efs/shaunxhwang/robotwin2.0_webdataset/dataset_stats.json" ]]; then
    STATS="/efs/shaunxhwang/robotwin2.0_webdataset/dataset_stats.json"
  elif [[ -f "${REPO_ROOT}/data/robotwin2.0_webdataset/dataset_stats.json" ]]; then
    STATS="${REPO_ROOT}/data/robotwin2.0_webdataset/dataset_stats.json"
  else
    STATS="${REPO_ROOT}/data/robotwin2.0/dataset_stats.json"
  fi
fi
export STATS

exec bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh" "$@"
