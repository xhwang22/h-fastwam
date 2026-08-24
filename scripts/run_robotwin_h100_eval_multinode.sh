#!/usr/bin/env bash
# Shard RoboTwin tasks over independent H100 nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
CURRENT_USER="${USER:-$(id -un)}"

MODEL_KIND="${MODEL_KIND:?Set MODEL_KIND=xr1, MODEL_KIND=idm, or MODEL_KIND=vjepa}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed training run directory}"
EXTRA_ARGS=("$@")

if [[ -n "${PET_NNODES:-}" ]]; then
  export FASTWAM_USE_EFA="${FASTWAM_USE_EFA:-0}"
  # shellcheck source=_aws_hyperpod_setup.sh
  source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
  fastwam_prepare_aws_hyperpod_runtime
fi

MANAGED_LAUNCH=0
NODE_INDEX=0
if [[ "${FASTWAM_MANAGED_DISTRIBUTED:-0}" == "1" ]]; then
  NODE_COUNT="${NNODES:?NNODES is required for a managed multi-node launch}"
  NODE_INDEX="${NODE_RANK:?NODE_RANK is required for a managed multi-node launch}"
  : "${MASTER_ADDR:?MASTER_ADDR is required for a managed multi-node launch}"
  : "${MASTER_PORT:?MASTER_PORT is required for a managed multi-node launch}"
  MANAGED_LAUNCH=1
else
  NODE_IP_LIST="${NODE_IP_LIST:?Set NODE_IP_LIST=chief_ip,node2_ip,... outside HyperPod/PET}"
  IFS=',' read -r -a NODES <<< "${NODE_IP_LIST}"
  NODE_COUNT="${#NODES[@]}"
fi
if [[ ! "${NODE_COUNT}" =~ ^[0-9]+$ ]] || \
   [[ ! "${NODE_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[h100-multinode] ERROR: node count and index must be non-negative integers." >&2
  exit 1
fi
if (( NODE_COUNT < 2 )); then
  echo "[h100-multinode] ERROR: at least two nodes are required." >&2
  exit 1
fi
if [[ -n "${EXPECTED_EVAL_NNODES:-}" ]]; then
  if [[ ! "${EXPECTED_EVAL_NNODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[h100-multinode] ERROR: EXPECTED_EVAL_NNODES must be a positive integer." >&2
    exit 1
  fi
  if (( NODE_COUNT != EXPECTED_EVAL_NNODES )); then
    echo "[h100-multinode] ERROR: expected ${EXPECTED_EVAL_NNODES} nodes, got ${NODE_COUNT}." >&2
    exit 1
  fi
fi
if (( NODE_INDEX >= NODE_COUNT )); then
  echo "[h100-multinode] ERROR: node index ${NODE_INDEX} is outside ${NODE_COUNT} nodes." >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-8}"
SSH_PORT="${SSH_PORT:-36000}"
RUN_DIR="$(realpath "${RUN_DIR}")"
if [[ -n "${CKPT:-}" ]]; then
  CKPT="$(realpath "${CKPT}")"
else
  CKPT="$(find "${RUN_DIR}/checkpoints/weights" -maxdepth 1 -type f -name 'step_*.pt' \
    | sort -V | tail -1)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[h100-multinode] ERROR: checkpoint not found under ${RUN_DIR}" >&2
  exit 1
fi

MODE="${MODE:-full}"
CKPT_NAME="$(basename "${CKPT}" .pt)"
# Keep the single-node default so rerunning with more nodes resumes existing
# per-task clean/random results instead of creating a separate output tree.
OUTPUT_TAG="${OUTPUT_TAG:-${MODEL_KIND}_${CKPT_NAME}_h100_${MODE}_aligned_v2}"
EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/checkpoints/h100_eval_configs/${MODEL_KIND}_$(basename "${RUN_DIR}").yaml}"
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

prepare_shared_eval() {
  MODEL_KIND="${MODEL_KIND}" \
  RUN_DIR="${RUN_DIR}" \
  CKPT="${CKPT}" \
  MODE="${MODE}" \
  NUM_GPUS="${NUM_GPUS}" \
  OUTPUT_TAG="${OUTPUT_TAG}" \
  EVAL_CONFIG="${EVAL_CONFIG}" \
  DRY_RUN=1 \
  CHECK_ENV="${CHECK_ENV:-1}" \
  bash "${SCRIPT_DIR}/run_robotwin_h100_eval.sh"
}

merge_shard_summaries() {
  local ckpt_tag
  local run_output_dir
  ckpt_tag="$(python3 - "${CKPT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve()
parts = path.parts
if "runs" in parts:
    index = parts.index("runs")
    print(f"{parts[index + 1]}_{parts[index + 2]}")
else:
    print(path.stem)
PY
)"
  run_output_dir="${REPO_ROOT}/evaluate_results/robotwin/${ckpt_tag}/${OUTPUT_TAG}"
  python3 scripts/merge_robotwin_eval_shards.py \
    --run-output-dir "${run_output_dir}" \
    --shard-count "${NODE_COUNT}"
}

export MODEL_KIND RUN_DIR CKPT MODE NUM_GPUS OUTPUT_TAG EVAL_CONFIG
export FASTWAM_EVAL_ENV CONDA_SH
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
export SKIP_EVAL_PREPARE=1
export CHECK_ENV="${CHECK_ENV:-1}"

FORWARD_VARS=(
  MODEL_KIND RUN_DIR CKPT MODE NUM_GPUS OUTPUT_TAG EVAL_CONFIG
  FASTWAM_EVAL_ENV CONDA_SH MAX_TASKS_PER_GPU SKIP_EVAL_PREPARE CHECK_ENV
  FASTWAM_EVAL_USE_CURRENT_ENV PYTHON_BIN MINIFORGE_ROOT
  SKIP_MODEL_PREFLIGHT QWEN_REVISION
  EXPECTED_EVAL_NNODES
  TRAIN_CONFIG TASK_NAME
  STATS TRAINING_STATS ROBOTWIN_ROOT QWEN_DIR XR1_CHECKPOINT
  VJEPA21_CHECKPOINT VJEPA21_REPO
  CUDA_TOOLKIT_ROOT USE_SYSTEM_NVIDIA_GRAPHICS
  NVIDIA_GRAPHICS_ENV NVIDIA_GRAPHICS_ROOT
  HF_HOME HF_HUB_CACHE TORCH_HOME DIFFSYNTH_MODEL_BASE_PATH
  DIFFSYNTH_DOWNLOAD_SOURCE PYTORCH_CUDA_ALLOC_CONF FASTWAM_SDPA_BACKEND
  RENDER_BACKEND CAMERA_TYPE EVAL_EPISODES EVAL_VIDEO_LOG MAX_RETRIES
  NUM_INFERENCE_STEPS REPLAN_STEPS TIMING_ENABLED
  CHECK_ALIGNMENT DRY_RUN
)

build_env_prefix() {
  local prefix=""
  local name
  for name in "${FORWARD_VARS[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      prefix+="${name}=$(printf '%q' "${!name}") "
    fi
  done
  printf '%s' "${prefix}"
}

if (( MANAGED_LAUNCH == 1 )); then
  if [[ ! "${MASTER_PORT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[h100-multinode] ERROR: MASTER_PORT must be a positive integer." >&2
    exit 1
  fi
  PREPARE_SYNC_PORT="${EVAL_PREPARE_SYNC_PORT:-${MASTER_PORT}}"
  EVAL_SYNC_PORT="${EVAL_SYNC_PORT:-$((MASTER_PORT + 1))}"
  if [[ ! "${PREPARE_SYNC_PORT}" =~ ^[1-9][0-9]*$ ]] || \
     [[ ! "${EVAL_SYNC_PORT}" =~ ^[1-9][0-9]*$ ]] || \
     (( PREPARE_SYNC_PORT > 65535 || \
        EVAL_SYNC_PORT < 1 || EVAL_SYNC_PORT > 65535 )); then
    echo "[h100-multinode] ERROR: synchronization ports must be in [1, 65535]." >&2
    exit 1
  fi
  if [[ "${FASTWAM_EVAL_USE_CURRENT_ENV:-0}" == "1" ]]; then
    SYNC_PYTHON="${PYTHON_BIN:-/opt/venv/bin/python}"
  else
    SYNC_PYTHON="${FASTWAM_EVAL_ENV}/bin/python"
  fi
  if [[ ! -x "${SYNC_PYTHON}" ]]; then
    echo "[h100-multinode] ERROR: synchronization Python not found: ${SYNC_PYTHON}" >&2
    exit 1
  fi

  PREPARE_STATUS=0
  if (( NODE_INDEX == 0 )); then
    set +e
    prepare_shared_eval
    PREPARE_STATUS=$?
    set -e
  fi
  "${SYNC_PYTHON}" scripts/sync_robotwin_eval_nodes.py \
    --phase prepare \
    --rank "${NODE_INDEX}" \
    --world-size "${NODE_COUNT}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${PREPARE_SYNC_PORT}" \
    --local-exit-code "${PREPARE_STATUS}" \
    --timeout-seconds "${PREPARE_SYNC_TIMEOUT_SECONDS:-1800}"

  export SKIP_EVAL_PREPARE=1
  echo "[h100-multinode] launch managed shard=${NODE_INDEX}/${NODE_COUNT}"
  set +e
  bash scripts/run_robotwin_h100_eval.sh \
    "${EXTRA_ARGS[@]}" \
    MULTIRUN.task_shard_count="${NODE_COUNT}" \
    MULTIRUN.task_shard_index="${NODE_INDEX}"
  LOCAL_STATUS=$?
  set -e

  "${SYNC_PYTHON}" scripts/sync_robotwin_eval_nodes.py \
    --phase evaluation \
    --rank "${NODE_INDEX}" \
    --world-size "${NODE_COUNT}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${EVAL_SYNC_PORT}" \
    --local-exit-code "${LOCAL_STATUS}" \
    --timeout-seconds "${EVAL_SYNC_TIMEOUT_SECONDS:-604800}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if (( NODE_INDEX == 0 )); then
      echo "[h100-multinode] dry-run completed for all ${NODE_COUNT} task shards."
    fi
    exit 0
  fi
  if (( NODE_INDEX == 0 )) && [[ "${SKIP_MERGE:-0}" != "1" ]]; then
    merge_shard_summaries
  fi
  echo "[h100-multinode] managed shard=${NODE_INDEX}/${NODE_COUNT} completed."
  exit 0
fi

# In SSH mode, prepare shared config/policy once before remote managers start.
prepare_shared_eval

ENV_PREFIX="$(build_env_prefix)"
REMOTE_EXTRA_ARGS=""
for argument in "${EXTRA_ARGS[@]}"; do
  REMOTE_EXTRA_ARGS+=" $(printf '%q' "${argument}")"
done
LOCAL_ENV_ARGS=()
for name in "${FORWARD_VARS[@]}"; do
  if [[ -n "${!name+x}" ]]; then
    LOCAL_ENV_ARGS+=("${name}=${!name}")
  fi
done
PIDS=()

echo "[h100-multinode] output_tag=${OUTPUT_TAG}"
echo "[h100-multinode] existing valid per-task results will be resumed"
for (( shard_index=1; shard_index<NODE_COUNT; shard_index++ )); do
  host="${NODES[$shard_index]%%:*}"
  remote_cmd="cd $(printf '%q' "${REPO_ROOT}") && ${ENV_PREFIX}bash scripts/run_robotwin_h100_eval.sh${REMOTE_EXTRA_ARGS} MULTIRUN.task_shard_count=${NODE_COUNT} MULTIRUN.task_shard_index=${shard_index}"
  echo "[h100-multinode] launch shard=${shard_index}/${NODE_COUNT} host=${host}"
  # shellcheck disable=SC2029
  ssh -p "${SSH_PORT}" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=no \
    "${host}" "${remote_cmd}" &
  PIDS+=($!)
done

echo "[h100-multinode] launch shard=0/${NODE_COUNT} locally"
set +e
env "${LOCAL_ENV_ARGS[@]}" bash scripts/run_robotwin_h100_eval.sh \
  "${EXTRA_ARGS[@]}" \
  MULTIRUN.task_shard_count="${NODE_COUNT}" \
  MULTIRUN.task_shard_index=0
LOCAL_STATUS=$?
set -e

REMOTE_STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    REMOTE_STATUS=1
  fi
done
if (( LOCAL_STATUS != 0 || REMOTE_STATUS != 0 )); then
  echo "[h100-multinode] ERROR: one or more task shards failed." >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[h100-multinode] dry-run completed for all ${NODE_COUNT} task shards."
  exit 0
fi
if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
  merge_shard_summaries
fi
echo "[h100-multinode] all ${NODE_COUNT} task shards completed."
