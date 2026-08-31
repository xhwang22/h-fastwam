#!/usr/bin/env bash
# Run one manually assigned RoboTwin task shard and merge when all shards finish.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EVAL_ENTRYPOINT="${EVAL_ENTRYPOINT:?Set EVAL_ENTRYPOINT to a single-node eval wrapper}"
MODEL_KIND="${MODEL_KIND:?Set MODEL_KIND}"
EVAL_LABEL="${EVAL_LABEL:?Set EVAL_LABEL}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR}"
TASK_SHARD_COUNT="${TASK_SHARD_COUNT:-5}"
TASK_SHARD_INDEX="${TASK_SHARD_INDEX:?Set TASK_SHARD_INDEX=0..$(( TASK_SHARD_COUNT - 1 ))}"
EXTRA_ARGS=("$@")

if [[ ! "${TASK_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TASK_SHARD_COUNT must be positive, got ${TASK_SHARD_COUNT}." >&2
  exit 2
fi
if [[ ! "${TASK_SHARD_INDEX}" =~ ^[0-9]+$ ]] || \
   (( TASK_SHARD_INDEX >= TASK_SHARD_COUNT )); then
  echo "ERROR: TASK_SHARD_INDEX must be in [0,$(( TASK_SHARD_COUNT - 1 ))], got ${TASK_SHARD_INDEX}." >&2
  exit 2
fi

RUN_DIR="$(realpath "${RUN_DIR}")"
if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(
    find "${RUN_DIR}/checkpoints/weights" -maxdepth 1 \
      -type f -name 'step_*.pt' | sort -V | tail -1
  )"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT:-<unset>}" >&2
  exit 1
fi
CKPT="$(realpath "${CKPT}")"

MODE="${MODE:-full}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
CKPT_NAME="$(basename "${CKPT}" .pt)"
OUTPUT_TAG="${OUTPUT_TAG:-${EVAL_LABEL}_${CKPT_NAME}_h100_${MODE}_${TASK_SHARD_COUNT}node}"
EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/checkpoints/h100_eval_configs/${MODEL_KIND}_$(basename "${RUN_DIR}").yaml}"
EVAL_SYNC_ROOT="${EVAL_SYNC_ROOT:-/efs/shaunxhwang/robotwin_eval_sync}"
EVAL_SYNC_DIR="${EVAL_SYNC_ROOT}/${OUTPUT_TAG}"
mkdir -p "${EVAL_SYNC_DIR}"

export MODEL_KIND EVAL_LABEL RUN_DIR CKPT MODE NUM_GPUS MAX_TASKS_PER_GPU
export OUTPUT_TAG EVAL_CONFIG

# Serialize config generation and compatibility patching, but validate every
# node's local simulator environment once before it launches its shard.
PREPARED_MARKER="${EVAL_SYNC_DIR}/prepared.shard${TASK_SHARD_INDEX}"
(
  flock -x 9
  if [[ ! -f "${PREPARED_MARKER}" ]] || \
     [[ "$(cat "${PREPARED_MARKER}")" != "${CKPT}" ]]; then
    DRY_RUN=1 \
    SKIP_EVAL_PREPARE=0 \
      bash "${EVAL_ENTRYPOINT}" "${EXTRA_ARGS[@]}"
    printf '%s\n' "${CKPT}" > "${PREPARED_MARKER}"
  fi
) 9>"${EVAL_SYNC_DIR}/prepare.lock"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  SKIP_EVAL_PREPARE=1 \
    bash "${EVAL_ENTRYPOINT}" \
      "${EXTRA_ARGS[@]}" \
      "MULTIRUN.task_shard_count=${TASK_SHARD_COUNT}" \
      "MULTIRUN.task_shard_index=${TASK_SHARD_INDEX}"
  exit 0
fi

SKIP_EVAL_PREPARE=1 \
  bash "${EVAL_ENTRYPOINT}" \
    "${EXTRA_ARGS[@]}" \
    "MULTIRUN.task_shard_count=${TASK_SHARD_COUNT}" \
    "MULTIRUN.task_shard_index=${TASK_SHARD_INDEX}"

SHARD_DONE="${EVAL_SYNC_DIR}/done.shard${TASK_SHARD_INDEX}"
printf '%s\n' "${CKPT}" > "${SHARD_DONE}"

CKPT_TAG="${CKPT_NAME}"
IFS='/' read -r -a _CKPT_PARTS <<< "${CKPT}"
for (( part=0; part<${#_CKPT_PARTS[@]}; part++ )); do
  if [[ "${_CKPT_PARTS[part]}" == "runs" ]]; then
    if (( part + 2 >= ${#_CKPT_PARTS[@]} )); then
      echo "ERROR: checkpoint under runs has invalid layout: ${CKPT}" >&2
      exit 1
    fi
    CKPT_TAG="${_CKPT_PARTS[part + 1]}_${_CKPT_PARTS[part + 2]}"
    break
  fi
done
RUN_OUTPUT_DIR="${REPO_ROOT}/evaluate_results/robotwin/${CKPT_TAG}/${OUTPUT_TAG}"
(
  flock -x 9
  all_done=1
  for (( shard=0; shard<TASK_SHARD_COUNT; shard++ )); do
    marker="${EVAL_SYNC_DIR}/done.shard${shard}"
    if [[ ! -f "${marker}" ]] || [[ "$(cat "${marker}")" != "${CKPT}" ]]; then
      all_done=0
      break
    fi
  done
  if (( all_done )) && [[ ! -f "${EVAL_SYNC_DIR}/merged" ]]; then
    if [[ -z "${PYTHON_BIN:-}" ]]; then
      if [[ -x "/fsx/conda-envs/fastwam-eval/bin/python" ]]; then
        PYTHON_BIN="/fsx/conda-envs/fastwam-eval/bin/python"
      else
        PYTHON_BIN="$(command -v python3 || command -v python)"
      fi
    fi
    "${PYTHON_BIN}" scripts/merge_robotwin_eval_shards.py \
      --run-output-dir "${RUN_OUTPUT_DIR}" \
      --shard-count "${TASK_SHARD_COUNT}"
    printf '%s\n' "${CKPT}" > "${EVAL_SYNC_DIR}/merged"
  fi
) 9>"${EVAL_SYNC_DIR}/merge.lock"

echo "[h100-manual-shard] completed shard ${TASK_SHARD_INDEX}/${TASK_SHARD_COUNT}"
echo "[h100-manual-shard] output=${RUN_OUTPUT_DIR}"
