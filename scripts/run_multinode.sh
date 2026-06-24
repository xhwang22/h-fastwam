#!/usr/bin/env bash
# Generic multi-node launcher for any FastWAM-style hydra+accelerate run.
#
# This script knows nothing about which model is being trained — it just:
#   1. Resolves multi-node topology from Tencent Taiji platform env vars
#      (or user overrides), exporting NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT.
#   2. Builds a hydra override list from MODEL / TASK / DATA / WANDB_NAME env.
#   3. Delegates to scripts/train_zero1.sh.
#
# Per-experiment thin wrappers (like run_libero_vjepa2ac_predictor_multinode.sh)
# can do model-specific sanity checks, set MODEL=... TASK=... etc, then
# `exec` to this script.
#
# Required env (or hydra overrides forwarded via EXTRA):
#   MODEL          hydra config name under configs/model/, e.g. fastwam_dino_xxx
#   TASK           hydra config name under configs/task/,  e.g. libero_uncond_2cam224_1e-4
#   DATA           hydra config name under configs/data/,  e.g. libero_2cam
#
# Optional env:
#   WANDB_NAME     wandb run name (default: ${TASK}_${RUN_PREFIX})
#   RUN_PREFIX     prefix for the run dir name (default: "run")
#   RUN_NAME       full run name; overrides RUN_PREFIX-based default
#   LOG_ROOT       parent dir for run logs (default: ./runs/${TASK})
#   FOREGROUND     1 = run in foreground with tee; 0 = nohup background (default 0)
#   EXTRA          extra hydra overrides, space-separated string
#                  e.g. EXTRA="model.loss.lambda_action=0.5 trainer.lr=2e-4"
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, NPROC_PER_NODE
#                  override Taiji-detected topology
#
# Usage examples:
#   # Plain DINO run on this Taiji job:
#   MODEL=fastwam_dino_xxx TASK=libero_uncond_2cam224_1e-4 DATA=libero_2cam \
#       RUN_PREFIX=dino bash scripts/run_multinode.sh
#
#   # Same, with an extra hydra override:
#   MODEL=fastwam_dino_xxx TASK=libero_uncond_2cam224_1e-4 DATA=libero_2cam \
#       RUN_PREFIX=dino EXTRA="trainer.lr=2e-4" \
#       bash scripts/run_multinode.sh
#
#   # Force single-node run (ignore Taiji multi-node setup):
#   NNODES=1 NODE_RANK=0 MODEL=... TASK=... DATA=... \
#       bash scripts/run_multinode.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

err()  { echo "[run_multinode] ERROR: $*" >&2; exit 1; }
info() { echo "[run_multinode] $*"; }

# ---------------------------------------------------------------------------
# 1. Required model/task/data inputs.
# ---------------------------------------------------------------------------
: "${MODEL:?Set MODEL=<hydra model config name>, e.g. fastwam_vjepa2ac_predictor}"
: "${TASK:?Set TASK=<hydra task config name>, e.g. libero_uncond_2cam224_1e-4}"
: "${DATA:?Set DATA=<hydra data config name>, e.g. libero_2cam}"

RUN_PREFIX="${RUN_PREFIX:-run}"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S)}"
WANDB_NAME="${WANDB_NAME:-${TASK}_${RUN_PREFIX}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/runs/${TASK}}"
FOREGROUND="${FOREGROUND:-0}"
EXTRA="${EXTRA:-}"

# ---------------------------------------------------------------------------
# 2. Resolve multi-node topology: explicit env > Taiji platform var > default.
# ---------------------------------------------------------------------------
NNODES="${NNODES:-${HOST_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${HOST_GPU_NUM:-8}}"
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-29500}"

is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
for v in NNODES NPROC_PER_NODE NODE_RANK MASTER_PORT; do
  if ! is_integer "${!v}"; then
    err "${v}=${!v} is not an integer."
  fi
done

# Export so train_zero1.sh's existing logic picks them up.
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT

# ---------------------------------------------------------------------------
# 3. Log dir + per-rank log file.
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log.rank${NODE_RANK}"
mkdir -p "${LOG_DIR}"

info "REPO_ROOT       = ${REPO_ROOT}"
info "MODEL           = ${MODEL}"
info "TASK            = ${TASK}"
info "DATA            = ${DATA}"
info "WANDB_NAME      = ${WANDB_NAME}"
info "NNODES          = ${NNODES}"
info "NPROC_PER_NODE  = ${NPROC_PER_NODE}"
info "NODE_RANK       = ${NODE_RANK}"
info "MASTER_ADDR     = ${MASTER_ADDR}"
info "MASTER_PORT     = ${MASTER_PORT}"
info "TOTAL_GPUS      = $(( NNODES * NPROC_PER_NODE ))"
info "RUN_NAME        = ${RUN_NAME}"
info "LOG_FILE        = ${LOG_FILE}"
info "FOREGROUND      = ${FOREGROUND}"
[[ -n "${EXTRA}" ]] && info "EXTRA hydra     = ${EXTRA}"

# ---------------------------------------------------------------------------
# 4. Hydra overrides.
# ---------------------------------------------------------------------------
HYDRA_OVERRIDES=(
  "task=${TASK}"
  "data=${DATA}"
  "model=${MODEL}"
  "output_dir=${LOG_DIR}"
  "wandb.name=${WANDB_NAME}"
)
if [[ -n "${EXTRA}" ]]; then
  # shellcheck disable=SC2206
  HYDRA_OVERRIDES+=( ${EXTRA} )
fi

# Force RUN_ID=RUN_NAME so all ranks share output_dir without TCPStore sync.
COMMON_ENV=(
  "NUM_GPUS=${NPROC_PER_NODE}"
  "RUN_ID=${RUN_NAME}"
  "NNODES=${NNODES}"
  "NODE_RANK=${NODE_RANK}"
  "MASTER_ADDR=${MASTER_ADDR}"
  "MASTER_PORT=${MASTER_PORT}"
)
# Pass through any pre-set env that experiment-specific wrappers may want
# (e.g. DIFFSYNTH_MODEL_BASE_PATH for V-JEPA / Wan checkpoints).
if [[ -n "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
  COMMON_ENV+=( "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}" )
fi
if [[ -n "${DS_CONFIG:-}" ]]; then
  COMMON_ENV+=( "DS_CONFIG=${DS_CONFIG}" )
fi
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  COMMON_ENV+=( "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" )
fi

# ---------------------------------------------------------------------------
# 5. Launch.
#
# Use launch_torchrun_multinode.sh (torchrun direct, per-node rendezvous).
# This avoids accelerate's --deepspeed_multinode_launcher standard which
# tries to SSH into other nodes from rank0 when we already launched there.
# ---------------------------------------------------------------------------
TORCHRUN_LAUNCHER="${SCRIPT_DIR}/launch_torchrun_multinode.sh"

# Pass everything launch_torchrun_multinode.sh needs via env.
export MODEL TASK DATA WANDB_NAME RUN_NAME LOG_ROOT FOREGROUND EXTRA
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
export DS_CONFIG PYTORCH_CUDA_ALLOC_CONF

if [[ "${FOREGROUND}" == "1" ]]; then
  info "Running in foreground (torchrun)."
  bash "${TORCHRUN_LAUNCHER}" 2>&1 | tee "${LOG_FILE}"
else
  # Write a self-contained wrapper script for the detached launcher.
  WRAPPER_SCRIPT="${LOG_DIR}/.launcher_wrapper.rank${NODE_RANK}.sh"
  {
    printf '#!/usr/bin/env bash\ncd %q\n' "${REPO_ROOT}"
    for kv in "${COMMON_ENV[@]}"; do
      key="${kv%%=*}"
      val="${kv#*=}"
      printf 'export %s=%q\n' "${key}" "${val}"
    done
    # Additional vars needed by launch_torchrun_multinode.sh
    printf 'export MODEL=%q\n' "${MODEL}"
    printf 'export TASK=%q\n'  "${TASK}"
    printf 'export DATA=%q\n'  "${DATA}"
    printf 'export WANDB_NAME=%q\n' "${WANDB_NAME}"
    printf 'export RUN_NAME=%q\n'   "${RUN_NAME}"
    printf 'export LOG_ROOT=%q\n'   "${LOG_ROOT}"
    printf 'export FOREGROUND=1\n'   # inside the detached child, run foreground so output goes to LOG_FILE
    [[ -n "${EXTRA}" ]] && printf 'export EXTRA=%q\n' "${EXTRA}"
    printf 'set -x\n'
    printf 'bash %q\n' "${TORCHRUN_LAUNCHER}"
  } > "${WRAPPER_SCRIPT}"
  chmod +x "${WRAPPER_SCRIPT}"

  # Use Python to launch as a fully detached process (survives shell exit).
  PY_LAUNCHER="${REPO_ROOT}/scripts/launch_detached.py"
  PID=$(python3 "${PY_LAUNCHER}" "${LOG_FILE}" "${WRAPPER_SCRIPT}")
  echo "${PID}" > "${LOG_DIR}/.launcher.pid.rank${NODE_RANK}"
  info "Launched in background on rank ${NODE_RANK}, PID=${PID}"
  info "Wrapper: ${WRAPPER_SCRIPT}"
  info "Tail this rank's log:   tail -f ${LOG_FILE}"
  info "Stop this rank's run:   pkill -TERM -f scripts/train.py && pkill -TERM -f torchrun"
  info "(Repeat the kill command on every node to fully stop the job.)"
fi
