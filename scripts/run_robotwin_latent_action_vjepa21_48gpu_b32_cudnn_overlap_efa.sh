#!/usr/bin/env bash
# Build DreamDojo caches with all 48 GPUs, then train at global batch 1536.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DREAMDOJO_ROOT="${DREAMDOJO_ROOT:-${REPO_ROOT}/external/DreamDojo}"
export DREAMDOJO_CHECKPOINT="${DREAMDOJO_CHECKPOINT:-${DREAMDOJO_ROOT}/checkpoints/DreamDojo/LAM_400k.ckpt}"
readonly DREAMDOJO_SOURCE_REVISION="02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"
readonly DREAMDOJO_CHECKPOINT_SHA256="d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f"
if [[ ! -f "${DREAMDOJO_ROOT}/external/lam/modules/lam.py" ]]; then
  echo "ERROR: DreamDojo LAM source not found under ${DREAMDOJO_ROOT}." >&2
  exit 1
fi
if [[ ! -f "${DREAMDOJO_CHECKPOINT}" ]]; then
  echo "ERROR: DreamDojo checkpoint not found: ${DREAMDOJO_CHECKPOINT}" >&2
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FASTWAM_EXPECTED_WORLD_SIZE=48
export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
export LATENT_ACTION_CACHE_ROOT="${LATENT_ACTION_CACHE_ROOT:-/efs/shaunxhwang/robotwin2.0_dreamdojo_latent_action_cache}"
export LATENT_ACTION_TRAIN_CACHE_DIR="${LATENT_ACTION_CACHE_ROOT}/train"
export LATENT_ACTION_VAL_CACHE_DIR="${LATENT_ACTION_CACHE_ROOT}/val"
export LATENT_ACTION_CACHE_SHARD_SIZE="${LATENT_ACTION_CACHE_SHARD_SIZE:-256}"
export LATENT_ACTION_SAMPLE_BATCH_SIZE="${LATENT_ACTION_SAMPLE_BATCH_SIZE:-24}"
export LATENT_ACTION_PAIR_BATCH_SIZE="${LATENT_ACTION_PAIR_BATCH_SIZE:-768}"
export LATENT_ACTION_CACHE_NUM_WORKERS="${LATENT_ACTION_CACHE_NUM_WORKERS:-8}"
export LATENT_ACTION_CACHE_PREFETCH_FACTOR="${LATENT_ACTION_CACHE_PREFETCH_FACTOR:-1}"
export LATENT_ACTION_CACHE_MP_CONTEXT="${LATENT_ACTION_CACHE_MP_CONTEXT:-spawn}"
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "ERROR: RoboTwin WebDataset is incomplete: ${ROBOTWIN_WEBDATASET_ROOT}" >&2
  exit 1
fi
for arg in "$@"; do
  normalized_arg="${arg}"
  while [[ "${normalized_arg}" == +* ]]; do
    normalized_arg="${normalized_arg#+}"
  done
  case "${normalized_arg}" in
    seed=*|data=*|data.*)
      echo "ERROR: dataset/seed Hydra overrides are not supported by this cache-first launcher: ${arg}" >&2
      echo "Set ROBOTWIN_WEBDATASET_ROOT or create a separate cache/run name instead." >&2
      exit 2
      ;;
  esac
done

# Resolve the six-node topology before starting the distributed cache phases.
# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod_runtime

mkdir -p "${LATENT_ACTION_CACHE_ROOT}/logs"
CACHE_TRAIN_PORT="${LATENT_ACTION_CACHE_TRAIN_PORT:-$((MASTER_PORT + 10))}"
CACHE_VAL_PORT="${LATENT_ACTION_CACHE_VAL_PORT:-$((MASTER_PORT + 11))}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: no Python interpreter found; set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi
if ! "${PYTHON_BIN}" -c "import torch, hydra" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} cannot import torch and hydra; set PYTHON_BIN to the FastWAM environment interpreter." >&2
  exit 1
fi

run_cache_phase() {
  local split="$1"
  local cache_dir="$2"
  local port="$3"
  local log_file="${LATENT_ACTION_CACHE_ROOT}/logs/${split}.rank${NODE_RANK}.log"
  local -a normalization_args=()
  if [[ "${split}" == "val" ]]; then
    normalization_args=(
      --normalization-stats-cache "${LATENT_ACTION_TRAIN_CACHE_DIR}"
    )
  fi
  echo "[latent-cache] split=${split} world=$((NNODES * NPROC_PER_NODE)) cache=${cache_dir}"
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${port}" \
    scripts/precompute_dreamdojo_latent_actions.py \
    --data-config robotwin_interleaved_webdataset \
    --split "${split}" \
    --cache-dir "${cache_dir}" \
    --dreamdojo-root "${DREAMDOJO_ROOT}" \
    --checkpoint "${DREAMDOJO_CHECKPOINT}" \
    --dreamdojo-source-revision "${DREAMDOJO_SOURCE_REVISION}" \
    --checkpoint-sha256 "${DREAMDOJO_CHECKPOINT_SHA256}" \
    --shard-size "${LATENT_ACTION_CACHE_SHARD_SIZE}" \
    --sample-batch-size "${LATENT_ACTION_SAMPLE_BATCH_SIZE}" \
    --pair-batch-size "${LATENT_ACTION_PAIR_BATCH_SIZE}" \
    --num-workers "${LATENT_ACTION_CACHE_NUM_WORKERS}" \
    --prefetch-factor "${LATENT_ACTION_CACHE_PREFETCH_FACTOR}" \
    --multiprocessing-context "${LATENT_ACTION_CACHE_MP_CONTEXT}" \
    --cache-dtype bfloat16 \
    "${normalization_args[@]}" \
    "data.train.preprocessed_root=${ROBOTWIN_WEBDATASET_ROOT}" \
    "data.val.preprocessed_root=${ROBOTWIN_WEBDATASET_ROOT}" \
    2>&1 | tee -a "${log_file}"
}

run_cache_phase train "${LATENT_ACTION_TRAIN_CACHE_DIR}" "${CACHE_TRAIN_PORT}"
run_cache_phase val "${LATENT_ACTION_VAL_CACHE_DIR}" "${CACHE_VAL_PORT}"
test -f "${LATENT_ACTION_TRAIN_CACHE_DIR}/manifest.json"
test -f "${LATENT_ACTION_VAL_CACHE_DIR}/manifest.json"

export PER_GPU_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=$(( \
  PER_GPU_BATCH_SIZE \
  * FASTWAM_EXPECTED_WORLD_SIZE \
  * GRADIENT_ACCUMULATION_STEPS \
))
export MODEL_CONFIG=hfastwam_latent_action_vjepa21
export DATA_CONFIG=robotwin_latent_action_interleaved_webdataset
export STANDARDISE_OUTPUT=true
export VIDEO_LATENT_CACHE_ENABLED=0
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_latent_action_vjepa21_48gpu_b32_cudnn_overlap_efa}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" \
  "$@"
