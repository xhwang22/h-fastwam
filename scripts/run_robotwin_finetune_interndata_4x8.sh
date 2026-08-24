#!/usr/bin/env bash
set -euo pipefail

cd /fsx/h-fastwam

# AWS distributed topology: 4 nodes x 8 GPUs.
JOB_NAME="${JOB_NAME:-finetune-robotwin}"
EXPECTED_NNODES=4
EXPECTED_GPUS_PER_NODE=8

# Support both HyperPod PET and Kubernetes Indexed Job.
if [[ -n "${PET_NNODES:-}" ]]; then
  [[ "${PET_NNODES}" == "${EXPECTED_NNODES}" ]] || {
    echo "ERROR: expected ${EXPECTED_NNODES} nodes, got PET_NNODES=${PET_NNODES}" >&2
    exit 1
  }

  export NNODES="${PET_NNODES}"
  export NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${EXPECTED_GPUS_PER_NODE}}"
  if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    export NPROC_PER_NODE="${EXPECTED_GPUS_PER_NODE}"
  fi
  export NODE_RANK="${PET_NODE_RANK:?PET_NODE_RANK is required}"
  export MASTER_ADDR="${PET_MASTER_ADDR:?PET_MASTER_ADDR is required}"
  export MASTER_PORT="${PET_MASTER_PORT:-29500}"
else
  export NNODES="${NNODES:-${EXPECTED_NNODES}}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-${EXPECTED_GPUS_PER_NODE}}"
  if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    export NPROC_PER_NODE="${EXPECTED_GPUS_PER_NODE}"
  fi
  export NODE_RANK="${NODE_RANK:-${JOB_COMPLETION_INDEX:?set JOB_COMPLETION_INDEX or NODE_RANK}}"
  export MASTER_ADDR="${MASTER_ADDR:-${JOB_NAME}-0.${JOB_NAME}}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi

[[ "${NNODES}" == "${EXPECTED_NNODES}" ]] || {
  echo "ERROR: expected ${EXPECTED_NNODES} nodes, got NNODES=${NNODES}" >&2
  exit 1
}

[[ "${NPROC_PER_NODE}" == "${EXPECTED_GPUS_PER_NODE}" ]] || {
  echo "ERROR: expected ${EXPECTED_GPUS_PER_NODE} GPUs per node, got NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
}

export FASTWAM_EXPECTED_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "Waiting for master DNS: ${MASTER_ADDR}"
until getent hosts "${MASTER_ADDR}" >/dev/null 2>&1; do
  sleep 1
done

echo "Topology: node_rank=${NODE_RANK}/${NNODES}, GPUs/node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"

# Prepare RoboTwin data, dependencies, and V-JEPA assets.
# Expected layout:
# /fsx/h-fastwam/data/robotwin2.0/
# ├── dataset_stats.json
# └── robotwin2.0/meta/info.json
export ROBOTWIN_ASSET_ROOT="${ROBOTWIN_ASSET_ROOT:-/fsx/h-fastwam/data/robotwin2.0}"

ROBOTWIN_DATA_ROOT="${ROBOTWIN_ASSET_ROOT}" \
INSTALL_PYTHON_DEPS="${INSTALL_PYTHON_DEPS:-1}" \
START_TRAINING=0 \
bash scripts/setup_robotwin_vjepa21_predictor_training.sh

# Select the latest InternData pretraining checkpoint unless explicitly set.
export SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/fsx/h-fastwam/runs/robotwin_hfastwam/interndata_a1_30hz_localnorm_8x8_b48}"

if [[ -z "${SOURCE_CHECKPOINT:-}" ]]; then
  shopt -s nullglob
  source_checkpoints=(
    "${SOURCE_RUN_DIR}"/checkpoints/weights/step_*.pt
  )
  shopt -u nullglob

  if (( ${#source_checkpoints[@]} == 0 )); then
    echo "ERROR: no pretraining weights found under:" >&2
    echo "  ${SOURCE_RUN_DIR}/checkpoints/weights" >&2
    exit 1
  fi

  SOURCE_CHECKPOINT="${source_checkpoints[$((${#source_checkpoints[@]} - 1))]}"
fi

[[ -f "${SOURCE_CHECKPOINT}" ]] || {
  echo "ERROR: source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
}

export SOURCE_CHECKPOINT

# Convert the InternData 20D checkpoint to a RoboTwin 14D transfer checkpoint.
source_step="$(basename "${SOURCE_CHECKPOINT}" .pt)"
export TRANSFER_CHECKPOINT="${TRANSFER_CHECKPOINT:-/fsx/h-fastwam/checkpoints/interndata_a1_${source_step}_robotwin14d_transfer.pt}"
ready_marker="${TRANSFER_CHECKPOINT}.ready"

# Only rank 0 performs the large conversion.
if [[ "${NODE_RANK}" == "0" ]]; then
  marker_source=""
  if [[ -f "${ready_marker}" ]]; then
    marker_source="$(<"${ready_marker}")"
  fi

  if [[ ! -f "${TRANSFER_CHECKPOINT}" || "${marker_source}" != "${SOURCE_CHECKPOINT}" ]]; then
    echo "Converting InternData checkpoint:"
    echo "  input:  ${SOURCE_CHECKPOINT}"
    echo "  output: ${TRANSFER_CHECKPOINT}"

    tmp_checkpoint="${TRANSFER_CHECKPOINT}.tmp.$$"
    tmp_marker="${ready_marker}.tmp.$$"
    trap 'rm -f "${tmp_checkpoint}" "${tmp_marker}"' EXIT

    python scripts/prepare_interndata_checkpoint_for_robotwin.py \
      --input "${SOURCE_CHECKPOINT}" \
      --output "${tmp_checkpoint}"

    # Publish atomically so other nodes never read a partial checkpoint.
    mv -f "${tmp_checkpoint}" "${TRANSFER_CHECKPOINT}"
    printf '%s\n' "${SOURCE_CHECKPOINT}" > "${tmp_marker}"
    mv -f "${tmp_marker}" "${ready_marker}"
    trap - EXIT
  else
    echo "Transfer checkpoint already ready: ${TRANSFER_CHECKPOINT}"
  fi
else
  echo "Waiting for rank 0 to prepare transfer checkpoint..."
  until [[ -f "${TRANSFER_CHECKPOINT}" && -f "${ready_marker}" ]] && \
        [[ "$(<"${ready_marker}")" == "${SOURCE_CHECKPOINT}" ]]; do
    sleep 5
  done
fi

export FASTWAM_CHECKPOINT="${TRANSFER_CHECKPOINT}"

# The downstream data selector appends robotwin2.0/robotwin2.0.
export ROBOTWIN_DATA_ROOT="$(dirname "${ROBOTWIN_ASSET_ROOT}")"
export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"

# This transfer run must use the indexed WebDataset. Do not silently fall back
# to random AV1 MP4 decoding when preprocessing is incomplete.
if [[ ! -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
  echo "ERROR: completed RoboTwin WebDataset not found: ${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" >&2
  echo "Wait for preprocessing to finish, or set ROBOTWIN_WEBDATASET_ROOT to a completed dataset." >&2
  exit 1
fi

# Match the InternData pretraining V-JEPA configuration.
unset VJEPA21_NORMALISE_STATS_PATH
export STANDARDISE_OUTPUT=true

export TASK_CONFIG=robotwin_uncond_3cam_384_1e-4
export MODEL_CONFIG=hfastwam_small_vjepa21_predictor
export USE_ROBOTWIN_DATA_OVERRIDES=1
export SET_NUM_SEGMENTS=1

# 32 GPUs x per-GPU batch 48 x accumulation 1 = global batch 1536.
export GRADIENT_ACCUMULATION_STEPS=1
export GLOBAL_BATCH_SIZE=$((48 * FASTWAM_EXPECTED_WORLD_SIZE))

export NUM_EPOCHS="${NUM_EPOCHS:-5}"
export MAX_STEPS="${MAX_STEPS:-null}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATALOADER_PERSISTENT_WORKERS=true

export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-1}"
export FASTWAM_KEEP_LAST_CKPT="${FASTWAM_KEEP_LAST_CKPT:-3}"

# FRESH=0 resumes an existing RoboTwin run, or initializes a new run from
# FASTWAM_CHECKPOINT when RUN_NAME has no training state. Use FRESH=1 to ignore
# existing RoboTwin state for the same RUN_NAME.
export FRESH="${FRESH:-0}"

# DeepSpeed, cuDNN SDPA, and EFA.
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA="${FASTWAM_USE_EFA:-1}"
export FASTWAM_DISABLE_PROXY="${FASTWAM_DISABLE_PROXY:-1}"

# W&B. Inject the key via the environment or a mounted secret file.
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin}"
export WANDB_GROUP="${WANDB_GROUP:-interndata-a1-transfer}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/fsx/.secrets/wandb_api_key}"

if [[ "${WANDB}" == "1" && -z "${WANDB_API_KEY:-}" && \
      ! -s "${WANDB_API_KEY_FILE}" && ! -s "${HOME}/.wandb_key" ]]; then
  echo "ERROR: inject WANDB_API_KEY or mount it at ${WANDB_API_KEY_FILE}." >&2
  echo "Do not hard-code the W&B key in this script." >&2
  exit 1
fi

export RUN_NAME="${RUN_NAME:-robotwin_ft_interndata_${source_step}_4x8_b48}"

# Keep Hugging Face's generated Arrow cache on node-local storage. Local rank 0
# builds it once per node; the other seven ranks wait and then reuse it.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/fastwam_hf_datasets/${RUN_NAME}}"
export FASTWAM_LOCAL_DATASET_CACHE_WARMUP=1
export FASTWAM_DATASET_WARMUP_ID="${FASTWAM_DATASET_WARMUP_ID:-launch-$$}"
export FASTWAM_DATASET_WARMUP_TIMEOUT="${FASTWAM_DATASET_WARMUP_TIMEOUT:-7200}"
mkdir -p "${HF_DATASETS_CACHE}"

echo
echo "Source checkpoint:   ${SOURCE_CHECKPOINT}"
echo "Transfer checkpoint: ${FASTWAM_CHECKPOINT}"
echo "Output directory:    /fsx/h-fastwam/runs/robotwin_hfastwam/${RUN_NAME}"
echo "HF datasets cache:   ${HF_DATASETS_CACHE}"
echo "Global batch:        ${GLOBAL_BATCH_SIZE}"
echo "Batch calculation:   ${FASTWAM_EXPECTED_WORLD_SIZE} GPUs x 48 x accum ${GRADIENT_ACCUMULATION_STEPS}"
echo

exec bash scripts/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh \
  "++model.language_pad_to_max_length=true" \
  "$@"
