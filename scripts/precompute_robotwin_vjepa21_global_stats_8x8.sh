#!/usr/bin/env bash
# Compute fixed V-JEPA 2.1 statistics on RoboTwin with 8 HyperPod/PET nodes.
set -euo pipefail

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PET_NNODES:-}" ]]; then
  echo "[robotwin-vjepa21-stats-8x8] ERROR: launch this script through HyperPod/PET." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=64
# Each node runs independently and exchanges results through the shared FS.
export FASTWAM_USE_EFA=0

# HyperPod/PET provides PET_NNODES, PET_NODE_RANK, PET_MASTER_ADDR, and
# optionally PET_MASTER_PORT. The helper maps them to torchrun topology.
# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod_runtime

if (( NNODES != 8 || NPROC_PER_NODE != 8 )); then
  echo "[robotwin-vjepa21-stats-8x8] ERROR: expected 8 nodes x 8 GPUs, got ${NNODES} x ${NPROC_PER_NODE}." >&2
  exit 1
fi

# shellcheck source=_robotwin_data_source.sh
source "${SCRIPT_DIR}/_robotwin_data_source.sh"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
fastwam_select_robotwin_data_source

export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
  echo "[robotwin-vjepa21-stats-8x8] ERROR: checkpoint not found: ${VJEPA21_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
  echo "[robotwin-vjepa21-stats-8x8] ERROR: source tree not found: ${VJEPA21_REPO}" >&2
  exit 1
fi

STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-16}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-8}"
STATS_PREFETCH_FACTOR="${STATS_PREFETCH_FACTOR:-2}"
STATS_MULTIPROCESSING_CONTEXT="${STATS_MULTIPROCESSING_CONTEXT:-spawn}"
TEMPORAL_DOWNSAMPLE="${TEMPORAL_DOWNSAMPLE:-4}"
OUTPUT_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
STATS_SHARD_RUN_NAME="${STATS_SHARD_RUN_NAME:-${MAX_SAMPLES:-all}}"
if [[ ! "${STATS_SHARD_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[robotwin-vjepa21-stats-8x8] ERROR: STATS_SHARD_RUN_NAME contains unsupported characters." >&2
  exit 1
fi
SHARD_DIR="${STATS_SHARD_DIR:-${OUTPUT_PATH}.shards/${STATS_SHARD_RUN_NAME}}"
STATS_MERGE_TIMEOUT="${STATS_MERGE_TIMEOUT:-7200}"
STATS_LOCAL_MASTER_PORT="${VJEPA21_STATS_LOCAL_MASTER_PORT:-29547}"
MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES}" != "all" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

DATA_OVERRIDE_ARGS=()
for override in "${ROBOTWIN_DATA_OVERRIDES[@]}"; do
  DATA_OVERRIDE_ARGS+=(--data-override "${override}")
done
DATA_OVERRIDE_ARGS+=(--data-override "data.train.num_segments=1")

GLOBAL_RANK_OFFSET=$(( NODE_RANK * NPROC_PER_NODE ))
echo "[robotwin-vjepa21-stats-8x8] node_rank=${NODE_RANK}/${NNODES} gpus_per_node=${NPROC_PER_NODE} rank_offset=${GLOBAL_RANK_OFFSET} local_rdzv=127.0.0.1:${STATS_LOCAL_MASTER_PORT}"
echo "[robotwin-vjepa21-stats-8x8] output=${OUTPUT_PATH}"
echo "[robotwin-vjepa21-stats-8x8] shards=${SHARD_DIR}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 \
  --master_port="${STATS_LOCAL_MASTER_PORT}" \
  scripts/precompute_vjepa21_stats.py \
  --data-config "${ROBOTWIN_DATA_CONFIG}" \
  --output-path "${OUTPUT_PATH}" \
  --checkpoint-path "${VJEPA21_CHECKPOINT}" \
  --repo-path "${VJEPA21_REPO}" \
  --model-name vjepa2_1_vit_gigantic_384 \
  "${MAX_SAMPLE_ARGS[@]}" \
  --batch-size "${STATS_BATCH_SIZE}" \
  --num-workers "${STATS_NUM_WORKERS}" \
  --prefetch-factor "${STATS_PREFETCH_FACTOR}" \
  --multiprocessing-context "${STATS_MULTIPROCESSING_CONTEXT}" \
  --temporal-downsample "${TEMPORAL_DOWNSAMPLE}" \
  --causal-tubelet-encoding \
  --shard-output-dir "${SHARD_DIR}" \
  --shard-rank-offset "${GLOBAL_RANK_OFFSET}" \
  --shard-world-size 64 \
  --resume-shards \
  "${DATA_OVERRIDE_ARGS[@]}" \
  "$@"

if (( NODE_RANK == 0 )); then
  python scripts/merge_vjepa21_stats_shards.py \
    --shard-dir "${SHARD_DIR}" \
    --output-path "${OUTPUT_PATH}" \
    --expected-world-size 64 \
    --wait-timeout "${STATS_MERGE_TIMEOUT}"
else
  echo "[robotwin-vjepa21-stats-8x8] node ${NODE_RANK} completed its local shards."
fi
