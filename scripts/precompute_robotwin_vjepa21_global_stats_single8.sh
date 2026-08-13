#!/usr/bin/env bash
# Compute fixed V-JEPA 2.1 statistics on the RoboTwin training distribution.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# This is intentionally a one-node job. HyperPod/PET and outer torchrun
# environments must not leak their multi-node rendezvous into the inner run.
unset PET_NNODES PET_NODE_RANK PET_MASTER_ADDR PET_MASTER_PORT PET_NPROC_PER_NODE
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK
unset GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT NODE_IP_LIST
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
unset FASTWAM_MANAGED_DISTRIBUTED _MULTINODE_LAUNCHED

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

# shellcheck source=_robotwin_data_source.sh
source "${SCRIPT_DIR}/_robotwin_data_source.sh"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-data}"
fastwam_select_robotwin_data_source

export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
  echo "[robotwin-vjepa21-stats] ERROR: checkpoint not found: ${VJEPA21_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
  echo "[robotwin-vjepa21-stats] ERROR: source tree not found: ${VJEPA21_REPO}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE=8
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-16}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-8}"
STATS_PREFETCH_FACTOR="${STATS_PREFETCH_FACTOR:-2}"
STATS_MULTIPROCESSING_CONTEXT="${STATS_MULTIPROCESSING_CONTEXT:-spawn}"
TEMPORAL_DOWNSAMPLE="${TEMPORAL_DOWNSAMPLE:-4}"
OUTPUT_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"
STATS_MASTER_PORT="${VJEPA21_STATS_MASTER_PORT:-29547}"
MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES}" != "all" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

DATA_OVERRIDE_ARGS=()
for override in "${ROBOTWIN_DATA_OVERRIDES[@]}"; do
  DATA_OVERRIDE_ARGS+=(--data-override "${override}")
done
DATA_OVERRIDE_ARGS+=(--data-override "data.train.num_segments=1")

exec torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 \
  --master_port="${STATS_MASTER_PORT}" \
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
  "${DATA_OVERRIDE_ARGS[@]}" \
  "$@"
