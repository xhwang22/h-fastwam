#!/usr/bin/env bash
# Compute fixed V-JEPA 2.1 statistics for the 85/15 multisource mixture.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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

PRETRAIN_DATA_ROOT="${PRETRAIN_DATA_ROOT:-/fsx/pretrain_data}"
export INTERN_A1_ROOT="${INTERN_A1_ROOT:-${PRETRAIN_DATA_ROOT}/InternData-A1}"
export AGIBOT_WORLD_ROOT="${AGIBOT_WORLD_ROOT:-${PRETRAIN_DATA_ROOT}/AgiBot-World-Beta_lerobotv3}"
export DROID_ROOT="${DROID_ROOT:-${PRETRAIN_DATA_ROOT}/DROID_LeRobot-v3.0}"
export OPEN_X_ROOT="${OPEN_X_ROOT:-${PRETRAIN_DATA_ROOT}/Open-X-Embodiment}"
export ROBOCOIN_ROOT="${ROBOCOIN_ROOT:-${PRETRAIN_DATA_ROOT}/RoboCOIN_v3.0_official}"
export GALAXEA_ROOT="${GALAXEA_ROOT:-${PRETRAIN_DATA_ROOT}/Galaxea-Open-World-Dataset-LeRobot-v3.0}"
for variable in \
  INTERN_A1_ROOT \
  AGIBOT_WORLD_ROOT \
  DROID_ROOT \
  OPEN_X_ROOT \
  ROBOCOIN_ROOT \
  GALAXEA_ROOT; do
  path="${!variable}"
  if [[ ! -d "${path}" ]]; then
    echo "[multisource-vjepa21-stats] ERROR: ${variable} does not exist: ${path}" >&2
    exit 1
  fi
done

export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v3}"
export MULTISOURCE_VIDEO_MANIFEST_DIR="${MULTISOURCE_VIDEO_MANIFEST_DIR:-${PRETRAIN_DATA_ROOT}/.fastwam_multisource/video_manifest_v2}"
MULTISOURCE_REGISTRY="${MULTISOURCE_REGISTRY:-configs/data/multisource_robot_v3_registry.yaml}"
python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}"
python scripts/build_multisource_video_manifest.py \
  --registry "${MULTISOURCE_REGISTRY}" \
  --output "${MULTISOURCE_VIDEO_MANIFEST_DIR}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-768}"
export MULTISOURCE_SAMPLES_PER_EPOCH="${MULTISOURCE_SAMPLES_PER_EPOCH:-768000}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
  echo "[multisource-vjepa21-stats] ERROR: checkpoint not found: ${VJEPA21_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
  echo "[multisource-vjepa21-stats] ERROR: source tree not found: ${VJEPA21_REPO}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-16}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-8}"
STATS_PREFETCH_FACTOR="${STATS_PREFETCH_FACTOR:-2}"
STATS_MULTIPROCESSING_CONTEXT="${STATS_MULTIPROCESSING_CONTEXT:-spawn}"
STATS_MASTER_PORT="${VJEPA21_STATS_MASTER_PORT:-29547}"
OUTPUT_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${MULTISOURCE_VIDEO_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_mixture85_15_stats.pt}"
MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES}" != "all" ]]; then
  MAX_SAMPLE_ARGS=(--max-samples "${MAX_SAMPLES}")
fi

exec torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port="${STATS_MASTER_PORT}" \
  scripts/precompute_vjepa21_stats.py \
  --data-config multisource_robot_v3 \
  --output-path "${OUTPUT_PATH}" \
  --checkpoint-path "${VJEPA21_CHECKPOINT}" \
  --repo-path "${VJEPA21_REPO}" \
  --model-name vjepa2_1_vit_gigantic_384 \
  "${MAX_SAMPLE_ARGS[@]}" \
  --batch-size "${STATS_BATCH_SIZE}" \
  --num-workers "${STATS_NUM_WORKERS}" \
  --prefetch-factor "${STATS_PREFETCH_FACTOR}" \
  --multiprocessing-context "${STATS_MULTIPROCESSING_CONTEXT}" \
  --temporal-downsample 4 \
  --causal-tubelet-encoding \
  "$@"
