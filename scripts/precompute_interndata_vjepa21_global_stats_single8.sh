#!/usr/bin/env bash
# Compute fixed V-JEPA 2.1 statistics on InternData-A1 with one 8-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="/apdcephfs_csgl/share_306089109/shaunxhwang/miniconda3/bin/activate"
if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" fastwam
fi

export INTERN_A1_ROOT="${INTERN_A1_ROOT:-/fsx/pretrain_data/InternData-A1}"
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-${INTERN_A1_ROOT}/.fastwam_intern_a1/manifest_v3}"
if [[ ! -d "${INTERN_A1_ROOT}" ]]; then
  echo "[vjepa21-stats] ERROR: InternData root does not exist: ${INTERN_A1_ROOT}" >&2
  exit 1
fi

python scripts/build_interndata_a1_manifest.py \
  --root "${INTERN_A1_ROOT}" \
  --output "${INTERN_A1_MANIFEST_DIR}"

export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/checkpoints/torch_hub}"
VJEPA21_CHECKPOINT="${VJEPA21_CHECKPOINT:-${TORCH_HOME}/hub/checkpoints/vjepa2_1_vitG_384.pt}"
VJEPA21_REPO="${VJEPA21_REPO:-${TORCH_HOME}/hub/facebookresearch_vjepa2_main}"
if [[ ! -f "${VJEPA21_CHECKPOINT}" ]]; then
  echo "[vjepa21-stats] ERROR: checkpoint not found: ${VJEPA21_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA21_REPO}/app/vjepa_2_1/models/vision_transformer.py" ]]; then
  echo "[vjepa21-stats] ERROR: source tree not found: ${VJEPA21_REPO}" >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-2}"
TEMPORAL_DOWNSAMPLE="${TEMPORAL_DOWNSAMPLE:-4}"
OUTPUT_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${INTERN_A1_MANIFEST_DIR}/vjepa21_vitG_causal_tubelet_global_stats.pt}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/precompute_vjepa21_stats.py \
  --data-config interndata_a1_v3 \
  --output-path "${OUTPUT_PATH}" \
  --checkpoint-path "${VJEPA21_CHECKPOINT}" \
  --repo-path "${VJEPA21_REPO}" \
  --model-name vjepa2_1_vit_gigantic_384 \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${STATS_BATCH_SIZE}" \
  --temporal-downsample "${TEMPORAL_DOWNSAMPLE}" \
  --causal-tubelet-encoding \
  "$@"
