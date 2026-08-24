#!/usr/bin/env bash
# Continue the same trained model for one epoch with the online encoder on/off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${REPO_ROOT}/data}"

MODE="${1:-}"
if [[ "${MODE}" != "on" && "${MODE}" != "off" ]]; then
  echo "Usage: bash $0 {on|off}" >&2
  exit 2
fi
shift
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: this A/B launcher does not accept positional Hydra overrides; use environment variables so every node receives identical settings." >&2
  exit 2
fi

export PET_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-8}"
if [[ -z "${PET_NNODES:-}" || -z "${PET_NODE_RANK:-}" || -z "${PET_MASTER_ADDR:-}" ]]; then
  echo "ERROR: this launcher requires a managed PET launch with PET_NNODES, PET_NODE_RANK, and PET_MASTER_ADDR set." >&2
  exit 2
fi
if [[ "${PET_NNODES}" != "2" ]]; then
  echo "ERROR: expected PET_NNODES=2, got ${PET_NNODES}." >&2
  exit 2
fi
if [[ "${PET_NPROC_PER_NODE}" != "8" && "${PET_NPROC_PER_NODE}" != "auto" ]]; then
  echo "ERROR: expected PET_NPROC_PER_NODE=8 or auto, got ${PET_NPROC_PER_NODE}." >&2
  exit 2
fi
export FASTWAM_EXPECTED_WORLD_SIZE=16

export FASTWAM_CHECKPOINT="${FASTWAM_CHECKPOINT:-/fsx/h-fastwam/runs/robotwin_hfastwam/robotwin_vjepa21_predictor_causal_tubelet_32gpu_b48_cudnn_overlap_efa/checkpoints/weights/step_019570.pt}"
if [[ ! -f "${FASTWAM_CHECKPOINT}" ]]; then
  echo "ERROR: FASTWAM_CHECKPOINT not found: ${FASTWAM_CHECKPOINT}" >&2
  exit 2
fi
export FASTWAM_CHECKPOINT_STRICT=true

export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
DATASET_DONE="${ROBOTWIN_WEBDATASET_ROOT}/dataset.done"
DATASET_MANIFEST="${ROBOTWIN_WEBDATASET_ROOT}/manifest.json"
if [[ ! -s "${DATASET_DONE}" || "$(tr -d '[:space:]' < "${DATASET_DONE}")" != "complete" ]]; then
  echo "ERROR: completed WebDataset marker is missing or invalid: ${DATASET_DONE}" >&2
  exit 2
fi
if [[ ! -s "${DATASET_MANIFEST}" ]]; then
  echo "ERROR: WebDataset manifest is missing or empty: ${DATASET_MANIFEST}" >&2
  exit 2
fi
python - "${DATASET_MANIFEST}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"ERROR: invalid WebDataset manifest {path}: {exc}")
if manifest.get("format") != "robotwin-webdataset" or manifest.get("version") != 1:
    raise SystemExit(
        f"ERROR: unsupported WebDataset manifest format/version in {path}: "
        f"{manifest.get('format')!r}/{manifest.get('version')!r}"
    )
if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
    raise SystemExit(f"ERROR: WebDataset manifest has no shards: {path}")
if int(manifest.get("converted_episodes", 0)) <= 0:
    raise SystemExit(f"ERROR: WebDataset manifest has no converted episodes: {path}")
PY

export DATA_CONFIG=robotwin_interleaved_webdataset
export USE_ROBOTWIN_DATA_OVERRIDES=1
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_EPOCHS=1
export MAX_STEPS=null
export FRESH=1
unset RESUME
export VIDEO_LATENT_CACHE_ENABLED=0
export NO_CKPT="${NO_CKPT:-0}"
export FASTWAM_ADAM_FUSED=1
export FASTWAM_PROFILE_STEPS="${FASTWAM_PROFILE_STEPS:-0}"
export VJEPA21_NORMALISE_STATS_PATH="${VJEPA21_NORMALISE_STATS_PATH:-${ROBOTWIN_DATA_ROOT}/robotwin2.0/vjepa21_vitG_causal_tubelet_global_stats.pt}"

if [[ "${MODE}" == "on" ]]; then
  export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_continue_1epoch_encoder_on_2x8_b48_acc2}"
  export FIXED_TARGET_ENCODER=true
  export VISUAL_ENCODER_FREEZE_BACKBONE=false
  export VISUAL_ENCODER_ACTIVATION_CHECKPOINTING=true
  export TRAINABLE_COMPONENTS='[dit,visual_encoder]'
  export VISUAL_ENCODER_LR_MULTIPLIER="${VISUAL_ENCODER_LR_MULTIPLIER:-0.1}"
else
  export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_continue_1epoch_encoder_off_2x8_b48_acc2}"
  export FIXED_TARGET_ENCODER=false
  export VISUAL_ENCODER_FREEZE_BACKBONE=true
  export VISUAL_ENCODER_ACTIVATION_CHECKPOINTING=false
  export TRAINABLE_COMPONENTS='[dit]'
  export VISUAL_ENCODER_LR_MULTIPLIER=1.0
fi

TARGET_OUTPUT_DIR="${REPO_ROOT}/runs/robotwin_hfastwam/${RUN_NAME}"
shopt -s nullglob
EXISTING_CHECKPOINTS=(
  "${TARGET_OUTPUT_DIR}/checkpoints/weights/"*
  "${TARGET_OUTPUT_DIR}/checkpoints/state/"*
)
shopt -u nullglob
if (( ${#EXISTING_CHECKPOINTS[@]} > 0 )); then
  echo "ERROR: target output directory already contains checkpoints: ${TARGET_OUTPUT_DIR}/checkpoints" >&2
  exit 2
fi

echo "[encoder-ab] mode=${MODE} run=${RUN_NAME} topology=2x8 world_size=16 global_batch=1536 grad_accum=2 micro_batch=48"
echo "[encoder-ab] checkpoint=${FASTWAM_CHECKPOINT} dataset=${ROBOTWIN_WEBDATASET_ROOT} fresh=1 no_ckpt=${NO_CKPT}"

exec bash \
  "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh"
