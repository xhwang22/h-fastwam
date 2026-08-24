#!/usr/bin/env bash
# 48-GPU latent-action training: 6 nodes x 8 GPUs, per-GPU batch 32.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -ne 0 ]]; then
  echo "ERROR: this fixed launcher does not accept positional Hydra overrides; use its documented environment variables." >&2
  exit 2
fi

export PET_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-8}"
if [[ -z "${PET_NNODES:-}" || -z "${PET_NODE_RANK:-}" || -z "${PET_MASTER_ADDR:-}" ]]; then
  echo "ERROR: this launcher requires PET_NNODES, PET_NODE_RANK, and PET_MASTER_ADDR." >&2
  exit 2
fi
if [[ "${PET_NNODES}" != "6" ]]; then
  echo "ERROR: expected PET_NNODES=6, got ${PET_NNODES}." >&2
  exit 2
fi
if [[ "${PET_NPROC_PER_NODE}" != "8" ]]; then
  echo "ERROR: expected PET_NPROC_PER_NODE=8, got ${PET_NPROC_PER_NODE}." >&2
  exit 2
fi
if [[ ! "${PET_NODE_RANK}" =~ ^[0-5]$ ]]; then
  echo "ERROR: expected PET_NODE_RANK in [0,5], got ${PET_NODE_RANK}." >&2
  exit 2
fi

LATENT_ACTION_CACHE_ROOT="${LATENT_ACTION_CACHE_ROOT:-/efs/shaunxhwang/robotwin2.0_latent_action_cache}"
LATENT_ACTION_CACHE_ROOT="${LATENT_ACTION_CACHE_ROOT%/}"
if [[ ! -d "${LATENT_ACTION_CACHE_ROOT}" ]]; then
  echo "ERROR: latent-action cache root does not exist: ${LATENT_ACTION_CACHE_ROOT}" >&2
  exit 2
fi
LATENT_ACTION_CACHE_SIGNATURE="${LATENT_ACTION_CACHE_SIGNATURE:?set LATENT_ACTION_CACHE_SIGNATURE to the expected 64-character SHA256 signature}"
if [[ ! "${LATENT_ACTION_CACHE_SIGNATURE}" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "ERROR: LATENT_ACTION_CACHE_SIGNATURE must be a 64-character SHA256 hex digest." >&2
  exit 2
fi
LATENT_ACTION_CACHE_SIGNATURE="${LATENT_ACTION_CACHE_SIGNATURE,,}"

python - \
  "${REPO_ROOT}" \
  "${LATENT_ACTION_CACHE_SIGNATURE}" \
  "${LATENT_ACTION_CACHE_ROOT}/train" \
  "${LATENT_ACTION_CACHE_ROOT}/val" <<'PY'
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))
from fastwam.utils.latent_action_cache import load_latent_action_cache_manifest

expected_signature = sys.argv[2]
expected_dreamdojo = {
    "git_revision": "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff",
    "checkpoint_revision": "89d029e10816d2995d700cb8ba06f171e0504203",
    "checkpoint_sha256": "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f",
}
manifests = []
for split, raw_cache in zip(("train", "val"), sys.argv[3:]):
    cache = pathlib.Path(raw_cache)
    try:
        manifest = load_latent_action_cache_manifest(
            cache,
            expected_signature=expected_signature,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: invalid {split} latent-action cache {cache}: {exc}") from exc
    if manifest.get("split") != split:
        raise SystemExit(
            f"ERROR: latent-action cache split mismatch: expected {split}, "
            f"got {manifest.get('split')!r}"
        )
    dreamdojo = manifest["signature_payload"].get("dreamdojo", {})
    for key, value in expected_dreamdojo.items():
        if dreamdojo.get(key) != value:
            raise SystemExit(
                f"ERROR: {split} DreamDojo provenance {key} mismatch: "
                f"expected {value!r}, got {dreamdojo.get(key)!r}"
            )
    manifests.append(manifest)
if manifests[0]["normalization"] != manifests[1]["normalization"]:
    raise SystemExit("ERROR: train/val latent-action normalization differs")
PY
export LATENT_ACTION_CACHE_ROOT LATENT_ACTION_CACHE_SIGNATURE
export ROBOTWIN_LATENT_ACTION_CACHE_ROOT="${LATENT_ACTION_CACHE_ROOT}"
export ROBOTWIN_LATENT_ACTION_CACHE_SIGNATURE="${LATENT_ACTION_CACHE_SIGNATURE}"

export ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-/efs/shaunxhwang/robotwin2.0_webdataset}"
if [[ ! -s "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" || ! -s "${ROBOTWIN_WEBDATASET_ROOT}/manifest.json" ]]; then
  echo "ERROR: completed RoboTwin WebDataset not found at ${ROBOTWIN_WEBDATASET_ROOT}." >&2
  exit 2
fi

export FASTWAM_EXPECTED_WORLD_SIZE=48
export GLOBAL_BATCH_SIZE=1536
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_EPOCHS=5
export MAX_STEPS=null
export MODEL_CONFIG=hfastwam_small_vjepa21_predictor_latent_action
export DATA_CONFIG=robotwin_latent_action_webdataset
export TRAINABLE_COMPONENTS='[dit,latent_action_decoder]'
export DETACH_VIDEO=false
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_latent_action_causal_tubelet_48gpu_b32_cudnn_overlap_efa}"
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FRESH=1
unset RESUME FASTWAM_CHECKPOINT

TARGET_OUTPUT_DIR="${REPO_ROOT}/runs/robotwin_hfastwam/${RUN_NAME}"
shopt -s nullglob
EXISTING_CHECKPOINTS=(
  "${TARGET_OUTPUT_DIR}/checkpoints/weights/"*
  "${TARGET_OUTPUT_DIR}/checkpoints/state/"*
)
shopt -u nullglob
if (( ${#EXISTING_CHECKPOINTS[@]} > 0 )); then
  echo "ERROR: refusing a fresh launch because checkpoints already exist: ${TARGET_OUTPUT_DIR}/checkpoints" >&2
  exit 2
fi

if [[ ! -f "${REPO_ROOT}/configs/model/${MODEL_CONFIG}.yaml" ]]; then
  echo "ERROR: missing model config: configs/model/${MODEL_CONFIG}.yaml" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/configs/data/${DATA_CONFIG}.yaml" ]]; then
  echo "ERROR: missing data config: configs/data/${DATA_CONFIG}.yaml" >&2
  exit 2
fi

echo "[latent-action-48gpu] topology=6x8 world_size=48 global_batch=1536 micro_batch=32 grad_accum=1 epochs=5"
echo "[latent-action-48gpu] model=${MODEL_CONFIG} data=${DATA_CONFIG} trainable=${TRAINABLE_COMPONENTS}"
echo "[latent-action-48gpu] cache=${LATENT_ACTION_CACHE_ROOT} signature=${LATENT_ACTION_CACHE_SIGNATURE} fresh=1"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh"
