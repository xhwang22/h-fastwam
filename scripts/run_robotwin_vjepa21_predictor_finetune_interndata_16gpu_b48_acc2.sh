#!/usr/bin/env bash
# Fine-tune a transferred InternData checkpoint on RoboTwin with fresh 14-d heads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${FASTWAM_CHECKPOINT:-}" ]]; then
  echo "ERROR: set FASTWAM_CHECKPOINT to the converted RobotTwin-transfer checkpoint." >&2
  echo "Run scripts/prepare_interndata_checkpoint_for_robotwin.py first." >&2
  exit 1
fi

export FASTWAM_EXPECTED_WORLD_SIZE=16
export GLOBAL_BATCH_SIZE=$(( 48 * FASTWAM_EXPECTED_WORLD_SIZE * 2 ))
export GRADIENT_ACCUMULATION_STEPS=2
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export LOG_EVERY="${LOG_EVERY:-10}"
export FASTWAM_SDPA_BACKEND=cudnn
export ACCEL_CONFIG=scripts/accelerate_configs/accelerate_zero2_bf16.yaml
export FASTWAM_USE_EFA=1
export RUN_NAME="${RUN_NAME:-robotwin_vjepa21_predictor_finetune_interndata_16gpu_b48_acc2}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" "$@"
