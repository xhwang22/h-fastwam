#!/usr/bin/env bash
# Single-node InternData-A1 v3 pretraining: 8 GPUs x batch 48 x accum 2 = 768.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FASTWAM_EXPECTED_WORLD_SIZE=8
export GLOBAL_BATCH_SIZE=$(( 8 * 48 * 2 ))
export GRADIENT_ACCUMULATION_STEPS=2
export FASTWAM_USE_EFA=0
export RUN_NAME="${RUN_NAME:-interndata_a1_vjepa21_predictor_pretrain_v3_single8_b48_acc2}"

exec bash \
  "${SCRIPT_DIR}/run_interndata_a1_vjepa21_predictor_pretrain_16gpu_b48_cudnn_overlap_efa.sh" \
  "$@"
