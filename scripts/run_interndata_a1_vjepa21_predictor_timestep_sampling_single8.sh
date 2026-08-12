#!/usr/bin/env bash
# InternData-A1 V-JEPA predictor action-timestep sampling ablation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_timestep_sampling_preset.sh
source "${SCRIPT_DIR}/_timestep_sampling_preset.sh"
fastwam_timestep_sampling_preset "${TIMESTEP_SAMPLING_PRESET:-baseline}" action

export RUN_NAME="${RUN_NAME:-interndata_a1_vjepa21_predictor_${TIMESTEP_PRESET_SUFFIX}_single8}"

exec bash \
  "${SCRIPT_DIR}/run_interndata_a1_vjepa21_predictor_pretrain_single8_b48_acc2_cudnn.sh" \
  "${TIMESTEP_SAMPLING_OVERRIDES[@]}" \
  "$@"
