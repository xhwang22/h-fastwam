#!/usr/bin/env bash
# V-JEPA 2.1 predictor IDM with teacher-forced future-latent action conditioning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

export TASK_CONFIG=robotwin_idm_3cam_384_1e-4
export MODEL_CONFIG=hfastwam_idm_vjepa21_predictor
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_idm_vjepa21_predictor_causal_tubelet}"
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true
export DETACH_VIDEO=false

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" "$@"
