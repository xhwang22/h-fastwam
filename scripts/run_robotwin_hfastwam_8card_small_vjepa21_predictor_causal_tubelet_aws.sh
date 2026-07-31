#!/usr/bin/env bash
# V-JEPA 2.1 JEPAPredictor with three independent causal tubelet states.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_t3_ds}"
export TEMPORAL_DOWNSAMPLE=4
export STANDARDISE_OUTPUT=true

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" "$@"
