#!/usr/bin/env bash
# RoboTwin training with Wan VAE latents and a deterministic video predictor on AWS HyperPod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=_aws_hyperpod_setup.sh
source "${SCRIPT_DIR}/_aws_hyperpod_setup.sh"
fastwam_prepare_aws_hyperpod

export MODEL_CONFIG="${MODEL_CONFIG:-hfastwam_small_vae_predictor}"
export USE_VJEPA21_VISUAL_ENCODER=0
export VIDEO_LATENT_CACHE_ENABLED=0
export RUN_NAME="${RUN_NAME:-robotwin_hfastwam_vae_predictor_aws}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastwam-robotwin-encoder-ablation}"
export WANDB_GROUP="${WANDB_GROUP:-vae-predictor}"

# The VAE is the encoder in this configuration, so visual-encoder overrides
# inherited from another experiment would produce invalid Hydra overrides.
unset VJEPA21_NORMALISE_STATS_PATH STANDARDISE_OUTPUT TEMPORAL_DOWNSAMPLE
unset CAUSAL_TUBELET_ENCODING FRAME_GAP FIXED_TARGET_ENCODER
unset VISUAL_ENCODER_FREEZE_BACKBONE VISUAL_ENCODER_ACTIVATION_CHECKPOINTING
unset TRAINABLE_COMPONENTS VISUAL_ENCODER_LR_MULTIPLIER

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor.sh" "$@"
