#!/usr/bin/env bash
# Multi-node launcher: H-FastWAM SMALL + V-JEPA 2-AC JEPAPredictor (from scratch).
#
# Thin wrapper around run_libero_hfastwam_small_multinode.sh. model=
# hfastwam_small_vjepa_predictor uses a deterministic JEPAPredictor video expert
# (L1 regression in frozen V-JEPA 2-AC ViT-g encoder space, in/out_dim=1408)
# instead of a flow-matching DiT. Video & action experts stay 2048/16/28
# random-init, language frozen, and the action stream attends to detached video
# K/V (the code fix in hfastwam.py), matching FastWAM's
# action_loss_detach_video_expert=True.
#
# NOTE: the video expert is a deterministic predictor, so the video loss is L1
# (not flow-matching MSE). The action expert is still a flow-matching denoiser,
# so the action-loss comparison against the VAE / DINO runs is apples-to-apples.
#
# Run this on ONE node (set the platform topology vars as usual).
#
# Usage:
#   WANDB=1 FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=small_vjepa_pred_mn \
#     bash scripts/run_libero_hfastwam_small_vjepa_predictor_multinode.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-hfastwam_small_vjepa_predictor}"
export RUN_PREFIX="${RUN_PREFIX:-libero_hfastwam_small_vjepa_pred_mn}"

exec bash "${SCRIPT_DIR}/run_libero_hfastwam_small_multinode.sh"
