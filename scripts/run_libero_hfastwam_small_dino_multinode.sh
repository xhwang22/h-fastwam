#!/usr/bin/env bash
# Multi-node launcher: H-FastWAM SMALL + DINOv3 vision encoder (from scratch).
#
# Thin wrapper around run_libero_hfastwam_small_multinode.sh: the ONLY difference
# vs the VAE run is the world-model representation space. model=hfastwam_small_dino
# swaps the VAE-48d latent for FROZEN DINOv3 ViT-S features (in/out_dim=384); the
# video & action experts stay 2048/16/28 random-init, language frozen, and the
# action stream attends to detached video K/V (the code fix in hfastwam.py),
# matching FastWAM's action_loss_detach_video_expert=True.
#
# Run this on ONE node (set the platform topology vars as usual). Same training
# hyperparameters / data / freeze policy as the VAE run — only the encoder differs.
#
# Usage:
#   WANDB=1 FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=small_dino_mn \
#     bash scripts/run_libero_hfastwam_small_dino_multinode.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-hfastwam_small_dino}"
export RUN_PREFIX="${RUN_PREFIX:-libero_hfastwam_small_dino_mn}"

exec bash "${SCRIPT_DIR}/run_libero_hfastwam_small_multinode.sh"
