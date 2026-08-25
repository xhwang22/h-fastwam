#!/usr/bin/env bash
# InternData-A1 30Hz VAE+DiT pretraining baseline: 8 nodes x 8 GPUs,
# micro-batch 24/GPU, gradient accumulation 2, global batch 3072.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export INTERN_A1_ROOT="${INTERN_A1_ROOT:-/s3/data/pretrain_data/InternData-A1}"
export INTERN_A1_MANIFEST_DIR="${INTERN_A1_MANIFEST_DIR:-/fsx/fastwam_manifests/interndata_a1/manifest_v5_30hz}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=64
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$(( 48 * FASTWAM_EXPECTED_WORLD_SIZE ))}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-24}"
export RUN_NAME="${RUN_NAME:-interndata_a1_vae_dit_pretrain_30hz_8x8_b${PER_GPU_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}}"
export LOG_DIR="${LOG_DIR:-/fsx/h-fastwam/runs/robotwin_hfastwam/${RUN_NAME}}"

exec bash \
  "${SCRIPT_DIR}/run_interndata_a1_vae_dit_pretrain_16gpu_b48_cudnn_overlap_efa.sh" \
  "$@"
