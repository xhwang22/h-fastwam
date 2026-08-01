#!/usr/bin/env bash
# Single-node 8-GPU V-JEPA21 causal-tubelet JEPAPredictor benchmark.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset PET_NNODES
unset PET_NPROC_PER_NODE
unset PET_NODE_RANK
unset PET_MASTER_ADDR
unset PET_MASTER_PORT
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset LOCAL_WORLD_SIZE
unset GROUP_RANK
unset NNODES
unset NODE_RANK
unset MASTER_ADDR
unset MASTER_PORT
unset NODE_IP_LIST
unset FASTWAM_MANAGED_DISTRIBUTED
unset _MULTINODE_LAUNCHED

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export GLOBAL_BATCH_SIZE=384
export GRADIENT_ACCUMULATION_STEPS=1
export FASTWAM_USE_EFA=0
export FASTWAM_SDPA_BACKEND="${FASTWAM_SDPA_BACKEND:-cudnn}"
export ACCEL_CONFIG="${ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero2_bf16.yaml}"
export RUN_NAME="${RUN_NAME:-single8_batch384_overlap_vjepa21_predictor_causal_tubelet}"

exec bash "${SCRIPT_DIR}/run_robotwin_hfastwam_8card_small_vjepa21_predictor_causal_tubelet_aws.sh" "$@"
