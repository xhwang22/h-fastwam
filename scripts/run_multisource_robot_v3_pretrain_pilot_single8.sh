#!/usr/bin/env bash
# Single-node: 8 GPUs x batch 48 x accumulation 2 = global batch 768.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset PET_NNODES PET_NODE_RANK PET_MASTER_ADDR PET_MASTER_PORT PET_NPROC_PER_NODE
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT NODE_IP_LIST
unset FASTWAM_MANAGED_DISTRIBUTED _MULTINODE_LAUNCHED

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE=8
export FASTWAM_EXPECTED_WORLD_SIZE=8
export GLOBAL_BATCH_SIZE=768
export GRADIENT_ACCUMULATION_STEPS=2
export FASTWAM_USE_EFA=0
export RUN_NAME="${RUN_NAME:-multisource_robot_v3_pilot_single8_b48_acc2}"

exec bash "${SCRIPT_DIR}/run_multisource_robot_v3_pretrain_pilot.sh" "$@"
