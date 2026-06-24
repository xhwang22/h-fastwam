#!/bin/bash
# =============================================================================
# Wan2.2 Continue-Pretrain with DINO visual encoder on OpenVid-1M
# =============================================================================
#
# Prerequisites:
#   1. Download OpenVid-1M dataset and update paths in configs/data/openvid.yaml
#   2. (Optional) Precompute text embeddings:
#      python scripts/precompute_openvid_text_embeds.py \
#          --csv_path /path/to/OpenVid-1M/OpenVid-1M.csv \
#          --output_dir /path/to/OpenVid-1M/text_embeds
#   3. Install decord: pip install decord
#   4. DINOv3 weights will be auto-downloaded from HuggingFace
#
# Usage:
#   # Single GPU
#   bash scripts/run_pretrain_dino.sh
#
#   # Multi-GPU (e.g., 8 GPUs)
#   NUM_GPUS=8 bash scripts/run_pretrain_dino.sh

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29501}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# DeepSpeed config for ZeRO-2
DEEPSPEED_CONFIG=$(cat <<'EOF'
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true
    },
    "bf16": {"enabled": true},
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
EOF
)

DEEPSPEED_FILE="/tmp/ds_config_pretrain_dino_$$.json"
echo "$DEEPSPEED_CONFIG" > "$DEEPSPEED_FILE"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --num_processes "$NUM_GPUS" \
        --num_machines 1 \
        --mixed_precision bf16 \
        --use_deepspeed \
        --deepspeed_config_file "$DEEPSPEED_FILE" \
        --main_process_port "$MASTER_PORT" \
        "${PROJECT_ROOT}/scripts/pretrain.py" \
        task=pretrain_dino_openvid \
        data=openvid \
        model=pretrain_dino \
        "$@"
else
    accelerate launch \
        --num_processes 1 \
        --mixed_precision bf16 \
        --use_deepspeed \
        --deepspeed_config_file "$DEEPSPEED_FILE" \
        "${PROJECT_ROOT}/scripts/pretrain.py" \
        task=pretrain_dino_openvid \
        data=openvid \
        model=pretrain_dino \
        "$@"
fi

rm -f "$DEEPSPEED_FILE"
echo "Done."
