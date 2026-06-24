#!/usr/bin/env bash
# Quick monitor for the currently running JEPA+RoboTwin multi-node training.
# Usage: bash scripts/monitor_current_run.sh
set -euo pipefail
REPO=/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM
RN=${RN:-$(cat /tmp/current_run_name.txt 2>/dev/null || ls -t $REPO/runs/robotwin_vjepa2ac_predictor/ | head -1)}
LD=$REPO/runs/robotwin_vjepa2ac_predictor/$RN
echo "RUN_NAME = $RN"
echo "LOG_DIR  = $LD"
date
echo ""
echo "=== procs per node ==="
echo "28.216.19.197 (rank0, local): $(ps aux | grep -E 'train\.py|torchrun' | grep -v grep | wc -l)"
for h in 28.216.19.17 28.216.19.7 28.216.19.159; do
  n=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 $h "ps aux | grep -E 'train\.py|torchrun' | grep -v grep | wc -l" 2>/dev/null)
  echo "$h: $n"
done
echo ""
echo "=== log sizes ==="
wc -l $LD/train.log.rank* 2>/dev/null
echo ""
echo "=== last 15 lines rank0 ==="
tail -15 $LD/train.log.rank0 2>/dev/null
echo ""
echo "=== errors ==="
grep -iE "error|traceback|failed|CUDA out" $LD/train.log.rank* 2>/dev/null | grep -vE "404|HTTP Request|INFO|warnings" | tail -10 || echo "(none)"
echo ""
echo "=== step progress ==="
grep -hE "global_step|train/loss=|Step [0-9]+/[0-9]+|epoch [0-9]+" $LD/train.log.rank0 2>/dev/null | tail -5 || echo "(no step logged yet — still in init)"
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -4
