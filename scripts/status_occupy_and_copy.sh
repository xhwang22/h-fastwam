#!/usr/bin/env bash
# Snapshot current status of GPU-occupy + dataset-copy across all 4 nodes.
set -euo pipefail
HOSTS=$(awk 'NF && $1!~/^#/ {print $1}' /etc/taiji/hostfile)

date

echo ""
echo "=========================================="
echo "GPU OCCUPY"
echo "=========================================="
echo "--- 28.216.19.197 (local) ---"
echo -n "  occupy proc: "; ps -p $(cat /tmp/gpu_occupy/pid.rank0 2>/dev/null) -o stat= 2>/dev/null || echo "dead"
echo -n "  last log: "; tail -1 /tmp/gpu_occupy/gpu_occupy.rank0.log 2>/dev/null
echo -n "  GPU util/mem: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1

for h in $HOSTS; do
  if [[ "$h" == "28.216.19.197" ]]; then continue; fi
  echo "--- $h ---"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 $h "
    echo -n '  occupy proc: '; ps -p \$(cat /tmp/gpu_occupy/pid.rank* 2>/dev/null | head -1) -o stat= 2>/dev/null || echo dead
    echo -n '  last log: '; tail -1 /tmp/gpu_occupy/gpu_occupy.rank*.log 2>/dev/null | tail -1
    echo -n '  GPU util/mem: '; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1
  " 2>&1
done

echo ""
echo "=========================================="
echo "DATASET COPY (target: ~75G to /tmp/fastwam_data/)"
echo "=========================================="
echo -n "28.216.19.197 (local): "
size=$(du -sh /tmp/fastwam_data/ 2>/dev/null | cut -f1)
done=$([[ -f /tmp/copy_robotwin/done.flag ]] && echo "DONE" || echo "running")
proc=$(ps -p $(cat /tmp/copy_robotwin/pid 2>/dev/null) -o stat= 2>/dev/null || echo "dead")
echo "size=$size status=$done proc=$proc"

for h in $HOSTS; do
  if [[ "$h" == "28.216.19.197" ]]; then continue; fi
  res=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 $h "
    size=\$(du -sh /tmp/fastwam_data/ 2>/dev/null | cut -f1)
    done=\$([[ -f /tmp/copy_robotwin/done.flag ]] && echo DONE || echo running)
    proc=\$(ps -p \$(cat /tmp/copy_robotwin/pid 2>/dev/null) -o stat= 2>/dev/null || echo dead)
    echo \"size=\$size status=\$done proc=\$proc\"
  " 2>&1)
  echo "$h: $res"
done
