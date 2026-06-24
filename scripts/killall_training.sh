#!/usr/bin/env bash
# Kill all training-related processes on this node
pkill -9 -f "scripts/train.py" 2>/dev/null || true
pkill -9 -f "deepspeed.launcher" 2>/dev/null || true
pkill -9 -f "deepspeed/launcher" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
pkill -9 -f "train_zero1" 2>/dev/null || true
pkill -9 -f "accelerate launch" 2>/dev/null || true
sleep 2
remaining=$(pgrep -c -f "train.py" 2>/dev/null || echo 0)
echo "[killall] $(hostname): $remaining train.py procs remaining"
