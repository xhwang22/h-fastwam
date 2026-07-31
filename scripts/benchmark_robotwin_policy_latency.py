#!/usr/bin/env python3
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.fastwam_policy.deploy_policy import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    args = parser.parse_args()

    policy = get_model(
        {
            "sim_resolved_cfg_path": args.config,
            "ckpt_setting": args.checkpoint,
            "dataset_stats_path": args.stats,
            "device": "cuda",
            "mixed_precision": "bf16",
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps,
            "seed": 42,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "timing_enabled": False,
        }
    )
    observation = {
        "observation": {
            "head_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
        },
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
    }
    instruction = "click the alarm clock"

    for _ in range(args.warmup):
        policy._infer_action_chunk(observation, instruction)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        policy._infer_action_chunk(observation, instruction)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)

    median_s = statistics.median(latencies)
    mean_s = statistics.mean(latencies)
    print(f"latencies_s={','.join(f'{value:.6f}' for value in latencies)}")
    print(f"median_s={median_s:.6f}")
    print(f"mean_s={mean_s:.6f}")
    print(f"chunk_hz={1.0 / median_s:.6f}")
    print(f"effective_action_hz={args.replan_steps / median_s:.6f}")


if __name__ == "__main__":
    main()
