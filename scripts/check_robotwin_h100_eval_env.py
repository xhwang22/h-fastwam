#!/usr/bin/env python3
"""Validate the pinned H100 RoboTwin evaluation environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_ROBOTWIN_REVISION = "bf44be51cf5717a5595ce59447f2cf5263d2aa95"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="checkpoints/RoboTwin")
    parser.add_argument(
        "--render-backend",
        choices=("gpu", "cpu"),
        default="gpu",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError(f"Expected Python >=3.10, found {sys.version}")

    import cv2
    import gymnasium
    import mplib
    import open3d
    import sapien
    import torch
    import transformers
    import warp
    import curobo
    from curobo.types.math import Pose  # noqa: F401
    from curobo.types.robot import JointState  # noqa: F401
    from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
    from experiments.robotwin.fastwam_policy.deploy_policy import (  # noqa: F401
        WorldActionRobotWinPolicy,
    )

    if not torch.__version__.startswith("2.7.1+cu128"):
        raise RuntimeError(f"Expected torch 2.7.1+cu128, found {torch.__version__}")
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("No CUDA GPU is visible.")
    for device_index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(device_index)
        if capability[0] < 9:
            raise RuntimeError(
                f"GPU {device_index} is not H100-class: "
                f"{torch.cuda.get_device_name(device_index)}, capability={capability}"
            )
    if str(warp.__version__) != "1.12.1":
        raise RuntimeError(f"Expected warp-lang 1.12.1, found {warp.__version__}")

    robotwin_root = Path(args.robotwin_root).expanduser().resolve()
    if not robotwin_root.is_dir():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    revision = subprocess.check_output(
        ["git", "-C", str(robotwin_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != EXPECTED_ROBOTWIN_REVISION:
        raise RuntimeError(
            f"Expected RoboTwin {EXPECTED_ROBOTWIN_REVISION}, found {revision}"
        )
    for relative_path in (
        "assets/objects",
        "assets/embodiments",
        "assets/background_texture",
        "script/eval_policy.py",
    ):
        path = robotwin_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing RoboTwin asset/path: {path}")

    if args.render_backend == "gpu":
        vulkan = subprocess.run(
            ["vulkaninfo", "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=os.environ.copy(),
        )
        vulkan_output = f"{vulkan.stdout}\n{vulkan.stderr}"
        if vulkan.returncode != 0 or "NVIDIA" not in vulkan_output:
            raise RuntimeError(
                "Vulkan failed to enumerate an NVIDIA GPU:\n"
                f"{vulkan_output[-4000:]}"
            )
        render = subprocess.run(
            [sys.executable, "script/test_render.py"],
            cwd=robotwin_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ.copy(),
        )
        render_output = f"{render.stdout}\n{render.stderr}"
        if render.returncode != 0 or "Render Well" not in render_output:
            raise RuntimeError(
                "RoboTwin SAPIEN GPU renderer preflight failed:\n"
                f"{render_output[-4000:]}"
            )

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"warp={warp.__version__}")
    print(f"curobo={curobo.__file__}")
    print(f"sapien={sapien.__version__ if hasattr(sapien, '__version__') else sapien.__file__}")
    print(f"mplib={mplib.__file__}")
    print(f"gymnasium={gymnasium.__version__}")
    print(f"open3d={open3d.__version__}")
    print(f"opencv={cv2.__version__}")
    for device_index in range(torch.cuda.device_count()):
        print(
            f"gpu[{device_index}]={torch.cuda.get_device_name(device_index)} "
            f"capability={torch.cuda.get_device_capability(device_index)}"
        )
    print(f"render_backend={args.render_backend}")
    print("H100 RoboTwin evaluation environment: OK")


if __name__ == "__main__":
    main()
