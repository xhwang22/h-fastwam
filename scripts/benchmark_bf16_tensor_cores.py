#!/usr/bin/env python3
"""Saturate all visible GPUs with BF16 GEMMs to validate Tensor Core metrics."""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0, help="Timed duration in seconds.")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Square GEMM matrix size.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No visible CUDA devices.")

    matrices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    gib_per_gpu = 3 * args.matrix_size**2 * 2 / 1024**3

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(
        f"gpus={device_count} matrix={args.matrix_size}x{args.matrix_size} "
        f"dtype=bf16 duration={args.duration:.1f}s allocation≈{gib_per_gpu:.2f}GiB/GPU"
    )

    for device_index in range(device_count):
        device = torch.device("cuda", device_index)
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"CUDA device {device_index} does not support BF16.")
            properties = torch.cuda.get_device_properties(device)
            print(
                f"gpu[{device_index}]={properties.name} "
                f"capability={properties.major}.{properties.minor}"
            )
            left = torch.randn(
                args.matrix_size,
                args.matrix_size,
                device=device,
                dtype=torch.bfloat16,
            )
            right = torch.randn_like(left)
            output = torch.empty_like(left)
            matrices.append((left, right, output))

    def run_iteration() -> None:
        for device_index, (left, right, output) in enumerate(matrices):
            with torch.cuda.device(device_index):
                torch.mm(left, right, out=output)
        for device_index in range(device_count):
            torch.cuda.synchronize(device_index)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_iteration()

        start = time.perf_counter()
        iterations = 0
        while time.perf_counter() - start < args.duration:
            run_iteration()
            iterations += 1
        elapsed = time.perf_counter() - start

    operations_per_gpu = 2 * args.matrix_size**3 * iterations
    tflops_per_gpu = operations_per_gpu / elapsed / 1e12
    aggregate_tflops = tflops_per_gpu * device_count

    print(
        f"iterations={iterations} elapsed={elapsed:.2f}s "
        f"per_gpu={tflops_per_gpu:.1f} TFLOPS aggregate={aggregate_tflops:.1f} TFLOPS"
    )
    print("Observe Tensor Active while this script runs. It should rise substantially.")


if __name__ == "__main__":
    main()
