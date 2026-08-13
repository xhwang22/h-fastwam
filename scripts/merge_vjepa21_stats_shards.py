#!/usr/bin/env python3
"""Merge independent V-JEPA statistics shards into one global stats file."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


class WelfordAccumulator:
    def __init__(self, num_channels: int):
        self.n = 0
        self.mean = torch.zeros(num_channels, dtype=torch.float64)
        self.m2 = torch.zeros(num_channels, dtype=torch.float64)

    def merge(self, n: int, mean: torch.Tensor, m2: torch.Tensor) -> None:
        if n == 0:
            return
        mean = mean.to(dtype=torch.float64, device="cpu")
        m2 = m2.to(dtype=torch.float64, device="cpu")
        if self.n == 0:
            self.n = int(n)
            self.mean.copy_(mean)
            self.m2.copy_(m2)
            return
        new_n = self.n + int(n)
        delta = mean - self.mean
        self.mean += delta * (int(n) / new_n)
        self.m2 += m2 + delta.square() * (self.n * int(n) / new_n)
        self.n = new_n

    @property
    def std(self) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("Cannot compute statistics without latent vectors.")
        return (self.m2 / self.n).sqrt()


def _load_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, got {type(payload).__name__}."
        )
    return payload


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _shard_paths(shard_dir: Path, world_size: int) -> list[Path]:
    return [
        shard_dir / f"rank-{rank:05d}-of-{world_size:05d}.pt"
        for rank in range(world_size)
    ]


def _wait_for_shards(
    paths: list[Path],
    timeout_seconds: int,
    poll_interval: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            preview = ", ".join(path.name for path in missing[:8])
            raise TimeoutError(
                f"Timed out waiting for {len(missing)} statistics shards: {preview}"
            )
        logger.info(
            "Waiting for %d/%d statistics shards; first missing: %s",
            len(missing),
            len(paths),
            missing[0].name,
        )
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independent V-JEPA statistics shard files."
    )
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive.")
    if args.wait_timeout < 0:
        parser.error("--wait-timeout must be non-negative.")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    shard_dir = args.shard_dir.expanduser().resolve()
    paths = _shard_paths(shard_dir, args.expected_world_size)
    _wait_for_shards(paths, args.wait_timeout, args.poll_interval)

    run_config = None
    num_channels = None
    merged = None
    processed_samples = 0
    processed_videos = 0
    failed_samples = 0

    for expected_rank, path in enumerate(paths):
        payload = _load_payload(path)
        if payload.get("format") != "fastwam_vjepa21_stats_shard":
            raise ValueError(f"Invalid V-JEPA statistics shard format in {path}.")
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported V-JEPA statistics shard version in {path}.")
        if int(payload.get("rank", -1)) != expected_rank:
            raise ValueError(f"Shard rank mismatch in {path}.")
        if int(payload.get("world_size", -1)) != args.expected_world_size:
            raise ValueError(f"Shard world-size mismatch in {path}.")
        if run_config is None:
            run_config = payload.get("run_config")
        elif payload.get("run_config") != run_config:
            raise ValueError(f"Shard configuration mismatch in {path}.")

        shard_channels = int(payload["num_channels"])
        mean = payload["mean"]
        m2 = payload["m2"]
        if mean.shape != (shard_channels,) or m2.shape != (shard_channels,):
            raise ValueError(f"Invalid accumulator shape in {path}.")
        if num_channels is None:
            num_channels = shard_channels
            merged = WelfordAccumulator(num_channels)
        elif shard_channels != num_channels:
            raise ValueError(f"Channel-count mismatch in {path}.")

        merged.merge(
            int(payload["latent_vector_count"]),
            mean,
            m2,
        )
        processed_samples += int(payload["processed_samples"])
        processed_videos += int(payload["processed_videos"])
        failed_samples += int(payload["failed_samples"])

    if run_config is None or num_channels is None or merged is None:
        raise RuntimeError("No statistics shards were loaded.")
    mean = merged.mean.float()
    std = merged.std.float()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("Merged V-JEPA statistics contain NaN/Inf.")
    if torch.any(std <= 0):
        raise RuntimeError("Merged V-JEPA statistics contain non-positive std.")

    metadata = {
        "version": 1,
        **run_config,
        "normalisation_axes": "global_sample_time_space_per_channel",
        "processed_samples": processed_samples,
        "processed_videos": processed_videos,
        "failed_samples": failed_samples,
        "latent_vector_count": merged.n,
        "num_channels": num_channels,
        "aggregation_mode": "independent_shards",
        "shard_count": args.expected_world_size,
    }
    output_payload = {
        "mean": mean,
        "std": std,
        **metadata,
    }
    output_path = args.output_path.expanduser().resolve()
    _atomic_torch_save(output_path, output_payload)
    _atomic_json_save(
        shard_dir / "MERGED.json",
        {
            "output_path": str(output_path),
            "processed_samples": processed_samples,
            "processed_videos": processed_videos,
            "failed_samples": failed_samples,
            "latent_vector_count": merged.n,
            "shard_count": args.expected_world_size,
        },
    )
    logger.info(
        "Merged %d shards into %s: samples=%d videos=%d vectors=%d channels=%d.",
        args.expected_world_size,
        output_path,
        processed_samples,
        processed_videos,
        merged.n,
        num_channels,
    )
    logger.info("Metadata: %s", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
