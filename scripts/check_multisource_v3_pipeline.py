#!/usr/bin/env python3
"""Validate a multisource VIDEO_ONLY manifest and route-safe sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.lerobot.multisource_v3_dataset import (
    MultiSourceRobotV3Dataset,
    VideoOnlyLeRobotV3Dataset,
)


class _DummyInternData:
    def iter_epoch_indices(self, epoch_seed, batch_size, num_processes):
        del epoch_seed, batch_size, num_processes
        for index in range(1 << 30):
            yield (1 << 62) | (index << 32) | 7


def _validate_route_sampler(
    video_dataset: VideoOnlyLeRobotV3Dataset,
    batch_size: int,
    num_processes: int,
    global_batches: int,
) -> dict:
    global_batch_size = int(batch_size) * int(num_processes)
    dataset = object.__new__(MultiSourceRobotV3Dataset)
    dataset.samples_per_epoch = global_batch_size * int(global_batches)
    dataset.full_batch_fraction = 0.85
    dataset.video_clips_per_episode = 8
    dataset.video_locality_stride = 1
    dataset.intern_a1 = _DummyInternData()
    dataset.video_only = video_dataset
    weights = np.asarray(video_dataset.source_weights, dtype=np.float64)
    dataset.video_source_probabilities = weights / weights.sum()

    indices = list(
        dataset.iter_epoch_indices(
            42,
            batch_size=batch_size,
            num_processes=num_processes,
        )
    )
    route_counts = {"FULL": 0, "VIDEO_ONLY": 0}
    for block_start in range(0, len(indices), global_batch_size):
        block = indices[block_start : block_start + global_batch_size]
        routes = [
            dataset._decode_mixed_index(index)[0]
            for index in block
        ]
        if not all(route == routes[0] for route in routes):
            raise ValueError(
                f"Mixed routes in global batch {block_start // global_batch_size}."
            )
        route_counts["VIDEO_ONLY" if routes[0] else "FULL"] += 1
        if routes[0]:
            for group_start in range(
                0,
                len(block),
                dataset.video_clips_per_episode,
            ):
                group = block[
                    group_start : group_start + dataset.video_clips_per_episode
                ]
                decoded = [
                    dataset._decode_mixed_index(index)
                    for index in group
                ]
                source_ids = {source_id for _, source_id, _ in decoded}
                if len(source_ids) != 1:
                    raise ValueError(
                        "VIDEO_ONLY locality group mixes source IDs: "
                        f"{sorted(source_ids)}"
                    )
                source_id = decoded[0][1]
                cumulative = video_dataset.source_clip_cumulative[source_id]
                episode_indices = {
                    int(np.searchsorted(cumulative, payload, side="right"))
                    for _, _, payload in decoded
                }
                if len(episode_indices) != 1:
                    raise ValueError(
                        "VIDEO_ONLY locality group crosses episode boundaries."
                    )
    return route_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--samples-per-source", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--global-batches", type=int, default=20)
    args = parser.parse_args()

    dataset = VideoOnlyLeRobotV3Dataset(
        manifest_dir=args.manifest_dir,
        seed=42,
        max_retries=0,
    )
    source_summaries = []
    for source_id, source in enumerate(dataset.sources):
        summary = {
            "source_id": source["source_id"],
            "episodes": int(dataset.source_episode_rows[source_id].size),
            "clips": dataset.source_clip_counts[source_id],
            "weight": dataset.source_weights[source_id],
        }
        if args.decode:
            decoded = []
            for sample_number in range(args.samples_per_source):
                sample = dataset.get_source_sample(
                    source_id,
                    sample_key=source_id * 1_000_003 + sample_number * 97_409 + 17,
                    sample_index=sample_number,
                )
                video = sample["video"]
                if video.shape != (3, 9, 384, 320):
                    raise ValueError(
                        f"{source['source_id']} video shape: {tuple(video.shape)}"
                    )
                if video.dtype != torch.float32 or not torch.isfinite(video).all():
                    raise ValueError(
                        f"{source['source_id']} video dtype/finiteness failure."
                    )
                if float(video.min()) < -1.0001 or float(video.max()) > 1.0001:
                    raise ValueError(
                        f"{source['source_id']} video outside [-1,1]."
                    )
                if int(sample["route_id"]) != 1:
                    raise ValueError(
                        f"{source['source_id']} must use VIDEO_ONLY route."
                    )
                decoded.append(
                    {
                        "view_role_valid_mask": sample[
                            "view_role_valid_mask"
                        ].tolist(),
                        "video_min": float(video.min()),
                        "video_max": float(video.max()),
                    }
                )
            summary["decoded"] = decoded
        source_summaries.append(summary)

    result = {
        "manifest": str(Path(args.manifest_dir).expanduser().resolve()),
        "manifest_version": int(dataset.manifest["version"]),
        "sources": source_summaries,
        "route_batches": _validate_route_sampler(
            dataset,
            batch_size=args.batch_size,
            num_processes=args.num_processes,
            global_batches=args.global_batches,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
