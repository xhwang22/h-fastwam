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
    ROUTE_FULL,
    ROUTE_VIDEO_ONLY,
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
    intern_weight: float,
) -> dict:
    global_batch_size = int(batch_size) * int(num_processes)
    dataset = object.__new__(MultiSourceRobotV3Dataset)
    dataset.samples_per_epoch = global_batch_size * int(global_batches)
    video_weight = float(sum(video_dataset.source_weights))
    dataset.full_batch_fraction = float(intern_weight) / (
        float(intern_weight) + video_weight
    )
    dataset.video_clips_per_episode = 8
    dataset.video_locality_stride = 1
    dataset.intern_a1 = _DummyInternData()
    dataset.external_sources = video_dataset
    full_source_ids = [-1]
    full_masses = [float(intern_weight)]
    video_source_ids = []
    video_masses = []
    for source_id, source_weight in enumerate(video_dataset.source_weights):
        full_count = video_dataset.source_route_clip_counts[
            (source_id, ROUTE_FULL)
        ]
        video_count = video_dataset.source_route_clip_counts[
            (source_id, ROUTE_VIDEO_ONLY)
        ]
        total_count = full_count + video_count
        if full_count:
            full_source_ids.append(source_id)
            full_masses.append(source_weight * full_count / total_count)
        if video_count:
            video_source_ids.append(source_id)
            video_masses.append(source_weight * video_count / total_count)
    dataset.full_source_ids = np.asarray(full_source_ids, dtype=np.int64)
    dataset.full_source_probabilities = np.asarray(full_masses, dtype=np.float64)
    dataset.full_source_probabilities /= dataset.full_source_probabilities.sum()
    dataset.video_source_ids = np.asarray(video_source_ids, dtype=np.int64)
    dataset.video_source_probabilities = np.asarray(video_masses, dtype=np.float64)
    if dataset.video_source_probabilities.size:
        dataset.video_source_probabilities /= dataset.video_source_probabilities.sum()
    full_mass = float(sum(full_masses))
    video_mass = float(sum(video_masses))
    dataset.full_batch_fraction = full_mass / (full_mass + video_mass)

    indices = list(
        dataset.iter_epoch_indices(
            42,
            batch_size=batch_size,
            num_processes=num_processes,
        )
    )
    route_counts = {"FULL": 0, "VIDEO_ONLY": 0}
    source_sample_counts = {
        "interndata_a1": 0,
        **{
            source["source_id"]: 0
            for source in video_dataset.sources
        },
    }
    for block_start in range(0, len(indices), global_batch_size):
        block = indices[block_start : block_start + global_batch_size]
        decoded_block = [
            dataset._decode_mixed_index(index)
            for index in block
        ]
        routes = [decoded[1] for decoded in decoded_block]
        for is_external, _, source_id, _ in decoded_block:
            source_name = (
                video_dataset.sources[source_id]["source_id"]
                if is_external
                else "interndata_a1"
            )
            source_sample_counts[source_name] += 1
        if not all(route == routes[0] for route in routes):
            raise ValueError(
                f"Mixed routes in global batch {block_start // global_batch_size}."
            )
        route_counts[
            "VIDEO_ONLY" if routes[0] == ROUTE_VIDEO_ONLY else "FULL"
        ] += 1
        for group_start in range(
            0,
            len(block),
            dataset.video_clips_per_episode,
        ):
            group = [
                dataset._decode_mixed_index(index)
                for index in block[
                    group_start : group_start + dataset.video_clips_per_episode
                ]
            ]
            external_flags = {is_external for is_external, _, _, _ in group}
            if external_flags == {False}:
                continue
            if external_flags != {True}:
                raise ValueError("Locality group mixes InternData and external samples.")
            if group:
                source_ids = {source_id for _, _, source_id, _ in group}
                if len(source_ids) != 1:
                    raise ValueError(
                        "VIDEO_ONLY locality group mixes source IDs: "
                        f"{sorted(source_ids)}"
                    )
                route_ids = {route_id for _, route_id, _, _ in group}
                if len(route_ids) != 1:
                    raise ValueError("Locality group mixes routes.")
                source_id = group[0][2]
                route_id = group[0][1]
                cumulative = video_dataset.source_route_clip_cumulative[
                    (source_id, route_id)
                ]
                episode_indices = {
                    int(np.searchsorted(cumulative, payload, side="right"))
                    for _, _, _, payload in group
                }
                if len(episode_indices) != 1:
                    raise ValueError(
                        "VIDEO_ONLY locality group crosses episode boundaries."
                    )
    return {
        "route_batches": route_counts,
        "source_sample_counts": source_sample_counts,
        "source_sample_fractions": {
            source: count / len(indices)
            for source, count in source_sample_counts.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--samples-per-source", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--global-batches", type=int, default=20)
    parser.add_argument("--intern-weight", type=float, default=0.20)
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
            "full_episodes": int(
                dataset.source_route_episode_rows[
                    (source_id, ROUTE_FULL)
                ].size
            ),
            "video_only_episodes": int(
                dataset.source_route_episode_rows[
                    (source_id, ROUTE_VIDEO_ONLY)
                ].size
            ),
            "full_clips": dataset.source_route_clip_counts[
                (source_id, ROUTE_FULL)
            ],
            "video_only_clips": dataset.source_route_clip_counts[
                (source_id, ROUTE_VIDEO_ONLY)
            ],
            "weight": dataset.source_weights[source_id],
        }
        if args.decode:
            decoded = []
            for sample_number in range(args.samples_per_source):
                route_id = (
                    ROUTE_FULL
                    if dataset.source_route_clip_counts[
                        (source_id, ROUTE_FULL)
                    ] > 0
                    else ROUTE_VIDEO_ONLY
                )
                sample = dataset.get_source_sample(
                    source_id,
                    route_id,
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
                if int(sample["route_id"]) != route_id:
                    raise ValueError(
                        f"{source['source_id']} route mismatch."
                    )
                if route_id == ROUTE_FULL:
                    if sample["action"].shape != (32, 20):
                        raise ValueError("FULL action must be [32,20].")
                    if sample["proprio"].shape != (32, 20):
                        raise ValueError("FULL proprio must be [32,20].")
                    if (
                        not torch.isfinite(sample["action"]).all()
                        or not torch.isfinite(sample["proprio"]).all()
                    ):
                        raise ValueError("FULL kinematics contain NaN/Inf.")
                decoded.append(
                    {
                        "route": (
                            "FULL"
                            if route_id == ROUTE_FULL
                            else "VIDEO_ONLY"
                        ),
                        "view_role_valid_mask": sample[
                            "view_role_valid_mask"
                        ].tolist(),
                        "video_min": float(video.min()),
                        "video_max": float(video.max()),
                        **(
                            {
                                "action_min": float(sample["action"].min()),
                                "action_max": float(sample["action"].max()),
                                "proprio_min": float(sample["proprio"].min()),
                                "proprio_max": float(sample["proprio"].max()),
                                "valid_action_dims": int(
                                    (~sample["action_dim_is_pad"]).sum()
                                ),
                            }
                            if route_id == ROUTE_FULL
                            else {}
                        ),
                    }
                )
            summary["decoded"] = decoded
        source_summaries.append(summary)

    result = {
        "manifest": str(Path(args.manifest_dir).expanduser().resolve()),
        "manifest_version": int(dataset.manifest["version"]),
        "sources": source_summaries,
        "sampler": _validate_route_sampler(
            dataset,
            batch_size=args.batch_size,
            num_processes=args.num_processes,
            global_batches=args.global_batches,
            intern_weight=args.intern_weight,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
