#!/usr/bin/env python3
"""Validate InternData-A1 v5 manifest and canonical temporal samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.lerobot.intern_a1_v3_dataset import (
    InternDataA1V3Dataset,
    build_temporal_contract,
)


def _family_summary(
    datasets: list[dict],
    dataset_ids: np.ndarray,
    lengths: np.ndarray,
    native_horizon: int,
) -> dict:
    summary = {}
    for family in ("dual", "single"):
        ids = np.asarray(
            [
                dataset_id
                for dataset_id, metadata in enumerate(datasets)
                if metadata["family"] == family
            ],
            dtype=np.int64,
        )
        mask = np.isin(dataset_ids, ids)
        summary[family] = {
            "dataset_roots": int(ids.size),
            "episodes": int(mask.sum()),
            "clip_starts": int(
                np.maximum(
                    lengths[mask].astype(np.int64) - native_horizon,
                    0,
                ).sum()
            ),
            "robot_types": sorted(
                {
                    str(datasets[dataset_id].get("robot_type"))
                    for dataset_id in ids
                }
            ),
        }
    return summary


def _validate_sample(sample: dict, family: str) -> dict:
    expected = {
        "video": (3, 9, 384, 320),
        "action": (32, 20),
        "proprio": (32, 20),
        "image_is_pad": (9,),
        "action_is_pad": (32,),
        "action_dim_is_pad": (20,),
        "view_role_valid_mask": (3,),
        "video_spatial_valid_mask": (384, 320),
    }
    for key, shape in expected.items():
        value = sample[key]
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{family} sample `{key}` shape {tuple(value.shape)} != {shape}."
            )
    for key in ("video", "action", "proprio"):
        if not torch.isfinite(sample[key]).all():
            raise ValueError(f"{family} sample `{key}` contains NaN/Inf.")
    video = sample["video"]
    if float(video.min()) < -1.0001 or float(video.max()) > 1.0001:
        raise ValueError(f"{family} sample video is outside [-1,1].")

    view_mask = sample["view_role_valid_mask"].to(torch.bool)
    spatial_mask = sample["video_spatial_valid_mask"].to(torch.bool)
    action_dim_is_pad = sample["action_dim_is_pad"].to(torch.bool)
    if family == "dual":
        if view_mask.tolist() != [True, True, True]:
            raise ValueError(f"Dual sample view mask is {view_mask.tolist()}.")
        if int((~action_dim_is_pad).sum()) != 20:
            raise ValueError("Dual sample must supervise 20 action dimensions.")
        generated_black_fraction = 0.0
    else:
        if view_mask.tolist() != [True, True, False]:
            raise ValueError(f"Single sample view mask is {view_mask.tolist()}.")
        if int((~action_dim_is_pad).sum()) != 10:
            raise ValueError("Single sample must supervise 10 action dimensions.")
        right_wrist = video[:, :, 256:, 160:]
        if not torch.all(right_wrist == -1):
            raise ValueError("Single-arm right-wrist fill is not normalized black.")
        if bool(spatial_mask[256:, 160:].any()):
            raise ValueError("Single-arm right-wrist loss mask is not zero.")
        generated_black_fraction = float((right_wrist == -1).float().mean())

    return {
        "video_min": float(video.min()),
        "video_max": float(video.max()),
        "valid_action_dims": int((~action_dim_is_pad).sum()),
        "view_role_valid_mask": view_mask.tolist(),
        "generated_black_fraction_in_missing_slot": generated_black_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--samples-per-family", type=int, default=2)
    parser.add_argument("--target-control-hz", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    if args.samples_per_family <= 0:
        parser.error("--samples-per-family must be positive.")

    dataset = InternDataA1V3Dataset(
        root=args.root,
        manifest_dir=args.manifest_dir,
        samples_per_epoch=max(args.samples_per_family * 2, 1),
        epoch_size_multiple=1,
        is_training_set=True,
        val_set_proportion=0.01,
        seed=args.seed,
        target_control_hz=args.target_control_hz,
        num_frames=33,
        action_video_freq_ratio=4,
        max_retries=0,
    )
    if int(dataset.manifest["version"]) != 5:
        raise ValueError("InternData data gate requires manifest v5.")
    contract = build_temporal_contract(args.target_control_hz)
    if dataset.state_native_indices.tolist() != contract["state_indices"].tolist():
        raise ValueError("InternData state indices are not aligned.")
    if dataset.action_native_indices.tolist() != contract["action_indices"].tolist():
        raise ValueError("InternData action indices are not next-state aligned.")
    if dataset.video_indices.tolist() != contract["video_indices"].tolist():
        raise ValueError("InternData video indices are not aligned.")

    family_summary = _family_summary(
        dataset.datasets,
        dataset.arrays["dataset_id"],
        dataset.arrays["length"],
        dataset.native_horizon,
    )
    rng = np.random.default_rng(args.seed)
    decoded = {"dual": [], "single": []}
    for family in decoded:
        candidate_rows = dataset.episode_rows[
            np.asarray(
                [
                    dataset.datasets[int(dataset.arrays["dataset_id"][row])][
                        "family"
                    ]
                    == family
                    for row in dataset.episode_rows
                ],
                dtype=np.bool_,
            )
        ]
        if candidate_rows.size < args.samples_per_family:
            raise ValueError(f"Not enough `{family}` episodes for the data gate.")
        chosen = rng.choice(
            candidate_rows,
            size=args.samples_per_family,
            replace=False,
        )
        for episode_row in chosen:
            clip_count = (
                int(dataset.arrays["length"][episode_row])
                - dataset.native_horizon
            )
            clip_start = int(rng.integers(max(clip_count, 1)))
            sample_index = dataset._encode_index(int(episode_row), clip_start)
            decoded[family].append(
                _validate_sample(dataset[sample_index], family)
            )

    result = {
        "status": "PASS",
        "target_control_hz": dataset.target_control_hz,
        "native_window_frames": dataset.native_window_frames,
        "manifest": dataset.manifest,
        "family_summary": family_summary,
        "decoded_samples": decoded,
        "black_fill_policy": {
            "dual": "No generated black slots; head/left/right are required.",
            "single": (
                "Right-wrist L3 slot is generated black, "
                "view_role_valid_mask[2]=0, and its video-loss mask is zero."
            ),
        },
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
