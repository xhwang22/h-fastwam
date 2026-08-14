#!/usr/bin/env python3
"""Check that fixed V-JEPA statistics match the intended experiment."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--temporal-downsample", type=int, required=True)
    parser.add_argument("--causal-tubelet-encoding", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.stats, map_location="cpu", weights_only=False)
    expected = {
        "encoder_type": "vjepa2_1",
        "data_config": args.data_config,
        "model_name": args.model_name,
        "temporal_downsample": args.temporal_downsample,
        "causal_tubelet_encoding": args.causal_tubelet_encoding,
        "skip_projection": True,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Incompatible V-JEPA statistics `{args.stats}`: {mismatches}"
        )
    mean = torch.as_tensor(payload.get("mean"))
    std = torch.as_tensor(payload.get("std"))
    if mean.shape != (1664,) or std.shape != (1664,):
        raise ValueError(
            f"Expected 1664-channel mean/std, got {mean.shape}/{std.shape}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("V-JEPA statistics contain NaN/Inf.")
    if torch.any(std <= 0):
        raise ValueError("V-JEPA statistics contain non-positive std.")
    print(
        f"V-JEPA stats compatible: data={args.data_config} "
        f"samples={payload.get('processed_samples')} "
        f"vectors={payload.get('latent_vector_count')}"
    )


if __name__ == "__main__":
    main()
