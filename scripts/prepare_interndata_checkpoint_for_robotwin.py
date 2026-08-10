#!/usr/bin/env python3
"""Drop embodiment-specific heads from an InternData checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ACTION_HEAD_PREFIXES = (
    "mixtures.action.action_encoder.",
    "mixtures.action.head.",
)


def convert(input_path: Path, output_path: Path) -> None:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    mot_state = payload.get("mot")
    if not isinstance(mot_state, dict):
        raise ValueError(f"Checkpoint has no `mot` state dict: {input_path}")

    filtered_mot = {
        key: value
        for key, value in mot_state.items()
        if not key.startswith(ACTION_HEAD_PREFIXES)
    }
    removed = sorted(set(mot_state) - set(filtered_mot))
    if not removed:
        raise ValueError("No embodiment-specific ActionDiT head keys were found.")

    converted = {
        key: value
        for key, value in payload.items()
        if key not in {"optimizer", "proprio_encoder"}
    }
    converted["mot"] = filtered_mot
    converted["transfer"] = {
        "source": str(input_path),
        "dropped_action_prefixes": list(ACTION_HEAD_PREFIXES),
        "dropped_proprio_encoder": "proprio_encoder" in payload,
        "action_head_policy": "reinitialize_for_robotwin_14d",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    print(f"Wrote {output_path}")
    print(f"Removed {len(removed)} ActionDiT input/output tensors.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
