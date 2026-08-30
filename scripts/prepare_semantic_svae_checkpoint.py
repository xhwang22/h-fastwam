#!/usr/bin/env python3
"""Extract frozen encoder weights from a semantic-wm S-VAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import torch

from fastwam.models.wan22.semantic_svae import (
    SEMANTIC_SVAE_SOURCE_PATH,
    SEMANTIC_SVAE_SOURCE_REVISION,
    SemanticSVAEEncoder,
    extract_semantic_svae_encoder_state,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_compact(path: Path) -> None:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid compact checkpoint: {path}")
    state_dict, _ = extract_semantic_svae_encoder_state(checkpoint)
    model = SemanticSVAEEncoder()
    model.load_state_dict(state_dict, strict=True)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("source_revision") != SEMANTIC_SVAE_SOURCE_REVISION:
        raise ValueError(f"Unexpected source revision in {path}.")
    if metadata.get("source_path") != SEMANTIC_SVAE_SOURCE_PATH:
        raise ValueError(f"Unexpected source path in {path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--remove-source", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if args.verify_only:
        _validate_compact(output)
        print(f"valid semantic S-VAE checkpoint: {output}")
        return

    if args.source is None:
        parser.error("--source is required unless --verify-only is used")
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")

    source_sha256 = _sha256(source)
    if (
        args.expected_source_sha256
        and source_sha256 != args.expected_source_sha256
    ):
        raise ValueError(
            f"Source SHA-256 mismatch: expected "
            f"{args.expected_source_sha256}, got {source_sha256}"
        )

    checkpoint = torch.load(
        source,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid source checkpoint: {source}")
    state_dict, adapter_config = extract_semantic_svae_encoder_state(checkpoint)
    state_dict = {
        key: value.to(dtype=torch.bfloat16, device="cpu")
        if torch.is_floating_point(value)
        else value.cpu()
        for key, value in state_dict.items()
    }

    compact = {
        "state_dict": state_dict,
        "adapter_config": adapter_config,
        "metadata": {
            "source_repo": "Nilaksh404/semantic-wm",
            "source_revision": SEMANTIC_SVAE_SOURCE_REVISION,
            "source_path": SEMANTIC_SVAE_SOURCE_PATH,
            "source_sha256": source_sha256,
            "source_license": "CC-BY-SA-4.0",
            "precision": "bfloat16",
            "contents": "semantic-wm V-JEPA image S-VAE encoder only",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(compact, temporary)
        _validate_compact(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"wrote compact semantic S-VAE checkpoint: {output}")
    if args.remove_source:
        source.unlink()


if __name__ == "__main__":
    main()
