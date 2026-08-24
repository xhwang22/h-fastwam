from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file


CACHE_FORMAT = "dreamdojo-latent-action"
CACHE_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


class LatentActionCacheError(RuntimeError):
    pass


@lru_cache(maxsize=16)
def _load_shard(path: str) -> dict[str, torch.Tensor]:
    return load_file(path, device="cpu")


def latent_action_shard_path(cache_dir: str | Path, shard_id: int) -> Path:
    return Path(cache_dir).expanduser().resolve() / (
        f"shard_{int(shard_id):08d}.safetensors"
    )


def latent_action_tensor_key(index: int) -> str:
    return f"latent_action_{int(index):012d}"


def load_latent_action_cache_manifest(
    cache_dir: str | Path,
    *,
    expected_length: int | None = None,
    expected_horizon: int | None = None,
    expected_dim: int | None = None,
) -> dict:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing latent-action cache manifest: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(
            f"Invalid latent-action cache format at {cache_dir}: "
            f"{manifest.get('format')!r}."
        )
    if int(manifest.get("version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"Latent-action cache version mismatch at {cache_dir}: "
            f"expected {CACHE_VERSION}, got {manifest.get('version')}."
        )
    if not bool(manifest.get("complete", False)):
        raise RuntimeError(f"Latent-action cache is incomplete: {cache_dir}.")

    expected_fields = {
        "dataset_length": expected_length,
        "latent_horizon": expected_horizon,
        "latent_dim": expected_dim,
    }
    for field, expected in expected_fields.items():
        if expected is None:
            continue
        actual = int(manifest.get(field, -1))
        if actual != int(expected):
            raise ValueError(
                f"Latent-action cache {field} mismatch at {cache_dir}: "
                f"expected {expected}, got {actual}."
            )
    shard_size = int(manifest.get("shard_size", 0))
    if shard_size <= 0:
        raise ValueError(
            f"Latent-action cache has invalid shard_size={shard_size}."
        )

    mean = manifest.get("mean")
    std = manifest.get("std")
    latent_dim = int(manifest.get("latent_dim", -1))
    if not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError("Latent-action cache manifest must contain mean/std.")
    if len(mean) != latent_dim or len(std) != latent_dim:
        raise ValueError(
            "Latent-action cache mean/std dimension mismatch: "
            f"dim={latent_dim}, mean={len(mean)}, std={len(std)}."
        )
    if min(float(value) for value in std) <= 0.0:
        raise ValueError("Latent-action cache std values must be positive.")
    return manifest


def load_latent_action(
    cache_dir: str | Path,
    manifest: dict,
    index: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    index = int(index)
    dataset_length = int(manifest["dataset_length"])
    if index < 0 or index >= dataset_length:
        raise IndexError(
            f"Latent-action cache index {index} is outside [0,{dataset_length})."
        )
    shard_size = int(manifest["shard_size"])
    shard_path = latent_action_shard_path(cache_dir, index // shard_size)
    if not shard_path.is_file():
        raise FileNotFoundError(
            f"Missing latent-action cache shard: {shard_path}"
        )
    key = latent_action_tensor_key(index)
    shard = _load_shard(str(shard_path))
    if key not in shard:
        raise KeyError(
            f"Missing `{key}` in latent-action shard {shard_path}."
        )
    latent_action = shard[key]

    expected_shape = (
        int(manifest["latent_horizon"]),
        int(manifest["latent_dim"]),
    )
    if tuple(latent_action.shape) != expected_shape:
        raise ValueError(
            f"Cached latent action must be {expected_shape}, "
            f"got {tuple(latent_action.shape)} for index {index}."
        )
    latent_action = latent_action.float()
    if normalize:
        mean = torch.tensor(manifest["mean"], dtype=torch.float32)
        std = torch.tensor(manifest["std"], dtype=torch.float32)
        latent_action = (latent_action - mean) / std
    if not bool(torch.isfinite(latent_action).all().item()):
        raise ValueError(
            f"Cached latent action contains non-finite values at index {index}."
        )
    return latent_action
