from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


CACHE_FORMAT = "dreamdojo-latent-action-cache"
CACHE_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = "manifest.partial.json"
LATENT_ACTION_SHAPE = (8, 32)
LATENT_ACTION_IS_PAD_SHAPE = (8,)


class LatentActionCacheError(RuntimeError):
    pass


def canonical_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_filename(shard_id: int) -> str:
    return f"shard_{int(shard_id):08d}.safetensors"


def shard_path(cache_dir: str | Path, shard_id: int) -> Path:
    return Path(cache_dir).expanduser().resolve() / shard_filename(shard_id)


def _require_int(manifest: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LatentActionCacheError(
            f"Latent action cache manifest `{key}` must be an integer >= {minimum}, got {value!r}."
        )
    return value


def _validate_shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    if not isinstance(value, list) or value != list(expected):
        raise LatentActionCacheError(
            f"Latent action cache `{name}` mismatch: expected {list(expected)}, got {value!r}."
        )


def validate_latent_action_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    expected_length: int | None = None,
    expected_signature: str | None = None,
    require_complete: bool = True,
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise LatentActionCacheError("Latent action cache manifest must be a JSON object.")
    manifest = dict(manifest)
    if manifest.get("format") != CACHE_FORMAT:
        raise LatentActionCacheError(
            f"Latent action cache format mismatch: expected {CACHE_FORMAT!r}, "
            f"got {manifest.get('format')!r}."
        )
    if manifest.get("version") != CACHE_VERSION:
        raise LatentActionCacheError(
            f"Latent action cache version mismatch: expected {CACHE_VERSION}, "
            f"got {manifest.get('version')!r}."
        )
    if require_complete and manifest.get("complete") is not True:
        raise LatentActionCacheError("Latent action cache is not complete.")
    if not isinstance(manifest.get("complete"), bool):
        raise LatentActionCacheError("Latent action cache `complete` must be boolean.")

    dataset_length = _require_int(manifest, "dataset_length", minimum=1)
    shard_size = _require_int(manifest, "shard_size", minimum=1)
    num_shards = _require_int(manifest, "num_shards", minimum=1)
    expected_num_shards = math.ceil(dataset_length / shard_size)
    if num_shards != expected_num_shards:
        raise LatentActionCacheError(
            f"Latent action cache shard count mismatch: expected {expected_num_shards}, got {num_shards}."
        )
    if expected_length is not None and dataset_length != int(expected_length):
        raise LatentActionCacheError(
            f"Latent action cache dataset length mismatch: expected {int(expected_length)}, "
            f"got {dataset_length}."
        )

    signature = manifest.get("signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise LatentActionCacheError(
            f"Latent action cache signature must be a SHA256 hex digest, got {signature!r}."
        )
    try:
        bytes.fromhex(signature)
    except ValueError as exc:
        raise LatentActionCacheError(
            f"Latent action cache signature is not hexadecimal: {signature!r}."
        ) from exc
    signature_payload = manifest.get("signature_payload")
    if not isinstance(signature_payload, Mapping):
        raise LatentActionCacheError(
            "Latent action cache `signature_payload` must be a JSON object."
        )
    try:
        actual_signature = canonical_signature(signature_payload)
    except (TypeError, ValueError) as exc:
        raise LatentActionCacheError(
            "Latent action cache `signature_payload` is not canonical JSON."
        ) from exc
    if signature != actual_signature:
        raise LatentActionCacheError(
            "Latent action cache signature does not match `signature_payload`: "
            f"expected {actual_signature}, got {signature}."
        )
    if expected_signature is not None and signature != str(expected_signature):
        raise LatentActionCacheError(
            "Latent action cache signature mismatch: "
            f"expected {expected_signature}, got {signature}."
        )

    _validate_shape(manifest.get("latent_action_shape"), LATENT_ACTION_SHAPE, "latent_action_shape")
    _validate_shape(
        manifest.get("latent_action_is_pad_shape"),
        LATENT_ACTION_IS_PAD_SHAPE,
        "latent_action_is_pad_shape",
    )
    normalization = manifest.get("normalization")
    if not isinstance(normalization, Mapping) or normalization.get("type") != "standardize":
        raise LatentActionCacheError(
            "Latent action cache `normalization` must describe standardize mean/std statistics."
        )
    signed_normalization = signature_payload.get("normalization")
    if not isinstance(signed_normalization, Mapping) or dict(normalization) != dict(signed_normalization):
        raise LatentActionCacheError(
            "Latent action cache `normalization` must exactly match "
            "`signature_payload.normalization`."
        )
    mean = torch.as_tensor(normalization.get("mean"), dtype=torch.float64)
    std = torch.as_tensor(normalization.get("std"), dtype=torch.float64)
    if mean.shape != (LATENT_ACTION_SHAPE[1],) or std.shape != (LATENT_ACTION_SHAPE[1],):
        raise LatentActionCacheError(
            "Latent action normalization mean/std must each have shape [32]."
        )
    if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(std).all().item()):
        raise LatentActionCacheError("Latent action normalization contains non-finite values.")
    if not bool((std > 0).all().item()):
        raise LatentActionCacheError("Latent action normalization std must be strictly positive.")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != num_shards:
        raise LatentActionCacheError(
            f"Latent action cache must list exactly {num_shards} shards, got "
            f"{len(shards) if isinstance(shards, list) else type(shards).__name__}."
        )
    normalized_shards = []
    root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    for shard_id, record in enumerate(shards):
        if not isinstance(record, Mapping):
            raise LatentActionCacheError(f"Shard record {shard_id} must be an object.")
        record = dict(record)
        start = shard_id * shard_size
        stop = min(start + shard_size, dataset_length)
        expected = {
            "shard_id": shard_id,
            "filename": shard_filename(shard_id),
            "index_start": start,
            "index_stop": stop,
            "sample_count": stop - start,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise LatentActionCacheError(
                    f"Shard {shard_id} `{key}` mismatch: expected {value!r}, got {record.get(key)!r}."
                )
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise LatentActionCacheError(f"Shard {shard_id} has an invalid SHA256 digest.")
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise LatentActionCacheError(
                f"Shard {shard_id} SHA256 digest is not hexadecimal."
            ) from exc
        if root is not None:
            path = root / expected["filename"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing latent action cache shard: {path}")
            if verify_shard_hashes:
                actual_digest = sha256_file(path)
                if actual_digest != digest:
                    raise LatentActionCacheError(
                        f"Shard {shard_id} SHA256 mismatch: expected {digest}, "
                        f"got {actual_digest}."
                    )
        normalized_shards.append(record)
    manifest["shards"] = normalized_shards
    manifest["_validated"] = True
    return manifest


def load_latent_action_cache_manifest(
    cache_dir: str | Path,
    *,
    expected_length: int | None = None,
    expected_signature: str | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing latent action cache manifest: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LatentActionCacheError(
            f"Failed to read latent action cache manifest: {manifest_path}"
        ) from exc
    return validate_latent_action_cache_manifest(
        manifest,
        cache_dir=cache_dir,
        expected_length=expected_length,
        expected_signature=expected_signature,
        require_complete=True,
    )


def validate_latent_action_tensors(
    latent_action: torch.Tensor,
    latent_action_is_pad: torch.Tensor,
    *,
    batch: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_shape = (-1, *LATENT_ACTION_SHAPE) if batch else LATENT_ACTION_SHAPE
    mask_shape = (-1, *LATENT_ACTION_IS_PAD_SHAPE) if batch else LATENT_ACTION_IS_PAD_SHAPE
    if not torch.is_tensor(latent_action) or latent_action.ndim != len(latent_shape):
        raise LatentActionCacheError(
            f"`latent_action` must have shape {latent_shape}, got "
            f"{getattr(latent_action, 'shape', None)}."
        )
    if tuple(latent_action.shape[-2:]) != LATENT_ACTION_SHAPE:
        raise LatentActionCacheError(
            f"`latent_action` must end with {LATENT_ACTION_SHAPE}, got {tuple(latent_action.shape)}."
        )
    if not latent_action.dtype.is_floating_point:
        raise LatentActionCacheError(
            f"`latent_action` must be floating point, got {latent_action.dtype}."
        )
    if not bool(torch.isfinite(latent_action).all().item()):
        raise LatentActionCacheError("`latent_action` contains non-finite values.")
    if not torch.is_tensor(latent_action_is_pad) or latent_action_is_pad.ndim != len(mask_shape):
        raise LatentActionCacheError(
            f"`latent_action_is_pad` must have shape {mask_shape}, got "
            f"{getattr(latent_action_is_pad, 'shape', None)}."
        )
    if tuple(latent_action_is_pad.shape[-1:]) != LATENT_ACTION_IS_PAD_SHAPE:
        raise LatentActionCacheError(
            "`latent_action_is_pad` must end with "
            f"{LATENT_ACTION_IS_PAD_SHAPE}, got {tuple(latent_action_is_pad.shape)}."
        )
    if latent_action_is_pad.dtype != torch.bool:
        raise LatentActionCacheError(
            f"`latent_action_is_pad` must be bool, got {latent_action_is_pad.dtype}."
        )
    if batch and latent_action.shape[0] != latent_action_is_pad.shape[0]:
        raise LatentActionCacheError(
            "Latent action batch size mismatch: "
            f"{latent_action.shape[0]} != {latent_action_is_pad.shape[0]}."
        )
    return latent_action.contiguous(), latent_action_is_pad.contiguous()


def write_latent_action_shard(
    cache_dir: str | Path,
    shard_id: int,
    sample_indices: Sequence[int] | torch.Tensor,
    latent_action: torch.Tensor,
    latent_action_is_pad: torch.Tensor,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    shard_id = int(shard_id)
    indices = torch.as_tensor(sample_indices, dtype=torch.int64, device="cpu").contiguous()
    if indices.ndim != 1 or indices.numel() == 0:
        raise LatentActionCacheError("`sample_indices` must be a non-empty 1D sequence.")
    latent_action, latent_action_is_pad = validate_latent_action_tensors(
        latent_action.detach().to(device="cpu"),
        latent_action_is_pad.detach().to(device="cpu"),
        batch=True,
    )
    if indices.shape[0] != latent_action.shape[0]:
        raise LatentActionCacheError(
            f"Shard sample count mismatch: {indices.shape[0]} indices and "
            f"{latent_action.shape[0]} latent actions."
        )
    if not torch.equal(indices, torch.arange(indices[0], indices[0] + len(indices))):
        raise LatentActionCacheError("Shard sample indices must be contiguous and increasing.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / shard_filename(shard_id)
    temporary_path = cache_dir / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    try:
        save_file(
            {
                "sample_indices": indices,
                "latent_action": latent_action,
                "latent_action_is_pad": latent_action_is_pad,
            },
            str(temporary_path),
            metadata={"format": CACHE_FORMAT, "version": str(CACHE_VERSION)},
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "shard_id": shard_id,
        "filename": output_path.name,
        "index_start": int(indices[0].item()),
        "index_stop": int(indices[-1].item()) + 1,
        "sample_count": int(indices.numel()),
        "sha256": sha256_file(output_path),
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_latent_action(
    cache_dir: str | Path,
    manifest: Mapping[str, Any],
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not isinstance(manifest, Mapping) or manifest.get("_validated") is not True:
        manifest = validate_latent_action_cache_manifest(
            manifest,
            cache_dir=cache_dir,
            require_complete=True,
            verify_shard_hashes=False,
        )
    index = int(index)
    dataset_length = int(manifest["dataset_length"])
    if index < 0 or index >= dataset_length:
        raise IndexError(
            f"Latent action cache index {index} is outside [0, {dataset_length})."
        )
    shard_size = int(manifest["shard_size"])
    shard_id = index // shard_size
    record = manifest["shards"][shard_id]
    row = index - int(record["index_start"])
    path = cache_dir / record["filename"]
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"sample_indices", "latent_action", "latent_action_is_pad"}
            if keys != required:
                raise LatentActionCacheError(
                    f"Shard {path} tensor keys mismatch: expected {sorted(required)}, got {sorted(keys)}."
                )
            stored_index = int(handle.get_slice("sample_indices")[row].item())
            latent_action = handle.get_slice("latent_action")[row]
            latent_action_is_pad = handle.get_slice("latent_action_is_pad")[row]
    except LatentActionCacheError:
        raise
    except Exception as exc:
        raise LatentActionCacheError(
            f"Failed to read latent action sample {index} from {path}."
        ) from exc
    if stored_index != index:
        raise LatentActionCacheError(
            f"Shard {path} row {row} stores sample {stored_index}, expected {index}."
        )
    latent_action, latent_action_is_pad = validate_latent_action_tensors(
        latent_action,
        latent_action_is_pad,
        batch=False,
    )
    normalization = manifest["normalization"]
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)
    latent_action = (latent_action.float() - mean) / std
    latent_action[latent_action_is_pad] = 0.0
    return latent_action, latent_action_is_pad
