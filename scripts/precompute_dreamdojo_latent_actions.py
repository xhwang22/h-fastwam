#!/usr/bin/env python3
"""Precompute DreamDojo latent-action targets for a FastWAM dataset split."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.utils.latent_action_cache import (  # noqa: E402
    CACHE_FORMAT,
    CACHE_VERSION,
    MANIFEST_FILENAME,
    latent_action_shard_path,
    latent_action_tensor_key,
    load_latent_action_cache_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="robotwin_interleaved_webdataset")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dreamdojo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--normalization-stats-cache",
        default=None,
        help=(
            "Completed training cache whose mean/std should be reused. "
            "Required for --split val."
        ),
    )
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument(
        "--cache-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides applied after the full-frame settings.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_index_fingerprint(dataset) -> str | None:
    selected_episode_ids = getattr(dataset, "selected_episode_ids", None)
    if selected_episode_ids is None:
        return None
    payload = json.dumps(
        [int(value) for value in selected_episode_ids],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("DreamDojo latent-action extraction requires CUDA.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(minutes=30),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size, torch.device("cuda", local_rank)


def _build_dataset(args: argparse.Namespace):
    overrides = [
        f"data={args.data_config}",
        f"seed={args.seed}",
        "data.train.action_video_freq_ratio=1",
        "data.val.action_video_freq_ratio=1",
        "data.train.num_segments=1",
        "data.val.num_segments=1",
        "++data.train.latent_action_cache_dir=null",
        "++data.val.latent_action_cache_dir=null",
        *args.overrides,
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs"),
    ):
        cfg = compose(config_name="train", overrides=overrides)
    dataset_cfg = cfg.data.get(args.split)
    if dataset_cfg is None:
        raise ValueError(
            f"Data config {args.data_config!r} has no {args.split!r} split."
        )
    return instantiate(dataset_cfg), OmegaConf.to_container(
        dataset_cfg,
        resolve=True,
    )


def _load_lam(
    dreamdojo_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not dreamdojo_root.is_dir():
        raise FileNotFoundError(f"DreamDojo checkout not found: {dreamdojo_root}")
    if not (
        dreamdojo_root / "external" / "lam" / "modules" / "lam.py"
    ).is_file():
        raise FileNotFoundError(
            "DreamDojo checkout is missing external/lam/modules/lam.py: "
            f"{dreamdojo_root}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DreamDojo LAM checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(dreamdojo_root))
    from external.lam.modules.lam import LatentActionModel

    model = LatentActionModel(
        in_dim=3,
        model_dim=1024,
        latent_dim=32,
        patch_size=16,
        enc_blocks=24,
        dec_blocks=24,
        num_heads=16,
        dropout=0.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"),
        dict,
    ):
        raise ValueError(
            "DreamDojo LAM checkpoint must contain a `state_dict` mapping."
        )
    wrapper_state = payload["state_dict"]
    non_lam_keys = [
        key for key in wrapper_state if not key.startswith("lam.")
    ]
    if non_lam_keys:
        raise ValueError(
            "DreamDojo checkpoint contains unexpected non-LAM state keys: "
            f"{non_lam_keys[:20]}."
        )
    lam_state = {
        key.removeprefix("lam."): value
        for key, value in wrapper_state.items()
    }
    missing, unexpected = model.load_state_dict(
        lam_state,
        strict=False,
        assign=True,
    )
    if missing or unexpected:
        raise ValueError(
            "DreamDojo LAM checkpoint does not exactly match the official "
            f"1024D/24+24/32D architecture: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}."
        )
    del model.patch_up
    del model.action_up
    del model.decoder
    model.eval()
    model.requires_grad_(False)
    return model.to(device=device, dtype=torch.bfloat16)


def _extract_full_video(
    dataset,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(dataset, "_get"):
        raise TypeError(
            "Latent-action precompute requires a RobotVideoDataset-compatible "
            "`_get(index)` method."
        )
    sample = dataset._get(int(index))
    actual_index = int(sample.get("sample_idx", index))
    if actual_index != int(index):
        raise RuntimeError(
            f"Requested sample {index}, but dataset returned {actual_index}."
        )
    video = sample.get("video")
    if not torch.is_tensor(video) or video.ndim != 4:
        raise ValueError(
            "Dataset must return unbatched video [3,33,H,W], "
            f"got {type(video)} with shape {getattr(video, 'shape', None)}."
        )
    if tuple(video.shape[:2]) != (3, 33):
        raise ValueError(
            "DreamDojo targets require the complete 33-frame window, "
            f"got {tuple(video.shape)}."
        )
    action_is_pad = sample.get("action_is_pad")
    if not torch.is_tensor(action_is_pad) or action_is_pad.shape != (32,):
        raise ValueError(
            "Dataset must return action_is_pad [32] for latent-target "
            f"normalization, got {getattr(action_is_pad, 'shape', None)}."
        )
    return video, ~action_is_pad.to(dtype=torch.bool)


@torch.inference_mode()
def _encode_video_pairs(
    model: torch.nn.Module,
    video: torch.Tensor,
    *,
    pair_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    frames = video.float().clamp(-1.0, 1.0)
    frames = F.interpolate(
        frames.permute(1, 0, 2, 3),
        size=(240, 320),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    frames = ((frames + 1.0) * 0.5).permute(0, 2, 3, 1).contiguous()
    pairs = torch.stack((frames[:-1], frames[1:]), dim=1)
    outputs = []
    for start in range(0, pairs.shape[0], pair_batch_size):
        pair_batch = pairs[start : start + pair_batch_size].to(
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = model.encode(pair_batch)
        z_rep = encoded.get("z_rep")
        if not torch.is_tensor(z_rep) or z_rep.shape[1:] != (1, 1, 32):
            raise ValueError(
                "DreamDojo LAM must return z_rep [B,1,1,32], "
                f"got {type(z_rep)} with shape {getattr(z_rep, 'shape', None)}."
            )
        outputs.append(z_rep[:, 0, 0].float().cpu())
    latent_actions = torch.cat(outputs, dim=0)
    if latent_actions.shape != (32, 32):
        raise ValueError(
            f"Expected latent-action target [32,32], got "
            f"{tuple(latent_actions.shape)}."
        )
    if not bool(torch.isfinite(latent_actions).all().item()):
        raise ValueError("DreamDojo LAM produced non-finite latent actions.")
    return latent_actions


def _cache_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _valid_existing_shard(
    path: Path,
    *,
    first_index: int,
    last_index: int,
) -> bool:
    if not path.is_file():
        return False
    expected = set()
    for index in range(first_index, last_index):
        expected.add(latent_action_tensor_key(index))
        expected.add(f"valid_{int(index):012d}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return set(handle.keys()) == expected
    except Exception:
        return False


def _write_shard(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def _compute_cache_stats(
    cache_dir: Path,
    *,
    dataset_length: int,
    shard_size: int,
) -> tuple[list[float], list[float]]:
    value_sum = torch.zeros(32, dtype=torch.float64)
    value_square_sum = torch.zeros(32, dtype=torch.float64)
    count = 0
    num_shards = math.ceil(dataset_length / shard_size)
    for shard_id in range(num_shards):
        shard_path = latent_action_shard_path(cache_dir, shard_id)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith("latent_action_"):
                    continue
                index = int(key.rsplit("_", 1)[-1])
                values = handle.get_tensor(key).double().reshape(-1, 32)
                valid = handle.get_tensor(
                    f"valid_{index:012d}"
                ).to(dtype=torch.bool)
                if valid.shape != (values.shape[0],):
                    raise ValueError(
                        f"Validity mask shape mismatch for cache index {index}: "
                        f"{tuple(valid.shape)} vs {(values.shape[0],)}."
                    )
                values = values[valid]
                if values.numel() == 0:
                    continue
                value_sum += values.sum(dim=0)
                value_square_sum += values.square().sum(dim=0)
                count += int(values.shape[0])
    if count == 0:
        raise RuntimeError("Cannot compute statistics for an empty cache.")
    mean = value_sum / count
    variance = (value_square_sum / count - mean.square()).clamp_min(1.0e-12)
    return mean.tolist(), variance.sqrt().tolist()


def main() -> None:
    args = _parse_args()
    if args.shard_size <= 0 or args.pair_batch_size <= 0:
        raise ValueError("shard-size and pair-batch-size must be positive.")
    if args.split == "val" and args.normalization_stats_cache is None:
        raise ValueError(
            "--split val requires --normalization-stats-cache pointing to "
            "the completed training cache."
        )
    rank, world_size, device = _init_distributed()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dreamdojo_root = Path(args.dreamdojo_root).expanduser().resolve()

    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    cache_lock = None
    if rank == 0:
        lock_path = cache_dir / ".cache.lock"
        cache_lock = lock_path.open("a+")
        try:
            fcntl.flock(
                cache_lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another latent-action cache writer holds {lock_path}."
            ) from exc

    dataset, data_cfg = _build_dataset(args)
    dataset_length = len(dataset)
    manifest_path = cache_dir / MANIFEST_FILENAME
    partial_manifest_path = cache_dir / "manifest.partial.json"
    cache_signature = None
    setup_status = [None]
    if rank == 0:
        try:
            data_root = Path(
                str(data_cfg["preprocessed_root"])
            ).expanduser().resolve()
            data_manifest = data_root / "manifest.json"
            dreamdojo_sources = [
                dreamdojo_root / "external" / "lam" / "model.py",
                dreamdojo_root / "external" / "lam" / "modules" / "lam.py",
                dreamdojo_root / "external" / "lam" / "modules" / "blocks.py",
            ]
            missing_sources = [
                str(path) for path in dreamdojo_sources if not path.is_file()
            ]
            if missing_sources:
                raise FileNotFoundError(
                    f"DreamDojo LAM source files are missing: {missing_sources}"
                )
            cache_signature = {
                "format": CACHE_FORMAT,
                "version": CACHE_VERSION,
                "split": args.split,
                "dataset_length": dataset_length,
                "shard_size": args.shard_size,
                "latent_horizon": 32,
                "latent_dim": 32,
                "cache_dtype": args.cache_dtype,
                "dreamdojo_checkpoint_sha256": _sha256(checkpoint),
                "dreamdojo_source_sha256": {
                    str(path.relative_to(dreamdojo_root)): _sha256(path)
                    for path in dreamdojo_sources
                },
                "dataset_manifest_sha256": (
                    _sha256(data_manifest) if data_manifest.is_file() else None
                ),
                "dataset_index_fingerprint": _dataset_index_fingerprint(dataset),
                "data_config": data_cfg,
                "implementation_sha256": _sha256(Path(__file__).resolve()),
                "normalization_stats_cache": (
                    None
                    if args.normalization_stats_cache is None
                    else str(
                        Path(args.normalization_stats_cache)
                        .expanduser()
                        .resolve()
                    )
                ),
                "normalization_stats_manifest_sha256": (
                    None
                    if args.normalization_stats_cache is None
                    else _sha256(
                        Path(args.normalization_stats_cache)
                        .expanduser()
                        .resolve()
                        / MANIFEST_FILENAME
                    )
                ),
            }
            if manifest_path.is_file():
                existing_manifest = load_latent_action_cache_manifest(
                    cache_dir,
                    expected_length=dataset_length,
                    expected_horizon=32,
                    expected_dim=32,
                )
                mismatched_fields = [
                    key
                    for key, value in cache_signature.items()
                    if existing_manifest.get(key) != value
                ]
                if mismatched_fields:
                    raise ValueError(
                        "Complete latent-action cache does not match the "
                        "requested checkpoint, dataset, configuration, or "
                        f"implementation; mismatched fields={mismatched_fields}."
                    )
                setup_status[0] = {"complete": True, "error": None}
            else:
                if partial_manifest_path.is_file():
                    with partial_manifest_path.open(
                        "r",
                        encoding="utf-8",
                    ) as handle:
                        existing_signature = json.load(handle)
                    if existing_signature != cache_signature:
                        raise ValueError(
                            "Existing partial latent-action cache was produced "
                            "by a different checkpoint, dataset, configuration, "
                            f"or implementation: {partial_manifest_path}"
                        )
                else:
                    _atomic_json_dump(cache_signature, partial_manifest_path)
                setup_status[0] = {"complete": False, "error": None}
        except Exception as exc:
            setup_status[0] = {
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if dist.is_initialized():
        dist.broadcast_object_list(setup_status, src=0)
    if setup_status[0]["error"] is not None:
        raise RuntimeError(
            "Failed to initialize DreamDojo latent-action cache: "
            f"{setup_status[0]['error']}"
        )
    if setup_status[0]["complete"]:
        if rank == 0:
            print(f"Complete latent-action cache already exists: {cache_dir}")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if cache_lock is not None:
            cache_lock.close()
        return
    if dist.is_initialized():
        dist.barrier()

    model = _load_lam(dreamdojo_root, checkpoint, device)
    num_shards = math.ceil(dataset_length / args.shard_size)
    output_dtype = _cache_dtype(args.cache_dtype)
    for shard_id in range(rank, num_shards, world_size):
        first = shard_id * args.shard_size
        last = min(first + args.shard_size, dataset_length)
        shard_path = latent_action_shard_path(cache_dir, shard_id)
        if _valid_existing_shard(
            shard_path,
            first_index=first,
            last_index=last,
        ):
            print(f"[rank {rank}] keeping complete shard {shard_id}/{num_shards}")
            continue
        tensors = {}
        for index in range(first, last):
            video, valid = _extract_full_video(dataset, index)
            latent_action = _encode_video_pairs(
                model,
                video,
                pair_batch_size=args.pair_batch_size,
                device=device,
            )
            tensors[latent_action_tensor_key(index)] = latent_action.to(
                dtype=output_dtype
            )
            tensors[f"valid_{index:012d}"] = valid
        _write_shard(
            shard_path,
            tensors,
            metadata={
                "format": CACHE_FORMAT,
                "version": str(CACHE_VERSION),
                "first_index": str(first),
                "last_index_exclusive": str(last),
            },
        )
        print(
            f"[rank {rank}] wrote shard {shard_id + 1}/{num_shards} "
            f"samples=[{first},{last})"
        )

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        missing = [
            str(latent_action_shard_path(cache_dir, shard_id))
            for shard_id in range(num_shards)
            if not _valid_existing_shard(
                latent_action_shard_path(cache_dir, shard_id),
                first_index=shard_id * args.shard_size,
                last_index=min(
                    (shard_id + 1) * args.shard_size,
                    dataset_length,
                ),
            )
        ]
        if missing:
            raise RuntimeError(
                f"Cannot finalize cache; incomplete shards: {missing[:20]}"
            )
        normalization_source = None
        checkpoint_sha256 = cache_signature["dreamdojo_checkpoint_sha256"]
        if args.normalization_stats_cache is None:
            mean, std = _compute_cache_stats(
                cache_dir,
                dataset_length=dataset_length,
                shard_size=args.shard_size,
            )
        else:
            normalization_source = str(
                Path(args.normalization_stats_cache).expanduser().resolve()
            )
            stats_manifest = load_latent_action_cache_manifest(
                normalization_source,
                expected_horizon=32,
                expected_dim=32,
            )
            reference_checkpoint_sha256 = stats_manifest.get(
                "dreamdojo_checkpoint_sha256"
            )
            if (
                reference_checkpoint_sha256 is not None
                and reference_checkpoint_sha256 != checkpoint_sha256
            ):
                raise ValueError(
                    "Validation and training latent-action caches must use the "
                    "same DreamDojo checkpoint."
                )
            mean = list(stats_manifest["mean"])
            std = list(stats_manifest["std"])
        manifest = {
            **cache_signature,
            "complete": True,
            "num_shards": num_shards,
            "mean": mean,
            "std": std,
            "normalization_stats_cache": normalization_source,
            "dreamdojo_root": str(dreamdojo_root),
            "dreamdojo_checkpoint": str(checkpoint),
            "dreamdojo_checkpoint_sha256": checkpoint_sha256,
        }
        _atomic_json_dump(manifest, manifest_path)
        partial_manifest_path.unlink()
        print(f"Finalized latent-action cache: {cache_dir}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if cache_lock is not None:
        cache_lock.close()


if __name__ == "__main__":
    main()
