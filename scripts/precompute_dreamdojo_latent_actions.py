#!/usr/bin/env python3
"""Precompute DreamDojo latent-action targets for a FastWAM dataset split."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.dreamdojo_lam import (  # noqa: E402
    encode_dreamdojo_latent_actions,
    load_dreamdojo_lam,
)
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
    parser.add_argument("--dreamdojo-source-revision", default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
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
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("forkserver", "spawn"),
        default="spawn",
    )
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


class _VideoExtractionDataset(Dataset):
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self.indices[position]
        video, valid = _extract_full_video(self.dataset, index)
        return torch.tensor(index, dtype=torch.long), video, valid


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
    if (
        args.shard_size <= 0
        or args.pair_batch_size <= 0
        or args.sample_batch_size <= 0
        or args.num_workers < 0
        or args.prefetch_factor <= 0
    ):
        raise ValueError(
            "shard-size, pair-batch-size, sample-batch-size, and "
            "prefetch-factor must be positive; num-workers must be "
            "non-negative."
        )
    if args.split == "val" and args.normalization_stats_cache is None:
        raise ValueError(
            "--split val requires --normalization-stats-cache pointing to "
            "the completed training cache."
        )
    if (args.dreamdojo_source_revision is None) != (
        args.checkpoint_sha256 is None
    ):
        raise ValueError(
            "--dreamdojo-source-revision and --checkpoint-sha256 must be "
            "provided together."
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
    num_shards = math.ceil(dataset_length / args.shard_size)
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
                "target_contract": "dreamdojo_adjacent_pair_z_mu_32d_v1",
                "split": args.split,
                "dataset_length": dataset_length,
                "shard_size": args.shard_size,
                "latent_horizon": 32,
                "latent_dim": 32,
                "cache_dtype": args.cache_dtype,
                "dreamdojo_checkpoint_sha256": (
                    args.checkpoint_sha256
                    if args.checkpoint_sha256 is not None
                    else _sha256(checkpoint)
                ),
                "dreamdojo_source_revision": (
                    args.dreamdojo_source_revision
                ),
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
                "encoder_implementation_sha256": _sha256(
                    Path(
                        inspect.unwrap(
                            encode_dreamdojo_latent_actions
                        ).__code__.co_filename
                    ).resolve()
                ),
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
            nonsemantic_fields = {
                "implementation_sha256",
                "encoder_implementation_sha256",
            }

            def mismatched_signature_fields(existing: dict) -> list[str]:
                mismatched = []
                for key, value in cache_signature.items():
                    if key in nonsemantic_fields:
                        continue
                    if key == "target_contract" and key not in existing:
                        continue
                    if existing.get(key) != value:
                        mismatched.append(key)
                return mismatched

            if manifest_path.is_file():
                existing_manifest = load_latent_action_cache_manifest(
                    cache_dir,
                    expected_length=dataset_length,
                    expected_horizon=32,
                    expected_dim=32,
                )
                mismatched_fields = mismatched_signature_fields(
                    existing_manifest
                )
                if mismatched_fields:
                    raise ValueError(
                        "Complete latent-action cache does not match the "
                        "requested checkpoint, dataset, configuration, or "
                        f"implementation; mismatched fields={mismatched_fields}."
                    )
                invalid_shards = [
                    shard_id
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
                setup_status[0] = {
                    "complete": not invalid_shards,
                    "error": None,
                }
                if invalid_shards:
                    print(
                        "Repairing incomplete latent-action cache shards: "
                        f"{invalid_shards[:20]}"
                    )
            else:
                if partial_manifest_path.is_file():
                    with partial_manifest_path.open(
                        "r",
                        encoding="utf-8",
                    ) as handle:
                        existing_signature = json.load(handle)
                    mismatched_fields = mismatched_signature_fields(
                        existing_signature
                    )
                    if mismatched_fields:
                        raise ValueError(
                            "Existing partial latent-action cache was produced "
                            "by a different checkpoint, dataset, configuration, "
                            "or target contract; mismatched fields="
                            f"{mismatched_fields}: {partial_manifest_path}"
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

    model = load_dreamdojo_lam(
        dreamdojo_root,
        checkpoint,
        device,
        expected_source_revision=args.dreamdojo_source_revision,
        expected_checkpoint_sha256=args.checkpoint_sha256,
    )
    output_dtype = _cache_dtype(args.cache_dtype)
    assigned_shards = range(rank, num_shards, world_size)
    pending_indices = []
    for shard_id in assigned_shards:
        first = shard_id * args.shard_size
        last = min(first + args.shard_size, dataset_length)
        if _valid_existing_shard(
            latent_action_shard_path(cache_dir, shard_id),
            first_index=first,
            last_index=last,
        ):
            continue
        pending_indices.extend(range(first, last))

    extraction_dataset = _VideoExtractionDataset(dataset, pending_indices)
    loader_kwargs = {
        "batch_size": args.sample_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = (
            args.multiprocessing_context
        )
        if "in_order" in inspect.signature(DataLoader).parameters:
            loader_kwargs["in_order"] = False
    extraction_loader = DataLoader(extraction_dataset, **loader_kwargs)
    sample_iterator = tqdm(
        extraction_loader,
        desc=f"{args.split} cache rank0/{world_size}",
        total=math.ceil(len(pending_indices) / args.sample_batch_size),
        dynamic_ncols=True,
        disable=rank != 0,
        initial=0,
        unit="batch",
    )
    shard_buffers: dict[int, dict[str, torch.Tensor]] = {}
    shard_sample_counts: dict[int, int] = {}

    def flush_shard(shard_id: int, shard_tensors: dict) -> None:
        first = shard_id * args.shard_size
        last = min(first + args.shard_size, dataset_length)
        _write_shard(
            latent_action_shard_path(cache_dir, shard_id),
            shard_tensors,
            metadata={
                "format": CACHE_FORMAT,
                "version": str(CACHE_VERSION),
                "first_index": str(first),
                "last_index_exclusive": str(last),
            },
        )
        if rank != 0:
            print(
                f"[rank {rank}] wrote shard {shard_id + 1}/{num_shards} "
                f"samples=[{first},{last})"
            )

    for batch_indices, videos, valid_masks in sample_iterator:
        latent_actions = encode_dreamdojo_latent_actions(
            model,
            videos,
            pair_batch_size=args.pair_batch_size,
            device=device,
            preprocess_all_frames=True,
        ).cpu()
        for offset, index_tensor in enumerate(batch_indices):
            index = int(index_tensor.item())
            shard_id = index // args.shard_size
            tensors = shard_buffers.setdefault(shard_id, {})
            tensors[latent_action_tensor_key(index)] = latent_actions[
                offset
            ].to(dtype=output_dtype)
            tensors[f"valid_{index:012d}"] = valid_masks[offset]
            shard_sample_counts[shard_id] = (
                shard_sample_counts.get(shard_id, 0) + 1
            )
            first = shard_id * args.shard_size
            expected_count = min(
                first + args.shard_size,
                dataset_length,
            ) - first
            if shard_sample_counts[shard_id] == expected_count:
                flush_shard(shard_id, tensors)
                del shard_buffers[shard_id]
                del shard_sample_counts[shard_id]
    if shard_buffers:
        raise RuntimeError(
            "DataLoader ended with incomplete latent-action shard buffers: "
            f"{sorted(shard_buffers)}."
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
        partial_manifest_path.unlink(missing_ok=True)
        print(f"Finalized latent-action cache: {cache_dir}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if cache_lock is not None:
        cache_lock.close()


if __name__ == "__main__":
    main()
