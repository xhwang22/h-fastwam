#!/usr/bin/env python3
"""Compute fixed global V-JEPA 2.1 output mean/std on a FastWAM dataset."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import tempfile
from datetime import timedelta
from itertools import islice
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger(__name__)


class WelfordAccumulator:
    """Numerically stable global per-channel statistics."""

    def __init__(self, num_channels: int):
        self.n = 0
        self.mean = torch.zeros(num_channels, dtype=torch.float64)
        self.m2 = torch.zeros(num_channels, dtype=torch.float64)

    def update(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5:
            raise ValueError(
                f"V-JEPA latents must be [B,D,T,H,W], got {tuple(latents.shape)}."
            )
        channels = int(latents.shape[1])
        values = (
            latents.detach()
            .to(device="cpu", dtype=torch.float64)
            .permute(0, 2, 3, 4, 1)
            .reshape(-1, channels)
        )
        batch_n = int(values.shape[0])
        if batch_n == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        new_n = self.n + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * (batch_n / new_n)
        self.m2 += (
            batch_var * batch_n
            + delta.square() * (self.n * batch_n / new_n)
        )
        self.n = new_n

    def merge(self, other: "WelfordAccumulator") -> None:
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean.copy_(other.mean)
            self.m2.copy_(other.m2)
            return
        new_n = self.n + other.n
        delta = other.mean - self.mean
        self.mean += delta * (other.n / new_n)
        self.m2 += (
            other.m2
            + delta.square() * (self.n * other.n / new_n)
        )
        self.n = new_n

    @property
    def std(self) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("Cannot compute statistics without latent vectors.")
        return (self.m2 / self.n).sqrt()


def _init_runtime(
    shard_rank_offset: int | None,
    shard_world_size: int | None,
) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("V-JEPA 2.1 statistics require CUDA.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if shard_rank_offset is not None:
        if shard_world_size is None:
            raise ValueError("--shard-world-size is required in sharded mode.")
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        rank = int(shard_rank_offset) + local_rank
        world_size = int(shard_world_size)
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"Sharded rank {rank} must be in [0, {world_size})."
            )
        if int(shard_rank_offset) + local_world_size > world_size:
            raise ValueError(
                "Local shard ranks exceed --shard-world-size: "
                f"offset={shard_rank_offset}, local_world_size={local_world_size}, "
                f"world_size={world_size}."
            )
        logger.info(
            "Running independent statistics shard rank %d/%d on local GPU %d.",
            rank,
            world_size,
            local_rank,
        )
        return rank, world_size, device

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(minutes=10),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dist.barrier()
        logger.info(
            "Rank %d/%d completed the distributed Gloo connectivity check.",
            rank,
            world_size,
        )
        return rank, world_size, device
    return 0, 1, device


def _build_dataset(data_config: str, seed: int, overrides: list[str]):
    hydra_overrides = [f"data={data_config}", f"seed={seed}", *overrides]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPO_ROOT / "configs"),
    ):
        config = compose(config_name="pretrain", overrides=hydra_overrides)
    if config.data is None or config.data.train is None:
        raise ValueError(f"Data config `{data_config}` has no training dataset.")
    return instantiate(config.data.train)


def _iter_indices(
    dataset,
    seed: int,
    local_batch_size: int,
    rank: int,
    world_size: int,
    max_samples: int | None,
) -> Iterator[int]:
    custom_iterator = getattr(dataset, "iter_epoch_indices", None)
    if callable(custom_iterator):
        indices = custom_iterator(
            seed,
            batch_size=local_batch_size,
            num_processes=world_size,
        )
    else:
        indices = iter(range(len(dataset)))

    stop = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    global_batch_size = int(local_batch_size) * int(world_size)
    consumed = 0
    while consumed < stop:
        take = min(global_batch_size, stop - consumed)
        block = list(islice(indices, take))
        if not block:
            return
        local_start = int(rank) * int(local_batch_size)
        local_end = local_start + int(local_batch_size)
        yield from block[local_start:local_end]
        consumed += len(block)


def _extract_videos(sample: dict) -> list[torch.Tensor]:
    video = sample.get("video")
    if torch.is_tensor(video):
        if video.ndim == 4:
            return [video]
        if video.ndim == 5:
            return list(video.unbind(dim=0))
        raise ValueError(f"Unexpected sample video shape: {tuple(video.shape)}")

    segments = sample.get("segments")
    if isinstance(segments, dict) and torch.is_tensor(segments.get("video")):
        segment_video = segments["video"]
        if segment_video.ndim == 4:
            return [segment_video]
        if segment_video.ndim == 5:
            return list(segment_video.unbind(dim=0))
        raise ValueError(
            f"Unexpected segmented video shape: {tuple(segment_video.shape)}"
        )
    if isinstance(segments, list):
        videos = []
        for segment in segments:
            if isinstance(segment, dict):
                videos.extend(_extract_videos(segment))
        if videos:
            return videos
    raise ValueError("Dataset sample contains no video tensor.")


class _RankShardSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        seed: int,
        local_batch_size: int,
        rank: int,
        world_size: int,
        max_samples: int | None,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.local_batch_size = int(local_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_sample_count = (
            len(dataset)
            if max_samples is None
            else min(int(max_samples), len(dataset))
        )

    def __iter__(self) -> Iterator[int]:
        return _iter_indices(
            dataset=self.dataset,
            seed=self.seed,
            local_batch_size=self.local_batch_size,
            rank=self.rank,
            world_size=self.world_size,
            max_samples=self.global_sample_count,
        )

    def __len__(self) -> int:
        global_batch_size = self.local_batch_size * self.world_size
        full_batches, remainder = divmod(
            self.global_sample_count,
            global_batch_size,
        )
        local_remainder = min(
            max(remainder - self.rank * self.local_batch_size, 0),
            self.local_batch_size,
        )
        return full_batches * self.local_batch_size + local_remainder


class _StatsVideoDataset(Dataset):
    """Strip action/text fields in workers so only video crosses process IPC."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        try:
            return {
                "videos": _extract_videos(self.dataset[int(index)]),
                "error": None,
            }
        except Exception as exc:
            return {
                "videos": [],
                "error": f"sample {index}: {type(exc).__name__}: {exc}",
            }


def _collate_stats_videos(samples: list[dict]) -> dict:
    videos = [
        video
        for sample in samples
        for video in sample["videos"]
    ]
    errors = [
        sample["error"]
        for sample in samples
        if sample["error"] is not None
    ]
    if not videos:
        return {
            "videos": None,
            "sample_count": len(samples),
            "errors": errors,
        }
    shapes = {tuple(video.shape) for video in videos}
    if len(shapes) != 1:
        raise ValueError(
            f"Stats batches require one video shape, got {sorted(shapes)}."
        )
    return {
        "videos": torch.stack(videos, dim=0),
        "sample_count": len(samples),
        "errors": errors,
    }


def _build_stats_loader(
    dataset,
    sampler: Sampler[int],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    multiprocessing_context: str,
) -> DataLoader:
    kwargs = {
        "dataset": _StatsVideoDataset(dataset),
        "batch_size": int(batch_size),
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": _collate_stats_videos,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = True
        kwargs["multiprocessing_context"] = multiprocessing_context
        if "in_order" in inspect.signature(DataLoader).parameters:
            kwargs["in_order"] = False
    return DataLoader(**kwargs)


@torch.no_grad()
def _encode_raw_latents(encoder, videos: torch.Tensor, device: torch.device) -> torch.Tensor:
    videos = videos.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    if bool(getattr(encoder, "causal_tubelet_encoding", False)):
        batch_size, channels, num_frames, height, width = videos.shape
        temporal_patch = int(encoder._temporal_patch)
        temporal_stride = int(encoder.temporal_downsample_factor)
        clips = []
        for frame_index in range(0, num_frames, temporal_stride):
            start = frame_index - temporal_patch + 1
            if start < 0:
                padding = videos[:, :, 0:1].expand(-1, -1, -start, -1, -1)
                clip = torch.cat([padding, videos[:, :, : frame_index + 1]], dim=2)
            else:
                clip = videos[:, :, start : frame_index + 1]
            if clip.shape[2] != temporal_patch:
                raise ValueError(
                    f"Expected {temporal_patch} causal frames, got {clip.shape[2]}."
                )
            clips.append(clip)
        num_states = len(clips)
        flat_clips = torch.stack(clips, dim=1).reshape(
            batch_size * num_states,
            channels,
            temporal_patch,
            height,
            width,
        )
        flat_latents = encoder.encode(flat_clips, device=device)
        if flat_latents.shape[2] != 1:
            raise ValueError(
                "Causal tubelet encoding must produce one latent state per clip, "
                f"got {tuple(flat_latents.shape)}."
            )
        return flat_latents.reshape(
            batch_size,
            num_states,
            flat_latents.shape[1],
            flat_latents.shape[3],
            flat_latents.shape[4],
        ).permute(0, 2, 1, 3, 4).contiguous()

    if bool(getattr(encoder, "causal_prefix_encoding", False)):
        temporal_patch = int(encoder._temporal_patch)
        temporal_stride = int(encoder.temporal_downsample_factor)
        states = []
        for frame_index in range(0, videos.shape[2], temporal_stride):
            prefix = videos[:, :, : frame_index + 1]
            pad_frames = (-prefix.shape[2]) % temporal_patch
            if pad_frames:
                padding = prefix[:, :, 0:1].expand(
                    -1, -1, pad_frames, -1, -1
                )
                prefix = torch.cat([padding, prefix], dim=2)
            prefix_latents = encoder.encode(prefix, device=device)
            states.append(prefix_latents[:, :, -1:])
        return torch.cat(states, dim=2)

    return encoder.encode(videos, device=device)


def _merge_distributed(
    accumulator: WelfordAccumulator,
    num_channels: int,
    rank: int,
    world_size: int,
) -> WelfordAccumulator:
    if world_size == 1:
        return accumulator
    local_payload = {
        "n": accumulator.n,
        "mean": accumulator.mean,
        "m2": accumulator.m2,
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_payload)
    merged = WelfordAccumulator(num_channels)
    if rank == 0:
        for payload in gathered:
            other = WelfordAccumulator(num_channels)
            other.n = int(payload["n"])
            other.mean.copy_(payload["mean"])
            other.m2.copy_(payload["m2"])
            merged.merge(other)
    return merged


def _save_stats(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_stats(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in {path}, got {type(payload).__name__}."
        )
    return payload


def _run_config(args: argparse.Namespace) -> dict:
    return {
        "encoder_type": "vjepa2_1",
        "model_name": args.model_name,
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "repo_path": str(Path(args.repo_path).expanduser().resolve()),
        "data_config": args.data_config,
        "data_overrides": list(args.data_override),
        "seed": args.seed,
        "spatial_downsample": args.spatial_downsample,
        "temporal_downsample": args.temporal_downsample,
        "causal_tubelet_encoding": args.causal_tubelet_encoding,
        "causal_prefix_encoding": args.causal_prefix_encoding,
        "skip_projection": True,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
    }


def _shard_path(shard_output_dir: Path, rank: int, world_size: int) -> Path:
    return shard_output_dir / f"rank-{rank:05d}-of-{world_size:05d}.pt"


def _validate_completed_shard(
    path: Path,
    payload: dict,
    rank: int,
    world_size: int,
    run_config: dict,
) -> None:
    if payload.get("format") != "fastwam_vjepa21_stats_shard":
        raise ValueError(f"Invalid V-JEPA statistics shard format in {path}.")
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported V-JEPA statistics shard version in {path}.")
    if int(payload.get("rank", -1)) != rank:
        raise ValueError(f"Shard rank mismatch in {path}.")
    if int(payload.get("world_size", -1)) != world_size:
        raise ValueError(f"Shard world-size mismatch in {path}.")
    if payload.get("run_config") != run_config:
        raise ValueError(
            f"Shard configuration mismatch in {path}; use a new shard directory."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute fixed global V-JEPA 2.1 output mean/std."
    )
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument(
        "--model-name",
        default="vjepa2_1_vit_gigantic_384",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum dataset samples across all ranks; omit to scan the full split.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("fork", "forkserver", "spawn"),
        default="spawn",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial-downsample", type=int, default=16)
    parser.add_argument("--temporal-downsample", type=int, default=4)
    parser.add_argument("--causal-tubelet-encoding", action="store_true")
    parser.add_argument("--causal-prefix-encoding", action="store_true")
    parser.add_argument(
        "--data-override",
        action="append",
        default=[],
        help="Hydra override, for example data.train.samples_per_epoch=10000.",
    )
    parser.add_argument(
        "--shard-output-dir",
        type=Path,
        default=None,
        help="Write this rank's accumulator to a file instead of using collectives.",
    )
    parser.add_argument(
        "--shard-rank-offset",
        type=int,
        default=None,
        help="Global rank of local rank 0 when writing independent shards.",
    )
    parser.add_argument(
        "--shard-world-size",
        type=int,
        default=None,
        help="Total number of independent statistics shards.",
    )
    parser.add_argument(
        "--resume-shards",
        action="store_true",
        help="Skip a rank when its compatible completed shard already exists.",
    )
    args = parser.parse_args()
    if args.causal_tubelet_encoding and args.causal_prefix_encoding:
        parser.error("Causal tubelet and causal prefix modes are mutually exclusive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive.")
    shard_args = (
        args.shard_output_dir,
        args.shard_rank_offset,
        args.shard_world_size,
    )
    if any(value is not None for value in shard_args) and any(
        value is None for value in shard_args
    ):
        parser.error(
            "--shard-output-dir, --shard-rank-offset, and --shard-world-size "
            "must be provided together."
        )
    if args.resume_shards and args.shard_output_dir is None:
        parser.error("--resume-shards requires sharded output arguments.")
    if args.shard_rank_offset is not None and args.shard_rank_offset < 0:
        parser.error("--shard-rank-offset must be non-negative.")
    if args.shard_world_size is not None and args.shard_world_size <= 0:
        parser.error("--shard-world-size must be positive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rank, world_size, device = _init_runtime(
        args.shard_rank_offset,
        args.shard_world_size,
    )
    run_config = _run_config(args)
    shard_path = None
    if args.shard_output_dir is not None:
        shard_path = _shard_path(
            args.shard_output_dir.expanduser().resolve(),
            rank,
            world_size,
        )
        if args.resume_shards and shard_path.is_file():
            completed_shard = _load_stats(shard_path)
            _validate_completed_shard(
                shard_path,
                completed_shard,
                rank,
                world_size,
                run_config,
            )
            logger.info("Reusing completed statistics shard: %s", shard_path)
            return

    try:
        from fastwam.models.wan22.visual_encoder import VJEPA21Encoder

        encoder = VJEPA21Encoder(
            model_name=args.model_name,
            checkpoint_source="local",
            checkpoint_path=args.checkpoint_path,
            repo_path=args.repo_path,
            skip_projection=True,
            freeze_backbone=True,
            spatial_downsample=args.spatial_downsample,
            temporal_downsample=args.temporal_downsample,
            standardise_output=False,
            causal_tubelet_encoding=args.causal_tubelet_encoding,
            causal_prefix_encoding=args.causal_prefix_encoding,
            normalise_stats_path=None,
            torch_dtype=torch.bfloat16,
        ).to(device)
        encoder.eval()
        dataset = _build_dataset(args.data_config, args.seed, args.data_override)
        sampler = _RankShardSampler(
            dataset=dataset,
            seed=args.seed,
            local_batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            max_samples=args.max_samples,
        )
        loader = _build_stats_loader(
            dataset=dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            multiprocessing_context=args.multiprocessing_context,
        )
        progress = tqdm(
            total=len(sampler),
            desc=f"rank {rank} V-JEPA stats",
            disable=rank != 0,
            unit="video",
        )
        accumulator = WelfordAccumulator(encoder.output_dim)
        processed_videos = 0
        failed_samples = 0
        processed_samples = 0

        for batch in loader:
            sample_count = int(batch["sample_count"])
            processed_samples += sample_count
            errors = batch["errors"]
            failed_samples += len(errors)
            for error in errors[:3]:
                logger.warning("%s", error)
            videos = batch["videos"]
            if videos is not None:
                latents = _encode_raw_latents(encoder, videos, device)
                accumulator.update(latents)
                processed_videos += int(videos.shape[0])
            if rank == 0:
                progress.set_postfix(
                    videos=processed_videos,
                    failed=failed_samples,
                )
                progress.update(sample_count)
        progress.close()
        logger.info(
            "Rank %d finished its local shard: samples=%d videos=%d failed=%d.",
            rank,
            processed_samples,
            processed_videos,
            failed_samples,
        )

        if shard_path is not None:
            shard_payload = {
                "format": "fastwam_vjepa21_stats_shard",
                "format_version": 1,
                "rank": rank,
                "world_size": world_size,
                "run_config": run_config,
                "processed_samples": processed_samples,
                "processed_videos": processed_videos,
                "failed_samples": failed_samples,
                "latent_vector_count": accumulator.n,
                "num_channels": encoder.output_dim,
                "mean": accumulator.mean,
                "m2": accumulator.m2,
            }
            _save_stats(shard_path, shard_payload)
            logger.info(
                "Saved statistics shard rank %d/%d to %s.",
                rank,
                world_size,
                shard_path,
            )
            return

        count_tensor = torch.tensor(
            [processed_samples, processed_videos, failed_samples],
            dtype=torch.long,
        )
        if world_size > 1:
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        merged = _merge_distributed(
            accumulator,
            num_channels=encoder.output_dim,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            mean = merged.mean.float()
            std = merged.std.float()
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise RuntimeError("Computed V-JEPA statistics contain NaN/Inf.")
            if torch.any(std <= 0):
                raise RuntimeError("Computed V-JEPA statistics contain non-positive std.")

            metadata = {
                "version": 1,
                **run_config,
                "normalisation_axes": "global_sample_time_space_per_channel",
                "processed_samples": int(count_tensor[0].item()),
                "processed_videos": int(count_tensor[1].item()),
                "failed_samples": int(count_tensor[2].item()),
                "latent_vector_count": merged.n,
                "num_channels": encoder.output_dim,
                "aggregation_mode": "distributed_gloo",
            }
            output_payload = {
                "mean": mean,
                "std": std,
                **metadata,
            }
            _save_stats(Path(args.output_path), output_payload)
            logger.info(
                "Saved V-JEPA global stats to %s: vectors=%d channels=%d "
                "mean=[%.5f, %.5f] std=[%.5f, %.5f]",
                args.output_path,
                merged.n,
                encoder.output_dim,
                mean.min().item(),
                mean.max().item(),
                std.min().item(),
                std.max().item(),
            )
            logger.info("Metadata: %s", json.dumps(metadata, sort_keys=True))
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
