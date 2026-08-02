import hashlib
import json
import math
import os
import time
import traceback
import uuid
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from accelerate import PartialState
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Sampler

from .logging_config import get_logger


logger = get_logger(__name__)

CACHE_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PARTIAL_MANIFEST_FILENAME = "manifest.partial.json"


class VideoLatentCacheError(RuntimeError):
    pass


def _json_payload(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _sampled_module_fingerprint(module) -> str:
    digest = hashlib.sha256()
    state_dict = module.state_dict()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        flat = tensor.reshape(-1)
        chunk_elements = 4 * 1024 * 1024
        for start in range(0, int(flat.numel()), chunk_elements):
            chunk = flat[start : start + chunk_elements].cpu().contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(data_cfg: dict) -> dict:
    fingerprints = {}
    preprocessed_root = data_cfg.get("preprocessed_root")
    if preprocessed_root:
        root = Path(str(preprocessed_root)).expanduser().resolve()
        record = {"path": str(root), "format": "indexed_webdataset"}
        metadata = {}
        for name in ("manifest.json", "dataset_stats.json", "dataset.done"):
            path = root / name
            if path.is_file():
                stat = path.stat()
                metadata[name] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _hash_file(path),
                }
        summaries = {}
        for path in sorted((root / "shards").glob("shard-*.summary.json")):
            stat = path.stat()
            summaries[path.name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _hash_file(path),
            }
        record["metadata"] = metadata
        record["shard_summaries"] = summaries
        payloads = {}
        for pattern in (
            "shard-*.tar",
            "shard-*.offsets.npy",
            "shard-*.sizes.npy",
            "shard-*.state.npy",
            "shard-*.action.npy",
            "shard-*.task_index.npy",
            "shard-*.episodes.json",
        ):
            for path in sorted((root / "shards").glob(pattern)):
                stat = path.stat()
                payloads[path.name] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        record["shard_payloads"] = payloads
        fingerprints[str(root)] = record

    for raw_root in data_cfg.get("dataset_dirs", []) or []:
        root = Path(str(raw_root)).expanduser().resolve()
        record = {"path": str(root)}
        metadata = {}
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            path = root / "meta" / name
            if path.is_file():
                stat = path.stat()
                metadata[name] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _hash_file(path),
                }
        record["metadata"] = metadata

        for subtree_name in ("data", "videos"):
            subtree = root / subtree_name
            file_count = 0
            total_size = 0
            latest_mtime_ns = 0
            tree_digest = hashlib.sha256()
            if subtree.is_dir():
                for dirpath, _, filenames in os.walk(subtree):
                    for filename in sorted(filenames):
                        path = Path(dirpath) / filename
                        stat = path.stat()
                        relative_path = path.relative_to(root)
                        file_count += 1
                        total_size += stat.st_size
                        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
                        tree_digest.update(str(relative_path).encode("utf-8"))
                        tree_digest.update(
                            f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_ino}".encode(
                                "utf-8"
                            )
                        )
            record[subtree_name] = {
                "file_count": file_count,
                "total_size": total_size,
                "latest_mtime_ns": latest_mtime_ns,
                "tree_signature": tree_digest.hexdigest(),
            }
        fingerprints[str(root)] = record
    return fingerprints


def _implementation_fingerprint() -> dict:
    fastwam_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        fastwam_root / "models" / "hfastwam" / "hfastwam.py",
        fastwam_root / "models" / "wan22" / "visual_encoder.py",
        fastwam_root / "datasets" / "lerobot" / "robot_video_dataset.py",
        fastwam_root / "datasets" / "lerobot" / "webdataset_robot_video_dataset.py",
        fastwam_root / "datasets" / "lerobot" / "processors" / "fastwam_processor.py",
        fastwam_root / "datasets" / "dataset_utils.py",
    ]
    return {
        str(path.relative_to(fastwam_root)): _hash_file(path)
        for path in source_paths
        if path.is_file()
    }


def _config_signature(
    cfg: DictConfig,
    dataset,
    model,
    split: str,
    cache_dtype: str,
    shard_size: int,
) -> tuple[str, dict]:
    model_cfg = _json_payload(cfg.model)
    data_cfg = _json_payload(cfg.data.get(split))
    representation_module = (
        model.visual_encoder if getattr(model, "use_visual_encoder", False) else model.vae
    )
    payload = {
        "version": CACHE_VERSION,
        "split": split,
        "dataset_length": len(dataset),
        "shard_size": int(shard_size),
        "model": model_cfg,
        "data": data_cfg,
        "dataset_fingerprint": _dataset_fingerprint(data_cfg),
        "representation_fingerprint": _sampled_module_fingerprint(representation_module),
        "implementation_fingerprint": _implementation_fingerprint(),
        "cache_dtype": cache_dtype,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _atomic_json_dump(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _shard_path(cache_dir: Path, shard_id: int) -> Path:
    return cache_dir / f"shard_{shard_id:08d}.safetensors"


def _tensor_key(index: int) -> str:
    return f"latent_{index:012d}"


def load_video_latent_cache_manifest(
    cache_dir: str | Path,
    *,
    expected_length: int | None = None,
) -> dict:
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing video latent cache manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not bool(manifest.get("complete", False)):
        raise RuntimeError(f"Video latent cache is not complete: {cache_dir}")
    if int(manifest.get("version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"Video latent cache version mismatch at {cache_dir}: "
            f"expected {CACHE_VERSION}, got {manifest.get('version')}"
        )
    if expected_length is not None and int(manifest.get("dataset_length", -1)) != int(expected_length):
        raise ValueError(
            f"Video latent cache length mismatch at {cache_dir}: expected "
            f"{expected_length}, got {manifest.get('dataset_length')}"
        )
    return manifest


def load_video_latent(cache_dir: str | Path, manifest: dict, index: int) -> torch.Tensor:
    cache_dir = Path(cache_dir).expanduser().resolve()
    shard_size = int(manifest["shard_size"])
    shard_id = int(index) // shard_size
    shard_path = _shard_path(cache_dir, shard_id)
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing video latent cache shard: {shard_path}")
    key = _tensor_key(int(index))
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"Missing `{key}` in video latent cache shard {shard_path}")
        latent = handle.get_tensor(key)
    if latent.ndim != 4:
        raise ValueError(
            f"Cached video latent must be [D,T,H,W], got {tuple(latent.shape)} "
            f"for index {index} in {shard_path}"
        )
    return latent


def _expected_shard_size(dataset_length: int, shard_size: int, shard_id: int) -> int:
    start = shard_id * shard_size
    return max(min(start + shard_size, dataset_length) - start, 0)


class _MissingShardIndexSampler(Sampler[int]):
    def __init__(
        self,
        *,
        dataset_length: int,
        shard_size: int,
        rank: int,
        world_size: int,
        missing_shards: set[int],
    ):
        self.dataset_length = int(dataset_length)
        self.shard_size = int(shard_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shard_ids = [
            shard_id
            for shard_id in sorted(missing_shards)
            if shard_id % self.world_size == self.rank
        ]

    def __iter__(self) -> Iterator[int]:
        for shard_id in self.shard_ids:
            start = shard_id * self.shard_size
            end = min(start + self.shard_size, self.dataset_length)
            yield from range(start, end)

    def __len__(self) -> int:
        return sum(
            _expected_shard_size(self.dataset_length, self.shard_size, shard_id)
            for shard_id in self.shard_ids
        )


class _VideoOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset._get(int(index))
        actual_index = int(sample.get("sample_idx", index))
        if actual_index != int(index):
            raise RuntimeError(
                f"Video cache requested index {index}, but dataset fallback returned {actual_index}."
            )
        video = sample.get("video")
        if not torch.is_tensor(video) or video.ndim != 4:
            raise ValueError(
                f"Video cache expects unbatched [3,T,H,W] video, got {type(video)} "
                f"with shape {getattr(video, 'shape', None)}."
            )
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "video": video,
        }


def _cache_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported video latent cache dtype: {name}")


def _validate_cacheable_encoder(model):
    if not getattr(model, "use_visual_encoder", False):
        return

    encoder = model.visual_encoder
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(
            "Cannot cache visual latents from a trainable encoder. "
            f"Trainable parameters include: {trainable[:8]}"
        )

    causal = bool(getattr(encoder, "causal_tubelet_encoding", False)) or bool(
        getattr(encoder, "causal_prefix_encoding", False)
    )
    fixed_stats = bool(getattr(encoder, "_has_fixed_stats", False))
    if bool(getattr(encoder, "standardise_output", False)) and not causal and not fixed_stats:
        raise ValueError(
            "The visual encoder uses batch-dependent output standardisation, so cached "
            "latents would depend on precompute batch composition. Set "
            "model.visual_encoder_config.standardise_output=false or provide fixed stats."
        )


def _write_shard(cache_dir: Path, shard_id: int, tensors: dict[str, torch.Tensor]):
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _shard_path(cache_dir, shard_id)
    tmp_path = cache_dir / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    save_file(tensors, str(tmp_path), metadata={"cache_version": str(CACHE_VERSION)})
    os.replace(tmp_path, output_path)


@contextmanager
def _distributed_cache_lock(cache_dir: Path, state: PartialState):
    lock_handle = None
    lock_path = cache_dir / ".build.lock"
    if state.is_main_process:
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+")
        logger.info("Waiting for video latent cache lock: %s", lock_path)

    while True:
        lock_status = [False, None]
        if state.is_main_process:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_status[0] = True
            except BlockingIOError:
                pass
            except Exception:
                lock_status[1] = traceback.format_exc()
        if dist.is_initialized():
            dist.broadcast_object_list(lock_status, src=0)
        if lock_status[1] is not None:
            raise RuntimeError(
                f"Failed to acquire video latent cache lock {lock_path}:\n{lock_status[1]}"
            )
        if lock_status[0]:
            break
        time.sleep(5.0)

    if state.is_main_process:
        logger.info("Acquired video latent cache lock: %s", lock_path)
    try:
        yield
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def _rank0_call(state: PartialState, operation, description: str):
    packet = [None, None]
    if state.is_main_process:
        try:
            packet[0] = operation()
        except Exception:
            packet[1] = traceback.format_exc()
    if dist.is_initialized():
        dist.broadcast_object_list(packet, src=0)
    if packet[1] is not None:
        raise RuntimeError(f"{description} failed on rank 0:\n{packet[1]}")
    return packet[0]


def _collective_validate(state: PartialState, operation, description: str):
    local_error = None
    try:
        operation()
    except Exception:
        local_error = traceback.format_exc()
    errors = [local_error]
    if dist.is_initialized():
        errors = [None for _ in range(int(state.num_processes))]
        dist.all_gather_object(errors, local_error)
    failures = [
        f"rank {rank}:\n{error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(f"{description} failed:\n" + "\n".join(failures))


@torch.no_grad()
def ensure_video_latent_cache(
    **kwargs,
):
    state = PartialState()
    cache_dir = Path(kwargs["cache_dir"]).expanduser().resolve()
    with _distributed_cache_lock(cache_dir, state):
        return _ensure_video_latent_cache_locked(**kwargs)


@torch.no_grad()
def _ensure_video_latent_cache_locked(
    *,
    cfg: DictConfig,
    model,
    dataset,
    split: str,
    cache_dir: str | Path,
    shard_size: int,
    batch_size: int,
    num_workers: int,
    cache_dtype: str,
    drop_video: bool,
):
    if not hasattr(dataset, "_get") or not hasattr(dataset, "set_video_latent_cache"):
        raise TypeError(
            "Video latent caching requires RobotVideoDataset-compatible `_get` and "
            "`set_video_latent_cache` methods."
        )
    if shard_size <= 0 or batch_size <= 0 or num_workers < 0:
        raise ValueError(
            f"Invalid cache loader settings: shard_size={shard_size}, "
            f"batch_size={batch_size}, num_workers={num_workers}."
        )

    state = PartialState()
    rank = int(state.process_index)
    world_size = int(state.num_processes)
    cache_dir = Path(cache_dir).expanduser().resolve()
    dataset_length = len(dataset)
    num_shards = math.ceil(dataset_length / shard_size)
    signature, signature_payload = _rank0_call(
        state,
        lambda: _config_signature(
            cfg,
            dataset,
            model,
            split,
            cache_dtype,
            shard_size,
        ),
        "Video latent cache signature generation",
    )
    manifest_path = cache_dir / MANIFEST_FILENAME
    partial_manifest_path = cache_dir / PARTIAL_MANIFEST_FILENAME

    def _inspect_cache():
        manifest = None
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("signature") != signature:
                raise ValueError(
                    f"Video latent cache signature mismatch at {cache_dir}. "
                    "Use a different VIDEO_LATENT_CACHE_DIR or remove the stale cache."
                )
            all_present = all(
                _shard_path(cache_dir, shard_id).is_file()
                for shard_id in range(num_shards)
            )
            if bool(manifest.get("complete")) and all_present:
                return True

        if partial_manifest_path.is_file():
            with partial_manifest_path.open("r", encoding="utf-8") as handle:
                partial_manifest = json.load(handle)
            if partial_manifest.get("signature") != signature:
                raise ValueError(
                    f"Partial video latent cache signature mismatch at {cache_dir}. "
                    "Use a different cache directory or remove the stale partial cache."
                )
        elif not manifest_path.is_file() and cache_dir.is_dir():
            existing_shards = list(cache_dir.glob("shard_*.safetensors"))
            if existing_shards:
                raise ValueError(
                    f"Found video latent shards without a manifest at {cache_dir}. "
                    "Use a new cache directory or remove the orphaned shards."
                )
        return False

    cache_complete = _rank0_call(state, _inspect_cache, "Video latent cache inspection")
    if cache_complete:
        dataset.set_video_latent_cache(
            cache_dir=cache_dir,
            expected_length=dataset_length,
            drop_video=drop_video,
        )
        if rank == 0:
            logger.info("Using complete %s video latent cache: %s", split, cache_dir)
        return

    _collective_validate(
        state,
        lambda: _validate_cacheable_encoder(model),
        "Video latent cache encoder validation",
    )

    def _write_partial_manifest():
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial_payload = {
            "version": CACHE_VERSION,
            "signature": signature,
            "signature_payload": signature_payload,
            "dataset_length": dataset_length,
            "shard_size": shard_size,
            "num_shards": num_shards,
            "cache_dtype": cache_dtype,
            "complete": False,
        }
        _atomic_json_dump(partial_payload, partial_manifest_path)

    _rank0_call(
        state,
        _write_partial_manifest,
        "Video latent partial manifest write",
    )

    missing_shards = {
        shard_id
        for shard_id in range(num_shards)
        if not _shard_path(cache_dir, shard_id).is_file()
    }
    local_sampler = _MissingShardIndexSampler(
        dataset_length=dataset_length,
        shard_size=shard_size,
        rank=rank,
        world_size=world_size,
        missing_shards=missing_shards,
    )

    if rank == 0:
        logger.info(
            "Building %s video latent cache: dir=%s samples=%d shards=%d missing=%d "
            "world_size=%d batch_size=%d workers=%d",
            split,
            cache_dir,
            dataset_length,
            num_shards,
            len(missing_shards),
            world_size,
            batch_size,
            num_workers,
        )

    buffers: dict[int, dict[str, torch.Tensor]] = {}
    local_shards_written = 0
    was_training = bool(model.training)
    local_error = None
    try:
        loader = DataLoader(
            _VideoOnlyDataset(dataset),
            batch_size=batch_size,
            sampler=local_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        output_dtype = _cache_dtype(cache_dtype)
        model.eval()
        for batch in loader:
            indices = batch["index"].tolist()
            videos = batch["video"].to(
                device=model.device,
                dtype=model.torch_dtype,
                non_blocking=True,
            )
            latents = model._encode_video_latents(videos, tiled=False)
            latents = model._align_first_conditioning_latent(videos, latents)
            latents = latents.detach().to(device="cpu", dtype=output_dtype).contiguous()

            for batch_index, dataset_index in enumerate(indices):
                shard_id = int(dataset_index) // shard_size
                shard_buffer = buffers.setdefault(shard_id, {})
                shard_buffer[_tensor_key(int(dataset_index))] = latents[batch_index]
                expected = _expected_shard_size(dataset_length, shard_size, shard_id)
                if len(shard_buffer) == expected:
                    _write_shard(cache_dir, shard_id, shard_buffer)
                    del buffers[shard_id]
                    local_shards_written += 1
                    if local_shards_written == 1 or local_shards_written % 10 == 0:
                        logger.info(
                            "Video latent cache rank %d wrote %d/%d assigned shards.",
                            rank,
                            local_shards_written,
                            len(local_sampler.shard_ids),
                        )
    except Exception:
        local_error = traceback.format_exc()
    finally:
        model.train(was_training)

    if local_error is None and buffers:
        incomplete = {shard_id: len(values) for shard_id, values in buffers.items()}
        local_error = f"Incomplete video latent shard buffers on rank {rank}: {incomplete}"

    rank_errors = [local_error]
    if dist.is_initialized():
        rank_errors = [None for _ in range(world_size)]
        dist.all_gather_object(rank_errors, local_error)
    failures = [
        f"rank {error_rank}:\n{error}"
        for error_rank, error in enumerate(rank_errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(
            "Video latent cache build failed on one or more ranks:\n" + "\n".join(failures)
        )

    def _finalize_cache():
        missing_after = [
            shard_id
            for shard_id in range(num_shards)
            if not _shard_path(cache_dir, shard_id).is_file()
        ]
        if missing_after:
            raise RuntimeError(
                f"Video latent cache remains incomplete at {cache_dir}; "
                f"missing shards: {missing_after[:16]}"
            )
        complete_payload = {
            "version": CACHE_VERSION,
            "signature": signature,
            "signature_payload": signature_payload,
            "dataset_length": dataset_length,
            "shard_size": shard_size,
            "num_shards": num_shards,
            "cache_dtype": cache_dtype,
            "complete": True,
        }
        _atomic_json_dump(complete_payload, manifest_path)
        partial_manifest_path.unlink(missing_ok=True)
        logger.info("Finished %s video latent cache: %s", split, cache_dir)

    _rank0_call(state, _finalize_cache, "Video latent cache finalization")

    dataset.set_video_latent_cache(
        cache_dir=cache_dir,
        expected_length=dataset_length,
        drop_video=drop_video,
    )


def ensure_training_video_latent_caches(cfg: DictConfig, model, train_dataset, val_dataset):
    cache_cfg = cfg.get("video_latent_cache")
    if cache_cfg is None or not bool(cache_cfg.get("enabled", False)):
        return

    root = cache_cfg.get("root")
    if root is None or not str(root).strip():
        raise ValueError("video_latent_cache.root is required when caching is enabled.")

    shard_size = int(cache_cfg.get("shard_size", 32))
    batch_size = int(cache_cfg.get("batch_size", 1))
    num_workers = int(cache_cfg.get("num_workers", 4))
    cache_dtype = str(cache_cfg.get("dtype", "bf16"))
    root = Path(str(root)).expanduser()
    drop_train_video = bool(cache_cfg.get("drop_train_video", True))
    if val_dataset is train_dataset and drop_train_video:
        drop_train_video = False
        logger.warning(
            "Train and validation share one dataset object; preserving raw video so "
            "evaluation can still run infer_action()."
        )

    ensure_video_latent_cache(
        cfg=cfg,
        model=model,
        dataset=train_dataset,
        split="train",
        cache_dir=root / "train",
        shard_size=shard_size,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dtype=cache_dtype,
        drop_video=drop_train_video,
    )

    if (
        bool(cache_cfg.get("include_val", False))
        and val_dataset is not None
        and val_dataset is not train_dataset
    ):
        ensure_video_latent_cache(
            cfg=cfg,
            model=model,
            dataset=val_dataset,
            split="val",
            cache_dir=root / "val",
            shard_size=shard_size,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dtype=cache_dtype,
            drop_video=False,
        )
