import argparse
import hashlib
import io
import json
import logging
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tarfile
import torch
from PIL import Image

from robotwin_s3 import AwsCliS3, S3Config

FORMAT_NAME = "robotwin-webdataset"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
DATASET_DONE_FILENAME = "dataset.done"
DONE_MARKER_TEXT = "complete\n"
STATE_DIM = 14
ACTION_DIM = 14
CANONICAL_CAMERAS: tuple[tuple[str, str], ...] = (
    ("cam_high", "observation.images.cam_high"),
    ("cam_left_wrist", "observation.images.cam_left_wrist"),
    ("cam_right_wrist", "observation.images.cam_right_wrist"),
)
REQUIRED_PARQUET_COLUMNS: tuple[str, ...] = (
    "observation.state",
    "action",
    "task_index",
    "frame_index",
    "timestamp",
    "episode_index",
)

LOGGER = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    length: int
    original_global_start: int


@dataclass(frozen=True)
class SourceMetadata:
    source_root: str
    data_path_template: str
    video_path_template: str
    chunks_size: int
    fps: int
    total_source_episodes: int
    total_source_frames: int
    total_tasks: int
    raw_height: int
    raw_width: int
    camera_short_keys: tuple[str, ...]
    camera_feature_keys: tuple[str, ...]
    episodes: tuple[EpisodeRecord, ...]


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    episode_start: int
    episode_stop: int
    episodes: tuple[EpisodeRecord, ...]
    original_global_start: int
    frame_count: int


@dataclass(frozen=True)
class WorkerConfig:
    output_root: str
    source_root: str
    episodes_per_shard: int
    png_compress_level: int
    decode_chunk_frames: int
    overwrite: bool
    s3_output_root: str | None = None
    aws_profile: str = "roboticsx"
    aws_region: str = "us-east-2"
    aws_credentials_file: str = "/fsx/.aws/credentials"
    aws_cli: str = "aws"


class UncompressedTarWriter:
    def __init__(self, handle: BinaryIO):
        self._handle = handle
        self._closed = False
        self._sha256 = hashlib.sha256()

    def _write(self, data: bytes) -> None:
        self._handle.write(data)
        self._sha256.update(data)

    def add_bytes(self, name: str, data: bytes) -> tuple[int, int]:
        if self._closed:
            raise RuntimeError("Cannot write to a closed tar writer")
        if len(name.encode("ascii")) > 100:
            raise ValueError(f"Tar member name must fit USTAR name field: {name}")
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        header = info.tobuf(format=tarfile.USTAR_FORMAT, encoding="ascii", errors="strict")
        if len(header) != 512:
            raise ValueError(f"Unexpected tar header size for {name}: {len(header)}")
        self._write(header)
        data_offset = self._handle.tell()
        self._write(data)
        pad = (-len(data)) % 512
        if pad:
            self._write(b"\0" * pad)
        return data_offset, len(data)

    def close(self) -> None:
        if self._closed:
            return
        self._write(b"\0" * 1024)
        self._closed = True

    def tell(self) -> int:
        return self._handle.tell()

    def hexdigest(self) -> str:
        if not self._closed:
            raise RuntimeError("Tar writer must be closed before reading its digest")
        return self._sha256.hexdigest()


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def default_workers() -> int:
    return 4


def default_stats_path(source_root: str | Path) -> Path:
    return Path(source_root).expanduser().resolve().parent / "dataset_stats.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the RoboTwin LeRobot dataset into resumable indexed WebDataset-style "
            "uncompressed tar shards."
        )
    )
    parser.add_argument(
        "--source-root",
        default="data/robotwin2.0/robotwin2.0",
        help="Path to the source RoboTwin LeRobot dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default="data/robotwin2.0_webdataset",
        help="Path to the local staging/output directory.",
    )
    parser.add_argument(
        "--s3-output-root",
        default=None,
        help="Optional s3:// destination. Completed shards are verified remotely then removed locally.",
    )
    parser.add_argument("--aws-profile", default="roboticsx")
    parser.add_argument("--aws-region", default="us-east-2")
    parser.add_argument("--aws-credentials-file", default="/fsx/.aws/credentials")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument(
        "--allow-partial-s3",
        action="store_true",
        help="Allow --max-episodes with S3 output. Use only with a dedicated non-production prefix.",
    )
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Path to normalization stats JSON. Default: <source-root parent>/dataset_stats.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Worker process count. Default: 4.",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=32,
        help="Number of full episodes per shard.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=3,
        help="Pillow PNG compression level in [0, 9].",
    )
    parser.add_argument(
        "--decode-chunk-frames",
        type=int,
        default=64,
        help="Sequential torchcodec chunk size per camera.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional prefix episode count for testing; preserves original episode ids starting at 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild requested shards even when they already have valid .done markers.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )


def ensure_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}, got {type(value).__name__}")
            yield value


def load_tasks(source_root: Path) -> list[str]:
    tasks_path = source_root / "meta" / "tasks.jsonl"
    tasks: list[str] = []
    for expected_index, record in enumerate(iter_jsonl(tasks_path)):
        task_index = int(record.get("task_index", -1))
        if task_index != expected_index:
            raise ValueError(
                f"Task indices must be contiguous from 0, got {task_index} at position {expected_index} in {tasks_path}"
            )
        task = record.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"Missing task text for task_index={task_index} in {tasks_path}")
        tasks.append(task)
    return tasks


def load_source_metadata(source_root: Path) -> SourceMetadata:
    info_path = source_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError(f"Expected object in {info_path}, got {type(info).__name__}")

    data_path_template = str(info.get("data_path", ""))
    video_path_template = str(info.get("video_path", ""))
    if not data_path_template:
        raise ValueError(f"Missing non-empty `data_path` in {info_path}")
    if not video_path_template:
        raise ValueError(f"Missing non-empty `video_path` in {info_path}")
    chunks_size = ensure_positive("chunks_size", int(info.get("chunks_size", 0)))
    fps = ensure_positive("fps", int(info.get("fps", 0)))
    total_source_episodes = ensure_positive("total_episodes", int(info.get("total_episodes", 0)))
    total_source_frames = ensure_positive("total_frames", int(info.get("total_frames", 0)))
    total_tasks = ensure_positive("total_tasks", int(info.get("total_tasks", 0)))

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Missing dict `features` in {info_path}")

    camera_short_keys = tuple(camera_name for camera_name, _ in CANONICAL_CAMERAS)
    camera_feature_keys = tuple(feature_key for _, feature_key in CANONICAL_CAMERAS)
    raw_height: int | None = None
    raw_width: int | None = None
    for camera_name, feature_key in CANONICAL_CAMERAS:
        feature = features.get(feature_key)
        if not isinstance(feature, dict):
            raise ValueError(f"Missing feature `{feature_key}` in {info_path}")
        shape = feature.get("shape")
        if not (isinstance(shape, list) and len(shape) == 3):
            raise ValueError(f"Expected [H, W, 3] shape for {feature_key}, got {shape!r}")
        height, width, channels = (int(shape[0]), int(shape[1]), int(shape[2]))
        if channels != 3:
            raise ValueError(f"Expected 3 channels for {feature_key}, got {channels}")
        if raw_height is None:
            raw_height = height
            raw_width = width
        elif raw_height != height or raw_width != width:
            raise ValueError(
                f"Camera {camera_name} shape {shape} does not match expected {(raw_height, raw_width, 3)}"
            )

    episodes_path = source_root / "meta" / "episodes.jsonl"
    episodes: list[EpisodeRecord] = []
    running_global_start = 0
    for expected_episode_id, record in enumerate(iter_jsonl(episodes_path)):
        episode_id = int(record.get("episode_index", -1))
        if episode_id != expected_episode_id:
            raise ValueError(
                f"Episode ids must be contiguous from 0, got {episode_id} at position {expected_episode_id} in {episodes_path}"
            )
        length = ensure_positive(f"episode {episode_id} length", int(record.get("length", 0)))
        episodes.append(
            EpisodeRecord(
                episode_id=episode_id,
                length=length,
                original_global_start=running_global_start,
            )
        )
        running_global_start += length

    if len(episodes) != total_source_episodes:
        raise ValueError(
            f"Episode count mismatch: info.json says {total_source_episodes}, episodes.jsonl has {len(episodes)}"
        )
    if running_global_start != total_source_frames:
        raise ValueError(
            f"Frame count mismatch: info.json says {total_source_frames}, episode lengths sum to {running_global_start}"
        )
    if raw_height is None or raw_width is None:
        raise ValueError(f"Could not determine raw frame size from {info_path}")

    return SourceMetadata(
        source_root=str(source_root),
        data_path_template=data_path_template,
        video_path_template=video_path_template,
        chunks_size=chunks_size,
        fps=fps,
        total_source_episodes=total_source_episodes,
        total_source_frames=total_source_frames,
        total_tasks=total_tasks,
        raw_height=raw_height,
        raw_width=raw_width,
        camera_short_keys=camera_short_keys,
        camera_feature_keys=camera_feature_keys,
        episodes=tuple(episodes),
    )


def build_shard_specs(
    metadata: SourceMetadata,
    episodes_per_shard: int,
    max_episodes: int | None,
) -> list[ShardSpec]:
    episodes_per_shard = ensure_positive("episodes_per_shard", int(episodes_per_shard))
    if max_episodes is None:
        limit = metadata.total_source_episodes
    else:
        limit = ensure_positive("max_episodes", int(max_episodes))
        if limit > metadata.total_source_episodes:
            raise ValueError(
                f"max_episodes={limit} exceeds source total {metadata.total_source_episodes}"
            )
    selected = metadata.episodes[:limit]
    shards: list[ShardSpec] = []
    for shard_index, episode_start in enumerate(range(0, limit, episodes_per_shard)):
        episode_stop = min(episode_start + episodes_per_shard, limit)
        shard_episodes = selected[episode_start:episode_stop]
        if not shard_episodes:
            continue
        first = shard_episodes[0]
        frame_count = sum(episode.length for episode in shard_episodes)
        shards.append(
            ShardSpec(
                shard_index=shard_index,
                episode_start=episode_start,
                episode_stop=episode_stop,
                episodes=tuple(shard_episodes),
                original_global_start=first.original_global_start,
                frame_count=frame_count,
            )
        )
    return shards


def shard_stem(shard_index: int) -> str:
    return f"shard-{shard_index:05d}"


def expected_shard_paths(output_root: Path, shard_index: int) -> dict[str, Path]:
    stem = shard_stem(shard_index)
    shard_dir = output_root / "shards"
    base = shard_dir / stem
    return {
        "tar": base.with_suffix(".tar"),
        "offsets": base.with_suffix(".offsets.npy"),
        "sizes": base.with_suffix(".sizes.npy"),
        "state": base.with_suffix(".state.npy"),
        "action": base.with_suffix(".action.npy"),
        "task_index": base.with_suffix(".task_index.npy"),
        "episodes": base.with_suffix(".episodes.json"),
        "summary": base.with_suffix(".summary.json"),
        "done": base.with_suffix(".done"),
    }


def partial_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.name}.partial.{os.getpid()}")


def episode_parquet_path(metadata: SourceMetadata, episode_id: int) -> Path:
    relative = metadata.data_path_template.format(
        episode_chunk=episode_id // metadata.chunks_size,
        episode_index=episode_id,
    )
    return Path(metadata.source_root) / relative


def episode_video_path(metadata: SourceMetadata, episode_id: int, feature_key: str) -> Path:
    relative = metadata.video_path_template.format(
        episode_chunk=episode_id // metadata.chunks_size,
        episode_index=episode_id,
        video_key=feature_key,
    )
    return Path(metadata.source_root) / relative


def source_file_fingerprint(
    metadata: SourceMetadata,
    spec: ShardSpec,
) -> dict[str, dict[str, int]]:
    source_root = Path(metadata.source_root)
    paths = []
    for episode in spec.episodes:
        paths.append(episode_parquet_path(metadata, episode.episode_id))
        for _camera_name, feature_key in CANONICAL_CAMERAS:
            paths.append(
                episode_video_path(metadata, episode.episode_id, feature_key)
            )
    records = {}
    for path in paths:
        stat = path.stat()
        records[str(path.relative_to(source_root))] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    return records


def source_metadata_fingerprint(metadata: SourceMetadata) -> dict[str, dict[str, int]]:
    source_root = Path(metadata.source_root)
    records = {}
    for relative in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        path = source_root / relative
        stat = path.stat()
        records[relative] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    return records


def member_name(original_global_frame_index: int) -> str:
    return f"{original_global_frame_index:012d}.png"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_int = int(round(seconds))
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _combine_list_column_to_float32(
    column: pa.ChunkedArray,
    *,
    name: str,
    length: int,
    expected_width: int,
    episode_id: int,
    path: Path,
) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_list(array.type):
        raise ConversionError(
            f"Episode {episode_id} parquet {path} column {name} must be list-like, got {array.type}"
        )
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    if offsets.shape[0] != length + 1:
        raise ConversionError(
            f"Episode {episode_id} parquet {path} column {name} offsets length {offsets.shape[0]} != {length + 1}"
        )
    widths = np.diff(offsets)
    if not np.all(widths == expected_width):
        unique = np.unique(widths)
        raise ConversionError(
            f"Episode {episode_id} parquet {path} column {name} expected width {expected_width}, got {unique.tolist()}"
        )
    values = array.values.to_numpy(zero_copy_only=False)
    if values.size != length * expected_width:
        raise ConversionError(
            f"Episode {episode_id} parquet {path} column {name} value count {values.size} != {length * expected_width}"
        )
    return np.asarray(values, dtype=np.float32).reshape(length, expected_width)


def load_episode_tabular(
    metadata: SourceMetadata,
    episode: EpisodeRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parquet_path = episode_parquet_path(metadata, episode.episode_id)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing parquet for episode {episode.episode_id}: {parquet_path}")
    table = pq.read_table(parquet_path, columns=list(REQUIRED_PARQUET_COLUMNS))
    if table.num_rows != episode.length:
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} row count {table.num_rows} != expected {episode.length}"
        )
    states = _combine_list_column_to_float32(
        table.column("observation.state"),
        name="observation.state",
        length=episode.length,
        expected_width=STATE_DIM,
        episode_id=episode.episode_id,
        path=parquet_path,
    )
    actions = _combine_list_column_to_float32(
        table.column("action"),
        name="action",
        length=episode.length,
        expected_width=ACTION_DIM,
        episode_id=episode.episode_id,
        path=parquet_path,
    )
    task_indices = np.asarray(
        table.column("task_index").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if task_indices.shape != (episode.length,):
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} task_index shape {task_indices.shape} != {(episode.length,)}"
        )
    if np.any(task_indices < 0) or np.any(task_indices >= metadata.total_tasks):
        min_value = int(task_indices.min(initial=0))
        max_value = int(task_indices.max(initial=0))
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} task_index range [{min_value}, {max_value}] "
            f"is outside [0, {metadata.total_tasks})"
        )
    frame_indices = np.asarray(
        table.column("frame_index").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    expected_frame_indices = np.arange(episode.length, dtype=np.int64)
    if frame_indices.shape != (episode.length,):
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} frame_index shape {frame_indices.shape} != {(episode.length,)}"
        )
    if not np.array_equal(frame_indices, expected_frame_indices):
        mismatch = np.flatnonzero(frame_indices != expected_frame_indices)[:8]
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} frame_index mismatch at positions "
            f"{mismatch.tolist()}: got {frame_indices[mismatch].tolist()} expected {expected_frame_indices[mismatch].tolist()}"
        )
    timestamps = np.asarray(
        table.column("timestamp").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    if timestamps.shape != (episode.length,):
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} timestamp shape {timestamps.shape} != {(episode.length,)}"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} contains non-finite timestamps"
        )
    episode_indices = np.asarray(
        table.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if episode_indices.shape != (episode.length,):
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} episode_index shape {episode_indices.shape} != {(episode.length,)}"
        )
    if not np.all(episode_indices == episode.episode_id):
        unique_episode_indices = np.unique(episode_indices)
        raise ConversionError(
            f"Episode {episode.episode_id} parquet {parquet_path} has incorrect episode_index values "
            f"{unique_episode_indices.tolist()}"
        )
    return states, actions, task_indices, frame_indices, timestamps


def decoder_average_fps(decoder: Any, *, episode_id: int, camera_name: str, video_path: Path) -> float:
    metadata = getattr(decoder, "metadata", None)
    average_fps = getattr(metadata, "average_fps", None)
    if average_fps is None:
        raise ConversionError(
            f"Episode {episode_id} camera {camera_name} video {video_path} is missing decoder average_fps"
        )
    value = float(average_fps)
    if not math.isfinite(value) or value <= 0:
        raise ConversionError(
            f"Episode {episode_id} camera {camera_name} video {video_path} has invalid decoder average_fps={average_fps}"
        )
    return value


def validate_timestamp_alignment(
    *,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    average_fps: float,
    episode_id: int,
    camera_name: str,
    video_path: Path,
) -> None:
    rounded_indices = np.rint(timestamps * average_fps).astype(np.int64)
    if not np.array_equal(rounded_indices, frame_indices):
        mismatch = np.flatnonzero(rounded_indices != frame_indices)[:8]
        raise ConversionError(
            f"Episode {episode_id} camera {camera_name} video {video_path} timestamp alignment failed at positions "
            f"{mismatch.tolist()}: rounded(timestamp*fps)={rounded_indices[mismatch].tolist()} "
            f"frame_index={frame_indices[mismatch].tolist()} average_fps={average_fps}"
        )


def decode_camera_chunk(
    decoder: Any,
    *,
    video_path: Path,
    chunk_start: int,
    chunk_stop: int,
    raw_height: int,
    raw_width: int,
    episode_id: int,
    camera_name: str,
) -> torch.Tensor:
    frames = decoder.get_frames_in_range(chunk_start, chunk_stop)
    data = getattr(frames, "data", None)
    if not torch.is_tensor(data):
        raise ConversionError(
            f"Episode {episode_id} camera {camera_name} video {video_path} returned non-tensor frames: {type(data).__name__}"
        )
    expected_frames = chunk_stop - chunk_start
    expected_shape = (expected_frames, 3, raw_height, raw_width)
    if tuple(data.shape) != expected_shape:
        raise ConversionError(
            f"Episode {episode_id} camera {camera_name} video {video_path} returned shape {tuple(data.shape)} "
            f"for chunk [{chunk_start}, {chunk_stop}), expected {expected_shape}"
        )
    if data.dtype != torch.uint8:
        raise ConversionError(
            f"Episode {episode_id} camera {camera_name} video {video_path} returned dtype {data.dtype}, expected torch.uint8"
        )
    return data


def png_bytes_from_frame(frame_chw: torch.Tensor, compress_level: int) -> bytes:
    if not torch.is_tensor(frame_chw):
        raise TypeError(f"Expected tensor frame, got {type(frame_chw).__name__}")
    if frame_chw.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 frame, got {frame_chw.dtype}")
    if frame_chw.ndim != 3 or frame_chw.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] frame, got {tuple(frame_chw.shape)}")
    rgb = frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level, optimize=False)
    return buffer.getvalue()


def write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, cls=_NumpyJSONEncoder)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def finalize_partial_outputs(final_to_partial: dict[Path, Path]) -> None:
    for final_path, partial in final_to_partial.items():
        os.replace(partial, final_path)


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except FileNotFoundError:
            continue


def remove_dataset_done_marker(output_root: Path) -> None:
    cleanup_paths([output_root / DATASET_DONE_FILENAME])


def write_dataset_done_marker(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    done_path = output_root / DATASET_DONE_FILENAME
    partial = done_path.with_name(f"{done_path.name}.partial.{os.getpid()}")
    cleanup_paths([partial])
    try:
        write_text(partial, DONE_MARKER_TEXT)
        os.replace(partial, done_path)
    finally:
        cleanup_paths([partial])
    return done_path


def validate_summary(
    summary: dict[str, Any],
    *,
    metadata: SourceMetadata,
    spec: ShardSpec,
    config: WorkerConfig,
    validate_local_payloads: bool = True,
) -> None:
    required_fields = [
        "format",
        "version",
        "source_root",
        "shard_index",
        "episode_ids",
        "frame_count",
        "original_global_start",
        "original_global_end_exclusive",
        "episodes_per_shard",
        "png_compress_level",
        "camera_keys",
        "raw_height",
        "raw_width",
        "tar_bytes",
        "output_bytes_total",
        "payload_files",
        "source_files",
        "source_metadata_files",
        "total_tasks",
        "fps",
        "validation",
    ]
    missing = [field for field in required_fields if field not in summary]
    if missing:
        raise ValueError(f"Missing summary fields: {missing}")
    if summary["format"] != FORMAT_NAME or int(summary["version"]) != FORMAT_VERSION:
        raise ValueError(f"Unexpected summary format/version: {summary.get('format')} {summary.get('version')}")
    if str(summary["source_root"]) != metadata.source_root:
        raise ValueError(f"Summary source_root mismatch: {summary['source_root']} != {metadata.source_root}")
    if int(summary["shard_index"]) != spec.shard_index:
        raise ValueError(f"Summary shard_index mismatch: {summary['shard_index']} != {spec.shard_index}")
    expected_episode_ids = [episode.episode_id for episode in spec.episodes]
    if [int(value) for value in summary["episode_ids"]] != expected_episode_ids:
        raise ValueError(f"Summary episode ids mismatch: {summary['episode_ids']} != {expected_episode_ids}")
    if int(summary["frame_count"]) != spec.frame_count:
        raise ValueError(f"Summary frame_count mismatch: {summary['frame_count']} != {spec.frame_count}")
    if int(summary["original_global_start"]) != spec.original_global_start:
        raise ValueError(
            f"Summary original_global_start mismatch: {summary['original_global_start']} != {spec.original_global_start}"
        )
    if int(summary["original_global_end_exclusive"]) != spec.original_global_start + spec.frame_count:
        raise ValueError(
            "Summary original_global_end_exclusive mismatch: "
            f"{summary['original_global_end_exclusive']} != {spec.original_global_start + spec.frame_count}"
        )
    if int(summary["episodes_per_shard"]) != config.episodes_per_shard:
        raise ValueError(
            f"Summary episodes_per_shard mismatch: {summary['episodes_per_shard']} != {config.episodes_per_shard}"
        )
    if int(summary["png_compress_level"]) != config.png_compress_level:
        raise ValueError(
            f"Summary png_compress_level mismatch: {summary['png_compress_level']} != {config.png_compress_level}"
        )
    if int(summary["fps"]) != metadata.fps:
        raise ValueError(f"Summary fps mismatch: {summary['fps']} != {metadata.fps}")
    if int(summary["total_tasks"]) != metadata.total_tasks:
        raise ValueError(
            f"Summary total_tasks mismatch: {summary['total_tasks']} != {metadata.total_tasks}"
        )
    if summary["source_metadata_files"] != source_metadata_fingerprint(metadata):
        raise ValueError("Source metadata files changed; rebuild required.")
    if list(summary["camera_keys"]) != list(metadata.camera_short_keys):
        raise ValueError(f"Summary camera_keys mismatch: {summary['camera_keys']} != {list(metadata.camera_short_keys)}")
    if int(summary["raw_height"]) != metadata.raw_height or int(summary["raw_width"]) != metadata.raw_width:
        raise ValueError(
            f"Summary raw size mismatch: {(summary['raw_height'], summary['raw_width'])} != {(metadata.raw_height, metadata.raw_width)}"
        )
    validation = summary["validation"]
    if not isinstance(validation, dict):
        raise ValueError(f"Summary validation must be a dict, got {type(validation).__name__}")
    required_validation_fields = [
        "frame_index_is_sequential",
        "episode_index_is_constant_and_correct",
        "timestamp_round_fps_matches_frame_index",
        "decoder_average_fps_by_camera",
    ]
    missing_validation = [field for field in required_validation_fields if field not in validation]
    if missing_validation:
        raise ValueError(f"Missing summary validation fields: {missing_validation}")
    if validation["frame_index_is_sequential"] is not True:
        raise ValueError("Summary validation.frame_index_is_sequential must be true")
    if validation["episode_index_is_constant_and_correct"] is not True:
        raise ValueError("Summary validation.episode_index_is_constant_and_correct must be true")
    if validation["timestamp_round_fps_matches_frame_index"] is not True:
        raise ValueError("Summary validation.timestamp_round_fps_matches_frame_index must be true")
    decoder_fps = validation["decoder_average_fps_by_camera"]
    if not isinstance(decoder_fps, dict):
        raise ValueError("Summary validation.decoder_average_fps_by_camera must be a dict")
    if sorted(decoder_fps) != sorted(metadata.camera_short_keys):
        raise ValueError(
            "Summary validation.decoder_average_fps_by_camera keys mismatch: "
            f"{sorted(decoder_fps)} != {sorted(metadata.camera_short_keys)}"
        )
    for camera_name, value in decoder_fps.items():
        if not math.isclose(float(value), float(metadata.fps), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Summary decoder fps mismatch for {camera_name}: {value} != {metadata.fps}"
            )
    payload_files = summary["payload_files"]
    if not isinstance(payload_files, dict) or not payload_files:
        raise ValueError("Summary payload_files must be a non-empty dict")
    expected_paths = expected_shard_paths(
        Path(config.output_root),
        spec.shard_index,
    )
    for key in (
        "tar",
        "offsets",
        "sizes",
        "state",
        "action",
        "task_index",
        "episodes",
    ):
        path = expected_paths[key]
        record = payload_files.get(path.name)
        if not isinstance(record, dict):
            raise ValueError(f"Summary payload_files is missing {path.name}")
        sha256 = record.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"Payload SHA256 is missing or invalid for {path.name}")
        if validate_local_payloads:
            stat = path.stat()
            if int(record.get("size", -1)) != stat.st_size:
                raise ValueError(
                    f"Payload size mismatch for {path}: "
                    f"{record.get('size')} != {stat.st_size}"
                )
            if hash_file(path) != sha256:
                raise ValueError(f"Payload SHA256 mismatch for {path}")
    current_source_files = source_file_fingerprint(metadata, spec)
    if summary["source_files"] != current_source_files:
        raise ValueError(
            f"Source files changed for shard {spec.shard_index}; rebuild required."
        )


def try_load_complete_summary(
    metadata: SourceMetadata,
    spec: ShardSpec,
    config: WorkerConfig,
) -> dict[str, Any] | None:
    paths = expected_shard_paths(Path(config.output_root), spec.shard_index)
    expected = [
        paths["tar"],
        paths["offsets"],
        paths["sizes"],
        paths["state"],
        paths["action"],
        paths["task_index"],
        paths["episodes"],
        paths["summary"],
        paths["done"],
    ]
    if not all(path.is_file() for path in expected):
        return None
    try:
        with paths["summary"].open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        validate_summary(summary, metadata=metadata, spec=spec, config=config)
    except Exception:
        LOGGER.warning("Shard %s summary validation failed; rebuilding.\n%s", shard_stem(spec.shard_index), traceback.format_exc())
        return None
    return summary


def make_s3_store(config: WorkerConfig) -> AwsCliS3 | None:
    if config.s3_output_root is None:
        return None
    return AwsCliS3(
        S3Config(
            root_uri=config.s3_output_root,
            profile=config.aws_profile,
            region=config.aws_region,
            credentials_file=config.aws_credentials_file,
            aws_cli=config.aws_cli,
        )
    )


def shard_relative_key(path: Path) -> str:
    return f"shards/{path.name}"


def try_load_remote_summary(
    metadata: SourceMetadata,
    spec: ShardSpec,
    config: WorkerConfig,
    store: AwsCliS3,
) -> dict[str, Any] | None:
    paths = expected_shard_paths(Path(config.output_root), spec.shard_index)
    done_key = shard_relative_key(paths["done"])
    summary_key = shard_relative_key(paths["summary"])
    try:
        if store.head(done_key) is None:
            return None
        done = json.loads(store.read_bytes(done_key))
        summary_bytes = store.read_bytes(summary_key)
        summary = json.loads(summary_bytes)
        if done.get("summary_sha256") != hash_bytes(summary_bytes):
            raise ValueError("Remote shard done marker does not match summary")
        validate_summary(
            summary,
            metadata=metadata,
            spec=spec,
            config=config,
            validate_local_payloads=False,
        )
        for key in (
            "tar",
            "offsets",
            "sizes",
            "state",
            "action",
            "task_index",
            "episodes",
        ):
            path = paths[key]
            record = summary["payload_files"][path.name]
            remote = store.head(shard_relative_key(path))
            metadata_fields = remote.get("Metadata", {}) if remote else {}
            if (
                remote is None
                or int(remote.get("ContentLength", -1)) != int(record["size"])
                or metadata_fields.get("sha256") != record["sha256"]
            ):
                raise ValueError(f"Remote payload validation failed for {path.name}")
        return summary
    except Exception:
        LOGGER.warning(
            "Remote shard %s validation failed; rebuilding or re-uploading without modifying the remote marker yet.\n%s",
            shard_stem(spec.shard_index),
            traceback.format_exc(),
        )
        return None


def upload_complete_shard(
    paths: dict[str, Path],
    summary: dict[str, Any],
    store: AwsCliS3,
) -> None:
    store.delete(shard_relative_key(paths["done"]))
    for key in (
        "tar",
        "offsets",
        "sizes",
        "state",
        "action",
        "task_index",
        "episodes",
    ):
        path = paths[key]
        record = summary["payload_files"][path.name]
        store.upload_file(path, shard_relative_key(path), record["sha256"])

    summary_sha256 = hash_file(paths["summary"])
    store.upload_file(
        paths["summary"],
        shard_relative_key(paths["summary"]),
        summary_sha256,
    )
    done_payload = {
        "format": "robotwin-webdataset-shard-done",
        "version": 1,
        "shard_index": int(summary["shard_index"]),
        "summary_sha256": summary_sha256,
        "payload_count": 7,
    }
    write_json(paths["done"], done_payload)
    store.upload_file(
        paths["done"],
        shard_relative_key(paths["done"]),
        hash_file(paths["done"]),
    )


def cleanup_complete_shard(paths: dict[str, Path]) -> None:
    cleanup_paths(paths.values())


def build_shard_summary(
    *,
    spec: ShardSpec,
    metadata: SourceMetadata,
    config: WorkerConfig,
    tar_path: Path,
    tar_sha256: str,
    sidecar_paths: Sequence[Path],
    png_bytes_total: int,
    source_files: dict[str, dict[str, int]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    tar_bytes = tar_path.stat().st_size
    sidecar_bytes_total = sum(path.stat().st_size for path in sidecar_paths)
    payload_paths = [tar_path, *sidecar_paths]
    payload_files = {
        path.name: {
            "size": path.stat().st_size,
            "sha256": tar_sha256 if path == tar_path else hash_file(path),
        }
        for path in payload_paths
    }
    summary = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source_root": metadata.source_root,
        "shard_index": spec.shard_index,
        "shard_name": shard_stem(spec.shard_index),
        "episode_ids": [episode.episode_id for episode in spec.episodes],
        "episode_count": len(spec.episodes),
        "frame_count": spec.frame_count,
        "original_global_start": spec.original_global_start,
        "original_global_end_exclusive": spec.original_global_start + spec.frame_count,
        "episodes_per_shard": config.episodes_per_shard,
        "png_compress_level": config.png_compress_level,
        "camera_keys": list(metadata.camera_short_keys),
        "raw_height": metadata.raw_height,
        "raw_width": metadata.raw_width,
        "tar_path": tar_path.name,
        "tar_bytes": tar_bytes,
        "png_bytes_total": int(png_bytes_total),
        "sidecar_bytes_total": int(sidecar_bytes_total),
        "output_bytes_total": int(tar_bytes + sidecar_bytes_total),
        "fps": metadata.fps,
        "total_tasks": metadata.total_tasks,
        "payload_files": payload_files,
        "source_files": source_files,
        "source_metadata_files": source_metadata_fingerprint(metadata),
        "validation": validation,
    }
    return summary


def copy_dataset_stats(
    *,
    source_path: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if not source_path.is_file():
        raise FileNotFoundError(
            "Normalization stats file not found: "
            f"{source_path}. Provide --stats-path or place dataset_stats.json next to the source root."
        )
    source_bytes = source_path.read_bytes()
    try:
        json.loads(source_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Normalization stats file is not valid UTF-8 JSON: {source_path}") from exc
    source_sha256 = hash_bytes(source_bytes)

    output_path = output_root / "dataset_stats.json"
    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(f"Expected file path for dataset stats, found directory: {output_path}")
        output_sha256 = hash_file(output_path)
        if output_sha256 == source_sha256:
            return {
                "path": output_path.name,
                "sha256": source_sha256,
                "source_path": str(source_path),
            }
        if not overwrite:
            raise FileExistsError(
                f"Existing output stats file {output_path} does not match source {source_path}. "
                "Pass --overwrite to replace it."
            )

    partial = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    cleanup_paths([partial])
    try:
        with partial.open("wb") as handle:
            handle.write(source_bytes)
        os.replace(partial, output_path)
    finally:
        cleanup_paths([partial])
    return {
        "path": output_path.name,
        "sha256": source_sha256,
        "source_path": str(source_path),
    }


def process_shard(spec: ShardSpec, metadata: SourceMetadata, config: WorkerConfig) -> dict[str, Any]:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "shards").mkdir(parents=True, exist_ok=True)
    paths = expected_shard_paths(output_root, spec.shard_index)
    store = make_s3_store(config)

    if store is not None and not config.overwrite:
        remote_summary = try_load_remote_summary(metadata, spec, config, store)
        if remote_summary is not None:
            cleanup_complete_shard(paths)
            return {"result": "remote-skipped", "summary": remote_summary}

    existing_summary = None if config.overwrite else try_load_complete_summary(metadata, spec, config)
    if existing_summary is not None:
        if store is not None:
            upload_complete_shard(paths, existing_summary, store)
            cleanup_complete_shard(paths)
            return {"result": "uploaded", "summary": existing_summary}
        return {"result": "skipped", "summary": existing_summary}
    source_files_before = source_file_fingerprint(metadata, spec)

    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise ImportError(
            "torchcodec is required to convert RoboTwin videos. Install project dependencies before running this script."
        ) from exc

    partials = {key: partial_path(final_path) for key, final_path in paths.items()}
    cleanup_paths(partials.values())

    offsets = np.empty(spec.frame_count, dtype=np.uint64)
    sizes = np.empty(spec.frame_count, dtype=np.uint32)
    state = np.empty((spec.frame_count, STATE_DIM), dtype=np.float32)
    action = np.empty((spec.frame_count, ACTION_DIM), dtype=np.float32)
    task_index = np.empty(spec.frame_count, dtype=np.int64)
    episodes_json: list[dict[str, int]] = []
    cursor = 0
    png_bytes_total = 0
    decoder_average_fps_by_camera: dict[str, float] = {}

    tar_handle = None
    tar_writer = None
    try:
        tar_handle = partials["tar"].open("wb")
        tar_writer = UncompressedTarWriter(tar_handle)

        for episode in spec.episodes:
            parquet_path = episode_parquet_path(metadata, episode.episode_id)
            try:
                state_rows, action_rows, task_rows, frame_indices, timestamps = load_episode_tabular(metadata, episode)
            except Exception as exc:
                raise ConversionError(
                    f"Failed to load tabular data for shard {spec.shard_index} episode {episode.episode_id} from {parquet_path}"
                ) from exc

            video_paths = {
                camera_name: episode_video_path(metadata, episode.episode_id, feature_key)
                for camera_name, feature_key in CANONICAL_CAMERAS
            }
            for camera_name, video_path in video_paths.items():
                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"Missing video for shard {spec.shard_index} episode {episode.episode_id} camera {camera_name}: {video_path}"
                    )

            decoders = {}
            try:
                for camera_name, video_path in video_paths.items():
                    decoders[camera_name] = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
            except Exception as exc:
                details = ", ".join(f"{name}={path}" for name, path in video_paths.items())
                raise ConversionError(
                    f"Failed to initialize decoders for shard {spec.shard_index} episode {episode.episode_id}: {details}"
                ) from exc
            for camera_name, video_path in video_paths.items():
                average_fps = decoder_average_fps(
                    decoders[camera_name],
                    episode_id=episode.episode_id,
                    camera_name=camera_name,
                    video_path=video_path,
                )
                if not math.isclose(average_fps, float(metadata.fps), rel_tol=0.0, abs_tol=1e-6):
                    raise ConversionError(
                        f"Episode {episode.episode_id} camera {camera_name} video {video_path} average_fps "
                        f"{average_fps} does not match metadata fps {metadata.fps}"
                    )
                validate_timestamp_alignment(
                    timestamps=timestamps,
                    frame_indices=frame_indices,
                    average_fps=average_fps,
                    episode_id=episode.episode_id,
                    camera_name=camera_name,
                    video_path=video_path,
                )
                previous_fps = decoder_average_fps_by_camera.get(camera_name)
                if previous_fps is None:
                    decoder_average_fps_by_camera[camera_name] = average_fps
                elif not math.isclose(previous_fps, average_fps, rel_tol=0.0, abs_tol=1e-6):
                    raise ConversionError(
                        f"Shard {spec.shard_index} camera {camera_name} average_fps changed across episodes: "
                        f"{previous_fps} vs {average_fps}"
                    )

            episodes_json.append(
                {
                    "episode_id": episode.episode_id,
                    "original_global_start": episode.original_global_start,
                    "shard_local_start": cursor,
                    "length": episode.length,
                }
            )

            try:
                for chunk_start in range(0, episode.length, config.decode_chunk_frames):
                    chunk_stop = min(chunk_start + config.decode_chunk_frames, episode.length)
                    chunk_frames = []
                    for camera_name, _feature_key in CANONICAL_CAMERAS:
                        video_path = video_paths[camera_name]
                        try:
                            decoded = decode_camera_chunk(
                                decoders[camera_name],
                                video_path=video_path,
                                chunk_start=chunk_start,
                                chunk_stop=chunk_stop,
                                raw_height=metadata.raw_height,
                                raw_width=metadata.raw_width,
                                episode_id=episode.episode_id,
                                camera_name=camera_name,
                            )
                        except Exception as exc:
                            raise ConversionError(
                                f"Failed decoding shard {spec.shard_index} episode {episode.episode_id} camera {camera_name} "
                                f"chunk [{chunk_start}, {chunk_stop}) from {video_path}"
                            ) from exc
                        chunk_frames.append(decoded)
                    merged = torch.cat(chunk_frames, dim=-1)
                    chunk_len = chunk_stop - chunk_start
                    if tuple(merged.shape) != (chunk_len, 3, metadata.raw_height, metadata.raw_width * len(CANONICAL_CAMERAS)):
                        raise ConversionError(
                            f"Episode {episode.episode_id} merged chunk shape {tuple(merged.shape)} is invalid"
                        )
                    for local_index in range(chunk_len):
                        source_index = chunk_start + local_index
                        global_frame_index = episode.original_global_start + source_index
                        png_bytes = png_bytes_from_frame(merged[local_index], config.png_compress_level)
                        if len(png_bytes) > np.iinfo(np.uint32).max:
                            raise ConversionError(
                                f"PNG for global frame {global_frame_index} exceeds uint32 size limit: {len(png_bytes)}"
                            )
                        offset, size = tar_writer.add_bytes(member_name(global_frame_index), png_bytes)
                        offsets[cursor] = offset
                        sizes[cursor] = size
                        state[cursor] = state_rows[source_index]
                        action[cursor] = action_rows[source_index]
                        task_index[cursor] = task_rows[source_index]
                        cursor += 1
                        png_bytes_total += size
            finally:
                decoders.clear()

        if cursor != spec.frame_count:
            raise ConversionError(
                f"Shard {spec.shard_index} wrote {cursor} frames but expected {spec.frame_count}"
            )

        tar_writer.close()
        tar_sha256 = tar_writer.hexdigest()
        tar_handle.flush()
        tar_handle.close()
        tar_handle = None
        tar_writer = None

        write_npy(partials["offsets"], offsets)
        write_npy(partials["sizes"], sizes)
        write_npy(partials["state"], state)
        write_npy(partials["action"], action)
        write_npy(partials["task_index"], task_index)
        write_json(partials["episodes"], episodes_json)
        source_files_after = source_file_fingerprint(metadata, spec)
        if source_files_after != source_files_before:
            raise ConversionError(
                f"Source files changed while converting shard {spec.shard_index}."
            )

        final_to_partial = {
            paths["tar"]: partials["tar"],
            paths["offsets"]: partials["offsets"],
            paths["sizes"]: partials["sizes"],
            paths["state"]: partials["state"],
            paths["action"]: partials["action"],
            paths["task_index"]: partials["task_index"],
            paths["episodes"]: partials["episodes"],
        }
        finalize_partial_outputs(final_to_partial)

        summary = build_shard_summary(
            spec=spec,
            metadata=metadata,
            config=config,
            tar_path=paths["tar"],
            tar_sha256=tar_sha256,
            sidecar_paths=[
                paths["offsets"],
                paths["sizes"],
                paths["state"],
                paths["action"],
                paths["task_index"],
                paths["episodes"],
            ],
            png_bytes_total=png_bytes_total,
            source_files=source_files_after,
            validation={
                "frame_index_is_sequential": True,
                "episode_index_is_constant_and_correct": True,
                "timestamp_round_fps_matches_frame_index": True,
                "decoder_average_fps_by_camera": decoder_average_fps_by_camera,
            },
        )
        write_json(partials["summary"], summary)
        os.replace(partials["summary"], paths["summary"])
        write_text(partials["done"], DONE_MARKER_TEXT)
        os.replace(partials["done"], paths["done"])
        if store is not None:
            upload_complete_shard(paths, summary, store)
            cleanup_complete_shard(paths)
            return {"result": "uploaded", "summary": summary}
        return {"result": "written", "summary": summary}
    except Exception:
        cleanup_paths(partials.values())
        raise
    finally:
        if tar_handle is not None:
            tar_handle.close()


def write_manifest(
    *,
    output_root: Path,
    metadata: SourceMetadata,
    shards: Sequence[ShardSpec],
    shard_summaries: Sequence[dict[str, Any]],
    tasks: Sequence[str],
    dataset_stats: dict[str, Any],
    episodes_per_shard: int,
    png_compress_level: int,
    max_episodes: int | None,
) -> Path:
    converted_episodes = sum(len(spec.episodes) for spec in shards)
    converted_frames = sum(spec.frame_count for spec in shards)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source_root": metadata.source_root,
        "total_source_episodes": metadata.total_source_episodes,
        "total_source_frames": metadata.total_source_frames,
        "converted_episodes": converted_episodes,
        "converted_frames": converted_frames,
        "requested_max_episodes": max_episodes,
        "fps": metadata.fps,
        "camera_keys": list(metadata.camera_short_keys),
        "camera_feature_keys": list(metadata.camera_feature_keys),
        "raw_height": metadata.raw_height,
        "raw_width": metadata.raw_width,
        "episodes_per_shard": episodes_per_shard,
        "png_compress_level": png_compress_level,
        "dataset_stats": dataset_stats,
        "tasks": {
            "count": len(tasks),
            "by_index": list(tasks),
        },
        "shards": list(shard_summaries),
    }
    manifest_path = output_root / MANIFEST_FILENAME
    partial = manifest_path.with_name(f"{manifest_path.name}.partial.{os.getpid()}")
    write_json(partial, manifest)
    os.replace(partial, manifest_path)
    return manifest_path


def self_check() -> None:
    tiny = torch.tensor(
        [
            [[0, 255], [10, 20]],
            [[30, 40], [50, 60]],
            [[70, 80], [90, 100]],
        ],
        dtype=torch.uint8,
    )
    encoded = png_bytes_from_frame(tiny, compress_level=3)
    decoded = np.asarray(Image.open(io.BytesIO(encoded)), dtype=np.uint8)
    expected = tiny.permute(1, 2, 0).cpu().numpy()
    if decoded.shape != expected.shape or not np.array_equal(decoded, expected):
        raise AssertionError("PNG round-trip failed")

    fake_metadata = SourceMetadata(
        source_root="/dataset",
        data_path_template="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path_template="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        chunks_size=1000,
        fps=50,
        total_source_episodes=5,
        total_source_frames=15,
        total_tasks=3,
        raw_height=2,
        raw_width=2,
        camera_short_keys=tuple(camera_name for camera_name, _ in CANONICAL_CAMERAS),
        camera_feature_keys=tuple(feature_key for _, feature_key in CANONICAL_CAMERAS),
        episodes=(
            EpisodeRecord(0, 3, 0),
            EpisodeRecord(1, 4, 3),
            EpisodeRecord(2, 2, 7),
            EpisodeRecord(3, 5, 9),
            EpisodeRecord(4, 1, 14),
        ),
    )
    shards = build_shard_specs(fake_metadata, episodes_per_shard=2, max_episodes=5)
    if [spec.frame_count for spec in shards] != [7, 7, 1]:
        raise AssertionError(f"Unexpected shard frame counts: {[spec.frame_count for spec in shards]}")
    if member_name(42) != "000000000042.png":
        raise AssertionError("Member naming is incorrect")

    buffer = io.BytesIO()
    writer = UncompressedTarWriter(buffer)
    offset, size = writer.add_bytes(member_name(7), b"abc")
    writer.close()
    raw = buffer.getvalue()
    if offset != 512 or size != 3:
        raise AssertionError(f"Unexpected tar offsets: offset={offset} size={size}")
    if raw[offset : offset + size] != b"abc":
        raise AssertionError("Tar payload mismatch")
    if raw[-1024:] != b"\0" * 1024:
        raise AssertionError("Tar terminator is missing")
    if default_stats_path("/repo/data/robotwin2.0/robotwin2.0") != Path("/repo/data/robotwin2.0/dataset_stats.json"):
        raise AssertionError("Default stats path is incorrect")
    if hash_bytes(b"abc") != hashlib.sha256(b"abc").hexdigest():
        raise AssertionError("Byte hashing is incorrect")
    if DATASET_DONE_FILENAME != "dataset.done":
        raise AssertionError("Dataset done filename is incorrect")


def run(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    ensure_positive("workers", int(args.workers))
    ensure_positive("episodes_per_shard", int(args.episodes_per_shard))
    ensure_positive("decode_chunk_frames", int(args.decode_chunk_frames))
    if not 0 <= int(args.png_compress_level) <= 9:
        raise ValueError(f"png_compress_level must be in [0, 9], got {args.png_compress_level}")

    source_root = resolve_path(args.source_root)
    output_root = resolve_path(args.output_root)
    stats_path = resolve_path(args.stats_path) if args.stats_path is not None else default_stats_path(source_root)
    if args.s3_output_root and str(args.output_root).startswith(("s3://", "/s3/")):
        raise ValueError("--output-root must be a local POSIX staging directory, not S3 or /s3 FUSE")
    if args.s3_output_root and args.max_episodes is not None and not args.allow_partial_s3:
        raise ValueError(
            "Partial S3 conversion requires --allow-partial-s3 and a dedicated non-production prefix."
        )
    LOGGER.info("Loading source metadata from %s", source_root)
    metadata = load_source_metadata(source_root)
    tasks = load_tasks(source_root)
    if len(tasks) != metadata.total_tasks:
        raise ValueError(
            f"tasks.jsonl count {len(tasks)} does not match info.json total_tasks {metadata.total_tasks}"
        )
    shards = build_shard_specs(metadata, args.episodes_per_shard, args.max_episodes)
    if not shards:
        raise ValueError("No shards were requested")

    requested_episodes = sum(len(spec.episodes) for spec in shards)
    requested_frames = sum(spec.frame_count for spec in shards)
    full_conversion_requested = (
        requested_episodes == metadata.total_source_episodes
        and requested_frames == metadata.total_source_frames
    )
    remove_dataset_done_marker(output_root)
    if output_root.exists():
        for stale_partial in output_root.rglob("*.partial.*"):
            cleanup_paths([stale_partial])
    dataset_stats = copy_dataset_stats(
        source_path=stats_path,
        output_root=output_root,
        overwrite=bool(args.overwrite),
    )
    LOGGER.info(
        "Preparing %d shard(s) covering %d episode(s) and %d frame(s) with %d worker(s)",
        len(shards),
        requested_episodes,
        requested_frames,
        args.workers,
    )

    config = WorkerConfig(
        output_root=str(output_root),
        source_root=metadata.source_root,
        episodes_per_shard=int(args.episodes_per_shard),
        png_compress_level=int(args.png_compress_level),
        decode_chunk_frames=int(args.decode_chunk_frames),
        overwrite=bool(args.overwrite),
        s3_output_root=args.s3_output_root,
        aws_profile=args.aws_profile,
        aws_region=args.aws_region,
        aws_credentials_file=args.aws_credentials_file,
        aws_cli=args.aws_cli,
    )
    s3_store = make_s3_store(config)
    if s3_store is not None:
        s3_store.delete(DATASET_DONE_FILENAME)
        LOGGER.info("Publishing shards to %s", args.s3_output_root)

    shard_summaries: dict[int, dict[str, Any]] = {}
    completed_shards = 0
    completed_frames = 0
    completed_bytes = 0
    skipped_shards = 0
    failures = 0
    start_time = time.perf_counter()

    mp_context = get_context("spawn")
    worker_metadata = replace(metadata, episodes=())
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=mp_context) as executor:
        future_to_spec = {
            executor.submit(process_shard, spec, worker_metadata, config): spec
            for spec in shards
        }
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                for pending in future_to_spec:
                    pending.cancel()
                raise RuntimeError(
                    f"Shard {spec.shard_index} failed for episodes {spec.episode_start}:{spec.episode_stop}"
                ) from exc

            summary = result["summary"]
            shard_summaries[spec.shard_index] = summary
            completed_shards += 1
            completed_frames += int(summary["frame_count"])
            completed_bytes += int(summary["output_bytes_total"])
            skipped_shards += int(result["result"].endswith("skipped"))

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            throughput = completed_frames / elapsed
            remaining_frames = max(requested_frames - completed_frames, 0)
            eta_seconds = remaining_frames / throughput if throughput > 0 else float("inf")
            LOGGER.info(
                "Progress %d/%d shards, %d/%d frames, %.1f frames/s, eta=%s, output=%.3f GiB, skipped=%d, failures=%d",
                completed_shards,
                len(shards),
                completed_frames,
                requested_frames,
                throughput,
                format_duration(eta_seconds),
                completed_bytes / float(1024 ** 3),
                skipped_shards,
                failures,
            )

    ordered_summaries = [shard_summaries[index] for index in sorted(shard_summaries)]
    manifest_path = write_manifest(
        output_root=output_root,
        metadata=metadata,
        shards=shards,
        shard_summaries=ordered_summaries,
        tasks=tasks,
        dataset_stats=dataset_stats,
        episodes_per_shard=int(args.episodes_per_shard),
        png_compress_level=int(args.png_compress_level),
        max_episodes=args.max_episodes,
    )
    converted_episodes = sum(int(summary["episode_count"]) for summary in ordered_summaries)
    converted_frames = sum(int(summary["frame_count"]) for summary in ordered_summaries)
    conversion_complete = (
        converted_episodes == metadata.total_source_episodes
        and converted_frames == metadata.total_source_frames
    )
    if conversion_complete and s3_store is None:
        write_dataset_done_marker(output_root)
    else:
        remove_dataset_done_marker(output_root)

    if s3_store is not None:
        stats_output_path = output_root / "dataset_stats.json"
        s3_store.upload_file(
            stats_output_path,
            stats_output_path.name,
            hash_file(stats_output_path),
        )
        s3_store.upload_file(
            manifest_path,
            manifest_path.name,
            hash_file(manifest_path),
        )
        if conversion_complete:
            dataset_done_path = output_root / DATASET_DONE_FILENAME
            write_json(
                dataset_done_path,
                {
                    "format": "robotwin-webdataset-done",
                    "version": 1,
                    "manifest_sha256": hash_file(manifest_path),
                    "shard_count": len(shards),
                    "converted_episodes": converted_episodes,
                    "converted_frames": converted_frames,
                },
            )
            s3_store.upload_file(
                dataset_done_path,
                DATASET_DONE_FILENAME,
                hash_file(dataset_done_path),
            )
        else:
            LOGGER.info("Partial conversion: remote dataset.done was intentionally not published.")
    total_elapsed = time.perf_counter() - start_time
    LOGGER.info(
        "Finished %d shard(s), %d frame(s) in %s. Manifest: %s",
        len(shards),
        requested_frames,
        format_duration(total_elapsed),
        manifest_path,
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
