import logging
import json
import inspect
import functools
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def _count_params(params) -> tuple[int, int]:
    tensor_count = 0
    param_count = 0
    for p in params:
        tensor_count += 1
        param_count += int(p.numel())
    return tensor_count, param_count


def _segment_count(segments) -> int:
    if isinstance(segments, list):
        return len(segments)
    if not isinstance(segments, dict):
        raise TypeError(f"`segments` must be a dict or list[dict], got {type(segments)}")
    video = segments.get("video")
    if torch.is_tensor(video):
        return int(video.shape[0]) if video.ndim >= 1 else 1
    for value in segments.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Unable to infer segment count from interleaved sample.")


def _segments_list_to_dict(segments: list[dict]) -> dict:
    if not segments:
        raise ValueError("Interleaved sample contains an empty segment list.")
    keys = segments[0].keys()
    out = {}
    for key in keys:
        values = [seg[key] for seg in segments]
        first = values[0]
        if torch.is_tensor(first):
            out[key] = torch.stack(values, dim=0)
        elif isinstance(first, str):
            out[key] = list(values)
        else:
            out[key] = values
    return out


def _pad_segment_tensor(values: list[torch.Tensor], max_segments: int) -> torch.Tensor:
    first = values[0]
    if any(value.shape[1:] != first.shape[1:] for value in values):
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"Segment tensor trailing shapes must match within a batch, got {shapes}")
    padded_shape = (len(values), max_segments, *first.shape[1:])
    padded = first.new_zeros(padded_shape)
    for batch_idx, value in enumerate(values):
        n = int(value.shape[0])
        padded[batch_idx, :n] = value
    return padded


def interleaved_collate(batch: list[dict]) -> dict:
    if not batch or not all(isinstance(sample, dict) and "segments" in sample for sample in batch):
        return default_collate(batch)

    normalized_segments = []
    counts = []
    for sample in batch:
        segments = sample["segments"]
        if isinstance(segments, list):
            segments = _segments_list_to_dict(segments)
        n = _segment_count(segments)
        if n <= 0:
            raise ValueError("Interleaved samples must contain at least one segment.")
        normalized_segments.append(segments)
        counts.append(n)

    max_segments = max(counts)
    segment_mask = torch.zeros((len(batch), max_segments), dtype=torch.bool)
    for batch_idx, n in enumerate(counts):
        segment_mask[batch_idx, :n] = True

    keys = set()
    for segments in normalized_segments:
        keys.update(segments.keys())

    collated_segments = {"segment_mask": segment_mask}
    for key in sorted(keys):
        values = [segments.get(key) for segments in normalized_segments]
        present = [value for value in values if value is not None]
        if not present:
            continue
        first = present[0]
        if torch.is_tensor(first):
            tensor_values = []
            for value, count in zip(values, counts):
                if value is None:
                    raise ValueError(f"Missing tensor segment key `{key}` in one batch item.")
                if value.ndim == 0 or int(value.shape[0]) != count:
                    raise ValueError(
                        f"Segment tensor `{key}` must start with segment dim {count}, got {tuple(value.shape)}"
                    )
                tensor_values.append(value)
            collated_segments[key] = _pad_segment_tensor(tensor_values, max_segments)
        elif isinstance(first, str):
            rows = []
            for value, count in zip(values, counts):
                if value is None:
                    row = ["" for _ in range(count)]
                elif isinstance(value, str):
                    if count != 1:
                        raise ValueError(f"String segment key `{key}` is only valid for one segment.")
                    row = [value]
                else:
                    row = list(value)
                if len(row) != count:
                    raise ValueError(f"Segment key `{key}` length {len(row)} != segment count {count}.")
                rows.append(row + ["" for _ in range(max_segments - count)])
            collated_segments[key] = rows
        else:
            rows = []
            for value, count in zip(values, counts):
                row = list(value) if isinstance(value, (list, tuple)) else [value]
                if len(row) != count:
                    raise ValueError(f"Segment key `{key}` length {len(row)} != segment count {count}.")
                rows.append(row + [None for _ in range(max_segments - count)])
            collated_segments[key] = rows

    return {"segments": collated_segments}


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.dataloader_timeout = float(cfg.get("dataloader_timeout", 0))
        self.dataloader_prefetch_factor = cfg.get("dataloader_prefetch_factor", None)
        self.dataloader_persistent_workers = bool(cfg.get("dataloader_persistent_workers", False))
        self.dataloader_multiprocessing_context = cfg.get("dataloader_multiprocessing_context", None)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.trainable_components = list(cfg.get("trainable_components", ["dit"]))
        self.projection_lr_multiplier = float(cfg.get("projection_lr_multiplier", 10.0))
        self.freeze_visual_encoder = bool(cfg.get("freeze_visual_encoder", False))
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # When running plain DDP (no DeepSpeed), some trainable params may not
        # receive grads on every step (route-dependent experts / skipped
        # cross-attention), which makes DDP raise unless we allow unused params.
        # DeepSpeed manages its own reduction and ignores these kwargs.
        _accel_kwargs = {}
        # Default: no manually-synced params (only populated on the FSDP path).
        self._fsdp_ignored_trainable_params = []
        _use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "").lower() in ("1", "true", "yes")
        if _use_fsdp:
            # FSDP path. The frozen language expert contains an nn.Embedding fed
            # by plain int token-id tensors. FSDP's *root* wrap flattens ALL
            # params not covered by a child wrap into a root flat-param and
            # shards them into DTensors -> language token_embedding.weight
            # becomes a DTensor while token_ids stay plain tensors -> the
            # "mixed torch.Tensor and DTensor" error in aten.embedding.
            #
            # An auto_wrap_policy alone cannot prevent this (it only controls
            # *child* wraps, not the root). We must additionally pass the frozen
            # modules as `ignored_modules` so FSDP never touches them. Frozen
            # modules also gain nothing from optimizer-state sharding.
            from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
            from accelerate import FullyShardedDataParallelPlugin
            from .models.wan22.wan_video_dit import DiTBlock

            def _wrap_only_trainable_blocks(module):
                if not isinstance(module, DiTBlock):
                    return False
                # Wrap only if this block has at least one trainable parameter.
                return any(p.requires_grad for p in module.parameters(recurse=True))

            _policy = functools.partial(
                lambda_auto_wrap_policy, lambda_fn=_wrap_only_trainable_blocks
            )

            # Ignore every top-level submodule that is fully frozen (no trainable
            # params): language_expert (embeddings), vae, etc. This keeps their
            # weights as plain tensors.
            #
            # Also ignore trainable modules that are *not* wrapped by the
            # DiTBlock policy, otherwise FSDP's root wrap will flatten/shard them
            # into DTensors. Their forwards consume plain dense tensors
            # (e.g. proprio_encoder(flat_proprio), video_expert.time_embedding(...),
            # action_expert.action_encoder(...)), which then trips the
            # "mixed torch.Tensor and DTensor" error in aten.addmm/aten.linear.
            # These modules are small relative to the DiT blocks, so keeping them
            # replicated is the pragmatic tradeoff.

            def _add_ignored(module, reason: str):
                if module is None:
                    return
                key = id(module)
                if key in _ignored_ids:
                    return
                _ignored_ids.add(key)
                _ignored.append(module)
                logger.info("FSDP: ignoring module '%s'.", reason)

            _ignored = []
            _ignored_ids = set()
            for _name, _sub in self.model.named_children():
                _has_trainable = any(p.requires_grad for p in _sub.parameters(recurse=True))
                if not _has_trainable:
                    _add_ignored(_sub, f"{_name} (fully frozen)")

            # Top-level small trainable modules.
            _add_ignored(getattr(self.model, "proprio_encoder", None), "proprio_encoder (trainable linear)")
            _add_ignored(getattr(self.model, "visual_encoder", None), "visual_encoder (non-DiT encoder/projection)")

            # Video expert pre/post modules outside DiTBlock.
            _video_expert = getattr(self.model, "video_expert", None)
            if _video_expert is not None:
                for _sub_name in (
                    "patch_embedding",
                    "text_embedding",
                    "time_embedding",
                    "time_projection",
                    "head",
                    "action_embedding",
                    "ref_conv",
                    "control_adapter",
                ):
                    _add_ignored(
                        getattr(_video_expert, _sub_name, None),
                        f"video_expert.{_sub_name} (non-DiT pre/post module)",
                    )

            # Action expert pre/post modules outside DiTBlock.
            _action_expert = getattr(self.model, "action_expert", None)
            if _action_expert is not None:
                for _sub_name in (
                    "action_encoder",
                    "text_embedding",
                    "time_embedding",
                    "time_projection",
                    "head",
                ):
                    _add_ignored(
                        getattr(_action_expert, _sub_name, None),
                        f"action_expert.{_sub_name} (non-DiT pre/post module)",
                    )

            _fsdp_plugin = FullyShardedDataParallelPlugin(
                auto_wrap_policy=_policy,
                use_orig_params=True,
                ignored_modules=_ignored if _ignored else None,
            )
            _accel_kwargs["fsdp_plugin"] = _fsdp_plugin
            logger.info("FSDP enabled: wrapping only trainable DiTBlocks; %d frozen submodule(s) ignored.", len(_ignored))

            # FSDP does NOT manage gradient synchronization for ignored_modules.
            # Some ignored modules are small but TRAINABLE (e.g.
            # video_expert.time_embedding / head, action_expert.head). Without a
            # fallback, their grads diverge across ranks -> silently wrong
            # training. We collect their params here and manually all-reduce
            # (mean) their grads before each optimizer step. These modules are
            # tiny relative to the DiT blocks, so the extra all-reduce is cheap.
            _ign_trainable = []
            for _m in _ignored:
                for _p in _m.parameters(recurse=True):
                    if _p.requires_grad:
                        _ign_trainable.append(_p)
            self._fsdp_ignored_trainable_params = _ign_trainable
            logger.info(
                "FSDP: %d trainable params in ignored modules will be grad-synced manually.",
                len(_ign_trainable),
            )
        elif os.environ.get("FASTWAM_DDP_FIND_UNUSED", "1") == "1":
            from accelerate import DistributedDataParallelKwargs
            _ddp_gradient_as_bucket_view = os.environ.get("FASTWAM_DDP_GRAD_BUCKET_VIEW", "1") == "1"
            _accel_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(
                    find_unused_parameters=True,
                    gradient_as_bucket_view=_ddp_gradient_as_bucket_view,
                )
            ]
            logger.info(
                "DDP enabled: find_unused_parameters=true gradient_as_bucket_view=%s",
                _ddp_gradient_as_bucket_view,
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            **_accel_kwargs,
        )
        
        _ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        _zero_stage = (
            _ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
            if _ds_plugin is not None
            else "none(non-deepspeed)"
        )
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            _zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)

        # Build parameter groups from the post-freeze trainable set only.
        # This is critical for H-FastWAM: model.dit is the whole MoT
        # (language + video + action). If we hand the full parameter list to the
        # optimizer/DeepSpeed, frozen experts still get optimizer state / ZeRO
        # bookkeeping, which can dominate backward time.
        dit_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            dit_params.extend([p for p in proprio_encoder.parameters() if p.requires_grad])

        dit_tensor_count, dit_param_count = _count_params(dit_params)
        logger.info(
            "Optimizer base group: trainable_tensors=%d trainable_params=%.3fB",
            dit_tensor_count,
            dit_param_count / 1e9,
        )

        param_groups = [
            {"params": dit_params, "lr": self.learning_rate},
        ]

        # Include DINO MLP projection in optimizer if using DINOEncoder and not frozen.
        if getattr(self.model, "use_visual_encoder", False) and not self.freeze_visual_encoder:
            proj_params = [p for p in self.model.visual_encoder.projection.parameters() if p.requires_grad]
            if proj_params:  # skip_projection=True → nn.Identity() → no params
                projection_lr = self.learning_rate * self.projection_lr_multiplier
                proj_tensor_count, proj_param_count = _count_params(proj_params)
                param_groups.append({
                    "params": proj_params,
                    "lr": projection_lr,
                })
                logger.info(
                    "Projection LR: %.2e (%.1fx base LR %.2e), trainable_tensors=%d trainable_params=%.3fM",
                    projection_lr,
                    self.projection_lr_multiplier,
                    self.learning_rate,
                    proj_tensor_count,
                    proj_param_count / 1e6,
                )
            else:
                logger.info(
                    "Visual encoder in skip_projection mode (DiT-side projection). "
                    "No separate projection params — patchify Conv3d is trained as part of DiT."
                )
        elif getattr(self.model, "use_visual_encoder", False) and self.freeze_visual_encoder:
            logger.info("Visual encoder projection is FROZEN (freeze_visual_encoder=true).")

        # AdamW implementation selection (memory vs speed):
        #   fused=True   -> fused CUDA kernel, NO large temporary copy in step()
        #                   (avoids the ~24G transient that torch._foreach_sqrt
        #                   allocates in the default foreach path) AND faster.
        #   foreach=True -> default multi-tensor path (fast, high transient mem).
        #   neither      -> per-tensor loop, lowest peak memory, slowest.
        # Default to fused to fix the optimizer.step() OOM under DDP.
        _adam_fused = os.environ.get("FASTWAM_ADAM_FUSED", "1") == "1"
        _adam_foreach = os.environ.get("FASTWAM_ADAM_FOREACH", "0") == "1"
        _adam_kwargs = {}
        if _adam_fused:
            _adam_kwargs["fused"] = True
        elif _adam_foreach:
            _adam_kwargs["foreach"] = True
        else:
            _adam_kwargs["foreach"] = False
        logger.info("AdamW impl kwargs: %s", _adam_kwargs)

        # ZeRO-1 (optimizer-state sharding) inside plain PyTorch DDP.
        #
        # The full fp32 AdamW state for ~6.7B trainable params is ~54G/GPU under
        # DDP, which OOMs an 80G card. ZeroRedundancyOptimizer shards that state
        # across the data-parallel ranks (54G / world_size), bringing per-GPU
        # optimizer memory down to a few GB while staying in DDP (no FSDP, no
        # DeepSpeed). Each rank steps its owned shard then all-gathers the
        # updated params -- the gradient reduction is still DDP's flat bucketed
        # all-reduce, so no per-param O(N^2) path.
        #
        # Compatibility note: accelerate's AcceleratedOptimizer.__init__ calls
        # optimizer.state_dict() when device_placement=True, which is unsafe for
        # a ZeRO optimizer that hasn't stepped/consolidated yet. We therefore
        # prepare() the optimizer with device_placement=False (a no-op anyway,
        # since there is no optimizer state to move before the first step).
        self._using_zero = (
            os.environ.get("FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER", "0") == "1"
            and str(self.accelerator.distributed_type) == "DistributedType.MULTI_GPU"
            and self.accelerator.num_processes > 1
        )
        if self._using_zero:
            from torch.distributed.optim import ZeroRedundancyOptimizer

            self.optimizer = ZeroRedundancyOptimizer(
                param_groups[0]["params"],
                optimizer_class=torch.optim.AdamW,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
                **_adam_kwargs,
            )
            for _extra_group in param_groups[1:]:
                self.optimizer.add_param_group(_extra_group)
            logger.info(
                "ZeRO-1 enabled: ZeroRedundancyOptimizer sharding AdamW state across %d ranks "
                "(prepare() will use device_placement=False for the optimizer).",
                self.accelerator.num_processes,
            )
        else:
            if os.environ.get("FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER", "0") == "1":
                logger.warning(
                    "FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 requested but not applicable "
                    "(distributed_type=%s, num_processes=%d). Using plain AdamW.",
                    self.accelerator.distributed_type,
                    self.accelerator.num_processes,
                )
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
                **_adam_kwargs,
            )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        # device_placement is per-object, matching the prepare() arg order
        # (model, optimizer, loader, scheduler). When using torch-ZeRO we must
        # skip device placement for the optimizer to avoid accelerate calling
        # state_dict() on an un-stepped ZeroRedundancyOptimizer.
        #
        # DeepSpeed/Megatron forbid passing device_placement at all (accelerate
        # raises "You can't customize device placements with DeepSpeed..."), so
        # only pass it on the torch-ZeRO path; otherwise use the default call.
        if self._using_zero:
            _dp = [None, False, None, None]
            self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler,
                device_placement=_dp,
            )
        else:
            self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.scheduler,
            )
        self.optimizer.zero_grad(set_to_none=True)

        # Gradient communication precision: register a bf16 compression comm
        # hook on the DDP-wrapped model. Gradients are cast to bf16 for the
        # all-reduce (and the reduction temporaries live in bf16), which roughly
        # halves the gradient-bucket / communication memory that shows up in the
        # backward-pass peak, while parameters and the optimizer master copy stay
        # fp32. Only meaningful for plain DDP (not DeepSpeed/FSDP, which manage
        # their own reduce dtype). On by default; set FASTWAM_DDP_BF16_COMM=0 to
        # fall back to full-precision all-reduce.
        if os.environ.get("FASTWAM_DDP_BF16_COMM", "1") == "1":
            self._maybe_register_bf16_comm_hook()

        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _sync_fsdp_ignored_grads(self):
        """All-reduce(mean) grads of trainable params in FSDP ignored_modules.

        FSDP does not synchronize gradients for ignored_modules, so without
        this their grads would diverge across ranks. No-op when the list is
        empty (non-FSDP paths) or world_size == 1. Cheap: these modules are
        tiny relative to the sharded DiT blocks.
        """
        params = getattr(self, "_fsdp_ignored_trainable_params", None)
        if not params:
            return
        if self.accelerator.num_processes <= 1:
            return
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        world = self.accelerator.num_processes
        grads = [p.grad for p in params if p.grad is not None]
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g.div_(world)

    def _maybe_register_bf16_comm_hook(self):
        """Register a bf16 gradient-compression comm hook on the DDP model.

        No-op unless self.model is a torch DistributedDataParallel instance
        (i.e. plain DDP under accelerate). DeepSpeed/FSDP wrap the model in
        their own classes and manage reduce precision internally, so we skip
        them. Safe to call once after accelerator.prepare().
        """
        from torch.nn.parallel import DistributedDataParallel as _DDP

        ddp_model = self.model
        # accelerate may return the bare DDP module; unwrap one level if needed.
        if not isinstance(ddp_model, _DDP):
            inner = getattr(ddp_model, "module", None)
            if isinstance(inner, _DDP):
                ddp_model = inner
        if not isinstance(ddp_model, _DDP):
            logger.info(
                "FASTWAM_DDP_BF16_COMM=1 but model is %s (not DDP); skipping bf16 comm hook.",
                type(self.model).__name__,
            )
            return
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as _dh

            # state=None -> the hook uses the default process group, which is
            # what accelerate sets up for single-node DDP.
            ddp_model.register_comm_hook(state=None, hook=_dh.bf16_compress_hook)
            logger.info(
                "Registered bf16 gradient-compression DDP comm hook "
                "(grads all-reduced in bf16; params/optimizer stay fp32)."
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to register bf16 comm hook (%s); using default all-reduce.", exc)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
            "timeout": self.dataloader_timeout,
            "collate_fn": interleaved_collate,
        }
        if self.num_workers > 0:
            if self.dataloader_prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(self.dataloader_prefetch_factor)
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            if self.dataloader_multiprocessing_context not in (None, "null", ""):
                loader_kwargs["multiprocessing_context"] = str(self.dataloader_multiprocessing_context)
        logger.info(
            "DataLoader config: batch_size=%d num_workers=%d timeout=%.1fs prefetch_factor=%s "
            "persistent_workers=%s multiprocessing_context=%s",
            self.batch_size,
            self.num_workers,
            self.dataloader_timeout,
            loader_kwargs.get("prefetch_factor", "default"),
            loader_kwargs.get("persistent_workers", False),
            loader_kwargs.get("multiprocessing_context", "default"),
        )
        return DataLoader(dataset, **loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _apply_dit_only_train_mode(self, model):
        """Pre-accelerator freeze: only DiT + optional components stay trainable."""
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)

        if bool(getattr(model, "freeze_language_expert", False)) and hasattr(model, "language_expert"):
            model.language_expert.eval()
            model.language_expert.requires_grad_(False)
        if bool(getattr(model, "freeze_video_expert", False)) and hasattr(model, "video_expert"):
            model.video_expert.eval()
            model.video_expert.requires_grad_(False)
        if bool(getattr(model, "freeze_action_expert", False)) and hasattr(model, "action_expert"):
            model.action_expert.eval()
            model.action_expert.requires_grad_(False)

        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        if getattr(model, "use_visual_encoder", False) and not self.freeze_visual_encoder:
            proj_params = list(model.visual_encoder.projection.parameters())
            if proj_params:  # has MLP (not skip_projection mode)
                model.visual_encoder.projection.train()
                model.visual_encoder.projection.requires_grad_(True)

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        # Extended to support configurable trainable_components.
        model = self.accelerator.unwrap_model(self.model)
        components = self.trainable_components
        logger.info("Setting trainable components: %s (freezing everything else).", components)

        # First freeze everything
        model.eval()
        model.requires_grad_(False)

        # Unfreeze DiT (always included by default)
        if "dit" in components:
            model.dit.train()
            model.dit.requires_grad_(True)

        if bool(getattr(model, "freeze_language_expert", False)) and hasattr(model, "language_expert"):
            model.language_expert.eval()
            model.language_expert.requires_grad_(False)
        if bool(getattr(model, "freeze_video_expert", False)) and hasattr(model, "video_expert"):
            model.video_expert.eval()
            model.video_expert.requires_grad_(False)
        if bool(getattr(model, "freeze_action_expert", False)) and hasattr(model, "action_expert"):
            model.action_expert.eval()
            model.action_expert.requires_grad_(False)

        # Unfreeze VAE
        if "vae" in components:
            if hasattr(model, "vae"):
                model.vae.train()
                model.vae.requires_grad_(True)
                logger.info("VAE encoder/decoder set to trainable.")
            else:
                logger.warning("trainable_components includes 'vae' but model has no 'vae' attribute.")

        # Unfreeze text encoder
        if "text_encoder" in components:
            if hasattr(model, "text_encoder") and model.text_encoder is not None:
                model.text_encoder.train()
                model.text_encoder.requires_grad_(True)
                logger.info("Text encoder set to trainable.")

        # Unfreeze proprio encoder (always if present, for backward compat)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

        # If using DINOEncoder, always unfreeze its MLP projection (backbone stays frozen).
        if getattr(model, "use_visual_encoder", False) and not self.freeze_visual_encoder:
            proj_params = list(model.visual_encoder.projection.parameters())
            if proj_params:  # has MLP (not skip_projection mode)
                model.visual_encoder.projection.train()
                model.visual_encoder.projection.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        if "segments" in sample:
            return Wan22Trainer._to_batched_interleaved_eval_sample(sample)

        video = sample["video"]
        prompt = sample.get("prompt", None)
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]
            elif isinstance(prompt, tuple):
                prompt = list(prompt)
            elif not isinstance(prompt, list):
                raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
            if len(prompt) != video.shape[0]:
                raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        elif context is None or context_mask is None:
            raise ValueError("Eval sample must contain either 'prompt' or both 'context' and 'context_mask'.")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @staticmethod
    def _first_segment_prompt(prompt):
        if prompt is None or isinstance(prompt, str):
            return prompt
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        if not isinstance(prompt, list) or len(prompt) == 0:
            raise TypeError(f"Expected segment prompt type str/list[str], got {type(prompt)}")

        first = prompt[0]
        if isinstance(first, str):
            return first
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], str):
            return first[0]
        raise TypeError(f"Expected segment prompt entries to be str/list[str], got {type(first)}")

    @staticmethod
    def _segment_prompt_at(prompt, *, batch_idx: int, segment_idx: int, batch_size: int, num_segments: int):
        if prompt is None or isinstance(prompt, str):
            return prompt
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        if not isinstance(prompt, list):
            raise TypeError(f"Expected segment prompt type str/list[str], got {type(prompt)}")
        if batch_size == 1 and len(prompt) == num_segments and all(isinstance(item, str) for item in prompt):
            return prompt[segment_idx]
        if len(prompt) == batch_size:
            row = prompt[batch_idx]
            if isinstance(row, str):
                if num_segments != 1:
                    raise ValueError("Flat prompt rows are only valid when num_segments=1.")
                return row
            if isinstance(row, tuple):
                row = list(row)
            if not isinstance(row, list) or len(row) != num_segments:
                raise ValueError("Segment prompts must be nested as [B][N] strings.")
            return row[segment_idx]
        if len(prompt) == num_segments and all(isinstance(row, (list, tuple)) for row in prompt):
            if any(len(row) != batch_size for row in prompt):
                raise ValueError("Transposed segment prompts must be nested as [N][B] strings.")
            return prompt[segment_idx][batch_idx]
        raise ValueError(
            f"Prompt batch mismatch: got outer len {len(prompt)}, "
            f"expected batch_size={batch_size} or num_segments={num_segments}"
        )

    @staticmethod
    def _batch_segment_prompts(prompt, *, batch_size: int, num_segments: int):
        if prompt is None:
            return None
        if isinstance(prompt, str):
            if batch_size == 1 and num_segments == 1:
                return [[prompt]]
            raise ValueError("A string segment prompt is only valid for [B=1,N=1].")
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        if not isinstance(prompt, list):
            raise TypeError(f"Expected segment prompt type str/list[str], got {type(prompt)}")

        if len(prompt) == num_segments and all(isinstance(item, str) for item in prompt):
            if batch_size != 1:
                raise ValueError("Unbatched segment prompt lists are only valid for batch_size=1.")
            return [prompt]
        return prompt

    @staticmethod
    def _to_batched_interleaved_eval_sample(sample):
        segments = sample["segments"]
        if isinstance(segments, list):
            segments = _segments_list_to_dict(segments)
        if not isinstance(segments, dict):
            raise TypeError(f"`sample['segments']` must be a dict or list[dict], got {type(segments)}")

        video = segments.get("video")
        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor `segments['video']` for evaluation, got {type(video)}. "
                "Interleaved evaluation expects [N,3,T,H,W] or [B,N,3,T,H,W]."
            )
        if video.ndim == 5:
            num_segments = int(video.shape[0])
            batch_size = 1
            batched_segments = {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
                for key, value in segments.items()
            }
        elif video.ndim == 6:
            batch_size = int(video.shape[0])
            num_segments = int(video.shape[1])
            batched_segments = dict(segments)
        else:
            raise ValueError(
                "Interleaved `segments['video']` must be [N,3,T,H,W] or [B,N,3,T,H,W], "
                f"got {tuple(video.shape)}"
            )
        if num_segments <= 0:
            raise ValueError("Interleaved eval sample must contain at least one segment.")
        if segments.get("prompt", None) is not None:
            batched_segments["prompt"] = Wan22Trainer._batch_segment_prompts(
                segments.get("prompt", None),
                batch_size=batch_size,
                num_segments=num_segments,
            )
        segment_mask = batched_segments.get("segment_mask", sample.get("segment_mask", None))
        if segment_mask is None:
            segment_mask = torch.ones((batch_size, num_segments), dtype=torch.bool)
        else:
            if not isinstance(segment_mask, torch.Tensor):
                segment_mask = torch.as_tensor(segment_mask, dtype=torch.bool)
            segment_mask = segment_mask.to(dtype=torch.bool)
            if segment_mask.ndim == 1 and batch_size == 1:
                segment_mask = segment_mask.unsqueeze(0)
            if segment_mask.shape != (batch_size, num_segments):
                raise ValueError(
                    f"`segment_mask` must be [B,N]={batch_size,num_segments}, got {tuple(segment_mask.shape)}"
                )
        if not bool(segment_mask[0].any().item()):
            raise ValueError("Interleaved eval sample has no valid segments for batch item 0.")
        first_valid_segment = int(segment_mask[0].nonzero(as_tuple=False)[0].item())
        batched_segments["segment_mask"] = segment_mask

        first_segment = {
            "video": batched_segments["video"][0, first_valid_segment],
            "prompt": Wan22Trainer._segment_prompt_at(
                batched_segments.get("prompt", None),
                batch_idx=0,
                segment_idx=first_valid_segment,
                batch_size=batch_size,
                num_segments=num_segments,
            ),
        }
        for key in ("action", "proprio", "context", "context_mask"):
            value = batched_segments.get(key)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"`segments['{key}']` must be a torch.Tensor, got {type(value)}")
                first_segment[key] = value[0, first_valid_segment]

        batched = Wan22Trainer._to_batched_eval_sample(first_segment)
        batched["segments"] = batched_segments
        return batched

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        # When using a non-VAE visual encoder (DINO / V-JEPA2), video decode
        # is unavailable. HFastWAM VAE mode can now use its FastWAM-compatible
        # infer() path; visual-encoder mode remains action-only.
        _is_hfastwam = hasattr(model, "language_expert") and hasattr(model, "infer_action")
        _can_decode_video = not getattr(model, "use_visual_encoder", False)
        _is_hfastwam_action_only = _is_hfastwam and not _can_decode_video

        prompt = sample["prompt"][0] if sample.get("prompt") is not None else None
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        pred_video = None
        pred_action = None
        psnr_rollout_vs_gt = 0.0
        ssim_rollout_vs_gt = 0.0
        psnr_decode_vs_gt = 0.0
        ssim_decode_vs_gt = 0.0
        psnr_rollout_vs_decode = 0.0
        ssim_rollout_vs_decode = 0.0

        if _is_hfastwam_action_only and action is not None:
            pred = model.infer_action(
                input_image=input_image,
                action_horizon=sample.get("action_horizon"),
                proprio=proprio,
                prompt=prompt,
                num_inference_steps=self.eval_num_inference_steps,
                seed=42,
                tiled=False,
            )
            pred_action = pred.get("action", None)

        if _can_decode_video:
            # 2. inference and video saving
            infer_kwargs = {
                "prompt": prompt,
                "input_image": input_image,
                "num_frames": num_frames,
                "action": action,
                "action_horizon": sample.get('action_horizon'),
                "proprio": proprio,
                "text_cfg_scale": 1.0,
                "action_cfg_scale": 1.0,
                "num_inference_steps": self.eval_num_inference_steps,
                "seed": 42,
                "tiled": False,
            }
            if sample["context"] is not None:
                # Pre-encoded context takes precedence over `prompt` (avoids
                # double-encoding the same text). Model APIs treat the two as
                # mutually exclusive, so explicitly drop `prompt` here.
                infer_kwargs["prompt"] = None
                infer_kwargs["context"] = sample["context"][0]
                infer_kwargs["context_mask"] = sample["context_mask"][0]

            pred = model.infer(
                **infer_kwargs,
            )

            pred_video = pred["video"]
            pred_action = pred.get("action", None)

            # 3. inference metrics against GT video
            pred_video_tensor = pil_frames_to_video_tensor(pred_video)
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

            assert pred_video_tensor.shape == gt_video_tensor.shape, (
                "Eval infer prediction/GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

            psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
            ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        if _can_decode_video:
            # 4. VAE reconstruction metrics against GT video
            gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
            vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
            vae_recon_video = model._decode_latents(vae_latents, tiled=False)
            vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

            assert vae_video_tensor.shape == gt_video_tensor.shape, (
                "Eval VAE reconstruction/GT shape mismatch: "
                f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

            psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
            ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

            psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
            ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

            stitched_video_tensor = torch.cat(
                [pred_video_tensor, vae_video_tensor, gt_video_tensor],
                dim=2,
            ).contiguous()
            stitched_frames = []
            for t in range(stitched_video_tensor.shape[1]):
                frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                stitched_frames.append(Image.fromarray(frame))

            video_path = os.path.join(
                self.eval_dir,
                f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
            )
            save_mp4(stitched_frames, video_path, fps=8)

        if not _can_decode_video:
            video_path = None

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_values = gathered_metrics[:, 7]
        action_l1_values = gathered_metrics[:, 8]
        valid_action_l2 = action_l2_values >= 0
        valid_action_l1 = action_l1_values >= 0
        action_l2_mean = action_l2_values[valid_action_l2].mean().item() if bool(valid_action_l2.any().item()) else None
        action_l1_mean = action_l1_values[valid_action_l1].mean().item() if bool(valid_action_l1.any().item()) else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        if self._using_zero:
            # ZeRO-1: ZeroRedundancyOptimizer.state_dict() requires prior
            # consolidate_state_dict(to=rank); accelerate.save_state() calls
            # state_dict() on EVERY rank without knowing this, so it crashes
            # on all non-target ranks.
            #
            # Fix: consolidate onto rank 0 (collective call — all ranks must
            # participate), then temporarily stub state_dict on non-zero ranks
            # to return {} so accelerate's save loop doesn't raise.  accelerate
            # only *writes* the optimizer file on rank 0 (single-node default),
            # so the stub value never reaches disk.  On resume,
            # accelerator.load_state reads rank 0's full-state file and passes
            # it to ZeroRedundancyOptimizer.load_state_dict on all ranks, which
            # redistributes shards correctly.
            _zero_opt = getattr(self.optimizer, "optimizer", self.optimizer)
            _zero_opt.consolidate_state_dict(to=0)  # collective barrier
            if not self.accelerator.is_main_process:
                _orig_state_dict = _zero_opt.state_dict
                _zero_opt.state_dict = lambda: {}
            try:
                self.accelerator.save_state(output_dir=state_path)
            finally:
                if not self.accelerator.is_main_process:
                    _zero_opt.state_dict = _orig_state_dict
        else:
            self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            is_first_step = self.global_step == self.run_start_step and self.batch_in_epoch == 0
            try:
                if is_first_step and self.accelerator.is_main_process:
                    logger.info("Fetching first training batch from dataloader...")
                sample = next(data_iter)
                self.batch_in_epoch += 1
                if is_first_step and self.accelerator.is_main_process:
                    logger.info("Fetched first training batch; running first training_loss forward.")
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            # --- lightweight step profiler (opt-in via FASTWAM_PROFILE_STEPS) --- #
            _profile_steps = int(os.environ.get("FASTWAM_PROFILE_STEPS", "0"))
            _do_profile = (
                _profile_steps > 0
                and self.global_step < _profile_steps
                and torch.cuda.is_available()
                and self.accelerator.is_main_process
            )

            def _cuda_t():
                if _do_profile:
                    torch.cuda.synchronize()
                return time.perf_counter()

            # --- deep torch profiler (opt-in via FASTWAM_TORCH_PROFILE=1, first step only) --- #
            _torch_prof_on = (
                os.environ.get("FASTWAM_TORCH_PROFILE", "0") == "1"
                and self.global_step == self.run_start_step
                and self.batch_in_epoch <= 1
                and self.accelerator.is_main_process
            )

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                if _torch_prof_on:
                    from torch.profiler import profile, ProfilerActivity
                    logger.info("[torch-profile] capturing first micro-step fwd+bwd ...")
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=False,
                        with_stack=False,
                    ) as _prof:
                        with self.accelerator.autocast():
                            _loss, _ld = train_model.training_loss(sample)
                        self.accelerator.backward(_loss)
                        torch.cuda.synchronize()
                    logger.info(
                        "[torch-profile] TOP BY CUDA TIME:\n%s",
                        _prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
                    )
                    logger.info(
                        "[torch-profile] TOP BY CPU TIME:\n%s",
                        _prof.key_averages().table(sort_by="cpu_time_total", row_limit=20),
                    )

                _t0 = _cuda_t()
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                _t_fwd = _cuda_t()
                if is_first_step and self.accelerator.is_main_process:
                    logger.info("Finished first training_loss forward; running backward.")
                self.accelerator.backward(loss)
                _t_bwd = _cuda_t()
                if is_first_step and self.accelerator.is_main_process:
                    logger.info("Finished first backward; waiting for optimizer step.")

                if self.accelerator.sync_gradients:
                    self._sync_fsdp_ignored_grads()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    _t_clip = _cuda_t()
                    self.optimizer.step()
                    _t_opt = _cuda_t()
                    if _do_profile:
                        _stats = torch.cuda.memory_stats()
                        _retries = _stats.get("num_alloc_retries", 0)
                        _alloc_gb = torch.cuda.memory_allocated() / 1e9
                        _reserved_gb = torch.cuda.memory_reserved() / 1e9
                        _max_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
                        logger.info(
                            "[profile] micro_step fwd=%.3fs bwd=%.3fs clip=%.3fs opt=%.3fs | "
                            "mem alloc=%.2fG reserved=%.2fG max_alloc=%.2fG alloc_retries=%d",
                            _t_fwd - _t0, _t_bwd - _t_fwd, _t_clip - _t_bwd, _t_opt - _t_clip,
                            _alloc_gb, _reserved_gb, _max_alloc_gb, _retries,
                        )
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
