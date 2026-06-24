import contextlib
import logging
import math
import os
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    build_datasets,
)
from fastwam.trainer import _count_params, interleaved_collate
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.fs import ensure_dir
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils.samplers import ResumableEpochSampler

register_default_resolvers()
logger = get_logger(__name__)


def _is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _init_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _apply_dit_only_train_mode(model, freeze_visual_encoder: bool):
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

    if getattr(model, "use_visual_encoder", False) and not freeze_visual_encoder:
        proj = getattr(model.visual_encoder, "projection", None)
        proj_params = list(proj.parameters()) if proj is not None else []
        if proj_params:
            proj.train()
            proj.requires_grad_(True)


def _make_mixed_precision_policy(mixed_precision: str):
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return None
    reduce_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return MixedPrecision(
        param_dtype=reduce_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=reduce_dtype,
    )


def _wrap_trainable_blocks(module, *, device_id: int, mp_policy) -> list[FSDP]:
    wrapped = []
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        return wrapped

    for idx, block in enumerate(blocks):
        if not any(p.requires_grad for p in block.parameters()):
            continue
        fsdp_block = FSDP(
            block,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            use_orig_params=True,
            device_id=device_id,
        )
        blocks[idx] = fsdp_block
        wrapped.append(fsdp_block)
    return wrapped


def _wrap_model_for_fsdp(model, *, local_rank: int, mixed_precision: str) -> list[FSDP]:
    mp_policy = _make_mixed_precision_policy(mixed_precision)
    wrapped = []
    for name in ("language_expert", "video_expert", "action_expert"):
        expert = getattr(model, name, None)
        if expert is None:
            continue
        cur = _wrap_trainable_blocks(expert, device_id=local_rank, mp_policy=mp_policy)
        if cur:
            logger.info("FSDP wrapped %d trainable block(s) under %s.", len(cur), name)
            wrapped.extend(cur)
    if not wrapped:
        raise RuntimeError("No trainable transformer blocks were wrapped by FSDP.")
    return wrapped


def _estimate_total_train_steps(dataset_len: int, *, batch_size: int, world_size: int, grad_accum: int, num_epochs: int, max_steps):
    if max_steps is not None:
        return max(int(max_steps), 1)
    global_batch = max(batch_size * world_size, 1)
    micro_steps_per_epoch = max(math.ceil(dataset_len / global_batch), 1)
    opt_steps_per_epoch = max(math.ceil(micro_steps_per_epoch / grad_accum), 1)
    return max(opt_steps_per_epoch * num_epochs, 1)


def _build_scheduler(optimizer, scheduler_type, *, total_train_steps: int, warmup_steps: int, learning_rate: float):
    scheduler_type = str(scheduler_type).strip().lower()
    total_train_steps = max(int(total_train_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
    remain = max(total_train_steps - warmup_steps, 1)
    if scheduler_type == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=remain, eta_min=learning_rate * 0.01)
    elif scheduler_type == "constant":
        main = ConstantLR(optimizer, factor=1.0, total_iters=remain)
    else:
        raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")
    if warmup_steps <= 0:
        return main
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_steps, 1),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])


def _autocast_context(mixed_precision: str):
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _all_reduce_mean(value: torch.Tensor) -> float:
    tensor = value.detach().float().reshape(1)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def _save_weights_only_checkpoint(model, path: str, step: int):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
    if _is_main_process():
        torch.save({"model": state, "step": int(step)}, path)
        logger.info("Saved FSDP weights checkpoint to %s", path)


def _load_resume_checkpoint_if_needed(model, resume: str | None):
    if not resume:
        return
    resume_path = Path(str(resume))
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    blob = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        logger.info(
            "Loaded standalone FSDP checkpoint %s (missing=%d unexpected=%d)",
            resume,
            len(missing),
            len(unexpected),
        )
        return
    model.load_checkpoint(str(resume_path), optimizer=None)
    logger.info("Loaded model checkpoint via model.load_checkpoint: %s", resume)


def _build_train_loader(cfg: DictConfig, dataset, *, rank: int, world_size: int):
    worker_init_fn = set_global_seed(int(cfg.seed), get_worker_init_fn=True)
    sampler = ResumableEpochSampler(
        dataset=dataset,
        seed=int(cfg.seed),
        batch_size=int(cfg.batch_size),
        num_processes=world_size,
    )
    loader_kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": False,
        "sampler": sampler,
        "num_workers": int(cfg.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "timeout": float(cfg.get("dataloader_timeout", 0)),
        "collate_fn": interleaved_collate,
    }
    if int(cfg.num_workers) > 0:
        prefetch = cfg.get("dataloader_prefetch_factor", None)
        if prefetch is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch)
        loader_kwargs["persistent_workers"] = bool(cfg.get("dataloader_persistent_workers", False))
        mp_ctx = cfg.get("dataloader_multiprocessing_context", None)
        if mp_ctx not in (None, "null", ""):
            loader_kwargs["multiprocessing_context"] = str(mp_ctx)
    if rank == 0:
        logger.info(
            "FSDP DataLoader config: batch_size=%d num_workers=%d timeout=%.1fs",
            int(cfg.batch_size),
            int(cfg.num_workers),
            float(cfg.get("dataloader_timeout", 0)),
        )
    return DataLoader(dataset, **loader_kwargs), sampler


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    rank, local_rank, world_size = _init_distributed()
    setup_logging(log_level=logging.INFO, is_main_process=(rank == 0))

    try:
        ensure_dir(str(cfg.output_dir))
        if rank == 0:
            with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
                OmegaConf.save(cfg, f)

        train_ds, _ = build_datasets(cfg.data)
        model_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
        model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
        _load_resume_checkpoint_if_needed(model, cfg.get("resume"))
        _apply_dit_only_train_mode(model, freeze_visual_encoder=bool(cfg.get("freeze_visual_encoder", False)))
        fsdp_modules = _wrap_model_for_fsdp(model, local_rank=local_rank, mixed_precision=mixed_precision)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        tensor_count, param_count = _count_params(trainable_params)
        if rank == 0:
            logger.info(
                "Pure PyTorch FSDP: wrapped_modules=%d trainable_tensors=%d trainable_params=%.3fB world_size=%d",
                len(fsdp_modules),
                tensor_count,
                param_count / 1e9,
                world_size,
            )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
            betas=(0.9, 0.95),
        )
        train_loader, train_sampler = _build_train_loader(cfg, train_ds, rank=rank, world_size=world_size)
        total_train_steps = _estimate_total_train_steps(
            len(train_ds),
            batch_size=int(cfg.batch_size),
            world_size=world_size,
            grad_accum=int(cfg.gradient_accumulation_steps),
            num_epochs=int(cfg.num_epochs),
            max_steps=cfg.get("max_steps"),
        )
        warmup_steps = int(total_train_steps * 0.05)
        scheduler = _build_scheduler(
            optimizer,
            cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            learning_rate=float(cfg.learning_rate),
        )

        checkpoint_root = Path(str(cfg.output_dir)) / "checkpoints"
        weights_dir = checkpoint_root / "weights"
        ensure_dir(str(weights_dir))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        batch_in_epoch = 0
        epoch = 0
        run_start_time = time.perf_counter()
        data_iter = iter(train_loader)
        grad_accum = int(cfg.gradient_accumulation_steps)

        while global_step < total_train_steps:
            try:
                sample = next(data_iter)
                batch_in_epoch += 1
            except StopIteration:
                epoch += 1
                batch_in_epoch = 0
                train_sampler.clear_resume_batch_offset()
                data_iter = iter(train_loader)
                continue

            sync_gradients = (batch_in_epoch % grad_accum) == 0
            no_sync_ctx = contextlib.ExitStack()
            if not sync_gradients:
                for module in fsdp_modules:
                    no_sync_ctx.enter_context(module.no_sync())

            _profile_steps = int(os.environ.get("FASTWAM_PROFILE_STEPS", "0"))
            _do_profile = _profile_steps > 0 and global_step < _profile_steps and rank == 0 and torch.cuda.is_available()

            def _cuda_t():
                if _do_profile:
                    torch.cuda.synchronize()
                return time.perf_counter()

            with no_sync_ctx:
                _t0 = _cuda_t()
                with _autocast_context(mixed_precision):
                    loss, loss_dict = model.training_loss(sample)
                _t_fwd = _cuda_t()
                (loss / grad_accum).backward()
                _t_bwd = _cuda_t()

            if not sync_gradients:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.max_grad_norm))
            _t_clip = _cuda_t()
            optimizer.step()
            _t_opt = _cuda_t()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if _do_profile:
                stats = torch.cuda.memory_stats()
                logger.info(
                    "[fsdp-pure-profile] micro_step fwd=%.3fs bwd=%.3fs clip=%.3fs opt=%.3fs | mem alloc=%.2fG reserved=%.2fG max_alloc=%.2fG alloc_retries=%d",
                    _t_fwd - _t0,
                    _t_bwd - _t_fwd,
                    _t_clip - _t_bwd,
                    _t_opt - _t_clip,
                    torch.cuda.memory_allocated() / 1e9,
                    torch.cuda.memory_reserved() / 1e9,
                    torch.cuda.max_memory_allocated() / 1e9,
                    stats.get("num_alloc_retries", 0),
                )

            if int(cfg.log_every) > 0 and global_step % int(cfg.log_every) == 0 and rank == 0:
                elapsed = max(time.perf_counter() - run_start_time, 1e-6)
                steps_per_sec = global_step / elapsed
                eta_seconds = int(max(total_train_steps - global_step, 0) / max(steps_per_sec, 1e-9))
                eta_h, eta_rem = divmod(eta_seconds, 3600)
                eta_m, eta_s = divmod(eta_rem, 60)
                global_loss = _all_reduce_mean(loss)
                details = []
                for key, value in sorted(loss_dict.items()):
                    val = torch.tensor(float(value), device=loss.device, dtype=torch.float32)
                    details.append(f"{key}={_all_reduce_mean(val):.4f}")
                grad_norm_val = _all_reduce_mean(torch.as_tensor(float(grad_norm), device=loss.device))
                logger.info(
                    "[fsdp-pure] epoch=%d step=%d/%d loss=%.4f %s lr=%.2e grad_norm=%.4f speed=%.2f step/s, %.2f samples/s eta=%02d:%02d:%02d",
                    epoch,
                    global_step,
                    total_train_steps,
                    global_loss,
                    " ".join(details),
                    float(optimizer.param_groups[0]["lr"]),
                    grad_norm_val,
                    steps_per_sec,
                    steps_per_sec * int(cfg.batch_size) * world_size,
                    eta_h,
                    eta_m,
                    eta_s,
                )

            if int(cfg.save_every) > 0 and global_step % int(cfg.save_every) == 0:
                ckpt_path = weights_dir / f"step_{global_step:08d}_fsdp_pure.pt"
                _save_weights_only_checkpoint(model, str(ckpt_path), step=global_step)
                if dist.is_initialized():
                    dist.barrier()

        final_ckpt = weights_dir / f"step_{global_step:08d}_final_fsdp_pure.pt"
        _save_weights_only_checkpoint(model, str(final_ckpt), step=global_step)
        if dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
