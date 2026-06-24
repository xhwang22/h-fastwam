"""Wan2.2 Continue-Pretrain model — video-only DiT training.

Supports two encoder modes:
    1. **VAE** (default) — uses Wan2.2's built-in WanVideoVAE38 for encoding *and*
       decoding.  This is the standard Wan2.2 pipeline.
    2. **DINO / V-JEPA2** — uses a frozen visual encoder backbone + trainable MLP
       projection.  Only encoding is available; decoding is not supported.

The model wraps a single ``WanVideoDiT`` (no action expert, no MoT) and trains
it with flow-matching on text-conditioned video generation using OpenVid-1M or
any similar text-video dataset.

Usage::

    model = Wan22Pretrain.from_wan22_pretrained(
        device="cuda",
        torch_dtype=torch.bfloat16,
        dit_config={...},
    )
    loss, loss_dict = model.training_loss(sample)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .helpers.loader import load_wan22_ti2v_5b_components
from .visual_encoder import BaseVisualEncoder, build_visual_encoder, sigreg_loss
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT

logger = get_logger(__name__)


class Wan22Pretrain(torch.nn.Module):
    """Video-only DiT for continue-pretraining Wan2.2 on text-video data.

    This is a simplified version of ``Wan22Core`` / ``FastWAM`` that does NOT
    include the action expert or MoT layer.  It trains a single WanVideoDiT
    with flow-matching loss, using either the original VAE encoder or a
    DINOv3/V-JEPA2 visual encoder.
    """

    def __init__(
        self,
        dit: WanVideoDiT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        visual_encoder=None,
        sigreg_lam_var: float = 1.0,
        sigreg_lam_cov: float = 1.0,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae

        # SIGReg hyperparameters (only used when visual_encoder is active)
        self.sigreg_lam_var = sigreg_lam_var
        self.sigreg_lam_cov = sigreg_lam_cov

        # Visual encoder: BaseVisualEncoder subclass or VAE (backward compat).
        self.use_visual_encoder = isinstance(visual_encoder, BaseVisualEncoder)
        if self.use_visual_encoder:
            self.visual_encoder = visual_encoder
        else:
            self.visual_encoder = vae

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)

        self.train_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=train_shift,
        )
        self.infer_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=infer_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        redirect_common_files: bool = True,
        dit_config: dict[str, Any] | None = None,
        skip_dit_load_from_pretrain: bool = False,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        visual_encoder_config: dict[str, Any] | None = None,
        sigreg_lam_var: float = 1.0,
        sigreg_lam_cov: float = 1.0,
    ):
        if dit_config is None:
            raise ValueError("`dit_config` is required.")
        if "text_dim" not in dit_config:
            raise ValueError("`dit_config['text_dim']` is required.")

        # --- Optional visual encoder (DINO / V-JEPA2) -------------------- #
        dino_visual_encoder = None
        if visual_encoder_config is not None:
            ve_cfg = dict(visual_encoder_config)
            encoder_type = ve_cfg.pop("encoder_type", "dino")
            dino_visual_encoder = build_visual_encoder(
                encoder_type=encoder_type,
                torch_dtype=torch_dtype,
                **ve_cfg,
            ).to(device=device)
            logger.info("Using %s visual encoder (VAE encoder bypassed).", encoder_type)

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            skip_vae_load=(dino_visual_encoder is not None),
        )

        model = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(dit_config["text_dim"]),
            device=device,
            torch_dtype=torch_dtype,
            train_shift=train_shift,
            infer_shift=infer_shift,
            num_train_timesteps=num_train_timesteps,
            visual_encoder=dino_visual_encoder,
            sigreg_lam_var=sigreg_lam_var,
            sigreg_lam_cov=sigreg_lam_cov,
        )
        model.model_paths = {
            "dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "visual_encoder": (
                visual_encoder_config.get("model_name", "dino") if visual_encoder_config else None
            ),
        }
        return model

    # ------------------------------------------------------------------ #
    # Module overrides
    # ------------------------------------------------------------------ #
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dit.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        if self.vae is not None:
            self.vae.to(*args, **kwargs)
        if self.use_visual_encoder:
            self.visual_encoder.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26), return_pre_standardise=False):
        if self.use_visual_encoder:
            return self.visual_encoder.encode(
                video_tensor, device=self.device,
                return_pre_standardise=return_pre_standardise,
            )
        else:
            with torch.no_grad():
                z = self.vae.encode(
                    video_tensor,
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            if return_pre_standardise:
                return z, None
            return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if self.use_visual_encoder:
            video_tensor = input_image.to(device=self.device).unsqueeze(2)
            z = self.visual_encoder.encode(video_tensor, device=self.device)
            return z
        else:
            image = input_image.to(device=self.device)[0].unsqueeze(1)
            z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if isinstance(z, list):
                z = z[0].unsqueeze(0)
            return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if self.use_visual_encoder:
            raise NotImplementedError(
                "Video decoding is not available in DINO encoder mode."
            )
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    # ------------------------------------------------------------------ #
    # Model forward (DiT only, no MoT)
    # ------------------------------------------------------------------ #
    def _model_fn(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents=False,
    ):
        return self.dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

    # ------------------------------------------------------------------ #
    # Build training inputs
    # ------------------------------------------------------------------ #
    def build_inputs(self, sample, tiled=False):
        """Build inputs for training.

        The sample dict must contain:
            - ``video``: Tensor ``[B, 3, T, H, W]`` in ``[-1, 1]``.
            - Either ``prompt`` (str or list[str]) **or** both ``context``
              and ``context_mask`` (precomputed text embeddings).
        """
        video = sample["video"]
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"`sample['video']` must be torch.Tensor, got {type(video)}")
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dim must be 3, got {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T%%4==1, got T={num_frames}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        encode_result = self._encode_video_latents(input_video, tiled=tiled, return_pre_standardise=self.use_visual_encoder)
        if self.use_visual_encoder:
            input_latents, latents_pre_std = encode_result
        else:
            input_latents = encode_result
            latents_pre_std = None

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.dit, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        # Context: precomputed or from text encoder
        has_context = "context" in sample and "context_mask" in sample
        has_prompt = "prompt" in sample

        if has_context:
            context = sample["context"]
            context_mask = sample["context_mask"]
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        elif has_prompt:
            prompt = sample["prompt"]
            if isinstance(prompt, str):
                prompt = [prompt]
            elif isinstance(prompt, (list, tuple)):
                prompt = list(prompt)
            else:
                raise TypeError(f"`sample['prompt']` must be str or list[str], got {type(prompt)}")
            if len(prompt) != batch_size:
                raise ValueError(f"Prompt batch mismatch: len={len(prompt)} vs video batch={batch_size}")
            context, context_mask = self.encode_prompt(prompt)
        else:
            raise ValueError("Sample must contain either 'prompt' or ('context' and 'context_mask').")

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "latents_pre_std": latents_pre_std,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
        }

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def training_loss(self, sample, tiled=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        latents_pre_std = inputs["latents_pre_std"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        noise = torch.randn_like(input_latents)
        timestep = self.train_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_scheduler.add_noise(input_latents, noise, timestep)
        target = self.train_scheduler.training_target(input_latents, noise, timestep)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        pred = self._model_fn(
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        if inputs["first_frame_latents"] is not None:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]

        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3, 4))
        sample_weight = self.train_scheduler.training_weight(timestep).to(
            loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        loss_total = (loss_per_sample * sample_weight).mean()
        loss_dict = {
            "loss_video": float(loss_total.detach().item()),
        }

        # SIGReg regularisation for visual encoder (DINO / V-JEPA2)
        if latents_pre_std is not None:
            loss_sigreg, sigreg_metrics = sigreg_loss(
                latents_pre_std.float(),
                lam_var=self.sigreg_lam_var,
                lam_cov=self.sigreg_lam_cov,
            )
            loss_total = loss_total + loss_sigreg
            loss_dict.update(sigreg_metrics)
            loss_dict["loss_sigreg"] = float(loss_sigreg.detach().item())

        return loss_total, loss_dict

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        num_frames: int = 17,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        height: int = 480,
        width: int = 832,
        # Accept and ignore action-related kwargs for trainer compatibility
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        action_cfg_scale: float = 1.0,
        proprio: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Generate video from text prompt (and optional first frame).

        Returns dict with ``"video"`` key containing list of PIL Images.
        """
        self.eval()

        if input_image is not None:
            if input_image.ndim == 3:
                input_image = input_image.unsqueeze(0)
            _, _, img_h, img_w = input_image.shape
            height = img_h
            width = img_w

        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_frames)

        latent_t = (checked_t - 1) // self.visual_encoder.temporal_downsample_factor + 1
        latent_h = checked_h // self.visual_encoder.upsampling_factor
        latent_w = checked_w // self.visual_encoder.upsampling_factor

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (1, self.visual_encoder.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        first_frame_latents = None
        fuse_flag = bool(getattr(self.dit, "fuse_vae_embedding_in_latents", False))
        if input_image is not None:
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            first_frame_latents = self._encode_input_image_latents_tensor(input_image, tiled=tiled)
            latents[:, :, 0:1] = first_frame_latents.clone()

        # Context
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context_posi, context_posi_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both be provided.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context_posi = context.to(device=self.device, dtype=self.torch_dtype)
            context_posi_mask = context_mask.to(device=self.device, dtype=torch.bool)

        context_nega = None
        context_nega_mask = None
        if text_cfg_scale != 1.0:
            context_nega, context_nega_mask = self.encode_prompt(
                "" if negative_prompt is None else negative_prompt
            )

        infer_timesteps, infer_deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )

        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep = step_t.unsqueeze(0).to(dtype=latents.dtype, device=self.device)
            noise_pred_posi = self._model_fn(
                latents=latents,
                timestep=timestep,
                context=context_posi,
                context_mask=context_posi_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            noise_pred = noise_pred_posi
            if context_nega is not None:
                noise_pred_nega = self._model_fn(
                    latents=latents,
                    timestep=timestep,
                    context=context_nega,
                    context_mask=context_nega_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                noise_pred = noise_pred + (text_cfg_scale - 1.0) * (noise_pred_posi - noise_pred_nega)

            latents = self.infer_scheduler.step(noise_pred, step_delta, latents)
            if first_frame_latents is not None:
                latents[:, :, 0:1] = first_frame_latents.clone()

        return {"video": self._decode_latents(latents, tiled=tiled)}

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.use_visual_encoder:
            payload["visual_encoder"] = self.visual_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location=self.device)
        if "dit" in payload:
            self.dit.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `dit` key: {path}")
        if self.use_visual_encoder and "visual_encoder" in payload:
            self.visual_encoder.load_state_dict(payload["visual_encoder"], strict=False)
            logger.info("Loaded `visual_encoder` from checkpoint.")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
