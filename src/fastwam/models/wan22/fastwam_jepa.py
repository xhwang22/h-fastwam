"""FastWAM-JEPA: deterministic next-frame latent world model.

Variant of :class:`fastwam.models.wan22.fastwam.FastWAM` where the **video
expert** is the deterministic :class:`JEPAPredictor` (next-frame latent
prediction, L1 loss in V-JEPA 2-AC encoder space) instead of a flow-matching
DiT.  The **action expert** is unchanged (the same Wan ``ActionDiT`` with its
flow-matching scheduler) and reaches the predictor purely through shared MoT
self-attention.  Because the predictor (1024-dim, 16 heads×64, 24 layers) and
the action expert (1024-dim, 24 heads×128, 30 layers) have different attention
widths and layer counts, the :class:`MoT` runs in **non-strict tail-overlap**
mode: the action expert runs its first ``30-24=6`` layers solo, then both
experts share 24 mixed-attention layers with q/k/v/o projection adapters
bridging 1024↔3072.

Key differences vs FastWAM:

* No video noise / video flow-matching scheduler. The predictor sees clean,
  frozen encoder latents of the *context* frames and predicts the *next*
  frame's latents; the loss is L1 against the encoder latents of those targets.
* ``build_inputs`` splits encoder latents into context (``[:-1]``) and target
  (``[1:]``) along the temporal axis.
* The MoT attention mask wires action group ``i`` ↔ context frame ``i``.
"""

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .jepa_predictor import JEPAPredictor, load_ac_predictor_weights
from .visual_encoder import BaseVisualEncoder, build_visual_encoder
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAMJepa(torch.nn.Module):
    """MoT world model: JEPA next-frame predictor + flow-matching action expert."""

    def __init__(
        self,
        video_expert: JEPAPredictor,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_loss_detach_video_expert: bool = False,
        video_loss_type: str = "l1",
        visual_encoder=None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Trainer compatibility: optimizer/freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
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
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        # Only the action expert is a flow-matching denoiser now.
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.action_loss_detach_video_expert = bool(action_loss_detach_video_expert)
        vlt = str(video_loss_type).lower()
        if vlt not in {"l1", "mse"}:
            raise ValueError(f"`video_loss_type` must be 'l1' or 'mse', got {video_loss_type}")
        self.video_loss_type = vlt

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
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        skip_video_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_loss_detach_video_expert: bool = False,
        video_loss_type: str = "l1",
        visual_encoder_config: dict[str, Any] | None = None,
        ac_predictor_checkpoint: str | None = None,
        pretrain_checkpoint: str | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAMJepa.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAMJepa.")

        # --- Visual encoder (required: V-JEPA2-AC / DINO) ----------------- #
        if visual_encoder_config is None:
            raise ValueError(
                "FastWAMJepa requires a `visual_encoder` (V-JEPA2-AC); VAE-only mode is not supported."
            )
        ve_cfg = dict(visual_encoder_config)
        encoder_type = ve_cfg.pop("encoder_type", "vjepa2_ac")
        visual_encoder = build_visual_encoder(
            encoder_type=encoder_type,
            torch_dtype=torch_dtype,
            **ve_cfg,
        ).to(device=device)
        logger.info("FastWAMJepa using %s visual encoder.", encoder_type)

        # --- Load Wan components only for text encoder/tokenizer ---------- #
        # The video expert is a JEPAPredictor (not WanVideoDiT) and the VAE is
        # replaced by an external visual encoder, so we skip building the Wan DiT
        # and VAE entirely. `skip_dit_build=True` also bypasses _validate_dit_config,
        # so the reused `video_dit_config` block (which carries JEPAPredictor-only
        # keys like use_text_context and omits WanVideoDiT-required args) is fine.
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=None,
            skip_dit_build=True,  # video expert is the JEPAPredictor built below
            load_text_encoder=load_text_encoder,
            skip_vae_load=True,
        )

        # --- Build the JEPA predictor (video expert) ---------------------- #
        predictor_kwargs = cls._predictor_kwargs_from_video_dit_config(video_dit_config, mot_checkpoint_mixed_attn)
        video_expert = JEPAPredictor(**predictor_kwargs).to(device=device, dtype=torch_dtype)

        # Optionally seed predictor with native V-JEPA2-AC predictor weights.
        if ac_predictor_checkpoint:
            cls._maybe_load_ac_predictor(video_expert, ac_predictor_checkpoint, device)

        # --- Action expert (unchanged) ------------------------------------ #
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        # --- MoT in non-strict tail-overlap mode -------------------------- #
        # Predictor and action expert differ in attn width (1024 vs 3072) and
        # layer count (24 vs 30); non-strict + tail_overlap inserts q/k/v/o
        # projection adapters and runs the action expert's extra prefix layers
        # solo before the shared overlap suffix.
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            strict_expert_compat=False,
            layer_alignment_mode="tail_overlap",
            shared_attention_expert="video",
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            action_loss_detach_video_expert=action_loss_detach_video_expert,
            video_loss_type=video_loss_type,
            visual_encoder=visual_encoder,
        )

        if pretrain_checkpoint is not None:
            logger.info("Loading continue-pretrain checkpoint: %s", pretrain_checkpoint)
            ckpt = torch.load(pretrain_checkpoint, map_location=device)
            if "mot" in ckpt:
                missing, unexpected = model.mot.load_state_dict(ckpt["mot"], strict=False)
                logger.info("Loaded MoT (missing=%d, unexpected=%d).", len(missing), len(unexpected))
            if model.use_visual_encoder and "visual_encoder" in ckpt:
                model.visual_encoder.load_state_dict(ckpt["visual_encoder"], strict=False)

        model.model_paths = {
            "video_predictor": "JEPAPredictor",
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "visual_encoder": visual_encoder_config.get("model_name", encoder_type),
            "ac_predictor_checkpoint": ac_predictor_checkpoint,
        }
        return model

    @staticmethod
    def _predictor_kwargs_from_video_dit_config(video_dit_config: dict, grad_ckpt: bool) -> dict:
        """Translate the ``video_dit_config`` block into JEPAPredictor kwargs."""
        return {
            "hidden_dim": int(video_dit_config["hidden_dim"]),
            "in_dim": int(video_dit_config["in_dim"]),
            "out_dim": int(video_dit_config["out_dim"]),
            "ffn_dim": int(video_dit_config["ffn_dim"]),
            "text_dim": int(video_dit_config["text_dim"]),
            "eps": float(video_dit_config.get("eps", 1e-6)),
            "patch_size": tuple(video_dit_config.get("patch_size", (1, 2, 2))),
            "num_heads": int(video_dit_config["num_heads"]),
            "attn_head_dim": int(video_dit_config["attn_head_dim"]),
            "num_layers": int(video_dit_config["num_layers"]),
            "video_attention_mask_mode": str(video_dit_config.get("video_attention_mask_mode", "per_frame_causal")),
            "action_group_causal_mask_mode": str(video_dit_config.get("action_group_causal_mask_mode", "group_diagonal")),
            "use_gradient_checkpointing": bool(video_dit_config.get("use_gradient_checkpointing", grad_ckpt)),
            "use_text_context": bool(video_dit_config.get("use_text_context", True)),
        }

    @staticmethod
    def _maybe_load_ac_predictor(video_expert: JEPAPredictor, checkpoint: str, device: str) -> None:
        import os

        if not os.path.isfile(checkpoint):
            logger.warning(
                "ac_predictor_checkpoint=%s not found; predictor stays randomly initialised.",
                checkpoint,
            )
            return
        blob = torch.load(checkpoint, map_location="cpu")
        native_predictor = None
        if isinstance(blob, (tuple, list)) and len(blob) >= 2:
            native_predictor = blob[1]
        elif isinstance(blob, dict):
            if "predictor" in blob:
                native_predictor = blob["predictor"]
            elif "target_encoder" in blob and "predictor" in blob:
                native_predictor = blob["predictor"]
            else:
                # Possibly a raw predictor state_dict.
                native_predictor = blob
        elif isinstance(blob, nn.Module):
            native_predictor = blob
        if native_predictor is None:
            logger.warning("Could not locate native predictor inside %s; skipping.", checkpoint)
            return
        coverage = load_ac_predictor_weights(video_expert, native_predictor, strict=False)
        logger.info("AC predictor weight load coverage: %s", coverage)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        if self.vae is not None:
            self.vae.to(*args, **kwargs)
        if self.use_visual_encoder:
            self.visual_encoder.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------ #
    # Shared helpers (copied/adapted from FastWAM)
    # ------------------------------------------------------------------ #
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

    def _append_proprio_to_context(self, context, context_mask, proprio):
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _encode_video_latents(self, video_tensor, tiled=False):
        # Encoder backbone is frozen; encode under no_grad to save memory.
        with torch.no_grad():
            return self.visual_encoder.encode(video_tensor, device=self.device)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        video_tensor = input_image.to(device=self.device).unsqueeze(2)  # [1,3,1,H,W]
        return self.visual_encoder.encode(video_tensor, device=self.device)

    # ------------------------------------------------------------------ #
    # Input building
    # ------------------------------------------------------------------ #
    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("FastWAMJepa training requires `sample['context']` and `sample['context_mask']`.")
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAMJepa training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B, T, a_dim], got {tuple(action.shape)}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        # [B, D, T_lat, H_lat, W_lat] in frozen encoder space.
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        T_lat = input_latents.shape[2]
        if T_lat <= 1:
            raise ValueError(
                f"Encoder produced T_lat={T_lat}; need >= 2 latent frames for next-frame prediction."
            )

        action_horizon = int(action.shape[1])
        num_transitions = T_lat - 1
        if action_horizon % num_transitions != 0:
            raise ValueError(
                f"`action` horizon ({action_horizon}) must be divisible by latent transitions ({num_transitions})."
            )

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3 or proprio.shape[2] != self.proprio_dim:
                raise ValueError(f"`sample['proprio']` must be [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}")
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    # ------------------------------------------------------------------ #
    # MoT attention mask
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        num_context_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video (causal across context frames)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # video frame i -> action group i (so the predictor "sees" the action
        # that drives transition i through shared attention).
        mask[:video_seq_len, video_seq_len:] = self.video_expert.build_video_to_action_mask(
            num_frames=num_context_frames,
            tokens_per_frame=video_tokens_per_frame,
            action_seq_len=action_seq_len,
            device=device,
        )
        # action -> action (full)
        mask[video_seq_len:, video_seq_len:] = True
        # action -> all video context frames (action expert conditions on obs)
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    # ------------------------------------------------------------------ #
    # Loss helpers
    # ------------------------------------------------------------------ #
    def _video_loss_token(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-(sample, frame) latent loss. Inputs ``[B, D, T, H, W]`` → ``[B, T]``."""
        if self.video_loss_type == "l1":
            err = F.l1_loss(pred.float(), target.float(), reduction="none")
        else:
            err = F.mse_loss(pred.float(), target.float(), reduction="none")
        return err.mean(dim=(1, 3, 4))  # mean over channels + spatial → [B, T]

    def _video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_tok = self._video_loss_token(pred_video, target_video)  # [B, T_pred]
        if image_is_pad is None:
            return loss_tok.mean(dim=1)

        temporal_factor = int(self.visual_encoder.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"temporal_downsample_factor must be positive, got {temporal_factor}.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align image_is_pad with latent steps: "
                f"num_frames={image_is_pad.shape[1]}, factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        # Targets are latent frames [1:], i.e. exactly the strided tail.
        target_is_pad = latent_tail_is_pad
        if target_is_pad.shape[1] != loss_tok.shape[1]:
            raise ValueError(
                f"Video-loss mask mismatch: mask steps={target_is_pad.shape[1]}, loss steps={loss_tok.shape[1]}."
            )
        valid = (~target_is_pad).to(device=loss_tok.device, dtype=loss_tok.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (loss_tok * valid).sum(dim=1) / valid_sum

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]              # [B, D, T_lat, H, W]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # Split into context (predict from) and target (predict).
        context_latents = input_latents[:, :, :-1]           # [B, D, T_lat-1, H, W]
        target_latents = input_latents[:, :, 1:]             # [B, D, T_lat-1, H, W]
        num_context_frames = context_latents.shape[2]

        # ---- Action flow-matching inputs -------------------------------- #
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # ---- pre_dit for each expert ------------------------------------ #
        video_pre = self.video_expert.pre_dit(
            x=context_latents,
            context=context if self.video_expert.use_text_context else None,
            context_mask=context_mask if self.video_expert.use_text_context else None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            num_context_frames=num_context_frames,
            device=video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": (
                    {"context": video_pre["context"], "mask": video_pre["context_mask"]}
                    if video_pre["context"] is not None else None
                ),
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            detach_video_for_action=self.action_loss_detach_video_expert,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)  # [B, D, T_pred, H, W]
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # ---- Video (next-frame latent) loss ----------------------------- #
        loss_video_per_sample = self._video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_latents,
            image_is_pad=image_is_pad,
        )
        loss_video = loss_video_per_sample.mean()

        # ---- Action flow-matching loss ----------------------------------- #
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    # ------------------------------------------------------------------ #
    # Inference — action prediction (single MoT forward, no diffusion video)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _predict_action_noise(
        self,
        context_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_context_frames: int,
    ) -> torch.Tensor:
        video_pre = self.video_expert.pre_dit(
            x=context_latents,
            context=context if self.video_expert.use_text_context else None,
            context_mask=context_mask if self.video_expert.use_text_context else None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=tokens_per_frame,
            num_context_frames=num_context_frames,
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": (
                    {"context": video_pre["context"], "mask": video_pre["context_mask"]}
                    if video_pre["context"] is not None else None
                ),
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return self.action_expert.post_dit(tokens_out["action"], action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: Optional[int] = None,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif not (proprio.ndim == 2 and proprio.shape[0] == 1):
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        # Single observation → 1 context latent frame.
        context_latents = self._encode_input_image_latents_tensor(input_image=input_image)
        num_context_frames = context_latents.shape[2]

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            prompt = None
            use_prompt = False
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=self.device,
            dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise(
                context_latents=context_latents,
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                num_context_frames=num_context_frames,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.use_visual_encoder:
            payload["visual_encoder"] = self.visual_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location=self.device)
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `mot` key: {path}")
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if self.use_visual_encoder and "visual_encoder" in payload:
            self.visual_encoder.load_state_dict(payload["visual_encoder"], strict=False)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
