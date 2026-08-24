"""Hierarchical Fast World-Action Model (H-FastWAM).

Three-expert composition on top of :class:`fastwam.models.wan22.mot.MoT`::

    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Language  │  │   Video    │  │   Action   │
    │  Expert    │  │  Expert    │  │  Expert    │
    │ (random)   │  │ (Wan2.2)   │  │ (Fastwam)  │
    └─────┬──────┘  └──────┬─────┘  └─────┬──────┘
          └─── shared multi-modal self-attention ───┘
                    (structured mask)

Token ordering inside MoT: ``[language ‖ video ‖ action]``.

Unified Vision Encoder
----------------------
All visual input flows through a **single** encoder (DINO or VAE) into
the video expert. The language expert has **no** dedicated image encoder.
Instead, it obtains visual grounding by attending to the video expert's
**first-frame tokens** (clean, t=0) through the shared MoT self-attention.
This eliminates a redundant SigLIP encoder and ensures language conditions
on the same features that drive video generation.

Knowledge Insulation
--------------------
The ``language`` expert is trained only by its own teacher-forced CE loss
(𝓛_lang). To prevent the video / action flow-matching losses from
leaking into the language weights, the language K/V is **detached** when
video or action queries attend to it. The language expert's own Q
attending to its own K/V is *not* detached, so it still trains from
𝓛_lang. This is enabled via
``MoT.forward(..., detach_kv_experts={"language"})``.

Attention mask (row = query, col = key)
---------------------------------------
Within language block (task / subtask tokens):
  - task ↔ task:     bidirectional
  - subtask → task + prev subtask: causal over subtask

Cross-expert:
  - language → video first frame: **allowed** (visual grounding).
  - language → video rest / action: blocked.
  - video → language (all):   allowed (subtask conditioning).
  - video → video:            ``video_expert.build_video_to_video_mask``
  - video → action:           blocked.
  - action → language (all):  allowed.
  - action → video first frame only: allowed (anti-leakage).
  - action → action:          bidirectional.

Weight init
-----------
- Language expert: fully random. Phase ``language_video`` warm-start is
  recommended so the language expert reaches a reasonable subtask
  distribution before adding the action expert.
- Video expert: continues to load from Wan2.2-TI2V-5B + optional
  ``pretrain_checkpoint``.
- Action expert: continues to load from ``action_dit_pretrained_path``
  (unchanged from fastwam).

Training phases::

    language_video  →  full
    (lang + video)     (lang + video + action)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.jepa_predictor import JEPAPredictor
from fastwam.models.wan22.latent_action_dit import LatentActionDiT
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.visual_encoder import BaseVisualEncoder, build_visual_encoder
from fastwam.utils.pytorch_utils import optimizer_to

from .language_expert import CROSS_ENTROPY_IGNORE_INDEX, LanguageExpert
from .latent_action_decoder import LatentActionDecoder
from .qwen_language_expert import QwenLanguageExpert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FrozenTeacherHandle:
    """Keep the target encoder outside nn.Module registration/checkpointing."""

    encoder: BaseVisualEncoder
    source_checkpoint: Optional[str]


class HFastWAM(nn.Module):
    """Hierarchical Fast World-Action Model (3-expert MoT)."""

    CHECKPOINT_SCHEMA_VERSION = 2
    LATENT_ACTION_CONTRACT = {
        "latent_horizon": 8,
        "latent_dim": 32,
        "physical_action_horizon": 32,
        "physical_action_dim": 14,
        "actions_per_latent": 4,
    }
    ACTION_HEAD_CHECKPOINT_PREFIXES = (
        "mixtures.action.action_encoder.",
        "mixtures.action.head.",
    )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        language_expert: LanguageExpert,
        video_expert: nn.Module,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        tokenizer=None,
        language_tokenizer=None,
        language_backend: str = "legacy",
        language_pad_to_max_length: bool = False,
        text_dim: int = 4096,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        # Schedulers
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_train_sampling_distribution: str = "shifted_uniform",
        video_train_logit_mean: float = 0.0,
        video_train_logit_std: float = 1.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        action_train_sampling_distribution: str = "shifted_uniform",
        action_train_logit_mean: float = 0.0,
        action_train_logit_std: float = 1.0,
        # Loss weights
        loss_lambda_language: float = 1.0,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_latent_action_decoder: float = 1.0,
        latent_action_decoder: Optional[LatentActionDecoder] = None,
        latent_action_config: Optional[dict] = None,
        latent_action_decoder_config: Optional[dict] = None,
        # Training phase & gradient policy
        training_phase: str = "full",
        knowledge_insulation: bool = True,
        action_loss_detach_video_expert: bool = False,
        strict_expert_compat: bool = True,
        freeze_language_expert: bool = False,
        freeze_video_expert: bool = False,
        freeze_action_expert: bool = False,
        # Optional DINO/VAE override for video expert
        # JEPA predictor vs flow-matching video expert
        video_loss_type: str = "flow_matching",
        visual_encoder=None,
        fixed_target_encoder: bool = False,
    ):
        super().__init__()
        self.strict_expert_compat = bool(strict_expert_compat)
        if self.strict_expert_compat:
            self._validate_expert_shapes(language_expert, video_expert, action_expert)
        self._validate_mot_membership(mot, language_expert, video_expert, action_expert)

        self.language_expert = language_expert
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.vae = vae
        self.tokenizer = tokenizer
        self.language_tokenizer = language_tokenizer
        self.language_backend = str(language_backend)
        self.language_pad_to_max_length = bool(language_pad_to_max_length)
        self.text_dim = int(text_dim)
        self.torch_dtype = torch_dtype

        # Video-expert visual encoder (DINO / V-JEPA2) or VAE fallback
        self.use_visual_encoder = isinstance(visual_encoder, BaseVisualEncoder)
        self.visual_encoder = visual_encoder if self.use_visual_encoder else vae
        self.fixed_target_encoder_enabled = bool(fixed_target_encoder)
        self.__dict__["_fixed_teacher_handle"] = None

        # Proprio → video/action context via a learned token
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.dit = self.mot  # trainer/optimizer compat

        # JEPA predictor detection — auto-detect from expert type.
        self.is_jepa_predictor = isinstance(video_expert, JEPAPredictor)
        # Accept explicit override; if "auto" or not set, derive from expert type.
        _video_loss_type = str(video_loss_type)
        if _video_loss_type == "auto":
            _video_loss_type = "l1" if self.is_jepa_predictor else "flow_matching"
        if self.is_jepa_predictor and _video_loss_type == "flow_matching":
            logger.warning(
                "video_loss_type='flow_matching' requested but video_expert is JEPAPredictor; "
                "overriding to 'l1'."
            )
            _video_loss_type = "l1"
        self.video_loss_type = _video_loss_type

        # Schedulers
        # The video scheduler is only used for flow-matching (WAN DiT) training.
        # When the video expert is a JEPAPredictor we skip creating it to save memory
        # and avoid spurious timestep sampling.
        if not self.is_jepa_predictor:
            self.train_video_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=video_num_train_timesteps,
                shift=video_train_shift,
                sampling_distribution=video_train_sampling_distribution,
                logit_mean=video_train_logit_mean,
                logit_std=video_train_logit_std,
            )
            self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift,
            )
        else:
            self.train_video_scheduler = None
            self.infer_video_scheduler = None
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
            sampling_distribution=action_train_sampling_distribution,
            logit_mean=action_train_logit_mean,
            logit_std=action_train_logit_std,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift,
        )

        self.loss_lambda_language = float(loss_lambda_language)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_latent_action_decoder = float(loss_lambda_latent_action_decoder)
        self.latent_action_decoder = latent_action_decoder
        self.latent_action_config = dict(latent_action_config or {})
        self.latent_action_decoder_config = dict(latent_action_decoder_config or {})
        self.latent_action_enabled = self.latent_action_decoder is not None
        self._training_epoch = 0
        self._training_phase = training_phase
        self.knowledge_insulation = bool(knowledge_insulation)
        self.action_loss_detach_video_expert = bool(action_loss_detach_video_expert)
        self.freeze_language_expert = bool(freeze_language_expert)
        self.freeze_video_expert = bool(freeze_video_expert)
        self.freeze_action_expert = bool(freeze_action_expert)

        if self.freeze_language_expert:
            self.language_expert.requires_grad_(False)
        if self.freeze_video_expert:
            self.video_expert.requires_grad_(False)
        if self.freeze_action_expert:
            self.action_expert.requires_grad_(False)

        self.device_str = device
        self.to(device=device, dtype=torch_dtype)

        logger.info(
            "HFastWAM: phase=%s, KI=%s, action_loss_detach_video_expert=%s, strict=%s, freeze(L/V/A)=(%s/%s/%s), "
            "λ_lang=%.2f λ_vid=%.2f λ_act=%.2f, experts=%s, "
            "is_jepa=%s, video_loss=%s",
            training_phase,
            self.knowledge_insulation,
            self.action_loss_detach_video_expert,
            self.strict_expert_compat,
            self.freeze_language_expert,
            self.freeze_video_expert,
            self.freeze_action_expert,
            self.loss_lambda_language,
            self.loss_lambda_video,
            self.loss_lambda_action,
            list(self.mot.expert_order),
            self.is_jepa_predictor,
            self.video_loss_type,
        )
        logger.info(
            "Training timestep sampling: video=%s shift=%.4g logit=(%.4g, %.4g), "
            "action=%s shift=%.4g logit=(%.4g, %.4g)",
            "none" if self.train_video_scheduler is None else self.train_video_scheduler.sampling_distribution,
            video_train_shift,
            video_train_logit_mean,
            video_train_logit_std,
            self.train_action_scheduler.sampling_distribution,
            action_train_shift,
            action_train_logit_mean,
            action_train_logit_std,
        )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @classmethod
    def _validate_latent_action_configs(
        cls,
        latent_action_config: Optional[dict],
        decoder_config: Optional[dict],
        action_dit_config: dict,
        proprio_dim: Optional[int],
        visual_dim: int,
    ) -> tuple[dict, dict]:
        if isinstance(latent_action_config, DictConfig):
            latent_action_config = OmegaConf.to_container(
                latent_action_config, resolve=True
            )
        if isinstance(decoder_config, DictConfig):
            decoder_config = OmegaConf.to_container(decoder_config, resolve=True)
        config = dict(latent_action_config or {})
        if not bool(config.get("enabled", False)):
            return {}, {}
        for key, expected in cls.LATENT_ACTION_CONTRACT.items():
            actual = config.get(key)
            if actual is None or int(actual) != expected:
                raise ValueError(
                    f"Latent-action contract requires `{key}={expected}`, got {actual!r}."
                )
        probabilities = config.get("oracle_probabilities")
        if probabilities is None or list(map(float, probabilities)) != [1.0, 0.75, 0.5, 0.25, 0.0]:
            raise ValueError(
                "Latent-action oracle probabilities must be [1, 0.75, 0.5, 0.25, 0]."
            )
        if str(config.get("decoder_loss_type", "smooth_l1")) != "smooth_l1":
            raise ValueError("Latent-action decoder_loss_type must be 'smooth_l1'.")
        beta = float(config.get("decoder_loss_beta", 1.0))
        if beta <= 0:
            raise ValueError("Latent-action decoder_loss_beta must be positive.")
        config["decoder_loss_beta"] = beta
        config["oracle_probabilities"] = list(map(float, probabilities))
        if int(action_dit_config.get("action_dim", -1)) != 32:
            raise ValueError("Latent-action ActionDiT must use action_dim=32.")

        if not isinstance(decoder_config, dict):
            raise ValueError("Latent-action mode requires `latent_action_decoder_config` as a dict.")
        decoder = dict(decoder_config)
        expected_decoder = {
            "latent_dim": 32,
            "num_latents": 8,
            "substeps_per_latent": 4,
            "action_dim": 14,
            "visual_dim": int(visual_dim),
        }
        if proprio_dim is None:
            raise ValueError("Latent-action decoding requires a configured proprio_dim.")
        expected_decoder["proprio_dim"] = int(proprio_dim)
        for key, expected in expected_decoder.items():
            actual = decoder.get(key)
            if actual is None or int(actual) != expected:
                raise ValueError(
                    f"Latent-action decoder requires `{key}={expected}`, got {actual!r}."
                )
        return config, decoder

    @staticmethod
    def _validate_expert_shapes(lang, video, action):
        # All three experts MUST share attn-space shape or MoT can't concat.
        if int(lang.num_heads) != int(video.num_heads) or int(action.num_heads) != int(video.num_heads):
            raise ValueError(
                f"num_heads mismatch: lang={lang.num_heads}, vid={video.num_heads}, act={action.num_heads}."
            )
        if int(lang.attn_head_dim) != int(video.attn_head_dim) or int(action.attn_head_dim) != int(video.attn_head_dim):
            raise ValueError(
                f"attn_head_dim mismatch: lang={lang.attn_head_dim}, "
                f"vid={video.attn_head_dim}, act={action.attn_head_dim}."
            )
        if int(len(lang.blocks)) != int(len(video.blocks)) or int(len(action.blocks)) != int(len(video.blocks)):
            raise ValueError(
                f"num_layers mismatch: lang={len(lang.blocks)}, "
                f"vid={len(video.blocks)}, act={len(action.blocks)}."
            )

    @staticmethod
    def _validate_mot_membership(mot, lang, video, action):
        if set(mot.expert_order) != {"language", "video", "action"}:
            raise ValueError(
                f"H-FastWAM expects MoT with experts {{language, video, action}}, got {mot.expert_order}."
            )
        if mot.mixtures["language"] is not lang:
            raise ValueError("MoT['language'] must be the same module as language_expert.")
        if mot.mixtures["video"] is not video:
            raise ValueError("MoT['video'] must be the same module as video_expert.")
        if mot.mixtures["action"] is not action:
            raise ValueError("MoT['action'] must be the same module as action_expert.")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def training_phase(self) -> str:
        return self._training_phase

    @training_phase.setter
    def training_phase(self, phase: str):
        valid = {"language_video", "full"}
        if phase not in valid:
            raise ValueError(f"Invalid training phase: {phase}. Must be one of {valid}")
        self._training_phase = phase
        logger.info("Training phase set to: %s", phase)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def fixed_target_encoder(self) -> Optional[BaseVisualEncoder]:
        handle = self.__dict__.get("_fixed_teacher_handle")
        return None if handle is None else handle.encoder

    def set_training_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"`epoch` must be non-negative, got {epoch}.")
        self._training_epoch = epoch

    def _latent_action_oracle_probability(self, epoch: Optional[int] = None) -> float:
        if not self.latent_action_enabled:
            return 0.0
        if self.latent_action_decoder is None or not self.latent_action_decoder.training:
            return 0.0
        probabilities = self.latent_action_config["oracle_probabilities"]
        index = min(self._training_epoch if epoch is None else int(epoch), len(probabilities) - 1)
        if index < 0:
            raise ValueError(f"`epoch` must be non-negative, got {index}.")
        return float(probabilities[index])

    @staticmethod
    def _estimate_clean_latent(
        noisy_latent: torch.Tensor,
        predicted_velocity: torch.Tensor,
        timestep: torch.Tensor,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        if noisy_latent.shape != predicted_velocity.shape:
            raise ValueError(
                "Noisy latent and predicted velocity shapes must match: "
                f"{tuple(noisy_latent.shape)} vs {tuple(predicted_velocity.shape)}."
            )
        sigma = timestep.to(device=noisy_latent.device, dtype=noisy_latent.dtype)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if sigma.ndim != 1 or sigma.shape[0] != noisy_latent.shape[0]:
            raise ValueError(
                f"`timestep` must be scalar or [B], got {tuple(timestep.shape)} for B={noisy_latent.shape[0]}."
            )
        sigma = sigma / float(num_train_timesteps)
        sigma = sigma.view(-1, *([1] * (noisy_latent.ndim - 1)))
        return noisy_latent - sigma * predicted_velocity

    @staticmethod
    def _select_decoder_latent(
        oracle_latent: torch.Tensor,
        generated_latent: torch.Tensor,
        oracle_probability: float,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if oracle_latent.shape != generated_latent.shape:
            raise ValueError(
                "Oracle/generated latent shapes must match: "
                f"{tuple(oracle_latent.shape)} vs {tuple(generated_latent.shape)}."
            )
        probability = float(oracle_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"`oracle_probability` must be in [0,1], got {probability}.")
        oracle_mask = torch.rand(
            (oracle_latent.shape[0],),
            device=oracle_latent.device,
            generator=generator,
        ) < probability
        selected = torch.where(
            oracle_mask.view(-1, *([1] * (oracle_latent.ndim - 1))),
            oracle_latent.detach(),
            generated_latent.detach(),
        )
        return selected, oracle_mask

    @staticmethod
    def _compute_latent_action_decoder_loss(
        predicted_action: torch.Tensor,
        physical_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        beta: float,
    ) -> torch.Tensor:
        if predicted_action.shape != physical_action.shape:
            raise ValueError(
                "Decoded/physical action shapes must match: "
                f"{tuple(predicted_action.shape)} vs {tuple(physical_action.shape)}."
            )
        element_loss = F.smooth_l1_loss(
            predicted_action.float(),
            physical_action.float(),
            reduction="none",
            beta=float(beta),
        )
        if action_is_pad is None:
            return element_loss.mean()
        action_is_pad = action_is_pad.to(device=element_loss.device, dtype=torch.bool)
        if action_is_pad.shape != element_loss.shape[:2]:
            raise ValueError(
                "Physical `action_is_pad` must match [B,T]: "
                f"got {tuple(action_is_pad.shape)}, expected {tuple(element_loss.shape[:2])}."
            )
        valid = (~action_is_pad).unsqueeze(-1).expand_as(element_loss)
        denominator = valid.sum().clamp_min(1)
        return (element_loss * valid).sum() / denominator

    def _initialize_fixed_target_encoder(self, source_checkpoint: Optional[str]) -> None:
        if not self.fixed_target_encoder_enabled:
            return
        if not self.is_jepa_predictor or not self.use_visual_encoder:
            raise ValueError("fixed_target_encoder requires a JEPA predictor with a visual encoder.")
        teacher = copy.deepcopy(self.visual_encoder)
        teacher.eval()
        teacher.requires_grad_(False)
        if hasattr(teacher, "_freeze_backbone"):
            teacher._freeze_backbone = True
        if hasattr(teacher, "use_activation_checkpointing"):
            teacher.use_activation_checkpointing = False
        backbone = getattr(teacher, "backbone", None)
        if backbone is not None and hasattr(backbone, "use_activation_checkpointing"):
            backbone.use_activation_checkpointing = False
        self.__dict__["_fixed_teacher_handle"] = _FrozenTeacherHandle(
            encoder=teacher,
            source_checkpoint=source_checkpoint,
        )
        logger.info("Initialized frozen target encoder from %s.", source_checkpoint or "online initialization")

    # ------------------------------------------------------------------ #
    # Encoders
    # ------------------------------------------------------------------ #
    def _encode_video_latents(self, video: torch.Tensor, tiled: bool = False):
        """Encode [B, 3, T, H, W] → video-expert latents."""
        if self.use_visual_encoder:
            return self._encode_video_with_visual_encoder(video, self.visual_encoder)
        with torch.no_grad():
            return self.vae.encode(video, device=self.device, tiled=tiled)

    def _encode_video_with_visual_encoder(
        self,
        video: torch.Tensor,
        encoder: BaseVisualEncoder,
        *,
        state_indices: Optional[list[int]] = None,
    ) -> torch.Tensor:
        causal_tubelets = bool(getattr(encoder, "causal_tubelet_encoding", False))
        causal_prefixes = bool(getattr(encoder, "causal_prefix_encoding", False))
        if causal_tubelets and causal_prefixes:
            raise ValueError(
                "causal_tubelet_encoding and causal_prefix_encoding are mutually exclusive."
            )
        if causal_prefixes:
            return self._encode_causal_visual_prefixes(
                video, encoder=encoder, state_indices=state_indices,
            )
        if causal_tubelets:
            return self._encode_causal_visual_states(
                video, encoder=encoder, state_indices=state_indices,
            )
        if state_indices is not None:
            raise ValueError("state_indices are only supported for causal visual encoding.")
        return encoder.encode(video, device=self.device)

    def _encode_causal_visual_prefixes(
        self,
        video: torch.Tensor,
        *,
        encoder: Optional[BaseVisualEncoder] = None,
        state_indices: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Encode prefixes ending at each latent-state anchor without future frames."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")

        encoder = self.visual_encoder if encoder is None else encoder
        temporal_patch = int(getattr(encoder, "_temporal_patch", 1))
        temporal_stride = int(getattr(encoder, "temporal_downsample_factor", 1))
        if temporal_patch < 1 or temporal_stride < 1:
            raise ValueError("Visual encoder temporal patch/downsample factors must be positive.")

        _, _, num_frames, _, _ = video.shape
        if num_frames < 1:
            raise ValueError("video must contain at least one frame.")
        if (num_frames - 1) % temporal_stride != 0:
            raise ValueError(
                "Causal prefix anchors must include the final frame: "
                f"num_frames={num_frames}, temporal_stride={temporal_stride}."
            )

        indices = (
            list(range(0, num_frames, temporal_stride))
            if state_indices is None
            else list(state_indices)
        )
        if not indices or any(index < 0 or index >= num_frames for index in indices):
            raise ValueError(f"Invalid causal prefix state indices {indices} for {num_frames} frames.")

        states = []
        for frame_index in indices:
            prefix = video[:, :, : frame_index + 1]
            pad_frames = (-prefix.shape[2]) % temporal_patch
            if pad_frames:
                first = prefix[:, :, 0:1].expand(-1, -1, pad_frames, -1, -1)
                prefix = torch.cat([first, prefix], dim=2)

            prefix_latents = encoder.encode(prefix, device=self.device)
            if prefix_latents.ndim != 5 or prefix_latents.shape[2] < 1:
                raise ValueError(
                    "Causal prefix encoding must produce [B,D,T,H,W] with T>=1, "
                    f"got {tuple(prefix_latents.shape)}."
                )
            states.append(prefix_latents[:, :, -1:])

        return torch.cat(states, dim=2)

    def _encode_causal_visual_states(
        self,
        video: torch.Tensor,
        *,
        encoder: Optional[BaseVisualEncoder] = None,
        state_indices: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Encode selected states from only the frames available at each state."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")

        encoder = self.visual_encoder if encoder is None else encoder
        temporal_patch = int(getattr(encoder, "_temporal_patch", 1))
        temporal_stride = int(getattr(encoder, "temporal_downsample_factor", 1))
        if temporal_patch < 1 or temporal_stride < 1:
            raise ValueError("Visual encoder temporal patch/downsample factors must be positive.")

        batch_size, channels, num_frames, height, width = video.shape
        if num_frames < 1:
            raise ValueError("video must contain at least one frame.")
        indices = (
            list(range(0, num_frames, temporal_stride))
            if state_indices is None
            else list(state_indices)
        )
        if not indices or any(index < 0 or index >= num_frames for index in indices):
            raise ValueError(f"Invalid causal tubelet state indices {indices} for {num_frames} frames.")
        clips = []
        for frame_index in indices:
            start = frame_index - temporal_patch + 1
            if start < 0:
                padding = video[:, :, 0:1].expand(
                    -1, -1, -start, -1, -1,
                )
                clip = torch.cat([padding, video[:, :, : frame_index + 1]], dim=2)
            else:
                clip = video[:, :, start : frame_index + 1]
            if clip.shape[2] != temporal_patch:
                raise ValueError(
                    f"Failed to build a {temporal_patch}-frame causal tubelet at "
                    f"frame {frame_index}: got {clip.shape[2]} frames."
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
        flat_latents = encoder.encode(flat_clips, device=self.device)
        if flat_latents.shape[2] != 1:
            raise ValueError(
                "Causal tubelet encoding must produce one latent state per clip, "
                f"got T_lat={flat_latents.shape[2]}."
            )

        latent_dim, latent_h, latent_w = (
            flat_latents.shape[1],
            flat_latents.shape[3],
            flat_latents.shape[4],
        )
        return flat_latents.reshape(
            batch_size,
            num_states,
            latent_dim,
            latent_h,
            latent_w,
        ).permute(0, 2, 1, 3, 4).contiguous()

    def _align_first_conditioning_latent(
        self,
        video: torch.Tensor,
        latents: torch.Tensor,
        *,
        encoder: Optional[BaseVisualEncoder] = None,
    ) -> torch.Tensor:
        """Match training's clean first latent to single-frame inference."""
        encoder = self.visual_encoder if encoder is None else encoder
        if (
            not isinstance(encoder, BaseVisualEncoder)
            or bool(getattr(encoder, "causal_tubelet_encoding", False))
            or bool(getattr(encoder, "causal_prefix_encoding", False))
            or not bool(getattr(encoder, "requires_independent_first_frame", False))
        ):
            return latents

        first_latent = self._encode_first_frame(video[:, :, 0], encoder=encoder)
        if first_latent.shape[1:] != latents[:, :, 0:1].shape[1:]:
            raise ValueError(
                "Single-frame and full-video visual latents have incompatible shapes: "
                f"{tuple(first_latent.shape)} vs {tuple(latents[:, :, 0:1].shape)}."
            )
        aligned = latents.clone()
        aligned[:, :, 0:1] = first_latent.to(device=latents.device, dtype=latents.dtype)
        return aligned

    def _encode_first_frame(
        self,
        image: torch.Tensor,
        tiled: bool = False,
        *,
        encoder: Optional[BaseVisualEncoder] = None,
    ) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B, 3, H, W], got {tuple(image.shape)}")
        video = image.to(device=self.device, dtype=self.torch_dtype).unsqueeze(2)
        if self.use_visual_encoder:
            encoder = self.visual_encoder if encoder is None else encoder
            return encoder.encode(video, device=self.device)
        with torch.no_grad():
            return self.vae.encode(video, device=self.device, tiled=tiled)

    def _decode_latents(self, latents: torch.Tensor, tiled: bool = False) -> list[Image.Image]:
        if self.use_visual_encoder:
            raise NotImplementedError(
                "Video decoding is not available in visual-encoder mode. "
                "Use VAE mode (`model.visual_encoder_config=null`) for rollout video eval."
            )
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _prepare_training_video_latents(
        self,
        *,
        video: Optional[torch.Tensor],
        cached_latents: Optional[torch.Tensor],
        tiled: bool = False,
        source: str,
    ) -> torch.Tensor:
        if cached_latents is not None:
            if not torch.is_tensor(cached_latents) or cached_latents.ndim != 5:
                raise ValueError(
                    f"{source} cached video latents must be [B,D,T,H,W], got "
                    f"{type(cached_latents)} with shape {getattr(cached_latents, 'shape', None)}"
                )
            expected_dim = int(
                self.visual_encoder.z_dim if self.use_visual_encoder else self.vae.model.z_dim
            )
            if int(cached_latents.shape[1]) != expected_dim:
                raise ValueError(
                    f"{source} cached latent channel mismatch: expected {expected_dim}, "
                    f"got {cached_latents.shape[1]}."
                )
            return cached_latents.detach().to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )

        if video is None:
            raise ValueError(f"{source} requires either raw video or cached video latents.")
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"{source} video must be [B,3,T,H,W], got {tuple(video.shape)}")
        latents = self._encode_video_latents(video, tiled=tiled)
        return self._align_first_conditioning_latent(video, latents)

    def _prepare_jepa_context_target_latents(
        self,
        *,
        video: Optional[torch.Tensor],
        cached_latents: Optional[torch.Tensor],
        tiled: bool = False,
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        teacher = self.fixed_target_encoder
        if teacher is None:
            latents = self._prepare_training_video_latents(
                video=video,
                cached_latents=cached_latents,
                tiled=tiled,
                source=source,
            )
            if latents.shape[2] < 2:
                raise ValueError(
                    f"JEPA predictor {source} requires at least 2 temporal latent states; "
                    f"got {latents.shape[2]}."
                )
            return latents[:, :, :-1], latents[:, :, 1:]

        if cached_latents is not None:
            raise ValueError(
                f"{source} cannot use cached video latents with fixed_target_encoder; "
                "raw video is required for separate online/teacher encoding."
            )
        if video is None:
            raise ValueError(f"{source} requires raw video with fixed_target_encoder.")
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"{source} video must be [B,3,T,H,W], got {tuple(video.shape)}")

        online = self.visual_encoder
        causal = bool(
            getattr(online, "causal_tubelet_encoding", False)
            or getattr(online, "causal_prefix_encoding", False)
        )
        if causal:
            temporal_stride = int(getattr(online, "temporal_downsample_factor", 1))
            anchors = list(range(0, int(video.shape[2]), temporal_stride))
            if len(anchors) < 2:
                raise ValueError(
                    f"JEPA predictor {source} requires at least 2 causal anchors; got {anchors}."
                )
            with torch.no_grad():
                target = self._encode_video_with_visual_encoder(
                    video,
                    teacher,
                    state_indices=anchors[1:],
                ).detach()
            context = self._encode_video_with_visual_encoder(
                video,
                online,
                state_indices=anchors[:-1],
            )
        else:
            with torch.no_grad():
                teacher_latents = self._encode_video_with_visual_encoder(video, teacher)
                teacher_latents = self._align_first_conditioning_latent(
                    video, teacher_latents, encoder=teacher,
                )
                target = teacher_latents[:, :, 1:].detach()
            online_latents = self._encode_video_with_visual_encoder(video, online)
            online_latents = self._align_first_conditioning_latent(
                video, online_latents, encoder=online,
            )
            context = online_latents[:, :, :-1]

        if context.shape != target.shape:
            raise ValueError(
                f"Online context and teacher target shapes differ: {tuple(context.shape)} "
                f"vs {tuple(target.shape)}."
            )
        return context, target

    # ------------------------------------------------------------------ #
    # Cross-attention context for video/action pre_dit
    # ------------------------------------------------------------------ #
    def _make_dummy_text_context(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single zero text token + empty mask.

        ``video_expert.pre_dit`` and ``action_expert.pre_dit`` both require
        ``context``/``context_mask`` arguments because Wan2.2 originally
        used a text encoder. Under the 3-expert MoT design, language
        conditioning happens via shared **self-attention** inside MoT, not
        via cross-attention. When no proprio token is available, we pass a
        1-token dummy here and then feed
        ``context_all["video"]=None`` / ``context_all["action"]=None``
        into :meth:`MoT.forward` so that
        :func:`MoT._apply_expert_post_block` skips cross-attention.
        """
        dummy_ctx = torch.zeros(
            (batch_size, 1, self.text_dim),
            dtype=self.torch_dtype, device=self.device,
        )
        dummy_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=self.device,
        )
        return dummy_ctx, dummy_mask

    def _select_initial_proprio(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
        *,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Select the initial proprio state as [B, D], matching FastWAM."""
        if self.proprio_encoder is None or proprio is None:
            return None
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        if proprio.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"`{source}` is 1D but batch_size={batch_size}; expected [B,D].")
            selected = proprio.unsqueeze(0)
        elif proprio.ndim == 2:
            if proprio.shape[0] == batch_size:
                selected = proprio
            elif batch_size == 1:
                selected = proprio[:1]
            else:
                raise ValueError(
                    f"`{source}` must have leading batch dim {batch_size}, got shape {tuple(proprio.shape)}"
                )
        elif proprio.ndim == 3:
            if proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`{source}` must be [B,T,D] with B={batch_size}, got shape {tuple(proprio.shape)}"
                )
            selected = proprio[:, 0, :]
        else:
            raise ValueError(f"`{source}` must be [D], [B,D], or [B,T,D], got shape {tuple(proprio.shape)}")

        if self.proprio_dim is None or selected.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` last dim must be proprio_dim={self.proprio_dim}, got {selected.shape[-1]}"
            )
        return selected.to(device=self.device, dtype=self.torch_dtype)

    def _make_proprio_text_context(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
        *,
        source: str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        initial_proprio = self._select_initial_proprio(proprio, batch_size, source=source)
        if initial_proprio is None:
            return None, None
        proprio_token = self.proprio_encoder(initial_proprio).unsqueeze(1).to(dtype=self.torch_dtype)
        proprio_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)
        return proprio_token, proprio_mask

    def _select_interleaved_initial_proprio(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
        num_segments: int,
        *,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Select segment initial proprio as [B, N, D]."""
        if self.proprio_encoder is None or proprio is None:
            return None
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        if proprio.ndim == 4:
            if proprio.shape[:2] != (batch_size, num_segments):
                raise ValueError(
                    f"`{source}` must be [B,N,T,D] with leading dims {(batch_size, num_segments)}, "
                    f"got {tuple(proprio.shape)}"
                )
            selected = proprio[:, :, 0, :]
        elif proprio.ndim == 3:
            if proprio.shape[:2] == (batch_size, num_segments):
                selected = proprio
            elif proprio.shape[0] == batch_size * num_segments:
                selected = proprio[:, 0, :].reshape(batch_size, num_segments, proprio.shape[-1])
            elif batch_size == 1 and proprio.shape[0] == num_segments:
                selected = proprio[:, 0, :].unsqueeze(0)
            else:
                raise ValueError(
                    f"`{source}` must be [B,N,D], [B,N,T,D], [B*N,T,D], or [N,T,D], "
                    f"got {tuple(proprio.shape)}"
                )
        elif proprio.ndim == 2:
            if proprio.shape[0] == batch_size * num_segments:
                selected = proprio.reshape(batch_size, num_segments, proprio.shape[-1])
            elif batch_size == 1 and proprio.shape[0] == num_segments:
                selected = proprio.unsqueeze(0)
            else:
                raise ValueError(
                    f"`{source}` must be [B*N,D] or [N,D] for interleaved input, got {tuple(proprio.shape)}"
                )
        else:
            raise ValueError(
                f"`{source}` must be [B,N,T,D], [B,N,D], [B*N,T,D], [B*N,D], [N,T,D], or [N,D], "
                f"got {tuple(proprio.shape)}"
            )

        if self.proprio_dim is None or selected.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` last dim must be proprio_dim={self.proprio_dim}, got {selected.shape[-1]}"
            )
        return selected.to(device=self.device, dtype=self.torch_dtype)

    def _make_interleaved_proprio_text_context(
        self,
        proprio: Optional[torch.Tensor],
        batch_size: int,
        num_segments: int,
        *,
        source: str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        initial_proprio = self._select_interleaved_initial_proprio(
            proprio,
            batch_size=batch_size,
            num_segments=num_segments,
            source=source,
        )
        if initial_proprio is None:
            return None, None
        flat_proprio = initial_proprio.reshape(batch_size * num_segments, self.proprio_dim)
        proprio_token = self.proprio_encoder(flat_proprio).unsqueeze(1).to(dtype=self.torch_dtype)
        proprio_mask = torch.ones((batch_size * num_segments, 1), dtype=torch.bool, device=self.device)
        return proprio_token, proprio_mask

    @staticmethod
    def _context_payload_from_pre_state(pre_state: dict, enabled: bool) -> Optional[dict]:
        if not enabled:
            return None
        return {
            "context": pre_state["context"],
            "mask": pre_state["context_mask"],
        }

    @staticmethod
    def _merge_segment_context_payload(
        pre_state: dict,
        batch_size: int,
        num_segments: int,
        segment_mask: Optional[torch.Tensor],
        enabled: bool,
    ) -> Optional[dict]:
        if not enabled:
            return None
        context = pre_state["context"]
        if context.shape[0] != batch_size * num_segments:
            raise ValueError(
                "Segment context batch mismatch: "
                f"got {context.shape[0]}, expected {batch_size * num_segments}"
            )
        query_len = int(pre_state["tokens"].shape[1])
        context_len = int(context.shape[1])
        context = context.reshape(batch_size, num_segments, context_len, context.shape[-1]).flatten(1, 2)

        key_valid = torch.ones((batch_size, num_segments, context_len), dtype=torch.bool, device=context.device)
        if segment_mask is not None:
            key_valid &= segment_mask.to(device=context.device, dtype=torch.bool).unsqueeze(-1)

        mask = torch.zeros(
            (batch_size, num_segments * query_len, num_segments * context_len),
            dtype=torch.bool,
            device=context.device,
        )
        for segment_idx in range(num_segments):
            q_start = segment_idx * query_len
            q_end = q_start + query_len
            k_end = (segment_idx + 1) * context_len
            mask[:, q_start:q_end, :k_end] = key_valid[:, :segment_idx + 1, :].flatten(1).unsqueeze(1)

        return {"context": context, "mask": mask}

    def _tokenize_prompt_batch(
        self,
        prompt: str | list[str] | tuple[str, ...],
        max_length: int,
    ) -> torch.Tensor:
        tokenizer = self.language_tokenizer if self.language_tokenizer is not None else self.tokenizer
        if tokenizer is None:
            raise ValueError(
                "No tokenizer is available for prompt -> language token conversion. "
                "Please pass `task_token_ids` and `subtask_token_ids` explicitly."
            )

        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, (list, tuple)):
            prompts = list(prompt)
        else:
            raise TypeError(f"`prompt` must be str/list[str]/tuple[str], got {type(prompt)}")
        if not prompts or any((not isinstance(p, str) or p.strip() == "") for p in prompts):
            raise ValueError("`prompt` must contain non-empty strings when language tokens are absent.")

        ids = None

        if self.language_backend == "qwen3":
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=(
                    "max_length"
                    if self.language_pad_to_max_length
                    else True
                ),
            )
            ids = encoded["input_ids"]
        else:
            rows = []
            for item in prompts:
                try:
                    cur_ids, _ = tokenizer(item, return_mask=True, add_special_tokens=True)
                except TypeError:
                    encoded = tokenizer(
                        item,
                        return_tensors="pt",
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_length,
                    )
                    cur_ids = encoded["input_ids"]
                if not torch.is_tensor(cur_ids):
                    cur_ids = torch.as_tensor(cur_ids, dtype=torch.long)
                if cur_ids.ndim == 2:
                    if cur_ids.shape[0] != 1:
                        raise ValueError(f"Tokenized prompt must have batch size 1, got {tuple(cur_ids.shape)}")
                    cur_ids = cur_ids[0]
                elif cur_ids.ndim != 1:
                    raise ValueError(f"Tokenized prompt must be 1D/2D, got {tuple(cur_ids.shape)}")
                rows.append(cur_ids[:max_length])
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", 0)
            padded = torch.full(
                (len(rows), max(int(row.numel()) for row in rows)),
                int(pad_id),
                dtype=torch.long,
            )
            for i, row in enumerate(rows):
                padded[i, : row.numel()] = row.to(dtype=torch.long)
            ids = padded

        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        elif ids.ndim != 2:
            raise ValueError(f"Tokenized task ids must be 2D [B,L], got shape={tuple(ids.shape)}")

        if ids.shape[1] > max_length:
            ids = ids[:, :max_length]

        return ids.to(device=self.device, dtype=torch.long)

    def _tokenize_task_prompt(self, prompt: str) -> torch.Tensor:
        if not isinstance(prompt, str) or prompt.strip() == "":
            raise ValueError("`prompt` must be a non-empty string when `task_token_ids` is absent.")
        max_task_len = int(getattr(self.language_expert, "max_task_len", 128))
        return self._tokenize_prompt_batch(prompt, max_length=max_task_len)

    def _ensure_language_tokens_from_prompt(self, sample: dict) -> dict:
        if "task_token_ids" in sample:
            if "subtask_token_ids" not in sample:
                sample = dict(sample)
                task_ids = sample["task_token_ids"]
                sample["subtask_token_ids"] = torch.empty(
                    (task_ids.shape[0], 0),
                    device=task_ids.device,
                    dtype=torch.long,
                )
            return sample
        prompt = sample.get("prompt", None)
        if prompt is None:
            return sample

        vision = sample.get("video", sample.get("video_latents", None))
        batch_size = int(vision.shape[0]) if torch.is_tensor(vision) and vision.ndim >= 1 else 1
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, (list, tuple)):
            prompts = list(prompt)
        else:
            raise TypeError(f"`sample['prompt']` must be str/list[str]/tuple[str], got {type(prompt)}")
        if len(prompts) != batch_size:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompts)} vs video batch={batch_size}")

        max_task_len = int(getattr(self.language_expert, "max_task_len", 128))
        task_ids = self._tokenize_prompt_batch(prompts, max_length=max_task_len)

        if not getattr(self, "_warned_prompt_language_fallback", False):
            logger.warning(
                "HFastWAM sample has no task/subtask token ids; using `prompt` as task "
                "tokens and applying causal language loss on the task stream. Provide "
                "explicit `subtask_token_ids` to supervise a separate subtask target."
            )
            self._warned_prompt_language_fallback = True

        sample = dict(sample)
        sample["task_token_ids"] = task_ids
        sample["subtask_token_ids"] = torch.empty(
            (task_ids.shape[0], 0),
            device=task_ids.device,
            dtype=torch.long,
        )
        return sample

    # ------------------------------------------------------------------ #
    # Structured attention mask
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_full_attention_mask(
        self,
        task_len: int,
        subtask_len: int,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the full ``[S_lang + S_vid + S_act]²`` mask.

        See module docstring for the rule set.
        """
        S_lang = task_len + subtask_len
        S_vid = int(video_seq_len)
        S_act = int(action_seq_len)
        S = S_lang + S_vid + S_act
        mask = torch.zeros((S, S), dtype=torch.bool, device=device)

        # Block ranges
        l_s, l_e = 0, S_lang
        v_s, v_e = S_lang, S_lang + S_vid
        a_s, a_e = S_lang + S_vid, S

        # ---- Language rows -------------------------------------------- #
        # Language self-attention (task + subtask)
        mask[l_s:l_e, l_s:l_e] = LanguageExpert.build_language_rows(
            task_len=task_len, subtask_len=subtask_len, device=device,
        )
        # Subtask language rows can use the first video frame for grounding.
        # Task/prompt rows stay text-only causal LM and must not see video.
        first_frame_tokens = min(int(video_tokens_per_frame), S_vid)
        subtask_start = l_s + int(task_len)
        mask[subtask_start:l_e, v_s:v_s + first_frame_tokens] = True
        # Language → action: blocked (default False).

        # ---- Video rows ----------------------------------------------- #
        # video → language (all) — subtask is the condition
        mask[v_s:v_e, l_s:l_e] = True
        # video → video (per video_expert rule)
        mask[v_s:v_e, v_s:v_e] = self.video_expert.build_video_to_video_mask(
            video_seq_len=S_vid,
            video_tokens_per_frame=int(video_tokens_per_frame),
            device=device,
        )
        # video → action: blocked for all video experts (both WAN flow-matching and
        # JEPAPredictor). Keeps the mask policy identical to DINO/VAE experiments so
        # that video-expert comparisons are not confounded by action conditioning.

        # ---- Action rows ---------------------------------------------- #
        # action → language (all)
        mask[a_s:a_e, l_s:l_e] = True
        # action → first video frame only (anti-leakage)
        mask[a_s:a_e, v_s:v_s + first_frame_tokens] = True
        # action → action: bidirectional
        mask[a_s:a_e, a_s:a_e] = True

        return mask

    @staticmethod
    def _repeat_freqs_for_segments(freqs: torch.Tensor, num_segments: int) -> torch.Tensor:
        if num_segments <= 1:
            return freqs
        return torch.cat([freqs] * int(num_segments), dim=0)

    @staticmethod
    def _merge_segment_tokens(tokens: torch.Tensor, batch_size: int, num_segments: int) -> torch.Tensor:
        if tokens.shape[0] != batch_size * num_segments:
            raise ValueError(
                "Segment token batch mismatch: "
                f"got {tokens.shape[0]}, expected {batch_size * num_segments}"
            )
        return tokens.reshape(batch_size, num_segments, *tokens.shape[1:]).flatten(1, 2)

    @staticmethod
    def _merge_segment_t_mod(
        t_mod: torch.Tensor,
        batch_size: int,
        num_segments: int,
        tokens_per_segment: int,
    ) -> torch.Tensor:
        if t_mod.ndim == 4:
            if t_mod.shape[0] != batch_size * num_segments:
                raise ValueError(
                    "Per-token t_mod batch mismatch: "
                    f"got {t_mod.shape[0]}, expected {batch_size * num_segments}"
                )
            return t_mod.reshape(batch_size, num_segments, *t_mod.shape[1:]).flatten(1, 2)

        if t_mod.ndim == 3 and t_mod.shape[0] == batch_size * num_segments:
            return (
                t_mod.reshape(batch_size, num_segments, 1, *t_mod.shape[1:])
                .expand(batch_size, num_segments, tokens_per_segment, *t_mod.shape[1:])
                .reshape(batch_size, num_segments * tokens_per_segment, *t_mod.shape[1:])
            )

        return t_mod

    def _merge_segment_pre_state(
        self,
        pre_state: dict,
        batch_size: int,
        num_segments: int,
    ) -> dict:
        tokens_per_segment = int(pre_state["tokens"].shape[1])
        merged = dict(pre_state)
        merged["tokens"] = self._merge_segment_tokens(
            pre_state["tokens"], batch_size=batch_size, num_segments=num_segments,
        )
        merged["freqs"] = self._repeat_freqs_for_segments(pre_state["freqs"], num_segments)
        merged["t_mod"] = self._merge_segment_t_mod(
            pre_state["t_mod"],
            batch_size=batch_size,
            num_segments=num_segments,
            tokens_per_segment=tokens_per_segment,
        )
        return merged

    def _prepare_jepa_training_video_pre(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[dict, dict, int]:
        """Prepare JEPA tokens for MoT and retain the prediction branch metadata."""
        del target_latents
        prediction_pre = self.video_expert.pre_dit(
            x=context_latents,
            context=context if self.video_expert.use_text_context else None,
            context_mask=context_mask if self.video_expert.use_text_context else None,
        )
        prediction_tokens = int(prediction_pre["tokens"].shape[1])
        return prediction_pre, prediction_pre, prediction_tokens

    @staticmethod
    def _unmerge_segment_tokens(
        tokens: torch.Tensor,
        batch_size: int,
        num_segments: int,
        tokens_per_segment: int,
    ) -> torch.Tensor:
        expected = num_segments * tokens_per_segment
        if int(tokens.shape[1]) != expected:
            raise ValueError(
                "Merged segment token length mismatch: "
                f"got {tokens.shape[1]}, expected {expected}"
            )
        return tokens.reshape(batch_size, num_segments, tokens_per_segment, tokens.shape[-1]).flatten(0, 1)

    @staticmethod
    def _zero_padded_action_dims(
        action: torch.Tensor,
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_dim_is_pad is None:
            return action
        mask = action_dim_is_pad.to(device=action.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (action.shape[0], action.shape[-1]):
            raise ValueError(
                "`action_dim_is_pad` shape mismatch: "
                f"got {tuple(mask.shape)}, expected {(action.shape[0], action.shape[-1])}."
            )
        return action.masked_fill(mask.unsqueeze(1), 0.0)

    @staticmethod
    def _flatten_segment_prompts(prompts: Any, batch_size: int, num_segments: int) -> list[str]:
        if isinstance(prompts, str):
            if batch_size != 1 or num_segments != 1:
                raise ValueError("A string prompt is only valid for a single [B=1,N=1] segment batch.")
            return [prompts]
        if isinstance(prompts, (list, tuple)):
            # Default PyTorch collation transposes per-sample prompt lists from
            # [B][N] to [N][B]. Accept both layouts and flatten in [B, N] order.
            if len(prompts) == num_segments and all(isinstance(row, (list, tuple)) for row in prompts):
                if any(len(row) != batch_size for row in prompts):
                    raise ValueError("Transposed segment prompts must be nested as [N][B] strings.")
                flat = [prompts[n][b] for b in range(batch_size) for n in range(num_segments)]
                if any((not isinstance(item, str) or item.strip() == "") for item in flat):
                    raise ValueError("Segment prompts must contain non-empty strings.")
                return flat

            if len(prompts) != batch_size:
                raise ValueError(
                    f"Prompt batch mismatch: got outer len {len(prompts)}, "
                    f"expected batch_size={batch_size} or num_segments={num_segments}"
                )
            flat: list[str] = []
            for row in prompts:
                if isinstance(row, str):
                    if num_segments != 1:
                        raise ValueError("Flat prompt lists are only valid when num_segments=1.")
                    flat.append(row)
                    continue
                if not isinstance(row, (list, tuple)) or len(row) != num_segments:
                    raise ValueError("Segment prompts must be nested as [B][N] strings.")
                flat.extend(row)
            if any((not isinstance(item, str) or item.strip() == "") for item in flat):
                raise ValueError("Segment prompts must contain non-empty strings.")
            return flat
        raise TypeError(f"Unsupported segment prompt container: {type(prompts)}")

    @staticmethod
    def _extract_segment_prompt_row(prompts: Any, batch_idx: int, valid_idx: torch.Tensor, batch_size: int, num_segments: int):
        if prompts is None:
            return None
        idxs = [int(i) for i in valid_idx.tolist()]
        if isinstance(prompts, str):
            if batch_size != 1 or num_segments != 1:
                raise ValueError("A string prompt is only valid for a single [B=1,N=1] segment batch.")
            return [[prompts]]
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        if not isinstance(prompts, list):
            raise TypeError(f"Unsupported segment prompt container: {type(prompts)}")

        if len(prompts) == batch_size:
            row = prompts[batch_idx]
            if isinstance(row, str):
                row = [row]
            if not isinstance(row, (list, tuple)) or len(row) != num_segments:
                raise ValueError("Segment prompts must be nested as [B][N] strings.")
            return [[row[i] for i in idxs]]

        # Default PyTorch collation can transpose nested strings into [N][B].
        if len(prompts) == num_segments and all(isinstance(row, (list, tuple)) for row in prompts):
            if any(len(row) != batch_size for row in prompts):
                raise ValueError("Transposed segment prompts must be nested as [N][B] strings.")
            return [[prompts[i][batch_idx] for i in idxs]]

        if batch_size == 1 and len(prompts) == num_segments and all(isinstance(item, str) for item in prompts):
            return [[prompts[i] for i in idxs]]

        raise ValueError(
            f"Prompt batch mismatch: got outer len {len(prompts)}, "
            f"expected batch_size={batch_size} or num_segments={num_segments}"
        )

    @staticmethod
    def _segment_list_to_dict(segments: list[dict]) -> dict:
        if not segments:
            raise ValueError("Interleaved input contains an empty segment list.")
        keys = segments[0].keys()
        out = {}
        for key in keys:
            values = [segment[key] for segment in segments]
            first = values[0]
            if torch.is_tensor(first):
                out[key] = torch.stack(values, dim=0)
            elif isinstance(first, str):
                out[key] = list(values)
            else:
                out[key] = values
        return out

    def _build_interleaved_attention_mask(
        self,
        task_len: int,
        subtask_len: int,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        num_segments: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a causal multi-segment mask over physical MoT order.

        MoT still stores tokens grouped by expert as
        ``[all_language | all_video | all_action]``. This mask gives that
        layout the semantics of temporal segments: segment ``i`` can attend
        to all tokens from segments ``< i`` and only the allowed tokens inside
        segment ``i``.
        """
        S_lang_seg = int(task_len) + int(subtask_len)
        S_vid_seg = int(video_seq_len)
        S_act_seg = int(action_seq_len)
        N = int(num_segments)

        lang_total = N * S_lang_seg
        video_total = N * S_vid_seg
        action_total = N * S_act_seg
        total = lang_total + video_total + action_total
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        def lang_range(seg_idx: int) -> tuple[int, int]:
            start = seg_idx * S_lang_seg
            return start, start + S_lang_seg

        def video_range(seg_idx: int) -> tuple[int, int]:
            start = lang_total + seg_idx * S_vid_seg
            return start, start + S_vid_seg

        def action_range(seg_idx: int) -> tuple[int, int]:
            start = lang_total + video_total + seg_idx * S_act_seg
            return start, start + S_act_seg

        first_frame_tokens = min(int(video_tokens_per_frame), S_vid_seg)
        for seg_idx in range(N):
            current_ranges = []
            if S_lang_seg > 0:
                current_ranges.append(lang_range(seg_idx))
            if S_vid_seg > 0:
                current_ranges.append(video_range(seg_idx))
            if S_act_seg > 0:
                current_ranges.append(action_range(seg_idx))

            previous_ranges = []
            for prev_idx in range(seg_idx):
                if S_lang_seg > 0:
                    previous_ranges.append(lang_range(prev_idx))
                if S_vid_seg > 0:
                    previous_ranges.append(video_range(prev_idx))
                if S_act_seg > 0:
                    previous_ranges.append(action_range(prev_idx))

            for row_s, row_e in current_ranges:
                for col_s, col_e in previous_ranges:
                    mask[row_s:row_e, col_s:col_e] = True

            if S_lang_seg > 0:
                l_s, l_e = lang_range(seg_idx)
                mask[l_s:l_e, l_s:l_e] = LanguageExpert.build_language_rows(
                    task_len=task_len, subtask_len=subtask_len, device=device,
                )
                if S_vid_seg > 0:
                    v_s, _ = video_range(seg_idx)
                    subtask_start = l_s + int(task_len)
                    mask[subtask_start:l_e, v_s:v_s + first_frame_tokens] = True

            if S_vid_seg > 0:
                v_s, v_e = video_range(seg_idx)
                if S_lang_seg > 0:
                    l_s, l_e = lang_range(seg_idx)
                    mask[v_s:v_e, l_s:l_e] = True
                mask[v_s:v_e, v_s:v_e] = self.video_expert.build_video_to_video_mask(
                    video_seq_len=S_vid_seg,
                    video_tokens_per_frame=int(video_tokens_per_frame),
                    device=device,
                )
                # video → action: blocked (same policy as WAN DiT / DINO / VAE experiments).

            if S_act_seg > 0:
                a_s, a_e = action_range(seg_idx)
                if S_lang_seg > 0:
                    l_s, l_e = lang_range(seg_idx)
                    mask[a_s:a_e, l_s:l_e] = True
                if S_vid_seg > 0:
                    v_s, _ = video_range(seg_idx)
                    mask[a_s:a_e, v_s:v_s + first_frame_tokens] = True
                mask[a_s:a_e, a_s:a_e] = True

        return mask

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def _resolve_training_route(self, sample: dict) -> tuple[str, dict[str, bool]]:
        has_task_tokens = "task_token_ids" in sample
        has_subtask_tokens = "subtask_token_ids" in sample
        if has_subtask_tokens and not has_task_tokens:
            raise ValueError(
                "`subtask_token_ids` requires `task_token_ids`. "
                "For task-only conditioning, provide `task_token_ids` without `subtask_token_ids`."
            )

        has_language = False
        if has_task_tokens:
            task_ids = sample["task_token_ids"]
            if task_ids.ndim != 2:
                raise ValueError(
                    f"`task_token_ids` must be 2D [B, L], got {tuple(task_ids.shape)}"
                )
            if has_subtask_tokens:
                subtask_ids = sample["subtask_token_ids"]
                if subtask_ids.ndim != 2:
                    raise ValueError(
                        f"`subtask_token_ids` must be 2D [B, L], got {tuple(subtask_ids.shape)}"
                    )
                if task_ids.shape[0] != subtask_ids.shape[0]:
                    raise ValueError(
                        "Batch mismatch between task/subtask token ids: "
                        f"{task_ids.shape[0]} vs {subtask_ids.shape[0]}"
                    )
            has_language = task_ids.shape[1] > 0 or (
                has_subtask_tokens and sample["subtask_token_ids"].shape[1] > 0
            )

        has_valid_action = False
        if "action" in sample:
            action = sample["action"]
            if action.ndim != 3:
                raise ValueError(f"sample['action'] must be [B,T,a_dim], got {tuple(action.shape)}")
            if action.shape[1] > 0:
                action_is_pad = sample.get("action_is_pad", None)
                if action_is_pad is None:
                    has_valid_action = True
                elif action_is_pad.ndim == 1:
                    has_valid_action = bool((~action_is_pad.to(torch.bool)).any().item())
                elif action_is_pad.ndim == 2:
                    if action_is_pad.shape[0] != action.shape[0] or action_is_pad.shape[1] != action.shape[1]:
                        raise ValueError(
                            "`action_is_pad` shape mismatch: "
                            f"got {tuple(action_is_pad.shape)} vs expected {tuple(action.shape[:2])}"
                        )
                    per_sample_valid = (~action_is_pad.to(torch.bool)).any(dim=1)
                    if bool(per_sample_valid.any().item()) and bool((~per_sample_valid).any().item()):
                        raise ValueError(
                            "Mixed action validity in one batch is not supported. "
                            "Please batch sequences with the same route together."
                        )
                    has_valid_action = bool(per_sample_valid.any().item())
                else:
                    raise ValueError(
                        f"`action_is_pad` must be 1D/2D boolean mask, got {tuple(action_is_pad.shape)}"
                    )

        if not has_language:
            if not has_valid_action:
                raise ValueError(
                    "Route `video_action` requires valid action tokens when language tokens are absent."
                )
            route = "video_action"
        elif has_valid_action:
            route = "full"
        else:
            route = "language_video"

        modality_mask = {
            "language": bool(has_language),
            "video": True,
            "action": bool(has_valid_action),
        }
        return route, modality_mask

    def _validate_sample(self, sample: dict) -> tuple[str, dict[str, bool]]:
        if self.latent_action_enabled:
            missing = [
                key for key in ("action", "latent_action", "latent_action_is_pad", "proprio")
                if sample.get(key) is None
            ]
            if missing:
                raise ValueError(
                    f"Latent-action training sample is missing required keys: {missing}."
                )
        if "video" not in sample and "video_latents" not in sample:
            raise ValueError(
                "H-FastWAM training needs sample['video'] or sample['video_latents']."
            )
        route, modality_mask = self._resolve_training_route(sample)
        if modality_mask["language"]:
            if "task_token_ids" not in sample:
                raise ValueError(f"Route '{route}' needs sample['task_token_ids'].")
        if modality_mask["action"] and "action" not in sample:
            raise ValueError(f"Route '{route}' needs sample['action'].")
        return route, modality_mask

    def _run_mot_interleaved_segments(
        self,
        lang_pre: Optional[dict],
        video_pre: dict,
        action_pre: Optional[dict],
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
        num_segments: int,
        video_context_payload: Optional[dict] = None,
        action_context_payload: Optional[dict] = None,
    ) -> dict:
        video_seq_len = int(video_pre["tokens"].shape[1]) // int(num_segments)
        action_seq_len = 0 if action_pre is None else int(action_pre["tokens"].shape[1]) // int(num_segments)
        attention_mask = self._build_interleaved_attention_mask(
            task_len=task_len,
            subtask_len=subtask_len,
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_segments=num_segments,
            device=video_pre["tokens"].device,
        )

        if lang_pre is not None and action_pre is not None:
            detach_set = set()
            if self.knowledge_insulation:
                detach_set.add("language")
            if self.action_loss_detach_video_expert:
                detach_set.add("video")
            return self.mot(
                embeds_all={
                    "language": lang_pre["tokens"],
                    "video": video_pre["tokens"],
                    "action": action_pre["tokens"],
                },
                attention_mask=attention_mask,
                freqs_all={
                    "language": lang_pre["freqs"],
                    "video": video_pre["freqs"],
                    "action": action_pre["freqs"],
                },
                context_all={
                    "language": None,
                    "video": video_context_payload,
                    "action": action_context_payload,
                },
                t_mod_all={
                    "language": lang_pre["t_mod"],
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                },
                detach_kv_experts=detach_set or None,
            )

        if lang_pre is not None:
            detach_set = {"language"} if self.knowledge_insulation else None
            return self.mot(
                embeds_all={"language": lang_pre["tokens"], "video": video_pre["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"language": lang_pre["freqs"], "video": video_pre["freqs"]},
                context_all={"language": None, "video": video_context_payload},
                t_mod_all={"language": lang_pre["t_mod"], "video": video_pre["t_mod"]},
                detach_kv_experts=detach_set,
                active_expert_order=("language", "video"),
            )

        if action_pre is None:
            raise ValueError("Interleaved training without language requires action tokens.")

        # Align with FastWAM's `action_loss_detach_video_expert=True` (see
        # `_run_mot_two_experts_va`): action attends to detached video K/V.
        return self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": video_context_payload, "action": action_context_payload},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            detach_kv_experts={"video"} if self.action_loss_detach_video_expert else None,
            active_expert_order=("video", "action"),
        )

    def _training_loss_interleaved_segments(
        self,
        sample: dict,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        segments = sample["segments"] if "segments" in sample else sample
        if isinstance(segments, list):
            segments = self._segment_list_to_dict(segments)
        if not isinstance(segments, dict):
            raise TypeError("`sample['segments']` must be a dict of tensors/lists.")

        video = segments.get("video")
        cached_latents = segments.get("video_latents")
        vision_ref = video if video is not None else cached_latents
        if vision_ref is None:
            raise ValueError(
                "Interleaved H-FastWAM input requires `segments['video']` or "
                "`segments['video_latents']`."
            )
        if vision_ref.ndim == 5:
            num_segments = int(vision_ref.shape[0])
            batched_segments = {}
            for key, value in segments.items():
                if key == "segment_mask":
                    continue
                if torch.is_tensor(value):
                    if value.ndim >= 1 and int(value.shape[0]) == num_segments:
                        batched_segments[key] = value.unsqueeze(0)
                    else:
                        batched_segments[key] = value
                elif key == "prompt":
                    if isinstance(value, str):
                        batched_segments[key] = [[value]]
                    elif isinstance(value, (list, tuple)):
                        batched_segments[key] = [list(value)]
                    else:
                        raise TypeError(f"Unsupported segment prompt container: {type(value)}")
                else:
                    batched_segments[key] = [list(value)] if isinstance(value, (list, tuple)) else value
            segment_mask = segments.get("segment_mask", sample.get("segment_mask", None))
            if segment_mask is None:
                segment_mask = torch.ones(
                    (1, num_segments), dtype=torch.bool, device=vision_ref.device
                )
            else:
                segment_mask = segment_mask.to(device=vision_ref.device, dtype=torch.bool)
                if segment_mask.ndim == 1:
                    segment_mask = segment_mask.unsqueeze(0)
            batched_segments["segment_mask"] = segment_mask
            return self._training_loss_interleaved_segments({"segments": batched_segments}, tiled=tiled)
        if vision_ref.ndim != 6:
            raise ValueError(
                "Interleaved vision input must be [N,D,T,H,W] or [B,N,D,T,H,W], "
                f"got {tuple(vision_ref.shape)}"
            )
        B, N = int(vision_ref.shape[0]), int(vision_ref.shape[1])
        if video is not None and cached_latents is not None:
            if tuple(video.shape[:2]) != tuple(cached_latents.shape[:2]):
                raise ValueError(
                    "Interleaved raw/cached vision batch mismatch: "
                    f"video={tuple(video.shape[:2])}, latents={tuple(cached_latents.shape[:2])}."
                )
        segment_mask = segments.get("segment_mask", sample.get("segment_mask", None))
        if segment_mask is not None:
            segment_mask = segment_mask.to(device=vision_ref.device, dtype=torch.bool)
            if segment_mask.ndim == 1 and B == 1:
                segment_mask = segment_mask.unsqueeze(0)
            if segment_mask.shape != (B, N):
                raise ValueError(f"`segment_mask` must be [B,N]={B,N}, got {tuple(segment_mask.shape)}")
            if not bool(segment_mask.any().item()):
                raise ValueError("`segment_mask` must mark at least one valid segment.")
            if not bool(segment_mask.all().item()):
                total_losses = []
                loss_sums: dict[str, float] = {}
                loss_counts: dict[str, int] = {}
                for batch_idx in range(B):
                    valid_idx = segment_mask[batch_idx].nonzero(as_tuple=False).flatten()
                    if valid_idx.numel() == 0:
                        continue
                    single_segments = {
                        "segment_mask": torch.ones(
                            (1, int(valid_idx.numel())),
                            dtype=torch.bool,
                            device=vision_ref.device,
                        )
                    }
                    for key, value in segments.items():
                        if key == "segment_mask":
                            continue
                        if torch.is_tensor(value):
                            if value.ndim >= 2 and value.shape[0] == B and value.shape[1] == N:
                                single_segments[key] = value[batch_idx:batch_idx + 1, valid_idx]
                            else:
                                single_segments[key] = value
                        elif key == "prompt":
                            single_segments[key] = self._extract_segment_prompt_row(
                                value,
                                batch_idx=batch_idx,
                                valid_idx=valid_idx,
                                batch_size=B,
                                num_segments=N,
                            )
                        else:
                            single_segments[key] = value

                    loss_i, dict_i = self._training_loss_interleaved_segments(
                        {"segments": single_segments},
                        tiled=tiled,
                    )
                    total_losses.append(loss_i)
                    for key, value in dict_i.items():
                        loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
                        loss_counts[key] = loss_counts.get(key, 0) + 1

                if not total_losses:
                    raise ValueError("No valid segments remain after applying `segment_mask`.")
                loss_dict = {
                    key: loss_sums[key] / max(loss_counts[key], 1)
                    for key in loss_sums
                }
                return torch.stack(total_losses).mean(), loss_dict
        flat_video = None
        if video is not None:
            if video.ndim != 6:
                raise ValueError(
                    f"Interleaved raw video must be [B,N,3,T,H,W], got {tuple(video.shape)}"
                )
            _, _, C, T, H, W = video.shape
            if C != 3:
                raise ValueError(f"Interleaved `video` channel dim must be 3, got {C}")
            flat_video = video.reshape(B * N, C, T, H, W)

        flat_cached_latents = None
        if cached_latents is not None:
            if cached_latents.ndim != 6:
                raise ValueError(
                    "Interleaved cached video latents must be [B,N,D,T,H,W], "
                    f"got {tuple(cached_latents.shape)}"
                )
            D, T_lat, H_lat, W_lat = cached_latents.shape[2:]
            flat_cached_latents = cached_latents.reshape(B * N, D, T_lat, H_lat, W_lat)

        flat_image_is_pad = None
        image_is_pad = segments.get("image_is_pad")
        if image_is_pad is not None:
            if image_is_pad.ndim != 3 or image_is_pad.shape[:2] != (B, N):
                raise ValueError(
                    "Interleaved `image_is_pad` must be [B,N,T], "
                    f"got {tuple(image_is_pad.shape)}."
                )
            flat_image_is_pad = image_is_pad.reshape(B * N, image_is_pad.shape[-1])

        flat_video_spatial_valid_mask = None
        video_spatial_valid_mask = segments.get("video_spatial_valid_mask")
        if video_spatial_valid_mask is not None:
            if (
                video_spatial_valid_mask.ndim != 4
                or video_spatial_valid_mask.shape[:2] != (B, N)
            ):
                raise ValueError(
                    "Interleaved `video_spatial_valid_mask` must be [B,N,H,W], "
                    f"got {tuple(video_spatial_valid_mask.shape)}."
                )
            flat_video_spatial_valid_mask = video_spatial_valid_mask.reshape(
                B * N,
                video_spatial_valid_mask.shape[-2],
                video_spatial_valid_mask.shape[-1],
            )

        task_ids = segments.get("task_token_ids")
        if task_ids is None and "prompt" in segments:
            flat_prompts = self._flatten_segment_prompts(segments["prompt"], batch_size=B, num_segments=N)
            task_ids = self._tokenize_prompt_batch(
                flat_prompts,
                max_length=int(getattr(self.language_expert, "max_task_len", 128)),
            ).reshape(B, N, -1)
        elif task_ids is not None and task_ids.ndim != 3:
            raise ValueError(f"`segments['task_token_ids']` must be [B,N,L], got {tuple(task_ids.shape)}")

        subtask_ids = segments.get("subtask_token_ids")
        if subtask_ids is not None and subtask_ids.ndim != 3:
            raise ValueError(f"`segments['subtask_token_ids']` must be [B,N,L], got {tuple(subtask_ids.shape)}")

        has_language = task_ids is not None and task_ids.shape[-1] > 0
        if subtask_ids is not None and not has_language:
            raise ValueError("`segments['subtask_token_ids']` requires non-empty `segments['task_token_ids']`.")

        lang_pre = None
        flat_subtask_ids = None
        task_len = 0
        subtask_len = 0
        if has_language:
            if task_ids.shape[:2] != (B, N):
                raise ValueError(f"`task_token_ids` leading dims must be {(B, N)}, got {tuple(task_ids.shape[:2])}")
            task_ids = task_ids.to(self.device, dtype=torch.long)
            if subtask_ids is None:
                subtask_ids = torch.empty(
                    (B, N, 0),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                if subtask_ids.shape[:2] != (B, N):
                    raise ValueError(
                        f"`subtask_token_ids` leading dims must be {(B, N)}, got {tuple(subtask_ids.shape[:2])}"
                    )
                subtask_ids = subtask_ids.to(self.device, dtype=torch.long)

            flat_task_ids = task_ids.reshape(B * N, task_ids.shape[-1])
            flat_subtask_ids = subtask_ids.reshape(B * N, subtask_ids.shape[-1])
            flat_lang_pre = self.language_expert.pre_dit(
                task_token_ids=flat_task_ids,
                subtask_token_ids=flat_subtask_ids,
            )
            task_len = int(flat_lang_pre["segments"]["task_len"])
            subtask_len = int(flat_lang_pre["segments"]["subtask_len"])
            lang_pre = self._merge_segment_pre_state(flat_lang_pre, batch_size=B, num_segments=N)

        if self.is_jepa_predictor:
            context_latents, target_video = self._prepare_jepa_context_target_latents(
                video=flat_video,
                cached_latents=flat_cached_latents,
                tiled=tiled,
                source="segments",
            )
            fuse_flag = False
            timestep_video = None
        else:
            input_latents = self._prepare_training_video_latents(
                video=flat_video,
                cached_latents=flat_cached_latents,
                tiled=tiled,
                source="segments",
            )
            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=B * N, device=self.device, dtype=input_latents.dtype,
            )
            noisy_latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
            fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
            if fuse_flag:
                noisy_latents[:, :, 0:1] = input_latents[:, :, 0:1]

        target_video_is_pad = None
        if self.is_jepa_predictor and flat_image_is_pad is not None:
            target_video_is_pad = self._causal_visual_target_is_pad(flat_image_is_pad)

        proprio_ctx, proprio_mask = self._make_interleaved_proprio_text_context(
            segments.get("proprio"),
            batch_size=B,
            num_segments=N,
            source="segments['proprio']",
        )
        has_proprio_context = proprio_ctx is not None
        if has_proprio_context:
            video_context, video_context_mask = proprio_ctx, proprio_mask
            action_context, action_context_mask = proprio_ctx, proprio_mask
        else:
            video_context, video_context_mask = self._make_dummy_text_context(B * N)
            action_context, action_context_mask = video_context, video_context_mask

        if self.is_jepa_predictor:
            flat_video_pre, flat_video_prediction_pre, prediction_video_tokens_per_segment = (
                self._prepare_jepa_training_video_pre(
                    context_latents=context_latents,
                    target_latents=target_video,
                    context=video_context,
                    context_mask=video_context_mask,
                )
            )
        else:
            flat_video_pre = self.video_expert.pre_dit(
                x=noisy_latents,
                timestep=timestep_video,
                context=video_context,
                context_mask=video_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            flat_video_prediction_pre = flat_video_pre
            prediction_video_tokens_per_segment = int(flat_video_pre["tokens"].shape[1])
        video_tokens_per_frame = int(flat_video_pre["meta"]["tokens_per_frame"])
        video_tokens_per_segment = int(flat_video_pre["tokens"].shape[1])
        video_context_payload = self._merge_segment_context_payload(
            flat_video_pre,
            batch_size=B,
            num_segments=N,
            segment_mask=segment_mask,
            enabled=has_proprio_context,
        )
        video_pre = self._merge_segment_pre_state(flat_video_pre, batch_size=B, num_segments=N)

        physical_action = segments.get("action")
        if self.latent_action_enabled and physical_action is None:
            raise ValueError("Latent-action mode requires physical `segments['action']` decoder targets.")
        action_pre = None
        target_action = None
        timestep_action = None
        noisy_action = None
        flat_action_is_pad = None
        flat_physical_action = None
        flat_physical_action_is_pad = None
        flat_action_dim_is_pad = None
        if physical_action is not None:
            if physical_action.ndim != 4:
                raise ValueError(
                    f"Interleaved `action` must be [B,N,T,D], got {tuple(physical_action.shape)}"
                )
            if physical_action.shape[:2] != (B, N):
                raise ValueError(
                    f"`action` leading dims must be {(B, N)}, got {tuple(physical_action.shape[:2])}"
                )
            flat_physical_action = physical_action.reshape(
                B * N, physical_action.shape[2], physical_action.shape[3]
            ).to(device=self.device, dtype=self.torch_dtype)
            physical_action_is_pad = segments.get(
                "action_is_pad", sample.get("action_is_pad", None)
            )
            if physical_action_is_pad is not None:
                if physical_action_is_pad.ndim != 3 or physical_action_is_pad.shape[:2] != (B, N):
                    raise ValueError(
                        "`action_is_pad` for interleaved input must be [B,N,T], "
                        f"got {tuple(physical_action_is_pad.shape)}"
                    )
                flat_physical_action_is_pad = physical_action_is_pad.reshape(
                    B * N, physical_action_is_pad.shape[-1]
                )

            if self.latent_action_enabled:
                latent_action = segments.get("latent_action")
                latent_action_is_pad = segments.get("latent_action_is_pad")
                if latent_action is None or latent_action_is_pad is None:
                    raise ValueError(
                        "Latent-action mode requires interleaved `latent_action` and "
                        "`latent_action_is_pad`."
                    )
                expected_latent = (B, N, 8, 32)
                expected_mask = (B, N, 8)
                if tuple(latent_action.shape) != expected_latent:
                    raise ValueError(
                        f"Interleaved `latent_action` must be {expected_latent}, got {tuple(latent_action.shape)}."
                    )
                if tuple(latent_action_is_pad.shape) != expected_mask:
                    raise ValueError(
                        "Interleaved `latent_action_is_pad` must be "
                        f"{expected_mask}, got {tuple(latent_action_is_pad.shape)}."
                    )
                flat_action = latent_action.reshape(B * N, 8, 32).to(
                    device=self.device, dtype=self.torch_dtype
                )
                flat_action_is_pad = latent_action_is_pad.reshape(B * N, 8)
                if tuple(flat_physical_action.shape[1:]) != (32, 14):
                    raise ValueError(
                        "Latent-action decoder target must be physical [B*N,32,14], got "
                        f"{tuple(flat_physical_action.shape)}."
                    )
            else:
                flat_action = flat_physical_action
                flat_action_is_pad = flat_physical_action_is_pad
                action_dim_is_pad = segments.get("action_dim_is_pad")
                flat_action_dim_is_pad = (
                    None
                    if action_dim_is_pad is None
                    else action_dim_is_pad.reshape(B * N, action_dim_is_pad.shape[-1])
                )
                flat_action = self._zero_padded_action_dims(flat_action, flat_action_dim_is_pad)

            noise_action = torch.randn_like(flat_action)
            if not self.latent_action_enabled:
                noise_action = self._zero_padded_action_dims(noise_action, flat_action_dim_is_pad)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=B * N, device=self.device, dtype=flat_action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(flat_action, noise_action, timestep_action)
            target_action = self.train_action_scheduler.training_target(flat_action, noise_action, timestep_action)
            flat_action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
            )
            action_context_payload = self._merge_segment_context_payload(
                flat_action_pre,
                batch_size=B,
                num_segments=N,
                segment_mask=segment_mask,
                enabled=has_proprio_context,
            )
            action_pre = self._merge_segment_pre_state(flat_action_pre, batch_size=B, num_segments=N)
        else:
            action_context_payload = None

        if lang_pre is None and action_pre is None:
            raise ValueError("Interleaved H-FastWAM input needs language tokens/prompts or action tokens.")

        tokens_out = self._run_mot_interleaved_segments(
            lang_pre=lang_pre,
            video_pre=video_pre,
            action_pre=action_pre,
            task_len=task_len,
            subtask_len=subtask_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_segments=N,
            video_context_payload=video_context_payload,
            action_context_payload=action_context_payload,
        )

        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_dict: dict[str, float] = {}

        flat_video_tokens_out = self._unmerge_segment_tokens(
            tokens_out["video"], batch_size=B, num_segments=N, tokens_per_segment=video_tokens_per_segment,
        )
        flat_prediction_tokens_out = flat_video_tokens_out[:, :prediction_video_tokens_per_segment]
        pred_video = self.video_expert.post_dit(
            flat_prediction_tokens_out,
            flat_video_prediction_pre,
        )
        loss_video = self._compute_video_loss(
            pred_video=pred_video,
            target_video=target_video,
            fuse_flag=fuse_flag,
            timestep_video=timestep_video,
            spatial_valid_mask=flat_video_spatial_valid_mask,
            video_is_pad=target_video_is_pad,
        )
        total_loss = total_loss + self.loss_lambda_video * loss_video
        loss_dict["loss_video"] = self.loss_lambda_video * float(loss_video.detach().item())

        if lang_pre is not None:
            lang_tokens_per_segment = task_len + subtask_len
            flat_lang_tokens_out = self._unmerge_segment_tokens(
                tokens_out["language"],
                batch_size=B,
                num_segments=N,
                tokens_per_segment=lang_tokens_per_segment,
            )
            if self.loss_lambda_language == 0.0:
                loss_dict["loss_language"] = 0.0
            else:
                lang_output = self.language_expert.post_dit(flat_lang_tokens_out, flat_lang_pre)
                language_losses = []
                if flat_task_ids.shape[1] > 1:
                    task_logits = self.language_expert.lm_head(
                        lang_output.hidden_states[:, :task_len, :]
                    )
                    language_losses.append(self._compute_language_token_loss(task_logits, flat_task_ids))
                if flat_subtask_ids is not None and flat_subtask_ids.shape[1] > 1:
                    language_losses.append(
                        self.language_expert.language_loss(
                            logits=lang_output.logits,
                            subtask_token_ids=flat_subtask_ids,
                        )
                    )
                if language_losses:
                    loss_language = torch.stack(language_losses).mean()
                    total_loss = total_loss + self.loss_lambda_language * loss_language
                    loss_dict["loss_language"] = self.loss_lambda_language * float(loss_language.detach().item())

        if action_pre is not None:
            action_tokens_per_segment = int(target_action.shape[1])
            flat_action_tokens_out = self._unmerge_segment_tokens(
                tokens_out["action"], batch_size=B, num_segments=N, tokens_per_segment=action_tokens_per_segment,
            )
            pred_action = self.action_expert.post_dit(flat_action_tokens_out, flat_action_pre)
            loss_action = self._compute_action_loss(
                pred_action=pred_action,
                target_action=target_action,
                timestep_action=timestep_action,
                action_is_pad=flat_action_is_pad,
                action_dim_is_pad=(
                    flat_action_dim_is_pad
                ),
            )
            total_loss = total_loss + self.loss_lambda_action * loss_action
            loss_dict["loss_action"] = self.loss_lambda_action * float(loss_action.detach().item())

            if self.latent_action_enabled:
                initial_proprio = self._select_interleaved_initial_proprio(
                    segments.get("proprio"),
                    batch_size=B,
                    num_segments=N,
                    source="segments['proprio']",
                )
                if initial_proprio is None:
                    raise ValueError("Latent-action decoding requires current normalized proprio.")
                generated_latent = self._estimate_clean_latent(
                    noisy_action,
                    pred_action,
                    timestep_action,
                    self.train_action_scheduler.num_train_timesteps,
                )
                decoder_latent, _ = self._select_decoder_latent(
                    flat_action,
                    generated_latent,
                    self._latent_action_oracle_probability(),
                )
                decoded_action = self.latent_action_decoder(
                    decoder_latent,
                    initial_proprio.reshape(B * N, self.proprio_dim).detach(),
                    context_latents[:, :, :1].detach(),
                    latent_is_pad=flat_action_is_pad.to(self.device, dtype=torch.bool),
                    flatten_output=True,
                )
                decoder_action_is_pad = flat_action_is_pad.to(
                    device=self.device,
                    dtype=torch.bool,
                ).repeat_interleave(
                    int(self.latent_action_config["actions_per_latent"]),
                    dim=1,
                )
                if flat_physical_action_is_pad is not None:
                    decoder_action_is_pad |= flat_physical_action_is_pad.to(
                        device=self.device,
                        dtype=torch.bool,
                    )
                decoder_loss = self._compute_latent_action_decoder_loss(
                    decoded_action,
                    flat_physical_action.detach(),
                    decoder_action_is_pad,
                    beta=float(self.latent_action_config["decoder_loss_beta"]),
                )
                total_loss = total_loss + self.loss_lambda_latent_action_decoder * decoder_loss
                loss_dict["loss_latent_action_decoder"] = (
                    self.loss_lambda_latent_action_decoder * float(decoder_loss.detach().item())
                )

        return total_loss, loss_dict

    def training_loss(
        self, sample: dict, tiled: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """Single training entry with token-driven automatic routing.

        Routes are inferred from sample tokens:
          - ``video_action``: no language tokens, valid action present
          - ``language_video``: language tokens present, action absent/invalid
          - ``full``: language tokens present, valid action present
        """
        if (
            "segments" in sample
            or ("video" in sample and getattr(sample["video"], "ndim", 0) == 6)
            or (
                "video_latents" in sample
                and getattr(sample["video_latents"], "ndim", 0) == 6
            )
        ):
            return self._training_loss_interleaved_segments(sample, tiled=tiled)

        sample = self._ensure_language_tokens_from_prompt(sample)
        route, modality_mask = self._validate_sample(sample)
        loss_dict: dict[str, float] = {}
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        # ---------- Optional language pre_dit ---------- #
        lang_pre = None
        subtask_ids = None
        lang_segments = {"task_len": 0, "subtask_len": 0}
        if modality_mask["language"]:
            task_ids = sample["task_token_ids"].to(self.device)
            if "subtask_token_ids" in sample:
                subtask_ids = sample["subtask_token_ids"].to(self.device)
            else:
                subtask_ids = torch.empty(
                    (task_ids.shape[0], 0),
                    device=self.device,
                    dtype=torch.long,
                )
            lang_pre = self.language_expert.pre_dit(
                task_token_ids=task_ids,
                subtask_token_ids=subtask_ids,
            )
            lang_segments = lang_pre["segments"]
            batch_size = int(task_ids.shape[0])
        else:
            vision_in = sample.get("video", sample.get("video_latents"))
            if vision_in.ndim != 5:
                raise ValueError(
                    "Non-interleaved vision input must be [B,D,T,H,W], "
                    f"got {tuple(vision_in.shape)}"
                )
            batch_size = int(vision_in.shape[0])

        # ---------- Prepare video pre_dit ---------- #
        video = sample.get("video")
        cached_latents = sample.get("video_latents")
        vision_batch = video if video is not None else cached_latents
        if video is not None and cached_latents is not None:
            if int(video.shape[0]) != int(cached_latents.shape[0]):
                raise ValueError(
                    "Raw/cached vision batch mismatch: "
                    f"video={video.shape[0]}, latents={cached_latents.shape[0]}."
                )
        if int(vision_batch.shape[0]) != batch_size:
            raise ValueError(
                f"Batch mismatch across modalities: vision batch={vision_batch.shape[0]} "
                f"vs expected {batch_size}"
            )

        input_latents = None
        if not self.is_jepa_predictor:
            input_latents = self._prepare_training_video_latents(
                video=video,
                cached_latents=cached_latents,
                tiled=tiled,
                source="sample",
            )

        proprio_ctx, proprio_mask = self._make_proprio_text_context(
            sample.get("proprio"),
            batch_size=batch_size,
            source="sample['proprio']",
        )
        has_proprio_context = proprio_ctx is not None
        if has_proprio_context:
            video_context, video_context_mask = proprio_ctx, proprio_mask
            action_context, action_context_mask = proprio_ctx, proprio_mask
        else:
            video_context, video_context_mask = self._make_dummy_text_context(batch_size)
            action_context, action_context_mask = video_context, video_context_mask

        if self.is_jepa_predictor:
            # JEPA: online context states predict the next frozen-teacher states.
            context_latents, target_video = self._prepare_jepa_context_target_latents(
                video=video,
                cached_latents=cached_latents,
                tiled=tiled,
                source="sample",
            )
            fuse_flag = False
            timestep_video = None
            video_pre = self.video_expert.pre_dit(
                x=context_latents,
                context=video_context if self.video_expert.use_text_context else None,
                context_mask=video_context_mask if self.video_expert.use_text_context else None,
            )
        else:
            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
            )
            noisy_latents = self.train_video_scheduler.add_noise(
                input_latents, noise_video, timestep_video,
            )
            target_video = self.train_video_scheduler.training_target(
                input_latents, noise_video, timestep_video,
            )
            fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
            if fuse_flag:
                noisy_latents[:, :, 0:1] = input_latents[:, :, 0:1]
            video_pre = self.video_expert.pre_dit(
                x=noisy_latents,
                timestep=timestep_video,
                context=video_context,
                context_mask=video_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        video_context_payload = self._context_payload_from_pre_state(video_pre, has_proprio_context)

        # ---------- Optional action pre_dit ---------- #
        action_pre = None
        target_action = None
        timestep_action = None
        noisy_action = None
        physical_action = None
        physical_action_is_pad = None
        action_dim_is_pad = None
        if modality_mask["action"]:
            physical_action = sample["action"].to(device=self.device, dtype=self.torch_dtype)
            if physical_action.ndim != 3:
                raise ValueError(
                    f"sample['action'] must be [B,T,a_dim], got {tuple(physical_action.shape)}"
                )
            if int(physical_action.shape[0]) != batch_size:
                raise ValueError(
                    "Batch mismatch across modalities: "
                    f"action batch={physical_action.shape[0]} vs expected {batch_size}"
                )
            physical_action_is_pad = sample.get("action_is_pad", None)
            if self.latent_action_enabled:
                action = sample.get("latent_action")
                action_is_pad = sample.get("latent_action_is_pad")
                if action is None or action_is_pad is None:
                    raise ValueError(
                        "Latent-action mode requires `latent_action` and `latent_action_is_pad`."
                    )
                if tuple(action.shape) != (batch_size, 8, 32):
                    raise ValueError(
                        f"`latent_action` must be [B,8,32], got {tuple(action.shape)}."
                    )
                if tuple(action_is_pad.shape) != (batch_size, 8):
                    raise ValueError(
                        f"`latent_action_is_pad` must be [B,8], got {tuple(action_is_pad.shape)}."
                    )
                if tuple(physical_action.shape[1:]) != (32, 14):
                    raise ValueError(
                        "Latent-action decoder target must be physical [B,32,14], got "
                        f"{tuple(physical_action.shape)}."
                    )
                action = action.to(device=self.device, dtype=self.torch_dtype)
            else:
                action = physical_action
                action_is_pad = physical_action_is_pad
                action_dim_is_pad = sample.get("action_dim_is_pad", None)
                action = self._zero_padded_action_dims(action, action_dim_is_pad)

            noise_action = torch.randn_like(action)
            if not self.latent_action_enabled:
                noise_action = self._zero_padded_action_dims(noise_action, action_dim_is_pad)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(
                action, noise_action, timestep_action,
            )
            target_action = self.train_action_scheduler.training_target(
                action, noise_action, timestep_action,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
            )
            action_context_payload = self._context_payload_from_pre_state(action_pre, has_proprio_context)
        else:
            action_context_payload = None

        # ---------- MoT forward by active experts ---------- #
        if route == "full":
            if lang_pre is None or action_pre is None:
                raise RuntimeError("Route 'full' requires both language and action pre states.")
            tokens_out = self._run_mot_three_experts(
                lang_pre=lang_pre,
                video_pre=video_pre,
                action_pre=action_pre,
                task_len=lang_segments["task_len"],
                subtask_len=lang_segments["subtask_len"],
                video_tokens_per_frame=video_tokens_per_frame,
                video_context_payload=video_context_payload,
                action_context_payload=action_context_payload,
            )
        elif route == "language_video":
            if lang_pre is None:
                raise RuntimeError("Route 'language_video' requires language pre state.")
            tokens_out = self._run_mot_two_experts_lv(
                lang_pre=lang_pre,
                video_pre=video_pre,
                task_len=lang_segments["task_len"],
                subtask_len=lang_segments["subtask_len"],
                video_tokens_per_frame=video_tokens_per_frame,
                video_context_payload=video_context_payload,
            )
        elif route == "video_action":
            if action_pre is None:
                raise RuntimeError("Route 'video_action' requires action pre state.")
            tokens_out = self._run_mot_two_experts_va(
                video_pre=video_pre,
                action_pre=action_pre,
                video_tokens_per_frame=video_tokens_per_frame,
                video_context_payload=video_context_payload,
                action_context_payload=action_context_payload,
            )
        else:
            raise ValueError(f"Unknown training route: {route}")

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        if modality_mask["language"]:
            if self.loss_lambda_language == 0.0:
                loss_dict["loss_language"] = 0.0
            else:
                lang_output = self.language_expert.post_dit(tokens_out["language"], lang_pre)
                language_losses = []
                task_len = int(lang_segments["task_len"])
                if task_ids.shape[1] > 1:
                    task_logits = self.language_expert.lm_head(
                        lang_output.hidden_states[:, :task_len, :]
                    )
                    language_losses.append(self._compute_language_token_loss(task_logits, task_ids))
                if subtask_ids is not None and subtask_ids.shape[1] > 1:
                    language_losses.append(
                        self.language_expert.language_loss(
                            logits=lang_output.logits,
                            subtask_token_ids=subtask_ids,
                        )
                    )
                if language_losses:
                    loss_language = torch.stack(language_losses).mean()
                    total_loss = total_loss + self.loss_lambda_language * loss_language
                    loss_dict["loss_language"] = self.loss_lambda_language * float(loss_language.detach().item())

        loss_video = self._compute_video_loss(
            pred_video=pred_video,
            target_video=target_video,
            fuse_flag=fuse_flag,
            timestep_video=timestep_video,
            spatial_valid_mask=sample.get("video_spatial_valid_mask"),
            video_is_pad=(
                self._causal_visual_target_is_pad(sample["image_is_pad"])
                if self.is_jepa_predictor and sample.get("image_is_pad") is not None
                else None
            ),
        )
        total_loss = total_loss + self.loss_lambda_video * loss_video
        loss_dict["loss_video"] = self.loss_lambda_video * float(loss_video.detach().item())

        if modality_mask["action"]:
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
            loss_action = self._compute_action_loss(
                pred_action=pred_action,
                target_action=target_action,
                timestep_action=timestep_action,
                action_is_pad=action_is_pad,
                action_dim_is_pad=action_dim_is_pad,
            )
            total_loss = total_loss + self.loss_lambda_action * loss_action
            loss_dict["loss_action"] = self.loss_lambda_action * float(loss_action.detach().item())

            if self.latent_action_enabled:
                initial_proprio = self._select_initial_proprio(
                    sample.get("proprio"),
                    batch_size=batch_size,
                    source="sample['proprio']",
                )
                if initial_proprio is None:
                    raise ValueError("Latent-action decoding requires current normalized proprio.")
                generated_latent = self._estimate_clean_latent(
                    noisy_action,
                    pred_action,
                    timestep_action,
                    self.train_action_scheduler.num_train_timesteps,
                )
                decoder_latent, _ = self._select_decoder_latent(
                    action,
                    generated_latent,
                    self._latent_action_oracle_probability(),
                )
                decoded_action = self.latent_action_decoder(
                    decoder_latent,
                    initial_proprio.detach(),
                    context_latents[:, :, :1].detach(),
                    latent_is_pad=action_is_pad.to(self.device, dtype=torch.bool),
                    flatten_output=True,
                )
                decoder_action_is_pad = action_is_pad.to(
                    device=self.device,
                    dtype=torch.bool,
                ).repeat_interleave(
                    int(self.latent_action_config["actions_per_latent"]),
                    dim=1,
                )
                if physical_action_is_pad is not None:
                    decoder_action_is_pad |= physical_action_is_pad.to(
                        device=self.device,
                        dtype=torch.bool,
                    )
                decoder_loss = self._compute_latent_action_decoder_loss(
                    decoded_action,
                    physical_action.detach(),
                    decoder_action_is_pad,
                    beta=float(self.latent_action_config["decoder_loss_beta"]),
                )
                total_loss = total_loss + self.loss_lambda_latent_action_decoder * decoder_loss
                loss_dict["loss_latent_action_decoder"] = (
                    self.loss_lambda_latent_action_decoder * float(decoder_loss.detach().item())
                )

        return total_loss, loss_dict

    # ------------------------------------------------------------------ #
    # Specialized forward helpers
    # ------------------------------------------------------------------ #
    def _run_mot_three_experts(
        self,
        lang_pre: dict,
        video_pre: dict,
        action_pre: dict,
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
        video_context_payload: Optional[dict] = None,
        action_context_payload: Optional[dict] = None,
    ) -> dict:
        video_seq_len = int(video_pre["tokens"].shape[1])
        action_seq_len = int(action_pre["tokens"].shape[1])

        attention_mask = self._build_full_attention_mask(
            task_len=task_len, subtask_len=subtask_len,
            video_seq_len=video_seq_len, action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )

        detach_set = set()
        if self.knowledge_insulation:
            detach_set.add("language")
        if self.action_loss_detach_video_expert:
            detach_set.add("video")

        return self.mot(
            embeds_all={
                "language": lang_pre["tokens"],
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "language": lang_pre["freqs"],
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "language": None,  # no cross-attn in language expert
                "video": video_context_payload,
                "action": action_context_payload,
            },
            t_mod_all={
                "language": lang_pre["t_mod"],
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            detach_kv_experts=detach_set or None,
        )

    def _prepare_inference_action_video_pre(
        self,
        lang_pre: dict,
        video_pre: dict,
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
        video_context_payload: Optional[dict],
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        first_frame_latents: torch.Tensor,
        num_video_frames: Optional[int],
    ) -> dict:
        """Prepare the video branch used by action denoising."""
        del (
            lang_pre,
            task_len,
            subtask_len,
            video_tokens_per_frame,
            video_context_payload,
            video_context,
            video_context_mask,
            first_frame_latents,
            num_video_frames,
        )
        return video_pre

    def _prepare_action_inference_cache(
        self,
        lang_pre: dict,
        video_pre: dict,
        task_len: int,
        subtask_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        video_context_payload: Optional[dict],
    ) -> Optional[dict]:
        del (
            lang_pre,
            video_pre,
            task_len,
            subtask_len,
            action_seq_len,
            video_tokens_per_frame,
            video_context_payload,
        )
        return None

    def _run_mot_action_inference(
        self,
        lang_pre: dict,
        video_pre: dict,
        action_pre: dict,
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
        video_context_payload: Optional[dict] = None,
        action_context_payload: Optional[dict] = None,
        action_inference_cache: Optional[dict] = None,
    ) -> dict:
        del action_inference_cache
        return self._run_mot_three_experts(
            lang_pre=lang_pre,
            video_pre=video_pre,
            action_pre=action_pre,
            task_len=task_len,
            subtask_len=subtask_len,
            video_tokens_per_frame=video_tokens_per_frame,
            video_context_payload=video_context_payload,
            action_context_payload=action_context_payload,
        )

    def _run_mot_two_experts_lv(
        self,
        lang_pre: dict,
        video_pre: dict,
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
        video_context_payload: Optional[dict] = None,
    ) -> dict:
        """2-expert MoT pass over {language, video}."""
        video_seq_len = int(video_pre["tokens"].shape[1])

        mask_full = self._build_full_attention_mask(
            task_len=task_len, subtask_len=subtask_len,
            video_seq_len=video_seq_len, action_seq_len=0,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        detach_set = {"language"} if self.knowledge_insulation else None

        return self.mot(
            embeds_all={"language": lang_pre["tokens"], "video": video_pre["tokens"]},
            attention_mask=mask_full,
            freqs_all={"language": lang_pre["freqs"], "video": video_pre["freqs"]},
            context_all={"language": None, "video": video_context_payload},
            t_mod_all={"language": lang_pre["t_mod"], "video": video_pre["t_mod"]},
            detach_kv_experts=detach_set,
            active_expert_order=("language", "video"),
        )

    def _run_mot_two_experts_va(
        self,
        video_pre: dict,
        action_pre: dict,
        video_tokens_per_frame: int,
        video_context_payload: Optional[dict] = None,
        action_context_payload: Optional[dict] = None,
    ) -> dict:
        """2-expert MoT pass over {video, action}."""
        video_seq_len = int(video_pre["tokens"].shape[1])
        action_seq_len = int(action_pre["tokens"].shape[1])

        mask_full = self._build_full_attention_mask(
            task_len=0,
            subtask_len=0,
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        # Align with FastWAM's `action_loss_detach_video_expert=True`: the action
        # stream attends to *detached* video K/V so that action-loss gradients do
        # not flow into the (from-scratch, still-noisy) video expert. Without this
        # the action expert chases a moving visual condition, which slows action
        # convergence and inflates action loss relative to FastWAM.
        return self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=mask_full,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": video_context_payload, "action": action_context_payload},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            detach_kv_experts={"video"} if self.action_loss_detach_video_expert else None,
            active_expert_order=("video", "action"),
        )

    # ------------------------------------------------------------------ #
    # Loss helpers
    # ------------------------------------------------------------------ #
    def _causal_visual_target_is_pad(self, image_is_pad: torch.Tensor) -> torch.Tensor:
        if image_is_pad.ndim != 2:
            raise ValueError(
                f"image_is_pad must be [B,T], got {tuple(image_is_pad.shape)}."
            )
        encoder = self.visual_encoder
        temporal_patch = int(getattr(encoder, "_temporal_patch", 1))
        temporal_stride = int(getattr(encoder, "temporal_downsample_factor", 1))
        num_frames = int(image_is_pad.shape[1])
        anchors = list(range(0, num_frames, temporal_stride))
        if len(anchors) < 2:
            raise ValueError(f"At least two visual anchors are required, got {anchors}.")
        target_masks = []
        for frame_index in anchors[1:]:
            start = max(frame_index - temporal_patch + 1, 0)
            target_masks.append(image_is_pad[:, start : frame_index + 1].all(dim=1))
        return torch.stack(target_masks, dim=1)

    def _compute_video_loss(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        fuse_flag: bool = False,
        timestep_video: Optional[torch.Tensor] = None,
        spatial_valid_mask: Optional[torch.Tensor] = None,
        video_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = None
        if spatial_valid_mask is not None:
            valid = spatial_valid_mask.to(
                device=pred_video.device,
                dtype=torch.float32,
            )
            if valid.ndim == 3:
                valid = valid.unsqueeze(1)
            if (
                valid.ndim != 4
                or valid.shape[0] != pred_video.shape[0]
                or valid.shape[1] != 1
            ):
                raise ValueError(
                    "`video_spatial_valid_mask` must be [B,H,W] or [B,1,H,W], "
                    f"got {tuple(valid.shape)} for video {tuple(pred_video.shape)}."
                )
            valid = F.interpolate(
                valid,
                size=pred_video.shape[-2:],
                mode="nearest",
            ).unsqueeze(2)

        if video_is_pad is not None:
            video_is_pad = video_is_pad.to(device=pred_video.device, dtype=torch.bool)
            if video_is_pad.ndim != 2 or video_is_pad.shape != (
                pred_video.shape[0], pred_video.shape[2]
            ):
                raise ValueError(
                    "video_is_pad must match [B,T] of the video prediction, "
                    f"got {tuple(video_is_pad.shape)} for {tuple(pred_video.shape)}."
                )
            temporal_valid = (~video_is_pad).to(dtype=torch.float32)[:, None, :, None, None]
            valid = temporal_valid if valid is None else valid * temporal_valid

        def reduce_error(error: torch.Tensor) -> torch.Tensor:
            if valid is None:
                return error.mean(dim=(1, 2, 3, 4))
            expanded_valid = valid.expand_as(error)
            denominator = expanded_valid.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
            return (error * expanded_valid).sum(dim=(1, 2, 3, 4)) / denominator

        if self.video_loss_type == "l1":
            # JEPA predictor: plain L1 regression in encoder-latent space.
            # No timestep weighting — the predictor is deterministic.
            error = F.l1_loss(
                pred_video.float(),
                target_video.float(),
                reduction="none",
            )
            return reduce_error(error).mean()
        # Flow-matching (WAN DiT): MSE weighted by the flow-matching training weight.
        if fuse_flag:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        error = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none",
        )
        per_sample = reduce_error(error)
        if timestep_video is None:
            raise ValueError("timestep_video is required for flow_matching video loss.")
        w = self.train_video_scheduler.training_weight(timestep_video).to(
            device=per_sample.device, dtype=per_sample.dtype,
        )
        return (per_sample * w).mean()

    def _compute_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        element_loss = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction="none",
        )
        valid = torch.ones_like(element_loss, dtype=torch.bool)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(self.device, dtype=torch.bool)
            valid &= ~action_is_pad.unsqueeze(-1)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(self.device, dtype=torch.bool)
            if action_dim_is_pad.ndim == 1:
                action_dim_is_pad = action_dim_is_pad.unsqueeze(0)
            if action_dim_is_pad.shape != (
                element_loss.shape[0],
                element_loss.shape[2],
            ):
                raise ValueError(
                    "`action_dim_is_pad` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)}, expected "
                    f"{(element_loss.shape[0], element_loss.shape[2])}."
                )
            valid &= ~action_dim_is_pad.unsqueeze(1)
        valid_float = valid.to(dtype=element_loss.dtype)
        valid_count = valid_float.sum(dim=(1, 2)).clamp(min=1.0)
        per_sample = (element_loss * valid_float).sum(dim=(1, 2)) / valid_count
        w = self.train_action_scheduler.training_weight(timestep_action).to(
            device=per_sample.device, dtype=per_sample.dtype,
        )
        return (per_sample * w).mean()

    @staticmethod
    def _compute_language_token_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"`logits` must be [B,L,V], got {tuple(logits.shape)}")
        if labels.ndim != 2:
            raise ValueError(f"`labels` must be [B,L], got {tuple(labels.shape)}")
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                "Language logits/labels shape mismatch: "
                f"{tuple(logits.shape[:2])} vs {tuple(labels.shape)}"
            )
        vocab_size = int(logits.shape[-1])
        shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        return F.cross_entropy(
            shift_logits.float(),
            shift_labels.to(device=shift_logits.device),
            ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
        )

    # ------------------------------------------------------------------ #
    # Inference (action-only)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer_action(
        self,
        image: Optional[torch.Tensor] = None,
        task_token_ids: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        subtask_token_ids: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_subtask_tokens: int = 64,
        generate_subtask: bool = False,
        *,
        # FastWAM-compatible aliases used by LIBERO evaluators.
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_video_frames: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Inference: language AR-generates subtask, then 1-shot video + action denoising.

        This method accepts both native H-FastWAM inputs
        (``image`` + ``task_token_ids``) and FastWAM-style LIBERO evaluator
        inputs (``prompt`` + ``input_image``).
        """
        del kwargs, context, context_mask, negative_prompt, text_cfg_scale

        if action_horizon is None or int(action_horizon) <= 0:
            raise ValueError(f"`action_horizon` must be a positive integer, got {action_horizon}")
        action_horizon = int(action_horizon)
        requested_action_horizon = action_horizon
        if self.latent_action_enabled:
            expected_physical_horizon = int(self.latent_action_config["physical_action_horizon"])
            if requested_action_horizon != expected_physical_horizon:
                raise ValueError(
                    "Latent-action inference requires external action_horizon="
                    f"{expected_physical_horizon}, got {requested_action_horizon}."
                )
            action_horizon = int(self.latent_action_config["latent_horizon"])

        self.eval()
        _vmask = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if not self.is_jepa_predictor and _vmask != "first_frame_causal":
            raise ValueError(
                "infer_action requires video_attention_mask_mode='first_frame_causal'; "
                f"got '{_vmask}'. (For JEPAPredictor per_frame_causal is accepted.)"
            )

        if image is None:
            image = input_image
        elif input_image is not None:
            logger.warning("Both `image` and `input_image` were provided; `image` takes priority.")

        if image is None:
            raise ValueError("Either `image` or `input_image` must be provided.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"image must be [1,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        image = image.to(device=self.device, dtype=self.torch_dtype)

        if task_token_ids is None:
            if prompt is None:
                raise ValueError("Either `task_token_ids` or `prompt` must be provided.")
            task_token_ids = self._tokenize_task_prompt(prompt)
        elif not torch.is_tensor(task_token_ids):
            task_token_ids = torch.as_tensor(task_token_ids, dtype=torch.long)

        if task_token_ids.ndim == 1:
            task_token_ids = task_token_ids.unsqueeze(0)
        elif task_token_ids.ndim != 2:
            raise ValueError(
                f"`task_token_ids` must be 1D/2D [L] or [B,L], got {tuple(task_token_ids.shape)}"
            )
        task_token_ids = task_token_ids.to(self.device, dtype=torch.long)

        if subtask_token_ids is not None:
            if not torch.is_tensor(subtask_token_ids):
                subtask_token_ids = torch.as_tensor(subtask_token_ids, dtype=torch.long)
            if subtask_token_ids.ndim == 1:
                subtask_token_ids = subtask_token_ids.unsqueeze(0)
            elif subtask_token_ids.ndim != 2:
                raise ValueError(
                    "`subtask_token_ids` must be 1D/2D [L] or [B,L], "
                    f"got {tuple(subtask_token_ids.shape)}"
                )


        # Encode first frame into video latents (for both subtask gen and action)
        first_frame_latents = self._encode_first_frame(image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        proprio_ctx, proprio_mask = self._make_proprio_text_context(
            proprio,
            batch_size=1,
            source="proprio",
        )
        has_proprio_context = proprio_ctx is not None
        if has_proprio_context:
            video_context, video_context_mask = proprio_ctx, proprio_mask
            action_context, action_context_mask = proprio_ctx, proprio_mask
        else:
            video_context, video_context_mask = self._make_dummy_text_context(1)
            action_context, action_context_mask = video_context, video_context_mask

        # Video pre_dit for first frame (used as visual grounding)
        if self.is_jepa_predictor:
            video_pre_ff = self.video_expert.pre_dit(
                x=first_frame_latents,
                context=video_context if self.video_expert.use_text_context else None,
                context_mask=video_context_mask if self.video_expert.use_text_context else None,
            )
        else:
            timestep_zero = torch.zeros(
                (first_frame_latents.shape[0],),
                dtype=first_frame_latents.dtype, device=self.device,
            )
            video_pre_ff = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_zero,
                context=video_context,
                context_mask=video_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
        video_tokens_per_frame = int(video_pre_ff["meta"]["tokens_per_frame"])
        video_context_payload = self._context_payload_from_pre_state(video_pre_ff, has_proprio_context)

        # ---- Step 1: subtask handling ---- #
        # Train/infer alignment: LIBERO (and any task-only run) trains with an
        # EMPTY subtask segment (subtask_len=0), so the action expert only ever
        # attends to language K/V = [task]. To match that exactly at inference we
        # must NOT autoregressively generate a subtask (the frozen LM was never
        # trained for it; a generated subtask injects an unseen language context
        # and shifts the action denoising distribution). Default behaviour is an
        # empty subtask. Pass an explicit `subtask_token_ids`, or set
        # `generate_subtask=True`, to opt into the AR-generated subtask.
        if subtask_token_ids is not None:
            subtask_ids_used = subtask_token_ids.to(self.device)
        elif generate_subtask:
            subtask_ids_used = self._generate_subtask(
                task_ids=task_token_ids,
                video_pre=video_pre_ff,
                video_tokens_per_frame=video_tokens_per_frame,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_subtask_tokens,
                video_context_payload=video_context_payload,
            )
        else:
            subtask_ids_used = torch.empty(
                (task_token_ids.shape[0], 0), dtype=torch.long, device=self.device,
            )

        # ---- Step 2: Full 3-expert MoT for action denoising ---------- #
        # Re-run language pre_dit with finalised subtask for consistent K/V
        lang_pre = self.language_expert.pre_dit(
            task_token_ids=task_token_ids,
            subtask_token_ids=subtask_ids_used,
        )
        seg = lang_pre["segments"]
        action_video_pre = self._prepare_inference_action_video_pre(
            lang_pre=lang_pre,
            video_pre=video_pre_ff,
            task_len=seg["task_len"],
            subtask_len=seg["subtask_len"],
            video_tokens_per_frame=video_tokens_per_frame,
            video_context_payload=video_context_payload,
            video_context=video_context,
            video_context_mask=video_context_mask,
            first_frame_latents=first_frame_latents,
            num_video_frames=num_video_frames,
        )
        action_video_tokens_per_frame = int(
            action_video_pre["meta"]["tokens_per_frame"]
        )
        action_video_context_payload = self._context_payload_from_pre_state(
            action_video_pre,
            has_proprio_context,
        )

        # Noisy action init
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        action_inference_cache = self._prepare_action_inference_cache(
            lang_pre=lang_pre,
            video_pre=action_video_pre,
            task_len=seg["task_len"],
            subtask_len=seg["subtask_len"],
            action_seq_len=int(latents_action.shape[1]),
            video_tokens_per_frame=action_video_tokens_per_frame,
            video_context_payload=action_video_context_payload,
        )

        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
            )
            action_context_payload = self._context_payload_from_pre_state(action_pre, has_proprio_context)

            tokens_out = self._run_mot_action_inference(
                lang_pre=lang_pre,
                video_pre=action_video_pre,
                action_pre=action_pre,
                task_len=seg["task_len"],
                subtask_len=seg["subtask_len"],
                video_tokens_per_frame=action_video_tokens_per_frame,
                video_context_payload=action_video_context_payload,
                action_context_payload=action_context_payload,
                action_inference_cache=action_inference_cache,
            )
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta, latents_action,
            )

        if self.latent_action_enabled:
            initial_proprio = self._select_initial_proprio(
                proprio,
                batch_size=1,
                source="proprio",
            )
            if initial_proprio is None:
                raise ValueError("Latent-action inference requires current normalized proprio.")
            output_action = self.latent_action_decoder(
                latents_action,
                initial_proprio,
                first_frame_latents[:, :, :1],
                flatten_output=True,
            )[0]
        else:
            output_action = latents_action[0]

        return {
            "action": output_action.detach().to(device="cpu", dtype=torch.float32),
            "subtask_tokens": subtask_ids_used[0].detach().cpu(),
        }

    # ------------------------------------------------------------------ #
    # Subtask generation (language + video 2-expert MoT)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate_subtask(
        self,
        task_ids: torch.Tensor,
        video_pre: dict,
        video_tokens_per_frame: int,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        video_context_payload: Optional[dict] = None,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive subtask decoding via 2-expert MoT (language + video).

        Language expert sees the video expert's first-frame tokens via MoT
        shared attention. Video tokens are **frozen** (just providing K/V).
        """
        B = task_ids.shape[0]
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            lang_pre = self.language_expert.pre_dit(
                task_token_ids=task_ids,
                subtask_token_ids=generated,
            )
            seg = lang_pre["segments"]

            # Build 2-expert attention mask (language + video, no action)
            video_seq_len = int(video_pre["tokens"].shape[1])
            mask = self._build_full_attention_mask(
                task_len=seg["task_len"],
                subtask_len=seg["subtask_len"],
                video_seq_len=video_seq_len,
                action_seq_len=0,
                video_tokens_per_frame=video_tokens_per_frame,
                device=lang_pre["tokens"].device,
            )

            # Run the language/video subset through the registered MoT so the
            # heterogeneous projection adapters are shared with training.
            detach_set = {"language"} if self.knowledge_insulation else None

            tokens_out = self.mot(
                embeds_all={"language": lang_pre["tokens"], "video": video_pre["tokens"]},
                attention_mask=mask,
                freqs_all={"language": lang_pre["freqs"], "video": video_pre["freqs"]},
                context_all={"language": None, "video": video_context_payload},
                t_mod_all={"language": lang_pre["t_mod"], "video": video_pre["t_mod"]},
                detach_kv_experts=detach_set,
                active_expert_order=("language", "video"),
            )

            logits_last = self.language_expert.step_logits(
                tokens_after_mot=tokens_out["language"],
                task_len=seg["task_len"],
            )  # [B, 1, vocab]

            if temperature > 0:
                logits_last = logits_last / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                    logits_last[logits_last < v[:, :, [-1]]] = -float("inf")
                probs = F.softmax(logits_last.float(), dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                next_token = logits_last.argmax(dim=-1)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #
    def set_latent_action_cache_manifest(self, manifest: dict) -> None:
        """Copy validated cache provenance into latent checkpoint metadata."""
        if not self.latent_action_enabled:
            return
        if not isinstance(manifest, dict):
            raise TypeError("Latent-action cache manifest must be a dict.")
        signature = manifest.get("signature")
        signature_payload = manifest.get("signature_payload")
        if not isinstance(signature, str) or not isinstance(signature_payload, dict):
            raise ValueError(
                "Latent-action cache manifest is missing signed provenance metadata."
            )
        normalization = signature_payload.get("normalization")
        dreamdojo = signature_payload.get("dreamdojo")
        if not isinstance(normalization, dict) or not isinstance(dreamdojo, dict):
            raise ValueError(
                "Latent-action cache signature payload must contain normalization and DreamDojo metadata."
            )
        provenance_fields = {
            "dreamdojo_code_revision": "git_revision",
            "dreamdojo_checkpoint_revision": "checkpoint_revision",
            "dreamdojo_checkpoint_hash": "checkpoint_sha256",
        }
        for config_key, manifest_key in provenance_fields.items():
            actual = dreamdojo.get(manifest_key)
            expected = self.latent_action_config.get(config_key)
            if not isinstance(actual, str) or not actual:
                raise ValueError(
                    "Latent-action cache signature payload is missing DreamDojo "
                    f"provenance field `{manifest_key}`."
                )
            if expected not in (None, "") and actual != str(expected):
                raise ValueError(
                    f"DreamDojo provenance mismatch for `{config_key}`: "
                    f"config={expected!r}, manifest={actual!r}."
                )
        expected_signature = self.latent_action_config.get("latent_cache_signature")
        if expected_signature not in (None, "") and signature != str(expected_signature):
            raise ValueError(
                "Latent cache signature mismatch between model config and dataset manifest: "
                f"config={expected_signature!r}, manifest={signature!r}."
            )
        self.latent_action_config["latent_cache_signature"] = signature
        self.latent_action_config["latent_normalization_stats"] = copy.deepcopy(
            normalization
        )
        for config_key, manifest_key in provenance_fields.items():
            self.latent_action_config[config_key] = dreamdojo[manifest_key]

    def _checkpoint_metadata(self) -> dict:
        metadata = {
            "checkpoint_schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "action_representation": "latent" if self.latent_action_enabled else "direct",
        }
        if self.latent_action_enabled:
            metadata.update(self.LATENT_ACTION_CONTRACT)
            metadata["latent_action_config"] = dict(self.latent_action_config)
            metadata["latent_action_decoder_config"] = dict(
                self.latent_action_decoder_config
            )
            for key in (
                "latent_cache_signature",
                "dreamdojo_code_revision",
                "dreamdojo_checkpoint_revision",
                "dreamdojo_checkpoint_hash",
                "latent_normalization_stats",
                "predictor_source_checkpoint",
                "predictor_source_hash",
            ):
                if key in self.latent_action_config:
                    metadata[key] = self.latent_action_config[key]
        return metadata

    def save_checkpoint(self, path: str, optimizer=None, step=None):
        payload = {
            "language_expert": self.language_expert.state_dict(),
            "mot": self.mot.state_dict(),
            "checkpoint_metadata": self._checkpoint_metadata(),
            "training_phase": self._training_phase,
            "torch_dtype": str(self.torch_dtype),
            "step": step,
        }
        if self.latent_action_enabled:
            payload["latent_action_decoder"] = self.latent_action_decoder.state_dict()
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.use_visual_encoder:
            payload["visual_encoder"] = self.visual_encoder.state_dict()
        payload["fixed_target_encoder"] = self.fixed_target_encoder is not None
        teacher_handle = self.__dict__.get("_fixed_teacher_handle")
        if teacher_handle is not None:
            payload["fixed_target_source"] = teacher_handle.source_checkpoint
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer=None, strict: bool = False):
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )

        metadata = payload.get("checkpoint_metadata")
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint `checkpoint_metadata` must be a dictionary.")
            schema_version = metadata.get("checkpoint_schema_version")
            if (
                isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version != self.CHECKPOINT_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Unsupported checkpoint schema version: "
                    f"got {schema_version!r}, expected {self.CHECKPOINT_SCHEMA_VERSION}."
                )
        checkpoint_representation = None if metadata is None else metadata.get("action_representation")
        expected_representation = "latent" if self.latent_action_enabled else "direct"
        if checkpoint_representation is not None and checkpoint_representation != expected_representation:
            raise ValueError(
                "Checkpoint/model action representation mismatch: "
                f"checkpoint={checkpoint_representation!r}, model={expected_representation!r}."
            )
        if strict and metadata is None and self.latent_action_enabled:
            raise ValueError(
                "Strict latent-action checkpoint load requires `checkpoint_metadata`."
            )
        if self.latent_action_enabled and metadata is not None:
            for key, expected in self.LATENT_ACTION_CONTRACT.items():
                if int(metadata.get(key, -1)) != expected:
                    raise ValueError(
                        f"Checkpoint latent-action `{key}` mismatch: "
                        f"got {metadata.get(key)!r}, expected {expected}."
                    )
            expected_signature = self.latent_action_config.get("latent_cache_signature")
            actual_signature = metadata.get("latent_cache_signature")
            if expected_signature is not None and actual_signature != expected_signature:
                raise ValueError(
                    "Latent cache signature mismatch: "
                    f"checkpoint={actual_signature!r}, expected={expected_signature!r}."
                )

        def _validate(name, missing, unexpected, allowed_missing=()):
            allowed_missing = set(allowed_missing)
            required_missing = [key for key in missing if key not in allowed_missing]
            logger.info(
                "Loaded %s (missing=%d, allowed_missing=%d, unexpected=%d).",
                name,
                len(required_missing),
                len(missing) - len(required_missing),
                len(unexpected),
            )
            if strict and (required_missing or unexpected):
                raise ValueError(
                    f"Strict checkpoint load failed for {name}: "
                    f"missing={required_missing[:20]}, unexpected={unexpected[:20]}"
                )

        if "language_expert" in payload:
            _validate("language_expert", *self.language_expert.load_state_dict(
                payload["language_expert"], strict=False,
            ))
        elif strict:
            raise ValueError(f"Strict checkpoint is missing `language_expert`: {path}")
        if "mot" in payload:
            mot_state = payload["mot"]
            if self.latent_action_enabled and checkpoint_representation is None:
                legacy_action_head_keys = [
                    key
                    for key in mot_state
                    if any(
                        key.startswith(prefix)
                        for prefix in self.ACTION_HEAD_CHECKPOINT_PREFIXES
                    )
                ]
                if legacy_action_head_keys:
                    logger.warning(
                        "Reinitializing legacy direct-action ActionDiT input/output layers "
                        "while loading latent mode: %s",
                        legacy_action_head_keys,
                    )
                    mot_state = {
                        key: value
                        for key, value in mot_state.items()
                        if key not in legacy_action_head_keys
                    }
            _validate("mot", *self.mot.load_state_dict(mot_state, strict=False))
        elif "dit" in payload:
            if strict:
                raise ValueError(
                    f"Strict HFastWAM checkpoint requires `mot`, found legacy `dit`: {path}"
                )
            logger.warning("Legacy ckpt: loading 'dit' into video_expert only.")
            _validate("video_expert (legacy dit)", *self.video_expert.load_state_dict(
                payload["dit"], strict=False,
            ))
        else:
            raise ValueError(f"Checkpoint missing both `mot` and legacy `dit` keys: {path}")
        if self.latent_action_enabled:
            if "latent_action_decoder" in payload:
                self.latent_action_decoder.load_state_dict(
                    payload["latent_action_decoder"], strict=True
                )
            elif strict or checkpoint_representation == "latent":
                raise ValueError(
                    f"Latent-action checkpoint is missing `latent_action_decoder`: {path}"
                )
            else:
                logger.warning(
                    "Checkpoint has no latent-action decoder weights; keeping random initialization."
                )
        elif "latent_action_decoder" in payload:
            raise ValueError(
                "Cannot load a latent-action decoder checkpoint into a direct-action model."
            )

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            elif strict:
                raise ValueError(f"Strict checkpoint is missing `proprio_encoder`: {path}")
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if self.use_visual_encoder and "visual_encoder" in payload:
            _validate(
                "visual_encoder",
                *self.visual_encoder.load_state_dict(
                    payload["visual_encoder"], strict=False,
                ),
                allowed_missing=("_norm_mean", "_norm_std"),
            )
        elif strict and self.use_visual_encoder:
            raise ValueError(f"Strict checkpoint is missing `visual_encoder`: {path}")
        if "training_phase" in payload:
            self._training_phase = payload["training_phase"]
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            optimizer_to(optimizer, self.device)
        return payload

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained_fastwam(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_dtype: Optional[torch.dtype] = None,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = False,
        redirect_common_files: bool = True,
        video_dit_config: dict | None = None,
        action_dit_config: dict | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        skip_video_dit_load_from_pretrain: bool = False,
        # Selects the module registered as the MoT video expert.
        video_expert_type: str = "wan_dit",
        mot_checkpoint_mixed_attn: bool = True,
        visual_encoder_config: dict | None = None,
        visual_encoder: dict | None = None,
        training_phase: str = "full",
        loss_config: dict | None = None,
        video_scheduler: dict | None = None,
        action_scheduler: dict | None = None,
        latent_action_config: dict | None = None,
        latent_action_decoder_config: dict | None = None,
        proprio_dim: int | None = None,
        knowledge_insulation: bool = True,
        action_loss_detach_video_expert: bool = False,
        strict_expert_compat: bool = True,
        layer_alignment_mode: str = "strict",
        shared_attention_expert: str = "video",
        freeze_language_expert: bool = False,
        freeze_video_expert: bool = False,
        freeze_action_expert: bool = False,
        fastwam_checkpoint: str | None = None,
        pretrain_checkpoint: str | None = None,
        fastwam_checkpoint_strict: bool = False,
        fixed_target_encoder: bool = False,
        # Language expert config
        language_backend: str = "legacy",
        language_model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        language_local_files_only: bool = False,
        language_pad_to_max_length: bool = False,
        language_vocab_size: int = 32000,
        language_ffn_dim: Optional[int] = None,
        language_max_task_len: int = 128,
        language_max_subtask_len: int = 128,
    ):
        """Build H-FastWAM: 3-expert MoT with a random-init language expert.

        The language expert gets visual grounding through MoT shared attention
        with the video expert's first-frame tokens. No separate image encoder.
        """
        if model_dtype is not None:
            torch_dtype = model_dtype
        if isinstance(video_dit_config, DictConfig):
            video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
        if not isinstance(video_dit_config, dict):
            raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

        if isinstance(action_dit_config, DictConfig):
            action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
        if action_dit_config is None:
            action_dit_config = {}
        if not isinstance(action_dit_config, dict):
            raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

        if isinstance(latent_action_config, DictConfig):
            latent_action_config = OmegaConf.to_container(latent_action_config, resolve=True)
        if latent_action_config is not None and not isinstance(latent_action_config, dict):
            raise ValueError(
                f"`latent_action_config` must resolve to a dict or null, got {type(latent_action_config)}"
            )
        if isinstance(latent_action_decoder_config, DictConfig):
            latent_action_decoder_config = OmegaConf.to_container(
                latent_action_decoder_config, resolve=True
            )
        if latent_action_decoder_config is not None and not isinstance(
            latent_action_decoder_config, dict
        ):
            raise ValueError(
                "`latent_action_decoder_config` must resolve to a dict or null, got "
                f"{type(latent_action_decoder_config)}"
            )

        if isinstance(visual_encoder_config, DictConfig):
            visual_encoder_config = OmegaConf.to_container(visual_encoder_config, resolve=True)
        if isinstance(visual_encoder, DictConfig):
            visual_encoder = OmegaConf.to_container(visual_encoder, resolve=True)
        if visual_encoder_config is None and visual_encoder is not None:
            visual_encoder_config = visual_encoder
        if visual_encoder_config is not None and not isinstance(visual_encoder_config, dict):
            raise ValueError(
                f"`visual_encoder_config` must resolve to a dict or null, got {type(visual_encoder_config)}"
            )

        if isinstance(loss_config, DictConfig):
            loss_config = OmegaConf.to_container(loss_config, resolve=True)
        if loss_config is None:
            loss_config = {}
        if not isinstance(loss_config, dict):
            raise ValueError(f"`loss_config` must resolve to a dict, got {type(loss_config)}")

        if isinstance(video_scheduler, DictConfig):
            video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
        if video_scheduler is None:
            video_scheduler = {}
        if not isinstance(video_scheduler, dict):
            raise ValueError(f"`video_scheduler` must resolve to a dict, got {type(video_scheduler)}")

        if isinstance(action_scheduler, DictConfig):
            action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
        if action_scheduler is None:
            action_scheduler = {}
        if not isinstance(action_scheduler, dict):
            raise ValueError(f"`action_scheduler` must resolve to a dict, got {type(action_scheduler)}")

        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")

        # Normalise and validate video_expert_type.
        _video_expert_type = str(video_expert_type).lower()
        if _video_expert_type not in (
            "wan_dit",
            "jepa_predictor",
            "latent_action_dit",
        ):
            raise ValueError(
                "video_expert_type must be one of "
                "{'wan_dit','jepa_predictor','latent_action_dit'}, "
                f"got {video_expert_type!r}"
            )
        if (
            _video_expert_type != "latent_action_dit"
            and "text_dim" not in video_dit_config
        ):
            raise ValueError("`video_dit_config['text_dim']` is required.")

        # Visual encoder (DINO/Qwen3-VL/V-JEPA) for video expert (optional)
        dino_visual_encoder = None
        if visual_encoder_config is not None:
            ve_cfg = dict(visual_encoder_config)
            encoder_type = ve_cfg.pop("encoder_type", "dino")
            dino_visual_encoder = build_visual_encoder(
                encoder_type=encoder_type, torch_dtype=torch_dtype, **ve_cfg,
            ).to(device=device)
            encoder_dim = int(dino_visual_encoder.z_dim)
            if _video_expert_type == "latent_action_dit":
                context_dim = int(video_dit_config.get("context_dim", -1))
                if context_dim != encoder_dim:
                    raise ValueError(
                        "Visual encoder and latent-action context dimensions "
                        f"must match: encoder={encoder_dim}, context_dim={context_dim}."
                    )
            else:
                video_in_dim = int(video_dit_config.get("in_dim", -1))
                video_out_dim = int(video_dit_config.get("out_dim", -1))
                if video_in_dim != encoder_dim or video_out_dim != encoder_dim:
                    raise ValueError(
                        "Visual encoder and video expert dimensions must match: "
                        f"encoder={encoder_dim}, in_dim={video_in_dim}, "
                        f"out_dim={video_out_dim}."
                    )

        if _video_expert_type == "jepa_predictor":
            latent_action_config, latent_action_decoder_config = (
                cls._validate_latent_action_configs(
                    latent_action_config,
                    latent_action_decoder_config,
                    action_dit_config,
                    proprio_dim,
                    int(video_dit_config["in_dim"]),
                )
            )
        elif latent_action_config is not None or latent_action_decoder_config is not None:
            raise ValueError(
                "The legacy latent_action_config/latent_action_decoder_config "
                "route is only supported with video_expert_type='jepa_predictor'."
            )

        # Wan2.2 components (VAE + tokenizer; Wan DiT only for wan_dit path).
        # Non-WAN experts pass skip_dit_build=True so
        # load_wan22_ti2v_5b_components skips both WAN DiT construction AND the
        # WAN-specific _validate_dit_config.
        _skip_dit_build = _video_expert_type in {
            "jepa_predictor",
            "latent_action_dit",
        }
        components = load_wan22_ti2v_5b_components(
            device=device, torch_dtype=torch_dtype,
            model_id=model_id, tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config if not _skip_dit_build else None,
            skip_dit_build=_skip_dit_build,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            skip_video_dit_load_from_pretrain=skip_video_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            skip_vae_load=(dino_visual_encoder is not None),
        )

        if _video_expert_type == "jepa_predictor":
            # Build JEPAPredictor from video_dit_config, filtering to accepted keys.
            _JEPA_KEYS = {
                "hidden_dim", "in_dim", "out_dim", "ffn_dim", "text_dim", "eps",
                "patch_size", "num_heads", "attn_head_dim", "num_layers",
                "video_attention_mask_mode", "action_group_causal_mask_mode",
                "use_gradient_checkpointing", "use_text_context",
            }
            jepa_cfg = {k: v for k, v in video_dit_config.items() if k in _JEPA_KEYS}
            logger.info(
                "Building JEPAPredictor from video_dit_config (keys: %s).",
                sorted(jepa_cfg),
            )
            video_expert = JEPAPredictor(**jepa_cfg).to(device=device, dtype=torch_dtype)
        elif _video_expert_type == "latent_action_dit":
            _LATENT_ACTION_KEYS = {
                "hidden_dim",
                "latent_dim",
                "latent_horizon",
                "ffn_dim",
                "context_dim",
                "context_spatial_pool",
                "freq_dim",
                "eps",
                "num_heads",
                "attn_head_dim",
                "num_layers",
                "use_gradient_checkpointing",
            }
            latent_action_cfg = {
                key: value
                for key, value in video_dit_config.items()
                if key in _LATENT_ACTION_KEYS
            }
            missing = sorted(
                _LATENT_ACTION_KEYS
                - {"use_gradient_checkpointing"}
                - set(latent_action_cfg)
            )
            if missing:
                raise ValueError(
                    "LatentActionDiT config is missing required keys: "
                    f"{missing}."
                )
            logger.info(
                "Building LatentActionDiT from video_dit_config (keys: %s).",
                sorted(latent_action_cfg),
            )
            video_expert = LatentActionDiT(**latent_action_cfg).to(
                device=device,
                dtype=torch_dtype,
            )
        else:
            video_expert = components.dit

        # Action expert (fastwam-style init)
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config or {},
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device, torch_dtype=torch_dtype,
        )
        # Strict shape matching with the video expert (legacy path).
        if strict_expert_compat:
            if int(action_expert.num_heads) != int(video_expert.num_heads):
                raise ValueError("ActionDiT num_heads must match video expert in strict mode.")
            if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError("ActionDiT attn_head_dim must match video expert in strict mode.")
            if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError("ActionDiT num_layers must match video expert in strict mode.")

        # Language expert
        lang_hidden = int(video_expert.blocks[0].hidden_dim)
        lang_ffn = int(language_ffn_dim) if language_ffn_dim is not None else int(
            video_expert.blocks[0].ffn_dim
        )
        language_tokenizer = None
        if language_backend == "legacy":
            language_expert = LanguageExpert(
                hidden_dim=lang_hidden,
                num_heads=int(video_expert.num_heads),
                attn_head_dim=int(video_expert.attn_head_dim),
                ffn_dim=lang_ffn,
                num_layers=int(len(video_expert.blocks)),
                vocab_size=int(language_vocab_size),
                max_task_len=int(language_max_task_len),
                max_subtask_len=int(language_max_subtask_len),
                eps=1e-6,
                use_gradient_checkpointing=bool(mot_checkpoint_mixed_attn),
                dtype=torch_dtype,
            ).to(device=device)
            language_tokenizer = components.tokenizer
        elif language_backend == "qwen3":
            language_expert = QwenLanguageExpert.from_pretrained_qwen(
                model_id=language_model_id,
                max_task_len=int(language_max_task_len),
                max_subtask_len=int(language_max_subtask_len),
                eps=1e-6,
                use_gradient_checkpointing=bool(mot_checkpoint_mixed_attn),
                dtype=torch_dtype,
                local_files_only=bool(language_local_files_only),
            ).to(device=device)
            if strict_expert_compat:
                if int(language_expert.num_heads) != int(video_expert.num_heads):
                    raise ValueError(
                        "language_backend='qwen3' requires strict_expert_compat=false when num_heads mismatch "
                        f"(lang={language_expert.num_heads}, video={video_expert.num_heads})."
                    )
                if int(language_expert.attn_head_dim) != int(video_expert.attn_head_dim):
                    raise ValueError(
                        "language_backend='qwen3' requires strict_expert_compat=false when attn_head_dim mismatch "
                        f"(lang={language_expert.attn_head_dim}, video={video_expert.attn_head_dim})."
                    )
                if int(len(language_expert.blocks)) != int(len(video_expert.blocks)):
                    raise ValueError(
                        "language_backend='qwen3' requires strict_expert_compat=false when num_layers mismatch "
                        f"(lang={len(language_expert.blocks)}, video={len(video_expert.blocks)})."
                    )
            try:
                from transformers import AutoTokenizer

                language_tokenizer = AutoTokenizer.from_pretrained(
                    language_model_id,
                    trust_remote_code=True,
                    local_files_only=bool(language_local_files_only),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load Qwen tokenizer (%s). Prompt-based inference may fail; "
                    "please provide task_token_ids directly.",
                    exc,
                )
        else:
            raise ValueError(
                f"Unsupported language_backend={language_backend}. Expected one of {{'legacy','qwen3'}}."
            )

        latent_action_decoder = None
        if latent_action_config:
            latent_action_decoder = LatentActionDecoder(
                **latent_action_decoder_config
            ).to(device=device, dtype=torch_dtype)

        # 3-expert MoT (order matters: language | video | action)
        mot = MoT(
            mixtures={
                "language": language_expert,
                "video": video_expert,
                "action": action_expert,
            },
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            strict_expert_compat=bool(strict_expert_compat),
            layer_alignment_mode=layer_alignment_mode,
            shared_attention_expert=shared_attention_expert,
        )

        model = cls(
            language_expert=language_expert,
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            tokenizer=components.tokenizer,
            language_tokenizer=language_tokenizer,
            language_backend=language_backend,
            language_pad_to_max_length=language_pad_to_max_length,
            text_dim=int(
                action_dit_config.get(
                    "text_dim",
                    video_dit_config.get("text_dim", 4096),
                )
                if _video_expert_type == "latent_action_dit"
                else video_dit_config["text_dim"]
            ),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
            video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
            video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
            video_train_sampling_distribution=str(
                video_scheduler.get("sampling_distribution", "shifted_uniform")
            ),
            video_train_logit_mean=float(video_scheduler.get("logit_mean", 0.0)),
            video_train_logit_std=float(video_scheduler.get("logit_std", 1.0)),
            action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
            action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
            action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
            action_train_sampling_distribution=str(
                action_scheduler.get("sampling_distribution", "shifted_uniform")
            ),
            action_train_logit_mean=float(action_scheduler.get("logit_mean", 0.0)),
            action_train_logit_std=float(action_scheduler.get("logit_std", 1.0)),
            loss_lambda_language=float(loss_config.get("lambda_language", 1.0)),
            loss_lambda_video=float(loss_config.get("lambda_video", 1.0)),
            loss_lambda_action=float(loss_config.get("lambda_action", 1.0)),
            loss_lambda_latent_action_decoder=float(
                loss_config.get("lambda_latent_action_decoder", 1.0)
            ),
            latent_action_decoder=latent_action_decoder,
            latent_action_config=latent_action_config,
            latent_action_decoder_config=latent_action_decoder_config,
            training_phase=training_phase,
            knowledge_insulation=knowledge_insulation,
            action_loss_detach_video_expert=action_loss_detach_video_expert,
            strict_expert_compat=bool(strict_expert_compat),
            freeze_language_expert=bool(freeze_language_expert),
            freeze_video_expert=bool(freeze_video_expert),
            freeze_action_expert=bool(freeze_action_expert),
            visual_encoder=dino_visual_encoder,
            video_loss_type="l1" if _video_expert_type == "jepa_predictor" else "flow_matching",
            fixed_target_encoder=bool(fixed_target_encoder),
        )

        # Optional weight-only initialization. Optimizer/scheduler/step are not restored.
        ckpt_path = fastwam_checkpoint or pretrain_checkpoint
        if ckpt_path is not None and _video_expert_type == "latent_action_dit":
            logger.info(
                "Loading latent-action initialization from: %s",
                ckpt_path,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "mot" not in ckpt:
                raise ValueError(
                    "HFastWAMLatentAction requires a checkpoint with `mot`; "
                    "legacy video-only `dit` checkpoints are not compatible."
                )
            current_mot = model.mot.state_dict()
            source_mot = ckpt["mot"]
            action_prefix = "mixtures.action."
            expected_action_keys = {
                key for key in current_mot if key.startswith(action_prefix)
            }
            compatible = {
                key: value
                for key, value in source_mot.items()
                if not key.startswith("mixtures.video.")
                and key in current_mot
                and tuple(value.shape) == tuple(current_mot[key].shape)
            }
            loaded_action_keys = {
                key for key in compatible if key.startswith(action_prefix)
            }
            missing_action_keys = sorted(expected_action_keys - loaded_action_keys)
            if missing_action_keys:
                raise ValueError(
                    "Latent-action initialization requires a complete, "
                    "shape-compatible ActionDiT in the source MoT; "
                    f"missing={missing_action_keys[:20]}."
                )
            model.mot.load_state_dict(compatible, strict=False)
            if model.proprio_encoder is not None and "proprio_encoder" in ckpt:
                model.proprio_encoder.load_state_dict(
                    ckpt["proprio_encoder"],
                    strict=True,
                )
            if model.use_visual_encoder and "visual_encoder" in ckpt:
                model.visual_encoder.load_state_dict(
                    ckpt["visual_encoder"],
                    strict=False,
                )
            del ckpt
        elif ckpt_path is not None:
            logger.info(
                "Loading fastwam pretrain checkpoint: %s (strict=%s)",
                ckpt_path,
                fastwam_checkpoint_strict,
            )
            model.load_checkpoint(
                ckpt_path,
                optimizer=None,
                strict=bool(fastwam_checkpoint_strict),
            )

        if model.fixed_target_encoder_enabled:
            model._initialize_fixed_target_encoder(ckpt_path)

        return model

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: Optional[int] = None,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        generate_subtask: bool = False,
        **kwargs,
    ) -> dict:
        """FastWAM-compatible eval inference.

        In VAE mode this returns both decoded rollout frames and predicted
        actions. In visual-encoder mode there is no decoder, so it falls back
        to action-only inference.
        """
        del action, context, context_mask, negative_prompt, text_cfg_scale, action_cfg_scale

        if self.use_visual_encoder or num_frames is None:
            return self.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                generate_subtask=generate_subtask,
                num_video_frames=num_frames,
                **kwargs,
            )

        if action_horizon is None or int(action_horizon) <= 0:
            raise ValueError(f"`action_horizon` must be a positive integer, got {action_horizon}")
        action_horizon = int(action_horizon)
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be a positive integer, got {num_frames}")
        _vmask2 = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if not self.is_jepa_predictor and _vmask2 != "first_frame_causal":
            raise ValueError(
                "HFastWAM video inference requires video_attention_mask_mode='first_frame_causal'; "
                f"got '{_vmask2}'."
            )

        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)

        task_token_ids = kwargs.pop("task_token_ids", None)
        subtask_token_ids = kwargs.pop("subtask_token_ids", None)
        bos_token_id = int(kwargs.pop("bos_token_id", 1))
        eos_token_id = int(kwargs.pop("eos_token_id", 2))
        max_subtask_tokens = int(kwargs.pop("max_subtask_tokens", 64))
        if kwargs:
            logger.debug("Ignoring unused HFastWAM infer kwargs: %s", sorted(kwargs.keys()))

        if task_token_ids is None:
            if prompt is None:
                raise ValueError("Either `prompt` or `task_token_ids` must be provided.")
            task_token_ids = self._tokenize_task_prompt(prompt)
        elif not torch.is_tensor(task_token_ids):
            task_token_ids = torch.as_tensor(task_token_ids, dtype=torch.long)
        if task_token_ids.ndim == 1:
            task_token_ids = task_token_ids.unsqueeze(0)
        elif task_token_ids.ndim != 2:
            raise ValueError(f"`task_token_ids` must be 1D/2D, got {tuple(task_token_ids.shape)}")
        task_token_ids = task_token_ids.to(self.device, dtype=torch.long)

        if subtask_token_ids is not None:
            if not torch.is_tensor(subtask_token_ids):
                subtask_token_ids = torch.as_tensor(subtask_token_ids, dtype=torch.long)
            if subtask_token_ids.ndim == 1:
                subtask_token_ids = subtask_token_ids.unsqueeze(0)
            elif subtask_token_ids.ndim != 2:
                raise ValueError(f"`subtask_token_ids` must be 1D/2D, got {tuple(subtask_token_ids.shape)}")
            subtask_token_ids = subtask_token_ids.to(self.device, dtype=torch.long)

        first_frame_latents = self._encode_first_frame(input_image, tiled=tiled)
        _, z_dim, _, latent_h, latent_w = first_frame_latents.shape
        temporal_factor = int(getattr(self.visual_encoder, "temporal_downsample_factor", 4))
        latent_t = (num_frames - 1) // temporal_factor + 1
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        proprio_ctx, proprio_mask = self._make_proprio_text_context(
            proprio,
            batch_size=1,
            source="proprio",
        )
        has_proprio_context = proprio_ctx is not None
        if has_proprio_context:
            video_context, video_context_mask = proprio_ctx, proprio_mask
            action_context, action_context_mask = proprio_ctx, proprio_mask
        else:
            video_context, video_context_mask = self._make_dummy_text_context(1)
            action_context, action_context_mask = video_context, video_context_mask

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = first_frame_latents
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        timestep_zero = torch.zeros((1,), dtype=latents_video.dtype, device=self.device)
        video_pre_ff = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_zero,
            context=video_context,
            context_mask=video_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_tokens_per_frame = int(video_pre_ff["meta"]["tokens_per_frame"])
        video_context_payload_ff = self._context_payload_from_pre_state(video_pre_ff, has_proprio_context)

        # Train/infer alignment: default to an EMPTY subtask (subtask_len=0),
        # matching task-only training. Only AR-generate a subtask when explicitly
        # requested via `generate_subtask=True` (or pass `subtask_token_ids`).
        if subtask_token_ids is not None:
            subtask_ids_used = subtask_token_ids
        elif generate_subtask:
            subtask_ids_used = self._generate_subtask(
                task_ids=task_token_ids,
                video_pre=video_pre_ff,
                video_tokens_per_frame=video_tokens_per_frame,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_subtask_tokens,
                video_context_payload=video_context_payload_ff,
            )
        else:
            subtask_ids_used = torch.empty(
                (task_token_ids.shape[0], 0), dtype=torch.long, device=self.device,
            )

        lang_pre = self.language_expert.pre_dit(
            task_token_ids=task_token_ids,
            subtask_token_ids=subtask_ids_used,
        )
        seg = lang_pre["segments"]

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=video_context,
                context_mask=video_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
            )
            video_context_payload = self._context_payload_from_pre_state(video_pre, has_proprio_context)
            action_context_payload = self._context_payload_from_pre_state(action_pre, has_proprio_context)
            tokens_out = self._run_mot_three_experts(
                lang_pre=lang_pre,
                video_pre=video_pre,
                action_pre=action_pre,
                task_len=seg["task_len"],
                subtask_len=seg["subtask_len"],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                video_context_payload=video_context_payload,
                action_context_payload=action_context_payload,
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "subtask_tokens": subtask_ids_used[0].detach().cpu(),
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
