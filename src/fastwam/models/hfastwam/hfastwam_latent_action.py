from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.models.dreamdojo_lam import (
    encode_dreamdojo_latent_actions,
    load_dreamdojo_lam,
)
from fastwam.models.wan22.latent_action_dit import LatentActionDiT

from .hfastwam import HFastWAM


class HFastWAMLatentAction(HFastWAM):
    """H-FastWAM with DreamDojo latent actions replacing video targets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_dreamdojo_lam", None)
        self.dreamdojo_pair_batch_size = 32
        self.dreamdojo_model_dtype = torch.bfloat16
        self.dreamdojo_provenance = None
        if not isinstance(self.video_expert, LatentActionDiT):
            raise TypeError(
                "HFastWAMLatentAction requires video_expert=LatentActionDiT, "
                f"got {type(self.video_expert).__name__}."
            )
        if not self.use_visual_encoder:
            raise ValueError(
                "HFastWAMLatentAction requires a frozen visual encoder for "
                "first-frame context."
            )
        if not bool(getattr(self.visual_encoder, "_freeze_backbone", False)):
            raise ValueError(
                "HFastWAMLatentAction requires visual_encoder.freeze_backbone=true."
            )

    @classmethod
    def from_pretrained_fastwam(
        cls,
        dreamdojo_config: Optional[dict] = None,
        **kwargs,
    ):
        model = super(HFastWAMLatentAction, cls).from_pretrained_fastwam(
            **kwargs
        )
        model._initialization_checkpoint_path = (
            kwargs.get("fastwam_checkpoint")
            or kwargs.get("pretrain_checkpoint")
        )
        config = dict(dreamdojo_config or {})
        enabled = bool(config.pop("enabled", False))
        if not enabled:
            if config:
                raise ValueError(
                    "DreamDojo settings were provided while "
                    "`dreamdojo_config.enabled=false`."
                )
            return model

        root = config.pop("root", None)
        checkpoint = config.pop("checkpoint", None)
        pair_batch_size = int(config.pop("pair_batch_size", 32))
        dtype_name = str(config.pop("dtype", "bfloat16"))
        checkpoint_sha256 = config.pop("checkpoint_sha256", None)
        source_revision = config.pop("source_revision", None)
        if config:
            raise ValueError(
                f"Unknown DreamDojo config fields: {sorted(config)}."
            )
        if root is None or checkpoint is None:
            raise ValueError(
                "Online DreamDojo targets require both "
                "`dreamdojo_config.root` and `dreamdojo_config.checkpoint`."
            )
        if checkpoint_sha256 in (None, "") or source_revision in (None, ""):
            raise ValueError(
                "Online DreamDojo targets require `checkpoint_sha256` and "
                "`source_revision` provenance."
            )
        if pair_batch_size <= 0:
            raise ValueError(
                "`dreamdojo_config.pair_batch_size` must be positive."
            )
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_by_name:
            raise ValueError(
                "`dreamdojo_config.dtype` must be one of "
                f"{sorted(dtype_by_name)}, got {dtype_name!r}."
            )
        dreamdojo_dtype = dtype_by_name[dtype_name]
        lam = load_dreamdojo_lam(
            root,
            checkpoint,
            device=torch.device(model.device),
            dtype=dreamdojo_dtype,
            expected_checkpoint_sha256=str(checkpoint_sha256),
            expected_source_revision=str(source_revision),
        )
        # Keep the frozen target encoder outside the registered module tree so
        # ZeRO and FastWAM checkpoints do not replicate/save its 8.5 GB weights.
        object.__setattr__(model, "_dreamdojo_lam", lam)
        model.dreamdojo_pair_batch_size = pair_batch_size
        model.dreamdojo_model_dtype = dreamdojo_dtype
        model.dreamdojo_provenance = {
            "checkpoint_sha256": str(checkpoint_sha256),
            "source_revision": str(source_revision),
            "target_mode": "online_adjacent_pairs",
            "normalization": "none",
        }
        model._validate_initialization_checkpoint()
        return model

    def _validate_initialization_checkpoint(self) -> None:
        initialization_checkpoint = getattr(
            self,
            "_initialization_checkpoint_path",
            None,
        )
        if initialization_checkpoint is None or self.dreamdojo_provenance is None:
            return
        payload = torch.load(
            initialization_checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        metadata = payload.get("checkpoint_metadata")
        if (
            isinstance(metadata, dict)
            and metadata.get("video_target_representation") is not None
        ):
            self._validate_checkpoint_metadata(
                metadata,
                strict=False,
                path=str(initialization_checkpoint),
            )
        mot_state = payload.get("mot")
        if (
            not isinstance(metadata, dict)
            or metadata.get("video_target_representation") is None
        ) and isinstance(mot_state, dict) and any(
            key.startswith("mixtures.video.action_encoder.")
            for key in mot_state
        ):
            raise ValueError(
                "Cannot initialize DreamDojo training from a latent-action "
                "checkpoint without target-space provenance. Use the original "
                "VAE-DiT H-FastWAM checkpoint instead."
            )

    def _checkpoint_metadata(self) -> dict:
        metadata = super()._checkpoint_metadata()
        if self.dreamdojo_provenance is not None:
            metadata["video_target_representation"] = "dreamdojo_latent_action"
            metadata["dreamdojo_target"] = dict(self.dreamdojo_provenance)
        return metadata

    def set_latent_action_cache_manifest(self, manifest: dict) -> None:
        if not isinstance(manifest, dict):
            raise TypeError("Latent-action cache manifest must be a dict.")
        if (
            int(manifest.get("latent_horizon", -1)) != 32
            or int(manifest.get("latent_dim", -1)) != 32
        ):
            raise ValueError(
                "HFastWAMLatentAction requires cached targets with shape "
                "[32,32]."
            )
        required = (
            "dreamdojo_checkpoint_sha256",
            "dreamdojo_source_sha256",
            "implementation_sha256",
            "dataset_manifest_sha256",
            "dataset_index_fingerprint",
            "mean",
            "std",
        )
        missing = [key for key in required if manifest.get(key) is None]
        if missing:
            raise ValueError(
                "Latent-action cache manifest is missing provenance fields: "
                f"{missing}."
            )
        self.dreamdojo_provenance = {
            "checkpoint_sha256": manifest["dreamdojo_checkpoint_sha256"],
            "source_revision": manifest.get("dreamdojo_source_revision"),
            "source_sha256": manifest["dreamdojo_source_sha256"],
            "implementation_sha256": manifest["implementation_sha256"],
            "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "dataset_index_fingerprint": manifest[
                "dataset_index_fingerprint"
            ],
            "target_mode": "cached_normalized_adjacent_pairs",
            "normalization": {
                "mean": manifest["mean"],
                "std": manifest["std"],
            },
        }
        self._validate_initialization_checkpoint()

    def _validate_checkpoint_metadata(
        self,
        metadata: Optional[dict],
        *,
        strict: bool,
        path: str,
    ) -> None:
        del strict
        if self.dreamdojo_provenance is None:
            return
        if not isinstance(metadata, dict):
            raise ValueError(
                "Online DreamDojo resume requires checkpoint metadata: "
                f"{path}"
            )
        if (
            metadata.get("video_target_representation")
            != "dreamdojo_latent_action"
        ):
            raise ValueError(
                "Checkpoint was not trained with online DreamDojo latent "
                f"targets: {path}"
            )
        actual = metadata.get("dreamdojo_target")
        if actual != self.dreamdojo_provenance:
            raise ValueError(
                "DreamDojo target provenance mismatch while resuming: "
                f"checkpoint={actual!r}, configured={self.dreamdojo_provenance!r}."
            )

    def _encode_online_latent_actions(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        lam = self.__dict__.get("_dreamdojo_lam")
        if lam is None:
            raise RuntimeError(
                "The sample has no cached `latent_actions`, and the online "
                "DreamDojo LAM is not configured."
            )
        if (
            self.video_expert.latent_horizon != 32
            or self.video_expert.latent_dim != 32
        ):
            raise ValueError(
                "Online DreamDojo targets require a [32,32] latent expert."
            )
        latent_actions = encode_dreamdojo_latent_actions(
            lam,
            video,
            pair_batch_size=self.dreamdojo_pair_batch_size,
            device=torch.device(self.device),
            model_dtype=self.dreamdojo_model_dtype,
        )
        return latent_actions.to(
            device=self.device,
            dtype=self.torch_dtype,
        )

    def _validate_latent_actions(
        self,
        latent_actions: torch.Tensor,
        *,
        source: str,
    ) -> torch.Tensor:
        if not torch.is_tensor(latent_actions):
            raise TypeError(f"`{source}` must be a tensor.")
        if latent_actions.ndim != 3:
            raise ValueError(
                f"`{source}` must be [B,T,D], got {tuple(latent_actions.shape)}."
            )
        expected = (
            self.video_expert.latent_horizon,
            self.video_expert.latent_dim,
        )
        if tuple(latent_actions.shape[1:]) != expected:
            raise ValueError(
                f"`{source}` must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(latent_actions.shape)}."
            )
        if not bool(torch.isfinite(latent_actions).all().item()):
            raise ValueError(f"`{source}` contains non-finite values.")
        return latent_actions.to(device=self.device, dtype=self.torch_dtype)

    def _prepare_first_frame_context(
        self,
        video: torch.Tensor,
        *,
        source: str,
    ) -> torch.Tensor:
        if not torch.is_tensor(video) or video.ndim != 5:
            raise ValueError(
                f"`{source}` must be [B,3,T,H,W], "
                f"got {type(video)} with shape {getattr(video, 'shape', None)}."
            )
        if video.shape[1] != 3 or video.shape[2] < 1:
            raise ValueError(
                f"`{source}` must contain at least one RGB frame, "
                f"got {tuple(video.shape)}."
            )
        return self._encode_first_frame(video[:, :, 0])

    def _compute_latent_action_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        latent_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "Latent-action prediction/target shape mismatch: "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}."
            )
        element_loss = F.mse_loss(
            pred.float(),
            target.float(),
            reduction="none",
        )
        valid = torch.ones_like(element_loss, dtype=torch.bool)
        if latent_is_pad is not None:
            latent_is_pad = latent_is_pad.to(
                device=element_loss.device,
                dtype=torch.bool,
            )
            if latent_is_pad.shape != element_loss.shape[:2]:
                raise ValueError(
                    "`latent_is_pad` must match [B,T]: "
                    f"got {tuple(latent_is_pad.shape)}, "
                    f"expected {tuple(element_loss.shape[:2])}."
                )
            valid &= ~latent_is_pad.unsqueeze(-1)
        valid_float = valid.to(dtype=element_loss.dtype)
        valid_count = valid_float.sum(dim=(1, 2)).clamp_min(1.0)
        per_sample = (element_loss * valid_float).sum(dim=(1, 2)) / valid_count
        weight = self.train_video_scheduler.training_weight(timestep).to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        return (per_sample * weight).mean()

    def _prepare_language(
        self,
        sample: dict,
        *,
        batch_size: int,
    ) -> tuple[Optional[dict], Optional[torch.Tensor], Optional[torch.Tensor]]:
        sample = self._ensure_language_tokens_from_prompt(sample)
        task_ids = sample.get("task_token_ids")
        if task_ids is None:
            return None, None, None
        task_ids = task_ids.to(self.device, dtype=torch.long)
        if task_ids.ndim != 2 or task_ids.shape[0] != batch_size:
            raise ValueError(
                "`task_token_ids` must be [B,L] with the modality batch size, "
                f"got {tuple(task_ids.shape)} for B={batch_size}."
            )
        subtask_ids = sample.get("subtask_token_ids")
        if subtask_ids is None:
            subtask_ids = torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=self.device,
            )
        else:
            subtask_ids = subtask_ids.to(self.device, dtype=torch.long)
        return (
            self.language_expert.pre_dit(
                task_token_ids=task_ids,
                subtask_token_ids=subtask_ids,
            ),
            task_ids,
            subtask_ids,
        )

    def _language_loss(
        self,
        tokens: torch.Tensor,
        lang_pre: dict,
        task_ids: torch.Tensor,
        subtask_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.loss_lambda_language == 0.0:
            return None
        lang_output = self.language_expert.post_dit(tokens, lang_pre)
        task_len = int(lang_pre["segments"]["task_len"])
        losses = []
        if task_ids.shape[1] > 1:
            task_logits = self.language_expert.lm_head(
                lang_output.hidden_states[:, :task_len]
            )
            losses.append(self._compute_language_token_loss(task_logits, task_ids))
        if subtask_ids.shape[1] > 1:
            losses.append(
                self.language_expert.language_loss(
                    logits=lang_output.logits,
                    subtask_token_ids=subtask_ids,
                )
            )
        return torch.stack(losses).mean() if losses else None

    def _training_loss_flat(
        self,
        sample: dict,
    ) -> tuple[torch.Tensor, dict]:
        video = sample.get("video")
        if not torch.is_tensor(video) or video.ndim != 5:
            raise ValueError(
                "Latent-action training requires raw video [B,3,T,H,W], "
                f"got {type(video)} with shape "
                f"{getattr(video, 'shape', None)}."
            )
        batch_size = int(video.shape[0])
        latent_actions = sample.get("latent_actions")
        if (
            latent_actions is not None
            and self.__dict__.get("_dreamdojo_lam") is not None
        ):
            raise ValueError(
                "Online DreamDojo training does not accept cached "
                "`latent_actions`; remove the latent-action cache override."
            )
        if latent_actions is None:
            latent_actions = self._encode_online_latent_actions(video)
        else:
            latent_actions = self._validate_latent_actions(
                latent_actions,
                source="sample['latent_actions']",
            )
            if int(latent_actions.shape[0]) != batch_size:
                raise ValueError(
                    "Raw video and latent-action batch sizes differ: "
                    f"{batch_size} vs {int(latent_actions.shape[0])}."
                )
        visual_context = self._prepare_first_frame_context(
            video,
            source="sample['video']",
        )
        lang_pre, task_ids, subtask_ids = self._prepare_language(
            sample,
            batch_size=batch_size,
        )

        noise_latent = torch.randn_like(latent_actions)
        timestep_latent = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=latent_actions.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(
            latent_actions,
            noise_latent,
            timestep_latent,
        )
        target_latent = self.train_video_scheduler.training_target(
            latent_actions,
            noise_latent,
            timestep_latent,
        )
        latent_pre = self.video_expert.pre_dit(
            latent_tokens=noisy_latent,
            timestep=timestep_latent,
            context_latents=visual_context,
        )
        latent_context_payload = self._context_payload_from_pre_state(
            latent_pre,
            True,
        )

        action = sample.get("action")
        action_pre = None
        target_action = None
        timestep_action = None
        action_context_payload = None
        if action is not None:
            action = action.to(device=self.device, dtype=self.torch_dtype)
            if action.ndim != 3 or action.shape[0] != batch_size:
                raise ValueError(
                    f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}."
                )
            action_dim_is_pad = sample.get("action_dim_is_pad")
            action = self._zero_padded_action_dims(action, action_dim_is_pad)
            noise_action = self._zero_padded_action_dims(
                torch.randn_like(action),
                action_dim_is_pad,
            )
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(
                action,
                noise_action,
                timestep_action,
            )
            target_action = self.train_action_scheduler.training_target(
                action,
                noise_action,
                timestep_action,
            )
            action_context, action_context_mask = self._make_proprio_text_context(
                sample.get("proprio"),
                batch_size=batch_size,
                source="sample['proprio']",
            )
            has_action_context = action_context is not None
            if not has_action_context:
                action_context, action_context_mask = (
                    self._make_dummy_text_context(batch_size)
                )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
            )
            action_context_payload = self._context_payload_from_pre_state(
                action_pre,
                has_action_context,
            )

        task_len = 0 if lang_pre is None else int(lang_pre["segments"]["task_len"])
        subtask_len = (
            0 if lang_pre is None else int(lang_pre["segments"]["subtask_len"])
        )
        if lang_pre is not None and action_pre is not None:
            tokens_out = self._run_mot_three_experts(
                lang_pre=lang_pre,
                video_pre=latent_pre,
                action_pre=action_pre,
                task_len=task_len,
                subtask_len=subtask_len,
                video_tokens_per_frame=self.video_expert.latent_horizon,
                video_context_payload=latent_context_payload,
                action_context_payload=action_context_payload,
            )
        elif lang_pre is not None:
            tokens_out = self._run_mot_two_experts_lv(
                lang_pre=lang_pre,
                video_pre=latent_pre,
                task_len=task_len,
                subtask_len=subtask_len,
                video_tokens_per_frame=self.video_expert.latent_horizon,
                video_context_payload=latent_context_payload,
            )
        elif action_pre is not None:
            tokens_out = self._run_mot_two_experts_va(
                video_pre=latent_pre,
                action_pre=action_pre,
                video_tokens_per_frame=self.video_expert.latent_horizon,
                video_context_payload=latent_context_payload,
                action_context_payload=action_context_payload,
            )
        else:
            raise ValueError(
                "Latent-action training requires language or raw action tokens."
            )

        pred_latent = self.video_expert.post_dit(
            tokens_out["video"],
            latent_pre,
        )
        loss_latent = self._compute_latent_action_loss(
            pred_latent,
            target_latent,
            timestep_latent,
            sample.get("action_is_pad"),
        )
        total_loss = self.loss_lambda_video * loss_latent
        loss_dict = {
            "loss_latent_action": self.loss_lambda_video
            * float(loss_latent.detach().item())
        }

        if lang_pre is not None:
            language_loss = self._language_loss(
                tokens_out["language"],
                lang_pre,
                task_ids,
                subtask_ids,
            )
            if language_loss is None:
                loss_dict["loss_language"] = 0.0
            else:
                total_loss = (
                    total_loss + self.loss_lambda_language * language_loss
                )
                loss_dict["loss_language"] = self.loss_lambda_language * float(
                    language_loss.detach().item()
                )

        if action_pre is not None:
            pred_action = self.action_expert.post_dit(
                tokens_out["action"],
                action_pre,
            )
            loss_action = self._compute_action_loss(
                pred_action=pred_action,
                target_action=target_action,
                timestep_action=timestep_action,
                action_is_pad=sample.get("action_is_pad"),
                action_dim_is_pad=sample.get("action_dim_is_pad"),
            )
            total_loss = total_loss + self.loss_lambda_action * loss_action
            loss_dict["loss_action"] = self.loss_lambda_action * float(
                loss_action.detach().item()
            )
        return total_loss, loss_dict

    def _training_loss_interleaved(
        self,
        sample: dict,
    ) -> tuple[torch.Tensor, dict]:
        segments = sample["segments"]
        if isinstance(segments, list):
            segments = self._segment_list_to_dict(segments)
        if not isinstance(segments, dict):
            raise TypeError("`sample['segments']` must be a dict or list[dict].")

        video = segments.get("video")
        if not torch.is_tensor(video):
            raise ValueError(
                "Interleaved latent-action training requires "
                "`segments['video']`."
            )
        latent_actions = segments.get("latent_actions")
        if video.ndim == 5:
            video = video.unsqueeze(0)
            num_segments = int(video.shape[1])
            segments = {
                key: (
                    value.unsqueeze(0)
                    if torch.is_tensor(value)
                    and value.ndim >= 1
                    and value.shape[0] == num_segments
                    else value
                )
                for key, value in segments.items()
            }
            segments["video"] = video
            latent_actions = segments.get("latent_actions")
        if video.ndim != 6:
            raise ValueError(
                "Interleaved video must be [B,N,3,T,H,W], "
                f"got {tuple(video.shape)}."
            )
        batch_size, num_segments = video.shape[:2]
        if latent_actions is not None:
            if not torch.is_tensor(latent_actions) or latent_actions.ndim != 4:
                raise ValueError(
                    "Cached interleaved latent actions must be [B,N,T,D], "
                    f"got {type(latent_actions)} with shape "
                    f"{getattr(latent_actions, 'shape', None)}."
                )
            if tuple(latent_actions.shape[:2]) != (
                batch_size,
                num_segments,
            ):
                raise ValueError(
                    "Interleaved video/latent-action leading dims differ."
                )
        segment_mask = segments.get("segment_mask")
        if segment_mask is not None:
            segment_mask = segment_mask.to(dtype=torch.bool)
            if segment_mask.ndim == 1 and batch_size == 1:
                segment_mask = segment_mask.unsqueeze(0)
            if segment_mask.shape != (batch_size, num_segments):
                raise ValueError(
                    "`segment_mask` must be [B,N], "
                    f"got {tuple(segment_mask.shape)}."
                )
            if not bool(segment_mask.all().item()):
                raise ValueError(
                    "Padded interleaved segments are not supported by the "
                    "latent-action training route."
                )

        flat_sample: dict[str, Any] = {}
        for key, value in segments.items():
            if key == "segment_mask":
                continue
            if torch.is_tensor(value):
                if value.ndim >= 2 and tuple(value.shape[:2]) == (
                    batch_size,
                    num_segments,
                ):
                    flat_sample[key] = value.flatten(0, 1)
                else:
                    flat_sample[key] = value
            else:
                flat_sample[key] = value
        if "prompt" in segments:
            flat_sample["prompt"] = self._flatten_segment_prompts(
                segments["prompt"],
                batch_size=batch_size,
                num_segments=num_segments,
            )
        return self._training_loss_flat(flat_sample)

    def training_loss(
        self,
        sample: dict,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        del tiled
        if "segments" in sample:
            return self._training_loss_interleaved(sample)
        return self._training_loss_flat(sample)

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
        *,
        prompt: Optional[str] = None,
        input_image: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        generate_subtask: bool = False,
        **kwargs,
    ) -> dict:
        del kwargs, tiled
        if generate_subtask:
            raise ValueError(
                "Latent-action inference does not support subtask generation; "
                "provide subtask_token_ids explicitly or use the default empty "
                "subtask stream."
            )
        if action_horizon is None or int(action_horizon) <= 0:
            raise ValueError(
                f"`action_horizon` must be positive, got {action_horizon}."
            )
        image = image if image is not None else input_image
        if image is None:
            raise ValueError("Either `image` or `input_image` is required.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError(
                f"Latent-action inference requires image [1,3,H,W], "
                f"got {tuple(image.shape)}."
            )
        image = image.to(device=self.device, dtype=self.torch_dtype)
        visual_context = self._encode_first_frame(image)

        if task_token_ids is None:
            if prompt is None:
                raise ValueError(
                    "Either `task_token_ids` or `prompt` is required."
                )
            task_token_ids = self._tokenize_task_prompt(prompt)
        elif not torch.is_tensor(task_token_ids):
            task_token_ids = torch.as_tensor(
                task_token_ids,
                dtype=torch.long,
            )
        if task_token_ids.ndim == 1:
            task_token_ids = task_token_ids.unsqueeze(0)
        if task_token_ids.ndim != 2 or task_token_ids.shape[0] != 1:
            raise ValueError(
                "`task_token_ids` must be [1,L], "
                f"got {tuple(task_token_ids.shape)}."
            )
        task_token_ids = task_token_ids.to(self.device, dtype=torch.long)
        if subtask_token_ids is None:
            subtask_token_ids = torch.empty(
                (1, 0),
                dtype=torch.long,
                device=self.device,
            )
        else:
            subtask_token_ids = torch.as_tensor(
                subtask_token_ids,
                dtype=torch.long,
                device=self.device,
            )
            if subtask_token_ids.ndim == 1:
                subtask_token_ids = subtask_token_ids.unsqueeze(0)
        lang_pre = self.language_expert.pre_dit(
            task_token_ids=task_token_ids,
            subtask_token_ids=subtask_token_ids,
        )

        action_context, action_context_mask = self._make_proprio_text_context(
            proprio,
            batch_size=1,
            source="proprio",
        )
        has_action_context = action_context is not None
        if not has_action_context:
            action_context, action_context_mask = self._make_dummy_text_context(1)

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latent_actions = torch.randn(
            (
                1,
                self.video_expert.latent_horizon,
                self.video_expert.latent_dim,
            ),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        actions = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        latent_timesteps, latent_deltas = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latent_actions.dtype,
                shift_override=sigma_shift,
            )
        )
        action_timesteps, action_deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=actions.dtype,
                shift_override=sigma_shift,
            )
        )
        for latent_t, latent_delta, action_t, action_delta in zip(
            latent_timesteps,
            latent_deltas,
            action_timesteps,
            action_deltas,
        ):
            latent_pre = self.video_expert.pre_dit(
                latent_tokens=latent_actions,
                timestep=latent_t.unsqueeze(0),
                context_latents=visual_context,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=actions,
                timestep=action_t.unsqueeze(0),
                context=action_context,
                context_mask=action_context_mask,
            )
            tokens_out = self._run_mot_three_experts(
                lang_pre=lang_pre,
                video_pre=latent_pre,
                action_pre=action_pre,
                task_len=int(lang_pre["segments"]["task_len"]),
                subtask_len=int(lang_pre["segments"]["subtask_len"]),
                video_tokens_per_frame=self.video_expert.latent_horizon,
                video_context_payload=self._context_payload_from_pre_state(
                    latent_pre,
                    True,
                ),
                action_context_payload=self._context_payload_from_pre_state(
                    action_pre,
                    has_action_context,
                ),
            )
            pred_latent = self.video_expert.post_dit(
                tokens_out["video"],
                latent_pre,
            )
            pred_action = self.action_expert.post_dit(
                tokens_out["action"],
                action_pre,
            )
            latent_actions = self.infer_video_scheduler.step(
                pred_latent,
                latent_delta,
                latent_actions,
            )
            actions = self.infer_action_scheduler.step(
                pred_action,
                action_delta,
                actions,
            )
        return {
            "action": actions[0].detach().float().cpu(),
            "latent_action": latent_actions[0].detach().float().cpu(),
            "subtask_tokens": subtask_token_ids[0].detach().cpu(),
        }
