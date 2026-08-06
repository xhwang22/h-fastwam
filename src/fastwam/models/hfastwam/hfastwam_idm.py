"""Inverse-dynamics H-FastWAM with a deterministic JEPA world model."""

from __future__ import annotations

from typing import Optional

import torch

from .hfastwam import HFastWAM
from .language_expert import LanguageExpert


class HFastWAMIDM(HFastWAM):
    """Train action denoising from ground-truth future JEPA latents.

    Each temporal segment contains two video branches backed by the same
    JEPAPredictor: a context branch used for next-latent prediction and a
    teacher-forcing future branch used only as the action condition.
    """

    def _build_training_condition_latents(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        del context_latents
        return target_latents

    def _build_inference_condition_latents(
        self,
        first_frame_latents: torch.Tensor,
        predicted_future_latents: torch.Tensor,
    ) -> torch.Tensor:
        del first_frame_latents
        return predicted_future_latents

    @staticmethod
    def _combine_idm_video_pre(
        prediction_pre: dict,
        condition_pre: dict,
    ) -> dict:
        if prediction_pre["meta"]["tokens_per_frame"] != condition_pre["meta"]["tokens_per_frame"]:
            raise ValueError("IDM prediction and future-condition branches must share the same spatial grid.")

        mot_pre = dict(prediction_pre)
        mot_pre["tokens"] = torch.cat(
            [prediction_pre["tokens"], condition_pre["tokens"]],
            dim=1,
        )
        mot_pre["freqs"] = torch.cat(
            [prediction_pre["freqs"], condition_pre["freqs"]],
            dim=0,
        )
        if prediction_pre["context_mask"] is not None:
            mot_pre["context_mask"] = torch.cat(
                [prediction_pre["context_mask"], condition_pre["context_mask"]],
                dim=1,
            )
        mot_pre["_idm_prediction_tokens"] = int(prediction_pre["tokens"].shape[1])
        mot_pre["_idm_condition_tokens"] = int(condition_pre["tokens"].shape[1])
        return mot_pre

    def _prepare_jepa_training_video_pre(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[dict, dict, int]:
        if not self.is_jepa_predictor:
            raise ValueError("HFastWAMIDM requires a JEPAPredictor video expert.")

        prediction_pre, _, prediction_tokens = super()._prepare_jepa_training_video_pre(
            context_latents=context_latents,
            target_latents=target_latents,
            context=context,
            context_mask=context_mask,
        )
        condition_latents = self._build_training_condition_latents(
            context_latents=context_latents,
            target_latents=target_latents,
        )
        condition_pre = self.video_expert.pre_dit(
            x=condition_latents,
            context=context if self.video_expert.use_text_context else None,
            context_mask=context_mask if self.video_expert.use_text_context else None,
        )
        mot_pre = self._combine_idm_video_pre(prediction_pre, condition_pre)
        return mot_pre, prediction_pre, int(prediction_tokens)

    @torch.no_grad()
    def _build_interleaved_idm_attention_mask(
        self,
        task_len: int,
        subtask_len: int,
        prediction_video_seq_len: int,
        condition_video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        num_segments: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build segment-causal IDM attention without future/action leakage."""
        lang_seg = int(task_len) + int(subtask_len)
        pred_seg = int(prediction_video_seq_len)
        cond_seg = int(condition_video_seq_len)
        video_seg = pred_seg + cond_seg
        action_seg = int(action_seq_len)
        num_segments = int(num_segments)

        lang_total = num_segments * lang_seg
        video_total = num_segments * video_seg
        action_total = num_segments * action_seg
        mask = torch.zeros(
            (lang_total + video_total + action_total,) * 2,
            dtype=torch.bool,
            device=device,
        )

        def lang_range(segment_idx: int) -> tuple[int, int]:
            start = segment_idx * lang_seg
            return start, start + lang_seg

        def prediction_range(segment_idx: int) -> tuple[int, int]:
            start = lang_total + segment_idx * video_seg
            return start, start + pred_seg

        def condition_range(segment_idx: int) -> tuple[int, int]:
            start = lang_total + segment_idx * video_seg + pred_seg
            return start, start + cond_seg

        def action_range(segment_idx: int) -> tuple[int, int]:
            start = lang_total + video_total + segment_idx * action_seg
            return start, start + action_seg

        def modality_ranges(segment_idx: int) -> list[tuple[int, int]]:
            ranges = []
            if lang_seg:
                ranges.append(lang_range(segment_idx))
            if pred_seg:
                ranges.append(prediction_range(segment_idx))
            if cond_seg:
                ranges.append(condition_range(segment_idx))
            if action_seg:
                ranges.append(action_range(segment_idx))
            return ranges

        first_frame_tokens = min(int(video_tokens_per_frame), pred_seg)
        for segment_idx in range(num_segments):
            current_ranges = modality_ranges(segment_idx)
            previous_ranges = [
                token_range
                for previous_idx in range(segment_idx)
                for token_range in modality_ranges(previous_idx)
            ]
            for row_start, row_end in current_ranges:
                for col_start, col_end in previous_ranges:
                    mask[row_start:row_end, col_start:col_end] = True

            if lang_seg:
                lang_start, lang_end = lang_range(segment_idx)
                mask[lang_start:lang_end, lang_start:lang_end] = (
                    LanguageExpert.build_language_rows(
                        task_len=task_len,
                        subtask_len=subtask_len,
                        device=device,
                    )
                )
                if pred_seg:
                    pred_start, _ = prediction_range(segment_idx)
                    subtask_start = lang_start + int(task_len)
                    mask[
                        subtask_start:lang_end,
                        pred_start:pred_start + first_frame_tokens,
                    ] = True

            if pred_seg:
                pred_start, pred_end = prediction_range(segment_idx)
                if lang_seg:
                    lang_start, lang_end = lang_range(segment_idx)
                    mask[pred_start:pred_end, lang_start:lang_end] = True
                mask[pred_start:pred_end, pred_start:pred_end] = (
                    self.video_expert.build_video_to_video_mask(
                        video_seq_len=pred_seg,
                        video_tokens_per_frame=int(video_tokens_per_frame),
                        device=device,
                    )
                )

            if cond_seg:
                cond_start, cond_end = condition_range(segment_idx)
                if lang_seg:
                    lang_start, lang_end = lang_range(segment_idx)
                    mask[cond_start:cond_end, lang_start:lang_end] = True
                mask[cond_start:cond_end, cond_start:cond_end] = (
                    self.video_expert.build_video_to_video_mask(
                        video_seq_len=cond_seg,
                        video_tokens_per_frame=int(video_tokens_per_frame),
                        device=device,
                    )
                )

            if action_seg:
                action_start, action_end = action_range(segment_idx)
                if lang_seg:
                    lang_start, lang_end = lang_range(segment_idx)
                    mask[action_start:action_end, lang_start:lang_end] = True
                if cond_seg:
                    cond_start, cond_end = condition_range(segment_idx)
                    mask[action_start:action_end, cond_start:cond_end] = True
                mask[action_start:action_end, action_start:action_end] = True

        return mask

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
        prediction_tokens = int(video_pre.get("_idm_prediction_tokens", 0))
        condition_tokens = int(video_pre.get("_idm_condition_tokens", 0))
        if prediction_tokens <= 0 or condition_tokens <= 0:
            raise ValueError("HFastWAMIDM requires both prediction and future-condition video tokens.")

        action_seq_len = (
            0
            if action_pre is None
            else int(action_pre["tokens"].shape[1]) // int(num_segments)
        )
        attention_mask = self._build_interleaved_idm_attention_mask(
            task_len=task_len,
            subtask_len=subtask_len,
            prediction_video_seq_len=prediction_tokens,
            condition_video_seq_len=condition_tokens,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_segments=num_segments,
            device=video_pre["tokens"].device,
        )

        if lang_pre is not None and action_pre is not None:
            detach_set = {"language"} if self.knowledge_insulation else None
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
                detach_kv_experts=detach_set,
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
            raise ValueError("IDM training without language requires action tokens.")
        return self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": video_context_payload, "action": action_context_payload},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            detach_kv_experts={"video"} if self.action_loss_detach_video_expert else None,
            active_expert_order=("video", "action"),
        )

    @torch.no_grad()
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
        temporal_factor = int(
            getattr(self.visual_encoder, "temporal_downsample_factor", 1)
        )
        if num_video_frames is None:
            num_future_latents = 1
        else:
            num_video_frames = int(num_video_frames)
            if num_video_frames <= 1:
                raise ValueError(
                    f"`num_video_frames` must be greater than 1, got {num_video_frames}."
                )
            num_latents = (num_video_frames - 1) // temporal_factor + 1
            num_future_latents = num_latents - 1
        if num_future_latents <= 0:
            raise ValueError(
                "IDM inference requires at least one future latent; "
                f"got num_video_frames={num_video_frames}, temporal_factor={temporal_factor}."
            )

        rollout_context = first_frame_latents
        predicted_future_latents = []
        prediction_pre = video_pre
        for _ in range(num_future_latents):
            prediction_pre = self.video_expert.pre_dit(
                x=rollout_context,
                context=video_context if self.video_expert.use_text_context else None,
                context_mask=video_context_mask if self.video_expert.use_text_context else None,
            )
            prediction_context_payload = self._context_payload_from_pre_state(
                prediction_pre,
                video_context_payload is not None,
            )
            prediction_out = self._run_mot_two_experts_lv(
                lang_pre=lang_pre,
                video_pre=prediction_pre,
                task_len=task_len,
                subtask_len=subtask_len,
                video_tokens_per_frame=int(
                    prediction_pre["meta"]["tokens_per_frame"]
                ),
                video_context_payload=prediction_context_payload,
            )
            predicted_sequence = self.video_expert.post_dit(
                prediction_out["video"],
                prediction_pre,
            )
            next_latent = predicted_sequence[:, :, -1:]
            predicted_future_latents.append(next_latent)
            rollout_context = torch.cat([rollout_context, next_latent], dim=2)

        predicted_future = torch.cat(predicted_future_latents, dim=2)
        condition_latents = self._build_inference_condition_latents(
            first_frame_latents=first_frame_latents,
            predicted_future_latents=predicted_future,
        )
        condition_pre = self.video_expert.pre_dit(
            x=condition_latents,
            context=video_context if self.video_expert.use_text_context else None,
            context_mask=video_context_mask if self.video_expert.use_text_context else None,
        )
        return self._combine_idm_video_pre(prediction_pre, condition_pre)

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
    ) -> dict:
        return self._run_mot_interleaved_segments(
            lang_pre=lang_pre,
            video_pre=video_pre,
            action_pre=action_pre,
            task_len=task_len,
            subtask_len=subtask_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_segments=1,
            video_context_payload=video_context_payload,
            action_context_payload=action_context_payload,
        )

    def training_loss(
        self,
        sample: dict,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        is_interleaved = (
            "segments" in sample
            or ("video" in sample and getattr(sample["video"], "ndim", 0) == 6)
            or (
                "video_latents" in sample
                and getattr(sample["video_latents"], "ndim", 0) == 6
            )
        )
        if not is_interleaved:
            raise ValueError(
                "HFastWAMIDM currently requires interleaved segment batches. "
                "Use an interleaved data config such as "
                "data=robotwin_interleaved_webdataset."
            )
        return super().training_loss(sample, tiled=tiled)


class HFastWAMFullConditionIDM(HFastWAMIDM):
    """FastWAM-style IDM whose action condition is ``[z0, z1, ..., zT]``."""

    def _build_training_condition_latents(
            self,
            context_latents: torch.Tensor,
            target_latents: torch.Tensor,
    ) -> torch.Tensor:
            return torch.cat([context_latents[:, :, :1], target_latents], dim=2)

    def _build_inference_condition_latents(
            self,
            first_frame_latents: torch.Tensor,
            predicted_future_latents: torch.Tensor,
    ) -> torch.Tensor:
            return torch.cat([first_frame_latents, predicted_future_latents], dim=2)
