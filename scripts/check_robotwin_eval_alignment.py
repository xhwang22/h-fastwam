#!/usr/bin/env python3
"""Fail fast when a RoboTwin eval config diverges from its training run."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _model_cfg(cfg):
    return cfg.model if "model" in cfg else cfg


def _redacted_model(cfg) -> dict:
    payload = copy.deepcopy(
        OmegaConf.to_container(_model_cfg(cfg), resolve=True)
    )
    for key in (
        "fastwam_checkpoint",
        "load_text_encoder",
        "language_model_id",
        "language_local_files_only",
        "tokenizer_model_id",
    ):
        payload.pop(key, None)
    visual = payload.get("visual_encoder_config")
    if isinstance(visual, dict):
        for key in (
            "checkpoint_path",
            "repo_path",
            "model_name",
            "local_files_only",
        ):
            visual.pop(key, None)
    return payload


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_training_stats(train_cfg, repo_root: Path) -> Path | None:
    data_train = train_cfg.data.train
    explicit = data_train.get("pretrained_norm_stats")
    if explicit not in (None, "", "null"):
        path = Path(str(explicit)).expanduser()
        return path if path.is_absolute() else (repo_root / path).resolve()
    preprocessed_root = data_train.get("preprocessed_root")
    if preprocessed_root not in (None, "", "null"):
        root = Path(str(preprocessed_root)).expanduser()
        root = root if root.is_absolute() else (repo_root / root).resolve()
        return root / "dataset_stats.json"
    return None


def _check_data_contract(train_cfg) -> None:
    data = train_cfg.data.train
    _require_equal(int(data.num_frames), 33, "data.train.num_frames")
    _require_equal(
        int(data.action_video_freq_ratio),
        4,
        "data.train.action_video_freq_ratio",
    )
    _require_equal(list(data.video_size), [384, 320], "data.train.video_size")
    _require_equal(str(data.concat_multi_camera), "robotwin", "camera layout")
    _require_equal(int(data.get("num_segments", 1)), 1, "data.train.num_segments")
    processor = data.processor
    _require_equal(int(processor.action_output_dim), 14, "action_output_dim")
    _require_equal(int(processor.proprio_output_dim), 14, "proprio_output_dim")

    transforms = processor.get("train_transforms")
    if transforms is not None:
        resize_entries = [
            entry
            for entry in transforms
            if str(entry.get("_target_", "")).endswith("transforms.Resize")
        ]
        if len(resize_entries) != 1:
            raise ValueError(
                "Expected exactly one training image Resize transform, "
                f"found {len(resize_entries)}."
            )
        _require_equal(
            list(resize_entries[0].size),
            [240, 320],
            "training camera pre-resize",
        )


def _check_model_contract(model_kind: str, model) -> None:
    latent_config = model.get("latent_action_config")
    latent_enabled = bool(latent_config and latent_config.get("enabled", False))
    if latent_enabled and model_kind != "vjepa":
        raise ValueError("Latent-action evaluation is only supported for the V-JEPA model.")

    _require_equal(str(model.language_backend), "qwen3", "language_backend")
    _require_equal(bool(model.freeze_language_expert), True, "freeze_language_expert")
    _require_equal(float(model.loss_config.lambda_language), 0.0, "lambda_language")
    _require_equal(bool(model.knowledge_insulation), False, "knowledge_insulation")
    _require_equal(
        bool(model.action_loss_detach_video_expert),
        False,
        "action detach",
    )
    _require_equal(str(model.layer_alignment_mode), "tail_overlap", "layer alignment")
    _require_equal(int(model.video_dit_config.hidden_dim), 2048, "video hidden_dim")
    _require_equal(int(model.video_dit_config.num_layers), 28, "video num_layers")
    _require_equal(int(model.action_dit_config.hidden_dim), 2048, "action hidden_dim")
    _require_equal(int(model.action_dit_config.num_layers), 28, "action num_layers")
    _require_equal(
        int(model.action_dit_config.action_dim),
        32 if latent_enabled else 14,
        "ActionDiT action_dim",
    )
    if latent_enabled:
        expected_contract = {
            "latent_horizon": 8,
            "latent_dim": 32,
            "physical_action_horizon": 32,
            "physical_action_dim": 14,
            "actions_per_latent": 4,
        }
        for key, expected in expected_contract.items():
            _require_equal(int(latent_config[key]), expected, f"latent action {key}")
        decoder = model.get("latent_action_decoder_config")
        if decoder is None:
            raise ValueError("Latent-action model is missing latent_action_decoder_config.")
        _require_equal(int(decoder.latent_dim), 32, "decoder latent_dim")
        _require_equal(int(decoder.num_latents), 8, "decoder num_latents")
        _require_equal(int(decoder.substeps_per_latent), 4, "decoder substeps")
        _require_equal(int(decoder.action_dim), 14, "decoder action_dim")
    visual = model.visual_encoder_config
    _require_equal(bool(visual.freeze_backbone), True, "visual freeze_backbone")
    _require_equal(bool(visual.skip_projection), True, "visual skip_projection")
    _require_equal(int(visual.temporal_downsample), 4, "temporal downsample")
    _require_equal(bool(visual.causal_tubelet_encoding), True, "causal tubelet")
    _require_equal(bool(visual.standardise_output), True, "standardise_output")

    target = str(model.get("_target_", ""))
    if model_kind == "xr1":
        _require_equal(
            target,
            "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam",
            "XR-1 model target",
        )
        _require_equal(str(visual.encoder_type), "xr1_vision", "XR-1 encoder")
        _require_equal(int(model.video_dit_config.in_dim), 1024, "XR-1 video in_dim")
        _require_equal(int(model.video_dit_config.out_dim), 1024, "XR-1 video out_dim")
        _require_equal(
            str(model.video_dit_config.video_attention_mask_mode),
            "first_frame_causal",
            "XR-1 video mask",
        )
    elif model_kind == "idm":
        _require_equal(
            target,
            "fastwam.models.hfastwam.hfastwam_idm."
            "HFastWAMFullConditionIDM.from_pretrained_fastwam",
            "full-condition IDM target",
        )
        _require_equal(str(visual.encoder_type), "vjepa2_1", "IDM encoder")
        _require_equal(int(model.video_dit_config.in_dim), 1664, "IDM video in_dim")
        _require_equal(int(model.video_dit_config.out_dim), 1664, "IDM video out_dim")
        _require_equal(
            str(model.video_dit_config.video_attention_mask_mode),
            "per_frame_causal",
            "IDM video mask",
        )
    elif model_kind == "vjepa":
        _require_equal(
            target,
            "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam",
            "V-JEPA model target",
        )
        _require_equal(str(model.video_expert_type), "jepa_predictor", "video expert")
        _require_equal(bool(model.fixed_target_encoder), False, "fixed_target_encoder")
        _require_equal(str(visual.encoder_type), "vjepa2_1", "V-JEPA encoder")
        _require_equal(int(model.video_dit_config.in_dim), 1664, "V-JEPA video in_dim")
        _require_equal(int(model.video_dit_config.out_dim), 1664, "V-JEPA video out_dim")
        _require_equal(
            str(model.video_dit_config.video_attention_mask_mode),
            "per_frame_causal",
            "V-JEPA video mask",
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")


def _check_checkpoint(
    checkpoint_path: Path,
    expected_video_dim: int,
    expected_action_dim: int = 14,
    latent_action_config=None,
    expected_fixed_target_encoder: bool | None = None,
) -> None:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    for key in ("mot", "language_expert", "proprio_encoder", "visual_encoder"):
        if key not in payload:
            raise ValueError(f"Checkpoint is missing `{key}`: {checkpoint_path}")
    if expected_fixed_target_encoder is not None:
        _require_equal(
            bool(payload.get("fixed_target_encoder", False)),
            expected_fixed_target_encoder,
            "checkpoint fixed_target_encoder",
        )
    mot = payload["mot"]
    action_input = mot.get("mixtures.action.action_encoder.weight")
    action_output = mot.get("mixtures.action.head.weight")
    video_patch = mot.get("mixtures.video.patch_embedding.weight")
    proprio = payload["proprio_encoder"].get("weight")
    if action_input is None or tuple(action_input.shape)[1] != expected_action_dim:
        raise ValueError(
            f"Checkpoint ActionDiT input is not {expected_action_dim}-d: "
            f"{None if action_input is None else tuple(action_input.shape)}"
        )
    if action_output is None or tuple(action_output.shape)[0] != expected_action_dim:
        raise ValueError(
            f"Checkpoint ActionDiT output is not {expected_action_dim}-d: "
            f"{None if action_output is None else tuple(action_output.shape)}"
        )
    if proprio is None or tuple(proprio.shape) != (4096, 14):
        raise ValueError(
            f"Checkpoint proprio encoder is not 14→4096: "
            f"{None if proprio is None else tuple(proprio.shape)}"
        )
    if video_patch is None or tuple(video_patch.shape)[1] != expected_video_dim:
        raise ValueError(
            f"Checkpoint video input dim is not {expected_video_dim}: "
            f"{None if video_patch is None else tuple(video_patch.shape)}"
        )

    if latent_action_config is not None:
        metadata = payload.get("checkpoint_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Latent-action checkpoint is missing checkpoint_metadata.")
        _require_equal(int(metadata.get("checkpoint_schema_version", -1)), 2, "checkpoint schema")
        _require_equal(metadata.get("action_representation"), "latent", "action representation")
        expected_signature = latent_action_config.get("latent_cache_signature")
        if expected_signature is not None:
            _require_equal(
                metadata.get("latent_cache_signature"),
                str(expected_signature),
                "latent cache signature",
            )
        decoder = payload.get("latent_action_decoder")
        if not isinstance(decoder, dict):
            raise ValueError("Latent-action checkpoint is missing `latent_action_decoder`.")
        latent_projection = decoder.get("latent_projection.weight")
        action_projection = decoder.get("action_projection.weight")
        if latent_projection is None or tuple(latent_projection.shape)[1] != 32:
            raise ValueError(
                "Checkpoint latent decoder input is not 32-d: "
                f"{None if latent_projection is None else tuple(latent_projection.shape)}"
            )
        if action_projection is None or tuple(action_projection.shape)[0] != 14:
            raise ValueError(
                "Checkpoint latent decoder output is not 14-d: "
                f"{None if action_projection is None else tuple(action_projection.shape)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xr1", "idm", "vjepa"], required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--training-stats")
    parser.add_argument("--camera-type", default="Large_D435")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    train_config_path = Path(args.train_config).expanduser().resolve()
    eval_config_path = Path(args.eval_config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    stats_path = Path(args.dataset_stats).expanduser().resolve()
    for path, label in (
        (train_config_path, "training config"),
        (eval_config_path, "eval config"),
        (checkpoint_path, "checkpoint"),
        (stats_path, "dataset stats"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    train_cfg = OmegaConf.load(train_config_path)
    eval_cfg = OmegaConf.load(eval_config_path)
    if _redacted_model(train_cfg) != _redacted_model(eval_cfg):
        raise ValueError(
            "Eval model config differs from training config outside allowed "
            "asset-path/load_text_encoder rewrites."
        )

    _check_data_contract(train_cfg)
    model = _model_cfg(eval_cfg)
    _check_model_contract(args.model, model)
    expected_video_dims = {"xr1": 1024, "idm": 1664, "vjepa": 1664}
    latent_action_config = model.get("latent_action_config")
    latent_enabled = bool(
        latent_action_config and latent_action_config.get("enabled", False)
    )
    _check_checkpoint(
        checkpoint_path,
        expected_video_dim=expected_video_dims[args.model],
        expected_action_dim=32 if latent_enabled else 14,
        latent_action_config=latent_action_config if latent_enabled else None,
        expected_fixed_target_encoder=False if args.model == "vjepa" else None,
    )

    training_stats = (
        Path(args.training_stats).expanduser().resolve()
        if args.training_stats
        else _resolve_training_stats(train_cfg, repo_root)
    )
    if training_stats is None or not training_stats.is_file():
        raise FileNotFoundError(
            "Training-time dataset stats are unavailable. Set TRAINING_STATS "
            "to the exact dataset_stats.json used by the training run."
        )
    if _sha256(training_stats) != _sha256(stats_path):
        raise ValueError(
            f"Dataset stats differ from training stats: "
            f"train={training_stats}, eval={stats_path}"
        )
    _require_equal(args.camera_type, "Large_D435", "evaluation camera type")
    stats_status = f"sha256 matched {training_stats}"

    num_video_frames = (
        int(train_cfg.data.train.num_frames) - 1
    ) // int(train_cfg.data.train.action_video_freq_ratio) + 1
    print("RoboTwin controllable eval alignment: OK")
    print(f"model={args.model}")
    print(f"checkpoint={checkpoint_path}")
    print("action_horizon=32")
    print(f"num_video_frames={num_video_frames}")
    if args.model == "idm":
        print("IDM condition=[z0,pred_z1,pred_z2], inference KV cache=disabled")
    elif args.model == "vjepa" and latent_enabled:
        print("V-JEPA predictor + latent ActionDiT=8x32 + physical decoder=32x14")
    elif args.model == "vjepa":
        print("V-JEPA predictor conditioning=non-IDM per-frame causal latent sequence")
    else:
        print("XR-1 action conditioning=current clean first-frame latent")
    print("image_pipeline=480x640 -> 240x320 -> RoboTwin 384x320 canvas")
    print(f"dataset_stats={stats_status}")
    print(
        "known_limitation=training used dynamic Qwen batch padding; a single-sample "
        "eval cannot reproduce the exact number of padded EOS tokens without retraining"
    )


if __name__ == "__main__":
    main()
