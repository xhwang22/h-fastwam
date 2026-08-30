#!/usr/bin/env python3
"""Rewrite a training config for offline RoboTwin evaluation on H100."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def _resolved_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return path


def _resolved_file_or_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def prepare(args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training config not found: {source}")

    qwen_dir = _resolved_dir(args.qwen_dir, "Qwen")
    cfg = OmegaConf.load(source)
    model_cfg = cfg.model if "model" in cfg else cfg
    target = str(model_cfg.get("_target_", ""))
    visual_cfg = model_cfg.get("visual_encoder_config")
    encoder_type = (
        ""
        if visual_cfg in (None, "null")
        else str(visual_cfg.get("encoder_type", ""))
    )
    language_backend = str(model_cfg.get("language_backend", "")).lower()
    if language_backend != "qwen3":
        raise ValueError(
            "The T5-free H100 launcher only supports language_backend=qwen3; "
            f"got {language_backend!r}."
        )

    if args.model == "xr1":
        if target != "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam":
            raise ValueError(f"XR-1 config has unexpected model target: {target}")
        if encoder_type != "xr1_vision":
            raise ValueError(f"XR-1 config has unexpected visual encoder: {encoder_type}")
        xr1_checkpoint = _resolved_file_or_dir(args.xr1_checkpoint, "XR-1 checkpoint")
        model_cfg.visual_encoder_config.checkpoint_path = str(xr1_checkpoint)
        model_cfg.visual_encoder_config.model_name = str(qwen_dir)
        model_cfg.visual_encoder_config.local_files_only = True
    elif args.model == "idm":
        if "HFastWAMIDM" not in target and "HFastWAMFullConditionIDM" not in target:
            raise ValueError(f"IDM config has unexpected model target: {target}")
        if encoder_type != "vjepa2_1":
            raise ValueError(f"IDM config has unexpected visual encoder: {encoder_type}")
        vjepa_checkpoint = _resolved_file_or_dir(
            args.vjepa_checkpoint,
            "V-JEPA2.1 checkpoint",
        )
        vjepa_repo = _resolved_dir(args.vjepa_repo, "V-JEPA2.1 source")
        vision_transformer = (
            vjepa_repo / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
        )
        if not vision_transformer.is_file():
            raise FileNotFoundError(
                f"V-JEPA2.1 source is incomplete; missing {vision_transformer}"
            )
        model_cfg.visual_encoder_config.checkpoint_path = str(vjepa_checkpoint)
        model_cfg.visual_encoder_config.repo_path = str(vjepa_repo)
    elif args.model == "vjepa21_flow":
        if target != "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam":
            raise ValueError(f"V-JEPA Flow-DiT config has unexpected target: {target}")
        if encoder_type != "vjepa2_1":
            raise ValueError(
                f"V-JEPA Flow-DiT config has unexpected visual encoder: {encoder_type}"
            )
        vjepa_checkpoint = _resolved_file_or_dir(
            args.vjepa_checkpoint,
            "V-JEPA2.1 checkpoint",
        )
        vjepa_repo = _resolved_dir(args.vjepa_repo, "V-JEPA2.1 source")
        vision_transformer = (
            vjepa_repo / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
        )
        if not vision_transformer.is_file():
            raise FileNotFoundError(
                f"V-JEPA2.1 source is incomplete; missing {vision_transformer}"
            )
        model_cfg.visual_encoder_config.checkpoint_path = str(vjepa_checkpoint)
        model_cfg.visual_encoder_config.repo_path = str(vjepa_repo)
        configured_stats = model_cfg.visual_encoder_config.get(
            "normalise_stats_path"
        )
        if configured_stats not in (None, "", "null"):
            if not args.vjepa_normalise_stats:
                raise ValueError(
                    "This training config uses fixed V-JEPA normalization; "
                    "--vjepa-normalise-stats is required."
                )
            normalise_stats = _resolved_file_or_dir(
                args.vjepa_normalise_stats,
                "V-JEPA2.1 normalization stats",
            )
            model_cfg.visual_encoder_config.normalise_stats_path = str(
                normalise_stats
            )
    elif args.model == "dinov3_flow":
        if target != "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam":
            raise ValueError(f"DINOv3 Flow-DiT config has unexpected target: {target}")
        if encoder_type != "dino":
            raise ValueError(
                f"DINOv3 Flow-DiT config has unexpected visual encoder: {encoder_type}"
            )
        dino_model = _resolved_dir(args.dinov3_model, "DINOv3 model")
        model_cfg.visual_encoder_config.model_name = str(dino_model)
    elif args.model == "siglip2_flow":
        if target != "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam":
            raise ValueError(f"SigLIP2 Flow-DiT config has unexpected target: {target}")
        if encoder_type not in {"siglip2_vision", "native_siglip2"}:
            raise ValueError(
                f"SigLIP2 Flow-DiT config has unexpected visual encoder: {encoder_type}"
            )
        siglip2_model = _resolved_dir(args.siglip2_model, "SigLIP2 model")
        model_cfg.visual_encoder_config.model_name = str(siglip2_model)
        model_cfg.visual_encoder_config.local_files_only = True
    elif args.model == "vae_predictor":
        if target != "fastwam.models.hfastwam.hfastwam.HFastWAM.from_pretrained_fastwam":
            raise ValueError(f"VAE predictor config has unexpected target: {target}")
        if visual_cfg not in (None, "null"):
            raise ValueError("VAE predictor must not configure a visual encoder.")
        if str(model_cfg.get("video_expert_type", "")) != "jepa_predictor":
            raise ValueError("VAE predictor must use video_expert_type=jepa_predictor.")
    else:
        raise ValueError(f"Unsupported model kind: {args.model}")

    model_cfg.fastwam_checkpoint = None
    model_cfg.load_text_encoder = False
    model_cfg.language_model_id = str(qwen_dir)
    model_cfg.language_local_files_only = True

    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    print(f"wrote: {output}")
    print(f"target: {target}")
    print(f"encoder: {encoder_type}")
    print(f"qwen: {qwen_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "xr1",
            "idm",
            "vjepa21_flow",
            "dinov3_flow",
            "siglip2_flow",
            "vae_predictor",
        ],
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--xr1-checkpoint")
    parser.add_argument("--vjepa-checkpoint")
    parser.add_argument("--vjepa-repo")
    parser.add_argument("--vjepa-normalise-stats")
    parser.add_argument("--dinov3-model")
    parser.add_argument("--siglip2-model")
    args = parser.parse_args()

    if args.model == "xr1" and not args.xr1_checkpoint:
        parser.error("--xr1-checkpoint is required for --model=xr1")
    if args.model in {"idm", "vjepa21_flow"} and (
        not args.vjepa_checkpoint or not args.vjepa_repo
    ):
        parser.error(
            "--vjepa-checkpoint and --vjepa-repo are required for "
            f"--model={args.model}"
        )
    if args.model == "dinov3_flow" and not args.dinov3_model:
        parser.error("--dinov3-model is required for --model=dinov3_flow")
    if args.model == "siglip2_flow" and not args.siglip2_model:
        parser.error("--siglip2-model is required for --model=siglip2_flow")
    prepare(args)


if __name__ == "__main__":
    main()
