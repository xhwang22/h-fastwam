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
    tokenizer_parent = _resolved_dir(args.tokenizer_parent, "Wan tokenizer parent")
    tokenizer_json = tokenizer_parent / "google" / "umt5-xxl" / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"UMT5 tokenizer.json not found: {tokenizer_json}")

    cfg = OmegaConf.load(source)
    model_cfg = cfg.model if "model" in cfg else cfg
    target = str(model_cfg.get("_target_", ""))
    encoder_type = str(model_cfg.visual_encoder_config.get("encoder_type", ""))

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
    else:
        raise ValueError(f"Unsupported model kind: {args.model}")

    model_cfg.fastwam_checkpoint = None
    model_cfg.tokenizer_model_id = str(tokenizer_parent)
    model_cfg.language_model_id = str(qwen_dir)
    model_cfg.language_local_files_only = True

    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    print(f"wrote: {output}")
    print(f"target: {target}")
    print(f"encoder: {encoder_type}")
    print(f"qwen: {qwen_dir}")
    print(f"tokenizer: {tokenizer_parent}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xr1", "idm"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--tokenizer-parent", required=True)
    parser.add_argument("--xr1-checkpoint")
    parser.add_argument("--vjepa-checkpoint")
    parser.add_argument("--vjepa-repo")
    args = parser.parse_args()

    if args.model == "xr1" and not args.xr1_checkpoint:
        parser.error("--xr1-checkpoint is required for --model=xr1")
    if args.model == "idm" and (
        not args.vjepa_checkpoint or not args.vjepa_repo
    ):
        parser.error("--vjepa-checkpoint and --vjepa-repo are required for --model=idm")
    prepare(args)


if __name__ == "__main__":
    main()
