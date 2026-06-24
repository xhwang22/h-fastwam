"""Same teacher-forcing test but with dinov3 model (which works well in eval).

This tells us if the vjepa2ac_predictor model has fundamentally worse action prediction.
"""
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/apdcephfs_tj5/share_302528826/shaunxhwang/fastwam/checkpoints/checkpoints/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pandas as pd
import imageio.v3 as iio
from PIL import Image
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json, SingleFieldLinearNormalizer,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


DATASET_DIR = "/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"


def _resize(img_np, target_h=224, target_w=224):
    pil = Image.fromarray(img_np)
    src_w, src_h = pil.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    pil = pil.crop((left, top, left + target_w, top + target_h))
    arr = np.asarray(pil, dtype=np.uint8)
    t = torch.tensor(arr).permute(2, 0, 1).float()
    t = t * (2.0 / 255.0) - 1.0
    return t


def _load_episode_video_frames(dataset_dir, episode_idx, key, frame_indices):
    video_path = (
        Path(dataset_dir) / "videos" / "chunk-000" / key
        / f"episode_{episode_idx:06d}.mp4"
    )
    frames = []
    for i, frame in enumerate(iio.imiter(str(video_path))):
        if i in frame_indices:
            frames.append(frame)
            if len(frames) == len(frame_indices):
                break
    return frames


def main():
    config_dir = str(project_root / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="sim_libero",
            overrides=[
                "model=fastwam_dino_ditproj",
                "task=libero_uncond_2cam224_1e-4",
                "ckpt=/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/runs/libero_dinov3/exp1/checkpoints/weights/step_021700.pt",
                "EVALUATION.output_dir=/tmp/debug_sanity_check",
                "EVALUATION.dataset_stats_path=/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/runs/libero_dinov3/exp1/dataset_stats.json",
                # Dinov3 specifics
                "model.visual_encoder.model_name=facebook/dinov3-vitl16-pretrain-lvd1689m",
                "model.visual_encoder.output_dim=48",
                "model.visual_encoder.freeze_backbone=true",
            ],
        )
    print("Note: trying with dino_ditproj fastwam config; visual_encoder may not be the same as dinov3 training run")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=== Loading dinov3 model ===")
    try:
        model = instantiate(cfg.model, model_dtype=dtype, device=device)
        model.load_checkpoint(str(cfg.ckpt))
        model = model.to(device).eval()
    except Exception as e:
        print(f"Failed to load dinov3: {e}")
        return

    dataset_stats = load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    # Load 5 episodes
    all_results = []
    for episode_idx in range(5):
        parquet_path = Path(DATASET_DIR) / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        df = pd.read_parquet(str(parquet_path))
        if len(df) < 35:
            continue
        start = max(0, len(df) // 4)
        num_frames = int(cfg.data.train.num_frames)
        stride = int(cfg.data.train.action_video_freq_ratio)
        if start + num_frames > len(df):
            start = len(df) - num_frames
        video_idxs = list(range(start, start + num_frames, stride))

        gt_actions_raw = np.stack(df["action"].iloc[start:start + num_frames - 1].values, axis=0).astype(np.float32)
        state_t0_raw = np.array(df["observation.state"].iloc[start], dtype=np.float32)

        tasks_jsonl = pd.read_json(Path(DATASET_DIR) / "meta" / "tasks.jsonl", lines=True)
        task_idx = int(df["task_index"].iloc[0])
        task_desc = tasks_jsonl[tasks_jsonl["task_index"] == task_idx]["task"].iloc[0]
        prompt = DEFAULT_PROMPT.format(task=task_desc)

        img_frames = _load_episode_video_frames(DATASET_DIR, episode_idx, "observation.images.image", video_idxs)
        wrist_frames = _load_episode_video_frames(DATASET_DIR, episode_idx, "observation.images.wrist_image", video_idxs)

        # First-frame only for dino
        primary = _resize(img_frames[0])
        wrist = _resize(wrist_frames[0])
        first_frame = torch.cat([primary, wrist], dim=2).unsqueeze(0).to(device=device, dtype=dtype)

        all_results.append({
            "first_frame": first_frame,
            "gt_actions_raw": gt_actions_raw,
            "state_t0_raw": state_t0_raw,
            "prompt": prompt,
            "task_desc": task_desc,
        })

    action_stats = dataset_stats['action']['default']
    cur_stats = {k.removeprefix('global_'): v for k, v in action_stats.items() if k.startswith('global_')}
    norm_action = SingleFieldLinearNormalizer(stats=cur_stats, mode='min/max')

    # === Run inference on each episode ===
    print("\n=== dino_ditproj inference MAE per episode (10-step, real first frame) ===")
    print(f"{'ep':>3} {'mae':>8}  task")
    all_maes = []
    for i, rec in enumerate(all_results):
        gt_norm = norm_action.forward(torch.tensor(rec["gt_actions_raw"]))
        action_horizon = rec["gt_actions_raw"].shape[0]

        state_meta = processor.shape_meta["state"]
        state_key = state_meta[0]["key"]
        sb = {"state": {state_key: torch.as_tensor(rec["state_t0_raw"], dtype=torch.float32).unsqueeze(0)}}
        sb = processor.action_state_transform(sb)
        sb = processor.normalizer.forward(sb)
        proprio_norm = sb["state"][state_key].to(device=device, dtype=dtype)

        with torch.no_grad():
            pred = model.infer_action(
                prompt=rec["prompt"],
                input_image=rec["first_frame"],
                action_horizon=action_horizon,
                negative_prompt="",
                text_cfg_scale=1.0,
                num_inference_steps=10,
                proprio=proprio_norm,
                sigma_shift=None,
                seed=42,
                rand_device="cpu",
                tiled=False,
            )
        pred_action = pred["action"].to(dtype=torch.float32, device="cpu")
        mae = (pred_action - gt_norm).abs().mean().item()
        all_maes.append(mae)
        print(f"{i:>3} {mae:>8.4f}  {rec['task_desc'][:60]}")
    print(f"\n  mean MAE: {np.mean(all_maes):.4f}")


if __name__ == "__main__":
    main()
