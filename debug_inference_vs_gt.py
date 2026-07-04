"""Sanity check: load a sample directly from parquet+mp4, run inference, compare to GT.

This bypasses the buggy lerobot dataset loader.
"""
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
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
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


DATASET_DIR = "/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"
EPISODE_IDX = 0


def _resize_for_dataset(img_np, target_h=224, target_w=224):
    """Mimic dataset transforms: ResizeSmallestSideAspectPreserving + CenterCrop + Normalize.

    Input: HxWx3 uint8.
    Output: torch [3,H,W] in [-1, 1].
    """
    pil = Image.fromarray(img_np)
    src_w, src_h = pil.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    # Center crop
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    pil = pil.crop((left, top, left + target_w, top + target_h))
    arr = np.asarray(pil, dtype=np.uint8)
    t = torch.tensor(arr).permute(2, 0, 1).float()
    t = t * (2.0 / 255.0) - 1.0
    return t


def _load_episode_video_frames(dataset_dir, episode_idx, key, frame_indices):
    """Load mp4 frames for a given episode + camera key + frame indices."""
    video_path = (
        Path(dataset_dir) / "videos" / "chunk-000" / key
        / f"episode_{episode_idx:06d}.mp4"
    )
    if not video_path.exists():
        # Maybe key is "observation.images.image" → directory key
        raise FileNotFoundError(f"Video not found: {video_path}")
    print(f"  Loading {video_path}")
    frames = []
    for i, frame in enumerate(iio.imiter(str(video_path))):
        if i in frame_indices:
            frames.append(frame)
            if len(frames) == len(frame_indices):
                break
    return frames


def main():
    # --- Load config ---
    config_dir = str(project_root / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="sim_libero",
            overrides=[
                "model=fastwam_vjepa2ac_predictor",
                "task=libero_uncond_2cam224_1e-4",
                "ckpt=",
                "EVALUATION.output_dir=/tmp/debug_sanity_check",
                "EVALUATION.dataset_stats_path=runs/libero_vjepa2ac_predictor/w1mn_2026-05-11_13-58-10/dataset_stats.json",
            ],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=== Loading model ===")
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()
    print(f"Model loaded.")

    # --- Load processor ---
    dataset_stats = load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    # --- Load episode parquet ---
    parquet_path = Path(DATASET_DIR) / "data" / "chunk-000" / f"episode_{EPISODE_IDX:06d}.parquet"
    print(f"\n=== Loading parquet {parquet_path} ===")
    df = pd.read_parquet(str(parquet_path))
    print(f"Episode length: {len(df)} frames")

    # Pick a starting frame index. Take 9 frames at stride 4 (matches video_sample_indices).
    start = 0  # episode start
    num_frames = int(cfg.data.train.num_frames)  # 33
    stride = int(cfg.data.train.action_video_freq_ratio)  # 4
    video_idxs = list(range(start, start + num_frames, stride))  # 9 indices
    print(f"video_idxs: {video_idxs}")

    # GT actions: [num_frames-1, action_dim] = [32, 7]
    gt_actions_raw = np.stack(df["action"].iloc[start:start + num_frames - 1].values, axis=0).astype(np.float32)
    print(f"gt_actions_raw shape: {gt_actions_raw.shape}, sample t0: {gt_actions_raw[0]}")

    # GT state at t=0
    state_t0_raw = np.array(df["observation.state"].iloc[start], dtype=np.float32)
    print(f"state_t0_raw (8-dim): {state_t0_raw}")

    # Episode task
    tasks_jsonl = pd.read_json(Path(DATASET_DIR) / "meta" / "tasks.jsonl", lines=True)
    episodes_jsonl = pd.read_json(Path(DATASET_DIR) / "meta" / "episodes.jsonl", lines=True)
    task_idx = int(df["task_index"].iloc[0])
    # tasks.jsonl maps task_index -> task description
    task_desc = tasks_jsonl[tasks_jsonl["task_index"] == task_idx]["task"].iloc[0]
    print(f"Task: {task_desc}")
    prompt = DEFAULT_PROMPT.format(task=task_desc)
    print(f"Prompt: {prompt}")

    # --- Load image+wrist frames ---
    print("\n=== Loading frames ===")
    img_frames = _load_episode_video_frames(DATASET_DIR, EPISODE_IDX, "observation.images.image", video_idxs)
    wrist_frames = _load_episode_video_frames(DATASET_DIR, EPISODE_IDX, "observation.images.wrist_image", video_idxs)
    print(f"loaded {len(img_frames)} primary frames, {len(wrist_frames)} wrist frames")
    print(f"primary frame shape: {img_frames[0].shape}, wrist: {wrist_frames[0].shape}")

    # Build first-frame (t=0) input image like _obs_to_model_input
    primary_t0 = img_frames[0]  # H, W, 3 uint8
    wrist_t0 = wrist_frames[0]

    # Crop+resize each camera to 224x224 then horizontally concat → 224x448
    primary_resized = _resize_for_dataset(primary_t0, target_h=224, target_w=224)  # [3,224,224]
    wrist_resized = _resize_for_dataset(wrist_t0, target_h=224, target_w=224)
    rgb = torch.cat([primary_resized, wrist_resized], dim=2)  # [3,224,448]
    input_image = rgb.unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,224,448]
    print(f"input_image shape: {input_image.shape}")

    # --- Build proprio (t=0) and normalize through processor ---
    # Same as _normalize_proprio in eval_libero_single
    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(state_t0_raw, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    proprio_norm = state_batch["state"][state_key]
    print(f"proprio_norm shape: {proprio_norm.shape} (expect [1, 8]). value: {proprio_norm[0].numpy()}")
    proprio_input = proprio_norm.to(device=device, dtype=dtype)

    # --- Run inference ---
    print("\n=== Running inference ===")
    action_horizon = num_frames - 1  # 32
    num_video_frames_arg = (num_frames - 1) // stride + 1  # = 9
    print(f"action_horizon: {action_horizon}")
    print(f"num_video_frames_arg passed to infer_action: {num_video_frames_arg}")

    with torch.no_grad():
        pred = model.infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            negative_prompt="",
            text_cfg_scale=1.0,
            num_inference_steps=int(cfg.get("eval_num_inference_steps", 10)),
            proprio=proprio_input,
            sigma_shift=None,
            seed=42,
            rand_device="cpu",
            tiled=False,
            num_video_frames=num_video_frames_arg,
        )
    pred_action_norm = pred["action"].to(dtype=torch.float32, device="cpu")  # [T, D]
    print(f"pred_action_norm shape: {pred_action_norm.shape}")

    # --- Denormalize predicted action ---
    action_meta = processor.shape_meta["action"]
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    pred_action_raw = normalizer.backward(pred_action_norm.unsqueeze(0))[0].numpy()
    print(f"pred_action_raw shape: {pred_action_raw.shape}")

    # The dataset flips gripper: (raw[-1] is 0/1) -> flip via *2-1, *-1 → (-1=open, +1=close).
    # During training, gripper was *not* flipped — the dataset just normalizes via min/max.
    # So pred_action_raw[..., -1] is in [0, 1] range (same as dataset action).
    # If we apply the eval flip (* 2 - 1, * -1), we get ([-1,+1] flipped).

    # For comparison let's compare raw (un-flipped) values directly with GT.
    print("\n=== Comparing pred vs GT (raw values, before eval-time gripper flip) ===")
    diff = np.abs(pred_action_raw - gt_actions_raw)
    print(f"Per-dim MAE (raw): {diff.mean(axis=0)}")
    print(f"Overall MAE: {diff.mean():.4f}")
    print(f"Overall max: {diff.max():.4f}")

    print("\nFirst 3 steps:")
    for t in range(min(3, pred_action_raw.shape[0])):
        print(f"  t={t}")
        print(f"    pred (raw): {pred_action_raw[t]}")
        print(f"    gt   (raw): {gt_actions_raw[t]}")

    print("\nLast 3 steps:")
    for t in range(max(0, pred_action_raw.shape[0]-3), pred_action_raw.shape[0]):
        print(f"  t={t}")
        print(f"    pred (raw): {pred_action_raw[t]}")
        print(f"    gt   (raw): {gt_actions_raw[t]}")

    # Save for inspection
    np.savez(
        "/tmp/debug_sanity_check.npz",
        pred_action_raw=pred_action_raw,
        gt_actions_raw=gt_actions_raw,
        pred_action_norm=pred_action_norm.numpy(),
    )
    print("\nSaved /tmp/debug_sanity_check.npz")


if __name__ == "__main__":
    main()
