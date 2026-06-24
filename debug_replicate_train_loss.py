"""Compute weighted action loss matching training, see if it matches val_loss=0.07."""
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
                "model=fastwam_vjepa2ac_predictor",
                "task=libero_uncond_2cam224_1e-4",
                "ckpt=/apdcephfs_tj5/share_302528826/shaunxhwang/ckpts/libero_vjepa2ac_predictor/step_021700.pt",
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

    dataset_stats = load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    sched = model.train_action_scheduler

    # Load 5 episodes
    samples = []
    for episode_idx in range(8):
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
        if len(img_frames) < 9:
            continue

        video_per_frame = []
        for i in range(9):
            p = _resize(img_frames[i])
            w = _resize(wrist_frames[i])
            rgb = torch.cat([p, w], dim=2)
            video_per_frame.append(rgb)
        video_full = torch.stack(video_per_frame, dim=1).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            real_latents = model.visual_encoder.encode(video_full, device=device)

        samples.append({
            "real_latents": real_latents,
            "gt_actions_raw": gt_actions_raw,
            "state_t0_raw": state_t0_raw,
            "prompt": prompt,
        })

    action_stats = dataset_stats['action']['default']
    cur_stats = {k.removeprefix('global_'): v for k, v in action_stats.items() if k.startswith('global_')}
    norm_action = SingleFieldLinearNormalizer(stats=cur_stats, mode='min/max')

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]

    # Replicate the exact training loss computation
    print("\n=== Replicating training-time action loss with N samples ===")
    print("This should match val_loss ~0.07 reported during training.")
    rng = torch.Generator(device='cpu').manual_seed(0)
    n_samples = 100
    weighted_losses = []
    unweighted_losses = []

    for k in range(n_samples):
        rec = samples[k % len(samples)]
        gt_norm = norm_action.forward(torch.tensor(rec["gt_actions_raw"]))
        gt_norm_dev = gt_norm.unsqueeze(0).to(device=device, dtype=dtype)

        sb = {"state": {state_key: torch.as_tensor(rec["state_t0_raw"], dtype=torch.float32).unsqueeze(0)}}
        sb = processor.action_state_transform(sb)
        sb = processor.normalizer.forward(sb)
        proprio_norm = sb["state"][state_key].to(device=device, dtype=dtype)

        context, context_mask = model.encode_prompt(rec["prompt"])
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio_norm,
        )

        # Sample t the same way training does
        timestep_action = sched.sample_training_t(batch_size=1, device=device, dtype=dtype)
        # Sample noise
        noise = torch.randn(gt_norm_dev.shape, generator=rng, device='cpu').to(device=device, dtype=dtype)
        noisy = sched.add_noise(gt_norm_dev, noise, timestep_action)
        target = sched.training_target(gt_norm_dev, noise, timestep_action)

        with torch.no_grad():
            pred = model._predict_action_noise(
                first_frame_latents=rec["real_latents"],
                latents_action=noisy,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )

        # MSE per token
        action_loss_token = (pred.float() - target.float()).pow(2).mean(dim=2)  # [B, T]
        action_loss_per_sample = action_loss_token.mean(dim=1)  # [B]
        weight = sched.training_weight(timestep_action).to(action_loss_per_sample.device)
        weighted_loss = (action_loss_per_sample * weight).mean().item()
        unweighted_loss = action_loss_per_sample.mean().item()
        weighted_losses.append(weighted_loss)
        unweighted_losses.append(unweighted_loss)

    print(f"\nResults averaged over {n_samples} samples:")
    print(f"  Mean weighted_loss   = {np.mean(weighted_losses):.4f}  (this should match training loss ~0.07)")
    print(f"  Mean unweighted_loss = {np.mean(unweighted_losses):.4f}")
    print()
    print(f"For comparison, training val_loss was ~0.07-0.08")


if __name__ == "__main__":
    main()
