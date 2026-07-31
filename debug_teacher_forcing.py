"""Run teacher-forcing across multiple sigma values to compare with training loss.

If model is well-trained, MSE(pred_velocity, target) should match training loss curve.
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
from fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json, SingleFieldLinearNormalizer,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


DATASET_DIR = "/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"


def _resize_for_dataset(img_np, target_h=224, target_w=224):
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

    dataset_stats = load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    # Load 8 episodes for variety
    all_results = []
    for episode_idx in range(8):
        parquet_path = Path(DATASET_DIR) / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        df = pd.read_parquet(str(parquet_path))
        if len(df) < 35:
            continue
        # take a window in the middle of the episode
        start = max(0, len(df) // 4)
        num_frames = int(cfg.data.train.num_frames)  # 33
        stride = int(cfg.data.train.action_video_freq_ratio)  # 4
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
            p = _resize_for_dataset(img_frames[i])
            w = _resize_for_dataset(wrist_frames[i])
            rgb = torch.cat([p, w], dim=2)
            video_per_frame.append(rgb)
        video_full = torch.stack(video_per_frame, dim=1).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            real_latents = model.visual_encoder.encode(video_full, device=device)
        all_results.append({
            "real_latents": real_latents,
            "video_full": video_full,
            "gt_actions_raw": gt_actions_raw,
            "state_t0_raw": state_t0_raw,
            "prompt": prompt,
            "task_desc": task_desc,
        })
        print(f"  Episode {episode_idx}: {task_desc[:60]}...")

    # Normalize GT
    action_stats = dataset_stats['action']['default']
    cur_stats = {k.removeprefix('global_'): v for k, v in action_stats.items() if k.startswith('global_')}
    norm_action = SingleFieldLinearNormalizer(stats=cur_stats, mode='min/max')

    print(f"\n=== Loaded {len(all_results)} episodes ===")

    # === Teacher forcing across sigma values ===
    print("\n=== Teacher forcing: MSE(pred_velocity, target_velocity) at different sigmas ===")
    print("This is what the training loss measures (without timestep weighting).")
    print()
    print(f"{'sigma':>8} {'mse':>8} {'pred_mag':>10} {'target_mag':>12}")

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
    seed = 0
    for sigma in sigmas:
        all_mse = []
        all_pred_mag = []
        all_target_mag = []
        for rec in all_results:
            gt_norm = norm_action.forward(torch.tensor(rec["gt_actions_raw"]))
            gt_norm_dev = gt_norm.unsqueeze(0).to(device=device, dtype=dtype)
            timestep_action = torch.tensor([sigma * 1000.0], dtype=dtype, device=device)
            noise = torch.randn(gt_norm_dev.shape, generator=torch.Generator(device="cpu").manual_seed(seed), device="cpu").to(device=device, dtype=dtype)
            noisy = (1 - sigma) * gt_norm_dev + sigma * noise

            # proprio
            state_meta = processor.shape_meta["state"]
            state_key = state_meta[0]["key"]
            sb = {"state": {state_key: torch.as_tensor(rec["state_t0_raw"], dtype=torch.float32).unsqueeze(0)}}
            sb = processor.action_state_transform(sb)
            sb = processor.normalizer.forward(sb)
            proprio_norm = sb["state"][state_key].to(device=device, dtype=dtype)

            context, context_mask = model.encode_prompt(rec["prompt"])
            context, context_mask = model._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio_norm,
            )

            with torch.no_grad():
                pred_velocity = model._predict_action_noise(
                    first_frame_latents=rec["real_latents"],
                    latents_action=noisy,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
            target_velocity = noise - gt_norm_dev
            mse = (pred_velocity.float() - target_velocity.float()).pow(2).mean().item()
            all_mse.append(mse)
            all_pred_mag.append(pred_velocity.float().abs().mean().item())
            all_target_mag.append(target_velocity.float().abs().mean().item())
        print(f"{sigma:>8.2f} {np.mean(all_mse):>8.4f} {np.mean(all_pred_mag):>10.4f} {np.mean(all_target_mag):>12.4f}")

    # === Per-episode MAE with current eval pipeline ===
    print("\n=== Per-episode action MAE (10-step inference, real-video latents) ===")
    print(f"{'ep':>3} {'mae':>8}  task")
    for i, rec in enumerate(all_results):
        gt_norm = norm_action.forward(torch.tensor(rec["gt_actions_raw"]))
        # proprio
        state_meta = processor.shape_meta["state"]
        state_key = state_meta[0]["key"]
        sb = {"state": {state_key: torch.as_tensor(rec["state_t0_raw"], dtype=torch.float32).unsqueeze(0)}}
        sb = processor.action_state_transform(sb)
        sb = processor.normalizer.forward(sb)
        proprio_norm = sb["state"][state_key].to(device=device, dtype=dtype)

        context, context_mask = model.encode_prompt(rec["prompt"])
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio_norm,
        )

        # full inference
        action_horizon = rec["gt_actions_raw"].shape[0]
        latents_action = torch.randn(
            (1, action_horizon, model.action_expert.action_dim),
            generator=torch.Generator(device="cpu").manual_seed(42),
            device="cpu", dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        infer_t, infer_d = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=10, device=device, dtype=latents_action.dtype,
        )
        for step_t, step_d in zip(infer_t, infer_d):
            tta = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=device)
            with torch.no_grad():
                pv = model._predict_action_noise(
                    first_frame_latents=rec["real_latents"],
                    latents_action=latents_action,
                    timestep_action=tta,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
            latents_action = model.infer_action_scheduler.step(pv, step_d, latents_action)
        pred = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        mae = (pred - gt_norm).abs().mean().item()
        print(f"{i:>3} {mae:>8.4f}  {rec['task_desc'][:60]}")


if __name__ == "__main__":
    main()
